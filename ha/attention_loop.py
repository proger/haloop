#
# This train loop finetunes GPT by updating on examples sequentially from a binary pretokenized file.
# 
# The code is derived from https://github.com/karpathy/nanoGPT/blob/master/train.py
# and https://github.com/proger/uk4b/blob/main/train.py
#
#
import argparse
import os
import time
import math
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from . import lora
from .checkpoint import construct_path_suffix, Checkpointer
from .init import load_model, Initializer
from .mlm import mask_tokens
from .optim import LR, configure_optimizers


class Formatter(argparse.ArgumentDefaultsHelpFormatter,
                argparse.MetavarTypeHelpFormatter,
                argparse.RawDescriptionHelpFormatter):
    pass

parser = argparse.ArgumentParser(description="hala trains attention models", formatter_class=Formatter)
Initializer.add_arguments(parser)
parser.add_argument("--train", type=str, help="Path to training data")
parser.add_argument("--eval", type=str, help="Path to validation data")
parser.add_argument("--objective", choices=["lm", "denoise", "cond"], default="lm", type=str, help="lm: predict next token; denoise: predict masked tokens; cond: predict one last token in the sequence")
parser.add_argument("--train-shuffle", action='store_true', help="If True, randomly samples batches from the training set")

Checkpointer.add_arguments(parser)

parser.add_argument("--eval-interval", type=int, default=100, help="Interval for evaluation")
parser.add_argument("--log-interval", type=int, default=1, help="Interval for logging")

parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of gradient accumulation steps")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
parser.add_argument("--block_size", type=int, default=1024, help="Block size")

parser.add_argument("--max_iters", type=int, default=200000, help="Total number of training iterations")
parser.add_argument("--grad_clip", type=float, default=1.0, help="Value to clip gradients at, set to 0.0 to disable")

parser.add_argument("--lora", action="store_true", help="Train LoRA adapter")

# lr schedule and adamw
LR.add_arguments(parser)

parser.add_argument("--backend", type=str, default="nccl", help="DDP backend")
parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")

args = parser.parse_args()

if args.train is None and args.eval is None:
   parser.error("at least one of --train and --eval is required")
print(args)

# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=args.backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    device = args.device

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

train_data = np.memmap(args.train, dtype=np.uint16, mode="r")
val_data = np.memmap(args.eval, dtype=np.uint16, mode="r")

if master_process:
    checkpoint = Checkpointer(path=args.exp, save=args.save)

def get_batch(data: np.ndarray,
              step: int,
              block_size=args.block_size,
              batch_size=args.batch_size,
              device_type=device_type,
              device=args.device):
    if args.train_shuffle:
        ix = torch.randint(len(data) - block_size, (batch_size,))
    else:
        ix = range(step * block_size * batch_size, (step + 1) * block_size * batch_size, block_size)
    x = torch.stack([torch.from_numpy((data[i : i + block_size].astype(np.int64))) for i in ix])

    match args.objective:
        case "lm":
            y = torch.cat((x[:, 1:], x.new_zeros((len(x), 1))), dim=1)
        case "denoise":
            x, y = mask_tokens(x)
        case "cond":
            # predict only final token in the sequence
            final_token = (x != 0).sum(dim=-1) - 2
            mask = torch.nn.functional.one_hot(final_token, num_classes=block_size)
            y = y*mask
            #print('cond', x, y[torch.arange(batch_size), final_token])

    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

iter_num = 0

initializer = Initializer()
model, _, _ = initializer(args)
model.train()

assert args.block_size == model.config.block_size, "Block sizes don't match"

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

if args.lora:
    lora.attach_to_c_attn(model)
    lora.mark_only_lora_as_trainable_(model)

model.to(device)

# optimizer
optimizer = configure_optimizers(model, args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)

# compile the model
if compile:
    print("Compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def evaluate():
    model.eval()
    eval_iters = len(val_data) // args.block_size // args.batch_size
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(val_data, k)

        with ctx:
            loss = model.forward_all(X, Y)

        losses[k] = loss.item()
    model.train()
    return losses.mean()

lr_ctl = LR(args)

if args.wandb and master_process:
    import wandb
    wandb.init(config=args)

train_batches = len(train_data) // args.block_size // args.batch_size // args.gradient_accumulation_steps

X, Y = get_batch(train_data, iter_num * args.gradient_accumulation_steps) # fetch the very first batch
if master_process:
    print("Trainable params", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Train batches:", train_batches)
    print(f"Tokens per step, update:", X.numel(), X.numel() * args.gradient_accumulation_steps)

t0 = time.time()
while args.train:
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(args.gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == args.gradient_accumulation_steps - 1)
        with ctx:
            loss = model.forward_all(X, Y)
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch(train_data, (iter_num * args.gradient_accumulation_steps + micro_step) % train_batches)
        # backward pass, with gradient scaling if training in fp16
        if torch.isnan(loss):
            break
        scaler.scale(loss).backward()
    if torch.isnan(loss):
        print("loss is NaN, skipping this update")
        continue

    lr = lr_ctl.apply_lr_(optimizer, iter_num)

    log_dict = {}

    # clip the gradient
    if args.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        log_dict["train/grad_norm"] = grad_norm

    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % args.log_interval == 0 and master_process:
        train_loss = loss.item() # loss as float. note: this is a CPU-GPU sync point
        print(f"iter {iter_num}: loss {train_loss:.4f}, time {dt*1000:.2f}ms, grad_norm: {grad_norm:.3f}, lr: {lr}")

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num and iter_num % args.eval_interval == 0:
            val_loss = evaluate()
            print(f"eval {iter_num}: val loss {val_loss:.4f}")
            log_dict["val/loss"] = val_loss
            if not math.isnan(val_loss):
                raw_model = model.module if ddp else model
                checkpoint(loss=val_loss, epoch=iter_num, checkpoint_fn=lambda: {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': raw_model.config.state_dict(),
                    'iter_num': iter_num,
                    'val_loss': val_loss,
                    'args': args,
                })
            else:
                print("NaN loss detected")
                break

        if args.wandb:
            wandb.log(log_dict | {
                "iter": iter_num,
                "train/loss": train_loss,
                "lr": lr,
            })

    iter_num += 1

    # termination conditions
    if iter_num > args.max_iters:
        break


if args.eval and master_process:
    val_loss = evaluate()
    print(f"step {iter_num}: val loss {val_loss}. final eval")


if ddp:
    destroy_process_group()


def main():
    pass # stub for hala
