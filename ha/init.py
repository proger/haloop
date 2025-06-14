import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import lora
from .attention import GPT
from .attention_audio import AudioEncoder, StridingAudioEncoder
from .checkpoint import Checkpointer
from .recognizer import TemporalClassifier, Transducer
from .resnet import FixupBasicBlock, FixupResNet
from . import transformer
from . import rnn


def log(*args, flush=False, **kwargs):
    print(*args, **kwargs, flush=flush, file=sys.stderr)


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    stable_embedding: bool = False
    causal: bool = True
    d_input: int = 1
    rotary_emb_dim: int = 0

    def state_dict(self):
        return asdict(self)


@dataclass
class AudioEncoderConfig(GPTConfig):
    block_size: int = 2048
    vocab_size: int = 128 # assume ascii
    causal: bool = False
    d_input: int = 80
    rotary_emb_dim: int = 64


@dataclass
class StridingAudioEncoderConfig(GPTConfig):
    block_size: int = 2048
    vocab_size: int = 16384 # assume bpe
    causal: bool = False
    d_input: int = 80
    rotary_emb_dim: int = 64
    d_conv: int = 256
    conv_strides: tuple[int, ...] = (2,2,2)



def load_model(ckpt_path, *, map_location='cpu'):
    checkpoint = torch.load(ckpt_path, map_location=map_location)

    if not 'vocab_size' in checkpoint['model_args']:
        # assume checkpoint for a large model

        checkpoint['model_args']['stable_embedding'] = True
        checkpoint['model_args']['vocab_size'] = 50257
        checkpoint['model_args']['bias'] = True

        gptconf = GPTConfig(**checkpoint['model_args'])
        model = nn.ModuleDict({'_orig_mod': GPT(gptconf)})
        model.load_state_dict(checkpoint['model'], strict=False)
    elif '_orig_mod.transformer.h.0.attn.c_attn.lora_A.weight' in checkpoint['model']:
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = nn.ModuleDict({'_orig_mod': GPT(gptconf)})
        lora.attach_to_c_attn(model)
        model.load_state_dict(checkpoint['model'])
    else:
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = nn.ModuleDict({'_orig_mod': GPT(gptconf)})
        model.load_state_dict(checkpoint['model'])

    model.eval()
    model.to(map_location)
    model = model._orig_mod

    return model


def create_model(arch: str, compile: bool = True):
    """
    Model architectures to initialize. Possible options:

        decoder
        decoder:vocab_size
        encoder
        lstm
        rnnlm
        r9
        audio-encoder
        recognizer:encoder:vocab_size
        rnn-transducer:encoder:vocab_size
        audio-transformer
    """
    match arch.split(':'):
        case ['decoder']:
            gptconf = GPTConfig()
            model = GPT(gptconf)
        case ['decoder', vocab_size]:
            gptconf = GPTConfig(vocab_size=int(vocab_size))
            model = GPT(gptconf)
        case ['encoder']:
            gptconf = GPTConfig(block_size=128, causal=False)
            model = GPT(gptconf)
        case ['lstm']:
            model = rnn.Encoder()
        case ['rnnlm']:
            model = Decoder(vocab_size=256,
                            emb_dim=2048,
                            hidden_dim=2048,
                            num_layers=1,
                            dropout=0.0)
        case ['r9']:
            model = FixupResNet(FixupBasicBlock, [5,5,5])
        case ['audio-encoder']:
            config = AudioEncoderConfig()
            config.rotary_emb_dim = 0
            encoder = AudioEncoder(config)
            model = nn.ModuleDict({
                'encoder': encoder,
                'recognizer': TemporalClassifier(feat_dim=config.n_embd, vocab_size=config.vocab_size),
            })
        case ['audio-encoder-rotary']:
            config = AudioEncoderConfig()
            encoder = AudioEncoder(config)
            model = nn.ModuleDict({
                'encoder': encoder,
                'recognizer': TemporalClassifier(feat_dim=config.n_embd, vocab_size=config.vocab_size),
            })
        case ['audio-encoder-rotary-dropout']:
            config = AudioEncoderConfig(dropout=0.1)
            encoder = AudioEncoder(config)
            model = nn.ModuleDict({
                'encoder': encoder,
                'recognizer': TemporalClassifier(feat_dim=config.n_embd, vocab_size=config.vocab_size),
            })
        case ['audio-encoder-rotary-dropout-e8']:
            config = AudioEncoderConfig(dropout=0.1, n_layer=8)
            encoder = AudioEncoder(config)
            model = nn.ModuleDict({
                'encoder': encoder,
                'recognizer': TemporalClassifier(feat_dim=config.n_embd, vocab_size=config.vocab_size),
            })
        case ['striding-e8']:
            config = StridingAudioEncoderConfig(dropout=0.1, n_layer=8)
            encoder = StridingAudioEncoder(config)
            model = nn.ModuleDict({
                'encoder': encoder,
                'recognizer': TemporalClassifier(feat_dim=config.n_embd, vocab_size=config.vocab_size),
            })
        case ['lstm', vocab_size]:
            vocab_size = int(vocab_size)
            model = nn.ModuleDict({
                'encoder': rnn.Encoder(hidden_dim=1536),
                'recognizer': TemporalClassifier(feat_dim=1536, vocab_size=vocab_size),
            })
        case ['recognizer', encoder_arch, vocab_size]:
            vocab_size = int(vocab_size)
            model = nn.ModuleDict({
                'encoder': create_model(encoder_arch, compile=False),
                'recognizer': TemporalClassifier(feat_dim=1024, vocab_size=vocab_size),
            })
        case ['rnn-transducer', encoder_arch, vocab_size]:
            vocab_size = int(vocab_size)
            model = nn.ModuleDict({
                'encoder': create_model(encoder_arch, compile=False),
                'recognizer': Transducer(feat_dim=1024, vocab_size=vocab_size),
            })
        case ['audio-transformer']:
            #config = AudioEncoderConfig(dropout=0.2, n_layer=6)
            #encoder = AudioEncoder(config)
            config = StridingAudioEncoderConfig(dropout=0.2, n_layer=6, n_head=8, n_embd=512, conv_strides=(2,2,1))
            encoder = StridingAudioEncoder(config)
            from ha.transformer import Decoder
            head_dim = config.n_embd // config.n_head
            decoder = Decoder(
                vocab=config.vocab_size,
                head_dim=head_dim,
                heads=config.n_head,
                p_drop=config.dropout,
                layers=4
            )
            model = nn.ModuleDict({'encoder': encoder, 'recognizer': decoder})
        case ['e6ctc-d4', vocab_size]:
            config = StridingAudioEncoderConfig(dropout=0.2, n_layer=6, n_head=8, n_embd=512, conv_strides=(2,2,1), vocab_size=int(vocab_size))
            encoder = StridingAudioEncoder(config)
            from ha.transformer import CTCAttentionDecoder
            head_dim = config.n_embd // config.n_head
            decoder = CTCAttentionDecoder(vocab=config.vocab_size,
                                          head_dim=head_dim, heads=config.n_head,
                                          p_drop=config.dropout, layers=4)
            model = nn.ModuleDict({'encoder': encoder, 'recognizer': decoder})
        case ['audio-transformer-ctc']:
            return create_model('e6ctc-d4:16384', compile=compile)
        case ['e6ctc-d6', vocab_size]:
            config = StridingAudioEncoderConfig(dropout=0.2, n_layer=6, n_head=8, n_embd=512, conv_strides=(2,2,1), vocab_size=int(vocab_size))
            encoder = StridingAudioEncoder(config)
            head_dim = config.n_embd // config.n_head
            decoder = transformer.CTCAttentionDecoder(vocab=config.vocab_size,
                                                      head_dim=head_dim, heads=config.n_head,
                                                      p_drop=config.dropout, layers=6)
            model = nn.ModuleDict({'encoder': encoder, 'recognizer': decoder})
        case ['e6d6', vocab_size]:
            config = StridingAudioEncoderConfig(dropout=0.2, n_layer=6, n_head=8, n_embd=512, conv_strides=(2,2,1), vocab_size=int(vocab_size))
            encoder = StridingAudioEncoder(config)
            head_dim = config.n_embd // config.n_head
            decoder = transformer.Decoder(vocab=config.vocab_size, head_dim=head_dim,
                                          heads=config.n_head, p_drop=config.dropout, layers=6)
            model = nn.ModuleDict({'encoder': encoder, 'recognizer': decoder})
        case ['e12ctc-d12', vocab_size]:
            config = StridingAudioEncoderConfig(dropout=0.2, n_layer=12, n_head=8, n_embd=512, conv_strides=(2,2,1), vocab_size=int(vocab_size))
            encoder = StridingAudioEncoder(config)
            head_dim = config.n_embd // config.n_head
            decoder = transformer.CTCAttentionDecoder(vocab=config.vocab_size, head_dim=head_dim,
                                          heads=config.n_head, p_drop=config.dropout, layers=12)
            model = nn.ModuleDict({'encoder': encoder, 'recognizer': decoder})
        case ['transformer', vocab_size]:
            encoder = transformer.AudioEncoder(head_dim=64, heads=8, layers=12, p_drop=0.2)
            decoder = transformer.CTCAttentionDecoder(
                vocab=int(vocab_size), head_dim=64,
                heads=8, p_drop=0.2, layers=12)
            model = nn.ModuleDict({'encoder': encoder, 'recognizer': decoder})
        case ['s222e12ctc-d12', vocab_size]:
            config = StridingAudioEncoderConfig(dropout=0.2, n_layer=12, n_head=8, n_embd=512, conv_strides=(2,2,2), vocab_size=int(vocab_size))
            encoder = StridingAudioEncoder(config)
            head_dim = config.n_embd // config.n_head
            decoder = transformer.CTCAttentionDecoder(vocab=config.vocab_size, head_dim=head_dim,
                                          heads=config.n_head, p_drop=config.dropout, layers=12)
            model = nn.ModuleDict({'encoder': encoder, 'recognizer': decoder})
        case ['e12d12', vocab_size]:
            encoder = transformer.AudioEncoder(head_dim=64, heads=8, layers=12, p_drop=0.2)
            decoder = transformer.Decoder(vocab=vocab_size, head_dim=64, heads=8,
                                          p_drop=0.2, layers=12)
            model = nn.ModuleDict({'encoder': encoder, 'recognizer': decoder})

        case _:
            raise ValueError(f'unknown architecture {arch}')

    if compile:
        model = torch.compile(model)
    return model


class Initializer:
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--init', type=Path, nargs='+', help="Path to checkpoint(s) to initialize from")
        parser.add_argument('--reset', action='store_true', help="Reset checkpoint epoch count (useful for LR scheduling)")
        parser.add_argument('--arch', type=str, default='transformer:512', help=create_model.__doc__)
        parser.add_argument('--compile', action='store_true', help="torch.compile the model (produces incompatible checkpoints)")
        parser.add_argument('--device', type=str, default='cuda', help="torch device to use")

    def __call__(self, args, make_module = lambda x: x):
        epoch, global_step = 0, 0

        if args.arch == "uk4b":
            assert args.init, "pass --init ckpt10m.pt"
            module = load_model(*args.init)

            log("initializing uk4b model")
        elif args.init:
            module = create_model(args.arch, compile=False).to(args.device)
            module = make_module(module)

            checkpoint = torch.load(args.init[0], map_location=args.device)
            module.load_state_dict(checkpoint)
            if len(args.init) > 1:
                log('averaging models')
                avg_model = torch.optim.swa_utils.AveragedModel(module)
                for m in args.init[1:]:
                    checkpoint = torch.load(m, map_location=args.device)
                    module.load_state_dict(checkpoint)
                    avg_model.update_parameters(module)
                module = avg_model.module

            if not args.reset:
                epoch = checkpoint.get('epoch', -1) + 1
                global_step = checkpoint.get('global_step', -1) + 1
        else:
            module = create_model(args.arch, compile=False).to(args.device)
            module = make_module(module)

            log('initializing randomly')

        if args.compile:
            module = torch.compile(module, mode='reduce-overhead')

        log('model parameters', sum(p.numel() for p in module.parameters() if p.requires_grad))

        return module, epoch, global_step


@torch.inference_mode()
def main():
    import argparse

    class Formatter(argparse.ArgumentDefaultsHelpFormatter,
                    argparse.MetavarTypeHelpFormatter):
        pass

    parser = argparse.ArgumentParser(description='hai initializes models', formatter_class=Formatter)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('arch', type=str, help=create_model.__doc__)
    parser.add_argument('path', type=Path)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = create_model(args.arch)
    print('creating a new model')
    print(model)
    if hasattr(model, 'config'):
        print(model.config)
        Checkpointer(args.path, save='all')(loss=float('inf'), epoch=-1, checkpoint_fn=lambda: {
            'model': model.state_dict(),
            'model_args': model.config.state_dict()
        })
    else:
        Checkpointer(args.path, save='all')(loss=float('inf'), epoch=-1, checkpoint_fn=lambda: model.state_dict())


if __name__ == '__main__':
    main()
