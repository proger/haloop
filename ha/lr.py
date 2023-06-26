import math


class LR:
    def __init__(self, args):
        self.args = args

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--lr', type=float, default=3e-4, help='Adam learning rate')
        parser.add_argument('--lr_schedule', type=str, choices=['const', 'cosine'], default='cosine', help='Learning rate schedule')
        parser.add_argument('--warmup_iters', type=int, default=2000, help='Number of warm-up steps')
        parser.add_argument('--lr_decay_iters', type=int, default=200000, help='Number of steps for learning rate decay')
        parser.add_argument('--min_lr', type=float, default=6e-5, help='Minimum learning rate')

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
        args = self.args

        if args.lr_schedule == 'cosine':
            # 1) linear warmup for warmup_iters steps
            if it < args.warmup_iters:
                return args.lr * it / args.warmup_iters
            # 2) if it > lr_decay_iters, return min learning rate
            if it > args.lr_decay_iters:
                return args.min_lr
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
            return args.min_lr + coeff * (args.lr - args.min_lr)
        else:
            return args.lr

    def apply_lr_(self, optimizer, step):
        lr = self.get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr