from pathlib import Path
from typing import Dict, List, Optional, Literal
import torch


class Checkpointer:
    def __init__(self, path: Path, save: Literal['all', 'best', 'last+best', 'none'] = 'best'):
        self.best_loss = float('inf')
        self.save = save
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

    def __call__(self, loss, epoch, checkpoint_fn):
        checkpoint = None
        if best := (loss <= self.best_loss):
            self.best_loss = loss

        if self.save == 'none':
            return

        if self.save == 'all':
            checkpoint = checkpoint_fn()
            path = self.path / f'epoch-{epoch}.pt'
            print(f'saving checkpoint to {str(path)}', flush=True)
            torch.save(checkpoint, str(path))
        elif self.save == 'last+best':
            checkpoint = checkpoint_fn()
            path = self.path / f'last.pt'
            print(f'saving checkpoint to {str(path)}', flush=True)
            torch.save(checkpoint, str(path))

        if best:
            path = self.path / 'best.pt'
            if not checkpoint:
                checkpoint = checkpoint_fn()
            print(f'saving checkpoint to {str(path)}', flush=True)
            torch.save(checkpoint, str(path))

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--exp', type=Path, default='exp/haloop', help="Path to checkpoint directory")
        parser.add_argument('--save', type=str, default='last+best', choices=['all', 'last+best', 'best', 'none'], help='What checkpoints to save after evaluation')

def construct_path_suffix(
    config: Dict,
    base_config: Dict,
    always_include: Optional[List[str]] = None,
    always_ignore: Optional[List[str]] = None,
) -> str:
    suffix_parts: List[str] = []
    if always_include is None:
        always_include = []
    if always_ignore is None:
        always_ignore = []

    for k in sorted(config.keys()):
        if k in always_ignore:
            continue
        if k in always_include or config[k] != base_config.get(k):
            suffix_parts.append(f"{k}-{str(config[k]).replace('.', '_').replace('/', '_')}")

    return ".".join(suffix_parts)
