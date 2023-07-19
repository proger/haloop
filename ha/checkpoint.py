from pathlib import Path
from typing import Dict, List, Optional
import torch


class Checkpointer:
    def __init__(self, path: Path, save_all: bool = False):
        self.best_loss = float('inf')
        self.save_all = save_all
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

    def __call__(self, loss, epoch, checkpoint_fn):
        checkpoint = None
        if self.save_all:
            checkpoint = checkpoint_fn()
            path = self.path / f'epoch-{epoch}.pt'
            print(f'saving checkpoint to {str(path)}', flush=True)
            torch.save(checkpoint, str(path))

        if loss <= self.best_loss:
            self.best_loss = loss
            path = self.path / 'best.pt'
            if not checkpoint:
                checkpoint = checkpoint_fn()
            print(f'saving checkpoint to {str(path)}', flush=True)
            torch.save(checkpoint, str(path))


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
