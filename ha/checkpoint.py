from pathlib import Path
from typing import Dict, List, Optional
import torch


class Checkpointer:
    def __init__(self, path: Path, save_all: bool = False):
        self.best_loss = float('inf')
        self.save_all = save_all
        self.path = path

    def __call__(self, loss, epoch, checkpoint_fn):
        should_save = self.save_all or loss <= self.best_loss
        self.best_loss = min(loss, self.best_loss)
        if should_save:
            path = self.path
            checkpoint = checkpoint_fn()
            print(f'saving checkpoint to {path}', flush=True)
            torch.save(checkpoint, path)


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
