from typing import Dict, List, Optional


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
