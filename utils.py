from __future__ import annotations
from typing import Iterable


def trim_block(s: str) -> str:
    return "\n".join([line.rstrip() for line in s.strip("\n").splitlines()]).strip()


def bulletify(items: Iterable[str]) -> str:
    items = [x for x in items if x]
    if not items:
        return "- (none)"
    return "\n".join([f"- {x}" for x in items])
