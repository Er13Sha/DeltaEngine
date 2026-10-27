"""Ревью открытых индуктором вилок (ворота G7 спецификации).

Индуктор помечает каждую открытую вилку как гипотезу со статусом «candidate».
Ответы, которые на неё опираются, не получают статус `proven`. Человек принимает
вилку — она становится констатацией — или отклоняет её.

    python -m delta_tools.review list   data/trees/intent.learned.json
    python -m delta_tools.review accept data/trees/intent.learned.json fork.root#1 [--by имя]
    python -m delta_tools.review reject data/trees/intent.learned.json fork.t#2
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Iterator

from delta_kernel import Tree


def forks(node: dict[str, Any]) -> Iterator[dict[str, Any]]:
    if "predicate" in node:
        yield node
        yield from forks(node["true"])
        yield from forks(node["false"])


def set_status(data: dict[str, Any], fork_id: str, status: str, by: str) -> dict[str, Any]:
    for fork in forks(data["root"]):
        if fork["id"] == fork_id:
            fork["knowledge"] = "constatation" if status == "accepted" else "hypothesis"
            evidence = fork.setdefault("evidence", [])
            evidence.append({"source": "review", "status": status, "by": by, "date": date.today().isoformat()})
            Tree.from_dict(data)  # дерево остаётся допустимым для ядра
            return fork
    raise KeyError(f"Нет вилки {fork_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ревью вилок, открытых индуктором")
    parser.add_argument("action", choices=["list", "accept", "reject"])
    parser.add_argument("path", type=Path)
    parser.add_argument("fork", nargs="?")
    parser.add_argument("--by", default="reviewer")
    args = parser.parse_args(argv)
    data = json.loads(args.path.read_text(encoding="utf-8"))
    if args.action == "list":
        for fork in forks(data["root"]):
            print(f"{fork['id']:20s} {fork.get('knowledge', 'constatation'):12s} {fork['predicate']}")
        return 0
    if not args.fork:
        parser.error("укажите id вилки")
    set_status(data, args.fork, "accepted" if args.action == "accept" else "rejected", args.by)
    args.path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{args.fork}: {args.action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
