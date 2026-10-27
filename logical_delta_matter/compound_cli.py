from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compound_space import compute_compound_space


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute possible/impossible compound space for a finite rule-pack.")
    parser.add_argument(
        "--concepts",
        default="data/confirmed-concepts.json",
        help="Path to confirmed element/compound concept catalog.",
    )
    parser.add_argument(
        "--config",
        default="data/compound-space-config.json",
        help="Path to compound space generation config.",
    )
    parser.add_argument(
        "--out",
        default="data/computed-compound-space.json",
        help="Output JSON consumed by the web UI.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = compute_compound_space(Path(args.concepts), Path(args.config), Path(args.out))
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
