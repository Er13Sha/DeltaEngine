from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import run_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run exact-discrete logical delta MVP.")
    parser.add_argument("--config", default="configs/exact_mvp.json", help="Path to JSON config.")
    parser.add_argument("--out", default="runs/exact_mvp", help="Output directory.")
    parser.add_argument(
        "--export-ui",
        default="data/exact-latest.json",
        help="Optional JSON export consumed by the current web UI.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_simulation(Path(args.config), Path(args.out), Path(args.export_ui) if args.export_ui else None)
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
