from __future__ import annotations

from pathlib import Path

from logical_delta_matter.runner import run_simulation


def main() -> None:
    summary = run_simulation(
        Path("configs/exact_mvp.json"),
        Path("runs/example"),
        Path("data/exact-latest.json"),
    )
    print(summary["metrics"])


if __name__ == "__main__":
    main()
