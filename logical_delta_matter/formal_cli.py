from __future__ import annotations

import argparse
import json

from .exact_algebra import build_space_formatting
from .formal_domains import DiscretenessTraits, MatterTraits, Plane, classify_discreteness, classify_matter, classify_plane_pair
from .proof_engine import StrictProofEngine
from .smt_certificates import prove_excluded_middle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run formal logical-delta domain demos.")
    parser.add_argument("--demo", choices=["all", "matter", "discreteness", "planes", "space", "smt"], default="all")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output: dict[str, object] = {}
    engine = StrictProofEngine()
    if args.demo in {"all", "matter"}:
        matter = classify_matter(
            MatterTraits(
                continuous=False,
                grouped=True,
                planetary=False,
                preserves_shape=False,
                preserves_volume=True,
                ionized=False,
            )
        )
        output["matter"] = {
            "decision": matter.to_dict(),
            "certificate": engine.certificate_for_decision(matter).to_dict(),
        }
    if args.demo in {"all", "discreteness"}:
        discreteness = classify_discreteness(DiscretenessTraits(bounded=True, internally_discontinuous=True))
        output["discreteness"] = {
            "decision": discreteness.to_dict(),
            "certificate": engine.certificate_for_decision(discreteness).to_dict(),
        }
    if args.demo in {"all", "planes"}:
        planes = classify_plane_pair(
            Plane.from_ints(1, 0, 0, 0),
            Plane.from_ints(0, 1, 0, 0),
        )
        output["planes"] = {
            "decision": planes.to_dict(),
            "certificate": engine.certificate_for_decision(planes).to_dict(),
        }
    if args.demo in {"all", "space"}:
        output["space_formatting"] = build_space_formatting().to_dict()
    if args.demo in {"all", "smt"}:
        output["smt"] = prove_excluded_middle("Continuous").to_dict()
    if args.demo == "all":
        output["partition_certificate"] = engine.partition_certificate(
            [
                "matter.continuous",
                "particles.grouped",
                "groups.planetary",
                "substance.preserves_shape",
                "fluid.preserves_volume",
                "gas.ionized",
            ]
        ).to_dict()
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
