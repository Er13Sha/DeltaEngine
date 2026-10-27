"""Exact-discrete logical delta prototype.

The package is intentionally dependency-light. The trusted core uses only
integer state, deterministic transitions, compact event logs, and JSON
checkpoints so it can run on the current project without a scientific Python
stack.
"""

from .delta_core import DeltaNode, compile_delta_tree, classify_delta, validate_delta_tree
from .exact_polynomial import LaurentPolynomial
from .formal_domains import DiscretenessTraits, MatterTraits, Plane, classify_discreteness, classify_matter, classify_plane_pair
from .proof_engine import REJECTED_CERTIFICATE, SOUND, UNKNOWN, StrictProofEngine
from .world_exact import WorldConfig, WorldState, init_world, run_world, step_world_exact

__all__ = [
    "DeltaNode",
    "DiscretenessTraits",
    "LaurentPolynomial",
    "MatterTraits",
    "Plane",
    "REJECTED_CERTIFICATE",
    "SOUND",
    "StrictProofEngine",
    "UNKNOWN",
    "WorldConfig",
    "WorldState",
    "classify_delta",
    "classify_discreteness",
    "classify_matter",
    "classify_plane_pair",
    "compile_delta_tree",
    "init_world",
    "run_world",
    "step_world_exact",
    "validate_delta_tree",
]
