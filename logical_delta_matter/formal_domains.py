from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any


@dataclass(frozen=True, slots=True)
class FormalDecision:
    domain: str
    leaf: str
    trace: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "leaf": self.leaf,
            "trace": list(self.trace),
            "obligations": {
                "total": "All predicates read finite boolean or rational fields defined by the domain dataclass.",
                "exclusive": "Every branch is selected from a boolean predicate P / not P.",
                "termination": "The decider executes a finite fixed sequence of predicate checks.",
            },
        }


@dataclass(frozen=True, slots=True)
class MatterTraits:
    continuous: bool
    grouped: bool
    planetary: bool
    preserves_shape: bool
    preserves_volume: bool
    ionized: bool


@dataclass(frozen=True, slots=True)
class DiscretenessTraits:
    bounded: bool
    internally_discontinuous: bool


@dataclass(frozen=True, slots=True)
class Plane:
    """Exact plane a*x + b*y + c*z = d."""

    normal: tuple[Fraction, Fraction, Fraction]
    offset: Fraction

    @classmethod
    def from_ints(cls, a: int, b: int, c: int, d: int) -> "Plane":
        return cls((Fraction(a), Fraction(b), Fraction(c)), Fraction(d))

    def __post_init__(self) -> None:
        if len(self.normal) != 3:
            raise ValueError("Plane normal must have exactly three components.")
        if all(value == 0 for value in self.normal):
            raise ValueError("Plane normal cannot be zero.")


def classify_matter(traits: MatterTraits) -> FormalDecision:
    trace: list[dict[str, Any]] = []

    if branch(trace, "matter.continuous", "Continuous(x)", traits.continuous, "Field", "Particles"):
        return FormalDecision("Matter", "Field", tuple(trace))
    if not branch(trace, "particles.grouped", "Grouped(x)", traits.grouped, "GroupsOfParticles", "FreeParticles"):
        return FormalDecision("Matter", "FreeParticles", tuple(trace))
    if branch(trace, "groups.planetary", "Planetary(x)", traits.planetary, "Subparticles", "Substance"):
        return FormalDecision("Matter", "Subparticles", tuple(trace))
    if branch(trace, "substance.preserves_shape", "PreservesShape(x)", traits.preserves_shape, "Solid", "GasOrLiquid"):
        return FormalDecision("Matter", "Solid", tuple(trace))
    if branch(trace, "fluid.preserves_volume", "PreservesVolume(x)", traits.preserves_volume, "Liquid", "Gas"):
        return FormalDecision("Matter", "Liquid", tuple(trace))
    if branch(trace, "gas.ionized", "Ionized(x)", traits.ionized, "Plasma", "NeutralGas"):
        return FormalDecision("Matter", "Plasma", tuple(trace))
    return FormalDecision("Matter", "NeutralGas", tuple(trace))


def classify_discreteness(traits: DiscretenessTraits) -> FormalDecision:
    trace: list[dict[str, Any]] = []
    bounded = branch(trace, "object.bounded", "Bounded(x)", traits.bounded, "ExternallyDiscrete", "Unbounded")
    internal = branch(
        trace,
        "object.internally_discontinuous",
        "InternallyDiscontinuous(x)",
        traits.internally_discontinuous,
        "InternallyDiscrete",
        "InternallyContinuous",
    )
    if not bounded and not internal:
        leaf = "ContinuousObject"
    elif not bounded and internal:
        leaf = "InternalDiscreteUnbounded"
    elif bounded and not internal:
        leaf = "ExternalDiscreteContinuous"
    else:
        leaf = "InternalExternalDiscrete"
    return FormalDecision("ObjectDiscreteness", leaf, tuple(trace))


def classify_plane_pair(left: Plane, right: Plane) -> FormalDecision:
    trace: list[dict[str, Any]] = []
    normals_parallel = are_vectors_parallel(left.normal, right.normal)
    same_plane = normals_parallel and are_same_plane(left, right)
    have_common_points = not normals_parallel or same_plane
    if not branch(trace, "planes.common_points", "HaveCommonPoints(p1,p2)", have_common_points, "Intersecting", "Parallel"):
        return FormalDecision("PlanePair", "Parallel", tuple(trace))
    if branch(trace, "planes.all_points_common", "AllPointsCommon(p1,p2)", same_plane, "Coincident", "LineIntersection"):
        return FormalDecision("PlanePair", "Coincident", tuple(trace))
    perpendicular = dot(left.normal, right.normal) == 0
    if branch(trace, "planes.right_angle", "Angle(p1,p2) = 90deg", perpendicular, "Perpendicular", "Oblique"):
        return FormalDecision("PlanePair", "Perpendicular", tuple(trace))
    return FormalDecision("PlanePair", "Oblique", tuple(trace))


def branch(
    trace: list[dict[str, Any]],
    delta_id: str,
    predicate: str,
    value: bool,
    true_branch: str,
    false_branch: str,
) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{delta_id} must evaluate to bool.")
    trace.append(
        {
            "delta_id": delta_id,
            "predicate": predicate,
            "value": value,
            "branch": true_branch if value else false_branch,
            "branches": {"true": true_branch, "false": false_branch},
        }
    )
    return value


def dot(left: tuple[Fraction, Fraction, Fraction], right: tuple[Fraction, Fraction, Fraction]) -> Fraction:
    return sum((a * b for a, b in zip(left, right)), Fraction(0))


def are_vectors_parallel(left: tuple[Fraction, Fraction, Fraction], right: tuple[Fraction, Fraction, Fraction]) -> bool:
    return (
        left[1] * right[2] - left[2] * right[1] == 0
        and left[2] * right[0] - left[0] * right[2] == 0
        and left[0] * right[1] - left[1] * right[0] == 0
    )


def are_same_plane(left: Plane, right: Plane) -> bool:
    left4 = (*left.normal, left.offset)
    right4 = (*right.normal, right.offset)
    return are_vectors4_parallel(left4, right4)


def are_vectors4_parallel(left: tuple[Fraction, Fraction, Fraction, Fraction], right: tuple[Fraction, Fraction, Fraction, Fraction]) -> bool:
    for i in range(4):
        for j in range(i + 1, 4):
            if left[i] * right[j] - left[j] * right[i] != 0:
                return False
    return True
