from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SmtCertificate:
    solver: str
    status: str
    claims: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver": self.solver,
            "status": self.status,
            "claims": self.claims,
        }


def prove_excluded_middle(predicate_name: str) -> SmtCertificate:
    """Prove total/exclusive boolean partition P / not P with z3 when available."""

    try:
        from z3 import And, Bool, Not, Or, Solver, unsat
    except ModuleNotFoundError:
        return SmtCertificate(
            solver="z3",
            status="unavailable",
            claims={
                "predicate": predicate_name,
                "total": "requires z3-solver: prove P or not P",
                "exclusive": "requires z3-solver: prove not(P and not P)",
                "termination": "finite propositional proof obligation",
            },
        )

    predicate = Bool(predicate_name)

    total_solver = Solver()
    total_solver.add(Not(Or(predicate, Not(predicate))))
    total_ok = total_solver.check() == unsat

    exclusive_solver = Solver()
    exclusive_solver.add(And(predicate, Not(predicate)))
    exclusive_ok = exclusive_solver.check() == unsat

    return SmtCertificate(
        solver="z3",
        status="proved" if total_ok and exclusive_ok else "failed",
        claims={
            "predicate": predicate_name,
            "total": total_ok,
            "exclusive": exclusive_ok,
            "termination": True,
        },
    )
