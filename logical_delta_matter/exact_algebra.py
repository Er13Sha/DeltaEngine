from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SpaceFormatting:
    R: Any
    n: Any
    L: Any
    T: Any
    backend: str

    def to_dict(self) -> dict[str, str]:
        return {
            "R": str(self.R),
            "n": str(self.n),
            "L": str(self.L),
            "T": str(self.T),
            "backend": self.backend,
            "definitions": "L = R*n; T = R/n",
        }


def build_space_formatting() -> SpaceFormatting:
    """Return symbolic constants for Katyushchik-style space formatting.

    SymPy is used when installed. The fallback keeps exact symbolic strings so
    the trusted logic still avoids float arithmetic.
    """

    try:
        from sympy import symbols
    except ModuleNotFoundError:
        R = SymbolicToken("R")
        n = SymbolicToken("n")
        return SpaceFormatting(R=R, n=n, L=SymbolicExpression("R*n"), T=SymbolicExpression("R/n"), backend="fallback")

    R, n = symbols("R n", positive=True)
    return SpaceFormatting(R=R, n=n, L=R * n, T=R / n, backend="sympy")


@dataclass(frozen=True, slots=True)
class SymbolicToken:
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class SymbolicExpression:
    expression: str

    def __str__(self) -> str:
        return self.expression
