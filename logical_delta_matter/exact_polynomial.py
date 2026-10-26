from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping


Number = int | Fraction


@dataclass(frozen=True, slots=True)
class LaurentPolynomial:
    """Exact element of Q[n, n^-1].

    Terms are stored as exponent -> rational coefficient. The trusted path
    rejects float coefficients so symbolic scale arithmetic stays exact.
    """

    terms: Mapping[int, Fraction]

    def __post_init__(self) -> None:
        normalized: dict[int, Fraction] = {}
        for exponent, coefficient in self.terms.items():
            if type(exponent) is not int:
                raise TypeError("Laurent exponents must be integers.")
            if type(coefficient) is float:
                raise TypeError("Float coefficients are forbidden in Q[n,n^-1].")
            rational = Fraction(coefficient)
            if rational:
                normalized[exponent] = normalized.get(exponent, Fraction(0)) + rational
        object.__setattr__(self, "terms", {key: value for key, value in sorted(normalized.items()) if value})

    @classmethod
    def constant(cls, value: Number) -> "LaurentPolynomial":
        return cls({0: Fraction(value)})

    @classmethod
    def n(cls) -> "LaurentPolynomial":
        return cls({1: Fraction(1)})

    @classmethod
    def n_inv(cls) -> "LaurentPolynomial":
        return cls({-1: Fraction(1)})

    def __add__(self, other: "LaurentPolynomial") -> "LaurentPolynomial":
        terms = dict(self.terms)
        for exponent, coefficient in other.terms.items():
            terms[exponent] = terms.get(exponent, Fraction(0)) + coefficient
        return LaurentPolynomial(terms)

    def __sub__(self, other: "LaurentPolynomial") -> "LaurentPolynomial":
        return self + (-other)

    def __neg__(self) -> "LaurentPolynomial":
        return LaurentPolynomial({exponent: -coefficient for exponent, coefficient in self.terms.items()})

    def __mul__(self, other: "LaurentPolynomial") -> "LaurentPolynomial":
        terms: dict[int, Fraction] = {}
        for left_exp, left_coeff in self.terms.items():
            for right_exp, right_coeff in other.terms.items():
                exponent = left_exp + right_exp
                terms[exponent] = terms.get(exponent, Fraction(0)) + left_coeff * right_coeff
        return LaurentPolynomial(terms)

    def scale(self, value: Number) -> "LaurentPolynomial":
        if type(value) is float:
            raise TypeError("Float scale is forbidden in Q[n,n^-1].")
        return LaurentPolynomial({exponent: coefficient * Fraction(value) for exponent, coefficient in self.terms.items()})

    def is_zero(self) -> bool:
        return not self.terms

    def to_dict(self) -> dict[str, str]:
        return {str(exponent): str(coefficient) for exponent, coefficient in self.terms.items()}

    def __str__(self) -> str:
        if not self.terms:
            return "0"
        parts: list[str] = []
        for exponent in sorted(self.terms, reverse=True):
            coefficient = self.terms[exponent]
            coeff_text = "" if coefficient == 1 and exponent != 0 else str(coefficient)
            if exponent == 0:
                term = str(coefficient)
            elif exponent == 1:
                term = f"{coeff_text}n"
            else:
                term = f"{coeff_text}n^{exponent}"
            parts.append(term)
        return " + ".join(parts).replace("+ -", "- ")
