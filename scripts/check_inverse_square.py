"""Точная проверка заданного закона; не является симуляцией поля."""
import json
from fractions import Fraction
from pathlib import Path

rows = []
for exponent in (0, 1, 2, 6, 1000):
    radius = Fraction(10**exponent)
    force = 1/(radius*radius)
    assert force > 0
    assert force*radius*radius == 1
    assert (1/((2*radius)**2))/force == Fraction(1, 4)
    rows.append({"distance_exact": f"10^{exponent}", "force_exact": f"10^{-2*exponent}",
                 "r_squared_times_force": "1", "F_2r_over_F_r": "1/4", "nonzero": force != 0})
result = {"scope": "model", "kind": "exact_arithmetic_check_of_postulate", "K": "1", "rows": rows}
output = Path(__file__).resolve().parents[1] / "data/experiments/field/inverse_square_check.json"
output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print("Exact checks passed: 5 distances; F > 0, r^2 F = 1, F(2r)/F(r) = 1/4.")
