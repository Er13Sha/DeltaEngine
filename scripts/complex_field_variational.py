"""Воспроизводимый вариационный расчёт согласно docs/complex-field-protocol-2026-10-27.md."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def energy(q, a, r):
    i2 = a**2 * math.pi**1.5 * r**3
    i4 = a**4 * (math.pi / 2)**1.5 * r**3
    i6 = a**6 * (math.pi / 3)**1.5 * r**3
    return q*q/(2*i2) + 3*i2/(4*r*r) + i2/2 - i4/4 + i6/6


def run(q, n):
    bounds = (0.1, 2.0, 0.5, 20.0)
    best = (math.inf, 0, 0)
    history = []
    for step in range(5):
        alo, ahi, rlo, rhi = bounds
        size = n if step == 0 else 41
        da = (ahi-alo)/(size-1)
        dr = (rhi-rlo)/(size-1)
        for ia in range(size):
            a = alo + ia*da
            for ir in range(size):
                r = rlo + ir*dr
                candidate = (energy(q, a, r), a, r)
                if candidate < best:
                    best = candidate
        history.append(best[0])
        _, a, r = best
        bounds = (max(0.1, a-da), min(2.0, a+da), max(0.5, r-dr), min(20.0, r+dr))
    e, a, r = best
    i2 = a*a * math.pi**1.5 * r**3
    assert all(y <= x for x, y in zip(history, history[1:]))
    return {"Q": q, "E": e, "E_per_Q": e/q, "omega": q/i2, "A": a, "R": r,
            "boundary": a <= 0.1 or a >= 2 or r <= 0.5 or r >= 20,
            "energy_history": history}


def main():
    # U/f² = 1/2 - x/4 + x²/6 has discriminant < 0 and positive leading coefficient.
    assert (-1/4)**2 - 4*(1/6)*(1/2) < 0
    result = {"scope": "model", "protocol": "docs/complex-field-protocol-2026-10-27.md",
              "coarse_201": [run(q, 201) for q in (10, 100, 1000)],
              "coarse_101": [run(q, 101) for q in (10, 100, 1000)]}
    output = ROOT / "data" / "complex-field-variational-2026-10-27.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
