"""Целочисленная решётчатая модель поля (трек E, этап E1: шип).

Определения — план, раздел 9.3 (вариант ассистента, к утверждению заказчиком):

- решётка — квадрат L × L с периодическими границами (шип в 2D; 3D — следующий этап);
- поле — целое отклонение от фона δ = φ − φ₀ в каждой ячейке; в семействах B и C
  δ ∈ {−1, 0, +1} (квант уплотнения), в семействе A — любое целое;
- правило сохраняет сумму δ: значения только переносятся между соседними ячейками;
- уплотнение — связная (по 4 соседям) область ячеек одного знака с |δ| > θ;
- частица — уплотнение, которое сохраняет размер в пределах допуска не менее T свипов;
- заряд Q — сумма δ по области; масса M — сумма |δ|.

Семейства:

- **A** (осыпание, детерминированное): ячейка с |δ| > θ_top отдаёт по единице своего знака
  каждому из четырёх соседей. Отрицательный контроль.
- **B** (сцепление): энергия E = −J · (число соседних пар одного ненулевого знака);
  динамика Кавасаки — обмен значениями соседних ячеек, принятие по Метрополису.
  Отрицательный контроль для K1: ожидается слияние одноимённых.
- **C** (сцепление + отталкивание одноимённых): к B добавлено +K · Σ G(r) по парам ячеек
  одного знака на расстоянии r ≤ R, G(r) = ⌊R/r⌋. Члена, притягивающего разноимённые,
  нет (условие трека: притяжение не задаётся правилом напрямую).

Энергии целые. Вероятность принятия exp(−ΔE/T) заранее переводится в целые пороги
из 2³¹; случайность — только через заданное зерно (воспроизводимость).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

NEIGHBORS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int64)
NEIGHBORS_N = NEIGHBORS  # имя для ядер numba


def kernel(radius: int) -> np.ndarray:
    """Смещения (dx, dy, G) дальнодействующего отталкивания: G(r) = ⌊R / r⌋ при 0 < r ≤ R."""
    rows = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            r2 = dx * dx + dy * dy
            if r2 == 0 or r2 > radius * radius:
                continue
            g = int(radius // math.sqrt(r2))
            if g > 0:
                rows.append((dx, dy, g))
    return np.array(rows, dtype=np.int64).reshape(-1, 3)


def acceptance(temperature_num: int, temperature_den: int, max_delta: int) -> np.ndarray:
    """Пороги принятия ⌊exp(−ΔE/T) · 2³¹⌋ для ΔE = 0 … max_delta (T = num/den)."""
    t = temperature_num / temperature_den
    return np.array([int(math.exp(-d / t) * 2 ** 31) for d in range(max_delta + 1)], dtype=np.int64)


@njit(cache=True)
def _local_energy_change(s, L, x, y, nx, ny, a, b, J, K, kern):
    """ΔE при обмене значений ячеек (x, y)=a и (nx, ny)=b."""
    de = 0
    # сцепление: связи x и соседа n с остальными соседями (связь x–n не меняется: a ≠ b)
    for k in range(4):
        ux = (x + NEIGHBORS_N[k, 0]) % L
        uy = (y + NEIGHBORS_N[k, 1]) % L
        if ux == nx and uy == ny:
            continue
        v = s[ux, uy]
        if a != 0 and v == a:
            de += J
        if b != 0 and v == b:
            de -= J
    for k in range(4):
        ux = (nx + NEIGHBORS_N[k, 0]) % L
        uy = (ny + NEIGHBORS_N[k, 1]) % L
        if ux == x and uy == y:
            continue
        v = s[ux, uy]
        if b != 0 and v == b:
            de += J
        if a != 0 and v == a:
            de -= J
    if K != 0:
        for k in range(kern.shape[0]):
            g = kern[k, 2]
            ux = (x + kern[k, 0]) % L
            uy = (y + kern[k, 1]) % L
            if not (ux == nx and uy == ny):
                v = s[ux, uy]
                if a != 0 and v == a:
                    de -= K * g
                if b != 0 and v == b:
                    de += K * g
            ux = (nx + kern[k, 0]) % L
            uy = (ny + kern[k, 1]) % L
            if not (ux == x and uy == y):
                v = s[ux, uy]
                if b != 0 and v == b:
                    de -= K * g
                if a != 0 and v == a:
                    de += K * g
    return de



@njit(cache=True)
def kawasaki(s, J, K, kern, thresholds, sweeps, seed):
    """Динамика Кавасаки: L² предложений обмена на свип; сумма и состав δ сохраняются."""
    np.random.seed(seed)
    L = s.shape[0]
    max_d = thresholds.shape[0] - 1
    accepted = 0
    for _ in range(sweeps):
        for _ in range(L * L):
            x = np.random.randint(0, L)
            y = np.random.randint(0, L)
            k = np.random.randint(0, 4)
            nx = (x + NEIGHBORS_N[k, 0]) % L
            ny = (y + NEIGHBORS_N[k, 1]) % L
            a = s[x, y]
            b = s[nx, ny]
            if a == b:
                continue
            de = _local_energy_change(s, L, x, y, nx, ny, a, b, J, K, kern)
            if de > 0:
                if de > max_d or np.random.randint(0, 2 ** 31) >= thresholds[de]:
                    continue
            s[x, y] = b
            s[nx, ny] = a
            accepted += 1
    return accepted


@njit(cache=True)
def topple(s, theta, max_steps):
    """Семейство A: осыпание до устойчивости (все |δ| ≤ θ) или до max_steps."""
    L = s.shape[0]
    for step in range(max_steps):
        moved = False
        for x in range(L):
            for y in range(L):
                v = s[x, y]
                if v > theta or v < -theta:
                    sign = 1 if v > 0 else -1
                    s[x, y] -= 4 * sign
                    for k in range(4):
                        s[(x + NEIGHBORS_N[k, 0]) % L, (y + NEIGHBORS_N[k, 1]) % L] += sign
                    moved = True
        if not moved:
            return step
    return max_steps


def energy(s: np.ndarray, J: int, K: int, kern: np.ndarray) -> int:
    """Полная энергия: −J · (соседние пары одного знака) + K · Σ G(r) по парам одного знака."""
    return _nn(s, J) + (K * _lr_pairs(s, kern) if K else 0)


def _nn(s, J):
    e = 0
    for dx, dy in ((1, 0), (0, 1)):
        t = np.roll(np.roll(s, -dx, 0), -dy, 1)
        e -= J * int(np.sum((s == t) & (s != 0)))
    return e


def _lr_pairs(s, kern):
    total = 0
    for dx, dy, g in kern:
        t = np.roll(np.roll(s, -dx, 0), -dy, 1)
        total += int(g) * int(np.sum((s == t) & (s != 0)))
    return total // 2


def clusters(s: np.ndarray, theta: int = 0) -> list[dict]:
    """Уплотнения: связные области одного знака с |δ| > θ. Центр — с учётом периодичности."""
    L = s.shape[0]
    seen = np.zeros_like(s, dtype=bool)
    out = []
    for x in range(L):
        for y in range(L):
            v = int(s[x, y])
            if seen[x, y] or abs(v) <= theta:
                continue
            sign = 1 if v > 0 else -1
            stack, cells = [(x, y)], []
            seen[x, y] = True
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ux, uy = (cx + dx) % L, (cy + dy) % L
                    w = int(s[ux, uy])
                    if not seen[ux, uy] and abs(w) > theta and (w > 0) == (sign > 0):
                        seen[ux, uy] = True
                        stack.append((ux, uy))
            q = sum(int(s[c]) for c in cells)
            out.append({"cells": len(cells), "charge": q, "mass": sum(abs(int(s[c])) for c in cells),
                        "center": _periodic_center(cells, L)})
    return out


def _periodic_center(cells, L):
    xs = np.array([c[0] for c in cells]) * 2 * math.pi / L
    ys = np.array([c[1] for c in cells]) * 2 * math.pi / L
    cx = (math.atan2(np.sin(xs).mean(), np.cos(xs).mean()) % (2 * math.pi)) * L / (2 * math.pi)
    cy = (math.atan2(np.sin(ys).mean(), np.cos(ys).mean()) % (2 * math.pi)) * L / (2 * math.pi)
    return (cx, cy)


def periodic_distance(a, b, L) -> float:
    dx = abs(a[0] - b[0]); dx = min(dx, L - dx)
    dy = abs(a[1] - b[1]); dy = min(dy, L - dy)
    return math.hypot(dx, dy)
