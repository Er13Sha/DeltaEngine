"""Формула физического воздействия («формула всего») — точная развёртка.

Источник: видео Rek8C322Q5k «Формула мироздания», «Катющик ТВ», 2022 (расшифровка
`data/sources/video-transcripts/Rek8C322Q5k.ru-auto.txt`, 08:34–21:29). Заказчик выбрал её
законом взаимодействия трека E (27.10.2026).

    W = Q·N                    воздействие = качество × количество
    Q = k_пр·c1 / (4π r²)      c1 — базовая интенсивность (одна на всю Вселенную),
                               k_пр — пространственная передача (однородное пространство: 1),
                               4πr² — распределение на площадь сферы
    N = h·S·ρ                  материальный слой: высота × площадь сечения × плотность
    F = k_в·N₂·Q·cos α         сила на воспринимающее тело, k_в — восприимчивость (тяготение: 1)

При C = c1·k_пр/(4π) получается F = C·N₁·N₂/r² — общая структура законов тяготения
(N — масса) и Кулона (N — заряд). Формула задаёт величину; знака (притяжение или
отталкивание) она не содержит, это отмечено и в видео (22:34).

Арифметика точная: числа — дроби, π держится символом (степень π), поэтому 4π в c1 и
в знаменателе сокращаются без округления. Плавающая точка — только при печати.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

SOURCE = {"video_id": "Rek8C322Q5k", "title": "Формула мироздания", "channel": "Катющик ТВ",
          "upload_date": "2022-02-11", "span": "08:34–21:29"}


@dataclass(frozen=True)
class PiNumber:
    """Число вида coef·π^power с рациональным coef: точное умножение и деление."""

    coef: Fraction
    power: int = 0

    @staticmethod
    def of(x) -> PiNumber:
        return x if isinstance(x, PiNumber) else PiNumber(Fraction(x))

    def __mul__(self, other) -> PiNumber:
        o = PiNumber.of(other)
        return PiNumber(self.coef * o.coef, self.power + o.power)

    __rmul__ = __mul__

    def __truediv__(self, other) -> PiNumber:
        o = PiNumber.of(other)
        return PiNumber(self.coef / o.coef, self.power - o.power)

    def __float__(self) -> float:
        return float(self.coef) * math.pi ** self.power

    def __str__(self) -> str:
        if self.power == 0:
            return str(self.coef)
        return f"{self.coef}·π^{self.power}"


PI = PiNumber(Fraction(1), 1)

# CODATA 2018: e и k_e (через ε₀) — с погрешностью таблицы, G и m_p — измеренные
G_NEWTON = Fraction("6.67430e-11")  # Н·м²/кг²
EPSILON_0 = Fraction("8.8541878128e-12")  # Ф/м
E_CHARGE = Fraction("1.602176634e-19")  # Кл, точно в СИ
M_PROTON = Fraction("1.67262192369e-27")  # кг

# Базовая интенсивность c1 для двух известных случаев: так формула видео переходит в
# законы из учебника. Для тяготения c1 = 4πG, для электричества c1 = 1/ε₀ (закон Гаусса).
C1 = {"тяготение": 4 * PI * G_NEWTON, "электричество": PiNumber(1 / EPSILON_0)}


def quality(c1: PiNumber, r, k_transfer=1) -> PiNumber:
    """Качественный показатель Q = k_пр·c1/(4πr²)."""
    return k_transfer * c1 / (4 * PI * Fraction(r) ** 2)


def quantity(h, s, rho) -> Fraction:
    """Количественный показатель N = h·S·ρ (масса или заряд материального слоя)."""
    return Fraction(h) * Fraction(s) * Fraction(rho)


def action(c1: PiNumber, r, n_source) -> PiNumber:
    """Воздействие W = Q·N, которое тело с количеством N оказывает на расстоянии r."""
    return quality(c1, r) * Fraction(n_source)


def force(c1: PiNumber, r, n_source, n_receiver, k_receive=1, cos_alpha=1) -> PiNumber:
    """Величина силы на воспринимающее тело: F = k_в·N₂·W·cos α (знака формула не задаёт)."""
    return action(c1, r, n_source) * (Fraction(k_receive) * Fraction(n_receiver) * Fraction(cos_alpha))


def coupling(c1: PiNumber, k_transfer=1) -> PiNumber:
    """Единый коэффициент C = c1·k_пр/(4π), при котором F = C·N₁·N₂/r²."""
    return k_transfer * c1 / (4 * PI)


def textbook_check() -> dict:
    """Сверка с законами Ньютона и Кулона для пары протонов на расстоянии 1 м."""
    newton = G_NEWTON * M_PROTON ** 2
    grav = force(C1["тяготение"], 1, M_PROTON, M_PROTON)
    k_e = coupling(C1["электричество"])
    elec = force(C1["электричество"], 1, E_CHARGE, E_CHARGE)
    return {
        "gravity_equals_newton_exactly": grav == PiNumber(newton),
        "gravity_coupling": str(coupling(C1["тяготение"])),
        "coulomb_constant": float(k_e),
        "coulomb_constant_table": 8.9875517923e9,
        "coulomb_over_gravity_protons": float(elec / grav),
        "sign": "формула задаёт величину; направление — из механизма (см. семейство D)",
    }
