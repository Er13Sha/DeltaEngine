import json
import math
import random
import unittest
from pathlib import Path

try:
    import numpy as np

    from adapters.field_lattice import experiments as ex
    from adapters.field_lattice.model import _local_energy_change, energy, kawasaki, acceptance, kernel, topple
except ImportError:  # numba и numpy — необязательные зависимости трека E
    np = None

ROOT = Path(__file__).resolve().parents[2]
FIELD = ROOT / "data" / "experiments" / "field"


@unittest.skipIf(np is None, "нужны numpy и numba (pip install -e .[field])")
class FieldModelTests(unittest.TestCase):
    def test_energy_change_matches_total_energy(self):
        rng = random.Random(7)
        kern = kernel(5)
        L = 16
        for _ in range(200):
            s = np.array([[rng.choice([-1, 0, 0, 1]) for _ in range(L)] for _ in range(L)], dtype=np.int64)
            x, y = rng.randrange(L), rng.randrange(L)
            dx, dy = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            nx, ny = (x + dx) % L, (y + dy) % L
            a, b = int(s[x, y]), int(s[nx, ny])
            if a == b:
                continue
            de = _local_energy_change(s, L, x, y, nx, ny, a, b, 40, 1, kern)
            e0 = energy(s, 40, 1, kern)
            s[x, y], s[nx, ny] = b, a
            self.assertEqual(energy(s, 40, 1, kern) - e0, de)

    def test_rules_conserve_field(self):
        rng = np.random.default_rng(3)
        s = rng.choice(np.array([-1, 0, 1], dtype=np.int64), size=(24, 24))
        total, plus = int(s.sum()), int((s == 1).sum())
        kawasaki(s, 40, 1, kernel(8), acceptance(8, 1, 400), 50, 5)
        self.assertEqual((int(s.sum()), int((s == 1).sum())), (total, plus))
        t = rng.integers(-9, 10, size=(24, 24)).astype(np.int64)
        before = int(t.sum())
        topple(t, 3, 100_000)
        self.assertEqual(int(t.sum()), before)
        self.assertTrue(int(np.abs(t).max()) <= 3)

    def test_interaction_energies(self):
        self.assertEqual(ex.interaction_energy("B", (1, 1), 0, "x")[0], -40)  # сцепление при касании
        self.assertEqual(ex.interaction_energy("C", (1, 1), 0, "x")[0], 524)  # отталкивание сильнее
        self.assertEqual(ex.interaction_energy("C", (1, -1), 0, "x")[0], 0)  # разноимённые: прямого притяжения нет

    def test_runs_are_reproducible(self):
        saved = [json.loads(line) for line in (FIELD / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
        run = next(r for r in saved if r["kind"] == "existence" and r["model_family"] == "C" and r["seed"] == 1)
        fresh = ex.existence_run("C", 1)
        self.assertEqual((fresh["config_hash"], fresh["observables"]), (run["config_hash"], run["observables"]))
        self.assertEqual(run["protocol_hash"], ex.protocol_hash())

    def test_negative_controls_and_admission(self):
        summary = json.loads((FIELD / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["negative_controls_failed_as_required"])
        self.assertEqual([f for f, v in summary["families"].items() if v["admitted"]], ["C"])


class FieldCharacteristicTests(unittest.TestCase):
    def test_model_tree_reproduces_scan(self):
        from fractions import Fraction

        from delta_kernel import Tree
        from delta_tools.induction import predict

        tree = Tree.from_dict(json.loads((ROOT / "data" / "trees" / "field_model.json").read_text(encoding="utf-8")))
        self.assertEqual(tree.check(), [])
        scan = json.loads((FIELD / "scan.json").read_text(encoding="utf-8"))
        for row in scan["rows"]:
            record = {k: row[k] for k in ("J", "K", "T")}
            record["t_ratio"], record["k_ratio"] = Fraction(row["t_ratio"]), Fraction(row["k_ratio"])
            with self.subTest(row=record):
                self.assertEqual(predict(tree, record)[0], row["label"])


@unittest.skipIf(np is None, "нужен numpy")
class ScreeningTests(unittest.TestCase):
    """Семейство D: притяжение через экранирование изотропного потока (H2)."""

    def test_single_body_feels_no_force(self):
        from adapters.field_lattice.screening import single_body_force

        self.assertLess(single_body_force(1.0, 0.3), 1e-12)

    def test_pair_attracts_with_inverse_square(self):
        from adapters.field_lattice.screening import power_law_exponent, shadow_force

        ds = [8, 16, 32]
        fs = [shadow_force(d, 1, 1, 0.05, 0.05, n_theta=24, n_phi=16, n_rho=12, n_psi=12) for d in ds]
        self.assertTrue(all(f > 0 for f in fs))
        self.assertAlmostEqual(power_law_exponent(ds, fs), -2.0, delta=0.02)

    def test_saturation_breaks_mass_proportionality(self):
        saved = json.loads((FIELD / "screening.json").read_text(encoding="utf-8"))
        ratios = [row["F_over_m2"] for row in saved["mass_proportionality"]]
        self.assertEqual(ratios, sorted(ratios, reverse=True))
        self.assertGreater(ratios[0] / ratios[-1], 100)

    def test_like_charges_condition_with_real_constants(self):
        saved = json.loads((FIELD / "screening.json").read_text(encoding="utf-8"))
        real = saved["like_charges"]["real_world"]
        self.assertGreater(real["proton_pair_coulomb_over_gravity"], 1e36)
        self.assertGreater(real["proton_charge_to_mass_C_per_kg"], real["threshold_charge_to_mass_C_per_kg"])

    def test_video_formula_holds_for_transparent_bodies(self):
        """Формула видео Rek8C322Q5k: F = c1·N_A·N_B/(4πd²) с c1 = 4π·I, N = μ·V.

        У прозрачных тел отношение силы к формуле → 1 при любом расстоянии; у плотных
        количество N = h·S·ρ насыщается, и формула завышает силу.
        """
        from adapters.field_lattice.screening import shadow_force

        n = 0.001 * 4 / 3 * math.pi
        for d in (12, 24):
            f = shadow_force(d, 1, 1, 0.001, 0.001, n_theta=24, n_phi=16, n_rho=12, n_psi=12)
            self.assertAlmostEqual(f * d * d / (n * n), 1.0, delta=0.005)
        saved = json.loads((FIELD / "screening.json").read_text(encoding="utf-8"))
        dense = [row["ratio"] for row in saved["video_formula"]["rows"] if row["mu"] == 10.0]
        self.assertTrue(all(r < 0.01 for r in dense))


class DragTests(unittest.TestCase):
    """Проверка C6: торможение и нагрев тела в потоке семейства D."""

    def test_body_at_rest_feels_no_drag(self):
        from adapters.field_lattice.drag import drag_integral

        self.assertLess(abs(drag_integral(0.0)), 1e-15)

    def test_drag_is_one_third_of_speed_ratio(self):
        from adapters.field_lattice.drag import drag_integral

        for s in (1e-4, 1e-3, 1e-2):
            self.assertAlmostEqual(drag_integral(s) / (-s / 3), 1.0, delta=1e-4)

    def test_elastic_reemission_cancels_attraction(self):
        saved = json.loads((FIELD / "drag.json").read_text(encoding="utf-8"))
        for row in saved["reemission"]:
            self.assertAlmostEqual(row["shadow_over_formula"], 1.0, delta=0.005)
            self.assertAlmostEqual(row["net_over_shadow"], 1 - row["beta"], delta=0.005)

    def test_earth_bounds_are_fatal(self):
        from adapters.field_lattice.drag import earth_bounds

        earth = earth_bounds()
        self.assertGreater(earth["u_min_over_c"], 1e9)
        self.assertGreater(earth["excess"], 1e30)
        self.assertLess(earth["if_u_is_c"]["stopping_time_days"], 365)


class UnifiedSchemeTests(unittest.TestCase):
    """Тезис: магнитные и ядерные силы — эффекты одного поля с законом W = Q·N."""

    def test_magnetism_follows_from_static_law_and_finite_speed(self):
        from fractions import Fraction

        from adapters.field_lattice.unified import magnetic_over_electric

        for beta in (Fraction(1, 10), Fraction(1, 2), Fraction(9, 10)):
            self.assertEqual(magnetic_over_electric(beta), beta * beta)

    def test_magnetism_is_carried_by_charge_not_mass(self):
        from adapters.field_lattice.unified import wires_check

        wires = wires_check()
        self.assertAlmostEqual(wires["ampere_force_N_per_m"], 2e-7, delta=1e-15)
        self.assertLess(wires["mass_over_charge"], 1e-42)

    def test_nuclear_binding_needs_short_range(self):
        saved = json.loads((FIELD / "unified.json").read_text(encoding="utf-8"))
        nuclear = saved["nuclear"]
        self.assertEqual(len(nuclear["nuclei"]), 20)
        self.assertAlmostEqual(nuclear["data_exponent_B_vs_A"], 1.0, delta=0.06)
        errors = [f["rms_relative_error"] for f in nuclear["fits"]]
        self.assertEqual(errors, sorted(errors))  # чем длиннее радиус, тем хуже
        self.assertGreater(nuclear["fits"][-1]["rms_relative_error"], 0.5)  # формула без затухания
        self.assertLess(nuclear["best"]["rms_relative_error"], 0.1)

    def test_deuteron_binding_dwarfs_gravity(self):
        from adapters.field_lattice.unified import deuteron

        d = deuteron()
        self.assertAlmostEqual(d["binding_MeV"], 2.22457, delta=1e-4)
        self.assertGreater(d["ratio"], 1e36)


class UniversalFluxTests(unittest.TestCase):
    """Уточнение заказчика: экранирует большее тело, поток Вселенной сильнее местного тяготения."""

    def test_big_body_screens_small_with_third_law(self):
        saved = json.loads((FIELD / "drag.json").read_text(encoding="utf-8"))["big_and_small"]
        self.assertAlmostEqual(saved["third_law_ratio"], 1.0, delta=0.002)
        self.assertAlmostEqual(saved["on_small_over_formula"], 1.0, delta=0.005)

    def test_stronger_universal_flux_makes_heating_worse(self):
        from adapters.field_lattice.drag import universal_over_local

        rows = universal_over_local()
        self.assertGreater(rows[0]["universal_over_local"], 1)  # условие выполняется само
        heats = [r["heat_excess"] for r in rows]
        self.assertEqual(heats, sorted(heats))
        self.assertGreater(heats[0], 1e30)

    def test_sink_medium_needs_superluminal_inflow(self):
        from adapters.field_lattice.drag import sink_bounds

        sink = sink_bounds()
        self.assertAlmostEqual(sink["inflow_speed_min_m_s"] / sink["g_times_age_m_s"], 1.0, delta=1e-9)
        self.assertGreater(sink["inflow_over_c"], 1e9)


class InstantGravityTests(unittest.TestCase):
    """Тезис заказчика: тяготение мгновенно, свет ограничен средой."""

    def test_instant_carriers_pass_c6_only_without_energy(self):
        from adapters.field_lattice.drag import instant_carriers

        inst = instant_carriers()
        self.assertLess(inst["e_over_p_max_m_s"], 1e-12)  # импульс почти без энергии
        self.assertGreater(inst["superluminal_u_min_over_c"], 1e21)
        self.assertLess(inst["orbit_change_over_age"], 1e-10)  # торможение тогда ничтожно

    def test_instant_gravity_vs_gw170817_and_binary_pulsar(self):
        from adapters.field_lattice.drag import instant_vs_observations

        obs = instant_vs_observations()
        self.assertGreater(obs["gw170817"]["instant_lead_years"], 1e8)
        self.assertLess(obs["gw170817"]["instant_tidal_over_wave"], 1e-70)
        self.assertGreater(obs["psr_b1913_16"]["deviation_sigma"], 600)


class LightInFieldTests(unittest.TestCase):
    """Тезис: свет — частица, движущаяся в базовом гравитационном поле."""

    def test_deflection_corpuscle_is_half_of_medium(self):
        from adapters.field_lattice.light import R_SUN, deflection

        corpuscle, medium = deflection(R_SUN, "corpuscle"), deflection(R_SUN, "medium")
        self.assertAlmostEqual(medium / corpuscle, 2.0, delta=1e-6)
        self.assertAlmostEqual(medium / (math.pi / 180 / 3600), 1.7512, delta=1e-3)

    def test_delay_sign_separates_models(self):
        saved = json.loads((FIELD / "light.json").read_text(encoding="utf-8"))
        rt = saved["shapiro_round_trip_us"]
        self.assertLess(rt["corpuscle"], 0)  # опережение
        self.assertAlmostEqual(rt["medium"] / saved["shapiro_medium_analytic_us"], 1.0, delta=1e-5)

    def test_cassini_selects_medium(self):
        saved = json.loads((FIELD / "light.json").read_text(encoding="utf-8"))
        self.assertLess(saved["medium_sigma"], 2)
        self.assertGreater(saved["corpuscle_deflection_sigma"], 1e4)

    def test_field_slows_clocks_and_light_equally(self):
        from adapters.field_lattice.light import light_speed_in_field

        s = light_speed_in_field()
        split = s["deflection_split_arcsec"]
        self.assertAlmostEqual(split["total"], 1.7512, delta=1e-3)
        self.assertAlmostEqual(s["pound_snider"]["predicted_shift"], 2.45e-15, delta=0.01e-15)
        self.assertAlmostEqual(s["gps_us_per_day"]["net"], 38.6, delta=0.3)  # заводская поправка часов GPS
