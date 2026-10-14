import unittest
import warnings
import numpy as np

from platon.wirth_model import WirthModel
from platon.constants import R_sun, R_earth, M_earth, AU, k_B, AMU

# LHS 3844b, with the grey CO2 of Wirth et al. (Table 1)
PLANET = dict(star_radius=0.19 * R_sun, T_star=3036, semimajor_axis=0.00622 * AU,
              planet_mass=2.25 * M_earth, planet_radius=1.303 * R_earth)
PERIOD = 0.4629 * 86400
CO2 = dict(kappa=5e-5, mmw=k_B / (189 * AMU), cp_over_R=849 / 189)


class TestWirthModel(unittest.TestCase):
    def test_energy_balance(self):
        for alpha in [0.05, 0.3, 1, "wtg"]:
            for p_s in [1e2, 1e4, 1e5]:
                r = WirthModel.solve(surface_pressure=p_s, alpha=alpha, **PLANET, **CO2)
                self.assertTrue(r["converged"])
                emitted = r["T_emit_day"]**4 + r["T_emit_night"]**4
                self.assertAlmostEqual(emitted / r["T_0"]**4, 1, places=6)

    def test_limits_and_regimes(self):
        # A vanishing atmosphere leaves a bare-rock dayside
        r = WirthModel.solve(surface_pressure=1, alpha=0.3, **PLANET, **CO2)
        self.assertAlmostEqual(r["T_surface_day"] / r["T_0"], 1, places=3)

        # Rotation-dominated flow carries less heat to the nightside
        wtg = WirthModel.solve(surface_pressure=1e4, alpha="wtg", **PLANET, **CO2)
        rotating = WirthModel.solve(surface_pressure=1e4, alpha="rotating",
                                    rotation_period=PERIOD, **PLANET, **CO2)
        self.assertLess(rotating["alpha"], wtg["alpha"])
        self.assertLess(rotating["T_surface_night"], wtg["T_surface_night"] - 50)

        # Slow transport: the dayside surface warms with pressure (Sect. 4.1)
        T_day = [WirthModel.solve(surface_pressure=p, alpha=0.05, **PLANET, **CO2)["T_surface_day"]
                 for p in [1e3, 1e4, 1e5]]
        self.assertTrue(np.all(np.diff(T_day) > 0))

    def test_grey_opacity(self):
        model = WirthModel(include_opacities=["CO2"])
        T_eq = PLANET["T_star"] * np.sqrt(PLANET["star_radius"] / (2 * PLANET["semimajor_axis"]))
        args = (PLANET["star_radius"], PLANET["planet_mass"], PLANET["planet_radius"])
        co2 = dict(logZ=None, CO_ratio=None, gases=["CO2"], vmrs=[1.0])
        taus = [model.grey_opacity(*args, p, T_eq, **co2)["tau"] for p in [1e3, 1e4, 1e5]]
        self.assertTrue(np.all(np.diff(taus) > 0))
        # CO2 becomes optically thick near 1 bar, as with the paper's opacity
        self.assertTrue(0.5 < taus[2] < 2)
        grey = model.grey_opacity(*args, 1e5, T_eq, **co2)
        self.assertAlmostEqual(grey["mmw"], 44.0, places=0)
        # The Planck mean is dominated by the band cores
        planck = model.grey_opacity(*args, 1e3, T_eq, method="planck", **co2)
        self.assertGreater(planck["tau"], 100)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = model.compute(PLANET["star_radius"], PLANET["T_star"], PLANET["semimajor_axis"],
                              PLANET["planet_mass"], PLANET["planet_radius"], 1e4,
                              alpha="rotating", rotation_period=PERIOD, **co2)
        self.assertTrue(r["converged"])
        self.assertAlmostEqual(r["tau"], taus[1], places=6)


if __name__ == '__main__':
    unittest.main()
