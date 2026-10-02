import unittest
import numpy as np
import matplotlib
matplotlib.use("Agg")

from platon.transit_depth_calculator import TransitDepthCalculator
from platon.eclipse_depth_calculator import EclipseDepthCalculator
from platon.combined_retriever import CombinedRetriever
from platon.retrieval_result import RetrievalResult
from platon.TP_profile import Profile
from platon.constants import R_sun, R_jup, M_jup

SPECIES = ["H2O", "CO2", "CO", "CH4"]
GASES = ["H2-He", "H2O", "CO2"]
VMRS = [0.98, 0.015, 0.005]
BINS = np.array([[a, b] for a, b in zip(np.linspace(1e-6, 5e-6, 21)[:-1],
                                        np.linspace(1e-6, 5e-6, 21)[1:])])


class TestMeanMolecularWeight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calc = TransitDepthCalculator(include_opacities=SPECIES)
        masses = cls.calc.atm.mass_data
        cls.self_consistent_mmw = sum(v * masses[g] for g, v in zip(GASES, VMRS))

    def compute(self, **kwargs):
        return self.calc.compute_depths(R_sun, M_jup, R_jup, 1200, logZ=None,
                                        CO_ratio=None, gases=GASES, vmrs=VMRS,
                                        full_output=True, **kwargs)

    def test_self_consistent_value_changes_nothing(self):
        _, depths, _ = self.compute()
        _, mmw_depths, info = self.compute(mmw=self.self_consistent_mmw)
        self.assertTrue(np.allclose(depths, mmw_depths, rtol=1e-5, atol=0))
        self.assertTrue(np.allclose(info["mu_profile"], self.self_consistent_mmw))

    def test_scale_height(self):
        # For an isothermal atmosphere of constant mean molecular weight,
        # 1/r - 1/Rp is proportional to 1/mmw
        _, _, info1 = self.compute(mmw=2.3)
        _, _, info2 = self.compute(mmw=4.6)
        height1 = 1 / info1["radii"] - 1 / R_jup
        height2 = 1 / info2["radii"] - 1 / R_jup
        self.assertTrue(np.allclose(height2, height1 * 2.3 / 4.6, rtol=1e-9, atol=0))
        self.assertTrue(np.all(info2["mu_profile"] == 4.6))

    def test_eclipse(self):
        calc = EclipseDepthCalculator(include_opacities=SPECIES)
        p = Profile()
        p.set_isothermal(1200)
        _, depths, info = calc.compute_depths(p, R_sun, M_jup, R_jup, 5700, logZ=None, CO_ratio=None,
                                              gases=GASES, vmrs=VMRS, mmw=3.0, full_output=True)
        self.assertTrue(np.all(np.isfinite(depths)))
        self.assertTrue(np.all(info["mu_profile"] == 3.0))

    def test_invalid(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                self.compute(mmw=bad)


class TestRetrieveMeanMolecularWeight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calc = TransitDepthCalculator(include_opacities=SPECIES)
        cls.calc.change_wavelength_bins(BINS)
        _, cls.depths, _ = cls.calc.compute_depths(
            R_sun, M_jup, R_jup, 1200, logZ=None, CO_ratio=None,
            gases=["H2O", "CO2"], vmrs=[1e-3, 1e-4], mmw=2.4)
        cls.errors = np.ones(len(cls.depths)) * 30e-6

    def fit_info(self, mode="vmr", **kwargs):
        fit_info = CombinedRetriever.get_default_fit_info(
            Rs=R_sun, Mp=M_jup, Rp=R_jup, T=1200, logZ=None, CO_ratio=None,
            fit_vmr=(mode == "vmr"), fit_clr=(mode == "clr"), **kwargs)
        fit_info.add_uniform_fit_param("Rp", 0.9 * R_jup, 1.1 * R_jup)
        return fit_info

    def ln_like(self, fit_info, **values):
        params = np.array([values[name] for name in fit_info.fit_param_names])
        return CombinedRetriever()._ln_like(params, self.calc, None, fit_info,
                                            self.depths, self.errors, None, None)

    def test_vmr_remainder(self):
        fit_info = self.fit_info(mmw=2.4)
        fit_info.add_gases_vmr(["H2O", "CO2"], 1e-12, 1, remainder=True)
        fit_info.add_uniform_fit_param("mmw", 1, 50)
        self.assertEqual(fit_info.fit_param_names, ["Rp", "log_H2O", "log_CO2", "mmw"])

        # H2O and CO2 at 10% each weigh 6.2 amu, leaving 0.8 of the atmosphere
        good = self.ln_like(fit_info, Rp=R_jup, log_H2O=-1, log_CO2=-1, mmw=10)
        self.assertTrue(np.isfinite(good))
        # Remainder gas would weigh (2.3 - 6.2) / 0.8 < 0 amu
        self.assertEqual(self.ln_like(fit_info, Rp=R_jup, log_H2O=-1, log_CO2=-1, mmw=2.3), -np.inf)
        # VMRs add up to more than 1
        self.assertEqual(self.ln_like(fit_info, Rp=R_jup, log_H2O=np.log10(0.6),
                                      log_CO2=np.log10(0.5), mmw=40), -np.inf)
        vmr, weight = CombinedRetriever.get_remainder(["H2O", "CO2"], [0.1, 0.1], 10,
                                                      self.calc.atm.mass_data)
        self.assertAlmostEqual(vmr, 0.8)
        self.assertAlmostEqual(weight, (10 - 0.1 * 18.0152 - 0.1 * 44.010) / 0.8)

    def test_inv_mmw_matches_mmw(self):
        mmw_info = self.fit_info(mmw=2.4)
        mmw_info.add_gases_vmr(["H2O", "CO2"], 1e-12, 1, remainder=True)
        mmw_info.add_uniform_fit_param("mmw", 1, 50)
        inv_info = self.fit_info(inv_mmw=1 / 2.4)
        inv_info.add_gases_vmr(["H2O", "CO2"], 1e-12, 1, remainder=True)
        inv_info.add_uniform_fit_param("inv_mmw", 0.02, 1)

        values = dict(Rp=R_jup, log_H2O=-3, log_CO2=-4)
        self.assertAlmostEqual(self.ln_like(mmw_info, mmw=4.0, **values),
                               self.ln_like(inv_info, inv_mmw=0.25, **values))
        # Too heavy for the trace gases to leave a remainder of positive weight
        self.assertEqual(self.ln_like(inv_info, inv_mmw=0.9, Rp=R_jup, log_H2O=-0.5, log_CO2=-0.5), -np.inf)

    def test_clr_remainder(self):
        fit_info = self.fit_info(mode="clr", mmw=30)
        fit_info.add_gases_clr(["H2O", "CO2"], remainder=True)
        self.assertEqual(fit_info.fit_param_names, ["Rp", "clr_H2O", "clr_CO2"])
        # CLRs of 0: H2O, CO2 and the remainder each make up 1/3, so the
        # remainder weighs 3 * (mmw - (18.0152 + 44.010) / 3)
        self.assertTrue(np.isfinite(self.ln_like(fit_info, Rp=R_jup, clr_H2O=0, clr_CO2=0)))
        fit_info.all_params["mmw"].best_guess = 10
        self.assertEqual(self.ln_like(fit_info, Rp=R_jup, clr_H2O=0, clr_CO2=0), -np.inf)

    def test_validation(self):
        both = self.fit_info(mmw=2.3, inv_mmw=1 / 2.3)
        equilibrium = CombinedRetriever.get_default_fit_info(Rs=R_sun, Mp=M_jup, Rp=R_jup, T=1200, mmw=2.3)
        no_mmw = self.fit_info()
        no_mmw.add_gases_vmr(["H2O", "CO2"], 1e-12, 1e-1, remainder=True)
        negative = self.fit_info(inv_mmw=0.5)
        negative.add_uniform_fit_param("inv_mmw", -1, 1)
        for fit_info in (both, equilibrium, no_mmw, negative):
            with self.assertRaises(ValueError):
                CombinedRetriever._validate_mmw_params(fit_info)

        valid = self.fit_info(inv_mmw=0.5)
        valid.add_gases_vmr(["H2O", "CO2"], 1e-12, 1e-1, remainder=True)
        valid.add_uniform_fit_param("inv_mmw", 0.05, 0.5)
        CombinedRetriever._validate_mmw_params(valid)

    def test_retrieval(self):
        fit_info = self.fit_info(inv_mmw=1 / 2.4)
        fit_info.add_gases_vmr(["H2O", "CO2"], 1e-8, 1e-1, remainder=True)
        fit_info.add_uniform_fit_param("inv_mmw", 0.05, 0.5)
        result = CombinedRetriever().run_dynesty(BINS, self.depths, self.errors, None, None, None,
                                                 fit_info, maxcall=300, num_final_samples=10)
        self.assertTrue(isinstance(result, RetrievalResult))
        self.assertIn("inv_mmw", result.fit_info.fit_param_names)
        self.assertTrue(result.best_fit_transit_dict["remainder_mmw"] >= 0)


if __name__ == '__main__':
    unittest.main()
