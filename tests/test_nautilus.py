import os
import tempfile
import unittest
import numpy as np

from platon.transit_depth_calculator import TransitDepthCalculator
from platon.combined_retriever import CombinedRetriever
from platon.constants import R_sun, R_jup, M_jup

EDGES = np.linspace(1e-6, 5e-6, 21)
BINS = np.array([EDGES[:-1], EDGES[1:]]).T


@unittest.skipIf(__import__("importlib").util.find_spec("nautilus") is None, "nautilus is not installed")
class TestNautilus(unittest.TestCase):
    def test_retrieval(self):
        calc = TransitDepthCalculator(include_opacities=["H2O", "CO2"])
        calc.change_wavelength_bins(BINS)
        _, depths, _ = calc.compute_depths(R_sun, M_jup, R_jup, 1200)
        errors = np.full(len(depths), 50e-6)

        fit_info = CombinedRetriever.get_default_fit_info(R_sun, M_jup, R_jup, 1200)
        fit_info.add_uniform_fit_param("Rp", 0.95 * R_jup, 1.05 * R_jup)
        fit_info.add_uniform_fit_param("T", 800, 1600)

        calls = {}
        for discard in [True, False]:
            cwd = os.getcwd()
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                try:
                    result = CombinedRetriever().run_nautilus(
                        BINS, depths, errors, None, None, None, fit_info, nlive=300, n_eff=500,
                        num_final_samples=10, include_opacities=["H2O", "CO2"], verbose=False, seed=0,
                        discard_exploration=discard)
                finally:
                    os.chdir(cwd)

            self.assertEqual(result.retrieval_type, "nautilus")
            self.assertTrue(np.isfinite(result.final_logz))
            self.assertAlmostEqual(np.sum(result.weights), 1)
            for i, truth in enumerate([R_jup, 1200]):
                lo, hi = np.percentile(result.equal_samples[:, i], [0.5, 99.5])
                self.assertTrue(lo <= truth <= hi)
            self.assertEqual(len(result.pointwise_lnlikes), 10)
            # About as many equally weighted samples as effective samples
            self.assertGreater(len(result.equal_samples), 0.8 * result.effective_sample_size)
            calls[discard] = result.ncall
        # Discarding the exploration phase needs more evaluations
        self.assertGreater(calls[True], calls[False])


if __name__ == '__main__':
    unittest.main()
