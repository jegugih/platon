import unittest
import numpy as np

from platon.transit_depth_calculator import TransitDepthCalculator
from platon.combined_retriever import CombinedRetriever, NUM_TRANSIT_OFFSETS
from platon.constants import R_sun, R_jup, M_jup

EDGES = np.linspace(1e-6, 5e-6, 21)
BINS = np.array([EDGES[:-1], EDGES[1:]]).T
ERRORS = np.full(len(BINS), 50e-6)
# One range per offset; offsets add where ranges overlap (offsets 1 and 5)
RANGES = [(0, 6), (4, 8), (8, 12), (12, 16), (2, 20)]
OFFSETS = [100e-6, -200e-6, 300e-6, -400e-6, 500e-6]


def offset_kwargs(ranges, offsets):
    kwargs = {}
    for n, ((start, end), offset) in enumerate(zip(ranges, offsets), start=1):
        value, start_name, end_name = CombinedRetriever._transit_offset_names(n)
        kwargs.update({value: offset, start_name: start, end_name: end})
    return kwargs


class TestTransitOffsets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calc = TransitDepthCalculator(include_opacities=["H2O", "CO2"])
        cls.calc.change_wavelength_bins(BINS)

    def model(self, **kwargs):
        fit_info = CombinedRetriever.get_default_fit_info(R_sun, M_jup, R_jup, 1200, **kwargs)
        fit_info.add_uniform_fit_param("T", 300, 2000)
        depths, _, _, _ = CombinedRetriever()._ln_like(
            [1200], self.calc, None, fit_info, np.zeros(len(BINS)), ERRORS,
            None, None, ret_best_fit=True)
        return fit_info, depths

    def test_offsets_shift_their_ranges(self):
        self.assertEqual(NUM_TRANSIT_OFFSETS, 5)
        _, plain = self.model()
        _, shifted = self.model(**offset_kwargs(RANGES, OFFSETS))
        expected = np.zeros(len(BINS))
        for (start, end), offset in zip(RANGES, OFFSETS):
            expected[start:end] += offset
        self.assertTrue(np.allclose(shifted - plain, expected, rtol=0, atol=1e-15))

    def test_likelihood_uses_offsets(self):
        # Data offset like the model: the residuals vanish
        _, plain = self.model()
        fit_info, shifted = self.model(**offset_kwargs(RANGES, OFFSETS))
        ln_like = CombinedRetriever()._ln_like(
            [1200], self.calc, None, fit_info, shifted, ERRORS, None, None)
        self.assertAlmostEqual(ln_like, np.sum(-0.5 * np.log(2 * np.pi * ERRORS**2)), places=6)

        # Without offsets, the same data fit worse
        no_offsets, _ = self.model()
        worse = CombinedRetriever()._ln_like(
            [1200], self.calc, None, no_offsets, shifted, ERRORS, None, None)
        self.assertLess(worse, ln_like - 10)

    def test_fitted_offset_needs_depths(self):
        for kwargs in ({}, {"offset3_start": 30, "offset3_end": 40}):
            fit_info = CombinedRetriever.get_default_fit_info(R_sun, M_jup, R_jup, 1200, **kwargs)
            fit_info.add_gaussian_fit_param("offset3_transit", 100e-6)
            with self.assertRaises(ValueError):
                CombinedRetriever().run_dynesty(BINS, np.zeros(len(BINS)), ERRORS,
                                                None, None, None, fit_info)


if __name__ == '__main__':
    unittest.main()
