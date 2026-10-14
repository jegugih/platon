import sys
import unittest
from unittest import mock
import numpy as np

from platon.combined_retriever import CombinedRetriever
from platon.constants import R_sun, R_jup, M_jup

EDGES = np.linspace(1e-6, 5e-6, 11)
BINS = np.array([EDGES[:-1], EDGES[1:]]).T
DEPTHS = np.full(len(BINS), 0.01)
ERRORS = np.full(len(BINS), 50e-6)


class Created(Exception):
    pass


class TestRetrievalOptions(unittest.TestCase):
    def calculator_kwargs(self, run, **kwargs):
        '''The arguments that `run` creates its transit and eclipse calculators with'''
        fit_info = CombinedRetriever.get_default_fit_info(R_sun, M_jup, R_jup, 1200)
        fit_info.add_uniform_fit_param("T", 300, 2000)
        created = []
        def calculator(**calc_kwargs):
            created.append(calc_kwargs)
            if len(created) == 2:
                raise Created()
        with mock.patch("platon.combined_retriever.TransitDepthCalculator", side_effect=calculator), \
             mock.patch("platon.combined_retriever.EclipseDepthCalculator", side_effect=calculator):
            with self.assertRaises(Created):
                getattr(CombinedRetriever(), run)(BINS, DEPTHS, ERRORS, BINS, DEPTHS, ERRORS,
                                                  fit_info, **kwargs)
        return created

    def test_options_reach_calculators(self):
        runs = ["run_emcee", "run_dynesty"]
        try:
            import nautilus
            runs.append("run_nautilus")
        except ImportError:
            pass
        for run in runs:
            for calc_kwargs in self.calculator_kwargs(
                    run, include_opacities=["H2O", "CO2"], downsample=5, rad_method="xsec"):
                self.assertEqual(calc_kwargs["include_opacities"], ["H2O", "CO2"])
                self.assertEqual(calc_kwargs["downsample"], 5)
                self.assertEqual(calc_kwargs["method"], "xsec")

    def test_default_opacities(self):
        # Without include_opacities, the calculators' default list is used
        for calc_kwargs in self.calculator_kwargs("run_dynesty"):
            self.assertNotIn("include_opacities", calc_kwargs)
            self.assertEqual(calc_kwargs["downsample"], 1)

    def test_multinest(self):
        # The calculators are created before MultiNest runs, so pymultinest
        # need not be installed
        with mock.patch.dict(sys.modules, {"pymultinest": mock.MagicMock()}):
            for calc_kwargs in self.calculator_kwargs(
                    "run_multinest", include_opacities=["H2O"], downsample=2):
                self.assertEqual(calc_kwargs["include_opacities"], ["H2O"])
                self.assertEqual(calc_kwargs["downsample"], 2)


if __name__ == '__main__':
    unittest.main()
