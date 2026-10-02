# PLanetary Atmospheric Tool for Observer Noobs
<img src="logo.png" height="315">

 [![codecov](https://codecov.io/gh/ideasrule/platon/branch/devel/graph/badge.svg)](https://codecov.io/gh/ideasrule/platon)
[![arXiv](http://img.shields.io/badge/arXiv-2004.09513-orange.svg)](https://arxiv.org/abs/2004.09513) 
[![arXiv](http://img.shields.io/badge/arXiv-1811.11761-orange.svg)](https://arxiv.org/abs/1811.11761)

**Feb 24, 2025: PLATON v6.3 is out!  The main difference from v6.2 is that we now include realistic emission from rocky surfaces using the emissivity library from [Paragas et al 2025](https://arxiv.org/abs/2502.04433).  The library contains 11 mineral compositions across 3 surface textures, and were obtained via lab reflectance measurements.**

**Oct 30, 2024: PLATON v6.2 is out, with major updates!  This corresponds most closely to the version described in [Zhang et al 2024](https://arxiv.org/abs/2410.22398).  Major changes include GPU support, free retrievals, surface emission, higher-resolution (R=20k) and more up-to-date opacities, leave-one-out cross-validation, pymultinest, and much better plotting tools.**

**February 2, 2020: PLEASE re-clone your repository if you cloned it before this
date!  On this date, all data files were deleted from the repository, including
from its history.  This shrank the repository's size from 1.2 GB to 36 MB,
which respects GitHub's size limit.**

**November 29, 2018: PLATON v3.0 is out!  Please check the release notes on GitHub for the list of new features.  Our paper actually describes v2.1, but v3 has a substantial number of improvements, and we strongly recommend everyone use it.**

**NEW (October 1): Version 2 is out, with a huge number of changes! Make sure to check out the release notes, and upgrade.**

PLATON is a fast and easy to use forward modelling and retrieval tool for
exoplanet atmospheres. It was written by Michael Zhang and Yayaati Chachan, and
is based on ExoTransmit by Eliza Kempton.

We welcome any feedback. Please send any comments, suggestions, or bug reports
to Michael Zhang: zmzhang "at" uchicago.edu
**
To find installation instructions and quickstart examples, read the
documentation at: https://platon.readthedocs.io

## MLX backend

On this branch the forward model runs on [MLX](https://github.com/ml-explore/mlx)
instead of NumPy/CuPy.  It has been tested on Apple silicon GPUs only, not on
MLX's CUDA backend.  The public API is unchanged and still returns NumPy
arrays.

Metal GPUs have no float64, so spectra are computed in float32 by default.
Densities, cross sections and abundances are rescaled or kept in log space so
that nothing under- or overflows, and the hydrostatic radii, ray path lengths
and T/P profiles are always computed in float64.  Against the float64 NumPy code,
transit and eclipse depths typically agree to 0.01 ppm or better.  The largest
difference seen was 0.13 ppm, at a single wavelength of a rocky planet's
eclipse spectrum, where two layers were almost equally good choices for the
photosphere.

To run everything in float64 on the CPU instead (e.g. for validation), call
this before creating a calculator:

    import platon
    platon.set_precision("float64")

Mie efficiencies (`_mie_multi_x.py`) are still computed in NumPy and cached,
and the hydrostatic solution and T/P profiles are computed in NumPy because
they involve only ~100 numbers, for which MLX's fixed cost per operation would
dominate.  Everything else that runs per model evaluation is in MLX.
Calculators in the same process share their opacity data.

*Now, these points of data  
make a beautiful line  
And we're out of beta  
We're releasing on time!*  
