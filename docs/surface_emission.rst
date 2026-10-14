Surface emission
=======================
PLATON can now calculate the emission from a rocky planet's surface, using realistic albedos newly measured in a lab by Caltech graduate student Kim Paragas (see Paragas et al. 2025 for more details).

To try out this feature, take a look at examples/surface_example.py. One begins by initializing an eclipse depth calculator with either the "HES2012" (Hu, Ehlmann, Seager 2012) spectral library or the new "Paragas" library::

  calc = EclipseDepthCalculator(surface_library="Paragas")

Optionally, one can estimate the surface temperature of a chosen surface_type from energy balance before computing the secondary eclipse depths by first retrieving the appropriate stellar spectrum and then calling calc_surface_temp::

    stellar_fluxes_orig, _ = calc.atm.get_stellar_spectrum(calc.atm.orig_lambda_grid, T_star, stellar_blackbody=False)
    surface_temp = calc.calc_surface_temp(surface_type, stellar_fluxes_orig, semi_major_axis / star_radius)

Then, call compute_depths with all the ordinary eclipse calculator arguments, but also the surface arguments::

  wavelengths, depths, info_dict = calc.compute_depths(...,
    surface_pressure = Psurf, surface_type=surface_type, semimajor_axis=semi_major_axis, surface_temp = surface_temp)

surface_temp can be None, in which case the temperature will be automatically calculated from energy balance.
The available surface types are the columns in data/HES2012/hemi_refls.csv or data/Paragas/hemi_refls.csv. The "Paragas" library surface types include a range of textures.

One can also see the surface temperature after the run::

    surface_temp = info_dict['surface_temperature']

A commonly asked question is **how do I generate the emission spectrum of an airless planet?**  We have yet to implement a simple way to do this, but it's easy enough to hack.
In surface_example.py, set Psurf to a negligible value, set the atmospheric temperature to the surface temperature, and give the atmosphere a composition that has no features::

  Psurf = 2e-4
  ...
  abundances = AbundanceGetter().get(0, 0.5)
  for key in abundances:
    abundances[key] *= 0
  abundances["N2"] += 1
  ...
  p.set_isothermal(surface_temp)

This is way overkill because the first and third steps alone should be enough to simulate an airless body, but better safe than sorry. 

Heat redistribution: the four-box model of Wirth et al.
--------------------------------------------------------

For tidally locked rocky planets, platon.wirth_model implements the four-box
model of `Wirth, Powell & Wordsworth (2026) <https://arxiv.org/abs/2601.12556>`_,
which gives the mean dayside and nightside temperatures of the surface and of
an optically thin atmosphere.  Unlike scalings based on the weak temperature
gradient approximation, it also applies to rapidly rotating planets.  The
grey infrared opacity and the mean molecular weight come from PLATON's
opacities and chemistry::

  from platon.wirth_model import WirthModel
  model = WirthModel()
  result = model.compute(Rs, T_star, semi_major_axis, Mp, Rp, Psurf,
                         logZ=None, CO_ratio=None, gases=["CO2"], vmrs=[1.0],
                         alpha="rotating", rotation_period=period, cp_over_R=4.5)

alpha, the day-to-night flow speed over the sound speed, can be a number
(the paper explores 0.05-1), "wtg" for slowly rotating planets, or
"rotating" for the Coriolis-dominated regime.  The result holds the four
temperatures (T_surface_day, T_atm_day, T_atm_night, T_surface_night),
brightness temperatures, and tau, the infrared optical depth; the model is
only valid for tau < 1.

The grey opacity is chosen so that the column absorbs the same fraction of
the surface's thermal emission as with the full opacities.  The paper
defines it as the Planck mean (method="planck" in grey_opacity), but for
line absorbers such as CO2 that is orders of magnitude larger and makes any
atmosphere optically thick.

To compute an eclipse spectrum, pass the dayside surface temperature as
surface_temp, with an isothermal atmosphere at T_atm_day::

  p.set_isothermal(result["T_atm_day"])
  wavelengths, depths, info_dict = calc.compute_depths(p, Rs, Mp, Rp, T_star, ..., surface_pressure=Psurf,
      surface_type=surface_type, semimajor_axis=semi_major_axis, surface_temp=result["T_surface_day"])
