# References

## Paper being replicated
- Bahmani, M.H., et al. (2018). "Investigation of turbulent heat transfer and
  nanofluid flow in a double pipe heat exchanger." *Advanced Powder Technology*
  29(2), 273-282.

## Numerical methods
- Van Doormaal, J.P., Raithby, G.D. (1984). "Enhancements of the SIMPLE method
  for predicting incompressible fluid flows." *Numerical Heat Transfer* 7,
  147-163. — SIMPLEC pressure-velocity coupling (`nanofluid_hx/flow/`).
- Patankar, S.V. (1980). *Numerical Heat Transfer and Fluid Flow.* Hemisphere.
  — staggered-grid FVM, implicit under-relaxation, upwind convection.
- Launder, B.E., Spalding, D.B. (1974). "The numerical computation of turbulent
  flows." *Comput. Methods Appl. Mech. Eng.* 3, 269-289. — standard k-epsilon
  model and equilibrium log-law wall functions
  (`nanofluid_hx/flow/wall_function.py`, `nanofluid_hx/flow/k_epsilon.py`).
- Versteeg, H.K., Malalasekera, W. (2007). *An Introduction to Computational
  Fluid Dynamics: The Finite Volume Method*, 2nd ed. Pearson.

## Validation benchmarks
- Hagen-Poiseuille exact solution: laminar centerline velocity and f = 64/Re
  (`tests/test_simplec_laminar_poiseuille.py`).
- Blasius correlation: f = 0.316 Re^-0.25 (Re < 2e4) / 0.184 Re^-0.20
  (Re > 2e4) (`tests/test_simplec_turbulent_pipe.py`,
  `tests/test_kepsilon_pipe.py`).
- Pak, B.-C., Cho, Y.-I. (1998). "Hydrodynamic and heat transfer study of
  dispersed fluids with submicron metallic oxide particles."
  *Experimental Heat Transfer* 11, 151-170. — paper's Fig. 3 validation
  benchmark (Stage 4).

## Nanofluid property correlations
- Density and specific heat: phase mixture rules. Viscosity and conductivity:
  polynomial correlations as adopted by Bahmani et al. (2018), Eqs. 7-10
  (`nanofluid_hx/properties.py`).
