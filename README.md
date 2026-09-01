# Nanofluid Double-Pipe Heat Exchanger Solver

2D axisymmetric finite-volume thermal solver for turbulent Al<sub>2</sub>O<sub>3</sub>-water
nanofluid flow in a double-pipe heat exchanger (parallel & counter flow).

## Physics
- Three zones: inner nanofluid pipe (0-13 mm), steel wall (13-15 mm), water annulus (15-25 mm)
- `flow/`: staggered-grid SIMPLEC momentum solver with standard k-&epsilon; turbulence and
  log-law wall functions &mdash; validated vs. exact Poiseuille (&lt;2%) and Blasius (2.5&ndash;4.5%)
- FVM energy equation, upwind convection, conjugate wall, sparse direct solve
- Legacy fast mode: prescribed 1/7<sup>th</sup> power-law velocity + mixing-length eddy diffusivity
- Outputs: temperature field, Nu, LMTD, overall U, effectiveness

## Structure
```
nanofluid_hx/
    flow/           SIMPLEC + k-epsilon flow solver (staggered grid, wall functions)
    turbulence/     mixing_length (legacy fast approximate mode)
    ...             properties, mesh, solver, postprocessing, plotting
scripts/            run_single_case.py, run_parameter_sweep.py, run_stage1_pipe_validation.py
tests/              pytest suite (mesh, properties, energy balance, SIMPLEC, k-epsilon)
results/            generated figures
docs/               references
```

## Install
```bash
pip install -e .
```

## Usage
```bash
python scripts/run_single_case.py        # both flow arrangements, saves contours
python scripts/run_parameter_sweep.py    # Re x phi sweep, Nu & effectiveness
pytest                                   # run the test suite
```

## Results
![Parallel flow](results/parallel_contour.png)
![Counter flow](results/counter_contour.png)
![Parameter sweep](results/sweep_nu_effectiveness.png)

## Roadmap
- [x] SIMPLEC pressure-velocity coupling (`nanofluid_hx/flow/`)
- [x] k-&epsilon; turbulence model with wall functions (`nanofluid_hx/flow/k_epsilon.py`)
- [ ] Couple solved flow field into the double-pipe thermal solver (Stage 3)
- [ ] Validation vs. Pak &amp; Cho; reproduce paper Figs. 4&ndash;12 (Stage 4)
- [ ] Temperature-dependent properties (beyond paper scope)

## References
See [docs/reference.md](docs/reference.md).

## License
MIT  see [LICENSE](LICENSE).
