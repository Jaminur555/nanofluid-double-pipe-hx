"""Thermal wall function (Jayatilleke T+): limits, conductance, solver coupling."""
import numpy as np
import pytest

from nanofluid_hx.flow.wall_function import t_plus, thermal_wall_h, YPLUS_VISCOUS


def test_t_plus_sublayer_is_molecular_conduction():
    # y+ -> 0: T+ = Pr*y+ so h = rho*cp*u_tau/T+ = k/y_P (pure conduction)
    y_plus = np.array([0.1, 1.0, 5.0])
    Pr, Pr_t = 5.0, 0.85
    assert np.allclose(t_plus(y_plus, Pr, Pr_t), Pr * y_plus)


def test_t_plus_log_layer_value():
    # Hand-computed Jayatilleke: Pr=1, Pr_t=0.9, y+=100
    Pr, Pr_t, yp = 1.0, 0.9, 100.0
    P = 9.24 * ((Pr / Pr_t) ** 0.75 - 1.0) * (1.0 + 0.28 * np.exp(-0.007 * Pr / Pr_t))
    expected = (Pr_t / 0.41) * np.log(yp) + P
    assert t_plus(yp, Pr, Pr_t) == pytest.approx(expected)
    assert t_plus(yp, Pr, Pr_t) > 0.0


def test_thermal_wall_h_positive_and_scales_with_velocity():
    # Higher near-wall velocity -> higher u_tau -> higher film coefficient
    Nz = 6
    u_slow = np.full(Nz + 1, 1.0)
    u_fast = np.full(Nz + 1, 5.0)
    y_P, rho, mu, cp, Pr, Pr_t = 1e-4, 1000.0, 8.9e-4, 4180.0, 5.0, 0.85
    h_slow = thermal_wall_h(u_slow, y_P, rho, mu, cp, Pr, Pr_t)
    h_fast = thermal_wall_h(u_fast, y_P, rho, mu, cp, Pr, Pr_t)
    assert h_slow.shape == (Nz,)
    assert np.all(h_slow > 0.0)
    assert np.all(h_fast > h_slow)


def test_wall_h_fed_to_solver_overrides_face_coupling():
    # On the tiny smoke mesh the SimplecFlow provider exposes per-column film
    # coefficients at both fluid-solid walls, and using them in the assembly
    # keeps Q_hot ~ Q_cold (conservative flux through the wall-function faces).
    from nanofluid_hx import AxisymmetricMesh, MaterialProperties, ThermalSolver
    from nanofluid_hx.turbulence import get_model
    from nanofluid_hx.postprocessing import evaluate_case

    mesh = AxisymmetricMesh(Nr_inner=8, Nr_wall=3, Nr_outer=8, Nz=40)
    fd = get_model("simplec_k_epsilon")(mesh, MaterialProperties(phi=0.05),
                                        MaterialProperties(phi=0),
                                        Re_inner=20000, Re_outer=20000)
    assert fd.wall_h is not None
    assert fd.wall_h["r1"].shape == (mesh.Nz,)
    assert fd.wall_h["r2"].shape == (mesh.Nz,)
    assert np.all(fd.wall_h["r1"] > 0.0)

    for parallel in (True, False):
        solver = ThermalSolver(mesh, fd, parallel_flow=parallel)
        solver.assemble_system()
        T = solver.solve()
        r = evaluate_case(mesh, fd, T, parallel_flow=parallel)
        assert r["Q_hot"] == pytest.approx(r["Q_cold"], rel=0.02)
        assert r["Nu_nf"] > 0.0
