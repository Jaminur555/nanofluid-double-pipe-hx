import pytest
from nanofluid_hx import AxisymmetricMesh, MaterialProperties, ThermalSolver
from nanofluid_hx.turbulence import get_model
from nanofluid_hx.postprocessing import evaluate_case


@pytest.fixture(scope="module")
def solved():
    mesh = AxisymmetricMesh(Nr_inner=10, Nr_wall=4, Nr_outer=10, Nz=60)
    pi, po = MaterialProperties(0.05), MaterialProperties(0.0)
    fd = get_model("mixing_length")(mesh, pi, po, 20000, 20000)
    solver = ThermalSolver(mesh, fd, parallel_flow=True)
    solver.assemble_system()
    return mesh, fd, solver, solver.solve()


def test_energy_balance(solved):
    mesh, fd, solver, T = solved
    m = evaluate_case(mesh, fd, T, True, solver.T_hot_in, solver.T_cold_in)
    assert m["Q_hot"] == pytest.approx(m["Q_cold"], rel=0.02)


def test_outlet_temps_bounded(solved):
    mesh, fd, solver, T = solved
    m = evaluate_case(mesh, fd, T, True, solver.T_hot_in, solver.T_cold_in)
    assert solver.T_cold_in < m["T_f_out"] < m["T_nf_out"] < solver.T_hot_in


@pytest.fixture(scope="module")
def solved_counter():
    mesh = AxisymmetricMesh(Nr_inner=10, Nr_wall=4, Nr_outer=10, Nz=60)
    pi, po = MaterialProperties(0.05), MaterialProperties(0.0)
    fd = get_model("mixing_length")(mesh, pi, po, 20000, 20000)
    solver = ThermalSolver(mesh, fd, parallel_flow=False)
    solver.assemble_system()
    return mesh, fd, solver, solver.solve()


def test_energy_balance_counter_flow(solved_counter):
    mesh, fd, solver, T = solved_counter
    m = evaluate_case(mesh, fd, T, False, solver.T_hot_in, solver.T_cold_in)
    assert m["Q_hot"] == pytest.approx(m["Q_cold"], rel=0.02)


def test_counter_flow_outlet_temps_bounded(solved_counter):
    mesh, fd, solver, T = solved_counter
    m = evaluate_case(mesh, fd, T, False, solver.T_hot_in, solver.T_cold_in)
    assert solver.T_cold_in < m["T_f_out"] < solver.T_hot_in
    assert solver.T_cold_in < m["T_nf_out"] < solver.T_hot_in
    assert 0.0 < m["effectiveness"] < 1.0