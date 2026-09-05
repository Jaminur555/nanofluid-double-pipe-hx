"""Energy balance with the paper's flow closure (SIMPLEC + k-epsilon).

Mirror of test_energy_balance.py, but with SimplecFlow and FULL radial
defaults (regraded 15/5/16 mesh). ONE SimplecFlow pair is reused for both
arrangements -- providers are arrangement-agnostic (decision E).
"""
import pytest

from nanofluid_hx import AxisymmetricMesh, MaterialProperties, ThermalSolver
from nanofluid_hx.flow.coupling import SimplecFlow
from nanofluid_hx.postprocessing import evaluate_case


@pytest.fixture(scope="module")
def fields():
    mesh = AxisymmetricMesh(Nz=60)              # radial defaults 15/5/16
    pi, po = MaterialProperties(0.05), MaterialProperties(0.0)
    fd = SimplecFlow(mesh, pi, po, 20000, 20000)
    return mesh, fd


@pytest.fixture(scope="module")
def results(fields):
    """Both arrangements from the SAME flow fields (one flow solve)."""
    out = {}
    for parallel_flow in (True, False):
        mesh, fd = fields
        solver = ThermalSolver(mesh, fd, parallel_flow=parallel_flow)
        solver.assemble_system()
        T = solver.solve()
        out[parallel_flow] = evaluate_case(
            mesh, fd, T, parallel_flow, solver.T_hot_in, solver.T_cold_in)
    return out


def test_energy_balance_parallel(results):
    m = results[True]
    assert m["Q_hot"] == pytest.approx(m["Q_cold"], rel=0.02)


def test_energy_balance_counter(results):
    m = results[False]
    assert m["Q_hot"] == pytest.approx(m["Q_cold"], rel=0.02)


def test_ordering_parallel(results):
    m = results[True]
    assert 285.0 < m["T_f_out"] < m["T_nf_out"] < 350.0


def test_outlets_bounded_counter(results):
    m = results[False]
    assert 285.0 < m["T_f_out"] < 350.0
    assert 285.0 < m["T_nf_out"] < 350.0
    assert 0.0 < m["effectiveness"] < 1.0


def test_counterflow_directional(results):
    """Counterflow must out-cool parallel: colder cold-outlet, higher eps."""
    par, ctr = results[True], results[False]
    assert ctr["T_f_out"] > par["T_f_out"]
    assert ctr["effectiveness"] >= par["effectiveness"] - 0.02
