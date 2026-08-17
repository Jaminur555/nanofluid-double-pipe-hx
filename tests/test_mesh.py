import numpy as np
from nanofluid_hx import AxisymmetricMesh

def test_total_volume_conservation():
    mesh = AxisymmetricMesh(Nr_inner=15, Nr_wall=5, Nr_outer=15, Nz=150)
    assert np.isclose(np.sum(mesh.V), np.pi * mesh.r3 ** 2 * mesh.L, rtol=1e-10)


def test_zone_map_covers_all_cells():
    mesh = AxisymmetricMesh()
    assert len(mesh.zone_map) == mesh.Nr
    assert set(np.unique(mesh.zone_map)) == {0, 1, 2}