"""Stage 3c flow -> energy coupling: SIMPLEC + k-epsilon provider.

Runs the real Stage 1/2 SIMPLEC solver on node-identical sub-meshes built
from the thermal mesh's OWN faces (StaggeredPipeMesh.from_faces) -- the
staggered u-nodes land exactly on the thermal z-faces, so the thermal solver
consumes the flow fields with ZERO interpolation.

Exposed fields (what ThermalSolver reads):
    u_face (Nr, Nz+1)  staggered axial velocity on the thermal z-faces
    mu_t   (Nr, Nz)    dynamic eddy viscosity [Pa s]
    k_eff  (Nr, Nz)    effective conductivity: k + cp*mu_t/Pr_t per zone
                       (steel wall rows: k_s; eddy terms zero)

Fields are ARRANGEMENT-AGNOSTIC: always parallel orientation (every stream
+z, entrance at z = 0). There is deliberately no parallel_flow argument and
no orientation method -- counterflow is handled by
ThermalSolver.assemble_system(), which reverses the annulus z-columns.
"""

import numpy as np

from .staggered_mesh import StaggeredPipeMesh
from .simple_solver import run_simplec


class SimplecFlow:
    """The paper's flow closure: SIMPLEC + standard k-epsilon on both zones."""

    name = "simplec_k_epsilon"

    def __init__(self, mesh, props_inner, props_outer, Re_inner, Re_outer,
                 Pr_t=0.85, turbulence_model="k_epsilon",
                 alpha_u=0.5, alpha_v=0.5, alpha_p=1.0,
                 alpha_k=0.5, alpha_eps=0.5,
                 max_outer_iter=300, mass_tol=1e-6, vel_tol=1e-6,
                 verbose=False):

        self.mesh = mesh
        self.pi   = props_inner
        self.po   = props_outer
        self.Re_in  = Re_inner
        self.Re_out = Re_outer
        self.Pr_t = Pr_t

        # Zone slice indices on the thermal mesh
        i1 = mesh.Nr_inner
        i2 = i1 + mesh.Nr_wall
        self.i1, self.i2 = i1, i2

        # Node-identical sub-meshes: the staggered grids ARE slices of the
        # thermal mesh faces (inner pipe 0..r1, annulus r2..r3).
        self.pipe_mesh = StaggeredPipeMesh.from_faces(
            mesh.r_faces[:i1 + 1], mesh.z_faces, south_is_wall=False)
        self.annulus_mesh = StaggeredPipeMesh.from_faces(
            mesh.r_faces[i2:], mesh.z_faces, south_is_wall=True)

        # Inlet velocities from the Reynolds definitions via each sub-mesh's
        # own hydraulic diameter (pipe: 2*r1, annulus: 2*(r3 - r2))
        self.U_in_inner = (Re_inner * props_inner.mu_nf
                           / (props_inner.rho_nf * self.pipe_mesh.Dh))
        self.U_in_outer = (Re_outer * props_outer.mu_f
                           / (props_outer.rho_f * self.annulus_mesh.Dh))

        solver_kwargs = dict(turbulence_model=turbulence_model,
                             alpha_u=alpha_u, alpha_v=alpha_v, alpha_p=alpha_p,
                             alpha_k=alpha_k, alpha_eps=alpha_eps,
                             max_outer_iter=max_outer_iter,
                             mass_tol=mass_tol, vel_tol=vel_tol,
                             verbose=verbose)

        self.result_inner = run_simplec(self.pipe_mesh, props_inner.rho_nf,
                                        props_inner.mu_nf, self.U_in_inner,
                                        **solver_kwargs)
        self.result_outer = run_simplec(self.annulus_mesh, props_outer.rho_f,
                                        props_outer.mu_f, self.U_in_outer,
                                        **solver_kwargs)

        self._assemble_fields()

    def _assemble_fields(self):
        """Assemble the full-grid (Nr, ...) fields from the two sub-results."""
        mesh = self.mesh
        Nr, Nz = mesh.Nr, mesh.Nz
        i1, i2 = self.i1, self.i2
        Pr_t = self.Pr_t

        # Staggered u (Nr_zone, Nz+1) maps 1:1 onto thermal rows/faces.
        u_face = np.zeros((Nr, Nz + 1))
        u_face[:i1, :] = self.result_inner["u"]
        # wall rows i1:i2 stay 0 (solid)
        u_face[i2:, :] = self.result_outer["u"]

        # Dynamic eddy viscosity [Pa s]; laminar runs have no mu_t.
        mu_t = np.zeros((Nr, Nz))
        mu_t[:i1, :] = self.result_inner.get("mu_t", 0.0)
        mu_t[i2:, :] = self.result_outer.get("mu_t", 0.0)

        # Effective conductivity. UNIT NOTE: run_simplec's mu_t is DYNAMIC
        # [Pa s] -- the eddy conductivity is cp*mu_t/Pr_t with NO rho factor
        # (the legacy mixing length used rho*cp*nu_t; identical since
        # rho*nu_t = mu_t).
        k_eff = np.empty((Nr, Nz))
        k_eff[:i1, :] = self.pi.k_nf + self.pi.cp_nf * mu_t[:i1, :] / Pr_t
        k_eff[i1:i2, :] = self.pi.k_s
        k_eff[i2:, :] = self.po.k_f + self.po.cp_f * mu_t[i2:, :] / Pr_t

        self.u_face = u_face
        self.mu_t   = mu_t
        self.k_eff  = k_eff

        # Backward compatibility: developed outlet column as a 1-D profile
        self.u = u_face[:, -1].copy()

        # ---- Diagnostics (parallel orientation; consumers orient) ----
        # Radial velocity on the global r-faces: pipe covers faces 0..i1,
        # annulus covers faces i2..Nr; faces inside the wall stay 0.
        v = np.zeros((Nr + 1, Nz))
        v[:i1 + 1, :] = self.result_inner["v"]
        v[i2:, :]     = self.result_outer["v"]
        self.v = v

        k_field = np.zeros((Nr, Nz))
        k_field[:i1, :] = self.result_inner.get("k", 0.0)
        k_field[i2:, :] = self.result_outer.get("k", 0.0)
        self.k = k_field

        eps_field = np.zeros((Nr, Nz))
        eps_field[:i1, :] = self.result_inner.get("eps", 0.0)
        eps_field[i2:, :] = self.result_outer.get("eps", 0.0)
        self.eps = eps_field

        self.iterations = {"inner": self.result_inner["iterations"],
                           "outer": self.result_outer["iterations"]}
        self.converged  = {"inner": self.result_inner["converged"],
                           "outer": self.result_outer["converged"]}
