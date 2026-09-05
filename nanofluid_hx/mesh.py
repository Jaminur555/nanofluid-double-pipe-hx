import numpy as np


def one_sided_power(start, end, n, power):
    """n cells between start and end via a power-law map s**power.

    power > 1 -> faces follow s**power which rises fast, so cells are FINE
    near `start` and COARSE near `end`. Use power < 1 to reverse that.
    """
    s = np.linspace(0.0, 1.0, n + 1)
    return start + (end - start) * s ** power


def two_sided_mirrored(start, end, n, power):
    """n cells between start and end, coarse at BOTH boundaries, fine at mid-gap.

    Cell widths from the power law are mirrored about the mid-gap, so both
    wall-adjacent cells are equal and largest. Works for odd n too: the even
    sequence is built first and the two mid cells are merged (sum preserved,
    so the last face still lands exactly on `end`).
    """
    W = 0.5 * (end - start)
    m = n + (n % 2)                      # even cell count to build
    half = m // 2
    s = np.linspace(0.0, 1.0, half + 1)
    w_low = np.diff(s ** power) * W       # widths, coarse at wall, sum = W
    widths = np.concatenate([w_low, w_low[::-1]])
    if n % 2:
        k = m // 2               # merge the two middle cells, keep symmetry
        widths = np.concatenate([widths[:k - 1], [widths[k - 1] + widths[k]], widths[k + 1:]])
    return start + np.concatenate([[0.0], np.cumsum(widths)])


class AxisymmetricMesh:
    def __init__(self, Nr_inner = 15, Nr_wall = 5, Nr_outer = 16, Nz = 150, L = 2):
        """
        Generates a non-uniform 2D axissymmetric cylinder mesh.

        Nr_inner: Number of radial grid cells in inner fluid (0-13) mm
        Nr_wall : Number of radial grid cells in steel wall (13-15) mm
        Nr_outer: Number of radial grid cells in outer fluid (15-25) mm
        Nz      : Number of axial grid cells along (L = 2 meters)

        Grading targets equilibrium wall functions: the wall-adjacent FLUID
        cells are the largest in their zone (first cell-center y+ ~ 16-140
        over Re 1e4-1e5), never sublayer-sized.
        """
        self.Nr_inner = Nr_inner
        self.Nr_wall  = Nr_wall
        self.Nr_outer = Nr_outer
        self.Nz       = Nz
        self.Nr       = Nr_inner + Nr_wall + Nr_outer
        self.L        = L

        # Physical Boundaries (in meters)
        self.r0 = 0.0
        self.r1 = 0.013    # Inner fluid interface
        self.r2 = 0.015    # Outer fluid interface
        self.r3 = 0.025    # Outer pipe wall

        # Generate Non-Unifrom Radial Coordinates (r_faces)
        self.r_faces = self.generate_radial_faces()

        # Generate Uniform Axial Coordinates (z_faces)
        self.z_faces = np.linspace(0.0, self.L, self.Nz + 1)

        # Calculate Cell Centers
        self.r_center = (self.r_faces[:-1] + self.r_faces[1:]) * 0.5
        self.z_center = (self.z_faces[:-1] + self.z_faces[1:]) * 0.5

        # Generate Cell Volumes and Faces Area
        self.V   = np.zeros((self.Nr, self.Nz))
        self.A_e = np.zeros((self.Nr, self.Nz))    # East face (constant - z, right side)
        self.A_w = np.zeros((self.Nr, self.Nz))    # West face (consta-z, left side)
        self.A_n = np.zeros((self.Nr, self.Nz))    # North face (Constant-r, outer cylinder)
        self.A_s = np.zeros((self.Nr, self.Nz))    # Sounth face (constant-r, inner cylinder)


        self.calculate_geometry()
        self.identity_zone()

    def generate_radial_faces(self):
        """Graded mesh sized for wall functions at every wall (r1, r2, r3)."""

        # Inner Fluid: fine at the axis, coarse at the wall r1 (wall functions
        # want the first cell CENTER in the log layer, i.e. not sublayer-fine)
        inner = one_sided_power(self.r0, self.r1, self.Nr_inner, 1.6)

        # Steel Wall: Thin (2mm), uniform spacing is sufficient
        wall = np.linspace(self.r1, self.r2, self.Nr_wall + 1)

        # Outer Fluid: two-sided -- coarse at BOTH walls (r2 and r3), fine at
        # the mid-gap (the 3a/3b mesh study: one-sided grading put the outer
        # wall first cell at y+ ~ 8, invalid for equilibrium wall functions)
        outer = two_sided_mirrored(self.r2, self.r3, self.Nr_outer, 0.7)

        faces = np.concatenate([inner, wall[1:], outer[1:]])

        # --- grading sanity guards (cheap, fail fast on bad parameters) ---
        dr = np.diff(faces)
        assert np.all(dr > 0.0), "radial faces must be strictly increasing"
        i1 = self.Nr_inner
        i2 = i1 + self.Nr_wall
        assert dr[:i1].max() == dr[i1 - 1], "pipe wall cell should be the zone's largest"
        assert dr[i2:].max() == max(dr[i2], dr[-1]), "annulus wall cells should be the zone's largest"
        return faces


    def calculate_geometry(self):
        """Calculate FVM geometry metrics for every cylindrical control volume"""

        for i in range(self.Nr):
            r_s = self.r_faces[i]     # South radius (inner)
            r_n = self.r_faces [i+1]  # North radius (outer)

            for j in range(self.Nz):
                z_w = self.z_faces[j]   # West position
                z_e = self.z_faces[j+1] # East position
                dz  = z_e - z_w

                # FVM Geometric formulas in axisymmetric coordinates
                self.V[i, j]   = np.pi * (r_n ** 2 - r_s ** 2) * dz
                self.A_e[i, j] = np.pi * (r_n ** 2 - r_s ** 2)
                self.A_w[i, j] = np.pi * (r_n ** 2 - r_s ** 2)
                self.A_n[i, j] = 2.0 * np.pi * r_n * dz
                self.A_s[i, j] = 2.0 * np.pi * r_s * dz


    def identity_zone(self):
        """Maps each cell index to its material zone (inner fluid, wall, or outer fluid)"""
        self.zone_map = np.zeros(self.Nr, dtype = int)

        self.zone_map[0:self.Nr_inner] = 0                             # 0 = Inner Fluid
        self.zone_map[self.Nr_inner: self.Nr_inner + self.Nr_wall] = 1 # 1 = Solid Steel wall
        self.zone_map[self.Nr_inner + self.Nr_wall :] = 2              # 2 = outer Fluid


# Verification test script
if __name__ == "__main__":
    mesh = AxisymmetricMesh()
    print("Mesh Succesfully generated!")
    print(f"Total radial cells (Nr): {mesh.Nr}")
    print(f"Total axial cells (Nz): {mesh.Nz}")
    dr = np.diff(mesh.r_faces)
    print(f"Pipe wall cell (at r1):    {dr[mesh.Nr_inner - 1] * 1000:.3f} mm")
    print(f"Wall cells (uniform):      {dr[mesh.Nr_inner] * 1000:.3f} mm")
    print(f"Annulus wall cells (r2/r3): {dr[mesh.Nr_inner + mesh.Nr_wall] * 1000:.3f} / {dr[-1] * 1000:.3f} mm")
    print(f"Annulus mid-gap cell:      {dr[mesh.Nr_inner + mesh.Nr_wall: ].min() * 1000:.3f} mm")
    print(f"Total volume of domain check: {np.sum(mesh.V): .6f} m^3")
    print(f"Analytical volume expectation: {np.pi * (0.025 ** 2) * 2.0: 0.6f} m^3")
