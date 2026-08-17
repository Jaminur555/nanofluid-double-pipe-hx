import numpy as np


class AxisymmetricMesh:
    def __init__(self, Nr_inner = 15, Nr_wall = 5, Nr_outer = 15, Nz = 150, L = 2):
        """
        Generates a non-uniform 2D axissymmetric cylinder mesh.

        Nr_inner: Number of radial grid cells in inner fluid (0-13) mm
        Nr_wall : Number of radial grid cells in steel wall (13-15) mm
        Nr_outer: Number of radial grid cells in outer fluid (15-25) mm
        Nz      : Number of axial grid cells along (L = 2 meters)
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
        self.z_faces = np.linspace(0, self.L, self.Nz + 1)

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
        """Generate a graded mesh clustered around key interfaces (r1 and r2)"""

        # Inner Fluid: Clustered towards r1 (13mm)
        inner = self.stretch_grid(self.r0, self.r1, self.Nr_inner, stretch_type = "end")

        # Steel Wall: Thin (2mm), uniform spacing is sufficient
        wall = np.linspace(self.r1, self.r2, self.Nr_wall + 1)

         # Outer Fluid: Clustered Towards r2 (15 mm)

        outer = self.stretch_grid(self.r2, self.r3, self.Nr_outer, stretch_type = "start")

        return np.concatenate([inner, wall[1:], outer[1:]])


    def stretch_grid(self, start, end, num_cells, stretch_type = "end"):
        """Helper to stretch grid spacing to resolve turbulent boundary layers"""

        # Simple Power law stretching function

        s = np.linspace(0, 1, num_cells + 1)
        if stretch_type == "end":
            s_stretched = s**1.5                 # Densified at the end (near interface)
        elif stretch_type == "start":
            s_stretched = 1.0 - (1.0 - s) ** 1.5 # Densified at the start (near interface)
        else:
            s_stretched = s
        return start + (end - start) * s_stretched


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
    mesh = AxisymmetricMesh(Nr_inner= 15, Nr_wall=5, Nr_outer=15, Nz=150)
    print("Mesh Succesfully generated!")
    print(f"Total radial cells (Nr): {mesh.Nr}")
    print(f"Total axial cells (Nz): {mesh.Nz}")
    print(f"Smallest dr in inner fluid: {(mesh.r_faces[15] - mesh.r_faces[14]) * 1000: .3f} mm")
    print(f"Wall cell spacing: {(mesh.r_faces[16] - mesh.r_faces[15]) * 1000:.3f} mm")
    print(f"Smallest dr in outer fluid: {(mesh.r_faces[21] - mesh.r_faces[20])*1000: 0.3f} mm")
    print(f"Total volume of domain check: {np.sum(mesh.V): .6f} m^3")
    print(f"Analytical volume expectation: {np.pi * (0.025 ** 2) * 2.0: 0.6f} m^3")


