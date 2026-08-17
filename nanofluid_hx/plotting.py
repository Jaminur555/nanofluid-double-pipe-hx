"""Plotting helpers: temperature contours and figure saving."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
import seaborn as sns


def plot_temperature_field(mesh, T_field, parallel_flow=True):
    """
    Plots the 2D axisymmetruc temperature distribution.
    """

    sns.set_theme(style = 'white', context='paper', font_scale=1.15)

    # Creae 2D coordinate grids for plotting
    Z, R = np.meshgrid(mesh.z_center, mesh.r_center)

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi = 120)

    levels = np.linspace(285.0, 350.0, 101)
    contour = ax.contourf(Z, R * 1000, T_field, levels = levels, cmap = 'jet', extend='both')

    cbar    = fig.colorbar(contour, ax = ax, pad = 0.03, aspect = 30)
    cbar.set_label('Temperature (K)', fontsize = 12, rotation = 270, labelpad = 15)
    cbar.ax.tick_params(labelsize = 10)

    # Draw physical domain lines (Interfaces at r = 13mm, and r = 15mm)
    innter_interface = ax.axhline(13, color = 'black', linewidth = 1.5, linestyle = '-', alpha = 0.5, label = "Inner Wall Interfaces")
    outer_interface  = ax.axhline(15, color = 'black', linewidth = 1.5, linestyle = '--', alpha = 0.5, label = "Outer Wall Interfaces")


    flow_type = "Parallel Flow" if parallel_flow else "Counter Flow"

    ax.set_title(f"2D Axissymmtric Temperature Field - {flow_type}", fontsize = 16, fontweight = 'bold', pad = 12)
    ax.set_xlabel("Axial Length $z$ (m)", fontsize = 12)
    ax.set_ylabel("Radial Radius $r$ (mm)", fontsize = 12)

    ax.tick_params(axis = "both", which = "major", labelsize = 10)

    ax.xaxis.set_major_locator(MaxNLocator(nbins = 10))
    ax.yaxis.set_major_locator(MultipleLocator(2))

    ax.grid(True, linestyle = ":", linewidth = 0.7, alpha = 0.5)
    ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.15), ncol = 2, frameon = True, fancybox = True, shadow = False, fontsize = 10)

    fig.tight_layout(rect = (0, 0.08, 1, 1))

    return fig, ax


def save_figure(fig, name, out_dir="results", dpi=200):
    """Save a figure into results/ (folder created if missing). Returns the path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Figure saved: {path}")
    return path