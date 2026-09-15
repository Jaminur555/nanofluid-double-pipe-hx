"""Grid independence study (paper Step 0.2): Nu_nf and effectiveness vs mesh.

Nr_inner {12,15,18,21} (Nr_outer scaled 16/15, Nr_wall=5), Nz {40,60,90},
plus the production default 15/5/16/150. Points (Re,phi): (1e4,0),
(3e4,0.05), (1e5,0.1), parallel side; deviations vs finest grid (21, 90).
Outputs: results/grid_independence.{csv,png} + console tables.
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_parameter_sweep import make_provider, analyze_case          # noqa: E402

from nanofluid_hx import AxisymmetricMesh                            # noqa: E402
from nanofluid_hx.plotting import save_figure                        # noqa: E402

NR_INNER = [12, 15, 18, 21]
NZ_LIST = [40, 60, 90]
POINTS = [(1.0e4, 0.0), (3.0e4, 0.05), (1.0e5, 0.1)]


def main(max_iter):
    grids = [(nri, round(nri * 16 / 15), nz)          # (Nr_inner, Nr_outer, Nz)
             for nri in NR_INNER for nz in NZ_LIST]
    grids.append((15, 16, 150))                        # production default

    rows = []
    for Re, phi in POINTS:
        for nri, nro, nz in grids:
            mesh = AxisymmetricMesh(Nr_inner=nri, Nr_wall=5, Nr_outer=nro, Nz=nz)
            fd = make_provider("simplec_k_epsilon", mesh, phi, Re,
                               max_outer_iter=max_iter)
            Nu, eff, *_ = analyze_case(fd, parallel_flow=True)
            rows.append({"Re": Re, "phi": phi, "Nr_inner": nri, "Nr_outer": nro,
                         "Nz": nz, "Nu": Nu, "eff": eff,
                         "it_inner": fd.iterations["inner"],
                         "it_outer": fd.iterations["outer"]})
            print(f"done: Re={Re:.0e} phi={phi} Nr={nri}/{nro} Nz={nz} "
                  f"Nu={Nu:.2f} eff={eff:.4f}", flush=True)

    csv_path = Path("results/grid_independence.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {csv_path}")

    # Deviation tables vs finest grid
    for Re, phi in POINTS:
        sub = [r for r in rows if r["Re"] == Re and r["phi"] == phi]
        ref = next(r for r in sub if r["Nr_inner"] == 21 and r["Nz"] == 90)
        print(f"\n=== Re={Re:.0e}, phi={phi}  (ref finest Nr=21/22, Nz=90: "
              f"Nu={ref['Nu']:.2f}, eff={ref['eff']:.4f}) ===")
        print(f"{'Nr':>7} {'Nz':>4} | {'Nu':>9} {'dNu%':>7} | {'eff':>8} {'deff%':>7}")
        for r in sorted(sub, key=lambda r: (r["Nr_inner"], r["Nz"])):
            dnu = 100.0 * (r["Nu"] - ref["Nu"]) / ref["Nu"]
            deff = 100.0 * (r["eff"] - ref["eff"]) / ref["eff"]
            tag = "  <- production" if (r["Nr_inner"], r["Nz"]) == (15, 150) else ""
            print(f"{r['Nr_inner']:>3}/{r['Nr_outer']:<3} {r['Nz']:>4} | "
                  f"{r['Nu']:>9.2f} {dnu:>+7.2f} | {r['eff']:>8.4f} "
                  f"{deff:>+7.2f}{tag}")

    # Figure: Nu vs Nz per Nr_inner, one panel per operating point
    fig, axes = plt.subplots(1, len(POINTS), figsize=(5 * len(POINTS), 4.5),
                             sharex=True)
    colors = ["blue", "green", "orange", "red"]
    for ax, (Re, phi) in zip(axes, POINTS):
        sub = [r for r in rows if r["Re"] == Re and r["phi"] == phi]
        for idx, nri in enumerate(NR_INNER):
            pts = sorted((r for r in sub if r["Nr_inner"] == nri and r["Nz"] <= 90),
                         key=lambda r: r["Nz"])
            ax.plot([p["Nz"] for p in pts], [p["Nu"] for p in pts], "o-",
                    color=colors[idx], label=f"Nr = {nri}/{round(nri*16/15)}")
        prod = next(r for r in sub if r["Nr_inner"] == 15 and r["Nz"] == 150)
        ax.plot(prod["Nz"], prod["Nu"], "k*", ms=14, label="production (150)")
        ax.set_title(f"Re = {Re:.0e}, $\\phi$ = {phi*100:g}%")
        ax.set_xlabel("Axial cells Nz (-)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Average Nusselt number Nu_nf (-)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Grid independence of Nu_nf", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, "grid_independence")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid independence study")
    parser.add_argument("--max-iter", type=int, default=500)
    main(parser.parse_args().max_iter)
