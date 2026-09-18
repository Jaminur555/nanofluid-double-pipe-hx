"""PEC / phi* robustness across property models and particles.

Combos: Al2O3 (Bahmani fits vs Maxwell/Brinkman vs Corcione) and CuO
(Maxwell/Brinkman). If PEC < 1 and phi* = 0 for every combo, the negative
result is robust to the property-model choice.
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_egm import PHIS, interp_phi, load
from nanofluid_hx.plotting import save_figure

RESULTS = Path("results")
COMBOS = [
    ("Al2O3 Bahmani", "sweep_nu_effectiveness_simplec_k_epsilon.csv"),
    ("Al2O3 Maxwell/Brinkman",
     "sweep_nu_effectiveness_simplec_k_epsilon_al2o3_maxwell_brinkman.csv"),
    ("Al2O3 Corcione",
     "sweep_nu_effectiveness_simplec_k_epsilon_al2o3_corcione_corcione.csv"),
    ("CuO Maxwell/Brinkman",
     "sweep_nu_effectiveness_simplec_k_epsilon_cuo_maxwell_brinkman.csv"),
    ("CuO Corcione",
     "sweep_nu_effectiveness_simplec_k_epsilon_cuo_corcione_corcione.csv"),
]
COLORS = ["blue", "green", "orange", "red", "purple"]


def pec_at_equal_re(data):
    """{(Re, phi): PEC} parallel side; baseline = phi 0 of the same combo."""
    base = data[0.0]
    out = {}
    for phi in PHIS[1:]:
        d = data[phi]
        for i, Re in enumerate(d["Re"]):
            j = np.searchsorted(base["Re"], Re)
            out[(Re, phi)] = (d["Nu_par"][i] / base["Nu_par"][j]) \
                * (base["W"][j] / d["W"][i]) ** (1 / 3)
    return out


def phi_star_at_fixed_w(data, n_w=5):
    """(w_grid, phi*(w), Nu(w*)) on the common W range (log-log interp)."""
    w_lo = max(d["W"].min() for d in data.values())
    w_hi = min(d["W"].max() for d in data.values())
    w_grid = np.geomspace(w_lo, w_hi, n_w)
    nu_w = {phi: interp_phi(data[phi], "Nu_par", w_grid) for phi in PHIS}
    best = [max(PHIS, key=lambda p: nu_w[p][k]) for k in range(n_w)]
    nu_best = [max(nu_w[p][k] for p in PHIS) for k in range(n_w)]
    return w_grid, best, nu_best


def main():
    data = {name: load(RESULTS / fname) for name, fname in COMBOS}
    pec = {name: pec_at_equal_re(d) for name, d in data.items()}

    print("PEC (equal Re) mean over Re, parallel side "
          "(range over Re in brackets):")
    header = f"{'combo':<22}" + "".join(f"phi={p:<18}" for p in PHIS[1:])
    print(header)
    csv_rows = []
    for name, _ in COMBOS:
        cells = []
        for phi in PHIS[1:]:
            vals = [v for (re_, p), v in pec[name].items() if p == phi]
            cells.append(f"{np.mean(vals):.2f} "
                         f"[{min(vals):.2f},{max(vals):.2f}]")
            for (re_, p), v in pec[name].items():
                if p == phi:
                    csv_rows.append([name, int(re_), p, round(v, 4)])
        print(f"{name:<22}" + "".join(f"{c:<20}" for c in cells))
    worst = max(v for pv in pec.values() for v in pv.values())
    print(f"\nWorst-case PEC over all combos/Re/phi: {worst:.2f} "
          f"({'< 1 everywhere' if worst < 1 else '>= 1 somewhere'})")

    print("\nphi* at fixed pumping power (5 W points per combo):")
    for name, _ in COMBOS:
        w_grid, best, nu_best = phi_star_at_fixed_w(data[name])
        uniq = sorted(set(best))
        print(f"  {name:<22} phi* values: {uniq}  "
              f"(Nu range {min(nu_best):.1f}-{max(nu_best):.1f})")

    out_csv = RESULTS / "property_uncertainty.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["combo", "Re", "phi", "PEC"])
        w.writerows(csv_rows)
    print(f"CSV saved: {out_csv}")

    # Figure: PEC vs phi (mean over Re, min-max band) and PEC vs Re at phi=5%
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    phis = PHIS[1:]
    for c, (name, _) in enumerate(COMBOS):
        mean = [np.mean([v for (_, p), v in pec[name].items() if p == phi])
                for phi in phis]
        lo = [min(v for (_, p), v in pec[name].items() if p == phi)
              for phi in phis]
        hi = [max(v for (_, p), v in pec[name].items() if p == phi)
              for phi in phis]
        x = np.array(phis) * 100
        ax[0].plot(x, mean, "o-", color=COLORS[c], label=name)
        ax[0].fill_between(x, lo, hi, color=COLORS[c], alpha=0.15)
        res = data[name][0.05]["Re"]
        pec5 = [pec[name][(r, 0.05)] for r in res]
        ax[1].plot(res, pec5, "s-", color=COLORS[c], label=name)
    for a, title in ((ax[0], "PEC vs phi (mean over Re, band = Re range)"),
                     (ax[1], "PEC vs Re at phi = 5%")):
        a.axhline(1.0, color="k", ls="--", lw=1)
        a.set_title(title)
        a.set_xlabel("phi (%)" if a is ax[0] else "Reynolds number")
        a.grid(True, alpha=0.3)
        a.legend(fontsize=8)
    ax[0].set_ylabel("PEC (-)")
    fig.suptitle("Property-model and particle robustness of the PEC result",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, "property_uncertainty")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Property-model comparison")
    parser.add_argument("--results", default=str(RESULTS))
    args = parser.parse_args()
    RESULTS = Path(args.results)
    main()
