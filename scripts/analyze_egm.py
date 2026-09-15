"""EGM / PEC analysis from the sweep CSV (Phase A4).

PEC (same Re) = (Nu/Nu_w)*(W_w/W)^(1/3), W = dP*U*A pipe side.
Fixed-W: phi* = argmax Nu (log-log interp). Min-S at fixed W is degenerate
(lower duty => lower S); S(W) curves are descriptive only.
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PHIS = [0.0, 0.025, 0.05, 0.075, 0.1]
COLORS = ['blue', 'green', 'orange', 'red', 'purple']
R_PIPE = 0.013                                  # default mesh r1 [m]
A_PIPE = np.pi * R_PIPE**2


def load(path):
    """-> {phi: {'Re','W','Nu_par','Nu_ctr','S_par','S_ctr'}}, sorted by Re."""
    rows = list(csv.DictReader(open(path)))
    data = {phi: {'Re': [], 'W': [], 'Nu_par': [], 'Nu_ctr': [],
                  'S_par': [], 'S_ctr': []} for phi in PHIS}
    for r in rows:
        phi = float(r['phi'])
        d = data[phi]
        d['Re'].append(float(r['Re']))
        d['W'].append(float(r['dP_pipe']) * float(r['U_pipe']) * A_PIPE)
        for key in ('Nu_par', 'Nu_ctr', 'S_par', 'S_ctr'):
            d[key].append(float(r[key]))
    for phi in PHIS:
        order = np.argsort(data[phi]['Re'])
        data[phi] = {k: np.asarray(v)[order] for k, v in data[phi].items()}
    return data


def interp_phi(d, key, w_target):
    """key(W) for one phi, interpolated in log-log space."""
    return np.exp(np.interp(np.log(w_target), np.log(d['W']),
                            np.log(d[key])))


def main(csv_path, out_png, out_csv):
    data = load(csv_path)

    # --- same-Re PEC vs phi=0 baseline -------------------------------
    print("PEC (Nu ratio weighted by pumping power, parallel side):")
    print("  Re      " + "".join(f"phi={p:<6}" for p in PHIS[1:]))
    pec = {Re: [] for Re in data[0.0]['Re']}
    for Re in data[0.0]['Re']:
        base = data[0.0]
        i0 = np.searchsorted(base['Re'], Re)
        w0, n0 = base['W'][i0], base['Nu_par'][i0]
        for phi in PHIS[1:]:
            d = data[phi]
            i = np.searchsorted(d['Re'], Re)
            pec[Re].append((d['Nu_par'][i] / n0) * (w0 / d['W'][i])**(1 / 3))
        print(f"  {Re:<7g}" + "".join(f"{v:<9.3f}" for v in pec[Re]))

    # --- fixed pumping power -----------------------------------------
    w_lo = max(d['W'].min() for d in data.values())
    w_hi = min(d['W'].max() for d in data.values())
    w_grid = np.geomspace(w_lo, w_hi, 5)
    nu_w = {phi: interp_phi(data[phi], 'Nu_par', w_grid) for phi in PHIS}

    print(f"\nFixed pumping power W in [{w_lo:.2f}, {w_hi:.2f}] W "
          f"(parallel side):")
    print("  W[W]    phi*_Nu  Nu")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(['W', 'phi_opt_Nu', 'Nu'])
        for k, wt in enumerate(w_grid):
            ph_n = max(PHIS, key=lambda p: nu_w[p][k])
            print(f"  {wt:<8.2f}{ph_n:<9g}{nu_w[ph_n][k]:.2f}")
            w.writerow([wt, ph_n, nu_w[ph_n][k]])
    print(f"CSV saved: {out_csv}")

    # --- figure -------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    for i, phi in enumerate(PHIS):
        d = data[phi]
        ax[0, 0].loglog(d['Re'], d['S_par'], 'o-', color=COLORS[i],
                        label=f'phi: {phi * 100:.1f}%')
        ax[1, 0].loglog(d['W'], d['Nu_par'], 'o-', color=COLORS[i],
                        label=f'phi: {phi * 100:.1f}%')
        ax[1, 1].loglog(d['W'], d['S_par'], 'o-', color=COLORS[i],
                        label=f'phi: {phi * 100:.1f}%')

    ax[0, 0].set_title("Total Entropy Generation vs Re (Parallel)")
    ax[0, 0].set_xlabel("Re"); ax[0, 0].set_ylabel("S_gen [W/K]")
    ax[0, 0].grid(True, which='both', alpha=0.3); ax[0, 0].legend()

    for i, Re in enumerate(data[0.0]['Re']):
        ax[0, 1].plot([p * 100 for p in PHIS[1:]], pec[Re], 'o-',
                      color=plt.cm.viridis(i / 5), label=f'Re = {Re / 1000:g}k')
    ax[0, 1].axhline(1.0, color='k', ls='--', lw=1)
    ax[0, 1].set_title("PEC vs Volume Fraction (equal-Re criterion)")
    ax[0, 1].set_xlabel("phi [%]"); ax[0, 1].set_ylabel("PEC")
    ax[0, 1].grid(True, alpha=0.3); ax[0, 1].legend(fontsize=8)

    ax[1, 0].set_title("Nu vs Pumping Power (Parallel)")
    ax[1, 0].set_xlabel("W = dP*U*A [W]"); ax[1, 0].set_ylabel("Nu")
    ax[1, 0].grid(True, which='both', alpha=0.3); ax[1, 0].legend()

    ax[1, 1].set_title("S_gen vs Pumping Power (Parallel)")
    ax[1, 1].set_xlabel("W = dP*U*A [W]"); ax[1, 1].set_ylabel("S_gen [W/K]")
    ax[1, 1].grid(True, which='both', alpha=0.3); ax[1, 1].legend()

    fig.suptitle("Al$_2$O$_3$-Water Nanofluid HX — EGM / PEC Analysis",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=150)
    print(f"Figure saved: {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EGM/PEC analysis of sweep CSV")
    parser.add_argument("--csv", default="results/"
                        "sweep_nu_effectiveness_simplec_k_epsilon.csv")
    args = parser.parse_args()
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    main(args.csv, out_dir / "egm_analysis.png", out_dir / "egm_optimum.csv")
