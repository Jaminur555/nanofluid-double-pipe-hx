"""Validate CFD Nu_nf vs Pak-Cho and Dittus-Boelter (paper Step 0.3').

Reads the sweep CSV, prints per-phi mean-|dev|% tables, saves the parity
figure results/validation_correlations.png.
dev% = (Nu_CFD - Nu_corr) / Nu_corr * 100 (Nu_par side).
"""
import argparse
import csv

import matplotlib.pyplot as plt

from nanofluid_hx import MaterialProperties
from nanofluid_hx.correlations import dittus_boelter, pak_cho, xuan_li
from nanofluid_hx.plotting import save_figure

# Experimental correlations only - Maiga (CFD-derived) is excluded by design.
MODELS = {"Pak-Cho": pak_cho, "Xuan-Li": xuan_li,
          "Dittus-Boelter": dittus_boelter}


def load_rows(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in ("Re", "phi", "Nu_par", "Nu_ctr"):
            r[key] = float(r[key])
    return rows


def main(csv_path):
    rows = load_rows(csv_path)
    phi_list = sorted({r["phi"] for r in rows})

    print(f"\nMean |deviation| % of CFD Nu vs correlation "
          f"({len(rows)} points; negative dev = CFD below correlation)\n")
    header = f"{'phi':>6} | " + " | ".join(f"{name:>16}" for name in MODELS)
    print(header)
    print("-" * len(header))
    overall = {name: [] for name in MODELS}
    for phi in phi_list:
        props = MaterialProperties(phi)
        sub = [r for r in rows if r["phi"] == phi]
        cells = []
        for name, fn in MODELS.items():
            devs = [100.0 * (r["Nu_par"] - fn(r["Re"], props)) / fn(r["Re"], props)
                    for r in sub]
            mean_abs = sum(abs(d) for d in devs) / len(devs)
            mean_signed = sum(devs) / len(devs)
            overall[name].extend(devs)
            cells.append(f"{mean_abs:7.1f} ({mean_signed:+6.1f})")
        print(f"{phi * 100:5.1f}% | " + " | ".join(f"{c:>16}" for c in cells))
    print("-" * len(header))
    cells = [f"{sum(abs(d) for d in overall[name]) / len(overall[name]):7.1f} "
             f"({sum(overall[name]) / len(overall[name]):+6.1f})"
             for name in MODELS]
    print(f"{'ALL':>6} | " + " | ".join(f"{c:>16}" for c in cells))

    pc = sum(abs(d) for d in overall["Pak-Cho"]) / len(overall["Pak-Cho"])
    verdict = "PASS (<=20%)" if pc <= 20.0 else "FAIL (>20%)"
    print(f"\nAcceptance vs Pak-Cho: mean |dev| = {pc:.1f}% -> {verdict}")

    # Parity figure: CFD Nu vs correlation Nu on equal log-log axes
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    colors = {"Pak-Cho": "blue", "Xuan-Li": "red", "Dittus-Boelter": "green"}
    markers = {phi: marker for phi, marker in
               zip(phi_list, ["o", "s", "^", "D", "v"])}
    lo = min(min(r["Nu_par"] for r in rows),
             min(fn(r["Re"], MaterialProperties(r["phi"])) for r in rows
                 for fn in MODELS.values()))
    hi = max(max(r["Nu_par"] for r in rows),
             max(fn(r["Re"], MaterialProperties(r["phi"])) for r in rows
                 for fn in MODELS.values()))
    ax.plot([lo * 0.8, hi * 1.25], [lo * 0.8, hi * 1.25], "k--", lw=1,
            label="perfect agreement")
    for name, fn in MODELS.items():
        x = [fn(r["Re"], MaterialProperties(r["phi"])) for r in rows]
        y = [r["Nu_par"] for r in rows]
        ax.plot(x, y, "o", color=colors[name], ms=0)      # legend proxy
        for xi, yi, r in zip(x, y, rows):
            ax.plot(xi, yi, marker=markers[r["phi"]], color=colors[name],
                    ms=7, mfc="none" if name != "Pak-Cho" else colors[name],
                    mec=colors[name])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Correlation Nusselt number (-)")
    ax.set_ylabel("CFD Nusselt number Nu_nf (-)")
    ax.set_title("CFD vs experimental correlations (parallel flow, hot side)")
    ax.grid(True, which="both", alpha=0.3)
    handles = [plt.Line2D([], [], color=colors[n], marker="o", ls="",
                          label=n) for n in MODELS]
    handles += [plt.Line2D([], [], color="k", marker=m, ls="",
                           label=f"phi = {phi * 100:g}%")
                for phi, m in markers.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    save_figure(fig, "validation_correlations")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correlation validation")
    parser.add_argument("--csv", default="results/sweep_nu_effectiveness_"
                        "simplec_k_epsilon.csv")
    main(parser.parse_args().csv)
