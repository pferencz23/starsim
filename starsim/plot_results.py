"""
Reproduce sim_plot.png from results_sim.csv.

Usage:
    python run_outputs/plot_results.py run_outputs/20260324T165104Z/results_sim.csv
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import sciris as sc

# ---------------------------------------------------------------------------
# Column metadata: (label, y-axis title)
# ---------------------------------------------------------------------------
COLUMNS = {
    "randomnet_n_edges":         ("Random network: n edges",          "Number of edges"),
    "epigame_n_quarantined":     ("Epigame: n quarantined",           "Number of agents"),
    "epigame_n_has_been_infected": ("Epigame: n has been infected",   "Number of agents"),
    "epigame_quarantine_rate":   ("Epigame: quarantine rate",         "Rate"),
    "epigame_mean_points":       ("Epigame: mean points",             "Points"),
    "seir_ams_n_susceptible":    ("SEIR AMS: n susceptible",          "Number of agents"),
    "seir_ams_n_exposed":        ("SEIR AMS: n exposed",              "Number of agents"),
    "seir_ams_n_infected":       ("SEIR AMS: n infected",             "Number of agents"),
    "seir_ams_n_recovered":      ("SEIR AMS: n recovered",            "Number of agents"),
    "seir_ams_prevalence":       ("SEIR AMS: prevalence",             "Prevalence"),
    "seir_ams_new_infections":   ("SEIR AMS: new infections",         "New infections"),
    "seir_ams_cum_infections":   ("SEIR AMS: cum infections",         "Cumulative infections"),
    "n_alive":                   ("n alive",                          "Number of agents"),
    "n_female":                  ("n female",                         "Number of agents"),
    "new_deaths":                ("New deaths",                       "Deaths"),
    "new_emigrants":             ("New emigrants",                    "Emigrants"),
    "cum_deaths":                ("Cum deaths",                       "Cumulative deaths"),
}


def plot_results(csv_path: str | Path, save_path: str | Path | None = None):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, index_col=0, parse_dates=["timevec"])
    t = df["timevec"]

    cols = [c for c in COLUMNS if c in df.columns]
    fig, axs = sc.getrowscols(len(cols), make=True, figsize=(14, 10))
    if not hasattr(axs, "__iter__"):
        axs = [axs]
    else:
        axs = list(axs.flatten())

    with sc.options.with_style("default"):
        for ax, col in zip(axs, cols):
            label, ylabel = COLUMNS[col]
            ax.plot(t, df[col], alpha=0.8)
            ax.set_title(label, fontsize=9, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(axis="x", labelsize=7, rotation=30)
            ax.tick_params(axis="y", labelsize=7)

        # Hide unused axes
        for ax in axs[len(cols):]:
            ax.set_visible(False)

    sc.figlayout(fig=fig)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "run_outputs/20260324T165104Z/results_sim.csv"
    out = Path(csv).parent / "sim_plot_reproduced.png"
    plot_results(csv, save_path=out)
