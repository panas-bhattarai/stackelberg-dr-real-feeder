"""House plotting style for the repo — matplotlib, consistent with the
previous prototypes (feeder-reconfiguration-invariance etc.)."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = ["#4053d3", "#00a348", "#dd2c2c", "#8c2db0", "#b8850a", "#00b3ad"]


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "figure.figsize": (8.5, 4.8),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "legend.frameon": False,
    })


def savefig(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, bbox_inches="tight")
