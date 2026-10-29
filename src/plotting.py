import matplotlib.pyplot as plt
from cycler import cycler

from src.config import FIGURES_DIR

# Okabe & Ito colour-universal-design palette, as recommended in Wong (2011),
# "Points of view: Color blindness", Nature Methods 8:441. The order keeps the first
# three colours distinguishable in every pair under colour-vision deficiency and at
# >= 3:1 contrast on white. Orange and sky blue are below 3:1, so charts that use them
# need direct labels. Yellow (#F0E442) is left out: it is too light on white.
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
SKY_BLUE = "#56B4E9"
CATEGORICAL = [BLUE, VERMILLION, GREEN, ORANGE, PURPLE, SKY_BLUE]

# Perceptually uniform, colour-blind safe and readable in grayscale
SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "RdBu_r"  # only for data with a meaningful centre, e.g. z-scores around 0

# Fixed colour per customer segment (notebook 02), so a segment looks the same in every
# chart. All four pairs pass the colour-blind checks. ORANGE is below 3:1 contrast on
# white, so segment charts carry direct labels.
SEGMENT_ORDER = ["Champions", "Promising", "At risk", "Lost"]
SEGMENT_COLORS = {"Champions": BLUE, "Promising": GREEN, "At risk": VERMILLION, "Lost": ORANGE}
NEUTRAL = "#666666"  # single-series charts that are not about a segment

INK = "#000000"
INK_SECONDARY = "#4d4d4d"
GRID = "#e5e5e5"
SURFACE = "#ffffff"


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (9, 4.5),
            "figure.dpi": 100,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.grid.axis": "y",
            "axes.prop_cycle": cycler(color=CATEGORICAL),
            "image.cmap": SEQUENTIAL_CMAP,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "font.size": 10,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,  # print resolution
        }
    )


def save_fig(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", facecolor=SURFACE)
