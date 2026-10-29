"""Simulate how each figure looks with colour-vision deficiencies and in grayscale.

For each `reports/figures/*.png` (or those matching a glob pattern), write a review
panel to `reports/figures/cvd/` with the original, deuteranopia, protanopia,
tritanopia and grayscale (print) versions.

Run from the project root: `.venv/bin/python -m src.cvd_check [pattern]`, e.g.
`.venv/bin/python -m src.cvd_check "01_*"`.
"""

import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with warnings.catch_warnings():  # colorspacious has an old docstring escape warning
    warnings.simplefilter("ignore", SyntaxWarning)
    from colorspacious import cspace_convert

from src.config import FIGURES_DIR

CVD_DIR = FIGURES_DIR / "cvd"
CVD_TYPES = {
    "Deuteranopia (red-green, most common)": "deuteranomaly",
    "Protanopia (red-green)": "protanomaly",
    "Tritanopia (blue-yellow, rare)": "tritanomaly",
}


def simulate_cvd(rgb: np.ndarray, cvd_type: str) -> np.ndarray:
    space = {"name": "sRGB1+CVD", "cvd_type": cvd_type, "severity": 100}
    return np.clip(cspace_convert(rgb, space, "sRGB1"), 0, 1)


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Relative luminance (Rec. 709 weights on linear RGB), shown as gray."""
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
    gray = np.where(luminance <= 0.0031308, 12.92 * luminance, 1.055 * luminance ** (1 / 2.4) - 0.055)
    return np.repeat(gray[..., None], 3, axis=-1)


def review_panel(path) -> None:
    rgb = plt.imread(path)[..., :3]
    versions = {"Original": rgb}
    versions.update({label: simulate_cvd(rgb, kind) for label, kind in CVD_TYPES.items()})
    versions["Grayscale (print)"] = to_grayscale(rgb)

    height, width = rgb.shape[:2]
    fig, axes = plt.subplots(3, 2, figsize=(14, 14 * 1.5 * height / width + 1))
    for ax, (label, image) in zip(axes.flat, versions.items()):
        ax.imshow(image)
        ax.set_title(label, fontsize=11, loc="left")
        ax.axis("off")
    axes.flat[-1].axis("off")
    fig.suptitle(path.name, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    CVD_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CVD_DIR / path.name, dpi=80)
    plt.close(fig)


def main() -> None:
    pattern = sys.argv[1] if len(sys.argv) > 1 else "*"
    paths = sorted(FIGURES_DIR.glob(f"{pattern}.png"))
    for path in paths:
        review_panel(path)
    print(f"{len(paths)} review panels written to {CVD_DIR}")


if __name__ == "__main__":
    main()
