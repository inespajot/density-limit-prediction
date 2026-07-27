import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "DL_DataFrame.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

shot_ids = df["discharge_ID"].drop_duplicates().head(3)
signals = [
    "density",
    "plasma_current",
    "toroidal_B_field",
    "density_limit_phase"
]
for shot_id in shot_ids:
    shot = df[df["discharge_ID"] == shot_id].sort_values("time")

    fig, axes = plt.subplots(
        nrows=len(signals),
        ncols=1,
        figsize=(10, 8),
        sharex=True
    )

    for ax, signal in zip(axes, signals):
        ax.plot(shot["time"], shot[signal])
        ax.set_ylabel(signal)

    axes[-1].set_xlabel("time")
    fig.suptitle(f"Discharge {shot_id}", y=1.02)
    plt.tight_layout()

    fig.savefig(
        FIGURES_DIR / f"discharge_{shot_id}_signals.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)