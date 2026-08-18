from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "DL_DataFrame.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "discharge_signals"

SIGNALS = [
    "density",
    "plasma_current",
    "toroidal_B_field",
    "density_limit_phase",
]


def plot_discharge(shot: pd.DataFrame, shot_id: int | str) -> None:
    """Create and save signal plots for one discharge."""
    shot = shot.sort_values("time")

    fig, axes = plt.subplots(
        nrows=len(SIGNALS),
        ncols=1,
        figsize=(10, 8),
        sharex=True,
    )

    for ax, signal in zip(axes, SIGNALS):
        ax.plot(shot["time"], shot[signal])
        ax.set_ylabel(signal)

    axes[-1].set_xlabel("time")
    fig.suptitle(f"Discharge {shot_id}")
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"discharge_{shot_id}_signals.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def main() -> None:
    """Generate plots for the first three discharges."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    shot_ids = df["discharge_ID"].drop_duplicates().head(3)

    for shot_id in shot_ids:
        shot = df[df["discharge_ID"] == shot_id]
        plot_discharge(shot, shot_id)


if __name__ == "__main__":
    main()
