import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


TRAIN_TEST_SPLIT = datetime(2023, 1, 1)
CRISIS_PERIODS = [
    ("2008 GFC", datetime(2007, 10, 1), datetime(2009, 6, 30), "#f4a261"),
    ("2020 COVID", datetime(2020, 2, 15), datetime(2020, 6, 30), "#e76f51"),
    ("2022 Downturn", datetime(2022, 1, 1), datetime(2022, 12, 31), "#8d99ae"),
]


def load_data(csv_path: Path) -> tuple[list[datetime], list[float]]:
    dates: list[datetime] = []
    closes: list[float] = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                dates.append(datetime.strptime(row["<DATE>"], "%Y.%m.%d"))
                closes.append(float(row["<CLOSE>"]))
            except (KeyError, TypeError, ValueError):
                continue

    return dates, closes


def plot_msft_close(dates, closes, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, closes, color="tab:blue", linewidth=1.3)

    for label, start, end, color in CRISIS_PERIODS:
        ax.axvspan(
            float(mdates.date2num(start)),
            float(mdates.date2num(end)),
            color=color,
            alpha=0.18,
        )

    ax.axvline(
        float(mdates.date2num(TRAIN_TEST_SPLIT)),
        color="black",
        linestyle="--",
        linewidth=1.2,
    )

    legend_handles: list[object] = [
        Line2D([0], [0], color="tab:blue", lw=1.8, label="MSFT Close"),
        Line2D(
            [0],
            [0],
            color="black",
            lw=1.2,
            linestyle="--",
            label="Train/Test Split (2023-01-01)",
        ),
    ]
    for label, _, _, color in CRISIS_PERIODS:
        legend_handles.append(
            Patch(facecolor=color, edgecolor="none", alpha=0.18, label=label)
        )

    ax.set_title("MSFT Daily Close Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price (USD)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(handles=legend_handles, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = repo_root / "data" / "MSFT" / "MSFT_1d.csv"
    output_path = repo_root / "figures" / "msft_1d_close.png"

    dates, closes = load_data(csv_path)
    plot_msft_close(dates, closes, output_path)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
