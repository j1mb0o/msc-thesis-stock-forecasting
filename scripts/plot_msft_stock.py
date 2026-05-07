from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


TRAIN_TEST_SPLIT = datetime(2023, 1, 1)
CRISIS_PERIODS = [
    ("2008 GFC", datetime(2007, 10, 1), datetime(2009, 6, 30), "#f4a261"),
    ("2020 COVID", datetime(2020, 2, 15), datetime(2020, 6, 30), "#e76f51"),
    ("2022 Downturn", datetime(2022, 1, 1), datetime(2022, 12, 31), "#8d99ae"),
]


def load_data(csv_path: Path) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(csv_path, sep="\t")

    if "<TIME>" in df.columns:
        dates = pd.to_datetime(
            df["<DATE>"] + " " + df["<TIME>"],
            format="%Y.%m.%d %H:%M:%S",
        )
    else:
        dates = pd.to_datetime(df["<DATE>"], format="%Y.%m.%d")

    closes = pd.to_numeric(df["<CLOSE>"], errors="coerce")
    valid_rows = dates.notna() & closes.notna()

    return dates[valid_rows], closes[valid_rows]


def calculate_pct_change(closes: pd.Series) -> pd.Series:
    return closes.pct_change().dropna()


def format_pct_change_stats(pct_changes: pd.Series) -> str:
    return (
        f"avg: {pct_changes.mean():.6g}\n"
        f"max: {pct_changes.max():.6g}\n"
        f"min: {pct_changes.min():.6g}"
    )


def add_stats_box(ax, pct_changes: pd.Series) -> None:
    ax.text(
        0.985,
        0.965,
        format_pct_change_stats(pct_changes),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.9,
        },
    )


def add_time_markers(ax) -> None:
    for _, start, end, color in CRISIS_PERIODS:
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
    ax.grid(True, linestyle="--", alpha=0.4)


def get_time_marker_handles() -> list[object]:
    legend_handles: list[object] = [
        Line2D(
            [0],
            [0],
            color="black",
            lw=1.2,
            linestyle="--",
            label="Train/Test Split (2023-01-01)",
        )
    ]

    for label, _, _, color in CRISIS_PERIODS:
        legend_handles.append(
            Patch(facecolor=color, edgecolor="none", alpha=0.18, label=label)
        )

    return legend_handles


def get_legend_handles(pct_change_label: str, color: str = "tab:green"):
    return [
        Line2D([0], [0], color=color, lw=1.2, label=pct_change_label),
        *get_time_marker_handles(),
    ]


def save_pdf(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, format="pdf")
    plt.close(fig)


def plot_msft_pct_change(
    dates, pct_changes, output_path: Path, period_label: str
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, pct_changes, color="tab:green", linewidth=0.9)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)

    add_time_markers(ax)
    add_stats_box(ax, pct_changes)

    ax.set_title(f"MSFT {period_label} Fractional Change")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{period_label} Fractional Change")
    ax.legend(
        handles=get_legend_handles(f"{period_label} Fractional Change"),
        loc="upper left",
    )
    fig.autofmt_xdate()
    fig.tight_layout()

    save_pdf(fig, output_path)


def plot_msft_pct_changes_together(
    daily_dates,
    daily_pct_changes,
    hourly_dates,
    hourly_pct_changes,
    output_path: Path,
) -> None:
    fig, (daily_ax, hourly_ax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 7),
        sharex=True,
    )

    daily_ax.plot(daily_dates, daily_pct_changes, color="tab:green", linewidth=0.9)
    hourly_ax.plot(
        hourly_dates,
        hourly_pct_changes,
        color="tab:purple",
        linewidth=0.6,
    )

    for ax in (daily_ax, hourly_ax):
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        add_time_markers(ax)

    add_stats_box(daily_ax, daily_pct_changes)
    add_stats_box(hourly_ax, hourly_pct_changes)

    daily_ax.set_title("MSFT Fractional Change")
    daily_ax.set_ylabel("Daily Fractional Change")
    daily_ax.legend(
        handles=get_legend_handles("Daily Fractional Change"),
        loc="upper left",
    )

    hourly_ax.set_xlabel("Date")
    hourly_ax.set_ylabel("Hourly Fractional Change")
    hourly_ax.legend(
        handles=get_legend_handles("Hourly Fractional Change", color="tab:purple"),
        loc="upper left",
    )

    fig.autofmt_xdate()
    fig.tight_layout()

    save_pdf(fig, output_path)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    daily_csv_path = repo_root / "data" / "MSFT" / "MSFT_1d.csv"
    hourly_csv_path = repo_root / "data" / "MSFT" / "MSFT_1h.csv"
    figures_dir = repo_root / "figures"

    daily_dates, daily_closes = load_data(daily_csv_path)
    hourly_dates, hourly_closes = load_data(hourly_csv_path)
    daily_pct_changes = calculate_pct_change(daily_closes)
    hourly_pct_changes = calculate_pct_change(hourly_closes)

    output_paths = [
        figures_dir / "msft_1d_pct_change.pdf",
        figures_dir / "msft_1h_pct_change.pdf",
        figures_dir / "msft_1d_1h_pct_change.pdf",
    ]

    plot_msft_pct_change(
        daily_dates.loc[daily_pct_changes.index],
        daily_pct_changes,
        output_paths[0],
        "Daily",
    )
    plot_msft_pct_change(
        hourly_dates.loc[hourly_pct_changes.index],
        hourly_pct_changes,
        output_paths[1],
        "Hourly",
    )
    plot_msft_pct_changes_together(
        daily_dates.loc[daily_pct_changes.index],
        daily_pct_changes,
        hourly_dates.loc[hourly_pct_changes.index],
        hourly_pct_changes,
        output_paths[2],
    )

    for output_path in output_paths:
        print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
