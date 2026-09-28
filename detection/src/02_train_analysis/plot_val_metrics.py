"""Trace les métriques de détection (mAP/AP50/AP75, puis mAP par classe)
à chaque checkpoint de validation, pour un run donné.

Usage :
    uv run python detection/src/02_train_analysis/plot_val_metrics.py <results_dir>
    (results_dir doit contenir val_metrics.csv, produit par
    parse_training_log.py)
"""

import csv
import sys

import matplotlib.pyplot as plt


def load_rows(csv_path: str) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [
            {k: (int(v) if k == "epoch" else float(v)) for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def main():
    results_dir = sys.argv[1]
    rows = load_rows(f"{results_dir}/val_metrics.csv")
    epochs = [r["epoch"] for r in rows]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": "#0b0b0b",
            "text.color": "#0b0b0b",
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
        }
    )
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(6.6, 6.2), dpi=300, sharex=True
    )
    for ax in (ax_top, ax_bottom):
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.patch.set_facecolor("#fcfcfb")

    overall_series = [
        ("mAP", "#2a78d6", "o"),
        ("AP_50", "#eb6834", "s"),
        ("AP_75", "#1baf7a", "^"),
    ]
    for col, color, marker in overall_series:
        ax_top.plot(
            epochs,
            [r[col] for r in rows],
            color=color,
            marker=marker,
            markersize=5,
            linewidth=1.8,
            label=col.replace("_", ""),
        )
    ax_top.set_ylabel("Score global")
    ax_top.legend(
        loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8.5
    )

    class_series = [
        ("person_mAP", "person", "#2a78d6", "o"),
        ("car_mAP", "car", "#eb6834", "s"),
    ]
    for col, label, color, marker in class_series:
        ax_bottom.plot(
            epochs,
            [r[col] / 100.0 for r in rows],
            color=color,
            marker=marker,
            markersize=5,
            linewidth=1.8,
            label=label,
        )
    ax_bottom.set_ylabel("mAP par classe")
    ax_bottom.set_xlabel("Epoch")
    ax_bottom.legend(
        loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8.5
    )

    fig.tight_layout()
    output_path = f"{results_dir}/val_metrics.png"
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    print(f"{output_path} écrit")


if __name__ == "__main__":
    main()
