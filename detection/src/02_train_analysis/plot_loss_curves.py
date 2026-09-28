"""Trace la loss d'entraînement vs la loss de validation par epoch, pour
diagnostiquer un éventuel sur-/sous-apprentissage.

NanoDet calcule et logue une loss de validation avec la même fonction de
perte que l'entraînement (`forward_train` est aussi appelé côté validation,
cf. `nanodet/trainer/task.py::validation_step`) -- ce n'est donc pas du
bruit ni une approximation, juste un signal qui n'était jusqu'ici pas
extrait du log texte par `parse_training_log.py`.

La loss totale (utilisée pour la rétropropagation) n'est pas logguée telle
quelle par NanoDet, seulement ses 6 composantes (loss_qfl/loss_bbox/
loss_dfl + leurs équivalents aux_*, cf. nanodet_plus_head.py::loss) --
reconstruite ici en sommant les colonnes, par ligne.

Usage :
    uv run python detection/src/02_train_analysis/plot_loss_curves.py <results_dir>
    (results_dir doit contenir train_losses.csv et val_loss.csv,
    produits par parse_training_log.py)
"""

import csv
import sys
from collections import defaultdict

import matplotlib.pyplot as plt

LOSS_COLUMNS = [
    "loss_qfl",
    "loss_bbox",
    "loss_dfl",
    "aux_loss_qfl",
    "aux_loss_bbox",
    "aux_loss_dfl",
]


def load_mean_loss_per_epoch(csv_path: str) -> dict[int, float]:
    """Loss totale moyenne par epoch (moyenne des lignes loggées à cette
    epoch, elles-mêmes déjà des moyennes de batch -- cf. task.py
    `loss_states[loss_name].mean().item()`)."""
    sums_by_epoch: dict[int, list[float]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total = sum(float(row[col]) for col in LOSS_COLUMNS)
            sums_by_epoch[int(row["epoch"])].append(total)
    return {epoch: sum(vals) / len(vals) for epoch, vals in sums_by_epoch.items()}


def main():
    results_dir = sys.argv[1]
    train_by_epoch = load_mean_loss_per_epoch(f"{results_dir}/train_losses.csv")
    # val_loss.csv : vraie moyenne sur tout le set de validation (task.py
    # validation_epoch_end) -- seule série de loss de validation parsée
    # (parse_training_log.py n'extrait plus l'échantillon par batch,
    # biaisé de façon fixe puisque le dataloader de validation n'est
    # jamais mélangé, jusqu'à ~0,10-0,12 d'écart selon la résolution,
    # cf. rapport section 4.1).
    val_by_epoch = load_mean_loss_per_epoch(f"{results_dir}/val_loss.csv")

    train_epochs = sorted(train_by_epoch)
    val_epochs = sorted(val_by_epoch)

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
    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=300)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    ax.plot(
        train_epochs,
        [train_by_epoch[e] for e in train_epochs],
        color="#2a78d6",
        linestyle="-",
        marker="o",
        markersize=4,
        linewidth=1.8,
        label="Loss entraînement",
    )
    ax.plot(
        val_epochs,
        [val_by_epoch[e] for e in val_epochs],
        color="#eb6834",
        linestyle="--",
        marker="s",
        markersize=5,
        linewidth=1.8,
        label="Loss validation",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss totale (moyenne par epoch)")
    ax.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)

    fig.tight_layout()
    output_path = f"{results_dir}/loss_curves.png"
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    print(f"{output_path} écrit")


if __name__ == "__main__":
    main()
