"""B2.7 — Archive stable du monitoring d'un entraînement NanoDet.

`workspace/` est gitignoré (regénéré à chaque run) : les courbes de perte
et métriques val ne survivraient pas à un futur réentraînement si elles
restaient seulement dans les `tfevents` ou `logs.txt` de ce dossier. Ce
script extrait les deux séries utiles du log texte du run (pertes par
itération, métriques val par epoch, y compris le détail par classe) vers
des CSV compacts, versionnés dans `detection/results/`, pour tracer des
graphes plus tard sans dépendre de tensorboard ni du workspace d'origine.

Usage :
    uv run python detection/src/02_train_analysis/parse_training_log.py <log_path...> <output_dir>

Accepte plusieurs `log_path` (dans l'ordre chronologique) : un
entraînement interrompu et repris (`resume: true`) crée un nouveau
dossier `logs-<horodatage>/logs.txt` à chaque relance dans
`workspace/.../`, chacun ne couvrant qu'un segment d'epochs -- il faut
tous les concaténer pour reconstruire l'historique complet."""

import csv
import re
import sys

EPOCH_RE = re.compile(r"Epoch(\d+)/\d+")


def make_step_re(phase: str) -> re.Pattern:
    """`phase` = "Train" ou "Val" -- NanoDet logue les deux avec le même format
    (task.py `training_step`/`validation_step`), seul le préfixe change. La loss
    de validation est calculée avec la même fonction de perte que l'entraînement
    (`forward_train` est aussi appelé côté validation) -- exploitable directement
    pour comparer les deux courbes (over/underfitting), pas seulement le mAP."""
    return re.compile(
        rf"{phase}\|Epoch(\d+)/\d+\|Iter(\d+)\((\d+)/(\d+)\)\|.*?"
        r"lr:([\d.eE+-]+)\|.*?"
        r"loss_qfl:([\d.eE+-]+)\|.*?"
        r"loss_bbox:([\d.eE+-]+)\|.*?"
        r"loss_dfl:([\d.eE+-]+)\|.*?"
        r"aux_loss_qfl:([\d.eE+-]+)\|.*?"
        r"aux_loss_bbox:([\d.eE+-]+)\|.*?"
        r"aux_loss_dfl:([\d.eE+-]+)\|"
    )


TRAIN_RE = make_step_re("Train")
# Vraie moyenne sur l'ensemble du set de validation (task.py
# validation_epoch_end, un seul message par epoch de validation) -- pas
# un échantillonnage par batch sur cfg.log.interval (comme pour Train
# ci-dessus) : NanoDet logue aussi ce type de ligne côté validation,
# mais toujours sur les mêmes images (dataloader de validation jamais
# mélangé), ce qui produit un biais fixe plutôt qu'un simple
# échantillonnage bruité (jusqu'à ~0,10-0,12 d'écart selon la résolution,
# cf. rapport section 4.1) -- non extrait ici, seule la vraie moyenne par
# epoch a un intérêt diagnostique.
VAL_LOSS_EPOCH_RE = re.compile(r"Val_loss_epoch\|Epoch(\d+)/\d+\|")
VAL_LOSS_EPOCH_FIELD_RE = re.compile(r"(\w+):([\d.eE+-]+)\|")
VAL_METRICS_RE = re.compile(r"Val_metrics: \{(.+)\}")
VAL_METRICS_FIELD_RE = re.compile(r"'(\w+)': np\.float64\(([\d.eE+-]+)\)")
PER_CLASS_RE = re.compile(
    r"\|\s*person\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*car\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
)


def _loss_row(match: re.Match) -> dict:
    epoch, global_iter, iter_in_epoch, iters_per_epoch, lr, qfl, bbox, dfl, aux_qfl, aux_bbox, aux_dfl = match.groups()
    return {
        "epoch": int(epoch),
        "global_iter": int(global_iter),
        "iter_in_epoch": int(iter_in_epoch),
        "iters_per_epoch": int(iters_per_epoch),
        "lr": float(lr),
        "loss_qfl": float(qfl),
        "loss_bbox": float(bbox),
        "loss_dfl": float(dfl),
        "aux_loss_qfl": float(aux_qfl),
        "aux_loss_bbox": float(aux_bbox),
        "aux_loss_dfl": float(aux_dfl),
    }


def parse(log_path: str) -> tuple[list[dict], list[dict], list[dict]]:
    train_rows = []
    val_loss_rows = []
    val_metric_rows = []
    current_epoch = 0
    pending_per_class = None

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))

            train_match = TRAIN_RE.search(line)
            if train_match:
                train_rows.append(_loss_row(train_match))
                continue

            val_loss_epoch_match = VAL_LOSS_EPOCH_RE.search(line)
            if val_loss_epoch_match:
                row = {"epoch": int(val_loss_epoch_match.group(1))}
                row.update(
                    {k: float(v) for k, v in VAL_LOSS_EPOCH_FIELD_RE.findall(line)}
                )
                val_loss_rows.append(row)
                continue

            per_class_match = PER_CLASS_RE.search(line)
            if per_class_match:
                p_ap50, p_map, c_ap50, c_map = per_class_match.groups()
                pending_per_class = {
                    "person_AP50": float(p_ap50),
                    "person_mAP": float(p_map),
                    "car_AP50": float(c_ap50),
                    "car_mAP": float(c_map),
                }
                continue

            val_match = VAL_METRICS_RE.search(line)
            if val_match:
                fields = dict(VAL_METRICS_FIELD_RE.findall(val_match.group(1)))
                row = {"epoch": current_epoch}
                row.update({k: float(v) for k, v in fields.items()})
                if pending_per_class:
                    row.update(pending_per_class)
                    pending_per_class = None
                val_metric_rows.append(row)

    return train_rows, val_loss_rows, val_metric_rows


def write_csv(rows: list[dict], output_path: str):
    if not rows:
        print(f"Rien à écrire pour {output_path} (aucune ligne trouvée)")
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{output_path}: {len(rows)} lignes")


if __name__ == "__main__":
    *log_paths, output_dir = sys.argv[1:]

    all_train_rows: list[dict] = []
    all_val_loss_rows: list[dict] = []
    all_val_metric_rows: list[dict] = []
    for log_path in log_paths:
        train_rows, val_loss_rows, val_metric_rows = parse(log_path)
        all_train_rows.extend(train_rows)
        all_val_loss_rows.extend(val_loss_rows)
        all_val_metric_rows.extend(val_metric_rows)
        print(
            f"{log_path}: {len(train_rows)} lignes train, "
            f"{len(val_loss_rows)} lignes val (loss, vraie moyenne), "
            f"{len(val_metric_rows)} lignes val (métriques)"
        )

    # Dédoublonne les métriques val par epoch (garde la dernière valeur
    # rencontrée) au cas où une reprise re-validerait exactement la même
    # epoch qu'un segment précédent -- ne devrait normalement pas arriver
    # (resume restaure la position exacte), mais coûte rien à sécuriser.
    # Le CSV par itération (train) n'est pas dédoublonné, même logique :
    # on fait confiance à la reprise pour ne pas re-logger les mêmes
    # itérations.
    val_by_epoch = {row["epoch"]: row for row in all_val_metric_rows}
    all_val_metric_rows = [val_by_epoch[e] for e in sorted(val_by_epoch)]
    # Même dédoublonnage par epoch pour la loss de validation (une seule
    # ligne par epoch de validation, contrairement au CSV train par
    # itération).
    val_loss_by_epoch = {row["epoch"]: row for row in all_val_loss_rows}
    all_val_loss_rows = [val_loss_by_epoch[e] for e in sorted(val_loss_by_epoch)]
    all_train_rows.sort(key=lambda r: r["global_iter"])

    write_csv(all_train_rows, f"{output_dir}/train_losses.csv")
    write_csv(all_val_loss_rows, f"{output_dir}/val_loss.csv")
    write_csv(all_val_metric_rows, f"{output_dir}/val_metrics.csv")
