"""Mise en place de NanoDet pour les démos, sans bruit dans la console.

Les démos n'utilisent NanoDet que pour le prétraitement et le
post-traitement, l'inférence passant par ONNX : inutile de charger les
poids ImageNet du backbone, que NanoDet téléchargerait sinon.

À importer avant tout module nanodet, pour filtrer ses avertissements.
"""

import contextlib
import io
import warnings

# Avertissements connus et sans effet : API pkg_resources utilisée par
# pytorch-lightning, et appel à torch.meshgrid dans NanoDet.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="torch.meshgrid", category=UserWarning)

from nanodet.model.arch import build_model  # noqa: E402
from nanodet.util import cfg, load_config  # noqa: E402


def build_postprocessor(config_path: str):
    """Charge la config dans `cfg` et construit le modèle, sans poids :
    seul model.head.post_process sert. Les messages de NanoDet sont masqués."""
    load_config(cfg, config_path)
    cfg.defrost()
    cfg.model.arch.backbone.pretrain = False
    cfg.freeze()
    with contextlib.redirect_stdout(io.StringIO()):
        return build_model(cfg.model)
