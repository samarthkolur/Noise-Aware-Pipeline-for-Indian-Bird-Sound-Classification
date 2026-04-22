"""
bird_rescue.py — Second-stage override when Noise Segregation V2 says "noise".

Runs BirdNET on the 3s segment, applies the trained binary EmbeddingClassifier
(no autoencoder), and if P(bird) >= threshold, treats the segment as bird for
folder routing. This recovers V2 false negatives (birds sent toward noise/) at
a configurable precision cost.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from models.classifier import EmbeddingClassifier
from utils.logger import get_logger

logger = get_logger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_checkpoint_dir(cfg: dict) -> Path:
    br = cfg.get("bird_rescue", {}) or {}
    override = br.get("checkpoint_dir")
    if override:
        p = Path(override).expanduser()
        if not p.is_absolute():
            p = _project_root() / p
        return p.resolve()
    tdir = cfg.get("training", {}).get("checkpoint_dir", "./checkpoints")
    p = Path(tdir)
    if not p.is_absolute():
        p = _project_root() / p
    return p.resolve()


def _load_threshold(cfg: dict, meta: dict) -> float:
    br = cfg.get("bird_rescue", {}) or {}
    raw = br.get("threshold", "auto")
    if raw is None or (isinstance(raw, str) and str(raw).strip().lower() == "auto"):
        t = float(meta.get("optimal_threshold", 0.5))
        logger.info(f"[bird_rescue] Using optimal_threshold from training meta: {t:.4f}")
        return t
    return float(raw)


class V2BirdRescueGate:
    """Lazy BirdNET + binary MLP; only used for V2 noise → bird overrides."""

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._device = self._resolve_device(cfg)
        self._encoder = None
        self._classifier: Optional[EmbeddingClassifier] = None
        self._threshold: float = 0.5
        self._ok = False
        self._init_error: Optional[str] = None
        self._try_init()

    @staticmethod
    def _resolve_device(cfg: dict) -> torch.device:
        ds = cfg.get("project", {}).get("device", "auto")
        if str(ds) == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(str(ds))

    def _try_init(self) -> None:
        chk_dir = _resolve_checkpoint_dir(self._cfg)
        meta_path = chk_dir / "best_model_meta.json"
        ckpt_path = chk_dir / "best_model.pt"
        if not meta_path.is_file() or not ckpt_path.is_file():
            self._init_error = (
                f"Missing {meta_path.name} or {ckpt_path.name} under {chk_dir}. "
                "Train the classifier or set bird_rescue.enabled: false."
            )
            logger.warning(f"[bird_rescue] Disabled: {self._init_error}")
            return
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except OSError as e:
            self._init_error = str(e)
            logger.warning(f"[bird_rescue] Disabled: {self._init_error}")
            return

        if not meta.get("binary", True):
            logger.warning(
                "[bird_rescue] best_model is not binary — rescue only supports "
                "binary bird vs noise. Disabling."
            )
            self._init_error = "multiclass checkpoint"
            return

        self._threshold = _load_threshold(self._cfg, meta)
        # Building BirdNET is deferred until first encode (heavy).
        from embedding.embedding import BirdNETModelNotFoundError, build_encoder

        try:
            self._encoder = build_encoder(self._cfg)
        except BirdNETModelNotFoundError as e:
            self._init_error = str(e)
            logger.warning(f"[bird_rescue] Disabled: {e}")
            return

        hdim = self._cfg.get("model", {}).get("hidden_dims", [512, 256])
        dropout = float(self._cfg.get("model", {}).get("dropout", 0.3))
        emb_dim = int(self._cfg["embedding"]["embedding_dim"])
        clf = EmbeddingClassifier(
            input_dim=emb_dim,
            num_classes=1,
            hidden_dims=list(hdim),
            dropout=dropout,
        ).to(self._device)
        try:
            chk = torch.load(ckpt_path, map_location=self._device, weights_only=True)
            clf.load_state_dict(chk["model_state_dict"], strict=True)
        except (OSError, RuntimeError) as e:
            self._init_error = str(e)
            logger.warning(f"[bird_rescue] Disabled: {e}")
            return
        clf.eval()
        self._classifier = clf
        self._ok = True
        logger.info(
            f"[bird_rescue] Active: checkpoint={chk_dir}  "
            f"threshold={self._threshold:.4f}  (V2 noise → MLP bird if P≥threshold)"
        )

    @property
    def is_active(self) -> bool:
        return self._ok

    @property
    def threshold(self) -> float:
        return self._threshold

    @torch.no_grad()
    def bird_probability(self, segment_mono: np.ndarray, sample_rate: int) -> float:
        """P(bird) in [0,1] for one mono float32 segment. Caller ensures active."""
        if not self._ok or self._encoder is None or self._classifier is None:
            return 0.0
        emb = self._encoder.encode(
            np.asarray(segment_mono, dtype=np.float32).reshape(-1), int(sample_rate)
        )
        x = torch.from_numpy(emb).float().unsqueeze(0).to(self._device)
        logit = self._classifier(x)
        if logit.ndim > 1:
            logit = logit.squeeze(-1)
        return float(torch.sigmoid(logit[0]).item())


def build_v2_bird_rescue_or_none(cfg: dict) -> Optional[V2BirdRescueGate]:
    """Return a configured gate, or None if disabled or init failed."""
    br = cfg.get("bird_rescue", {}) or {}
    if not bool(br.get("enabled", False)):
        return None
    gate = V2BirdRescueGate(cfg)
    return gate if gate.is_active else None
