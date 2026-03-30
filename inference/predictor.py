"""
predictor.py — End-to-end inference pipeline with three-class routing.

Flow:
1. Load raw audio file
2. Segment (3s) and remove silence via Preprocessor
3. Extract embeddings via configured Encoder
4. Classify each segment via trained EmbeddingClassifier
5. Route segments into:
   - `outputs/clean_birds/`  (prob > high_threshold)
   - `outputs/noise/`        (prob < low_threshold)
   - `outputs/uncertain/`    (between thresholds — for manual review)
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from embedding.embedding import build_encoder
from models.classifier import EmbeddingClassifier
from preprocessing.preprocessing import Preprocessor, SegmentMeta
from utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """Run end-to-end inference on raw audio files."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.device = self._resolve_device(cfg)
        
        # Output routing
        self.output_dir = Path(cfg["data"]["output_dir"])
        self.birds_dir = self.output_dir / "clean_birds"
        self.noise_dir = self.output_dir / "noise"
        self.uncertain_dir = self.output_dir / "uncertain"
        
        self.birds_dir.mkdir(parents=True, exist_ok=True)
        self.noise_dir.mkdir(parents=True, exist_ok=True)
        self.uncertain_dir.mkdir(parents=True, exist_ok=True)

        # Thresholds
        inf_cfg = cfg.get("inference", {})
        self.high_threshold = inf_cfg.get("high_threshold", 0.7)
        self.low_threshold = inf_cfg.get("low_threshold", 0.3)

        # Load optimal threshold from training (if available)
        conf_thresh_raw = inf_cfg.get("confidence_threshold", "auto")
        if conf_thresh_raw == "auto":
            self.conf_thresh = self._load_optimal_threshold(cfg)
        else:
            self.conf_thresh = float(conf_thresh_raw)

        logger.info(
            f"Thresholds: optimal={self.conf_thresh:.3f}, "
            f"high={self.high_threshold:.3f}, low={self.low_threshold:.3f}"
        )

        # 1. Preprocessor (resample, segment, silence removal)
        self.preprocessor = Preprocessor(cfg)
        
        # 2. Encoder (BirdNET or placeholder)
        self.encoder = build_encoder(cfg)

        # 3. Classifier + Label Map
        self.classifier, self.label_map, self.binary = self._load_model(cfg)
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

    # ── Public API ──────────────────────────────────────────

    def run(self) -> None:
        """Run full inference pipeline on `cfg.data.raw_dir`."""
        raw_dir = Path(self.cfg["data"]["raw_dir"])
        audio_files = []
        for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg"):
            audio_files.extend(list(raw_dir.rglob(ext)))
            
        if not audio_files:
            logger.warning(f"No audio files found in {raw_dir}")
            return

        logger.info(f"Starting inference on {len(audio_files)} files...")
        
        counts = {"bird": 0, "noise": 0, "uncertain": 0, "total": 0}

        for audio_path in tqdm(sorted(audio_files), desc="Inference"):
            try:
                results = self.predict_file(audio_path)
                
                for res in results:
                    counts["total"] += 1
                    counts[res["decision"]] += 1
                    
            except Exception as e:
                logger.error(f"Failed to process {audio_path.name}: {e}")

        logger.info("Inference complete ✓")
        logger.info(f"  Segments processed:      {counts['total']}")
        logger.info(f"  Routed to clean_birds:   {counts['bird']}")
        logger.info(f"  Routed to noise:         {counts['noise']}")
        logger.info(f"  Routed to uncertain:     {counts['uncertain']}")

    def predict_file(self, audio_path: Path) -> List[Dict]:
        """Process a single file end-to-end and route segments."""
        import tempfile
        
        results = []
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # 1. Preprocess → list of SegmentMeta
            metas = self.preprocessor.process_file(
                audio_path, output_dir=tmp_path, species="unknown"
            )
            
            for meta in metas:
                seg_wav_path = Path(meta.output_path)
                
                # 2. Extract embedding
                waveform, sr = self._load_wav_fast(seg_wav_path)
                emb_np = self.encoder.encode(waveform, sr)
                
                # 3. Classify with three-class routing
                emb_pt = torch.from_numpy(emb_np).unsqueeze(0).to(self.device)
                decision, top_species, prob = self._classify_segment(emb_pt)
                
                res = {
                    "source_file": meta.source_file,
                    "segment_index": meta.segment_index,
                    "start_sec": meta.start_sec,
                    "end_sec": meta.end_sec,
                    "decision": decision,  # "bird", "noise", or "uncertain"
                    "is_bird": decision == "bird",
                    "predicted_species": top_species,
                    "confidence": float(prob),
                    "threshold_used": self.conf_thresh,
                    "tmp_wav_path": seg_wav_path,
                }
                
                # 4. Route audio
                if decision == "bird":
                    target_dir = self.birds_dir
                elif decision == "noise":
                    target_dir = self.noise_dir
                else:
                    target_dir = self.uncertain_dir
                    
                self._save_prediction(res, target_dir)
                results.append(res)
                
        return results

    # ── Private ─────────────────────────────────────────────

    @torch.no_grad()
    def _classify_segment(self, embedding: torch.Tensor) -> Tuple[str, str, float]:
        """Run classifier on a single embedding.
        
        Returns:
            (decision, species_name, probability)
            decision is one of: "bird", "noise", "uncertain"
        """
        logits = self.classifier(embedding)
        
        if self.binary:
            if logits.ndim > 1:
                logits = logits.squeeze(-1)
            prob = torch.sigmoid(logits).item()
            
            # Three-class routing
            if prob >= self.high_threshold:
                decision = "bird"
                species = "bird"
            elif prob <= self.low_threshold:
                decision = "noise"
                species = "noise"
            else:
                decision = "uncertain"
                species = "uncertain"
                
            return decision, species, prob
            
        else:
            probs = F.softmax(logits, dim=1).squeeze(0)
            conf, pred_idx = probs.max(dim=0)
            conf = conf.item()
            pred_idx = pred_idx.item()
            species_name = self.inv_label_map.get(pred_idx, "unknown")
            
            if species_name == "noise":
                decision = "noise"
            elif conf >= self.conf_thresh:
                decision = "bird"
            else:
                decision = "uncertain"
                
            return decision, species_name, conf

    def _save_prediction(self, res: dict, target_dir: Path) -> None:
        """Move temporary wav to target dir and write JSON metadata."""
        src_wav = Path(res["tmp_wav_path"])
        
        stem = Path(res["source_file"]).stem
        dest_stem = f"{stem}_seg{res['segment_index']:04d}"
        dest_wav = target_dir / f"{dest_stem}.wav"
        dest_json = target_dir / f"{dest_stem}.json"
        
        if src_wav.exists():
            shutil.move(str(src_wav), str(dest_wav))
        else:
            logger.error(f"Missing temp wav: {src_wav}")
        
        meta = {k: v for k, v in res.items() if k != "tmp_wav_path"}
        meta["output_path"] = str(dest_wav)
        
        with open(dest_json, "w") as f:
            json.dump(meta, f, indent=2)

    def _load_optimal_threshold(self, cfg: dict) -> float:
        """Load optimal threshold from training metadata, fallback to 0.5."""
        chkpt_dir = Path(cfg["training"]["checkpoint_dir"])
        meta_path = chkpt_dir / "best_model_meta.json"
        
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if "optimal_threshold" in meta:
                thresh = meta["optimal_threshold"]
                logger.info(f"Using optimal threshold from training: {thresh:.3f}")
                return thresh
                
        logger.info("No optimal threshold found, using default 0.5")
        return 0.5

    def _load_model(self, cfg: dict) -> Tuple[torch.nn.Module, dict, bool]:
        """Load the trained MLP checkpoint and metadata."""
        chkpt_dir = Path(cfg["training"]["checkpoint_dir"])
        meta_path = chkpt_dir / "best_model_meta.json"
        
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Missing {meta_path}. Run training first."
            )
            
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        binary = meta["binary"]
        label_map = meta["label_map"]
        num_classes = 1 if binary else len(label_map)
        
        classifier = EmbeddingClassifier(
            input_dim=cfg["embedding"]["embedding_dim"],
            num_classes=num_classes,
            hidden_dims=cfg["model"].get("hidden_dims", [512, 256]),
        ).to(self.device)
        
        chkpt_path = chkpt_dir / "best_model.pt"
        chkpt = torch.load(chkpt_path, map_location=self.device, weights_only=True)
        classifier.load_state_dict(chkpt["model_state_dict"])
        classifier.eval()
        
        logger.info(f"Loaded classifier from {chkpt_path} (binary={binary})")
        return classifier, label_map, binary

    @staticmethod
    def _load_wav_fast(path: Path) -> Tuple:
        """Load temporary WAV file using soundfile."""
        import soundfile as sf
        waveform_np, sr = sf.read(str(path), dtype="float32", always_2d=True)
        waveform = waveform_np.mean(axis=1)  # mono
        return waveform, sr

    @staticmethod
    def _resolve_device(cfg: dict) -> torch.device:
        device_str = cfg.get("project", {}).get("device", "auto")
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)
