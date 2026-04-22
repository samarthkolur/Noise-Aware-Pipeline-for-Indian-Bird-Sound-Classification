"""
mining.py — Hard negative mining and dataset update loops.

1. Mine: Run inference over a source directory (e.g., unlabeled or known noise),
   identify false positives (predicted as birds), and export them.
2. Update: Take a directory of human-verified clips and append their
   embeddings to the training dataset.
"""

import json
import shutil
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from inference.prediction_api import decision_binary_gated_single
from inference.predictor import Predictor
from embedding.embedding import build_encoder
from utils.logger import get_logger

logger = get_logger(__name__)


def _parse_segment_index_from_name(name: str) -> int:
    stem = Path(name).stem
    parts = stem.rsplit("_seg", 1)
    if len(parts) == 2:
        try:
            return int(parts[1])
        except ValueError:
            return 0
    return 0


class Miner:
    """End-to-end active learning and hard-negative mining loop."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.output_dir = Path(cfg["data"]["output_dir"])
        
        # We subclass/wrap Predictor for mining
        self.predictor = Predictor(cfg)

    def mine_false_positives(self, source_dir: Path, export_dir: Path) -> None:
        """Run inference to find False Positives (noise predicted as bird).

        Uses in-memory predictions only; does not mutate ``Predictor`` routing
        directories. Routing and AE gating use ``inference.prediction_api``
        (same policy as ``Predictor``) without calling ``Predictor.predict_file``.

        Args:
            source_dir: Directory of unlabeled or known-noise audio.
            export_dir: Directory where False Positives will be copied.
        """
        export_dir.mkdir(parents=True, exist_ok=True)
        audio_files = []
        for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg"):
            audio_files.extend(list(source_dir.rglob(ext)))

        if not audio_files:
            logger.warning(f"No audio files found in {source_dir} for mining.")
            return

        logger.info(f"Mining False Positives in {len(audio_files)} files...")
        sr_out = int(self.cfg["audio"]["sample_rate"])
        device = self.predictor.device
        pred = self.predictor

        fp_count = 0
        for audio_path in tqdm(sorted(audio_files), desc="Mining"):
            try:
                metas = pred.preprocessor.process_file(
                    audio_path,
                    Path("."),
                    species="unknown",
                    persist_segments=False,
                )
                for meta in metas:
                    wf = meta.waveform_mono
                    if wf is None:
                        continue
                    emb_pt = torch.from_numpy(
                        pred.encoder.encode(wf, meta.sample_rate)
                    ).unsqueeze(0).to(device, non_blocking=True)
                    decision, _species, prob, recon_error, ae_rejected = (
                        decision_binary_gated_single(
                            pred.autoencoder,
                            pred.classifier,
                            emb_pt,
                            threshold=pred.recon_threshold,
                            high_threshold=float(pred.high_threshold),
                            low_threshold=float(pred.low_threshold),
                        )
                    )
                    if decision != "bird":
                        continue
                    res = {
                        "source_file": meta.source_file,
                        "segment_index": meta.segment_index,
                        "start_sec": meta.start_sec,
                        "end_sec": meta.end_sec,
                        "decision": decision,
                        "is_bird": True,
                        "predicted_species": "bird",
                        "confidence": float(prob),
                        "threshold_used": pred.conf_thresh,
                        "recon_error": float(recon_error),
                        "recon_error_rejected": ae_rejected,
                    }
                    dest_stem = (
                        f"{Path(res['source_file']).stem}_seg{res['segment_index']:04d}"
                    )
                    out_wav = export_dir / f"{dest_stem}.wav"
                    sf.write(
                        str(out_wav),
                        np.asarray(wf, dtype=np.float32),
                        sr_out,
                        subtype="PCM_16",
                    )
                    with open(out_wav.with_suffix(".json"), "w", encoding="utf-8") as f:
                        json.dump(res, f, indent=2)
                    fp_count += 1

            except Exception as e:
                logger.error(f"Mining failed on {audio_path.name}: {e}")

        logger.info(f"Mining complete ✓ Exported {fp_count} false positives to {export_dir}")

    def mine_false_negatives(self, noise_dir: Optional[Path] = None, export_dir: Optional[Path] = None) -> None:
        """Re-run inference on segments in outputs/noise/ to find false negatives.

        **Intentionally bypasses the autoencoder OOD gate** (no reconstruction
        rejection): embeddings go straight to the classifier. This is
        exploratory / review tooling only and does **not** reflect the deployed
        policy (deployed inference always applies the AE gate before the MLP).

        False negatives are bird sounds that were incorrectly routed to noise/.
        These are identified by re-running the classifier and finding segments
        where the model's probability is above a recovery threshold.

        Args:
            noise_dir: Directory containing noise-classified segments (default: outputs/noise/).
            export_dir: Where to copy potential false negatives for review.
        """
        if noise_dir is None:
            noise_dir = self.output_dir / "noise"
        if export_dir is None:
            export_dir = self.output_dir / "false_negatives"
        export_dir.mkdir(parents=True, exist_ok=True)

        wav_files = sorted(noise_dir.glob("*.wav"))
        if not wav_files:
            logger.warning(f"No WAV files in {noise_dir} to mine.")
            return

        logger.info(f"Mining false negatives from {len(wav_files)} noise-classified segments...")

        # Use a lower threshold to catch borderline cases
        recovery_threshold = self.cfg.get("inference", {}).get("low_threshold", 0.3)
        fn_count = 0

        for wav_path in tqdm(wav_files, desc="Mining FN"):
            try:
                waveform, sr = self.predictor._load_wav_fast(wav_path)
                emb_np = self.predictor.encoder.encode(waveform, sr)
                emb_pt = torch.from_numpy(emb_np).unsqueeze(0).to(
                    self.predictor.device
                )

                logits = self.predictor.classifier(emb_pt)
                if logits.ndim > 1:
                    logits = logits.squeeze(-1)
                prob = torch.sigmoid(logits).item()

                # If probability is above recovery threshold, this may be a
                # false negative (bird misclassified as noise)
                if prob > recovery_threshold:
                    import shutil
                    dest = export_dir / wav_path.name
                    shutil.copy2(str(wav_path), str(dest))

                    # Save metadata
                    meta = {
                        "source": str(wav_path),
                        "bird_probability": prob,
                        "recovery_threshold": recovery_threshold,
                        "action": "review_as_potential_bird",
                    }
                    with open(dest.with_suffix(".json"), "w") as f:
                        json.dump(meta, f, indent=2)

                    fn_count += 1

            except Exception as e:
                logger.error(f"Failed on {wav_path.name}: {e}")

        logger.info(
            f"False negative mining complete ✓ "
            f"Found {fn_count}/{len(wav_files)} potential false negatives "
            f"in {export_dir}"
        )

    def mine_uncertain(self, export_dir: Optional[Path] = None) -> None:
        """Export all uncertain-classified segments for manual review.

        Args:
            export_dir: Where to copy uncertain segments (default: outputs/uncertain/).
        """
        uncertain_dir = self.output_dir / "uncertain"
        if export_dir is None:
            export_dir = uncertain_dir  # They're already there

        wav_files = sorted(uncertain_dir.glob("*.wav"))
        logger.info(
            f"Uncertain segments for review: {len(wav_files)} files in {uncertain_dir}"
        )
        if wav_files:
            logger.info(
                "Review these segments and move confirmed birds to a 'verified_birds/' "
                "folder, then run: miner.update_dataset(verified_dir, target_species='<species>')"
            )

    def update_dataset(self, verified_dir: Path, target_species: str = "noise") -> None:
        """Process verified false positives and append them to the HDF5 dataset.

        Args:
            verified_dir: Directory containing human-verified .wav files.
            target_species: The label they should be given (defaults to "noise").
        """
        verified_files = sorted(list(verified_dir.glob("*.wav")))
        if not verified_files:
            logger.warning(f"No fully verified WAVs in {verified_dir}")
            return
            
        logger.info(f"Updating dataset with {len(verified_files)} verified clips as '{target_species}'...")
        
        # 1. Setup encoder & target HDF5
        encoder = build_encoder(self.cfg)
        emb_dir = Path(self.cfg["data"].get("embeddings_dir", "./data/embeddings"))
        target_h5 = emb_dir / target_species / "embeddings.h5"
        
        if not target_h5.exists():
            target_h5.parent.mkdir(parents=True, exist_ok=True)
            _init_h5(target_h5, self.cfg["embedding"]["embedding_dim"])
            
        # 2. Extract embeddings
        new_embeddings = []
        new_filenames = []
        
        for wav_path in tqdm(verified_files, desc="Encoding"):
            waveform, sr = self.predictor._load_wav_fast(wav_path)
            # Since these are already 3s chunks from mining, we just encode!
            emb_np = encoder.encode(waveform, sr)
            new_embeddings.append(emb_np)
            new_filenames.append(str(wav_path.resolve()))
            
        # 3. Append to HDF5
        with h5py.File(target_h5, "a") as h5f:
            emb_ds = h5f["embeddings"]
            fn_ds = h5f["filenames"]
            
            curr_len = len(emb_ds)
            add_len = len(new_embeddings)
            
            emb_ds.resize(curr_len + add_len, axis=0)
            fn_ds.resize(curr_len + add_len, axis=0)
            
            emb_ds[curr_len:] = np.stack(new_embeddings)
            fn_ds[curr_len:] = new_filenames
            
        logger.info(f"Appended {len(new_embeddings)} items to {target_h5}")
        
        # 4. Update global manifest
        self._update_manifest(emb_dir)
        logger.info("Dataset update complete ✓ (ready for re-training)")

    def _update_manifest(self, emb_dir: Path) -> None:
        """Rebuild the global manifest CSV using the canonical schema from EmbeddingStore.

        Columns match exactly what EmbeddingDataset.from_manifest() expects:
            species, source_file, segment_index, embedding_dim,
            model_name, sample_rate, duration_sec, hdf5_path, hdf5_row
        """
        import csv

        rows = []
        for h5_path in sorted(emb_dir.rglob("embeddings.h5")):
            species = h5_path.parent.name
            with h5py.File(h5_path, "r") as h5f:
                filenames = h5f["filenames"][:]
                n = len(filenames)
            for i, fn in enumerate(filenames):
                rows.append({
                    "species": species,
                    "source_file": fn.decode("utf-8") if isinstance(fn, bytes) else fn,
                    "segment_index": _parse_segment_index_from_name(
                        fn.decode("utf-8") if isinstance(fn, bytes) else fn
                    ),
                    "embedding_dim": self.cfg["embedding"]["embedding_dim"],
                    "model_name": "birdnet_v2.4",
                    "sample_rate": self.cfg["audio"]["sample_rate"],
                    "duration_sec": self.cfg["audio"]["segment_duration_s"],
                    "hdf5_path": str(h5_path),
                    "hdf5_row": i,
                })

        fieldnames = [
            "species", "source_file", "segment_index", "embedding_dim",
            "model_name", "sample_rate", "duration_sec", "hdf5_path", "hdf5_row",
        ]
        manifest_path = emb_dir / "manifest.csv"
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        logger.info(f"Manifest rebuilt → {manifest_path} ({len(rows)} rows)")


def _init_h5(path: Path, dim: int) -> None:
    """Initialize an empty resizeable HDF5 dataset."""
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset(
            "embeddings",
            shape=(0, dim),
            maxshape=(None, dim),
            dtype=np.float32,
            compression="gzip"
        )
        dt = h5py.string_dtype(encoding="utf-8")
        h5f.create_dataset(
            "filenames",
            shape=(0,),
            maxshape=(None,),
            dtype=dt,
            compression="gzip"
        )
