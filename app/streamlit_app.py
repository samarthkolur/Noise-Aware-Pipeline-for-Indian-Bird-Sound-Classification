"""Production Streamlit demo for the noise-aware bird sound pipeline."""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embedding.embedding import build_encoder
from models.autoencoder import EmbeddingAutoencoder
from models.classifier import EmbeddingClassifier
from preprocessing.preprocessing import Preprocessor, SegmentMeta
from utils.ae_checkpoint import load_autoencoder_state
from utils.config import load_config


def _resolve_config_path() -> Path:
    env = os.environ.get("PIPELINE_CONFIG", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    cfg = PROJECT_ROOT / "configs" / "config.yaml"
    if cfg.is_file():
        return cfg
    return PROJECT_ROOT / "config.yaml"


def _resolve_relative_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve all relative paths in the config to be absolute from PROJECT_ROOT.

    This is critical because Streamlit's CWD is ``app/``, but config paths
    like ``./checkpoints`` are relative to the project root.
    """
    import copy
    cfg = copy.deepcopy(cfg)

    # Keys known to contain filesystem paths
    path_keys = {
        ("data", "raw_dir"),
        ("data", "processed_dir"),
        ("data", "embeddings_dir"),
        ("data", "output_dir"),
        ("training", "checkpoint_dir"),
        ("training", "log_dir"),
        ("inference", "checkpoint"),
        ("autoencoder", "checkpoint_path"),
        ("evaluation", "results_dir"),
    }
    for section, key in path_keys:
        if section in cfg and key in cfg[section]:
            val = cfg[section][key]
            if isinstance(val, str) and not Path(val).is_absolute():
                cfg[section][key] = str((PROJECT_ROOT / val).resolve())

    return cfg


@st.cache_data(show_spinner=False)
def load_pipeline_config(config_path_str: str) -> dict[str, Any]:
    cfg = load_config(config_path_str)
    return _resolve_relative_paths(cfg)


@st.cache_resource(show_spinner=False)
def build_cached_encoder(cfg: dict[str, Any]):
    return build_encoder(cfg)


@st.cache_resource(show_spinner=False)
def load_cached_classifier(cfg: dict[str, Any]) -> tuple[torch.nn.Module, bool, float]:
    chkpt_dir = PROJECT_ROOT / cfg["training"]["checkpoint_dir"]
    meta_path = chkpt_dir / "best_model_meta.json"
    chkpt_path = chkpt_dir / "best_model.pt"
    if not meta_path.exists() or not chkpt_path.exists():
        raise FileNotFoundError(
            "Missing classifier checkpoint metadata. Run `python scripts/run_pipeline.py "
            "--config configs/config.yaml --stage train` first."
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    binary = bool(meta.get("binary", True))
    optimal_threshold = float(meta.get("optimal_threshold", 0.5))
    num_classes = 1 if binary else len(meta.get("label_map", {}))
    classifier = EmbeddingClassifier(
        input_dim=int(cfg["embedding"]["embedding_dim"]),
        num_classes=num_classes,
        hidden_dims=cfg["model"].get("hidden_dims", [512, 256]),
        dropout=float(cfg["model"].get("dropout", 0.3)),
    )
    chkpt = torch.load(chkpt_path, map_location="cpu", weights_only=True)
    classifier.load_state_dict(chkpt["model_state_dict"])
    classifier.eval()
    return classifier, binary, optimal_threshold


@st.cache_resource(show_spinner=False)
def load_cached_autoencoder(cfg: dict[str, Any]) -> tuple[torch.nn.Module, float]:
    """Load AE + τ_AE (required; same artifacts as CLI inference)."""
    ae_cfg = dict(cfg.get("autoencoder", {}))
    cp = ae_cfg.get("checkpoint_path", "./checkpoints/autoencoder.pt")
    p = Path(cp)
    if not p.is_absolute():
        ae_cfg = {**ae_cfg, "checkpoint_path": str((PROJECT_ROOT / p).resolve())}
    cfg_resolved = {**cfg, "autoencoder": ae_cfg}
    ae, tau, _meta = load_autoencoder_state(cfg_resolved, torch.device("cpu"))
    return ae, tau


def _work_dir() -> Path:
    path = PROJECT_ROOT / ".streamlit_demo"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _materialize_upload(upload_name: str, raw_bytes: bytes, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload_name).suffix.lower() or ".wav"
    src = out_dir / f"upload{suffix}"
    src.write_bytes(raw_bytes)

    if suffix == ".wav":
        return src

    try:
        waveform_np, sr = sf.read(str(src), always_2d=True)
        waveform = torch.from_numpy(waveform_np).transpose(0, 1)
    except Exception as e:
        raise RuntimeError(f"Could not decode {suffix}: {e}") from e

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    decoded = out_dir / "upload_decoded.wav"
    sf.write(str(decoded), waveform.squeeze(0).numpy().astype(np.float32), int(sr), subtype="PCM_16")
    return decoded


def _audio_metadata(path: Path) -> dict[str, Any]:
    info = sf.info(str(path))
    return {
        "samplerate": int(info.samplerate),
        "channels": int(info.channels),
        "duration_sec": round(float(info.duration), 2),
        "frames": int(info.frames),
        "format": info.format,
        "subtype": info.subtype,
    }


def _load_wav_mono_np(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1).astype(np.float32), int(sr)


def _waveform_figure(path: Path) -> plt.Figure:
    x, sr = _load_wav_mono_np(path)
    t = np.arange(len(x)) / float(sr)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(t, x, linewidth=0.5, color="#1d4ed8")
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig


def _spectrogram_figure(path: Path) -> plt.Figure:
    x, sr = _load_wav_mono_np(path)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.specgram(x, NFFT=1024, Fs=sr, noverlap=512, cmap="magma")
    ax.set_title("Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.tight_layout()
    return fig


def _raw_vote_decision(
    noise_votes: int | None,
    total: int | None,
    majority_tie: str = "bird",
) -> str:
    """Compute the raw majority-vote V2 decision from vote counts.

    This bypasses the bird guard override that bakes into ``v2_label``.
    The raw votes are the ground truth of the subframe scoring.
    """
    if noise_votes is None or total is None or total == 0:
        return "n/a"
    half = total // 2
    if noise_votes > half:
        return "noise"
    if noise_votes < half:
        return "bird"
    # Tie (e.g. 3/6) — use config majority_tie
    return majority_tie


def _segment_summary_rows(
    metas: list[SegmentMeta],
    majority_tie: str = "bird",
    pred_map: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for meta in metas:
        raw_decision = _raw_vote_decision(
            meta.v2_noise_subframe_votes, meta.v2_total_subframes, majority_tie
        )
        row: dict[str, Any] = {
            "segment": meta.segment_index,
            "time_window": f"{meta.start_sec:.2f}-{meta.end_sec:.2f}",
            "rms_db": meta.rms_db,
            "v2_label (bird-guard)": meta.v2_label,
            "v2_score": meta.v2_mean_score,
            "noise_votes": (
                f"{meta.v2_noise_subframe_votes}/{meta.v2_total_subframes}"
                if meta.v2_noise_subframe_votes is not None and meta.v2_total_subframes is not None
                else None
            ),
            "raw_vote_decision": raw_decision,
        }
        if pred_map and meta.segment_index in pred_map:
            pred = pred_map[meta.segment_index]
            row["final_label"] = pred["label"]
            row["mlp_prob"] = pred.get("prob")
            row["ae_error"] = pred.get("recon_error")
            row["routed_by"] = pred.get("routed_by", "")
        row["output_path"] = meta.output_path
        rows.append(row)
    return rows


# _classify_segment has been removed in favor of Predictor.predict_file


def _render_segment_group(title: str, items: list[dict[str, Any]]) -> None:
    st.markdown(f"### {title}")
    if not items:
        st.caption("No segments in this bucket.")
        return

    for idx, item in enumerate(items):
        meta: SegmentMeta = item["meta"]
        pred = item["pred"]
        routed_by = pred.get("routed_by", "")
        votes_str = (
            f"{meta.v2_noise_subframe_votes}/{meta.v2_total_subframes}"
            if meta.v2_noise_subframe_votes is not None else "-"
        )
        parts = [
            f"Segment {meta.segment_index}",
            f"{meta.start_sec:.2f}-{meta.end_sec:.2f}s",
            f"RMS {meta.rms_db:.1f} dB",
            f"votes {votes_str}",
            f"→ {pred['label']}",
            f"p={pred['prob']:.3f}",
        ]
        if pred.get("recon_error") is not None:
            parts.append(f"AE={pred['recon_error']:.5f}")
        if routed_by:
            parts.append(f"via {routed_by}")
        with st.expander(" | ".join(parts), expanded=(idx == 0)):
            wav_path = Path(meta.output_path)
            if wav_path.is_file():
                st.audio(wav_path.read_bytes(), format="audio/wav")
            st.json(
                {
                    "segment_meta": asdict(meta),
                    "prediction": pred,
                },
                expanded=False,
            )


def main() -> None:
    st.set_page_config(page_title="Noise-Aware Bird Sound Demo", layout="wide")
    st.title("Noise-Aware Pipeline for Indian Bird Sound Classification")
    st.caption(
        "Production demo for the report architecture: preprocessing, Noise Segregation V2, "
        "BirdNET embeddings, autoencoder-based OOD rejection gate, focal-loss MLP, "
        "and three-band routing."
    )

    cfg_path = _resolve_config_path()
    if not cfg_path.is_file():
        st.error(f"Config not found: {cfg_path}")
        st.stop()

    try:
        cfg = load_pipeline_config(str(cfg_path))
    except Exception as e:
        st.error(f"Failed to load config: {e}")
        st.stop()

    st.sidebar.header("Runtime")
    st.sidebar.code(str(cfg_path))
    st.sidebar.write(
        {
            "pipeline.mode": cfg.get("pipeline", {}).get("mode"),
            "silence_rms_db": cfg.get("silence_removal", {}).get("rms_threshold_db"),
            "tau_low": cfg.get("inference", {}).get("low_threshold"),
            "tau_high": cfg.get("inference", {}).get("high_threshold"),
            "autoencoder.latent_dim": cfg.get("autoencoder", {}).get("latent_dim"),
        }
    )

    landing, upload_tab, results_tab = st.tabs(["Overview", "Upload", "Results"])

    with landing:
        col1, col2 = st.columns([1.2, 1.0])
        with col1:
            st.markdown(
                """
This demo follows the report architecture end-to-end:

1. Raw audio is standardized to 48 kHz mono and segmented into 3-second clips.
2. RMS filtering removes low-energy segments.
3. Noise Segregation V2 scores six 0.5-second subframes per segment.
4. BirdNET V2.4 extracts 1024-dimensional embeddings.
5. A bird-trained autoencoder rejects OOD embeddings (high reconstruction error) as noise.
6. A focal-loss MLP routes each in-distribution segment into Bird, Noise, or Uncertain.
                """
            )
        with col2:
            st.info(
                "Expected deployment thresholds from the report:\n\n"
                f"- tau_low = {cfg['inference']['low_threshold']}\n"
                f"- tau_high = {cfg['inference']['high_threshold']}\n"
                f"- RMS gate = {cfg['silence_removal']['rms_threshold_db']} dB"
            )

    uploads = upload_tab.file_uploader(
        "Upload one or more WAV/MP3 recordings",
        type=["wav", "mp3"],
        accept_multiple_files=True,
    )
    if not uploads:
        with upload_tab:
            st.caption("Upload audio files to start the demo.")
        return

    selected_name = upload_tab.selectbox(
        "Preview file",
        options=[u.name for u in uploads],
        index=0,
    )
    selected = next(u for u in uploads if u.name == selected_name)
    raw_bytes = selected.getvalue()

    upload_key = f"{selected.name}:{len(raw_bytes)}"
    if st.session_state.get("upload_key") != upload_key:
        st.session_state["upload_key"] = upload_key
        st.session_state.pop("segment_metas", None)
        st.session_state.pop("segment_predictions", None)
        st.session_state["preprocessing_run"] = False

    source_dir = _work_dir() / upload_key.replace(":", "_")
    source_path = _materialize_upload(selected.name, raw_bytes, source_dir)
    metadata = _audio_metadata(source_path)

    with upload_tab:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duration", f"{metadata['duration_sec']} s")
        c2.metric("Sample rate", f"{metadata['samplerate']} Hz")
        c3.metric("Channels", metadata["channels"])
        c4.metric("Format", metadata["format"])
        st.audio(raw_bytes, format="audio/wav" if selected.name.lower().endswith(".wav") else "audio/mp3")

        wf_col, spec_col = st.columns(2)
        with wf_col:
            wf_fig = _waveform_figure(source_path)
            st.pyplot(wf_fig)
            plt.close(wf_fig)
        with spec_col:
            spec_fig = _spectrogram_figure(source_path)
            st.pyplot(spec_fig)
            plt.close(spec_fig)

        if st.button("Run preprocessing + V2 segregation", type="primary"):
            seg_out = source_dir / "segments"
            if seg_out.exists():
                shutil.rmtree(seg_out, ignore_errors=True)
            preprocessor = Preprocessor(cfg)
            with st.spinner("Segmenting audio and applying Noise Segregation V2..."):
                metas = preprocessor.process_file(source_path, seg_out, species="uploaded_demo")
            st.session_state["segment_metas"] = metas
            st.session_state["preprocessing_run"] = True
            st.session_state.pop("segment_predictions", None)

    metas: list[SegmentMeta] = st.session_state.get("segment_metas", [])
    has_run = st.session_state.get("preprocessing_run", False)
    
    if not metas:
        with results_tab:
            if has_run:
                st.info("Preprocessing complete, but 0 segments were kept. All segments were filtered out by the Silence Gate or Noise Reduction (RMS too low). If you uploaded pure stationary noise, this is the expected behavior.")
            else:
                st.caption("Run preprocessing first to generate segments.")
        return

    majority_tie = cfg.get("noise_segregation", {}).get("v2", {}).get("majority_tie", "bird")
    raw_bird = [
        m for m in metas
        if _raw_vote_decision(m.v2_noise_subframe_votes, m.v2_total_subframes, majority_tie) == "bird"
    ]
    raw_noise = [
        m for m in metas
        if _raw_vote_decision(m.v2_noise_subframe_votes, m.v2_total_subframes, majority_tie) == "noise"
    ]
    pred_map_for_table: dict[int, dict[str, Any]] = st.session_state.get("segment_predictions", {})
    with results_tab:
        s1, s2, s3 = st.columns(3)
        s1.metric("Segments written", len(metas))
        s2.metric("V2 bird (raw votes)", len(raw_bird))
        s3.metric("V2 noise (raw votes)", len(raw_noise))
        st.dataframe(
            _segment_summary_rows(metas, majority_tie, pred_map_for_table),
            use_container_width=True,
            hide_index=True,
        )

    device = torch.device("cpu")
    try:
        classifier, binary, optimal_threshold = load_cached_classifier(cfg)
    except Exception as e:
        with results_tab:
            st.warning(str(e))
        return

    if not binary:
        with results_tab:
            st.error("The current checkpoint is not binary bird/noise.")
        return

    encoder = build_cached_encoder(cfg)
    try:
        autoencoder, recon_threshold = load_cached_autoencoder(cfg)
    except FileNotFoundError as e:
        with results_tab:
            st.error(str(e))
        return

    with results_tab:
        st.info(
            f"Classifier ready. Validation-optimal threshold={optimal_threshold:.4f}; "
            f"deployment bands use tau_low={cfg['inference']['low_threshold']:.2f}, "
            f"tau_high={cfg['inference']['high_threshold']:.2f}."
        )

        if st.button("Run full routing demo"):
            from inference.predictor import Predictor
            predictor = Predictor(cfg)

            progress = st.progress(0.0, text="Running Predictor pipeline...")
            
            # Use Predictor on the source file to guarantee perfect tracing.
            # (In-memory predictions to avoid side effects)
            predictor_results = predictor.predict_file(source_path, persist_outputs=False)
            
            preds: dict[int, dict[str, Any]] = {}
            for i, res in enumerate(predictor_results):
                preds[res["segment_index"]] = {
                    "label": res["decision"].capitalize(),
                    "prob": res["confidence"],
                    "recon_error": res["recon_error"],
                    "ae_rejected": res["recon_error_rejected"],
                    "routed_by": res["routed_by"],
                }
                progress.progress((i + 1) / max(len(predictor_results), 1), text=f"Processed {i+1}/{len(predictor_results)} segments")

            st.session_state["segment_predictions"] = preds
            progress.empty()

    pred_map: dict[int, dict[str, Any]] = st.session_state.get("segment_predictions", {})
    if not pred_map:
        return

    grouped = {"Bird": [], "Noise": [], "Uncertain": []}
    for meta in metas:
        pred = pred_map.get(meta.segment_index, {"label": "Uncertain", "prob": 0.0, "routed_by": ""})
        grouped[pred["label"]].append({"meta": meta, "pred": pred})

    with results_tab:
        r1, r2, r3 = st.columns(3)
        r1.metric("Bird", len(grouped["Bird"]))
        r2.metric("Noise", len(grouped["Noise"]))
        r3.metric("Uncertain", len(grouped["Uncertain"]))

        bird_tab, noise_tab, uncertain_tab = st.tabs(["Bird", "Noise", "Uncertain"])
        with bird_tab:
            _render_segment_group("Confident Bird Segments", grouped["Bird"])
        with noise_tab:
            _render_segment_group("Confident Noise Segments", grouped["Noise"])
        with uncertain_tab:
            _render_segment_group("Uncertain Segments For Review", grouped["Uncertain"])


if __name__ == "__main__":
    main()
