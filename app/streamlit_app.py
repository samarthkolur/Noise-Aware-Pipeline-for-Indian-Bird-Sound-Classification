"""
Streamlit demo: upload audio → existing Preprocessor.segmentation (via process_file).

Run from project root:  streamlit run app.py
Optional: PIPELINE_CONFIG=/path/to/config.yaml
"""

from __future__ import annotations
import sys
from pathlib import Path as _PathSetup

# Ensure project root is on sys.path
_PROJECT_ROOT = _PathSetup(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import torchaudio
import torch

from src.preprocessing.preprocessing import Preprocessor, SegmentMeta
from src.embedding.embedding import build_encoder
from src.models.autoencoder import EmbeddingAutoencoder
from src.models.classifier import EmbeddingClassifier
from src.utils.config import load_config


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_config_path() -> Path:
    env = os.environ.get("PIPELINE_CONFIG", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return _project_root() / "configs" / "config.yaml"


@st.cache_data(show_spinner=False)
def load_pipeline_config(config_path_str: str) -> dict:
    """Cache YAML config by resolved path string (no hardcoded repo paths in logic)."""
    return load_config(config_path_str)


@st.cache_resource(show_spinner=False)
def build_cached_encoder(cfg: dict):
    """Build embedding encoder once per session."""
    return build_encoder(cfg)


@st.cache_resource(show_spinner=False)
def load_cached_classifier(cfg: dict) -> tuple[torch.nn.Module, dict, bool, float]:
    """Load trained classifier checkpoint (no training in UI).

    Returns: (classifier, label_map, binary, optimal_threshold)
    """
    chkpt_dir = _PROJECT_ROOT / cfg["training"]["checkpoint_dir"]
    meta_path = chkpt_dir / "best_model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing `{meta_path}`. Run `python run_pipeline.py --stage train` first.")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = __import__("json").load(f)

    binary = bool(meta.get("binary", True))
    label_map = meta.get("label_map", {"noise": 0, "bird": 1})
    optimal_threshold = float(meta.get("optimal_threshold", 0.5))

    num_classes = 1 if binary else len(label_map)
    classifier = EmbeddingClassifier(
        input_dim=int(cfg["embedding"]["embedding_dim"]),
        num_classes=num_classes,
        hidden_dims=cfg["model"].get("hidden_dims", [512, 256]),
        dropout=float(cfg["model"].get("dropout", 0.3)),
    )

    chkpt_path = chkpt_dir / "best_model.pt"
    if not chkpt_path.exists():
        raise FileNotFoundError(f"Missing `{chkpt_path}`. Run training first.")

    chkpt = torch.load(chkpt_path, map_location="cpu", weights_only=True)
    classifier.load_state_dict(chkpt["model_state_dict"])
    classifier.eval()
    return classifier, label_map, binary, optimal_threshold


@st.cache_resource(show_spinner=False)
def load_cached_autoencoder(cfg: dict) -> tuple[torch.nn.Module | None, float]:
    """Load AE gate if enabled and checkpoint exists; else return (None, 0.0)."""
    ae_cfg = cfg.get("autoencoder", {})
    if not ae_cfg.get("enabled", False):
        return None, 0.0

    ae_path = cfg.get("autoencoder", {}).get("checkpoint_path", "./checkpoints/autoencoder.pt")
    chkpt_path = _PROJECT_ROOT / ae_path
    meta_path = chkpt_path.with_name("autoencoder_meta.json")
    if not chkpt_path.exists():
        return None, 0.0

    emb_dim = int(cfg["embedding"]["embedding_dim"])
    ae = EmbeddingAutoencoder(input_dim=emb_dim)
    chk = torch.load(chkpt_path, map_location="cpu", weights_only=True)
    ae.load_state_dict(chk["model_state_dict"])
    ae.eval()

    thresh_cfg = ae_cfg.get("recon_threshold", "auto")
    if thresh_cfg == "auto" and meta_path.exists():
        import json

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        threshold = float(meta.get("recon_threshold", 0.01))
    elif isinstance(thresh_cfg, (int, float)):
        threshold = float(thresh_cfg)
    else:
        threshold = 0.01

    return ae, threshold


def _load_wav_mono_np(path: Path) -> tuple[np.ndarray, int]:
    """Match predictor._load_wav_fast behavior: float32 mono waveform + sr."""
    waveform_np, sr = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = waveform_np.mean(axis=1).astype(np.float32)
    return waveform, int(sr)


@torch.no_grad()
def classify_segment_embedding(
    *,
    classifier: torch.nn.Module,
    autoencoder: torch.nn.Module | None,
    embedding_np: np.ndarray,
    high_threshold: float,
    low_threshold: float,
    device: torch.device,
    recon_threshold: float,
) -> tuple[str, float]:
    """Binary classifier probability → three-class routing label."""
    emb = torch.from_numpy(embedding_np).unsqueeze(0).to(device)

    # Optional AE gating (matches inference/predictor.py behavior)
    if autoencoder is not None:
        reconstructed, _ = autoencoder(emb)
        recon_err = float(
            EmbeddingAutoencoder.compute_reconstruction_error(emb, reconstructed).item()
        )
        if recon_err > recon_threshold:
            # Rejected as OOD → route to Noise
            return "Noise", 0.0

    logits = classifier(emb)
    if logits.ndim > 1:
        logits = logits.squeeze(-1)
    prob = float(torch.sigmoid(logits).item())

    if prob >= high_threshold:
        return "Bird", prob
    if prob <= low_threshold:
        return "Noise", prob
    return "Uncertain", prob


def _materialize_upload_for_preprocessor(
    uploaded_name: str,
    raw_bytes: bytes,
    work_dir: Path,
) -> Path:
    """Write upload to disk; decode non-WAV with torchaudio to WAV for soundfile-based loader.

    Segmentation itself always runs inside ``Preprocessor.process_file`` — this is only I/O adaptation.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_name).suffix.lower() or ".wav"
    src = work_dir / f"upload{suffix}"
    src.write_bytes(raw_bytes)

    if suffix == ".wav":
        return src

    try:
        wav, sr = torchaudio.load(str(src))
    except Exception as e:
        raise RuntimeError(
            f"Could not decode {suffix} (install ffmpeg if needed for MP3): {e}"
        ) from e

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    decoded = work_dir / "upload_decoded_for_preprocess.wav"
    # Preprocessor uses soundfile.read; give it PCM WAV at native sr (Preprocessor resamples).
    x = wav.squeeze(0).numpy().astype(np.float32)
    sf.write(str(decoded), x, int(sr), subtype="PCM_16")
    return decoded


def run_segmentation(cfg: dict, audio_path: Path, out_dir: Path, species: str) -> List[SegmentMeta]:
    """Delegate to existing ``Preprocessor.process_file`` (no duplicate segmentation logic)."""
    preprocessor = Preprocessor(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    return preprocessor.process_file(audio_path, out_dir, species=species)


def _waveform_figure(path: Path, title: str, max_seconds: float = 30.0) -> Optional[plt.Figure]:
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        return None
    # Mono mix if stereo
    if data.ndim == 2 and data.shape[1] > 1:
        x = data.mean(axis=1)
    else:
        x = data.ravel()
    n = min(len(x), int(max_seconds * sr))
    t = np.arange(n) / float(sr)
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.plot(t, x[:n], linewidth=0.5, color="#1f77b4")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.set_xlim(0, t[-1] if n else 1)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="Audio segmentation demo", layout="wide")
    st.title("Pipeline segmentation demo")
    st.caption(
        "Uses `preprocessing.Preprocessor` and `config.yaml` — same code path as the CLI preprocess stage."
    )

    cfg_path = _resolve_config_path()
    if not cfg_path.is_file():
        st.error(f"Config not found: `{cfg_path}`. Set PIPELINE_CONFIG or add config.yaml at project root.")
        st.stop()

    try:
        cfg = load_pipeline_config(str(cfg_path))
    except Exception as e:
        st.error(f"Failed to load config: {e}")
        st.stop()

    mode = cfg.get("pipeline", {}).get("mode", "baseline")
    seg_s = cfg.get("audio", {}).get("segment_duration_s", 3.0)
    st.info(
        f"Config: `{cfg_path.name}` · pipeline.mode=**{mode}** · segment length **{seg_s}s** "
        f"(silent segments may be dropped; V2 may filter/route when not baseline)."
    )

    with st.expander("Sanity checks (what models/data are actually used)", expanded=False):
        st.write(
            {
                "embedding.model_name": cfg.get("embedding", {}).get("model_name"),
                "embedding.embedding_dim": cfg.get("embedding", {}).get("embedding_dim"),
                "training.checkpoint_dir": cfg.get("training", {}).get("checkpoint_dir"),
                "inference.low_threshold": cfg.get("inference", {}).get("low_threshold"),
                "inference.high_threshold": cfg.get("inference", {}).get("high_threshold"),
                "inference.streamlit_use_autoencoder": cfg.get("inference", {}).get(
                    "streamlit_use_autoencoder", False
                ),
                "autoencoder.enabled": cfg.get("autoencoder", {}).get("enabled"),
                "autoencoder.checkpoint_path": cfg.get("autoencoder", {}).get("checkpoint_path"),
                "data.embeddings_dir": cfg.get("data", {}).get("embeddings_dir"),
            }
        )
        try:
            from pathlib import Path as _P

            chk = _P(cfg["training"]["checkpoint_dir"])
            st.write(
                {
                    "best_model.pt exists": (chk / "best_model.pt").exists(),
                    "best_model_meta.json exists": (chk / "best_model_meta.json").exists(),
                    "autoencoder.pt exists": _P(cfg.get("autoencoder", {}).get("checkpoint_path", "")).exists()
                    if cfg.get("autoencoder", {}).get("enabled", False)
                    else False,
                }
            )
        except Exception as e:
            st.warning(f"Could not check checkpoint files: {e}")

    up = st.file_uploader("Upload audio", type=["wav", "mp3"])
    if up is None:
        st.stop()

    raw_bytes = up.getvalue()
    upload_id = f"{up.name}:{len(raw_bytes)}"
    if st.session_state.get("upload_id") != upload_id:
        st.session_state["upload_id"] = upload_id
        st.session_state.pop("segment_metas", None)

    work = Path(st.session_state.get("work_dir", _project_root() / ".streamlit_preprocess_tmp"))
    st.session_state["work_dir"] = str(work)

    try:
        audio_path = _materialize_upload_for_preprocessor(up.name, raw_bytes, work)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.subheader("Original")
    st.audio(raw_bytes, format="audio/wav" if up.name.lower().endswith(".wav") else "audio/mp3")

    wf_fig = _waveform_figure(audio_path, "Waveform (file fed to preprocessor)")
    if wf_fig is not None:
        st.pyplot(wf_fig)
        plt.close(wf_fig)

    if st.button("Run segmentation", type="primary"):
        seg_out = work / "segments_out"
        if seg_out.exists():
            shutil.rmtree(seg_out, ignore_errors=True)

        with st.spinner("Running Preprocessor.process_file …"):
            try:
                metas = run_segmentation(cfg, audio_path, seg_out, species="streamlit_demo")
            except Exception as e:
                st.exception(e)
                st.stop()

        st.session_state["segment_metas"] = metas
        st.session_state.pop("segment_predictions", None)
        st.success(f"Done — {len(metas)} segment(s) written under `{seg_out}`.")

    metas: List[SegmentMeta] = st.session_state.get("segment_metas", [])

    if not metas:
        st.caption("Click **Run segmentation** to populate segment list.")
        st.stop()

    st.subheader("Results")
    st.metric("Total segments (kept)", len(metas))

    # ─────────────────────────────────────────────────────────────
    # DEMO MODE: Preprocessing-only Noise Segregation V2 separation
    # (no MLP training required)
    # ─────────────────────────────────────────────────────────────
    st.subheader("Noise separation (Preprocessing V2)")
    has_v2 = any(m.v2_label is not None for m in metas)
    if not has_v2:
        st.warning(
            "No V2 labels found in segment metadata. "
            "To demo noise separation, set `pipeline.mode: full` (or `filtered`) in `config.yaml` "
            "and re-run segmentation."
        )
    else:
        v2_bird = [m for m in metas if m.v2_label == "bird"]
        v2_noise = [m for m in metas if m.v2_label == "noise"]
        v2_other = [m for m in metas if m.v2_label not in ("bird", "noise")]

        c1, c2, c3 = st.columns(3)
        c1.metric("V2 bird-like", len(v2_bird))
        c2.metric("V2 noise-like", len(v2_noise))
        c3.metric("V2 unknown", len(v2_other))

        t1, t2, t3 = st.tabs(
            [f"Bird-like ({len(v2_bird)})", f"Noise-like ({len(v2_noise)})", f"Other ({len(v2_other)})"]
        )

        def _render_v2(items: List[SegmentMeta], *, expand_first: bool) -> None:
            for i, m in enumerate(sorted(items, key=lambda x: x.segment_index)):
                parts = [
                    f"Segment **{m.segment_index}**",
                    f"{m.start_sec:.2f}–{m.end_sec:.2f}s",
                    f"RMS {m.rms_db:.1f} dB",
                    f"V2: **{m.v2_label or '—'}**",
                ]
                if m.v2_mean_score is not None:
                    parts.append(f"S={m.v2_mean_score:.3f}")
                if m.v2_noise_subframe_votes is not None and m.v2_total_subframes is not None:
                    parts.append(f"votes={m.v2_noise_subframe_votes}/{m.v2_total_subframes}")
                title = " · ".join(parts)

                with st.expander(title, expanded=(expand_first and i == 0)):
                    p = Path(m.output_path)
                    if p.is_file():
                        st.audio(p.read_bytes(), format="audio/wav")
                    else:
                        st.warning(f"Missing file: {p}")
                    st.caption(f"`{m.output_path}`")

        with t1:
            _render_v2(v2_bird, expand_first=True)
        with t2:
            _render_v2(v2_noise, expand_first=True)
        with t3:
            _render_v2(v2_other, expand_first=False)

    st.divider()
    st.subheader("Classifier routing (Embedding → MLP → thresholds)")

    # --- FULL PIPELINE (embedding + trained MLP) ---
    # IMPORTANT: do NOT use BirdNET species names as final output; only the trained binary classifier.
    try:
        classifier, _, binary, optimal_threshold = load_cached_classifier(cfg)
    except Exception as e:
        st.error(str(e))
        st.stop()

    if not binary:
        st.error("This UI expects a binary bird/noise checkpoint (`best_model_meta.json` says binary=false).")
        st.stop()

    device = torch.device("cpu")
    classifier = classifier.to(device)
    encoder = build_cached_encoder(cfg)
    autoencoder, recon_threshold = load_cached_autoencoder(cfg)
    if autoencoder is not None:
        autoencoder = autoencoder.to(device)

    inf_cfg = cfg.get("inference", {})
    low_threshold = float(inf_cfg.get("low_threshold", 0.2))
    high_threshold = float(inf_cfg.get("high_threshold", 0.6))
    # AE before MLP often marks real segments as OOD → all "Noise" @ p=0. Default: MLP only (matches CLI).
    use_ae_in_streamlit = bool(inf_cfg.get("streamlit_use_autoencoder", False))
    ae_for_classify = autoencoder if use_ae_in_streamlit else None

    if use_ae_in_streamlit and ae_for_classify is not None:
        st.caption(
            f"Streamlit: **AE gating on** (recon_threshold={recon_threshold:.6f}), then MLP + thresholds."
        )
    elif autoencoder is not None and not use_ae_in_streamlit:
        st.caption(
            "Streamlit: **AE gating off** (`inference.streamlit_use_autoencoder: false`) — "
            "MLP routing only, same bands as `python run_pipeline.py --stage infer`."
        )
    else:
        st.caption("AE not loaded — MLP routing only.")

    st.caption(
        f"Routing: **Bird** if prob ≥ **{high_threshold:.2f}**, **Noise** if prob ≤ **{low_threshold:.2f}**, "
        f"**Uncertain** between (from `config.yaml` `inference.*`). "
        f"Training F1-optimal threshold (metadata) = **{optimal_threshold:.4f}** — shown for reference only."
    )

    if st.button("Quick white-noise sanity test (embed → MLP → routing)"):
        sr = int(cfg.get("audio", {}).get("sample_rate", 48000))
        dur = float(cfg.get("audio", {}).get("segment_duration_s", 3.0))
        n = int(sr * dur)
        rng = np.random.default_rng(0)
        wn = rng.standard_normal(n).astype(np.float32)
        # scale to a reasonable RMS
        wn = wn / (np.std(wn) + 1e-12) * 0.05
        emb_np = encoder.encode(wn, sr)
        emb_mu, emb_std = float(emb_np.mean()), float(emb_np.std())
        label, prob = classify_segment_embedding(
            classifier=classifier,
            autoencoder=ae_for_classify,
            embedding_np=emb_np,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            device=device,
            recon_threshold=recon_threshold,
        )
        st.write(
            {
                "embedding_mean": emb_mu,
                "embedding_std": emb_std,
                "pred_label": label,
                "prob": prob,
                "note": "If this is Bird, add noise-like training data (e.g. generate_synthetic_noise.py) and retrain.",
            }
        )

    if st.button("Run classification (embedding → MLP → routing)"):
        preds = []
        with st.spinner("Encoding segments + running classifier …"):
            for m in sorted(metas, key=lambda x: x.segment_index):
                wav_path = Path(m.output_path)
                waveform, sr = _load_wav_mono_np(wav_path)
                emb_np = encoder.encode(waveform, sr)

                label, prob = classify_segment_embedding(
                    classifier=classifier,
                    autoencoder=ae_for_classify,
                    embedding_np=emb_np,
                    high_threshold=high_threshold,
                    low_threshold=low_threshold,
                    device=device,
                    recon_threshold=recon_threshold,
                )

                # Mandatory debug output to console
                print(
                    f"[streamlit] file={Path(m.output_path).name} "
                    f"seg={m.segment_index:04d} prob={prob:.4f} label={label}",
                    flush=True,
                )

                preds.append({"segment_index": m.segment_index, "label": label, "prob": prob})

        st.session_state["segment_predictions"] = {p["segment_index"]: p for p in preds}
        st.success("Classification complete. Check console for per-segment debug lines.")

    pred_map = st.session_state.get("segment_predictions", {})
    if not pred_map:
        st.caption("Click **Run classification** to see Bird / Noise / Uncertain routing per segment.")
        st.stop()

    # Note: has_v2 already computed above; reuse it here.

    # Group by classifier routing label
    bird_metas: List[SegmentMeta] = []
    noise_metas: List[SegmentMeta] = []
    uncertain_metas: List[SegmentMeta] = []
    for m in metas:
        p = pred_map.get(m.segment_index, {})
        lbl = p.get("label", "Uncertain")
        if lbl == "Bird":
            bird_metas.append(m)
        elif lbl == "Noise":
            noise_metas.append(m)
        else:
            uncertain_metas.append(m)

    t_bird, t_noise, t_unc = st.tabs(
        [f"Bird ({len(bird_metas)})", f"Noise ({len(noise_metas)})", f"Uncertain ({len(uncertain_metas)})"]
    )

    def _render_segment_list(items: List[SegmentMeta], *, default_expand_first: bool) -> None:
        for i, m in enumerate(sorted(items, key=lambda x: x.segment_index)):
            pred = pred_map.get(m.segment_index, {})
            pred_label = pred.get("label", "Uncertain")
            prob = float(pred.get("prob", 0.0))
            parts = [
                f"Segment **{m.segment_index}**",
                f"{m.start_sec:.2f}–{m.end_sec:.2f}s",
                f"RMS {m.rms_db:.1f} dB",
                f"Pred: **{pred_label}**",
                f"p={prob:.3f}",
            ]
            if has_v2:
                parts.append(f"V2: **{m.v2_label or '—'}**")
                if m.v2_mean_score is not None:
                    parts.append(f"S={m.v2_mean_score:.3f}")
            title = " · ".join(parts)

            with st.expander(title, expanded=(default_expand_first and i == 0)):
                p = Path(m.output_path)
                if p.is_file():
                    st.audio(p.read_bytes(), format="audio/wav")
                else:
                    st.warning(f"Missing file: {p}")
                st.caption(f"`{m.output_path}`")

    with t_bird:
        _render_segment_list(bird_metas, default_expand_first=True)

    with t_noise:
        _render_segment_list(noise_metas, default_expand_first=True)

    with t_unc:
        _render_segment_list(uncertain_metas, default_expand_first=False)


if __name__ == "__main__":
    main()
