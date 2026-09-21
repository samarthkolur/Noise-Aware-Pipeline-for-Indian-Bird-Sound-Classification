"""Streamlit dashboard for the Noise-Aware Pipeline for Indian Bird Sound
Classification (design.md DD-021).

Covers every delivery phase (design.md §10) from one UI: run each pipeline
script, inspect its output, run inference on a WAV file, browse the
benchmark/analytics results, run the test suite, and read the living
engineering doc — without needing the CLI for day-to-day exploration.

Run with:
    streamlit run app/dashboard.py

This is a development/analytics dashboard, not a production service — it
shells out to the same scripts/*.py entry points design.md already treats as
the source of truth (DD-021), so behavior here always matches the CLI.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import PROJECT_ROOT, load_config  # noqa: E402
from pipeline.inference import InferencePipeline  # noqa: E402
from pipeline.mining import export_user_correction  # noqa: E402

st.set_page_config(page_title="Noise-Aware BirdNET Dashboard", layout="wide")

PYTHON = sys.executable


# ─── Shared helpers ──────────────────────────────────────────────────────────

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


@st.cache_data(ttl=10)
def get_config():
    load_config.cache_clear()
    return load_config()


def run_script(args: list[str], label: str) -> tuple[int, str]:
    """Run a scripts/*.py entry point, streaming its output into the UI."""
    cfg = get_config()
    cmd = [PYTHON, *args]
    env = {**os.environ, "FORCE_COLOR": "0", "NO_COLOR": "1"}
    st.caption(f"Running: `{' '.join(cmd)}`")
    placeholder = st.empty()
    lines: list[str] = []
    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(_ANSI_RE.sub("", line.rstrip("\n")))
        placeholder.code("\n".join(lines[-200:]) or "(no output yet)")
    process.wait()
    del cfg
    if process.returncode == 0:
        st.success(f"{label} finished (exit code 0).")
    else:
        st.error(f"{label} failed (exit code {process.returncode}).")
    return process.returncode, "\n".join(lines)


BAND_COLORS = {
    "bird": "#4CAF50",
    "uncertain": "#FFC107",
    "noise": "#F44336",
    None: "#9E9E9E",  # silence-rejected / not routed
}


def audio_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """Encode a raw audio array as in-memory WAV bytes, for st.audio playback
    of a single segment without writing a scratch file to disk."""
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def render_band_timeline(df: pd.DataFrame) -> None:
    """Color-coded horizontal strip of the whole clip: green=bird,
    amber=uncertain, red=noise, gray=silence-rejected — the 'range of
    seconds the noise exists in' view (design.md DD-022)."""
    total = float(df["end_sec"].max()) if len(df) else 1.0
    segments_html = []
    for _, row in df.iterrows():
        width_pct = max((row["end_sec"] - row["start_sec"]) / total * 100, 0.5)
        color = BAND_COLORS.get(
            row["final_band"] if pd.notna(row["final_band"]) else None, "#9E9E9E"
        )
        band_label = row["final_band"] or "silence"
        title = (
            f"segment {int(row['segment_index'])}: "
            f"{row['start_sec']:.1f}s-{row['end_sec']:.1f}s ({band_label})"
        )
        segments_html.append(
            f'<div title="{title}" style="width:{width_pct}%;background:{color};'
            f"height:36px;display:inline-block;border-right:1px solid #fff;"
            f'box-sizing:border-box;"></div>'
        )
    strip_style = "width:100%;white-space:nowrap;line-height:0;border-radius:4px;overflow:hidden;"
    st.markdown(
        f'<div style="{strip_style}">{"".join(segments_html)}</div>',
        unsafe_allow_html=True,
    )
    legend = "  ".join(
        f'<span style="color:{color}">●</span> {label or "silence"}'
        for label, color in BAND_COLORS.items()
        if label is not None
    )
    st.markdown(f"<small>{legend}</small>", unsafe_allow_html=True)


def read_manifest() -> pd.DataFrame | None:
    cfg = get_config()
    path = cfg.resolve_path(cfg.paths.manifest_path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def extract_design_sections(design_md_path: Path) -> dict[str, str]:
    text = design_md_path.read_text()
    parts = re.split(r"\n(?=## \d+\. )", text)
    sections = {}
    for part in parts:
        m = re.match(r"## (\d+\. .+)", part)
        if m:
            sections[m.group(1)] = part
    return sections


# ─── Sidebar navigation ──────────────────────────────────────────────────────

SECTIONS = [
    "Overview",
    "1. Data Pipeline",
    "2. Embeddings",
    "3. Training",
    "4. Inference",
    "5. Benchmark & Analytics",
    "6. Testing",
    "7. Design Doc",
]

st.sidebar.title("Noise-Aware BirdNET")
section = st.sidebar.radio("Section", SECTIONS)
st.sidebar.markdown("---")
st.sidebar.caption(f"Project root:\n`{PROJECT_ROOT}`")

cfg = get_config()

# ─── Overview ────────────────────────────────────────────────────────────────

if section == "Overview":
    st.title("Noise-Aware Pipeline for Indian Bird Sound Classification")
    st.markdown(
        "BirdNET embeddings → Focal-loss MLP → Autoencoder OOD gate → "
        "three-band router. See `design.md` for the full architecture."
    )

    col1, col2, col3, col4 = st.columns(4)
    manifest_df = read_manifest()
    if manifest_df is not None:
        col1.metric("Manifest segments", len(manifest_df))
        col2.metric("Bird segments", int((manifest_df["class"] == "bird").sum()))
        col3.metric("Noise segments", int((manifest_df["class"] == "noise").sum()))
    else:
        col1.metric("Manifest segments", "—")
        col2.metric("Bird segments", "—")
        col3.metric("Noise segments", "—")

    cache_path = cfg.resolve_path(cfg.paths.embeddings_cache_path)
    if cache_path.exists():
        with h5py.File(cache_path, "r") as f:
            col4.metric("Cached embeddings", len(f.keys()))
    else:
        col4.metric("Cached embeddings", "—")

    st.markdown("### Pipeline status")
    status_rows = [
        ("Manifest (`data/manifest.csv`)", cfg.resolve_path(cfg.paths.manifest_path).exists()),
        ("Embedding cache (`data/embeddings.h5`)", cache_path.exists()),
        (
            "Trained MLP (`artifacts/mlp_best.pt`)",
            (cfg.resolve_path(cfg.paths.artifacts_dir) / "mlp_best.pt").exists(),
        ),
        (
            "Trained AE (`artifacts/ae_best.pt`)",
            (cfg.resolve_path(cfg.paths.artifacts_dir) / "ae_best.pt").exists(),
        ),
        (
            "Benchmark report (`outputs/evaluation_report.json`)",
            (cfg.resolve_path(cfg.paths.outputs_dir) / "evaluation_report.json").exists(),
        ),
    ]
    status_df = pd.DataFrame(status_rows, columns=["Artifact", "Present"])
    st.dataframe(status_df, hide_index=True, use_container_width=True)

    artifacts_config = cfg.resolve_path(cfg.paths.artifacts_dir) / "config.json"
    if artifacts_config.exists():
        with open(artifacts_config) as f:
            saved = json.load(f)
        st.markdown("### Latest training run")
        st.json(saved)

# ─── 1. Data Pipeline ────────────────────────────────────────────────────────

elif section == "1. Data Pipeline":
    st.title("Phase 2 — Data Pipeline")
    st.markdown(
        "Segments raw audio from `data/iBC53/<species>/*.wav` "
        "(non-overlapping, per design.md §6.1) and builds the manifest + "
        "stratified split (DD-020)."
    )

    with st.form("prepare_data_form"):
        c1, c2 = st.columns(2)
        max_segments = c1.number_input(
            "Max segments per species (-1 = full corpus)", value=20, min_value=-1
        )
        force_resplit = c2.checkbox("Force re-split (overwrite existing manifest)", value=False)
        allow_synthetic = c1.checkbox("Include synthetic noise class (DD-010, opt-in)", value=True)
        n_synthetic = c2.number_input("Synthetic noise segments", value=200, min_value=0)
        submitted = st.form_submit_button("Run prepare_data.py")

    if submitted:
        args = [
            "scripts/prepare_data.py",
            "--max-segments-per-species",
            str(int(max_segments)),
        ]
        if force_resplit:
            args.append("--force-resplit")
        if allow_synthetic:
            args += ["--allow-synthetic", "--n-synthetic", str(int(n_synthetic))]
        run_script(args, "prepare_data.py")
        st.cache_data.clear()

    st.markdown("### Current manifest")
    manifest_df = read_manifest()
    if manifest_df is None:
        st.info("No manifest yet — run prepare_data.py above.")
    else:
        col1, col2 = st.columns(2)
        col1.markdown("**Class distribution**")
        col1.bar_chart(manifest_df["class"].value_counts())
        col2.markdown("**Split distribution**")
        col2.bar_chart(manifest_df["split"].value_counts())
        st.markdown("**Sample rows**")
        st.dataframe(manifest_df.sample(min(20, len(manifest_df))), use_container_width=True)

# ─── 2. Embeddings ───────────────────────────────────────────────────────────

elif section == "2. Embeddings":
    st.title("Phase 4 — BirdNET Embedding Extraction")
    st.markdown(
        "Runs real BirdNET V2.4 inference (`tensorflow` backend, DD-009) on "
        "every manifest segment and caches 1024-D embeddings to HDF5 "
        "(idempotent, DD-003)."
    )

    with st.form("extract_embeddings_form"):
        c1, c2 = st.columns(2)
        batch_size = c1.number_input("Batch size", value=16, min_value=1)
        limit = c2.number_input("Limit segments (0 = all)", value=0, min_value=0)
        submitted = st.form_submit_button("Run extract_embeddings.py")

    if submitted:
        args = ["scripts/extract_embeddings.py", "--batch-size", str(int(batch_size))]
        if limit > 0:
            args += ["--limit", str(int(limit))]
        run_script(args, "extract_embeddings.py")
        st.cache_data.clear()

    st.markdown("### Cache status")
    cache_path = cfg.resolve_path(cfg.paths.embeddings_cache_path)
    if cache_path.exists():
        with h5py.File(cache_path, "r") as f:
            keys = list(f.keys())
            st.metric("Cached embeddings", len(keys))
            if keys:
                sample = f[keys[0]][()]
                st.caption(f"Sample embedding shape: {sample.shape}, dtype: {sample.dtype}")
    else:
        st.info("No embedding cache yet — run extraction above.")

# ─── 3. Training ─────────────────────────────────────────────────────────────

elif section == "3. Training":
    st.title("Phase 5 — MLP + Autoencoder Training")
    st.markdown(
        "Trains the Focal-loss MLP classifier and the bird-only Autoencoder "
        "OOD gate on cached embeddings; derives τ_AE from the validation "
        "split (DD-005) and saves artifacts."
    )

    if st.button("Run train.py"):
        _, output = run_script(["scripts/train.py"], "train.py")

        mlp_rows = re.findall(r"epoch (\d+)/\d+ train_loss=([\d.]+) val_f1=([\d.]+)", output)
        ae_rows = re.findall(r"AE epoch (\d+)/\d+ train_loss=([\d.]+) val_mse=([\d.]+)", output)

        if mlp_rows:
            mlp_df = pd.DataFrame(mlp_rows, columns=["epoch", "train_loss", "val_f1"]).astype(float)
            st.markdown("**FocalMLP training curve**")
            st.line_chart(mlp_df.set_index("epoch")[["train_loss", "val_f1"]])
        if ae_rows:
            ae_df = pd.DataFrame(ae_rows, columns=["epoch", "train_loss", "val_mse"]).astype(float)
            st.markdown("**Autoencoder training curve**")
            st.line_chart(ae_df.set_index("epoch")[["train_loss", "val_mse"]])

    st.markdown("### Saved artifacts")
    artifacts_config = cfg.resolve_path(cfg.paths.artifacts_dir) / "config.json"
    if artifacts_config.exists():
        with open(artifacts_config) as f:
            saved = json.load(f)
        c1, c2, c3 = st.columns(3)
        c1.metric("τ_AE", f"{saved.get('tau_ae', float('nan')):.6f}")
        c2.metric("Best val F1 (MLP)", f"{saved.get('mlp_best_val_f1', float('nan')):.4f}")
        c3.metric("Best val MSE (AE)", f"{saved.get('ae_best_val_mse', float('nan')):.6f}")
        with st.expander("Full training config"):
            st.json(saved)
    else:
        st.info("No trained artifacts yet — run training above.")

# ─── 4. Inference ────────────────────────────────────────────────────────────

elif section == "4. Inference":
    st.title("Phase 6 — Inference Pipeline")
    st.markdown(
        "Runs a WAV file through the full pipeline: RMS gate → Noise "
        "Segregation V2 → Bird Guard → BirdNET embedding → MLP + AE gate → "
        "three-band router (design.md §6.0)."
    )

    artifacts_dir = cfg.resolve_path(cfg.paths.artifacts_dir)
    if not (artifacts_dir / "mlp_best.pt").exists():
        st.warning("No trained model found. Run Phase 5 (Training) first.")
    else:

        @st.cache_resource
        def load_pipeline():
            return InferencePipeline.from_artifacts(get_config())

        source = st.radio("Audio source", ["Upload a WAV file", "Pick from data/segments/"])
        wav_path: Path | None = None
        uploaded_bytes: bytes | None = None

        if source == "Upload a WAV file":
            uploaded = st.file_uploader("WAV file", type=["wav"])
            if uploaded is not None:
                uploaded_bytes = uploaded.getvalue()
                tmp_path = Path("/tmp") / uploaded.name
                tmp_path.write_bytes(uploaded_bytes)
                wav_path = tmp_path
        else:
            segments_dir = cfg.resolve_path(cfg.paths.segments_dir)
            if segments_dir.exists():
                species_dirs = sorted(p.name for p in segments_dir.iterdir() if p.is_dir())
                species = st.selectbox("Species", species_dirs) if species_dirs else None
                if species:
                    files = sorted((segments_dir / species).glob("*.wav"))
                    file_choice = st.selectbox("File", [f.name for f in files])
                    if file_choice:
                        wav_path = segments_dir / species / file_choice
                        uploaded_bytes = wav_path.read_bytes()
            else:
                st.info("data/segments/ doesn't exist yet — run Phase 2 (Data Pipeline) first.")

        if wav_path is not None and st.button("Run inference"):
            pipeline = load_pipeline()
            with st.spinner("Running full pipeline (real BirdNET inference)..."):
                records, segments, sr = pipeline.process_file_with_segments(str(wav_path))
            # Persist across reruns (e.g. clicking a "Flag" button below)
            # so we don't re-run real BirdNET inference on every click.
            st.session_state["inference_wav_path"] = str(wav_path)
            st.session_state["inference_records"] = records
            st.session_state["inference_segments"] = segments
            st.session_state["inference_sr"] = sr
            st.session_state["flagged_segments"] = set()

        if st.session_state.get("inference_records") and st.session_state.get(
            "inference_wav_path"
        ) == str(wav_path):
            records = st.session_state["inference_records"]
            segments = st.session_state["inference_segments"]
            sr = st.session_state["inference_sr"]

            if uploaded_bytes:
                st.audio(uploaded_bytes)

            rows = [asdict(r) for r in records]
            df = pd.DataFrame(rows).drop(columns=["extra"], errors="ignore")
            st.markdown(f"**{len(records)} segment(s) processed**")

            st.markdown("### Timeline")
            render_band_timeline(df)

            st.markdown("### Full results")
            st.dataframe(df, use_container_width=True)
            st.bar_chart(df["final_band"].value_counts(dropna=False))

            # ── Noise analysis: exact time ranges flagged as noise, with
            # per-segment playback and a way to flag a mislabeled one back
            # into the training loop (design.md DD-022). ──
            st.markdown("### Noise analysis")
            noise_rows = df[df["final_band"] == "noise"]
            if noise_rows.empty:
                st.success("No segments were routed to the noise band in this clip.")
            else:
                st.markdown(
                    f"**{len(noise_rows)} of {len(records)} segment(s)** were classified as "
                    "noise. Listen to each below — if one is actually a bird call, flag it "
                    "so it can be used as a training correction."
                )
                for idx in noise_rows.index:
                    record = records[idx]
                    seg_audio = segments[idx]
                    flag_key = (wav_path_str := str(wav_path), record.segment_index, "bird")

                    with st.container(border=True):
                        c1, c2 = st.columns([2, 1])
                        with c1:
                            st.markdown(
                                f"**Segment {record.segment_index}** — "
                                f"`{record.start_sec:.1f}s – {record.end_sec:.1f}s`"
                            )
                            st.audio(audio_to_wav_bytes(seg_audio, sr))
                        with c2:
                            st.caption(
                                f"noise_score: {record.noise_score:.3f}"
                                if record.noise_score is not None
                                else "noise_score: —"
                            )
                            st.caption(f"bird_guard_triggered: {record.bird_guard_triggered}")
                            if record.mlp_prob is not None:
                                st.caption(f"mlp_prob: {record.mlp_prob:.3f}")
                            if record.ae_mse is not None:
                                st.caption(f"ae_mse: {record.ae_mse:.6f}")

                        if flag_key in st.session_state["flagged_segments"]:
                            st.success("Flagged as bird — saved for retraining.")
                        elif st.button(
                            "🚩 Flag as bird (mislabeled)",
                            key=f"flag_{record.segment_index}_{wav_path_str}",
                        ):
                            review_dir = cfg.resolve_path(cfg.paths.outputs_dir) / "review"
                            export_user_correction(
                                audio=seg_audio,
                                sr=sr,
                                source_path=wav_path_str,
                                segment_index=record.segment_index,
                                start_sec=record.start_sec,
                                end_sec=record.end_sec,
                                pipeline_band="noise",
                                corrected_label="bird",
                                review_dir=review_dir,
                            )
                            st.session_state["flagged_segments"].add(flag_key)
                            st.rerun()

            with st.expander("Also review bird-band segments (flag as noise instead)"):
                bird_rows = df[df["final_band"] == "bird"]
                if bird_rows.empty:
                    st.caption("No bird-band segments in this clip.")
                for idx in bird_rows.index:
                    record = records[idx]
                    seg_audio = segments[idx]
                    flag_key = (str(wav_path), record.segment_index, "noise")
                    c1, c2 = st.columns([3, 1])
                    time_range = f"{record.start_sec:.1f}s–{record.end_sec:.1f}s"
                    c1.markdown(f"Segment {record.segment_index} — {time_range}")
                    c1.audio(audio_to_wav_bytes(seg_audio, sr))
                    if flag_key in st.session_state["flagged_segments"]:
                        c2.success("Flagged")
                    elif c2.button(
                        "🚩 Flag as noise", key=f"flag_noise_{record.segment_index}_{wav_path}"
                    ):
                        review_dir = cfg.resolve_path(cfg.paths.outputs_dir) / "review"
                        export_user_correction(
                            audio=seg_audio,
                            sr=sr,
                            source_path=str(wav_path),
                            segment_index=record.segment_index,
                            start_sec=record.start_sec,
                            end_sec=record.end_sec,
                            pipeline_band="bird",
                            corrected_label="noise",
                            review_dir=review_dir,
                        )
                        st.session_state["flagged_segments"].add(flag_key)
                        st.rerun()

# ─── 5. Benchmark & Analytics ────────────────────────────────────────────────

elif section == "5. Benchmark & Analytics":
    st.title("Phase 7 — Benchmark & Analytics")
    st.markdown(
        "Three-way comparison (BirdNET Baseline / MLP Only / MLP+AE Gate) "
        "on the held-out test split, with statistical validation "
        "(design.md §9, DD-012). Read-only over data + artifacts — "
        "running it twice produces identical results."
    )

    if st.button("Run benchmark.py"):
        run_script(["scripts/benchmark.py"], "benchmark.py")
        st.cache_data.clear()

    report_path = cfg.resolve_path(cfg.paths.outputs_dir) / "evaluation_report.json"
    if not report_path.exists():
        st.info("No evaluation report yet — run the benchmark above.")
    else:
        with open(report_path) as f:
            report = json.load(f)

        st.warning(report.get("scope_note", ""))

        st.markdown("### Three-way comparison")
        systems = ["birdnet_baseline", "mlp_only", "mlp_ae_gate"]
        labels = ["BirdNET Baseline", "MLP Only", "MLP + AE Gate"]
        metrics = ["accuracy", "precision", "recall", "f1", "fpr", "fnr", "roc_auc", "pr_auc"]
        comparison = pd.DataFrame(
            {
                label: [report[sys_key].get(m) for m in metrics]
                for label, sys_key in zip(labels, systems, strict=True)
            },
            index=metrics,
        )
        st.dataframe(comparison.style.format("{:.4f}"), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Statistical significance")
            st.write("**Baseline → MLP**")
            st.json(report.get("significance_baseline_to_mlp", {}))
            st.write("**MLP → MLP+AE Gate**")
            st.json(report.get("significance_mlp_to_gated", {}))
        with col2:
            st.markdown("### Three-band router")
            router = report.get("three_band_router", {})
            st.json(router)
            if "band_counts" in router:
                st.bar_chart(pd.Series(router["band_counts"]))

        tau_ae = report.get("tau_ae", float("nan"))
        n_test = report.get("n_test", "—")
        st.markdown(f"τ_AE = **{tau_ae:.6f}**  |  n_test = **{n_test}**")

        plots_dir = cfg.resolve_path(cfg.paths.outputs_dir) / "plots"
        if plots_dir.exists():
            st.markdown("### Plots")
            images = sorted(plots_dir.glob("*.png"))
            cols = st.columns(3)
            for i, img_path in enumerate(images):
                cols[i % 3].image(str(img_path), caption=img_path.stem, use_container_width=True)

# ─── 6. Testing ──────────────────────────────────────────────────────────────

elif section == "6. Testing":
    st.title("Test Suite")
    st.markdown("Runs `pytest` against the `tests/` suite (design.md §14).")

    include_slow = st.checkbox(
        "Include slow tests (real BirdNET integration — several minutes)", value=False
    )

    if st.button("Run pytest"):
        args = ["-m", "pytest", "-q", "--color=no"]
        if not include_slow:
            args += ["-m", "not slow"]
        returncode, output = run_script(args, "pytest")

        summary_match = re.search(r"^(\d+ (?:passed|failed|error).*)$", output, re.MULTILINE)
        if summary_match:
            if returncode == 0:
                st.success(summary_match.group(1))
            else:
                st.error(summary_match.group(1))

        coverage_match = re.search(r"TOTAL\s+(\d+)\s+(\d+)\s+(\d+)%", output)
        if coverage_match:
            stmts, miss, pct = coverage_match.groups()
            c1, c2, c3 = st.columns(3)
            c1.metric("Statements", stmts)
            c2.metric("Missed", miss)
            c3.metric("Coverage", f"{pct}%")

# ─── 7. Design Doc ───────────────────────────────────────────────────────────

elif section == "7. Design Doc":
    st.title("design.md")
    st.caption(
        "The canonical engineering document (per CLAUDE.md). This view is "
        "read-only — edit design.md directly to change it."
    )

    design_path = PROJECT_ROOT / "design.md"
    sections = extract_design_sections(design_path)
    if not sections:
        st.error("Could not parse design.md sections.")
    else:
        default_idx = 0
        for i, title in enumerate(sections):
            if "Known Issues" in title or "Current Phase" in title:
                default_idx = i
                break
        choice = st.selectbox("Jump to section", list(sections.keys()), index=default_idx)
        st.markdown(sections[choice])

        with st.expander("View full document"):
            st.markdown(design_path.read_text())
