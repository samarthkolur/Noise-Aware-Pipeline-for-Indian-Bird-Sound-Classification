"""Full inference pipeline: raw WAV -> SegmentRecord -> routed output
(design.md §6.0 Pipeline State Model, §6.1)."""

from __future__ import annotations

import csv
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from pipeline.audio import is_silent, load_audio, rms_db, segment_audio
from pipeline.bird_guard import apply_bird_guard
from pipeline.config import PipelineConfig
from pipeline.embedding import extract_embedding, l2_normalize
from pipeline.model import BirdAutoencoder, FocalMLP
from pipeline.noise_segregation import classify_segment
from pipeline.router import route_segment


@dataclass
class SegmentRecord:
    path: str
    segment_index: int
    start_sec: float = 0.0
    end_sec: float = 0.0
    rms_db: float = 0.0
    silence_rejected: bool = False
    noise_score: float | None = None
    noise_class: str | None = None
    bird_guard_triggered: bool = False
    rescued: bool = False
    mlp_prob: float | None = None
    mlp_prediction: str | None = None
    ae_mse: float | None = None
    ood_rejected: bool = False
    final_band: str | None = None
    routed_to: str | None = None
    error: str | None = None
    extra: dict = field(default_factory=dict)


class InferencePipeline:
    """Loads trained MLP + AE artifacts and runs the full per-segment pipeline."""

    def __init__(self, config: PipelineConfig, mlp: FocalMLP, ae: BirdAutoencoder, tau_ae: float):
        self.config = config
        self.mlp = mlp.eval()
        self.ae = ae.eval()
        self.tau_ae = tau_ae

    @classmethod
    def from_artifacts(cls, config: PipelineConfig) -> InferencePipeline:
        artifacts_dir = config.resolve_path(config.paths.artifacts_dir)
        mlp = FocalMLP(config.embedding.embedding_dim, config.mlp)
        mlp.load_state_dict(torch.load(artifacts_dir / "mlp_best.pt", map_location="cpu"))
        ae = BirdAutoencoder(config.embedding.embedding_dim, config.autoencoder)
        ae.load_state_dict(torch.load(artifacts_dir / "ae_best.pt", map_location="cpu"))

        import json

        with open(artifacts_dir / "config.json") as f:
            saved = json.load(f)
        return cls(config, mlp, ae, tau_ae=saved["tau_ae"])

    def process_segment(
        self, audio, sr: int, source_path: str, segment_index: int
    ) -> SegmentRecord:
        start_sec = segment_index * self.config.audio.segment_length_s
        record = SegmentRecord(
            path=source_path,
            segment_index=segment_index,
            start_sec=start_sec,
            end_sec=start_sec + len(audio) / sr,
        )
        record.rms_db = rms_db(audio)
        record.silence_rejected = is_silent(audio, self.config.audio.silence_db_threshold)
        if record.silence_rejected:
            return record

        v2_result = classify_segment(audio, sr, self.config.noise_segregation)
        record.noise_score = v2_result.noise_score
        record.noise_class = v2_result.noise_class

        if v2_result.noise_class == "noise":
            guard = apply_bird_guard(audio, self.config.bird_guard, sr)
            record.bird_guard_triggered = guard.triggered
            if guard.triggered:
                record.noise_class = "bird"

        if record.noise_class == "noise":
            record.final_band = "noise"
            record.routed_to = "outputs/noise/"
            return record

        embedding = extract_embedding(audio, sr, self.config.embedding)
        embedding_t = torch.from_numpy(l2_normalize(embedding)).unsqueeze(0)

        with torch.no_grad():
            self.mlp.eval()
            record.mlp_prob = float(self.mlp(embedding_t).item())
            record.ae_mse = float(self.ae.reconstruction_error(embedding_t).item())
        record.mlp_prediction = "bird" if record.mlp_prob >= 0.5 else "noise"
        record.ood_rejected = record.ae_mse > self.tau_ae

        routing = route_segment(record.mlp_prob, record.ood_rejected, self.config.router)
        record.final_band = routing.final_band
        record.routed_to = routing.routed_to
        return record

    def process_file_with_segments(
        self, wav_path: str
    ) -> tuple[list[SegmentRecord], list, int]:
        """Like process_file, but also returns each segment's raw audio + sample
        rate — needed to play back or export a specific segment (e.g. the
        dashboard's noise-analysis / user-correction flow, design.md DD-022).
        """
        audio, sr = load_audio(wav_path, self.config.audio.target_sr)
        segments = segment_audio(audio, sr, self.config.audio.segment_length_s)
        records = [self.process_segment(seg, sr, wav_path, i) for i, seg in enumerate(segments)]
        return records, segments, sr

    def process_file(self, wav_path: str) -> list[SegmentRecord]:
        records, _, _ = self.process_file_with_segments(wav_path)
        return records

    def process_directory(self, input_dir: str) -> list[SegmentRecord]:
        records = []
        for wav_path in sorted(Path(input_dir).rglob("*.wav")):
            records.extend(self.process_file(str(wav_path)))
        return records


def route_output_file(record: SegmentRecord, outputs_dir: Path, audio_path: str) -> None:
    """Physically copy a source WAV into its routed band directory."""
    if record.final_band is None:
        return
    dest_dir = outputs_dir / record.final_band
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_name = f"{Path(audio_path).stem}_seg{record.segment_index}.wav"
    shutil.copy2(audio_path, dest_dir / dest_name)


def write_manifest_results(records: list[SegmentRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0]).keys()) if records else []
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=[k for k in fieldnames if k != "extra"], extrasaction="ignore"
        )
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))
