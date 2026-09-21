import csv

import numpy as np

from pipeline.mining import export_hard_examples, export_user_correction, find_hard_examples


def test_likely_fp_is_noise_labeled_with_high_prob():
    result = find_hard_examples(
        segment_ids=["a"], true_labels=["noise"], mlp_probs=[0.9], ood_rejected=[False]
    )
    assert result.likely_fp == ["a"]
    assert result.likely_fn == []


def test_likely_fn_is_bird_labeled_with_low_prob():
    result = find_hard_examples(
        segment_ids=["a"], true_labels=["bird"], mlp_probs=[0.1], ood_rejected=[False]
    )
    assert result.likely_fn == ["a"]
    assert result.likely_fp == []


def test_likely_fn_is_bird_labeled_and_ood_rejected():
    result = find_hard_examples(
        segment_ids=["a"], true_labels=["bird"], mlp_probs=[0.9], ood_rejected=[True]
    )
    assert result.likely_fn == ["a"]


def test_correctly_classified_segments_are_not_flagged():
    result = find_hard_examples(
        segment_ids=["bird_ok", "noise_ok"],
        true_labels=["bird", "noise"],
        mlp_probs=[0.9, 0.1],
        ood_rejected=[False, False],
    )
    assert result.likely_fp == []
    assert result.likely_fn == []


def test_export_hard_examples_writes_manifest_and_copies_files(tmp_path):
    src_fp = tmp_path / "src_fp.wav"
    src_fn = tmp_path / "src_fn.wav"
    src_fp.write_bytes(b"RIFF....FAKEDATA")
    src_fn.write_bytes(b"RIFF....FAKEDATA")

    result = find_hard_examples(
        segment_ids=["fp1", "fn1"],
        true_labels=["noise", "bird"],
        mlp_probs=[0.9, 0.1],
        ood_rejected=[False, False],
    )
    review_dir = tmp_path / "review"
    export_hard_examples(result, {"fp1": str(src_fp), "fn1": str(src_fn)}, review_dir)

    assert (review_dir / "likely_fp" / "src_fp.wav").exists()
    assert (review_dir / "likely_fn" / "src_fn.wav").exists()
    assert (review_dir / "hard_examples_manifest.csv").exists()

    manifest_text = (review_dir / "hard_examples_manifest.csv").read_text()
    assert "fp1,likely_fp" in manifest_text
    assert "fn1,likely_fn" in manifest_text


def test_export_user_correction_writes_audio_and_csv_row(tmp_path):
    sr = 48000
    audio = np.zeros(sr * 3, dtype=np.float32)
    review_dir = tmp_path / "review"

    seg_path = export_user_correction(
        audio=audio,
        sr=sr,
        source_path="field_recording.wav",
        segment_index=2,
        start_sec=6.0,
        end_sec=9.0,
        pipeline_band="noise",
        corrected_label="bird",
        review_dir=review_dir,
    )

    assert seg_path.exists()
    assert seg_path.parent == review_dir / "user_corrections"

    csv_path = review_dir / "user_corrections.csv"
    assert csv_path.exists()
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["source_path"] == "field_recording.wav"
    assert rows[0]["segment_index"] == "2"
    assert rows[0]["pipeline_band"] == "noise"
    assert rows[0]["corrected_label"] == "bird"
    assert rows[0]["start_sec"] == "6.000"


def test_export_user_correction_appends_without_duplicating_header(tmp_path):
    sr = 48000
    audio = np.zeros(sr * 3, dtype=np.float32)
    review_dir = tmp_path / "review"

    for i in range(3):
        export_user_correction(
            audio=audio,
            sr=sr,
            source_path=f"file_{i}.wav",
            segment_index=0,
            start_sec=0.0,
            end_sec=3.0,
            pipeline_band="noise",
            corrected_label="bird",
            review_dir=review_dir,
        )

    csv_path = review_dir / "user_corrections.csv"
    lines = csv_path.read_text().splitlines()
    assert lines[0].startswith("timestamp,")
    assert len(lines) == 4  # header + 3 rows
