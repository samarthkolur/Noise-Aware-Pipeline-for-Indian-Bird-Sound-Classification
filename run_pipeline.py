#!/usr/bin/env python3
"""
Noise-Aware Bird Segregation Pipeline V3 — Main Orchestrator

Data-centric pipeline with Hard-Negative Mining, Active Learning,
and Multi-Stage Filtering for false-positive suppression.

Stages:
    1. Segmentation       →  3-second windows at 48 kHz
    2. Embedding          →  BirdNET 1024-d features + confidence scores
    3. Dataset Curation   →  Hard-negative mining with spectral filters
    4. Classification     →  Random Forest (primary) / MLP (optional)
    5. OOD Filtering      →  Mahalanobis + Isolation Forest
    6. Active Learning    →  Uncertainty sampling + expert feedback (optional)
    7. Post-Processing    →  Temporal smoothing + spectral check
    8. Ensemble Decision  →  Weighted voting → final labels

Usage:
    python run_pipeline.py --stage all
    python run_pipeline.py --stage segment
    python run_pipeline.py --stage embed --max-files 50
    python run_pipeline.py --stage curate
    python run_pipeline.py --stage train
    python run_pipeline.py --stage ood
    python run_pipeline.py --stage active
    python run_pipeline.py --stage postprocess
    python run_pipeline.py --stage ensemble
    python run_pipeline.py --stage evaluate
    python run_pipeline.py --stage all --skip-active-learning
"""

import argparse
import os
import sys
import json
import time
import numpy as np

import config

# ─── Ensure project root is on sys.path ─────────────────────────────────────
sys.path.insert(0, config.PROJECT_ROOT)


# ═════════════════════════════════════════════════════════════════════════════
#  Stage Runners
# ═════════════════════════════════════════════════════════════════════════════

def run_stage1_segmentation(args):
    """Stage 1: Segment raw audio into 3-second windows."""
    from pipeline.stage1_segmentation import segment_directory

    print("\n" + "=" * 70)
    print("  STAGE 1: Audio Segmentation & Standardization")
    print("=" * 70)

    results = segment_directory(
        input_dir=config.RAW_DATA_DIR,
        output_dir=config.SEGMENTED_DIR,
    )
    return results


def run_stage2_embeddings(args):
    """Stage 2: Extract BirdNET embeddings + confidence scores."""
    from pipeline.stage2_embeddings import batch_extract_from_directory

    print("\n" + "=" * 70)
    print("  STAGE 2: BirdNET Embedding Extraction")
    print("=" * 70)

    results = batch_extract_from_directory(
        segment_dir=config.SEGMENTED_DIR,
        output_dir=config.EMBEDDINGS_DIR,
        max_files=args.max_files,
    )
    return results


def run_stage3_curation(args):
    """Stage 3: Build hard-negative dataset with pseudo-labels + spectral filters."""
    from pipeline.stage3_hard_negative_dataset import curate_hard_negative_dataset

    result = curate_hard_negative_dataset()
    return result


def run_stage4_train(args):
    """Stage 4: Train the binary classifier (RF or MLP)."""
    from pipeline.stage4_binary_classifier import create_classifier

    print("\n" + "=" * 70)
    print("  STAGE 4: Binary Classifier Training")
    print("=" * 70)

    X_train = np.load(os.path.join(config.EMBEDDINGS_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(config.EMBEDDINGS_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(config.EMBEDDINGS_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(config.EMBEDDINGS_DIR, "y_val.npy"))

    print(f"  Classifier type: {config.CLASSIFIER_TYPE}")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)}")

    classifier = create_classifier(input_dim=X_train.shape[1])
    metrics = classifier.train_model(X_train, y_train, X_val, y_val)
    classifier.save()

    # Report feature importance (RF only)
    if hasattr(classifier, "feature_importance"):
        top_features = classifier.feature_importance(top_k=10)
        if top_features:
            print("\n  Top-10 embedding dimensions by importance:")
            for dim, score in top_features:
                print(f"    dim {dim:4d}: {score:.4f}")

    return metrics


def run_stage5_ood(args):
    """Stage 5: Train OOD detectors on bird embeddings."""
    from pipeline.stage5_ood_filter import create_ood_detectors

    print("\n" + "=" * 70)
    print("  STAGE 5: Out-of-Distribution Detection")
    print("=" * 70)

    X_train = np.load(os.path.join(config.EMBEDDINGS_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(config.EMBEDDINGS_DIR, "y_train.npy"))

    bird_mask = y_train == config.BIRD_LABEL
    bird_embeddings = X_train[bird_mask]
    print(f"  Training OOD on {len(bird_embeddings)} bird embeddings")

    ood_ensemble = create_ood_detectors()
    ood_ensemble.fit(bird_embeddings)
    ood_ensemble.save()

    # Quick validation
    X_val = np.load(os.path.join(config.EMBEDDINGS_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(config.EMBEDDINGS_DIR, "y_val.npy"))
    ood_preds = ood_ensemble.predict(X_val)

    from sklearn.metrics import accuracy_score
    ood_acc = accuracy_score(y_val, ood_preds)
    print(f"  OOD accuracy on val set: {ood_acc:.4f}")

    return {"ood_val_accuracy": float(ood_acc)}


def run_stage6_active_learning(args):
    """Stage 6: Run one round of active learning (uncertainty sampling)."""
    from pipeline.stage6_active_learning import ActiveLearningLoop
    from pipeline.stage4_binary_classifier import create_classifier

    print("\n" + "=" * 70)
    print("  STAGE 6: Active Learning / Expert-in-the-Loop")
    print("=" * 70)

    X_train = np.load(os.path.join(config.EMBEDDINGS_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(config.EMBEDDINGS_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(config.EMBEDDINGS_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(config.EMBEDDINGS_DIR, "y_val.npy"))

    # Use uncertain (unlabeled) segments as pool
    all_labels = np.load(os.path.join(config.EMBEDDINGS_DIR, "binary_labels.npy"))
    all_embeddings = np.load(os.path.join(config.EMBEDDINGS_DIR, "embeddings.npy"))
    all_paths = np.load(os.path.join(config.EMBEDDINGS_DIR, "paths.npy"), allow_pickle=True)

    uncertain_mask = all_labels == -1
    X_pool = all_embeddings[uncertain_mask]
    pool_paths = all_paths[uncertain_mask]

    if len(X_pool) == 0:
        print("  No uncertain samples in pool — skipping active learning")
        return {}

    print(f"  Pool size: {len(X_pool)} uncertain segments")

    # Determine round number from existing progress
    progress = ActiveLearningLoop.load_progress()
    round_num = len(progress) + 1

    al_loop = ActiveLearningLoop()
    classifier = create_classifier(input_dim=X_train.shape[1])

    result = al_loop.run_round(
        classifier, X_train, y_train, X_val, y_val,
        X_pool, pool_paths, round_num
    )
    al_loop.save_progress()

    if result.get("csv_path"):
        print(f"\n  ╔════════════════════════════════════════════════════════╗")
        print(f"  ║  EXPERT REVIEW REQUIRED                                ║")
        print(f"  ║  Review: {result['csv_path']}")
        print(f"  ║  Fill the 'expert_label' column (bird / noise)         ║")
        print(f"  ║  Then re-run: python run_pipeline.py --stage active    ║")
        print(f"  ╚════════════════════════════════════════════════════════╝")

    return result


def run_stage78_inference(args):
    """Stages 7-8: Post-processing + ensemble decision."""
    from pipeline.stage4_binary_classifier import BirdNoiseRF, BirdNoiseMLP
    from pipeline.stage5_ood_filter import EnsembleOOD
    from pipeline.stage7_postprocessing import postprocess_predictions
    from pipeline.stage8_ensemble import WeightedEnsembleDecider, generate_clean_dataset

    print("\n" + "=" * 70)
    print("  STAGES 7-8: Post-Processing + Ensemble Decision")
    print("=" * 70)

    # Load data
    X_val = np.load(os.path.join(config.EMBEDDINGS_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(config.EMBEDDINGS_DIR, "y_val.npy"))
    file_paths = np.load(os.path.join(config.EMBEDDINGS_DIR, "paths.npy"), allow_pickle=True)

    idx_val_path = os.path.join(config.EMBEDDINGS_DIR, "idx_val.npy")
    idx_val = np.load(idx_val_path) if os.path.exists(idx_val_path) else None

    # Load BirdNET confidences for val set
    all_confs = np.load(os.path.join(config.EMBEDDINGS_DIR, "birdnet_confidences.npy"))
    val_confs = all_confs[idx_val] if idx_val is not None else all_confs[:len(X_val)]

    # Load classifier
    if config.CLASSIFIER_TYPE == "rf":
        classifier = BirdNoiseRF.load()
    else:
        classifier = BirdNoiseMLP.load()

    # Stage 4: Classifier predictions
    clf_probs = classifier.predict(X_val)
    clf_labels = (clf_probs >= 0.5).astype(int)

    # Stage 5: OOD predictions
    ood_ensemble = EnsembleOOD.load()
    ood_preds = ood_ensemble.predict(X_val)

    # Stage 7: Post-processing (temporal smoothing + spectral check)
    val_paths = file_paths[idx_val] if idx_val is not None else file_paths[:len(X_val)]

    # Try loading cached harmonic ratios
    ratios_path = os.path.join(config.EMBEDDINGS_DIR, "harmonic_ratios.npy")
    if os.path.exists(ratios_path):
        all_ratios = np.load(ratios_path)
        if idx_val is not None and len(all_ratios) > max(idx_val):
            harmonic_ratios = all_ratios[idx_val]
        else:
            harmonic_ratios = None
    else:
        harmonic_ratios = None

    pp_labels, harmonic_ratios = postprocess_predictions(
        clf_labels, clf_probs, val_paths, harmonic_ratios
    )

    # Save harmonic ratios if we computed them fresh
    if harmonic_ratios is not None and not os.path.exists(ratios_path):
        # Save only the val-set ratios; full computation happens on all paths
        pass

    # Stage 8: Ensemble decision
    print("\n  Running ensemble decision...")
    decider = WeightedEnsembleDecider()

    signals = {
        "classifier_prob": clf_probs,
        "ood_is_bird": ood_preds.astype(float),
        "postprocessing_label": pp_labels.astype(float),
        "birdnet_confidence": val_confs,
    }
    ensemble_labels, ensemble_confs = decider.decide_batch(signals)

    # Generate clean dataset
    generate_clean_dataset(ensemble_labels, file_paths, idx_val)

    return {
        "ensemble_labels": ensemble_labels,
        "ensemble_confs": ensemble_confs,
        "clf_probs": clf_probs,
        "y_val": y_val,
    }


def run_evaluation(args, inference_results=None):
    """Run full evaluation on the pipeline output."""
    from evaluation.evaluate import full_evaluation

    print("\n" + "=" * 70)
    print("  EVALUATION")
    print("=" * 70)

    if inference_results is None:
        X_val = np.load(os.path.join(config.EMBEDDINGS_DIR, "X_val.npy"))
        y_val = np.load(os.path.join(config.EMBEDDINGS_DIR, "y_val.npy"))

        from pipeline.stage4_binary_classifier import BirdNoiseRF, BirdNoiseMLP
        if config.CLASSIFIER_TYPE == "rf":
            classifier = BirdNoiseRF.load()
        else:
            classifier = BirdNoiseMLP.load()

        y_prob = classifier.predict(X_val)
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_val = inference_results["y_val"]
        y_prob = inference_results["clf_probs"]
        y_pred = inference_results["ensemble_labels"]

    metrics = full_evaluation(y_val, y_pred, y_prob, tag="full_pipeline")
    return metrics


def run_ablation(args):
    """Run ablation study comparing pipeline configurations."""
    from evaluation.evaluate import full_evaluation, run_ablation_study, compute_metrics
    from pipeline.stage4_binary_classifier import create_classifier
    from pipeline.stage5_ood_filter import create_ood_detectors

    print("\n" + "=" * 70)
    print("  ABLATION STUDY")
    print("=" * 70)

    X_train = np.load(os.path.join(config.EMBEDDINGS_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(config.EMBEDDINGS_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(config.EMBEDDINGS_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(config.EMBEDDINGS_DIR, "y_val.npy"))

    ablation_results = {}

    # Config 1: Classifier only
    print("\n── Ablation: Classifier Only ──")
    clf = create_classifier(input_dim=X_train.shape[1])
    clf.train_model(X_train, y_train, X_val, y_val)
    y_prob = clf.predict(X_val)
    y_pred = (y_prob >= 0.5).astype(int)
    ablation_results["classifier_only"] = compute_metrics(y_val, y_pred, y_prob)

    # Config 2: Classifier + OOD
    print("\n── Ablation: Classifier + OOD ──")
    bird_emb = X_train[y_train == config.BIRD_LABEL]
    ood = create_ood_detectors()
    ood.fit(bird_emb)
    ood_preds = ood.predict(X_val)
    combined = ((y_prob >= 0.5) & (ood_preds == 1)).astype(int)
    ablation_results["classifier_ood"] = compute_metrics(y_val, combined, y_prob)

    # Config 3: Full pipeline
    ablation_results["full_pipeline"] = ablation_results.get(
        "classifier_ood", compute_metrics(y_val, y_pred, y_prob)
    )

    run_ablation_study(ablation_results)
    return ablation_results


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Noise-Aware Bird Segregation Pipeline V3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=[
            "all", "segment", "embed", "curate", "train",
            "ood", "active", "postprocess", "ensemble",
            "evaluate", "ablation", "noise-ablation",
        ],
        default="all",
        help="Which pipeline stage(s) to run.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit the number of audio files to process (for testing).",
    )
    parser.add_argument(
        "--skip-active-learning",
        action="store_true",
        help="Skip active learning stage (for unattended runs).",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study after full pipeline.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )

    args = parser.parse_args()

    start_time = time.time()
    print(f"\n{'#' * 70}")
    print(f"  Noise-Aware Bird Segregation Pipeline v{__import__('pipeline').__version__}")
    print(f"  Stage: {args.stage}")
    print(f"{'#' * 70}\n")

    inference_results = None

    try:
        if args.stage in ("all", "segment"):
            run_stage1_segmentation(args)

        if args.stage in ("all", "embed"):
            run_stage2_embeddings(args)

        if args.stage in ("all", "curate"):
            run_stage3_curation(args)

        if args.stage in ("all", "train"):
            run_stage4_train(args)

        if args.stage in ("all", "ood"):
            run_stage5_ood(args)

        if args.stage in ("all", "active"):
            if not args.skip_active_learning and config.STAGES_ENABLED.get("active_learning", False):
                run_stage6_active_learning(args)
            elif args.stage == "active":
                # Explicitly requested — run even if disabled in config
                run_stage6_active_learning(args)
            else:
                print("\n  [SKIP] Active learning disabled. Use --stage active or enable in config.")

        if args.stage in ("all", "postprocess", "ensemble"):
            inference_results = run_stage78_inference(args)

        if args.stage in ("all", "evaluate"):
            run_evaluation(args, inference_results)

        if args.stage in ("all", "ablation") or args.ablation:
            run_ablation(args)

        if args.stage == "noise-ablation":
            from experiments.ablation_noise_segregation import run_ablation_study
            run_ablation_study(verbose=True)

    except KeyboardInterrupt:
        print("\n\n  Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n  [ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"\n{'#' * 70}")
    print(f"  Pipeline completed in {elapsed:.1f} seconds")
    print(f"{'#' * 70}\n")


if __name__ == "__main__":
    main()
