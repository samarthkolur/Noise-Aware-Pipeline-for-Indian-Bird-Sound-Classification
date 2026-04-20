# DEPRECATED -- DO NOT USE
# This script iterated data/segmented/bird/ expecting flat .wav files,
# but that directory contains species subdirectories -- making it a no-op
# (moves directories instead of files, produces an incorrect structure).
#
# Train/val splitting is now handled by:
#   etl/load.py  ->  splits/train.csv  +  splits/val.csv
#
# To regenerate splits:
#   python run_pipeline.py --only 3
raise SystemExit(
    "create_train_val_split.py is deprecated. "
    "Use etl/load.py or run_pipeline.py --only 3 instead."
)