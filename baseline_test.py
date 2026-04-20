# DEPRECATED -- DO NOT USE
# This file used a non-existent 'birdnet' package with an incorrect API.
# BirdNET inference is now handled by:
#   run_baseline_birdnet.py   (raw audio)
#   run_processed_birdnet.py  (preprocessed segments)
#
# To run BirdNET evaluation:
#   python run_pipeline.py --from 4
raise SystemExit(
    "baseline_test.py is deprecated. "
    "Use run_baseline_birdnet.py or run_pipeline.py instead."
)
