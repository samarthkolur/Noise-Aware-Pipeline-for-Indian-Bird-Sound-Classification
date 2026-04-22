Optional local copy (exact filename) — checked first when birdnet_model_path is auto:

  BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite

If this file is missing here, the pipeline uses the copy bundled with pip package birdnetlib
(site-packages/.../birdnetlib/models/analyzer/). If neither exists, it errors.

How to obtain the file:
  1) pip install birdnetlib — then either rely on the bundle or copy the .tflite into this folder
  2) BirdNET-Analyzer-V2.4.zip: https://birdnet-team.github.io/BirdNET-Analyzer/models.html

This folder is named birdnet_weights (not "models/") to avoid clashing with the MLP Python package.
