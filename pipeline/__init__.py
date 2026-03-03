"""
Noise-Aware Bird Segregation Pipeline

A multi-stage pipeline for separating bird vocalizations from noise,
designed to work upstream of BirdNET for species classification.

Stages:
    1. Audio Segmentation & Standardization
    2. Deep Embedding Extraction (BirdNET, YAMNet, OpenL3)
    3. Supervised Binary Classifier (Bird vs Noise)
    4. Out-of-Distribution Detection
    5. Source Separation Refinement (HPSS)
    6. Temporal Consistency Modeling
    7. Ensemble Decision Mechanism
    8. Hard-Negative Mining
"""

__version__ = "2.0.0"
