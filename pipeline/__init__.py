"""
Noise-Aware Bird Segregation Pipeline V3

A data-centric pipeline for false-positive suppression in passive acoustic
monitoring. Designed to work upstream of BirdNET for species classification.

Architecture:
    1. Audio Segmentation & Standardization
    2. BirdNET Embedding Extraction (1024-d)
    3. Hard-Negative Dataset Curation
    4. Binary Classifier (Random Forest / MLP)
    5. OOD Detection (Mahalanobis + Isolation Forest)
    6. Active Learning / Expert-in-the-Loop Feedback
    7. Post-Processing (Spectral + Temporal + Ecological)
    8. Ensemble Decision & Final Output
"""

__version__ = "3.0.0"
