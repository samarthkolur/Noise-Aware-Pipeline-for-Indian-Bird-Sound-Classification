"""
Noise-Aware Pipeline for Indian Bird Sound Classification.

Reproduces "Noise-Aware Pipeline for Indian Bird Sound Classification Using
BirdNET Embeddings, Focal-Loss MLP, and Autoencoder Gating" (Kolur et al.,
DSU). See design.md for the full architecture and rationale.

Pipeline stages (design.md §6):
    1. RMS-based silence rejection + segmentation (audio.py)
    2. Noise Segregation V2 + Bird Guard + Bird Rescue
    3. Frozen BirdNET V2.4 embedding extraction (1024-D), HDF5-cached
    4. Focal-loss binary MLP classifier (bird vs. noise)
    5. Bird-only autoencoder OOD gate
    6. Three-band confidence router + hard-example mining
"""

__version__ = "1.0.0"
