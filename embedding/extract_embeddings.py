"""
extract_embeddings.py — Batch embedding extraction from processed features.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.logger import get_logger

logger = get_logger(__name__)


class _FeatureDataset(Dataset):
    """Simple dataset that loads pre-saved feature tensors."""

    def __init__(self, feature_dir: Path) -> None:
        self.paths = sorted(feature_dir.rglob("*.pt"))
        if not self.paths:
            raise FileNotFoundError(
                f"No .pt feature files found in {feature_dir}"
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        feature = torch.load(str(path), weights_only=True)
        # Extract species label from parent directory name
        species = path.parent.name
        return feature, species, str(path)


class EmbeddingExtractor:
    """Extract embeddings from pre-computed features using a trained encoder."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.device = self._resolve_device(cfg)
        self.batch_size = cfg["embedding"]["batch_size"]
        self.num_workers = cfg["embedding"]["num_workers"]
        self.feature_dir = Path(cfg["data"]["processed_dir"])
        self.output_dir = Path(cfg["data"]["embeddings_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Extract embeddings for all processed features and save to disk."""
        from embedding.embedding_model import EmbeddingModel

        model = EmbeddingModel(self.cfg).to(self.device)
        model.eval()

        dataset = _FeatureDataset(self.feature_dir)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        logger.info(
            f"Extracting embeddings for {len(dataset)} features → {self.output_dir}"
        )

        with torch.no_grad():
            for features, species_list, paths in tqdm(loader, desc="Embedding"):
                features = features.to(self.device)
                embeddings = model(features)

                for emb, species, src_path in zip(
                    embeddings.cpu(), species_list, paths
                ):
                    save_dir = self.output_dir / species
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_name = Path(src_path).stem + "_emb.pt"
                    torch.save(emb, str(save_dir / save_name))

        logger.info("Embedding extraction finished.")

    @staticmethod
    def _resolve_device(cfg: dict) -> torch.device:
        device_str = cfg["project"]["device"]
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)
