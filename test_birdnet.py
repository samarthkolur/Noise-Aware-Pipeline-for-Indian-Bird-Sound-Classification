import sys
import pandas as pd
from pathlib import Path

try:
    from birdnetlib.analyzer import Analyzer
    from birdnetlib.recording import Recording
except ImportError as e:
    print(f"Error importing birdnetlib: {e}")
    sys.exit(1)

def main():
    analyzer = Analyzer()
    print("Analyzer loaded successfully!")
    
    # Read manifest to find a test file
    manifest_path = "data/embeddings/manifest.csv"
    if not Path(manifest_path).exists():
        print("Manifest not found")
        sys.exit(1)
        
    df = pd.read_csv(manifest_path)
    test_df = df[df["split"] == "test"]
    print(f"Test split contains {len(test_df)} segments")
    
    # Try one recording
    sample = test_df.iloc[0]["source_file"]
    recording = Recording(analyzer, sample, min_conf=0.1)
    recording.analyze()
    print(f"Sample: {sample}")
    print(f"Detections: {recording.detections}")

if __name__ == "__main__":
    main()
