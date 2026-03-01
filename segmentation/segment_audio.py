import librosa
import soundfile as sf
import os
import glob
from tqdm import tqdm

INPUT_ROOT = "data/iBC53"
OUTPUT_ROOT = "data/segmented"
SEGMENT_LENGTH = 3.0
TARGET_SR = 48000


def segment_audio(file_path, species_name):
    y, sr = librosa.load(file_path, sr=TARGET_SR)
    samples_per_segment = int(SEGMENT_LENGTH * TARGET_SR)

    total_segments = len(y) // samples_per_segment

    species_output_dir = os.path.join(OUTPUT_ROOT, species_name)
    os.makedirs(species_output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(file_path))[0]

    for i in range(total_segments):
        start = i * samples_per_segment
        end = start + samples_per_segment
        segment = y[start:end]

        out_file = os.path.join(
            species_output_dir,
            f"{base_name}_seg{i:03d}.wav"
        )
        sf.write(out_file, segment, TARGET_SR)


def process_all():
    audio_files = glob.glob(f"{INPUT_ROOT}/**/*.wav", recursive=True)
    for file_path in tqdm(audio_files):
        species_name = os.path.basename(os.path.dirname(file_path))
        segment_audio(file_path, species_name)
    print(f"Searching inside: {INPUT_ROOT}")
    print(f"Found {len(audio_files)} files")


if __name__ == "__main__":
    process_all()