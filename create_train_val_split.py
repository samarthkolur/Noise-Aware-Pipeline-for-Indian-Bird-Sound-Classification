import os
import random
import shutil

SOURCE_DIR = "data/segmented"
TRAIN_RATIO = 0.8

for label in ["bird", "noise"]:
    label_path = os.path.join(SOURCE_DIR, label)

    if not os.path.isdir(label_path):
        print(f"[INFO] Skipping label '{label}' — folder not found.")
        continue

    files = os.listdir(label_path)
    random.shuffle(files)

    split_index = int(len(files) * TRAIN_RATIO)

    train_files = files[:split_index]
    val_files = files[split_index:]

    for split, file_list in [("train", train_files), ("val", val_files)]:
        target_dir = os.path.join(SOURCE_DIR, split, label)
        os.makedirs(target_dir, exist_ok=True)

        for f in file_list:
            shutil.move(
                os.path.join(label_path, f),
                os.path.join(target_dir, f)
            )

print("Train/Val split complete.")