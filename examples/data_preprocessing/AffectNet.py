import os
import numpy as np
import pandas as pd

# === CONFIG ===
dataset_root = "datasets/AffectNet"  # 修改为你实际路径
image_folder = "images"
annotation_folder = os.path.join(dataset_root, "annotations")
output_csv_path = os.path.join(dataset_root, "affectnet_cleaned.csv")

# === Label mapping (1–8 only) ===
emotion_map = {
    1: "Happiness",
    2: "Sadness",
    3: "Surprise",
    4: "Fear",
    5: "Disgust",
    6: "Anger",
    7: "Contempt"
}

records = []

for fname in os.listdir(annotation_folder):
    if not fname.endswith("_exp.npy"):
        continue

    prefix = fname.split("_")[0]
    print(f"🔍 Processing sample group: {prefix}")

    try:
        exp = np.load(os.path.join(annotation_folder, f"{prefix}_exp.npy"), allow_pickle=True)
        aro = np.load(os.path.join(annotation_folder, f"{prefix}_aro.npy"), allow_pickle=True)
        val = np.load(os.path.join(annotation_folder, f"{prefix}_val.npy"), allow_pickle=True)

        # 标量：只处理一张图 like 904.jpg
        if isinstance(exp, np.ndarray) and exp.ndim == 0:
            label = int(exp.item())
            if label in [0, 9, 10]:
                continue

            img_name = f"{prefix}.jpg"
            img_path = os.path.join(dataset_root, image_folder, img_name)
            if not os.path.exists(img_path):
                continue

            records.append({
                "img_name": img_name,
                "img_folder": image_folder,
                "emotion_cat": emotion_map[label],
                "emotion_v": float(val),
                "emotion_a": float(aro),
                "source_dataset": "AffectNet"
            })

        # 批量样本：prefix_0.jpg, prefix_1.jpg, ...
        elif isinstance(exp, np.ndarray) and exp.ndim >= 1:
            if not (len(exp) == len(aro) == len(val)):
                print(f"⚠️ Skipped {prefix}: Length mismatch among arrays.")
                continue

            for i in range(len(exp)):
                label = int(exp[i])
                if label in [0, 9, 10]:
                    continue

                img_name = f"{prefix}_{i}.jpg"
                img_path = os.path.join(dataset_root, image_folder, img_name)
                if not os.path.exists(img_path):
                    continue

                records.append({
                    "img_name": img_name,
                    "img_folder": image_folder,
                    "emotion_cat": emotion_map[label],
                    "emotion_v": float(val[i]),
                    "emotion_a": float(aro[i]),
                    "source_dataset": "AffectNet"
                })

        else:
            print(f"⚠️ Skipped {prefix}: Unrecognized .npy format.")

    except Exception as e:
        print(f"❌ Error processing {prefix}: {e}")
        continue

# === Save result ===
df = pd.DataFrame(records)
df.to_csv(output_csv_path, index=False)
print(f"\n✅ Saved {len(df)} valid samples to {output_csv_path}")
