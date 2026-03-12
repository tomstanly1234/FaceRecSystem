# src/generate_embeddings.py
import os, pickle, cv2
import numpy as np
from utils import get_face_embedding, ensure_dir


def generate_embeddings(dataset_dir="../dataset", output_path="../models/embeddings.pkl"):
    """Generate embeddings for all faces in dataset"""
    ensure_dir(os.path.dirname(output_path))

    embeddings = []
    labels = []

    print("[INFO] Generating embeddings from dataset...")

    for person in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person)
        if not os.path.isdir(person_dir):
            continue

        person_embeddings = []

        for img_name in os.listdir(person_dir):
            path = os.path.join(person_dir, img_name)
            img = cv2.imread(path)

            if img is None:
                print(f"[WARNING] Could not read {path}")
                continue

            emb = get_face_embedding(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if emb is not None:
                embeddings.append(emb)
                labels.append(person)
                person_embeddings.append(emb)

        if person_embeddings:
            print(f"[INFO] {person}: {len(person_embeddings)} embeddings generated")
        else:
            print(f"[WARNING] {person}: No valid embeddings generated")

    if not embeddings:
        print("[ERROR] No embeddings generated! Check your dataset.")
        return

    # Save as dictionary with embeddings and labels
    data = {
        "embeddings": embeddings,
        "labels": labels
    }

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"[INFO] Saved {len(embeddings)} embeddings → {output_path}")
    print(f"[INFO] Classes: {set(labels)}")


if __name__ == "__main__":
    generate_embeddings()
