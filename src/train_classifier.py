# src/train_model.py
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from pathlib import Path


def train_model(embeddings_path="../models/embeddings.pkl",
                model_path="../models/classifier.pkl"):
    # Create models directory if it doesn't exist
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    try:
        with open(embeddings_path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: {embeddings_path} not found. Run data collection first!")
        return
    except EOFError:
        print(f"ERROR: {embeddings_path} is empty or corrupted!")
        return

    embeddings = data["embeddings"]
    labels = data["labels"]

    if len(embeddings) == 0:
        print("ERROR: No embeddings found. Collect face data first!")
        return

    print(f"[INFO] Training on {len(embeddings)} samples...")

    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    # Train classifier
    clf = SVC(kernel='linear', probability=True, C=1.0)
    clf.fit(embeddings, encoded_labels)

    # Save model
    with open(model_path, "wb") as f:
        pickle.dump((clf, le), f)

    print(f"[INFO] Model saved to {model_path}")
    print(f"[INFO] Classes: {le.classes_}")


if __name__ == "__main__":
    train_model()
