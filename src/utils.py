# src/utils.py
import os
import cv2
import numpy as np
from datetime import datetime
import sqlite3
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

# Initialize models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_face_embedding(frame):
    """Return 512D embedding for a detected face."""
    face = mtcnn(frame)
    if face is None:
        return None
    with torch.no_grad():
        embedding = resnet(face.unsqueeze(0).to(device))
    return embedding.squeeze().cpu().numpy()

def log_attendance(db_path, name):
    # ✅ Ensure the folder exists before connecting
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    # ✅ Connect to database (creates it if missing)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # ✅ Create table if it doesn’t exist
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    timestamp TEXT
                )''')

    # ✅ Insert the attendance record
    c.execute("INSERT INTO attendance (name, timestamp) VALUES (?, ?)",
              (name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    # ✅ Commit and close
    conn.commit()
    conn.close()

    print(f"[INFO] Attendance logged for: {name}")
