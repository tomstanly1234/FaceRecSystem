# src/realtime_recognition.py
import cv2, pickle, numpy as np
from utils import mtcnn, resnet, device, log_attendance, get_face_embedding
import time


def recognize(threshold=0.6, model_path="../models/classifier.pkl", db_path="../db/attendance.db"):
    with open(model_path, "rb") as f:
        clf, le = pickle.load(f)
    print("[INFO] Model loaded. Starting webcam...")

    # Track last attendance time for each person (cooldown period)
    last_attendance = {}
    ATTENDANCE_COOLDOWN = 30  # seconds between logging same person

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break

        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                face = frame[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                emb = get_face_embedding(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                if emb is not None:
                    probs = clf.predict_proba([emb])[0]
                    idx = np.argmax(probs)
                    confidence = probs[idx]
                    label = le.inverse_transform([idx])[0] if confidence > threshold else "Unknown"

                    # Color coding based on confidence
                    color = (0, 255, 0) if confidence > threshold else (0, 0, 255)

                    # Attendance logging with cooldown
                    current_time = time.time()
                    if (label != "Unknown" and
                            confidence > threshold and
                            (label not in last_attendance or
                             current_time - last_attendance[label] > ATTENDANCE_COOLDOWN)):
                        log_attendance(db_path, label)
                        last_attendance[label] = current_time
                        print(f"✅ Attendance logged for {label}")

                    # Display bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Real-Time Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):  # Manual recording trigger
            print("Manual recording triggered")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize()
