# src/collect_dataset.py
import cv2
import os
from utils import mtcnn, ensure_dir

def collect_images(person_name, output_dir="../dataset", num_images=60):
    ensure_dir(output_dir)
    person_dir = os.path.join(output_dir, person_name)
    ensure_dir(person_dir)

    cap = cv2.VideoCapture(0)
    count = 0
    print(f"[INFO] Capturing images for '{person_name}' (press 'q' to quit)...")

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                face = frame[y1:y2, x1:x2]
                if face.size > 0:
                    filename = os.path.join(person_dir, f"{count}.jpg")
                    cv2.imwrite(filename, cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    count += 1
                    print(f"Saved {count}/{num_images}")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.imshow("Dataset Collector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Collected {count} images for {person_name}.")

if __name__ == "__main__":
    name = input("Enter person's name: ").strip().lower()
    collect_images(name)
