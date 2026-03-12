# Face Recognition Attendance System

A real-time **face recognition attendance system** built using **Python** and computer vision techniques. The system detects faces from a webcam, identifies known individuals using a trained model, and records attendance automatically.

---

# Overview

This project implements a complete **face recognition pipeline**:

1. Collect face images of users
2. Convert images into numerical face embeddings
3. Train a machine learning classifier
4. Recognize faces in real time
5. Store attendance records in a database

The system converts each face into a high-dimensional feature vector and uses a trained classifier to identify individuals.

---

# Features

* Real-time face detection using webcam
* Face recognition using deep learning embeddings
* Machine learning classifier for identity prediction
* Automatic attendance logging
* Dataset creation for new users
* Local database storage for attendance records

---

# Technologies Used

* Python
* OpenCV
* facenet-pytorch
* scikit-learn
* NumPy
* SQLite
* Streamlit (optional dashboard)

---

# Project Structure

```
FaceRecSys/
│
├── dataset/
│   ├── person1/
│   ├── person2/
│
├── models/
│   ├── embeddings.pkl
│   └── classifier.pkl
│
├── db/
│   └── attendance.db
│
├── src/
│   ├── collect_dataset.py
│   ├── generate_embeddings.py
│   ├── train_classifier.py
│   ├── realtime_recognition.py
│   ├── utils.py
│   ├── api_server.py
│   └── streamlit_app.py
│
└── requirements.txt
```

---

# System Workflow

The program follows this pipeline:

```
Dataset Images
      ↓
Face Detection
      ↓
Face Embedding Generation
      ↓
Train Classifier
      ↓
Real-Time Recognition
      ↓
Attendance Logging
```

---

# Mathematical Approach

### Face Embedding

Each detected face is converted into a **512-dimensional embedding vector**:

```
x = (x1, x2, x3, ..., x512)
```

These embeddings represent facial geometry.

### Classification

A Support Vector Machine (SVM) classifier is trained to separate embeddings belonging to different individuals using a hyperplane:

```
w · x + b = 0
```

Prediction is made based on which side of the hyperplane the embedding lies.

---

# Installation

Clone the repository:

```
git clone https://github.com/yourusername/FaceRecSystem.git
cd FaceRecSystem
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Usage

### 1. Collect Dataset

Capture images for each person:

```
python src/collect_dataset.py
```

Images will be stored inside the `dataset` folder.

---

### 2. Generate Face Embeddings

Convert images into feature vectors:

```
python src/generate_embeddings.py
```

This creates:

```
models/embeddings.pkl
```

---

### 3. Train Classifier

Train the recognition model:

```
python src/train_classifier.py
```

This generates:

```
models/classifier.pkl
```

---

### 4. Run Face Recognition

Start real-time recognition:

```
python src/realtime_recognition.py
```

The webcam will open and recognized faces will be labeled.

---

# Attendance Storage

Attendance is stored using a local SQLite database:

```
db/attendance.db
```

Each recognition inserts a record containing:

* Person name
* Timestamp
* Attendance status

---

# Example Recognition Flow

```
Camera Frame
      ↓
Face Detection
      ↓
Face Embedding Extraction
      ↓
Classifier Prediction
      ↓
Display Name
      ↓
Save Attendance
```

---

# Optional Web Interface

To launch the dashboard:

```
streamlit run src/streamlit_app.py
```

This allows you to view attendance logs and control the system from a browser.

---

# Future Improvements

* Cloud database integration
* Multi-camera support
* Face mask detection
* Mobile app integration
* Improved recognition accuracy

---

# Author

Developed as a computer vision project for automated attendance using face recognition.

---

# License

This project is for educational purposes.
