import cv2
import os

CASCADE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "lbph_model.yml")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "labels.txt")

def detect_faces(gray_frame):
    return face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

def load_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    labels = {}
    if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
        recognizer.read(MODEL_PATH)
        with open(LABELS_PATH, "r") as f:
            for line in f:
                idx, name = line.strip().split(",", 1)
                labels[int(idx)] = name
    return recognizer, labels

def save_recognizer(recognizer, labels):
    recognizer.write(MODEL_PATH)
    with open(LABELS_PATH, "w") as f:
        for idx, name in labels.items():
            f.write(f"{idx},{name}\n")