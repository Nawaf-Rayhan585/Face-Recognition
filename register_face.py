import cv2
import sys
import os
import numpy as np
from core.detector import detect_faces, save_recognizer

FACES_DIR = os.path.join("data", "faces")

def register(name, image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("could not read image")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    if len(faces) == 0:
        print("no face found in image, try another photo")
        return
    x, y, w, h = faces[0]
    face_crop = cv2.resize(gray[y:y + h, x:x + w], (200, 200))

    person_dir = os.path.join(FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    count = len(os.listdir(person_dir))
    cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), face_crop)
    print(f"saved photo for {name}, retraining model...")

    retrain()
    print(f"registered {name}")

def retrain():
    faces, ids, labels = [], [], {}
    if not os.path.exists(FACES_DIR):
        return
    for idx, name in enumerate(sorted(os.listdir(FACES_DIR))):
        person_dir = os.path.join(FACES_DIR, name)
        if not os.path.isdir(person_dir):
            continue
        labels[idx] = name
        for fname in os.listdir(person_dir):
            img = cv2.imread(os.path.join(person_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                ids.append(idx)

    if not faces:
        print("no registered faces yet")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    save_recognizer(recognizer, labels)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python register_face.py <name> <image_path>")
        sys.exit(1)
    register(sys.argv[1], sys.argv[2])
