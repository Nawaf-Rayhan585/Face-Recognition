# Face Recognition Template

Reusable face recognition with jarvis-style dot overlay. Uses OpenCV only —
Haar cascade for detection, LBPH for recognition. No dlib, no cmake, no downloads.

## Setup

```
pip install -r requirements.txt
```
That's it. `opencv-contrib-python` includes everything needed.

## Register a person

```
python register_face.py John john.jpg
```
Use a clear front-facing photo. You can register the same person multiple times
with different photos (different angles/lighting) — it improves accuracy, each
call retrains the model automatically.

## Run

```
python main.py                                   # webcam
python main.py --source photo.jpg --output result.jpg
python main.py --source video.mp4 --output result.mp4
```
Known face = name + cyan box + jarvis dots. Unknown = red box.
Press `q` to quit (webcam/video mode).

## Reuse for a new client

1. Copy this whole folder.
2. Delete everything inside `data/faces/`, plus `data/lbph_model.yml` and `data/labels.txt`.
3. Register new client's people.
4. Done — same code, new data.

## Notes
- `CONFIDENCE_THRESHOLD` in main.py (70) = strictness. LBPH confidence is a
  distance score — LOWER means better match, so lower threshold = stricter.
- Accuracy improves a lot with 3-5 photos per person vs just 1.
- Haar cascade is decent but not perfect at odd angles/low light — swap in a
  DNN detector later if a client needs higher accuracy.
