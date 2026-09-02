import cv2
import argparse
import os
import time
from core.detector import detect_faces, load_recognizer
from core.ui import draw_box, draw_corners, draw_mesh, draw_hud

CONFIDENCE_THRESHOLD = 70  # lower = stricter match (LBPH distance)
WINDOW_NAME = "Face Recognition"

def process_frame(frame, recognizer, labels):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)

    for (x, y, w, h) in faces:
        x, y, w, h = int(x), int(y), int(w), int(h)
        face_crop = cv2.resize(gray[y:y + h, x:x + w], (200, 200))

        pad_x, pad_top, pad_bottom = int(w * 0.15), int(h * 0.35), int(h * 0.15)
        x = max(0, x - pad_x)
        y = max(0, y - pad_top)
        w = min(frame.shape[1] - x, w + pad_x * 2)
        h = min(frame.shape[0] - y, h + pad_top + pad_bottom)
        name = "Unknown"
        match_pct = 0
        if labels:
            label_id, distance = recognizer.predict(face_crop)
            if distance < CONFIDENCE_THRESHOLD:
                name = labels.get(label_id, "Unknown")
                match_pct = max(0, 100 - distance)

        draw_mesh(frame, x, y, w, h)
        draw_corners(frame, x, y, w, h)
        draw_box(frame, x, y, w, h, name, match_pct)

    return frame, len(faces)

def run_image(path, recognizer, labels, output):
    frame = cv2.imread(path)
    frame, _ = process_frame(frame, recognizer, labels)
    out_path = output or "output.jpg"
    cv2.imwrite(out_path, frame)
    print(f"saved to {out_path}")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_video(source, recognizer, labels, output):
    cam = cv2.VideoCapture(source)
    writer = None
    if output:
        fps = cam.get(cv2.CAP_PROP_FPS) or 20
        w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 800, 600)
    print("press q to quit")

    prev_time = time.time()
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        frame, face_count = process_frame(frame, recognizer, labels)

        now = time.time()
        fps = 1 / (now - prev_time) if now != prev_time else 0
        prev_time = now
        draw_hud(frame, fps, face_count)

        if writer:
            writer.write(frame)
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    if writer:
        writer.release()
        print(f"saved to {output}")
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="0 for webcam, or path to image/video file")
    parser.add_argument("--output", default=None, help="path to save result (image or video)")
    args = parser.parse_args()

    recognizer, labels = load_recognizer()

    source = args.source
    ext = os.path.splitext(source)[1].lower()

    if ext in (".jpg", ".jpeg", ".png", ".bmp"):
        run_image(source, recognizer, labels, args.output)
    elif ext in (".mp4", ".avi", ".mov", ".mkv"):
        run_video(source, recognizer, labels, args.output)
    else:
        run_video(int(source), recognizer, labels, args.output)

if __name__ == "__main__":
    main()