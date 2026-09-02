import cv2

CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)

def draw_box(frame, x, y, w, h, name, confidence):
    color = CYAN if name != "Unknown" else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    label = name.upper()
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1.6, 3)
    label_y = y - 45 if y - 45 > th else y + h + th + 55

    cv2.rectangle(frame, (x - 4, label_y - th - 8), (x + tw + 8, label_y + 8), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_DUPLEX, 1.6, color, 3)

    if name != "Unknown":
        conf_text = f"{confidence:.0f}% MATCH"
        cv2.putText(frame, conf_text, (x, label_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def draw_corners(frame, x, y, w, h, size=28):
    color = YELLOW
    t = 3
    for px, py, dx, dy in [(x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)]:
        cv2.line(frame, (px, py), (px + dx * size, py), color, t)
        cv2.line(frame, (px, py), (px, py + dy * size), color, t)

# fuller face mesh: jawline outline + temples + brows + eyes + nose + mouth
MESH_POINTS = [
    (0.10, 0.45), (0.13, 0.60), (0.20, 0.75), (0.32, 0.88), (0.50, 0.95),
    (0.68, 0.88), (0.80, 0.75), (0.87, 0.60), (0.90, 0.45),          # 0-8 jawline
    (0.12, 0.22), (0.88, 0.22),                                     # 9-10 temples
    (0.28, 0.30), (0.40, 0.24), (0.60, 0.24), (0.72, 0.30),         # 11-14 brows
    (0.30, 0.42), (0.42, 0.40), (0.58, 0.40), (0.70, 0.42),         # 15-18 eyes
    (0.50, 0.46), (0.44, 0.60), (0.56, 0.60), (0.50, 0.63),         # 19-22 nose
    (0.34, 0.73), (0.50, 0.70), (0.66, 0.73),                       # 23-25 mouth top
    (0.40, 0.82), (0.50, 0.84), (0.60, 0.82),                       # 26-28 mouth bottom
]

MESH_LINES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
    (9, 0), (10, 8), (9, 11), (10, 14),
    (11, 12), (12, 13), (13, 14),
    (11, 15), (12, 16), (13, 17), (14, 18),
    (15, 16), (16, 19), (17, 19), (17, 18),
    (19, 20), (19, 21), (20, 22), (21, 22),
    (16, 23), (17, 25), (22, 24),
    (23, 24), (24, 25), (23, 26), (25, 28), (26, 27), (27, 28),
    (2, 23), (6, 25), (3, 26), (5, 28),
]

def draw_mesh(frame, x, y, w, h):
    overlay = frame.copy()
    pts = [(int(x + rx * w), int(y + ry * h)) for rx, ry in MESH_POINTS]
    for a, b in MESH_LINES:
        cv2.line(overlay, pts[a], pts[b], CYAN, 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(overlay, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

def draw_hud(frame, fps, face_count):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 55), (220, h), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 1)
    cv2.putText(frame, f"FACES: {face_count}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 1)
    cv2.rectangle(frame, (w - 100, 0), (w, 35), (0, 0, 255), cv2.FILLED)
    cv2.putText(frame, "LIVE", (w - 85, 24), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)