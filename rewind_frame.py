"""
Rewind Frame - a hand-made window into the past

Hold up both hands and make a frame with your thumbs and index fingers.
Inside the frame you see the video from a few seconds ago; outside it, live.

Install:
    pip install opencv-python mediapipe numpy

Run:
    python rewind_frame.py

Controls (click on the camera window first so it has focus):
    [ / ]   -> decrease / increase rewind delay (0.5 s steps)
    - / =   -> decrease / increase edge feather
    , / .   -> decrease / increase smoothing
    d       -> toggle debug overlay (fingertips + frame outline)
    q       -> quit
"""

import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

THUMB_TIP = 4
INDEX_TIP = 8
WRIST = 0

FRAME_W, FRAME_H = 960, 720   # modest size keeps the frame buffer's memory in check
MIN_DELAY, MAX_DELAY = 0.5, 6.0


class HandSmoother:
    """Exponential moving average per hand slot (slot 0 = leftmost hand on screen)."""

    def __init__(self):
        self.state = {}  # slot -> {"thumb": (x, y), "index": (x, y)}

    def update(self, slot, thumb, index, a):
        if slot not in self.state:
            self.state[slot] = {"thumb": thumb, "index": index}
        else:
            s = self.state[slot]
            s["thumb"] = (s["thumb"][0] * a + thumb[0] * (1 - a),
                          s["thumb"][1] * a + thumb[1] * (1 - a))
            s["index"] = (s["index"][0] * a + index[0] * (1 - a),
                          s["index"][1] * a + index[1] * (1 - a))
        return self.state[slot]

    def reset(self):
        self.state.clear()


def build_polygon(points):
    """Convex hull: always a clean, non-self-intersecting outline."""
    if len(points) < 3:
        return None
    pts = np.array(points, dtype=np.float32)
    return cv2.convexHull(pts).astype(np.int32)


def blend_region(output, live, past, polygon, feather):
    """Blend `past` over `live` inside the polygon, working only on its
    bounding box (plus feather margin) instead of the whole frame."""
    h, w = live.shape[:2]
    k = max(1, feather | 1)  # Gaussian kernel must be odd
    margin = k
    x, y, bw, bh = cv2.boundingRect(polygon)
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(w, x + bw + margin), min(h, y + bh + margin)
    if x1 <= x0 or y1 <= y0:
        return

    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon - np.array([x0, y0], dtype=np.int32)], 255)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    m = (mask.astype(np.float32) / 255.0)[..., None]

    p = past[y0:y1, x0:x1].astype(np.float32)
    l = live[y0:y1, x0:x1].astype(np.float32)
    output[y0:y1, x0:x1] = (p * m + l * (1 - m)).astype(np.uint8)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Try a different camera index.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    smoother = HandSmoother()
    buffer = deque()          # (timestamp, frame) pairs, oldest on the left
    delay = 5.0               # seconds into the past
    feather = 21
    smoothing_amt = 0.55
    debug = False
    prev_hand_count = 0

    print("Rewind Frame running.")
    print("[ ] delay   - = feather   , . smoothing   d debug   q quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        frame = cv2.flip(frame, 1)  # mirror so movement feels natural
        h, w = frame.shape[:2]

        # --- rolling buffer, based on real timestamps (not a fixed frame count)
        buffer.append((now, frame))
        target = now - delay
        while len(buffer) > 1 and buffer[1][0] <= target:
            buffer.popleft()
        past = buffer[0][1] if buffer[0][0] <= target else None  # None = still buffering

        # --- hand tracking
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)

        detected = []
        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                detected.append(lm.landmark)
            # identify hands by screen position instead of MediaPipe's Left/Right label,
            # which can flip or duplicate when hands cross or get occluded
            detected.sort(key=lambda l: l[WRIST].x)

        if len(detected) != prev_hand_count:
            smoother.reset()  # avoid blending state between different hands
        prev_hand_count = len(detected)

        points = []
        for slot, l in enumerate(detected):
            thumb = (l[THUMB_TIP].x * w, l[THUMB_TIP].y * h)
            index = (l[INDEX_TIP].x * w, l[INDEX_TIP].y * h)
            s = smoother.update(slot, thumb, index, smoothing_amt)
            points.append(s["thumb"])
            points.append(s["index"])

        # --- composite
        output = frame.copy()
        polygon = build_polygon(points)

        if polygon is not None and past is not None:
            blend_region(output, frame, past, polygon, feather)

        if debug and polygon is not None:
            cv2.polylines(output, [polygon], True, (0, 255, 255), 2)
            for px, py in points:
                cv2.circle(output, (int(px), int(py)), 6, (0, 0, 255), -1)

        # --- status text
        if past is None:
            filled = now - buffer[0][0]
            status = f"buffering past... {filled:.1f}/{delay:.1f}s"
            color = (0, 200, 255)
        elif len(detected) < 2:
            status = "show both hands to open the portal"
            color = (200, 200, 200)
        else:
            status = f"portal open - showing {delay:.1f}s ago"
            color = (212, 234, 94)

        cv2.putText(output, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(output,
                    f"hands: {len(detected)}  delay: {delay:.1f}s  "
                    f"feather: {feather}  smoothing: {smoothing_amt:.2f}",
                    (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Rewind Frame", output)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('['):
            delay = max(MIN_DELAY, round(delay - 0.5, 1))
        elif key == ord(']'):
            delay = min(MAX_DELAY, round(delay + 0.5, 1))
        elif key == ord('-'):
            feather = max(1, feather - 2)
        elif key == ord('='):
            feather = min(101, feather + 2)
        elif key == ord(','):
            smoothing_amt = max(0.0, round(smoothing_amt - 0.05, 2))
        elif key == ord('.'):
            smoothing_amt = min(0.95, round(smoothing_amt + 0.05, 2))
        elif key == ord('d'):
            debug = not debug

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
