# -*- coding: utf-8 -*-
"""
Car detection + color filter + speed & acceleration estimation (OpenCV + Pygame)
- Improved red detection (dual HSV ranges)
- Simple centroid tracker to estimate speed using v = d / t
- Acceleration estimation using a = Δv / Δt
- Scale can be set via --ppm (meters per pixel) or interactive calibration (press C and click two points, then enter meters)
"""

import argparse
import sys
import time
from collections import deque

import cv2
import numpy as np
import pygame

# --- Color ranges in HSV ---
COLOR_RANGES = {
    "red": [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (180, 255, 255))],
    "blue": [((100, 150, 0), (140, 255, 255))],
    "yellow": [((20, 100, 100), (30, 255, 255))],
    "white": [((0, 0, 200), (180, 25, 255))],
    "black": [((0, 0, 0), (180, 255, 50))],
}

# --- Display colors for checkboxes ---
DISPLAY_COLORS = {
    "red": (200, 0, 0),
    "blue": (0, 100, 200),
    "yellow": (200, 200, 0),
    "white": (220, 220, 220),
    "black": (20, 20, 20),
}


def open_capture(video_path: str | None):
    if video_path is None:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video source")
    return cap


def init_pygame(width: int, height: int):
    pygame.init()
    pygame.display.set_caption("Car Detector — OpenCV + Pygame")
    screen = pygame.display.set_mode((width + 280, height))  # extra panel width for speed/calibration
    font = pygame.font.SysFont("consolas", 18)
    small = pygame.font.SysFont("consolas", 14)
    clock = pygame.time.Clock()
    return screen, font, small, clock


class Checkbox:
    def __init__(self, label, x, y, checked=False):
        self.label = label
        self.rect = pygame.Rect(x, y, 20, 20)
        self.checked = checked

    def draw(self, surface, font):
        pygame.draw.rect(surface, (255, 255, 255), self.rect, 0 if self.checked else 2)
        if self.checked:
            pygame.draw.line(surface, (0, 200, 0), (self.rect.left, self.rect.top), (self.rect.right, self.rect.bottom), 2)
            pygame.draw.line(surface, (0, 200, 0), (self.rect.left, self.rect.bottom), (self.rect.right, self.rect.top), 2)

        color_box = pygame.Rect(self.rect.right + 10, self.rect.top, 20, 20)
        disp_color = DISPLAY_COLORS[self.label.lower()]
        pygame.draw.rect(surface, disp_color, color_box)

        text = font.render(self.label, True, (255, 255, 255))
        surface.blit(text, (color_box.right + 10, self.rect.top - 2))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.checked = not self.checked


# --- Simple centroid tracker for speed & acceleration ---
class Track:
    _next_id = 1
    def __init__(self, cx, cy, timestamp):
        self.id = Track._next_id; Track._next_id += 1
        self.history = deque(maxlen=20)  # (time, x, y, speed_mps)
        self.update(cx, cy, timestamp)
        self.last_update = timestamp
        self.lost_frames = 0
        self.speed_kmh = 0.0
        self.accel = 0.0

    def update(self, cx, cy, t):
        self.history.append((t, float(cx), float(cy), None))
        self.last_update = t
        self.lost_frames = 0

    def estimate_speed_and_accel(self, meters_per_pixel: float | None):
        if len(self.history) < 5:
            return 0.0, 0.0
        t0, x0, y0, _ = self.history[0]
        t1, x1, y1, _ = self.history[-1]
        dt = max(1e-6, t1 - t0)
        d_pix = np.hypot(x1 - x0, y1 - y0)
        if meters_per_pixel is None:
            return 0.0, 0.0
        d_m = d_pix * meters_per_pixel
        v_mps = d_m / dt
        v_kmh = v_mps * 3.6
        self.speed_kmh = 0.8 * self.speed_kmh + 0.2 * v_kmh

        # --- acceleration ---
        if len(self.history) >= 2:
            t_prev, _, _, _ = self.history[-2]
            dt2 = max(1e-6, t1 - t_prev)
            v_prev = (np.hypot(x1 - x0, y1 - y0) * meters_per_pixel) / dt
            a = (v_mps - v_prev) / dt2
            self.accel = 0.7 * self.accel + 0.3 * a

        return self.speed_kmh, self.accel


def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1 = max(ax, bx); y1 = max(ay, by)
    x2 = min(ax+aw, bx+bw); y2 = min(ay+ah, by+bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw*ah + bw*bh - inter + 1e-6
    return inter / union


def match_tracks(tracks, boxes, t):
    # Greedy matching by IoU + proximity
    assigned = set()
    for tr in tracks:
        best_j = -1; best_score = -1
        for j, b in enumerate(boxes):
            if j in assigned: 
                continue
            i = iou(b, tr._last_box) if hasattr(tr, "_last_box") else 0
            cx, cy = b[0]+b[2]/2, b[1]+b[3]/2
            hx, hy = tr.history[-1][1], tr.history[-1][2]
            dist = np.hypot(cx-hx, cy-hy)
            score = i + (1.0 / (1.0 + dist/50.0)) * 0.3
            if score > best_score:
                best_score = score; best_j = j
        if best_j >= 0:
            assigned.add(best_j)
            bx, by, bw, bh = boxes[best_j]
            tr._last_box = (bx, by, bw, bh)
            tr.update(bx + bw/2, by + bh/2, t)
        else:
            tr.lost_frames += 1
    new_tracks = []
    for j, b in enumerate(boxes):
        if j not in assigned:
            bx, by, bw, bh = b
            tr = Track(bx + bw/2, by + bh/2, t)
            tr._last_box = (bx, by, bw, bh)
            new_tracks.append(tr)
    tracks[:] = [tr for tr in tracks if tr.lost_frames <= 10] + new_tracks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--min-area", type=int, default=900)
    parser.add_argument("--ppm", type=float, default=None, help="meters per pixel (e.g., 0.02 means 1px = 2cm)")
    parser.add_argument("--show-trails", action="store_true", help="draw motion trails for tracks")
    args = parser.parse_args()

    cap = open_capture(args.video)
    ret, frame = cap.read()
    if not ret:
        return
    h, w = frame.shape[:2]
    screen, font, small, clock = init_pygame(w, h)

    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    paused = False
    last_time = time.time()
    fps = 0.0

    meters_per_pixel = args.ppm  # can be set later via calibration
    calibrating = False
    calib_points = []  # two points in image coords

    checkboxes = [
        Checkbox("Blue", w + 20, 60),
        Checkbox("Red", w + 20, 100, checked=True),
        Checkbox("Yellow", w + 20, 140),
        Checkbox("White", w + 20, 180),
        Checkbox("Black", w + 20, 220),
    ]

    tracks = []

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release(); pygame.quit(); return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    cap.release(); pygame.quit(); return
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_c:
                    calibrating = True
                    calib_points.clear()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if calibrating and mx < w and my < h:
                    calib_points.append((mx, my))
                    if len(calib_points) == 2:
                        calibrating = False
                        (x1, y1), (x2, y2) = calib_points
                        pix_dist = float(np.hypot(x2 - x1, y2 - y1))
                        try:
                            print("Enter real distance in meters for the two clicked points (default 3.5): ", end="", flush=True)
                            import sys as _sys
                            import select
                            meters = 3.5
                            start = time.time()
                            while time.time() - start < 2.0:
                                if _sys.stdin in select.select([_sys.stdin], [], [], 0)[0]:
                                    val = _sys.stdin.readline().strip()
                                    if val:
                                        meters = float(val)
                                    break
                            meters_per_pixel = meters / pix_dist
                            print(f"Calibrated: {meters_per_pixel:.6f} m/px (distance {meters} m over {pix_dist:.1f} px)")
                        except Exception:
                            meters_per_pixel = 3.5 / pix_dist
                            print(f"Calibrated with default 3.5m: {meters_per_pixel:.6f} m/px")
            for cb in checkboxes:
                cb.handle_event(event)

        if paused:
            clock.tick(30); continue

        ret, frame = cap.read()
        if not ret:
            if args.video is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            else:
                break

        t = time.time()
        fg = backSub.apply(frame)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
        fg = cv2.dilate(fg, np.ones((5, 5), np.uint8), iterations=2)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        active_colors = [cb.label.lower() for cb in checkboxes if cb.checked]
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < args.min_area:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            roi = frame[y:y+ch, x:x+cw]
            if roi.size == 0:
                continue
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            match = False
            for cname in active_colors:
                for lower, upper in COLOR_RANGES[cname]:
                    lower = np.array(lower); upper = np.array(upper)
                    mask = cv2.inRange(hsv, lower, upper)
                    ratio = cv2.countNonZero(mask) / float(cw * ch)
                    if ratio > 0.2:
                        match = True; break
                if match: break
            if not match:
                continue
            boxes.append((x, y, cw, ch))

        match_tracks(tracks, boxes, t)

        frame_draw = frame.copy()
        for tr in tracks:
            if not hasattr(tr, "_last_box"): 
                continue
            x, y, cw, ch = map(int, tr._last_box)
            cv2.rectangle(frame_draw, (x, y), (x+cw, y+ch), (0, 255, 0), 2)
            v, a = tr.estimate_speed_and_accel(meters_per_pixel)
            label = f"ID {tr.id}"
            if meters_per_pixel is not None:
                label += f" | {v:4.1f} km/h | a={a:.2f} m/s²"
            else:
                label += " | set scale (C)"
            cv2.putText(frame_draw, label, (x, max(0, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
            if args.show_trails and len(tr.history) >= 2:
                pts = np.array([(int(px), int(py)) for _, px, py, _ in tr.history])
                for i in range(1, len(pts)):
                    cv2.line(frame_draw, tuple(pts[i-1]), tuple(pts[i]), (0,255,0), 2)

        now = time.time()
        dt = now - last_time; last_time = now
        fps = (0.9 * fps + 0.1 * (1.0 / dt)) if dt > 0 else fps

        frame_rgb = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)
        surface = pygame.image.frombuffer(frame_rgb.tobytes(), (w, h), "RGB")
        screen.blit(surface, (0, 0))

        pygame.draw.rect(screen, (40, 40, 40), (w, 0, 280, h))
        title = font.render("Filters", True, (0, 255, 0)); screen.blit(title, (w + 20, 20))
        for cb in checkboxes: cb.draw(screen, font)

        screen.blit(font.render("Physics", True, (0, 255, 0)), (w + 20, 260))
        if meters_per_pixel is None:
            lines = ["Calibrate scale:", "Press C, click two", "points on road,", "enter meters in", "console (default 3.5)"]
        else:
            lines = [f"Scale: {meters_per_pixel:.5f} m/px", "v = d / t", "a = Δv / Δt", "km/h = m/s * 3.6"]
        y0 = 290
        for ln in lines:
            screen.blit(small.render(ln, True, (220, 220, 220)), (w + 20, y0)); y0 += 20

        screen.blit(small.render("Keys: Space pause, Q/Esc quit, C calibrate", True, (200,200,200)), (w + 20, h - 40))

        pygame.display.flip(); clock.tick(120)

    cap.release(); pygame.quit()


if __name__ == "__main__":
    main()
