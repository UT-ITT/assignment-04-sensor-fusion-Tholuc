"""
Controls
--------
  Arrow UP / DOWN   → increase / decrease alpha by 0.05
  ESC               → quit
  DIPPID Button 1   → reset prediction to current camera position
"""

import sys
import time
import threading

import cv2
import cv2.aruco as aruco
import numpy as np
import pyglet
import pyglet.window.key as key

# ── DIPPID import (graceful fallback if library not installed) ────────────────
from DIPPID import SensorUDP



# ── Configuration ─────────────────────────────────────────────────────────────
DIPPID_PORT   = 5700   # UDP port the DIPPID app sends data to
TRACKER_ID    = 5      # ArUco ID of the marker attached to the phone
ALPHA_INIT    = 0.7    # Initial complementary-filter weight (0 = cam, 1 = accel)
ALPHA_STEP    = 0.05   # Amount alpha changes per arrow-key press
ACC_SCALE     = 1.5   # Scalar multiplier for integrated acceleration (tune this)

# ── Video capture ─────────────────────────────────────────────────────────────
video_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(video_id)

WINDOW_WIDTH  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
WINDOW_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
print(f"[INFO] Camera resolution: {WINDOW_WIDTH}×{WINDOW_HEIGHT}")

# ── Pyglet window ─────────────────────────────────────────────────────────────
window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT,
                               caption="Sensor Fusion – ArUco + DIPPID")

# ── ArUco setup ───────────────────────────────────────────────────────────────
aruco_dict   = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()
detector     = aruco.ArucoDetector(aruco_dict, aruco_params)

# ── Shared state ──────────────────────────────────────────────────────────────
current_texture = None   # Frame rendered each draw call
M_last          = None   # Last valid perspective transform
alpha           = ALPHA_INIT

# Camera-observed position of marker ID 5 (pixel coords in warped space)
cam_pos         = None   # (x, y) or None if not visible

# Complementary-filter prediction (pixel coords)
pred_pos        = np.array([WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2], dtype=float)

# Velocity accumulated from accelerometer integration (pixels / s)
vel             = np.array([0.0, 0.0])

# ── Utilities ─────────────────────────────────────────────────────────────────

def cv2glet(img, fmt='BGR'):
    """Convert an OpenCV BGR image to a Pyglet ImageData for blitting."""
    rows, cols = img.shape[:2]
    channels   = 1 if img.ndim == 2 else img.shape[2]
    raw        = img.tobytes()
    pitch      = -(channels * cols)           # negative → flip vertically
    return pyglet.image.ImageData(cols, rows, fmt, raw, pitch=pitch)


def order_points(pts):
    """Return 4 points ordered: TL, TR, BR, BL."""
    pts = np.array(pts, dtype="float32")
    s   = pts.sum(axis=1)
    tl  = pts[np.argmin(s)]
    br  = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).flatten()
    tr  = pts[np.argmin(diff)]
    bl  = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def find_board_transform(corners, ids):
    """
    Pick 4 ArUco markers that form the board corners (any IDs ≠ TRACKER_ID),
    compute and return the perspective transform M, or None if not enough.
    """
    if ids is None or len(ids) < 4:
        return None

    # Exclude the tracker marker; use the first 4 remaining as board corners
    board_markers = [(c, i[0]) for c, i in zip(corners, ids) if i[0] != TRACKER_ID]
    if len(board_markers) < 4:
        return None

    centers = [np.mean(m[0][0], axis=0) for m in board_markers[:4]]
    src_pts = order_points(centers)

    dst_pts = np.array([
        [0,              0             ],
        [WINDOW_WIDTH,   0             ],
        [WINDOW_WIDTH,   WINDOW_HEIGHT ],
        [0,              WINDOW_HEIGHT ],
    ], dtype="float32")

    return cv2.getPerspectiveTransform(src_pts, dst_pts)


def find_tracker_in_warped(corners, ids, M):
    """
    Return the (x, y) centre of marker TRACKER_ID projected into warped space,
    or None if it is not visible.
    """
    if ids is None:
        return None
    for corner, mid in zip(corners, ids):
        if mid[0] == TRACKER_ID:
            centre = np.mean(corner[0], axis=0)          # (x, y) in raw frame
            pt     = np.array([[centre]], dtype="float32")
            warped = cv2.perspectiveTransform(pt, M)      # project into board
            wx, wy = warped[0][0]
            # Only accept positions inside the board rectangle
            if 0 <= wx <= WINDOW_WIDTH and 0 <= wy <= WINDOW_HEIGHT:
                return np.array([wx, wy], dtype=float)
    return None

# ── DIPPID accelerometer thread ───────────────────────────────────────────────

class AccelReader:
    """
    Runs in a background thread.  Continuously integrates accelerometer data
    to produce a velocity estimate.  Thread-safe via a lock.
    """
    def __init__(self):
        self._lock    = threading.Lock()
        self._vel     = np.array([0.0, 0.0])
        self._running = True
        self._last_t  = time.time()
        self._sensor  = None

        self._sensor = SensorUDP(DIPPID_PORT)
        self._sensor.register_callback('button_1', self._on_button)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_velocity(self):
        """Return integrated velocity (pixels / s) as a copy."""
        with self._lock:
            return self._vel.copy()

    def reset_velocity(self):
        """Zero the integrated velocity (also called by DIPPID button 1)."""
        with self._lock:
            self._vel[:] = 0.0

    def stop(self):
        self._running = False
        if self._sensor:
            self._sensor.disconnect()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_button(self, data):
        """DIPPID Button 1 pressed → reset velocity and prediction."""
        if int(data) == 1:
            self.reset_velocity()
            # Signal main thread to snap prediction back to last known cam pos
            global pred_pos, cam_pos
            if cam_pos is not None:
                pred_pos = cam_pos.copy()

    def _run(self):
        rate = 50          # Hz – poll rate for accelerometer
        dt   = 1.0 / rate
        while self._running:
            try:
                if 'accelerometer' in self._sensor.get_capabilities():
                    acc  = self._sensor.get_value('accelerometer')
                    ax   = acc.get('x', 0.0)
                    ay   = acc.get('y', 0.0)
                    # Map phone axes → screen pixel axes and scale
                    dpx  = ax * ACC_SCALE
                    dpy  = -ay * ACC_SCALE   # y is inverted between phone & screen
                    with self._lock:
                        self._vel[0] += dpx
                        self._vel[1] += dpy
            except Exception:
                pass
            time.sleep(dt)


accel = AccelReader()

# ── Main update loop ───────────────────────────────────────────────────────────

def update(dt):
    """Called by Pyglet's scheduler every frame (~30 fps)."""
    global M_last, cam_pos, pred_pos, current_texture

    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    # ── 1. Update board perspective transform ─────────────────────────────────
    M = find_board_transform(corners, ids)
    if M is not None:
        M_last = M

    if M_last is None:
        # Board not yet found – show raw frame with instructions
        msg_frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        cv2.putText(msg_frame,
                    "Show all 4 board ArUco markers to start",
                    (30, WINDOW_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        current_texture = cv2glet(msg_frame)
        return

    # ── 2. Warp board into rectangle ──────────────────────────────────────────
    warped = cv2.warpPerspective(frame, M_last, (WINDOW_WIDTH, WINDOW_HEIGHT))

    # Re-detect markers in the warped image (more stable for tracker)
    gray_w      = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    corners_w, ids_w, _ = detector.detectMarkers(gray_w)

    # ── 3. Locate tracker marker (ID 5) ───────────────────────────────────────
    new_cam_pos = None
    if ids_w is not None:
        for corner, mid in zip(corners_w, ids_w):
            if mid[0] == TRACKER_ID:
                cx, cy = np.mean(corner[0], axis=0)
                new_cam_pos = np.array([cx, cy], dtype=float)
                break

    if new_cam_pos is not None:
        cam_pos = new_cam_pos

    # ── 4. Complementary filter ───────────────────────────────────────────────
    # Accelerometer gives us a velocity-derived displacement each frame
    acc_vel   = accel.get_velocity()
    # Displacement this frame from integrated acceleration
    acc_delta = acc_vel * dt  # pixels

    if cam_pos is not None:
        # Blend: alpha weights the accelerometer/inertial prediction,
        # (1-alpha) snaps toward the camera measurement.
        pred_pos = alpha * (pred_pos + acc_delta) + (1 - alpha) * cam_pos
    else:
        # No camera fix available; rely on accelerometer alone
        pred_pos = pred_pos + acc_delta

    # Clamp to board bounds
    pred_pos[0] = np.clip(pred_pos[0], 0, WINDOW_WIDTH)
    pred_pos[1] = np.clip(pred_pos[1], 0, WINDOW_HEIGHT)

    # ── 5. Draw overlays ──────────────────────────────────────────────────────
    # Camera position → RED dot
    if cam_pos is not None:
        cx, cy = int(cam_pos[0]), int(cam_pos[1])
        cv2.circle(warped, (cx, cy), 14, (0, 0, 220), -1)
        cv2.circle(warped, (cx, cy), 14, (255, 255, 255), 2)
        cv2.putText(warped, "CAM", (cx + 17, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1, cv2.LINE_AA)
    else:
        cv2.putText(warped, "Marker ID 5 not detected",
                    (20, WINDOW_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1, cv2.LINE_AA)

    # Prediction → GREEN dot
    px, py = int(pred_pos[0]), int(pred_pos[1])
    cv2.circle(warped, (px, py), 10, (0, 200, 0), -1)
    cv2.circle(warped, (px, py), 10, (255, 255, 255), 2)
    cv2.putText(warped, "PRED", (px + 13, py + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)

    # HUD – alpha value and key hints
    cv2.putText(warped,
                f"Alpha: {alpha:.2f}  (UP/DOWN to adjust)",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(warped,
                "Red=Camera  Green=Prediction  DIPPID Btn1=Reset",
                (10, WINDOW_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    current_texture = cv2glet(warped)


# ── Pyglet event handlers ──────────────────────────────────────────────────────

@window.event
def on_draw():
    window.clear()
    if current_texture is not None:
        current_texture.blit(0, 0)


@window.event
def on_key_press(symbol, modifiers):
    global alpha, pred_pos, cam_pos
    if symbol == key.UP:
        alpha = min(1.0, round(alpha + ALPHA_STEP, 2))
        print(f"[INFO] Alpha → {alpha:.2f}  (accelerometer weight ↑)")
    elif symbol == key.DOWN:
        alpha = max(0.0, round(alpha - ALPHA_STEP, 2))
        print(f"[INFO] Alpha → {alpha:.2f}  (camera weight ↑)")
    elif symbol == key.ESCAPE:
        pyglet.app.exit()


@window.event
def on_close():
    accel.stop()
    cap.release()
    cv2.destroyAllWindows()


# ── Start ─────────────────────────────────────────────────────────────────────
pyglet.clock.schedule_interval(update, 1 / 50.0)

if __name__ == '__main__':
    print("[INFO] sensor_fusion.py started.")
    print(f"[INFO] Tracking marker ID {TRACKER_ID} | Initial alpha = {alpha}")
    print("[INFO] Arrow UP/DOWN → adjust alpha | ESC → quit | DIPPID Btn1 → reset")
    pyglet.app.run()
    cap.release()
    cv2.destroyAllWindows()