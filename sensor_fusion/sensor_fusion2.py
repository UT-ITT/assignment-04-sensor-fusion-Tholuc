"""
sensor_fusion.py
Assignment 4 – Task 3: Sensor Fusion

Tracks an ArUco marker (ID 5, or ID 23 for DIPPID app) on a smartphone
within a 4-corner ArUco board using a webcam.

- Red dot   = raw camera position of the tracked marker
- Green dot = complementary-filter prediction (camera + accelerometer)

Controls:
  Arrow UP / DOWN      → increase / decrease alpha (camera weight)
  ESC or close window  → quit cleanly
DIPPID Button 1        → reset the prediction to current camera position
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import pyglet
import pyglet.window.key as key
import sys
from DIPPID import SensorUDP

# ── Configuration ─────────────────────────────────────────────────────────────
DIPPID_PORT     = 5700    # Default DIPPID UDP port
TRACKED_ID      = 5       # ArUco ID on phone (use 23 for DIPPID app marker)
ALPHA           = 0.85    # Initial camera weight (0 = only accel, 1 = only camera)
ALPHA_STEP      = 0.05    # Arrow-key adjustment step
ACCEL_SCALE     = 800.0   # Scales gravity-free acceleration (g-units) → pixels/s²
ACCEL_DEADZONE  = 0.02    # g-units below which linear accel is treated as zero
VELOCITY_DECAY  = 0.999    # Velocity damping per frame — bleeds drift when still

# ── Video / window setup ──────────────────────────────────────────────────────
video_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cap      = cv2.VideoCapture(video_id)

WINDOW_WIDTH  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
WINDOW_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
print(f"Webcam resolution: {WINDOW_WIDTH}x{WINDOW_HEIGHT}")

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT, caption="Sensor Fusion")

# ── ArUco setup ───────────────────────────────────────────────────────────────
aruco_dict   = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()
detector     = aruco.ArucoDetector(aruco_dict, aruco_params)

# ── DIPPID sensor ─────────────────────────────────────────────────────────────
sensor = SensorUDP(DIPPID_PORT)
print(f"Listening for DIPPID on UDP port {DIPPID_PORT}")

# ── State ─────────────────────────────────────────────────────────────────────
M_last          = None        
current_texture = None

cam_pos   = None              # Camera-observed (x, y) or None
pred_pos  = None              # Complementary-filter predicted (x, y) or None
velocity  = np.zeros(2)       # Integrated velocity in pixels/s (x, y)

alpha     = ALPHA

# ── Helpers ───────────────────────────────────────────────────────────────────
def cv2glet(img):
    """Convert a BGR OpenCV image to a Pyglet ImageData (bottom-up)."""
    rows, cols, ch = img.shape
    return pyglet.image.ImageData(
        width=cols, height=rows,
        fmt='BGR',
        data=img.tobytes(),
        pitch=-(ch * cols)
    )


def order_points(pts):
    """Sort 4 points into: TL, TR, BR, BL."""
    pts  = np.array(pts, dtype="float32")
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    return np.array([pts[np.argmin(s)],
                     pts[np.argmin(diff)],
                     pts[np.argmax(s)],
                     pts[np.argmax(diff)]], dtype="float32")


def marker_center(corners_single):
    """Return the integer (x, y) pixel center of one marker's corner array."""
    return tuple(np.mean(corners_single[0], axis=0).astype(int))


def cleanup():
    """Release all resources and exit — called from both ESC and window close."""
    pyglet.clock.unschedule(update)
    cap.release()
    cv2.destroyAllWindows()
    sensor.disconnect()
    pyglet.app.exit()


# ── DIPPID callback: Button 1 resets prediction and velocity ──────────────────
def on_button1(value):
    global pred_pos, velocity
    if value == 1 and cam_pos is not None:
        pred_pos = cam_pos
        velocity = np.zeros(2)
        print("Prediction and velocity reset via Button 1")

sensor.register_callback('button_1', on_button1)

# ── Main update loop ──────────────────────────────────────────────────────────
def update(dt):
    global M_last, cam_pos, pred_pos, velocity, current_texture

    ret, frame = cap.read()
    if not ret:
        return

    gray            = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    # ── 1. Update board from the 4 board corner markers ─────────────────
    if ids is not None:
        board_mask    = (ids.flatten() != TRACKED_ID)
        board_corners = [corners[i] for i in range(len(ids)) if board_mask[i]]

        if len(board_corners) == 4:
            centers = [np.mean(c[0], axis=0) for c in board_corners]
            src_pts = order_points(centers)
            dst_pts = np.array([
                [0,            0             ],
                [WINDOW_WIDTH, 0             ],
                [WINDOW_WIDTH, WINDOW_HEIGHT ],
                [0,            WINDOW_HEIGHT ]
            ], dtype="float32")
            M_last = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # ── 2. Warp the board region ──────────────────────────────────────────────
    if M_last is None:
        hint = frame.copy()
        cv2.putText(hint, "Show all 4 board ArUco markers",
                    (30, WINDOW_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)
        current_texture = cv2glet(hint)
        return

    warped = cv2.warpPerspective(frame, M_last, (WINDOW_WIDTH, WINDOW_HEIGHT))

    # ── 3. Detect the tracked marker inside the warped image ─────────────────
    gray_w              = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    w_corners, w_ids, _ = detector.detectMarkers(gray_w)

    new_cam_pos = None
    if w_ids is not None:
        for i, mid in enumerate(w_ids.flatten()):
            if mid == TRACKED_ID:
                new_cam_pos = marker_center(w_corners[i])
                break

    if new_cam_pos is not None:
        cam_pos = new_cam_pos
    else:
        cam_pos = None

    # ── 4. Read accelerometer and gravity from DIPPID ────────────────────────
    raw_accel = np.zeros(2)
    grav      = np.zeros(2)

    accel_data   = sensor.get_value('accelerometer')
    gravity_data = sensor.get_value('gravity')

    if accel_data and isinstance(accel_data, dict):
        raw_accel[0] = float(accel_data.get('x', 0.0))
        raw_accel[1] = float(accel_data.get('y', 0.0))

    if gravity_data and isinstance(gravity_data, dict):
        # DIPPID sends gravity in m/s² — normalise back to g-units to match accelerometer
        grav[0] = float(gravity_data.get('x', 0.0)) / 9.81
        grav[1] = float(gravity_data.get('y', 0.0)) / 9.81

    # ── 5. Gravity separation using DIPPID's gravity channel ─────────────────
    # just like we did in class
    linear_accel = raw_accel - grav

    # ── 6. Dead zone: zero out noise / residual when nearly still ─────────────
    linear_accel[np.abs(linear_accel) < ACCEL_DEADZONE] = 0.0

    # ── 7. Double-integrate with velocity damping ─────────────────────────────
    #
    #   velocity     = decay * velocity + linear_accel * scale * dt
    #   displacement = velocity * dt
    velocity   = VELOCITY_DECAY * velocity + linear_accel * ACCEL_SCALE * dt
    accel_disp = velocity * dt

    # ── 8. Complementary filter ───────────────────────────────────────────────
    #
    # Blends the accelerometer-predicted position with the camera observation:
    #
    #   pred = alpha * cam_pos + (1-alpha) * (pred + accel_disp)
    #
    # alpha → 1 : trusts camera heavily; prediction sticks tightly to red dot
    # alpha → 0 : trusts accelerometer; smoother but drifts over time


    if pred_pos is None and cam_pos is not None:
        pred_pos = cam_pos          # initialise on first detection

    if pred_pos is not None:
        px, py = float(pred_pos[0]), float(pred_pos[1])

        # Accelerometer prediction step
        px += accel_disp[0]
        py += accel_disp[1]

        # Blend with camera if the marker is currently visible
        if cam_pos is not None:
            cx, cy = cam_pos
            px = alpha * cx + (1 - alpha) * px
            py = alpha * cy + (1 - alpha) * py

        pred_pos = (int(np.clip(px, 0, WINDOW_WIDTH  - 1)),
                    int(np.clip(py, 0, WINDOW_HEIGHT - 1)))

    # ── 9. Draw overlays ──────────────────────────────────────────────────────
    # Red dot = camera position
    if cam_pos is not None:
        cv2.circle(warped, cam_pos, 10, (0, 0, 255), -1)
        cv2.putText(warped, "Camera",
                    (cam_pos[0] + 12, cam_pos[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    # Green dot = predicted position
    if pred_pos is not None:
        cv2.circle(warped, pred_pos, 10, (0, 220, 0), -1)
        cv2.putText(warped, "Predicted",
                    (pred_pos[0] + 12, pred_pos[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1, cv2.LINE_AA)

    # HUD
    cv2.putText(warped, f"alpha = {alpha:.2f}  (UP/DOWN to adjust)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(warped, f"Tracking marker ID {TRACKED_ID}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    if cam_pos is None:
        cv2.putText(warped, f"Marker {TRACKED_ID} not visible",
                    (10, WINDOW_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 165, 255), 1, cv2.LINE_AA)

    current_texture = cv2glet(warped)


# ── Pyglet callbacks ──────────────────────────────────────────────────────────
@window.event
def on_draw():
    window.clear()
    if current_texture is not None:
        current_texture.blit(0, 0)


@window.event
def on_close():
    """Triggered when the user closes the window — cleans up everything."""
    cleanup()


@window.event
def on_key_press(symbol, modifiers):
    global alpha
    if symbol == key.UP:
        alpha = min(1.0, round(alpha + ALPHA_STEP, 2))
        print(f"alpha → {alpha:.2f}  (more camera)")
    elif symbol == key.DOWN:
        alpha = max(0.0, round(alpha - ALPHA_STEP, 2))
        print(f"alpha → {alpha:.2f}  (more accelerometer)")
    elif symbol == key.ESCAPE:
        cleanup()


# ── Run ───────────────────────────────────────────────────────────────────────
pyglet.clock.schedule_interval(update, 1 / 30.0)

if __name__ == '__main__':
    pyglet.app.run()