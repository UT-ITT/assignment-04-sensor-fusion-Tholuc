import cv2
import cv2.aruco as aruco
import numpy as np
import pyglet
import sys
import random

# Setup Video ID from command line arguments (defaults to 0 for default webcam)
video_id = 0
if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

# Initialize video capture
cap = cv2.VideoCapture(video_id)

#  NATIVE RESOLUTION EXTRACTION
WINDOW_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
WINDOW_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if WINDOW_WIDTH == 0 or WINDOW_HEIGHT == 0:
    WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480

print(f" Webcam Native Resolution Detected: {WINDOW_WIDTH}x{WINDOW_HEIGHT}")

# Window Configuration matches the camera properties exactly
window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT, caption="AR Bubble Popper")

# Setup ArUco Detector
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

# ---  AUDIO SETUP ---
try:
    pop_sound = pyglet.media.load('pop.wav', streaming=False)
    error_sound = pyglet.media.load('error.wav', streaming=False)
    audio_enabled = True
    print(" Sound effects loaded successfully!")
except Exception as e:
    audio_enabled = False
    print(" Audio files ('pop.wav' / 'error.wav') not found. Running in silent mode.")

# --- GAME STATE SETUP ---
score = 0
combo = 1                
bubbles = []
particles = []           
M_last = None
MAX_BUBBLES = 8
current_texture = None
game_won = False         #  Tracks if the player has won the game

# --- TUNING PARAMETERS ---
OBJECT_THRESHOLD = 90
BORDER_MASK = 45 
WIN_SCORE = 500          #  Target goal score required to win!

def cv2glet(img, fmt):
    '''Converts an OpenCV image matrix directly to raw bytes for Pyglet rendering'''
    if fmt == 'GRAY':
        rows, cols = img.shape
        channels = 1
    else:
        rows, cols, channels = img.shape

    raw_img = img.tobytes() 
    top_to_bottom_flag = -1
    bytes_per_row = channels * cols
    
    return pyglet.image.ImageData(width=cols, 
                                  height=rows, 
                                  fmt=fmt, 
                                  data=raw_img, 
                                  pitch=top_to_bottom_flag * bytes_per_row)

def order_points(pts):
    '''Sorts 4 coordinates uniformly: Top-Left, Top-Right, Bottom-Right, Bottom-Left'''
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).flatten()
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")

def manage_bubbles():
    '''Maintains the active target count and equips new spawns with physics velocities'''
    global bubbles
    if game_won:  # Stop spawning new targets if the player has won
        return
        
    while len(bubbles) < MAX_BUBBLES:
        bx = random.randint(BORDER_MASK + 20, WINDOW_WIDTH - BORDER_MASK - 20)
        by = random.randint(BORDER_MASK + 20, WINDOW_HEIGHT - BORDER_MASK - 20)
        br = random.randint(20, 35)
        
        vx = random.choice([-5, -3, 3, 5])
        vy = random.choice([-5, -3, 3, 5])
        
        btype = 'green' if random.random() > 0.30 else 'red'
        bubbles.append({'x': bx, 'y': by, 'vx': vx, 'vy': vy, 'r': br, 'type': btype})

def spawn_particles(x, y, color):
    '''Spawns an outward-bursting ring of particles when a bubble pops'''
    global particles
    for _ in range(10):
        particles.append({
            'x': x,
            'y': y,
            'vx': random.uniform(-6, 6),
            'vy': random.uniform(-6, 6),
            'r': random.randint(3, 6),
            'life': 12,  
            'color': color
        })

def update_game(dt):
    '''Main loop: processes input frames, runs background filter pipelines, and updates physics'''
    global M_last, score, combo, bubbles, particles, current_texture, game_won
    
    # If the game is won, freeze camera sampling and skip calculation cycles
    if game_won:
        return

    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is not None and len(corners) == 4:
        centers = [np.mean(marker_corners[0], axis=0) for marker_corners in corners]
        src_pts = order_points(centers)
        
        dst_pts = np.array([
            [-20, -20],
            [WINDOW_WIDTH + 20, -20],
            [WINDOW_WIDTH + 20, WINDOW_HEIGHT + 20],
            [-20, WINDOW_HEIGHT + 20]
        ], dtype="float32")
        
        M_last = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    if M_last is not None:
        warped = cv2.warpPerspective(frame, M_last, (WINDOW_WIDTH, WINDOW_HEIGHT))
        gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        
        # ---  Image Preprocessing Pipeline ---
        blurred = cv2.GaussianBlur(gray_warped, (11, 11), 0)
        _, thresh = cv2.threshold(blurred, OBJECT_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        
        # Edge Dead-zone Blocker
        cv2.rectangle(thresh, (0, 0), (WINDOW_WIDTH, BORDER_MASK), 0, -1)
        cv2.rectangle(thresh, (0, WINDOW_HEIGHT - BORDER_MASK), (WINDOW_WIDTH, WINDOW_HEIGHT), 0, -1)
        cv2.rectangle(thresh, (0, 0), (BORDER_MASK, WINDOW_HEIGHT), 0, -1)
        cv2.rectangle(thresh, (WINDOW_WIDTH - BORDER_MASK, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), 0, -1)
        
        kernel = np.ones((7, 7), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fX, fY = None, None  
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 1200: 
                extLeft  = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
                extRight = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
                extTop   = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
                extBot   = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
                
                # Default: Bottom Entry
                fX, fY = extTop[0], extTop[1]
                
                # Full 4-Directional Tracking Smart Compensation
                if extTop[1] <= BORDER_MASK + 5:    
                    fX, fY = extBot[0], extBot[1]
                elif extLeft[0] <= BORDER_MASK + 5:    
                    fX, fY = extRight[0], extRight[1]
                elif extRight[0] >= WINDOW_WIDTH - BORDER_MASK - 5: 
                    fX, fY = extLeft[0], extLeft[1]

        # --- Game Mechanics: Physics Loop ---
        manage_bubbles()
        remaining_bubbles = []
        
        for b in bubbles:
            b['x'] += b['vx']
            b['y'] += b['vy']
            
            if b['x'] - b['r'] <= BORDER_MASK or b['x'] + b['r'] >= WINDOW_WIDTH - BORDER_MASK:
                b['vx'] *= -1
                b['x'] = np.clip(b['x'], BORDER_MASK + b['r'], WINDOW_WIDTH - b['r'] - b['r'])
                
            if b['y'] - b['r'] <= BORDER_MASK or b['y'] + b['r'] >= WINDOW_HEIGHT - BORDER_MASK:
                b['vy'] *= -1
                b['y'] = np.clip(b['y'], BORDER_MASK + b['r'], WINDOW_HEIGHT - BORDER_MASK - b['r'])

            # Collision Detection
            popped = False
            if fX is not None and fY is not None:
                distance = np.sqrt((fX - b['x'])**2 + (fY - b['y'])**2)
                if distance < b['r']:
                    popped = True
                    b_color = (0, 255, 0) if b['type'] == 'green' else (0, 0, 255)
                    
                    spawn_particles(int(b['x']), int(b['y']), b_color)
                    
                    if b['type'] == 'green':
                        score += 10 * combo
                        combo += 1  
                        if audio_enabled:
                            pop_sound.play() 
                    else:
                        score -= 20
                        combo = 1   
                        if audio_enabled:
                            error_sound.play() 
            
            if not popped:
                remaining_bubbles.append(b)
                
        bubbles = remaining_bubbles

        # --- Check for Win Condition Trigger ---
        if score >= WIN_SCORE:
            game_won = True
            bubbles.clear()
            particles.clear()
            
            # Draw a  dark tinted glass semi-transparent overlay box across the entire board
            overlay = warped.copy()
            cv2.rectangle(overlay, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, warped, 0.25, 0, warped)
            
            # Render  Victory Screen Dashboard
            cv2.putText(warped, " VICTORY ", (WINDOW_WIDTH // 2 - 160, WINDOW_HEIGHT // 2 - 40), 
                        cv2.FONT_HERSHEY_TRIPLEX, 1.3, (0, 215, 255), 3, cv2.LINE_AA)
            cv2.putText(warped, f"FINAL SCORE: {score}", (WINDOW_WIDTH // 2 - 130, WINDOW_HEIGHT // 2 + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(warped, "Press 'ESC' to Quit", (WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 + 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
            
            current_texture = cv2glet(warped, 'BGR')
            return  # Stop processing further frame modifications

        # --- Update & Process Particle System Simulation ---
        next_particles = []
        for p in particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            p['r'] = max(1, int(p['r'] * 0.9)) 
            if p['life'] > 0:
                next_particles.append(p)
                cv2.circle(warped, (int(p['x']), int(p['y'])), p['r'], p['color'], -1)
        particles = next_particles

        # --- Draw Glossy 3D Game Objects ---
        for b in bubbles:
            color = (0, 220, 0) if b['type'] == 'green' else (0, 0, 220)
            cv2.circle(warped, (int(b['x']), int(b['y'])), b['r'], color, -1)
            cv2.circle(warped, (int(b['x']), int(b['y'])), b['r'], (0, 0, 0), 1)
            hx = int(b['x'] - b['r'] * 0.35)
            hy = int(b['y'] - b['r'] * 0.35)
            hr = max(2, int(b['r'] * 0.2))
            cv2.circle(warped, (hx, hy), hr, (255, 255, 255), -1)
        
        if fX is not None and fY is not None:
            cv2.circle(warped, (fX, fY), 6, (0, 255, 255), -1)  
            cv2.drawContours(warped, [largest_contour], -1, (0, 255, 255), 1)  
        else:
            cv2.putText(warped, "Bring pointer over board", (BORDER_MASK + 10, WINDOW_HEIGHT - BORDER_MASK - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # --- Arcade Dashboard UI Overlay ---
        cv2.putText(warped, f"SCORE: {score} / {WIN_SCORE}", (BORDER_MASK + 10, BORDER_MASK + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        
        if combo > 1:
            cv2.putText(warped, f"COMBO x{combo}", (BORDER_MASK + 10, BORDER_MASK + 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        current_texture = cv2glet(warped, 'BGR')
    else:
        msg_frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        cv2.putText(msg_frame, "Show All 4 ArUco Markers To Start Game", (50, WINDOW_HEIGHT // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        current_texture = cv2glet(msg_frame, 'BGR')

@window.event
def on_draw():
    window.clear()
    if current_texture is not None:
        current_texture.blit(0, 0, 0)

pyglet.clock.schedule_interval(update_game, 1/30.0)

if __name__ == '__main__':
    pyglet.app.run()
    cap.release()
    cv2.destroyAllWindows()