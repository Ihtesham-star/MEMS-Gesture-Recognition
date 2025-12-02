import depthai as dai
import cv2
import mediapipe as mp
import socket
import struct
import csv
import time
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)
mp_draw = mp.solutions.drawing_utils

# Class to smooth coordinates for stability
class CoordinateSmoothing:
    def __init__(self, smoothing_factor=0.5):
        self.smoothing_factor = smoothing_factor
        self.prev_x = None
        self.prev_y = None

    def smooth(self, x, y):
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y
        smooth_x = self.prev_x * self.smoothing_factor + x * (1 - self.smoothing_factor)
        smooth_y = self.prev_y * self.smoothing_factor + y * (1 - self.smoothing_factor)
        self.prev_x, self.prev_y = smooth_x, smooth_y
        return int(smooth_x), int(smooth_y)

# Class to manage gesture states
class GestureState:
    def __init__(self):
        self.is_position_locked = False
        self.locked_position = None
        self.is_measuring = False
        self.current_side = 0  # 0: width, 1: height
        self.width = None
        self.height = None
        self.measuring_start_pos = None
        self.was_pinching = False
        self.last_index_state = None
        self.index_gesture_count = 0
        self.current_finger_pos = None
        self.square_drawn = False
        self.locking_enabled = True
        self.last_thumbs_up_state = False

    def reset(self):
        self.is_measuring = False
        self.measuring_start_pos = None
        self.width = None
        self.height = None
        self.current_side = 0
        self.last_index_state = None
        self.was_pinching = False
        self.square_drawn = False
        if not self.is_position_locked:
            self.locked_position = None

# Gesture detection functions
def is_fist(landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    all_fingers_closed = True
    for tip, pip in zip(finger_tips, finger_pips):
        if landmarks.landmark[tip].y < landmarks.landmark[pip].y:
            all_fingers_closed = False
            break
    return all_fingers_closed

def is_hand_fully_open(landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    all_open = True
    for tip, pip in zip(finger_tips, finger_pips):
        if landmarks.landmark[tip].y > landmarks.landmark[pip].y:
            all_open = False
            break
    return all_open

def is_index_finger_open(landmarks):
    index_tip = landmarks.landmark[8].y
    index_pip = landmarks.landmark[6].y
    return index_tip < index_pip

def is_thumbs_up(landmarks):
    thumb_tip = landmarks.landmark[4]
    thumb_ip = landmarks.landmark[3]
    thumb_mcp = landmarks.landmark[2]
    wrist = landmarks.landmark[0]
    thumb_extended_up = (thumb_tip.y < thumb_ip.y < thumb_mcp.y) and (thumb_tip.y < wrist.y - 0.1)
    thumb_vertical = abs(thumb_tip.x - thumb_mcp.x) < 0.1
    index_pip = landmarks.landmark[6]
    index_tip = landmarks.landmark[8]
    middle_pip = landmarks.landmark[10]
    middle_tip = landmarks.landmark[12]
    fingers_curled = (index_tip.y > index_pip.y) and (middle_tip.y > middle_pip.y)
    thumb_above_fingers = (thumb_tip.y < index_tip.y - 0.05) and (thumb_tip.y < middle_tip.y - 0.05)
    return thumb_extended_up and thumb_vertical and fingers_curled and thumb_above_fingers

def is_pinch_gesture(landmarks):
    thumb_tip = landmarks.landmark[4]
    index_tip = landmarks.landmark[8]
    distance = ((thumb_tip.x - index_tip.x)**2 + 
                (thumb_tip.y - index_tip.y)**2 + 
                (thumb_tip.z - index_tip.z)**2)**0.5
    return distance < 0.05

# Detect hand orientation (front or back)
def get_hand_orientation(landmarks):
    index_tip = landmarks.landmark[8]
    index_pip = landmarks.landmark[6]
    return "front" if index_tip.z < index_pip.z else "back"

# Function to draw a measured square (used in gesture interactions)
def draw_measured_square(s, center_x, center_y, width, height):
    try:
        s.sendall(b'release')
        time.sleep(0.1)
        s.sendall(struct.pack('dd', width, height))
        s.sendall(b'dimensions')
        time.sleep(0.1)
        half_width = width / 2
        half_height = height / 2
        square_points = [
            (center_x - half_width, center_y - half_height),
            (center_x + half_width, center_y - half_height),
            (center_x + half_width, center_y + half_height),
            (center_x - half_width, center_y + half_height),
            (center_x - half_width, center_y - half_height)
        ]
        for i in range(len(square_points)-1):
            start = square_points[i]
            end = square_points[i+1]
            steps = 20
            for j in range(steps):
                t = j / steps
                x = start[0] + (end[0] - start[0]) * t
                y = start[1] + (end[1] - start[1]) * t
                s.sendall(struct.pack('dd', x, y))
                s.sendall(b'draw')
                time.sleep(0.02)
        A = width / 2 * 0.9
        B = height / 2 * 0.9
        a = 3
        b = 2
        delta = 3.14/2
        steps = 1000
        start_x = center_x + A * math.sin(0)
        start_y = center_y + B * math.sin(delta)
        s.sendall(struct.pack('dd', start_x, start_y))
        time.sleep(0.1)
        for t in range(steps):
            t_normalized = (t / steps) * 2 * 3.14159
            x = center_x + A * math.sin(a * t_normalized)
            y = center_y + B * math.sin(b * t_normalized + delta)
            s.sendall(struct.pack('dd', x, y))
            s.sendall(b'draw')
            time.sleep(0.01)
        s.sendall(b'release')
        time.sleep(0.1)
        return True
    except Exception as e:
        print(f"Error in draw_measured_square: {e}")
        s.sendall(b'release')
        return False

# Utility function to display status text on the image
def show_status(image, text, position=(10, 30), color=(0, 255, 255)):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# Pipeline setup for DepthAI camera
pipeline = dai.Pipeline()
color_cam = pipeline.createColorCamera()
color_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
color_cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
xout_video = pipeline.createXLinkOut()
xout_video.setStreamName("video")
color_cam.video.link(xout_video.input)

# Setup TCP client for communication
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost', 30001))

# Initialize state variables
smoother = CoordinateSmoothing(smoothing_factor=0.3)
gesture_state = GestureState()

# Open timing data log file
timing_file = open('tracking1_analysis.csv', 'w', newline='')
timing_writer = csv.writer(timing_file)
timing_writer.writerow([
    "Timestamp", "Frame_Time", "Detection_Time", "Processing_Time",
    "Finger_X", "Finger_Y", "Is_Measuring", "Measured_Width",
    "Measured_Height", "Is_Drawing_Square", "Square_Center_X",
    "Square_Center_Y", "Is_Position_Locked", "Locked_X", "Locked_Y"
])

try:
    with dai.Device(pipeline) as device:
        q_video = device.getOutputQueue(name="video", maxSize=4, blocking=False)
        
        while True:
            frame_start_time = time.perf_counter()
            
            # Get video frame
            in_video = q_video.get()
            frame = in_video.getCvFrame()
            image = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process hand detection
            detection_start_time = time.perf_counter()
            results = hands.process(image_rgb)
            detection_time = time.perf_counter() - detection_start_time

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Get raw TIP position
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                raw_x = int(index_tip.x * image.shape[1])
                raw_y = int(index_tip.y * image.shape[0])
                
                # Determine hand orientation
                orientation = get_hand_orientation(hand_landmarks)
                
                # Adjust position based on orientation
                if orientation == "front":
                    adj_x, adj_y = raw_x, raw_y
                else:
                    dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
                    dip_x = int(dip.x * image.shape[1])
                    dip_y = int(dip.y * image.shape[0])
                    adj_x = int(0.7 * raw_x + 0.3 * dip_x)
                    adj_y = int(0.7 * raw_y + 0.3 * dip_y)
                
                # Smooth the adjusted position
                smooth_x, smooth_y = smoother.smooth(adj_x, adj_y)
                
                # Update current finger position
                gesture_state.current_finger_pos = (smooth_x, smooth_y)

                # Check hand gestures
                is_fist_gesture = is_fist(hand_landmarks)
                is_open_hand = is_hand_fully_open(hand_landmarks)
                is_index_open = is_index_finger_open(hand_landmarks)
                current_thumbs_up = is_thumbs_up(hand_landmarks)
                
                # Check for thumbs up toggle to enable/disable locking
                if current_thumbs_up and not gesture_state.last_thumbs_up_state:
                    gesture_state.locking_enabled = not gesture_state.locking_enabled
                    if not gesture_state.locking_enabled and gesture_state.is_position_locked:
                        gesture_state.is_position_locked = False
                        gesture_state.reset()
                
                gesture_state.last_thumbs_up_state = current_thumbs_up
                
                # Visualization
                display_x, display_y = gesture_state.locked_position if gesture_state.is_position_locked else gesture_state.current_finger_pos
                if gesture_state.is_measuring:
                    cv2.circle(image, (gesture_state.current_finger_pos[0], gesture_state.current_finger_pos[1]), 8, (0, 255, 255), -1)
                    cv2.circle(image, (gesture_state.locked_position[0], gesture_state.locked_position[1]), 8, (255, 0, 0), -1)
                else:
                    circle_color = (255, 0, 0) if gesture_state.is_position_locked else (0, 255, 0)
                    cv2.circle(image, (display_x, display_y), 8, circle_color, -1)
                
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Send position and record timing data
                if not gesture_state.is_measuring:
                    send_time = time.perf_counter()
                    send_x, send_y = gesture_state.locked_position if gesture_state.is_position_locked else gesture_state.current_finger_pos
                    s.sendall(struct.pack('dd', send_x, send_y))
                    processing_time = send_time - frame_start_time
                    
                    timing_writer.writerow([
                        time.time(),
                        frame_start_time,
                        detection_time,
                        processing_time,
                        send_x,
                        send_y,
                        1 if gesture_state.is_measuring else 0,
                        gesture_state.width if gesture_state.width else -1,
                        gesture_state.height if gesture_state.height else -1,
                        1 if gesture_state.square_drawn else 0,
                        send_x if gesture_state.square_drawn else -1,
                        send_y if gesture_state.square_drawn else -1,
                        1 if gesture_state.is_position_locked else 0,
                        gesture_state.locked_position[0] if gesture_state.locked_position else -1,
                        gesture_state.locked_position[1] if gesture_state.locked_position else -1
                    ])

            # Display the image
            cv2.imshow('Hand Tracking', image)
            if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
                break

finally:
    timing_file.close()
    cv2.destroyAllWindows()
    s.close()