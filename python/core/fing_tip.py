import depthai as dai
import cv2
import mediapipe as mp
import socket
import struct
import csv
import time
import math

# Set this flag to True for laser to shine directly ON your fingertip
FINGERTIP_TARGETING_MODE = True

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)
mp_draw = mp.solutions.drawing_utils

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

def is_pinch_gesture(landmarks):
    thumb_tip = landmarks.landmark[4]
    index_tip = landmarks.landmark[8]
    
    distance = ((thumb_tip.x - index_tip.x)**2 + 
                (thumb_tip.y - index_tip.y)**2 + 
                (thumb_tip.z - index_tip.z)**2)**0.5
    
    return distance < 0.05

def draw_measured_square(s, center_x, center_y, width, height):
    try:
        # Clear any pending states first
        s.sendall(b'release')
        time.sleep(0.1)  # Small delay to ensure clean state

        # NEW CODE: First move smoothly to the starting corner
        # Calculate the top-left corner position
        start_x = center_x - width/2
        start_y = center_y - height/2
        
        # Send a specific command to indicate movement to start position
        s.sendall(struct.pack('dd', start_x, start_y))
        s.sendall(b'move_to_start')  # New command for smooth transition
        time.sleep(0.1)
        
        # Step 1: Send dimensions and initiate square drawing mode
        s.sendall(struct.pack('dd', width, height))
        s.sendall(b'dimensions')
        time.sleep(0.1)

        # Step 2: Draw the square outline
        # Calculate corner points
        half_width = width / 2
        half_height = height / 2
        square_points = [
            (center_x - half_width, center_y - half_height),  # Top-left
            (center_x + half_width, center_y - half_height),  # Top-right
            (center_x + half_width, center_y + half_height),  # Bottom-right
            (center_x - half_width, center_y + half_height),  # Bottom-left
            (center_x - half_width, center_y - half_height)   # Back to top-left
        ]
        
        # Draw square outline with interpolation
        for i in range(len(square_points)-1):
            start = square_points[i]
            end = square_points[i+1]
            steps = 20  # Number of interpolation steps
            
            for j in range(steps):
                t = j / steps
                x = start[0] + (end[0] - start[0]) * t
                y = start[1] + (end[1] - start[1]) * t
                s.sendall(struct.pack('dd', x, y))
                s.sendall(b'draw')
                time.sleep(0.02)  # Small delay for smooth motion
        
        # Step 3: Perform Lissajous scan
        A = width / 2 * 0.9   # Slightly smaller than square to ensure it fits
        B = height / 2 * 0.9
        a = 3  # Frequency ratio x
        b = 2  # Frequency ratio y
        delta = 3.14/2  # Phase difference
        steps = 1000     # Increased points for smoother pattern
        
        # Move to starting position of Lissajous pattern
        start_x = center_x + A * math.sin(0)
        start_y = center_y + B * math.sin(delta)
        s.sendall(struct.pack('dd', start_x, start_y))
        time.sleep(0.1)
        
        # Draw Lissajous pattern
        for t in range(steps):
            t_normalized = (t / steps) * 2 * 3.14159
            x = center_x + A * math.sin(a * t_normalized)
            y = center_y + B * math.sin(b * t_normalized + delta)
            
            s.sendall(struct.pack('dd', x, y))
            s.sendall(b'draw')
            time.sleep(0.01)  # Fast enough for smooth motion
        
        # Step 4: Complete the drawing
        s.sendall(b'release')
        time.sleep(0.1)
        
        return True
        
    except Exception as e:
        print(f"Error in draw_measured_square: {e}")
        s.sendall(b'release')
        return False

def show_status(image, text, position=(10, 30), color=(0, 255, 255)):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# Pipeline setup
pipeline = dai.Pipeline()
color_cam = pipeline.createColorCamera()
color_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
color_cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
xout_video = pipeline.createXLinkOut()
xout_video.setStreamName("video")
color_cam.video.link(xout_video.input)

# Setup TCP client
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost', 30001))

# Initialize state variables
smoother = CoordinateSmoothing(smoothing_factor=0.7 if FINGERTIP_TARGETING_MODE else 0.3)
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
            
            in_video = q_video.get()
            frame = in_video.getCvFrame()
            image = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            detection_start_time = time.perf_counter()
            results = hands.process(image_rgb)
            detection_time = time.perf_counter() - detection_start_time

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Get current finger position
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                current_x = int(index_tip.x * image.shape[1])
                current_y = int(index_tip.y * image.shape[0])
                smooth_x, smooth_y = smoother.smooth(current_x, current_y)
                
                # Update current finger position
                gesture_state.current_finger_pos = (smooth_x, smooth_y)

                # In fingertip targeting mode, the laser will shine directly ON the fingertip
                if FINGERTIP_TARGETING_MODE:
                    # Visualization
                    cv2.circle(image, (smooth_x, smooth_y), 8, (0, 255, 0), -1)
                    mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Send special position with a flag value
                    send_time = time.perf_counter()
                    # Use a special marker by modifying the y value
                    s.sendall(struct.pack('dd', smooth_x, -9999))  # Special y value as flag
                    time.sleep(0.01)  # Small delay
                    s.sendall(struct.pack('dd', smooth_x, smooth_y))  # Now send actual coordinates
                    processing_time = send_time - frame_start_time
                    
                    show_status(image, "Targeting Fingertip", position=(10, 60), color=(0, 255, 0))
                    
                    # Record timing data
                    timing_writer.writerow([
                        time.time(),
                        frame_start_time,
                        detection_time,
                        processing_time,
                        smooth_x,
                        smooth_y,
                        0,  # Not measuring
                        -1,  # No width
                        -1,  # No height
                        0,   # Not drawing square
                        -1,  # No square center x
                        -1,  # No square center y
                        0,   # Not position locked
                        -1,  # No locked x
                        -1   # No locked y
                    ])
                else:
                    # Original gesture control mode
                    # Check hand gestures
                    is_fist_gesture = is_fist(hand_landmarks)
                    is_open_hand = is_hand_fully_open(hand_landmarks)
                    is_index_open = is_index_finger_open(hand_landmarks)
                    
                    # Handle position locking
                    if is_fist_gesture and not gesture_state.is_position_locked:
                        gesture_state.is_position_locked = True
                        gesture_state.locked_position = (smooth_x, smooth_y)  # This will be the square center
                        show_status(image, "Position Locked!")
                    elif is_open_hand:
                        if gesture_state.is_position_locked:
                            gesture_state.is_position_locked = False
                            gesture_state.reset()
                            show_status(image, "Position Unlocked!")

                    # Handle measurements when position is locked
                    if gesture_state.is_position_locked:
                        # Only handle measurements if we don't have both width and height yet
                        if not (gesture_state.width and gesture_state.height):
                            # Track index finger open/close cycles
                            if gesture_state.last_index_state is not None and is_index_open != gesture_state.last_index_state:
                                if not is_index_open:  # Index finger just closed
                                    if not gesture_state.is_measuring:
                                        gesture_state.is_measuring = True
                                        gesture_state.measuring_start_pos = gesture_state.current_finger_pos
                                        show_status(image, "Measuring width...")
                                    else:
                                        if gesture_state.current_side == 0:
                                            gesture_state.width = abs(gesture_state.current_finger_pos[0] - gesture_state.measuring_start_pos[0])
                                            gesture_state.current_side = 1
                                            gesture_state.measuring_start_pos = gesture_state.current_finger_pos
                                            show_status(image, f"Width set: {gesture_state.width}px")
                                        else:
                                            gesture_state.height = abs(gesture_state.current_finger_pos[1] - gesture_state.measuring_start_pos[1])
                                            gesture_state.is_measuring = False
                                            show_status(image, f"Height set: {gesture_state.height}px. Pinch to draw square.")

                            gesture_state.last_index_state = is_index_open

                            # Draw current measurements
                            if gesture_state.is_measuring:
                                if gesture_state.current_side == 0:  # Width
                                    cv2.line(image, 
                                            (gesture_state.measuring_start_pos[0], gesture_state.measuring_start_pos[1]),
                                            (gesture_state.current_finger_pos[0], gesture_state.measuring_start_pos[1]),
                                            (0, 255, 255), 2)
                                    show_status(image, f"Width: {abs(gesture_state.current_finger_pos[0] - gesture_state.measuring_start_pos[0])}px")
                                else:  # Height
                                    cv2.line(image,
                                            (gesture_state.measuring_start_pos[0], gesture_state.measuring_start_pos[1]),
                                            (gesture_state.measuring_start_pos[0], gesture_state.current_finger_pos[1]),
                                            (0, 255, 255), 2)
                                    show_status(image, f"Height: {abs(gesture_state.current_finger_pos[1] - gesture_state.measuring_start_pos[1])}px")

                        # Handle drawing square when measurements are complete
                        if gesture_state.width and gesture_state.height and not gesture_state.is_measuring:
                            current_pinch = is_pinch_gesture(hand_landmarks)
                            
                            if current_pinch:
                                if not gesture_state.was_pinching and not gesture_state.square_drawn:
                                    # Draw square centered at locked position
                                    draw_measured_square(s, 
                                                      gesture_state.locked_position[0],
                                                      gesture_state.locked_position[1],
                                                      gesture_state.width,
                                                      gesture_state.height)
                                    gesture_state.square_drawn = True
                                    show_status(image, "Square drawn! Open hand to reset.")
                                gesture_state.was_pinching = True
                            else:
                                gesture_state.was_pinching = False
                                if not gesture_state.square_drawn:
                                    show_status(image, "Pinch to draw square")

                    # Visualization
                    display_x, display_y = gesture_state.locked_position if gesture_state.is_position_locked else gesture_state.current_finger_pos
                    if gesture_state.is_measuring:
                        cv2.circle(image, (gesture_state.current_finger_pos[0], gesture_state.current_finger_pos[1]), 8, (0, 255, 255), -1)
                        # Draw locked position
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

            cv2.imshow('Hand Tracking', image)
            if cv2.waitKey(1) & 0xFF == 27:
                break

finally:
    timing_file.close()
    cv2.destroyAllWindows()
    s.close()