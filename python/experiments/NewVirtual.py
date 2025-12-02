import depthai as dai
import cv2
import mediapipe as mp
import socket
import struct
import csv
import time
import math
import numpy as np


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)
mp_draw = mp.solutions.drawing_utils

# ==================== SMOOTHING OPTIONS ====================
# Original smoothing method
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

# Enhanced Moving Average Filter
class EnhancedSmoothing:
    def __init__(self, window_size=10, position_weight=0.7, velocity_weight=0.3):
        self.window_size = window_size
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.position_history_x = []
        self.position_history_y = []
        self.last_velocity_x = 0
        self.last_velocity_y = 0
        self.prev_x = None
        self.prev_y = None

    def smooth(self, x, y):
        # Initialize if this is the first point
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y
            self.position_history_x = [x] * self.window_size
            self.position_history_y = [y] * self.window_size
            return x, y

        # Calculate current velocity
        velocity_x = x - self.prev_x
        velocity_y = y - self.prev_y

        # Smooth velocity with previous velocity
        smooth_velocity_x = velocity_x * (1 - self.velocity_weight) + self.last_velocity_x * self.velocity_weight
        smooth_velocity_y = velocity_y * (1 - self.velocity_weight) + self.last_velocity_y * self.velocity_weight

        # Update position history
        self.position_history_x.append(x)
        self.position_history_y.append(y)
        if len(self.position_history_x) > self.window_size:
            self.position_history_x.pop(0)
            self.position_history_y.pop(0)

        # Calculate average position
        avg_x = sum(self.position_history_x) / len(self.position_history_x)
        avg_y = sum(self.position_history_y) / len(self.position_history_y)

        # Combine position and velocity information
        predicted_x = avg_x * self.position_weight + (self.prev_x + smooth_velocity_x) * (1 - self.position_weight)
        predicted_y = avg_y * self.position_weight + (self.prev_y + smooth_velocity_y) * (1 - self.position_weight)

        # Update state for next iteration
        self.prev_x, self.prev_y = predicted_x, predicted_y
        self.last_velocity_x, self.last_velocity_y = smooth_velocity_x, smooth_velocity_y

        return int(predicted_x), int(predicted_y)

# One Euro Filter (great for reducing jitter while preserving responsiveness)
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_previous = None
        self.dx_previous = 0
        self.t_previous = None
        self.y_previous = None
        self.dy_previous = 0

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_previous):
        return a * x + (1 - a) * x_previous

    def smooth(self, x, y, timestamp=None):
        if timestamp is None:
            timestamp = time.perf_counter()

        # Initialize on first call
        if self.x_previous is None:
            self.x_previous = x
            self.y_previous = y
            self.t_previous = timestamp
            return x, y

        # Time delta
        t_e = timestamp - self.t_previous
        if t_e <= 0:
            t_e = 1/30  # Assume 30fps if timestamps are invalid

        # Calculate new derivatives
        dx = (x - self.x_previous) / t_e
        dy = (y - self.y_previous) / t_e

        # Apply smoothing to derivatives
        edx = self.exponential_smoothing(
            self.smoothing_factor(t_e, self.d_cutoff), dx, self.dx_previous)
        edy = self.exponential_smoothing(
            self.smoothing_factor(t_e, self.d_cutoff), dy, self.dy_previous)

        # Calculate cutoff frequency with dynamic component
        cutoff = self.min_cutoff + self.beta * abs(edx)
        cutoff_y = self.min_cutoff + self.beta * abs(edy)

        # Apply smoothing to position
        x_hat = self.exponential_smoothing(
            self.smoothing_factor(t_e, cutoff), x, self.x_previous)
        y_hat = self.exponential_smoothing(
            self.smoothing_factor(t_e, cutoff_y), y, self.y_previous)

        # Update state
        self.x_previous = x_hat
        self.y_previous = y_hat
        self.dx_previous = edx
        self.dy_previous = edy
        self.t_previous = timestamp

        return int(x_hat), int(y_hat)

# Kalman Filter
class KalmanFilter:
    def __init__(self, process_noise=0.001, measurement_noise=0.1):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.posX = 0
        self.posY = 0
        self.errX = 1.0
        self.errY = 1.0
        self.initialized = False

    def smooth(self, x, y):
        if not self.initialized:
            self.posX = x
            self.posY = y
            self.initialized = True
            return int(x), int(y)

        # Prediction step
        p_errX = self.errX + self.process_noise
        p_errY = self.errY + self.process_noise

        # Update step
        K_x = p_errX / (p_errX + self.measurement_noise)
        K_y = p_errY / (p_errY + self.measurement_noise)

        self.posX = self.posX + K_x * (x - self.posX)
        self.posY = self.posY + K_y * (y - self.posY)

        self.errX = (1 - K_x) * p_errX
        self.errY = (1 - K_y) * p_errY

        return int(self.posX), int(self.posY)

# Double exponential smoothing
class DoubleExponentialSmoothing:
    def __init__(self, alpha=0.6, beta=0.4):
        self.alpha = alpha  # Level smoothing factor
        self.beta = beta    # Trend smoothing factor
        self.level_x = None
        self.trend_x = None
        self.level_y = None
        self.trend_y = None

    def smooth(self, x, y):
        # Initialize on first call
        if self.level_x is None:
            self.level_x = x
            self.trend_x = 0
            self.level_y = y
            self.trend_y = 0
            return int(x), int(y)

        # Save previous values
        prev_level_x = self.level_x
        prev_level_y = self.level_y

        # Update level and trend
        self.level_x = self.alpha * x + (1 - self.alpha) * (prev_level_x + self.trend_x)
        self.trend_x = self.beta * (self.level_x - prev_level_x) + (1 - self.beta) * self.trend_x

        self.level_y = self.alpha * y + (1 - self.alpha) * (prev_level_y + self.trend_y)
        self.trend_y = self.beta * (self.level_y - prev_level_y) + (1 - self.beta) * self.trend_y

        return int(self.level_x), int(self.level_y)

# Median Filter - good for removing outliers
class MedianFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.x_buffer = []
        self.y_buffer = []

    def smooth(self, x, y):
        # Add new position to buffer
        self.x_buffer.append(x)
        self.y_buffer.append(y)

        # Keep buffer at window size
        if len(self.x_buffer) > self.window_size:
            self.x_buffer.pop(0)
            self.y_buffer.pop(0)

        # Take median of buffer
        x_sorted = sorted(self.x_buffer)
        y_sorted = sorted(self.y_buffer)

        x_median = x_sorted[len(self.x_buffer) // 2]
        y_median = y_sorted[len(self.y_buffer) // 2]

        return int(x_median), int(y_median)

# ==================== GESTURE STATE CLASS ====================
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
        # New variable to disable position locking
        self.locking_enabled = True
        self.last_thumbs_up_state = False
        # Add these new variables for guided cursor mode
        self.guided_mode_active = False
        self.guided_cursor_pos = None
        self.guided_cursor_index = 0

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

# ==================== HAND GESTURE FUNCTIONS ====================
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
    # Thumb points
    thumb_tip = landmarks.landmark[4]
    thumb_ip = landmarks.landmark[3]
    thumb_mcp = landmarks.landmark[2]
    wrist = landmarks.landmark[0]

    # Check if thumb is extended upward (lower y value means higher up in image)
    thumb_extended_up = (thumb_tip.y < thumb_ip.y < thumb_mcp.y) and (thumb_tip.y < wrist.y - 0.1)

    # Check if thumb is relatively vertical (less horizontal movement)
    thumb_vertical = abs(thumb_tip.x - thumb_mcp.x) < 0.1

    # Check other fingers are curved
    index_pip = landmarks.landmark[6]
    index_tip = landmarks.landmark[8]
    middle_pip = landmarks.landmark[10]
    middle_tip = landmarks.landmark[12]

    # More pronounced bending of other fingers (they should be curled toward palm)
    fingers_curled = (index_tip.y > index_pip.y) and (middle_tip.y > middle_pip.y)

    # For a thumbs up, the thumb should be clearly above other finger tips
    thumb_above_fingers = (thumb_tip.y < index_tip.y - 0.05) and (thumb_tip.y < middle_tip.y - 0.05)

    return thumb_extended_up and thumb_vertical and fingers_curled and thumb_above_fingers

def is_pinch_gesture(landmarks):
    thumb_tip = landmarks.landmark[4]
    index_tip = landmarks.landmark[8]

    distance = ((thumb_tip.x - index_tip.x)**2 +
                (thumb_tip.y - index_tip.y)**2 +
                (thumb_tip.z - index_tip.z)**2)**0.5

    return distance < 0.05

def get_hand_orientation(landmarks):
    # Get index finger joints
    index_tip = landmarks.landmark[8]
    index_pip = landmarks.landmark[6]

    # Check if hand is facing front or back based on Z coordinate
    # Negative Z means finger is pointing towards camera (front)
    # Positive Z means finger is pointing away from camera (back)
    return "front" if index_tip.z < index_pip.z else "back"

def draw_measured_square(s, center_x, center_y, width, height):
    try:
        # Clear any pending states first
        s.sendall(b'release')
        time.sleep(0.1)  # Small delay to ensure clean state

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

def create_virtual_pattern(image, pattern_type="sine", scale=100, color=(0, 255, 255), thickness=2):
    """
    Creates a virtual pattern overlay on the image

    Parameters:
    - image: The camera frame to draw on
    - pattern_type: Type of pattern ("sine", "square", "circle", "grid", "text", "target")
    - scale: Size scaling of the pattern
    - color: BGR color tuple
    - thickness: Line thickness

    Returns:
    - image with pattern overlay
    - list of key points along the pattern (for accuracy measurement)
    """
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    key_points = []
    all_points = []  # Add this line to store all points of the pattern

    if pattern_type == "sine":
        # Create a sine wave across the middle of the screen
        points = []
        for x in range(w//4, 3*w//4):
            # Calculate sine wave
            y = int(center_y + scale * math.sin((x - w//4) * 0.02))
            points.append((x, y))
            # Store key points at peaks and troughs
            if x % 50 == 0:
                key_points.append((x, y))
                all_points.append((x, y))
            else:
                all_points.append((x,y))

        # Draw the sine wave
        for i in range(1, len(points)):
            cv2.line(image, points[i-1], points[i], color, thickness)

        # Add scientific labels
        cv2.putText(image, "Sinusoidal Waveform", (w//4, center_y - scale - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(image, "f(x) = A·sin(ωx)", (w//4, center_y - scale - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Add amplitude markers
        cv2.line(image, (w//4 + 100, center_y), (w//4 + 100, center_y + scale), color, 1)
        cv2.putText(image, f"A={scale}px", (w//4 + 105, center_y + scale//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    elif pattern_type == "square":
        side = scale * 2
        top_left = (center_x - side//2, center_y - side//2)
        bottom_right = (top_left[0] + side, top_left[1] + side)
        num_samples = 20  # Increase for smoother motion

        # Generate points along the four edges
        points = []
        # Top edge (left to right)
        for i in range(num_samples):
            x = top_left[0] + i * (side / (num_samples - 1))
            y = top_left[1]
            points.append((int(x), int(y)))
        # Right edge (top to bottom)
        for i in range(num_samples):
            x = bottom_right[0]
            y = top_left[1] + i * (side / (num_samples - 1))
            points.append((int(x), int(y)))
        # Bottom edge (right to left)
        for i in range(num_samples):
            x = bottom_right[0] - i * (side / (num_samples - 1))
            y = bottom_right[1]
            points.append((int(x), int(y)))
        # Left edge (bottom to top)
        for i in range(num_samples):
            x = top_left[0]
            y = bottom_right[1] - i * (side / (num_samples - 1))
            points.append((int(x), int(y)))

        key_points = points[:]  # you can choose key_points as a subset if desired
        all_points = points[:]

        # Draw the square outline using the dense list
        cv2.polylines(image, [np.array(points)], isClosed=True, color=color, thickness=thickness)


    elif pattern_type == "circle":
        radius = scale
        points = []
        for angle in range(0, 360, 5):  # 5° steps for smoother curves
            rad = math.radians(angle)
            x = int(center_x + radius * math.cos(rad))
            y = int(center_y + radius * math.sin(rad))
            points.append((x, y))
            cv2.circle(image, (x, y), 3, color, -1)  # draw the points if desired
        key_points = points[:]   # or select a subset for specific key markers
        all_points = points[:]
        # Optionally, draw the continuous circle
        cv2.polylines(image, [np.array(points)], isClosed=True, color=color, thickness=thickness)
        cv2.putText(image, f"Circle: r={radius}px", (center_x - radius, center_y - radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


    elif pattern_type == "grid":
        # Create a grid for precision tests
        grid_size = scale // 5
        grid_cols = 5
        grid_rows = 5

        start_x = center_x - (grid_size * grid_cols) // 2
        start_y = center_y - (grid_size * grid_rows) // 2

        # Draw horizontal lines
        for i in range(grid_rows + 1):
            y = start_y + i * grid_size
            cv2.line(image, (start_x, y), (start_x + grid_cols * grid_size, y), color, thickness)

        # Draw vertical lines
        for i in range(grid_cols + 1):
            x = start_x + i * grid_size
            cv2.line(image, (x, start_y), (x, start_y + grid_rows * grid_size), color, thickness)

        # Add calibration points at intersections
        for row in range(grid_rows + 1):
            for col in range(grid_cols + 1):
                x = start_x + col * grid_size
                y = start_y + row * grid_size
                cv2.circle(image, (x, y), 3, color, -1)
                key_points.append((x, y))
                all_points.append((x, y))

        # Label
        cv2.putText(image, f"Calibration Grid ({grid_cols}x{grid_rows})",
                    (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    elif pattern_type == "text":
        # Scientific formula to trace
        formulas = [
            "NU",
        ]
        all_points = []
        key_points = []
        # Draw each formula at a different position
        y_offset = center_y - 100
        font_scale = 15.5 # Increased from 1.5 to 5.0 for larger text
        thickness = 3  # Set a moderate thickness
        for formula in formulas:
            # Get text size to center it
            text_size = cv2.getTextSize(formula, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = center_x - text_size[0] // 2  # Add this line
            text_y = y_offset + text_size[1]

            # Draw the formula
            cv2.putText(image, formula, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.putText(mask, formula, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, thickness)

            # Find contours on the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
                combined_points = []
                for cnt in contours:
                    cnt = cnt.squeeze()
                    if cnt.ndim == 1:
                        cnt = np.expand_dims(cnt, 0)
                    for pt in cnt:
                        combined_points.append(tuple(pt))
                all_points.extend(combined_points)
                step = max(1, len(combined_points) // 20)
                key_points = combined_points[::step]
            y_offset += text_size[1] + 60
            # points = np.column_stack(np.where(mask.T > 0))
            # points = [(p[1], p[0]) for p in points]
            # points.sort(key=lambda p: (p[1], p[0]))
            #
            # all_points.extend(points)
            # key_points = points[::20]
            #
            # y_offset += text_size[1] + 20  # Move down for next formula if there were multiple
            #
            # # Add key points along the text
            # for i in range(len(formula)):
            #     char_size = cv2.getTextSize(formula[:i+1], cv2.FONT_HERSHEY_SIMPLEX, 1.5, thickness)[0]
            #     point_x = text_x + char_size[0]
            #     key_points.append((point_x, y_offset))
            #     all_points.append((point_x, y_offset))
            # y_offset += 60

    elif pattern_type == "target":
        # Create a target with concentric circles
        for r in range(scale, 0, -scale//4):
            circle_color = (0, 0, 255) if r == scale else color
            cv2.circle(image, (center_x, center_y), r, circle_color, thickness if r == scale else 1)
            key_points.append((center_x + r, center_y))  # Right point

        # Add crosshair
        cv2.line(image, (center_x - scale, center_y), (center_x + scale, center_y), color, 1)
        cv2.line(image, (center_x, center_y - scale), (center_x, center_y + scale), color, 1)

        # Add bullseye
        cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
        key_points.append((center_x, center_y))
        all_points.append((center_x+r, center_y))
        all_points.append((center_x, center_y))
        # Label
        cv2.putText(image, "Precision Target", (center_x - scale, center_y - scale - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Add pattern selector instructions
    cv2.putText(image, "Press 1-6 to change patterns", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return image, key_points, all_points

def calculate_accuracy(current_pos, key_points, max_distance=30):
    """
    Calculate the accuracy of finger tracing against key points

    Parameters:
    - current_pos: Current finger position (x, y)
    - key_points: List of key points along the pattern
    - max_distance: Maximum distance to consider a point "reached"

    Returns:
    - closest_point: The nearest key point
    - distance: Distance to closest point
    - reached: Whether the point was successfully reached
    """
    if not key_points:
        return None, float('inf'), False

    closest_point = None
    min_distance = float('inf')

    for point in key_points:
        distance = math.sqrt((point[0] - current_pos[0])**2 + (point[1] - current_pos[1])**2)
        if distance < min_distance:
            min_distance = distance
            closest_point = point

    return closest_point, min_distance, min_distance <= max_distance

def find_closest_point_index(pattern_points, finger_pos, start_idx=0):
    """
    Find the index of the closest point in the pattern to the finger position

    Parameters:
    - pattern_points: List of points defining the pattern
    - finger_pos: Current finger position (x, y)
    - start_idx: Index to start searching from (for optimization)

    Returns:
    - Index of the closest point
    """
    if not pattern_points:
        return 0

    min_distance = float('inf')
    closest_idx = start_idx

    # Look ahead first (from start_idx to end)
    for i in range(start_idx, len(pattern_points)):
        point = pattern_points[i]
        distance = math.sqrt((point[0] - finger_pos[0])**2 + (point[1] - finger_pos[1])**2)
        if distance < min_distance:
            min_distance = distance
            closest_idx = i

    # Then look behind if needed (from 0 to start_idx)
    for i in range(0, start_idx):
        point = pattern_points[i]
        distance = math.sqrt((point[0] - finger_pos[0])**2 + (point[1] - finger_pos[1])**2)
        if distance < min_distance:
            min_distance = distance
            closest_idx = i

    return closest_idx

def guided_cursor_mode(pattern_points, finger_pos, current_index=0):
    if not pattern_points or len(pattern_points) < 2:
        return finger_pos, current_index  # fallback to finger position if not enough points

    best_dist = float('inf')
    best_point = None
    best_seg_index = current_index

    # Loop through all segments of the pattern
    for i in range(len(pattern_points) - 1):
        A = pattern_points[i]
        B = pattern_points[i+1]
        # Compute the vector from A to B
        AB = (B[0] - A[0], B[1] - A[1])
        # Vector from A to finger position
        AP = (finger_pos[0] - A[0], finger_pos[1] - A[1])

        # Compute projection scalar t (clamped between 0 and 1)
        norm_AB_sq = AB[0]**2 + AB[1]**2
        if norm_AB_sq == 0:
            continue
        t = (AP[0]*AB[0] + AP[1]*AB[1]) / norm_AB_sq
        t = max(0.0, min(1.0, t))

        # Compute the point on the segment corresponding to t
        P = (A[0] + t * AB[0], A[1] + t * AB[1])

        # Compute the distance from finger_pos to the projected point P
        dist = math.hypot(finger_pos[0] - P[0], finger_pos[1] - P[1])
        if dist < best_dist:
            best_dist = dist
            best_point = P
            best_seg_index = i

    # Optionally update current_index if needed
    return best_point, best_seg_index


# ==================== MAIN APPLICATION ====================
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
gesture_state = GestureState()

# CHOOSE ONE SMOOTHING METHOD BY UNCOMMENTING IT:

# OPTION 1: Original smoothing (try with higher smoothing factor)
smoother = CoordinateSmoothing(smoothing_factor=0.7)  # Try values between 0.6-0.9

# OPTION 2: Enhanced Moving Average with window and velocity
# smoother = EnhancedSmoothing(window_size=15, position_weight=0.8, velocity_weight=0.5)

# OPTION 3: One Euro Filter - good balance between smoothness and responsiveness
# smoother = OneEuroFilter(min_cutoff=0.5, beta=0.01, d_cutoff=1.0)

# OPTION 4: Kalman Filter - statistical approach
# smoother = KalmanFilter(process_noise=0.001, measurement_noise=0.1)

# OPTION 5: Double Exponential Smoothing - good for predicting movement
# smoother = DoubleExponentialSmoothing(alpha=0.7, beta=0.3)

# OPTION 6: Median Filter - good for removing outliers
# smoother = MedianFilter(window_size=5)

# Display active smoothing method
active_method = type(smoother).__name__
print(f"Active smoothing method: {active_method}")

# Open timing data log file
timing_file = open('tracking_analysis.csv', 'w', newline='')
timing_writer = csv.writer(timing_file)
timing_writer.writerow([
    "Timestamp", "Frame_Time", "Detection_Time", "Processing_Time",
    "Finger_X", "Finger_Y", "Smoothed_X", "Smoothed_Y", "Smoothing_Method",
    "Is_Measuring", "Measured_Width", "Measured_Height", "Is_Drawing_Square",
    "Square_Center_X", "Square_Center_Y", "Is_Position_Locked", "Locked_X", "Locked_Y"
])

# Virtual pattern control
current_pattern = "sine"
pattern_scale = 200
pattern_thickness = 8  # New variable for thickness
pattern_color = (0, 255, 255)  # Yellow by default
key_points = []
reached_points = set()
accuracy_score = 0

try:
    with dai.Device(pipeline) as device:
        q_video = device.getOutputQueue(name="video", maxSize=4, blocking=False)

        while True:
            frame_start_time = time.perf_counter()

            in_video = q_video.get()
            frame = in_video.getCvFrame()
            image = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Generate virtual pattern overlay
            image, key_points, all_pattern_points = create_virtual_pattern(image, pattern_type=current_pattern,
                                                                           scale=pattern_scale, color=pattern_color, thickness=pattern_thickness)

            detection_start_time = time.perf_counter()
            results = hands.process(image_rgb)
            detection_time = time.perf_counter() - detection_start_time

            # Show active smoothing method on screen
            show_status(image, f"Method: {active_method}", position=(10, 90), color=(0, 255, 0))

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                # Get current finger position
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                current_x = int(index_tip.x * image.shape[1])
                current_y = int(index_tip.y * image.shape[0])

                # Apply smoothing based on active method
                if isinstance(smoother, OneEuroFilter):
                    smooth_x, smooth_y = smoother.smooth(current_x, current_y, time.perf_counter())
                else:
                    smooth_x, smooth_y = smoother.smooth(current_x, current_y)

                # Update current finger position with smoothed coordinates
                gesture_state.current_finger_pos = (smooth_x, smooth_y)

                # Check hand gestures
                is_fist_gesture = is_fist(hand_landmarks)
                is_open_hand = is_hand_fully_open(hand_landmarks)
                is_index_open = is_index_finger_open(hand_landmarks)
                current_thumbs_up = is_thumbs_up(hand_landmarks)
                hand_orientation = get_hand_orientation(hand_landmarks)

                # Show raw vs smoothed coordinates for debugging
                show_status(image, f"Raw: ({current_x}, {current_y})", position=(10, 50), color=(255, 0, 0))
                show_status(image, f"Smooth: ({smooth_x}, {smooth_y})", position=(10, 130), color=(0, 255, 0))
                show_status(image, f"Orientation: {hand_orientation}", position=(10, 170), color=(255, 255, 0))

                # Calculate accuracy against virtual pattern
                if key_points:
                    closest_point, distance, point_reached = calculate_accuracy(
                        gesture_state.current_finger_pos, key_points)

                    if closest_point and point_reached and tuple(closest_point) not in reached_points:
                        reached_points.add(tuple(closest_point))
                        accuracy_score += 1

                    # Show accuracy metrics
                    point_count = len(key_points)
                    reached_count = len(reached_points)
                    accuracy_percentage = int((reached_count / max(1, point_count)) * 100)
                    show_status(image, f"Accuracy: {accuracy_percentage}% ({reached_count}/{point_count})",
                                position=(10, 250), color=(255, 255, 0))

                    # Highlight closest point
                    if closest_point:
                        cv2.circle(image, closest_point, 8, (0, 165, 255), -1)

                        # Show distance to nearest point
                        distance_text = f"Distance: {distance:.1f}px"
                        show_status(image, distance_text, position=(10, 290), color=(0, 165, 255))

                # Visualization
                display_x, display_y = gesture_state.locked_position if gesture_state.is_position_locked else gesture_state.current_finger_pos
                if gesture_state.is_measuring:
                    cv2.circle(image, (gesture_state.current_finger_pos[0], gesture_state.current_finger_pos[1]), 8, (0, 255, 255), -1)
                    # Draw locked position
                    if gesture_state.locked_position:
                        cv2.circle(image, (gesture_state.locked_position[0], gesture_state.locked_position[1]), 8, (255, 0, 0), -1)
                else:
                    circle_color = (255, 0, 0) if gesture_state.is_position_locked else (0, 255, 0)
                    cv2.circle(image, (display_x, display_y), 8, circle_color, -1)

                # Draw raw position (red) and smoothed position (green)
                cv2.circle(image, (current_x, current_y), 5, (0, 0, 255), -1)  # Raw in red
                cv2.circle(image, (smooth_x, smooth_y), 8, (0, 255, 0), -1)    # Smoothed in green

                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Process guided cursor mode if active and we have pattern points
            if gesture_state.guided_mode_active and all_pattern_points:
                if gesture_state.current_finger_pos:
                    guided_pos, guided_idx = guided_cursor_mode(
                        all_pattern_points,
                        gesture_state.current_finger_pos,
                        current_index=gesture_state.guided_cursor_index
                    )
                    gesture_state.guided_cursor_pos = guided_pos
                    gesture_state.guided_cursor_index = guided_idx

                    # Draw guided cursor
                    if gesture_state.guided_cursor_pos is not None:
                        cursor = (int(gesture_state.guided_cursor_pos[0]), int(gesture_state.guided_cursor_pos[1]))
                        cv2.circle(image, cursor, 12, (0, 0, 255), -1)
                    # cv2.circle(image, gesture_state.guided_cursor_pos, 8, (255, 255, 255), -1)

                    # Use guided position for drawing
                    if not gesture_state.is_measuring:
                        send_time = time.perf_counter()
                        if gesture_state.guided_mode_active and gesture_state.guided_cursor_pos:
                            send_x, send_y = gesture_state.guided_cursor_pos
                        else:
                            send_x, send_y = gesture_state.locked_position if gesture_state.is_position_locked else gesture_state.current_finger_pos
                        s.sendall(struct.pack('dd', send_x, send_y))


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
                        current_x,
                        current_y,
                        smooth_x,
                        smooth_y,
                        active_method,
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

                # ----- LOCKING STATUS DISPLAY COMMENTED OUT -----
                # if not gesture_state.locking_enabled:
                #     show_status(image, "Locking Disabled", position=(10, 210), color=(0, 255, 0))
                # ----- END OF LOCKING STATUS DISPLAY -----



            cv2.imshow('Hand Tracking', image)

            # Replace the existing keyboard check with this expanded version
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('1'):
                current_pattern = "sine"
                gesture_state.guided_cursor_index = 0  # Reset guided index
                print("Changed pattern to sine wave")
            elif key == ord('2'):
                current_pattern = "square"
                gesture_state.guided_cursor_index = 0  # Reset guided index
                print("Changed pattern to square")
            elif key == ord('3'):
                current_pattern = "circle"
                gesture_state.guided_cursor_index = 0  # Reset guided index
                print("Changed pattern to circle")
            elif key == ord('4'):
                current_pattern = "grid"
                gesture_state.guided_cursor_index = 0  # Reset guided index
                print("Changed pattern to grid")
            elif key == ord('5'):
                current_pattern = "text"
                gesture_state.guided_cursor_index = 0  # Reset guided index
                print("Changed pattern to text")
            elif key == ord('6'):
                current_pattern = "target"
                gesture_state.guided_cursor_index = 0  # Reset guided index
                print("Changed pattern to target")
            elif key == ord('+') or key == ord('='):
                pattern_scale += 20
                print(f"Increased pattern scale to {pattern_scale}")
            elif key == ord('-'):
                pattern_scale = max(50, pattern_scale - 20)
                print(f"Decreased pattern scale to {pattern_scale}")
            elif key == ord('t'):
                pattern_color = (pattern_color[2], pattern_color[0], pattern_color[1])  # Cycle colors
                print(f"Changed pattern color")
            elif key == ord('g'):
                gesture_state.guided_mode_active = not  gesture_state.guided_mode_active
                print(f"Guided cursor mode {'enabled' if gesture_state.guided_mode_active else 'disabled'}")

finally:
    timing_file.close()
    cv2.destroyAllWindows()
    s.close()