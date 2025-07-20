import cv2
import time
from threading import Thread
import pyvjoy
import numpy as np
import math
import mediapipe as mp

# === OneEuroFilter ===
class OneEuroFilter:
    def __init__(self, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.last_time = None

    def alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x

        dx = (x - self.x_prev) * self.freq
        alpha_d = self.alpha(self.d_cutoff)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self.alpha(cutoff)
        x_hat = alpha * x + (1 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return x_hat

# === VJoy ===
j = pyvjoy.VJoyDevice(1)
CENTER = 16384
steering_value = CENTER

# Buttons
race_pressed = False
brake_pressed = False

RACE_PRESS_THRESHOLD = 20
RACE_RELEASE_THRESHOLD = 28
BRAKE_PRESS_THRESHOLD = 30
BRAKE_RELEASE_THRESHOLD = 45

# Steering config
ANGLE_RANGE = 45  # degrees for full lock
filter = OneEuroFilter(freq=60, min_cutoff=1.0, beta=0.02)

# Calibration baseline
baseline_angle = 0.0

# === VJoy updater ===
def vjoy_updater():
    global steering_value
    while True:
        j.set_axis(pyvjoy.HID_USAGE_X, steering_value)
        time.sleep(0.01)

Thread(target=vjoy_updater, daemon=True).start()

# === Mediapipe ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# === Camera ===
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
cap.set(cv2.CAP_PROP_FPS, 60)

pTime = 0

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results_pose = pose.process(img_rgb)
    results_hands = hands.process(img_rgb)

    left_shoulder = None
    right_shoulder = None
    left_wrist = None
    right_wrist = None

    if results_pose.pose_landmarks:
        h, w, _ = img.shape
        left_shoulder = (
            int(results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
            int(results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)
        )
        right_shoulder = (
            int(results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
            int(results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)
        )

    if results_hands.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
            label = handedness.classification[0].label
            cx = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * wCam)
            cy = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * hCam)
            if label == "Left":
                left_wrist = (cx, cy)
            else:
                right_wrist = (cx, cy)

    # === Throttle ===
    if results_hands.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
            label = handedness.classification[0].label
            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            dist = math.hypot(
                (index.x - thumb.x) * wCam,
                (index.y - thumb.y) * hCam
            )
            if label == "Right":
                if dist < RACE_PRESS_THRESHOLD and not race_pressed:
                    j.set_button(1, 1)
                    race_pressed = True
                elif dist > RACE_RELEASE_THRESHOLD and race_pressed:
                    j.set_button(1, 0)
                    race_pressed = False
            elif label == "Left":
                if dist < BRAKE_PRESS_THRESHOLD and not brake_pressed:
                    j.set_button(3, 0)
                    brake_pressed = True
                elif dist > BRAKE_RELEASE_THRESHOLD and brake_pressed:
                    j.set_button(3, 1)
                    brake_pressed = False

    # === Steering: shoulder line ===
    if left_shoulder and right_shoulder and left_wrist and right_wrist:
        # Shoulder vector
        dx_shoulder = right_shoulder[0] - left_shoulder[0]
        dy_shoulder = right_shoulder[1] - left_shoulder[1]
        shoulder_angle = math.degrees(math.atan2(dy_shoulder, dx_shoulder))

        # Wrist vector
        dx_wrist = right_wrist[0] - left_wrist[0]
        dy_wrist = right_wrist[1] - left_wrist[1]
        wrist_angle = math.degrees(math.atan2(dy_wrist, dx_wrist))

        relative_angle = wrist_angle - shoulder_angle - baseline_angle
        relative_angle = np.clip(relative_angle, -ANGLE_RANGE, ANGLE_RANGE)

        filtered_angle = filter.filter(relative_angle)
        normalized = np.clip(filtered_angle / ANGLE_RANGE, -1.0, 1.0)

        target_steering = int(CENTER + normalized * (32768 // 2))
        steering_value = target_steering

        # === Overlay ===
        bar_length = 300
        bar_center = wCam // 2
        bar_x = int(bar_center + normalized * (bar_length // 2))
        cv2.rectangle(img, (bar_center - bar_length // 2, 30), (bar_center + bar_length // 2, 60), (50, 50, 50), -1)
        cv2.rectangle(img, (bar_center - 2, 30), (bar_center + 2, 60), (255, 255, 255), -1)
        cv2.rectangle(img, (bar_x - 5, 30), (bar_x + 5, 60), (0, 255, 255), -1)

        cv2.putText(img, f"Angle: {filtered_angle:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, f"Norm: {normalized:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # === Calibrate baseline ===
    key = cv2.waitKey(1)
    if key & 0xFF == ord('c'):
        baseline_angle = wrist_angle - shoulder_angle
        print(f"Calibrated! Baseline: {baseline_angle:.2f}")

    if key & 0xFF == ord('q'):
        break

    # === FPS ===
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Hand Steering", img)

cap.release()
cv2.destroyAllWindows()
