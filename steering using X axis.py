import cv2
import time
from threading import Thread
import pyvjoy
import HandTrackingModule as htm

# === vJoy setup ===
j = pyvjoy.VJoyDevice(1)
center = 16384
steering_value = center  # Global steering value

# === Background thread to send vJoy input ===
def vjoy_updater():
    global steering_value
    while True:
        j.set_axis(pyvjoy.HID_USAGE_X, steering_value)
        time.sleep(0.05)  # 20 Hz update rate

Thread(target=vjoy_updater, daemon=True).start()

# === Camera setup ===
wCam, hCam = 640, 360
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# === Mediapipe Hand Detector ===
detector = htm.handDetector(detectionCon=0.8)
pTime = 0

# === Main loop ===
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList1 = detector.findPosition(img, handNo=0)
    lmList2 = detector.findPosition(img, handNo=1)

    def get_point(lmList, idx):
        if lmList and len(lmList) > idx:
            return (lmList[idx][1], lmList[idx][2])
        return None

    # Draw cross lines
    p1 = get_point(lmList1, 20)
    p2 = get_point(lmList2, 4)
    p3 = get_point(lmList1, 4)
    p4 = get_point(lmList2, 20)

    if p1 and p2:
        cv2.line(img, p1, p2, (0, 150, 255), 2)
    if p3 and p4:
        cv2.line(img, p3, p4, (0, 150, 255), 2)

    # === Accelerator (Race) & Brake Logic ===
    if lmList1 and lmList2 and len(lmList1) > 6 and len(lmList2) > 6:
        r1, r2 = lmList2[4][1], lmList2[6][1]
        b1, b2 = lmList1[4][2], lmList1[6][2]

        race = (r2 - r1)
        brake = (b2 - b1)

        # Accelerator
        j.set_button(1, 1 if race < 15 else 0)

        # Brake
        j.set_button(2, 1 if brake < 15 else 0)

    # === Stable Steering Based on Wrist X-Axis ===
    if lmList1 and lmList2 and len(lmList1) > 0 and len(lmList2) > 0:
        x1 = lmList1[0][1]  # Left wrist X
        x2 = lmList2[0][1]  # Right wrist X
        diff = x2 - x1      # Positive = turning right, Negative = turning left

        sensitivity = 35    # Increase to make steering more sensitive
        raw_value = center + int(diff * sensitivity)
        steering_value = max(0, min(32768, raw_value))  # Clamp

        # Visual debug
        direction = "Right" if diff > 20 else "Left" if diff < -20 else "Center"
        cv2.putText(img, f"Steering: {direction}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # === FPS Display ===
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime
    cv2.putText(img, f'fps: {int(fps)}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Hand Steering", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
