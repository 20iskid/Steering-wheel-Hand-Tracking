import cv2
import time
import pyvjoy
from threading import Thread
import HandTrackingModule as htm

# === TrackPilot: AI Hand Racing Controller === #

# Initialize vJoy device
j = pyvjoy.VJoyDevice(1)
CENTER = 16384
steering_value = CENTER
race_pressed = False
brake_pressed = False

# Settings
STEERING_SCALE = 60
STEERING_MAX = 32768
RACE_THRESHOLD_ON = 25
RACE_THRESHOLD_OFF = 35
BRAKE_THRESHOLD_ON = 25
BRAKE_THRESHOLD_OFF = 35

# Launch vJoy updater in background thread
def vjoy_loop():
    global steering_value
    while True:
        j.set_axis(pyvjoy.HID_USAGE_X, steering_value)
        time.sleep(0.05)

Thread(target=vjoy_loop, daemon=True).start()

# Camera Setup
wCam, hCam = 640, 360
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(detectionCon=0.8)
pTime = 0

def get_palm_center_y(lmList):
    ids = [0, 6, 10, 14, 19]
    y_vals = [lmList[i][2] for i in ids if i < len(lmList)]
    return sum(y_vals) / len(y_vals) if y_vals else 0

def get_point(lmList, idx):
    return (lmList[idx][1], lmList[idx][2]) if lmList and len(lmList) > idx else None

# Main Loop
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList1 = detector.findPosition(img, handNo=0)
    lmList2 = detector.findPosition(img, handNo=1)

    # Gesture Controls: Race & Brake
    if lmList1 and lmList2 and len(lmList1) > 8 and len(lmList2) > 8:
        # Right hand → Race
        race_diff = abs(lmList2[6][1] - lmList2[4][1])
        # Left hand → Brake
        brake_diff = abs(lmList1[6][2] - lmList1[4][2])

        if race_diff < RACE_THRESHOLD_ON and not race_pressed:
            j.set_button(1, 1)
            race_pressed = True
        elif race_diff > RACE_THRESHOLD_OFF and race_pressed:
            j.set_button(1, 0)
            race_pressed = False

        if brake_diff < BRAKE_THRESHOLD_ON and not brake_pressed:
            j.set_button(3, 1)
            brake_pressed = True
        elif brake_diff > BRAKE_THRESHOLD_OFF and brake_pressed:
            j.set_button(3, 0)
            brake_pressed = False
    else:
        # Reset if hands are lost
        j.set_button(1, 0)
        j.set_button(3, 0)
        race_pressed = False
        brake_pressed = False

    # Steering Logic
    if lmList1 and lmList2:
        y1 = get_palm_center_y(lmList1)
        y2 = get_palm_center_y(lmList2)
        diff = y1 - y2
        raw_value = CENTER + int(diff * STEERING_SCALE)
        steering_value = max(0, min(STEERING_MAX, raw_value))

        # Visual Feedback
        direction = "Right" if diff > 30 else "Left" if diff < -30 else "Center"
        cv2.putText(img, f"Steering: {direction}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Debug visuals: Hand connection lines
    p1 = get_point(lmList1, 20)
    p2 = get_point(lmList2, 4)
    p3 = get_point(lmList1, 4)
    p4 = get_point(lmList2, 20)
    if p1 and p2:
        cv2.line(img, p1, p2, (255, 100, 50), 2)
    if p3 and p4:
        cv2.line(img, p3, p4, (255, 100, 50), 2)

    # FPS Counter
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Show Output
    cv2.imshow("TrackPilot - AI Racing Controller", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
