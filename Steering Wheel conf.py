import cv2
import time
from threading import Thread
import pyvjoy
import HandTrackingModule as htm



j = pyvjoy.VJoyDevice(1)
center = 16384
steering_value = center


race_pressed = False
brake_pressed = False


def vjoy_updater():
    global steering_value
    while True:
        j.set_axis(pyvjoy.HID_USAGE_X, steering_value)
        time.sleep(0.05)


Thread(target=vjoy_updater, daemon=True).start()


wCam, hCam = 640, 360
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


detector = htm.handDetector(detectionCon=0.8)
pTime = 0


def palm_center_y(lmList):
    ids = [0, 6, 10, 14, 19]
    y_vals = [lmList[i][2] for i in ids if i < len(lmList)]
    return sum(y_vals) / len(y_vals) if y_vals else 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)

    lmList1 = detector.findPosition(img, handNo=0)
    lmList2 = detector.findPosition(img, handNo=1)


    if lmList1 and lmList2 and len(lmList1) > 8 and len(lmList2) > 8:
        r1 = lmList2[4][1]
        r2 = lmList2[6][1]
        b1 = lmList1[4][2]
        b2 = lmList1[6][2]

        race = abs(r2 - r1)
        brake = abs(b2 - b1)


        if race < 25:
            if not race_pressed:
                j.set_button(1, 1)
                race_pressed = True
        elif race > 35:
            if race_pressed:
                j.set_button(1, 0)
                race_pressed = False


        if brake < 25:
            if not brake_pressed:
                j.set_button(3, 1)
                brake_pressed = True
        elif brake > 35:
            if brake_pressed:
                j.set_button(3, 0)
                brake_pressed = False
    else:

        j.set_button(1, 0)
        j.set_button(3, 0)
        race_pressed = False
        brake_pressed = False

    def get_point(lmList, idx):
        if lmList and len(lmList) > idx:
            return (lmList[idx][1], lmList[idx][2])
        return None


    p1 = get_point(lmList1, 20)
    p2 = get_point(lmList2, 4)
    p3 = get_point(lmList1, 4)
    p4 = get_point(lmList2, 20)




    if p1 and p2:
        cv2.line(img, p1, p2, (0, 150, 255), 2)
    if p3 and p4:
        cv2.line(img, p3, p4, (0, 150, 255), 2)


    if lmList1 and lmList2:
        y1 = palm_center_y(lmList1)
        y2 = palm_center_y(lmList2)
        diff = y1 - y2
        raw_value = center + int(diff * 60)
        steering_value = max(0, min(32768, raw_value))


        direction = "Right" if diff > 30 else "Left" if diff < -30 else "Center"
        cv2.putText(img, f"Steering: {direction}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime
    cv2.putText(img, f'fps: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Hand Steering", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
