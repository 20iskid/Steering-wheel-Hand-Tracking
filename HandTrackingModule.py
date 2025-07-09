import cv2
import mediapipe as mp
import time
from collections import defaultdict

try:
    from filterpy.kalman import KalmanFilter
    import numpy as np
except ImportError:
    KalmanFilter = None  # only needed for kalman mode

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5, smoothing=None):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.smoothing = smoothing  # "average", "kalman", or None

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

        # Smoothing buffers
        self.history = defaultdict(list)
        self.smooth_factor = 5  # for moving average
        self.kalman_filters = {}  # for kalman filter

    def _init_kalman(self, id, x, y):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([x, y, 0, 0])  # x, y, dx, dy
        kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        kf.P *= 1000.
        kf.R = 5
        kf.Q = 0.1
        return kf

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                h, w, c = img.shape

                for id, lm in enumerate(myHand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    if self.smoothing == "average":
                        self.history[id].append((cx, cy))
                        if len(self.history[id]) > self.smooth_factor:
                            self.history[id].pop(0)
                        avg_x = int(sum(p[0] for p in self.history[id]) / len(self.history[id]))
                        avg_y = int(sum(p[1] for p in self.history[id]) / len(self.history[id]))
                        lmList.append([id, avg_x, avg_y])

                    elif self.smoothing == "kalman" and KalmanFilter:
                        if id not in self.kalman_filters:
                            self.kalman_filters[id] = self._init_kalman(id, cx, cy)
                        kf = self.kalman_filters[id]
                        kf.predict()
                        kf.update(np.array([cx, cy]))
                        kx, ky = int(kf.x[0]), int(kf.x[1])
                        lmList.append([id, kx, ky])
                    else:
                        lmList.append([id, cx, cy])

        return lmList


# Demo usage
def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector(detectionCon=0.9, smoothing="average")  # options: "average", "kalman", None

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        if lmList:
            print(lmList[4])  # Example: thumb tip

        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime - pTime else 0
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
