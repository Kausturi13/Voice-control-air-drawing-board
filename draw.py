import cv2
import mediapipe as mp
import numpy as np

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7)

mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

canvas = None

xp, yp = 0, 0

while True:

    success, img = cap.read()

    if not success:
        continue

    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:

            lmList = []

            for id, lm in enumerate(handLms.landmark):

                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)

                lmList.append((cx, cy))

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            if len(lmList) != 0:

                x1, y1 = lmList[8]   # Index finger tip

                cv2.circle(img, (x1,y1), 10, (255,0,255), cv2.FILLED)

                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                cv2.line(canvas,(xp,yp),(x1,y1),(0,255,0),5)

                xp, yp = x1, y1

    imgGray = cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,canvas)

    cv2.imshow("Air Drawing",img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()