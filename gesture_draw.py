import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)

mpDraw = mp.solutions.drawing_utils

canvas = None

xp, yp = 0, 0

color = (255,0,0)

while True:

    success, img = cap.read()

    img = cv2.flip(img,1)

    if canvas is None:
        canvas = np.zeros_like(img)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:

            lmList = []

            for id,lm in enumerate(handLms.landmark):

                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)

                lmList.append((cx,cy))

            mpDraw.draw_landmarks(img,handLms,
                                  mpHands.HAND_CONNECTIONS)

            if len(lmList)!=0:

                x1,y1 = lmList[8]   # index finger
                x2,y2 = lmList[12]  # middle finger

                # Draw Mode (1 finger)
                if y1 < y2:

                    cv2.circle(img,(x1,y1),10,color,-1)

                    if xp==0 and yp==0:
                        xp,yp = x1,y1

                    cv2.line(canvas,(xp,yp),(x1,y1),
                             color,5)

                    xp,yp = x1,y1

                # Stop Drawing (2 fingers)
                else:
                    xp,yp=0,0

    imgGray = cv2.cvtColor(canvas,
                           cv2.COLOR_BGR2GRAY)

    _,imgInv = cv2.threshold(imgGray,
                             50,
                             255,
                             cv2.THRESH_BINARY_INV)

    imgInv = cv2.cvtColor(imgInv,
                          cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,canvas)

    cv2.imshow("Gesture Drawing",img)

    if cv2.waitKey(1)==27:
        break