import cv2
import mediapipe as mp
import numpy as np
import speech_recognition as sr
import threading

# Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

canvas = None
xp, yp = 0, 0

# Default color
color = (255,0,0)

# Voice recognition
recognizer = sr.Recognizer()

def listen_voice():
    global color, canvas

    while True:
        try:
            with sr.Microphone() as source:

                print("Listening...")

                audio = recognizer.listen(source)

                command = recognizer.recognize_google(audio)

                command = command.lower()

                print("You said:",command)

                if "red" in command:
                    color = (0,0,255)

                if "blue" in command:
                    color = (255,0,0)

                if "green" in command:
                    color = (0,255,0)

                if "clear" in command:
                    canvas = None

                if "save" in command:
                    cv2.imwrite("drawing.png",canvas)
                    print("Saved")

        except:
            pass

# Run voice in background
threading.Thread(target=listen_voice,
                 daemon=True).start()

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

            mpDraw.draw_landmarks(img,
                                  handLms,
                                  mpHands.HAND_CONNECTIONS)

            if len(lmList)!=0:

                x1,y1 = lmList[8]
                x2,y2 = lmList[12]

                # Draw mode
                if y1 < y2:

                    if xp==0 and yp==0:
                        xp,yp = x1,y1

                    cv2.line(canvas,
                             (xp,yp),
                             (x1,y1),
                             color,
                             5)

                    xp,yp = x1,y1

                else:
                    xp,yp=0,0

    imgGray=cv2.cvtColor(canvas,
                         cv2.COLOR_BGR2GRAY)

    _,imgInv=cv2.threshold(imgGray,
                           50,
                           255,
                           cv2.THRESH_BINARY_INV)

    imgInv=cv2.cvtColor(imgInv,
                        cv2.COLOR_GRAY2BGR)

    img=cv2.bitwise_and(img,imgInv)
    img=cv2.bitwise_or(img,canvas)

    cv2.imshow("AI Gesture + Voice Drawing",img)

    if cv2.waitKey(1)==27:
        break