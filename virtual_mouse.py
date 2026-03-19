import cv2
import mediapipe as mp
import pyautogui

# Initialize camera FIRST
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera failed")
    exit()

# Initialize Mediapipe AFTER camera
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)

mpDraw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()

while True:

    success, img = cap.read()

    if success == False:
        print("Camera not working")
        break

    img = cv2.flip(img,1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:

            h, w, c = img.shape

            for id, lm in enumerate(handLms.landmark):

                cx, cy = int(lm.x*w), int(lm.y*h)

                if id == 8:  # Index finger

                    mouse_x = screen_w/w * cx
                    mouse_y = screen_h/h * cy

                    pyautogui.moveTo(mouse_x, mouse_y)

            mpDraw.draw_landmarks(img, handLms,
                                  mpHands.HAND_CONNECTIONS)

    cv2.imshow("Virtual Mouse", img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()