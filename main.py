import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time


wCam, hCam = 640, 480
frameR = 100
clickCooldown = 0.5
doubleClickThreshold = 0.3
lastClickTime = 0
prevClickTime = 0
prevTime = 0

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

screenWidth, screenHeight = pyautogui.size()
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


def fingersUp(lmList):
    fingers = []

    if lmList[4][0] > lmList[3][0]:
        fingers.append(1)
    else:
        fingers.append(0)

    tipIds = [8, 12, 16, 20]
    for i in tipIds:
        if lmList[i][1] < lmList[i - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers


def drawHandUI(img, lmList, bbox):
    global prevTime

    if len(lmList) >= 9:
        x, y = lmList[8]
        cv2.circle(img, (x, y), 12, (255, 0, 255), cv2.FILLED)

    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(img, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)

    # FPS
    currTime = time.time()
    fps = 1 / (currTime - prevTime) if (currTime - prevTime) != 0 else 0
    prevTime = currTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return img


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            xList = []
            yList = []

            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * wCam), int(lm.y * hCam)
                lmList.append((cx, cy))
                xList.append(cx)
                yList.append(cy)

            if lmList:
                x_min, x_max = min(xList), max(xList)
                y_min, y_max = min(yList), max(yList)
                bbox = (x_min, y_min, x_max, y_max)

                fingers = fingersUp(lmList)

                x1, y1 = lmList[8]
                screenX = np.interp(x1, (frameR, wCam - frameR), (0, screenWidth))
                screenY = np.interp(y1, (frameR, hCam - frameR), (0, screenHeight))
                pyautogui.moveTo(screenX, screenY)

                x2, y2 = lmList[12]
                distance = math.hypot(x2 - x1, y2 - y1)

                currentTime = time.time()
                if fingers[1] == 1 and fingers[2] == 1 and distance < 40:
                    if currentTime - lastClickTime < doubleClickThreshold:
                        pyautogui.doubleClick()
                        lastClickTime = 0
                        cv2.putText(img, 'Double Click', (x1, y1 - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif currentTime - lastClickTime > clickCooldown:
                        pyautogui.click()
                        lastClickTime = currentTime
                        cv2.putText(img, 'Click', (x1, y1 - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if fingers == [0, 0, 1, 0, 0] and currentTime - lastClickTime > clickCooldown:
                    pyautogui.rightClick()
                    lastClickTime = currentTime
                    cv2.putText(img, 'Right Click', (x1, y1 - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)


                img = drawHandUI(img, lmList, bbox)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
