import cv2
import mediapipe as mp
import time
# time is to check the frame rates


cap = cv2.VideoCapture(0)

# using the hand detection module provided by the mp lib
# we are going to create an object from our class hands

mpHands = mp.solutions.hands
hands = mpHands.Hands()
# there are four parameters in this function
# static image mode(false), max num of hands(2)
# min detecting confidence(0.5 i.e. 50%)
# min tracking confidence(0.5 i.e. 50%)
# there are default already set, if we wish to change, we can
# we will let the default parameters be set

mpDraw = mp.solutions.drawing_utils

# to display the frames per second: fps
pTime=0
cTime=0

while True:
    success, img = cap.read()

    # hands class, object only uses the RGB image
    # so, we have to change the color space from BGR to RGB
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
    # print(results.multi_hand_landmarks)

    # to draw the landmarks on the detected hand, we will iterate over the points in
    # results.multi_hand_landmarks, and use the mediapipe method draw_landmarks
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id == 4:
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    cTime = time.time();
    fps = 1/(cTime-pTime);
    pTime = cTime
    
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_TRIPLEX, 3, (255,0,255), 3)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)