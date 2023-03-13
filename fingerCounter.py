import cv2
import time
import mediapipe as mp
import os
import numpy as np
import HandTrackingModule as htm

wCam, hCam = 1080, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = 'counter images'
mylist = os.listdir(folderPath)
print(mylist)

#copy the images into the overlay list
overlay = []
for imPath in mylist:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlay.append(image)

def resize (img, scaleFactor):
    nh=int(img.shape[0]*scaleFactor/100)
    nw=int(img.shape[1]*scaleFactor/100)
    dim=(nw,nh)
    img=cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
    return img

pTime=0

detector = htm.handDetector(detectionConf=0.75)

tipId = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()

    img = detector.find_hands(img)
    lmList = detector.findPosition(img,draw=False)
    # print(lmList)
    ones = 0
    if len(lmList) != 0:
        fingers = []
        #thumb
        if lmList[4][1] > lmList[2][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #4 fingers
        for i in range(1,5):
            if lmList[tipId[i]][2] < lmList[tipId[i]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        ones = fingers.count(1)
        print(ones)




    ih,iw=overlay[ones-1].shape[:2]
    img[0:ih,0:iw]=overlay[ones-1]

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'FPS: {str(int(fps))}', (800,100), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0),1)
    cv2.imshow("Image", img)
    cv2.waitKey(1)