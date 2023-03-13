import cv2
import mediapipe as mp
import numpy as np
import time
import HandTrackingModule as htm
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam = 1080
hCam = 720

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

pTime = 0
lmList = []
hands = htm.handDetector()


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(0, None)
minVol = volRange[0]
maxVol = volRange[1]



while True:
    success, img = cap.read()

    img = hands.find_hands(img,draw=True)
    lmList = hands.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        cv2.circle(img, (x1,y1), 7, (255,0,0), -1)
        cv2.circle(img, (x2,y2), 7, (255,0,0), -1)

        xmid, ymid = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (xmid,ymid), 7, (255,0,0), -1)
        cv2.line(img, (x1,y1), (x2,y2), (255,0,0), 2)

        dist = int(math.hypot(x1-x2,y1-y2))
        # print(dist)

        # range of the dist
        d_min, d_max = 30, 230
        
        vol1 = np.interp(dist, [d_min,d_max], [minVol, maxVol])
        vol2 = int(np.interp(vol1, [minVol, maxVol], [0, 100]))
        vol = int(np.interp(vol2, [0,100], [minVol, maxVol]))
        volume.SetMasterVolumeLevel(vol, None)
        top_l = (50,100)
        bottom_r = (100,500)

        cv2.rectangle(img, top_l, bottom_r, (255,0,0), 2)
        cv2.circle(img, top_l, 7, (0, 255, 0), -1)
        cv2.circle(img, bottom_r, 7, (255, 0, 0), -1)
        h_bar = int(np.interp(vol2, [0,100], [500,100]))
        cv2.rectangle(img, (50, h_bar) ,bottom_r, (255,0,0), -1)

        cv2.putText(img, f'{vol2}%', (60,80), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,0,0), 2)

        if dist<30:
            cv2.circle(img, (x1, y1), 7, (0, 0, 255), -1)
            cv2.circle(img, (x2, y2), 7, (0, 0, 255), -1)
            cv2.circle(img, (xmid, ymid), 7, (0, 0, 255), -1)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if dist>230:
                cv2.circle(img, (x1, y1), 7, (0, 255, 0), -1)
                cv2.circle(img, (x2, y2), 7, (0, 255, 0), -1)
                cv2.circle(img, (xmid, ymid), 7, (0, 255, 0), -1)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {str(int(fps))}', (700,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0),4)


    cv2.imshow("image",img)
    cv2.waitKey(1)
