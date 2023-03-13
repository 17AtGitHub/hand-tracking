import cv2
import mediapipe as mp
import time


# going for object-oriented programming instead of functional programming
# better to create classes
class handDetector():
    def __init__(self, mode=False, maxHands=2, mod_complexity=1, detectionConf=0.5, trackConf=0.5):
        # creating an object self, being assigned the value entered by the user
        self.mode = mode
        self.maxHands = maxHands
        self.mod_complexity = mod_complexity
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        # the initializations from the mediapipe library also inside the function
        # and we also have to assign this to the self object
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.mod_complexity, self.detectionConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    # initialization is done, now we can move on to define the functions inside the class
    # basically the methods we are interested to call on from this class
    # the flag set to be true, if changed to false, hands get detected but not drawn
    def find_hands(self, img, draw=True):
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        # to draw the landmarks on the detected hand, we will iterate over the points in
        # results.multi_hand_landmarks, and use the mediapipe method draw_landmarks

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    # to get the positions of the landmarks in the stream, we can create and call another method from the class
    def findPosition(self, img, Handno=0, draw=True):
        #we are going to return a landmark list
        lmList = []
        # if any hand is detected, only then I would go for looking the position
        # so check if the result is containing anything or not
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[Handno]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.putText(img, f'{id}', (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),2)
        return lmList

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0

    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_TRIPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
    # whatever we write in the main part will acting like a dummy code to showcase
            # what can this module do
