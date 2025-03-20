import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)

offset = 20
imgsize = 300

while True:
    success , img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgwhite = np.ones((imgsize,imgsize,3),np.uint8)*255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropshape = imgCrop.shape
        aspectratioo = h/w

        if aspectratioo >1:
            k = imgsize/h
            wCal = math.ceil(k*w)
            imgresize = cv2.resize(imgCrop,(wCal,imgsize))
            imgresizeshape = imgresize.shape
            wGap = math.ceil((imgsize - wCal)/2)
            imgwhite[:, wGap:wCal+wGap] = imgresize

        else:
            k = imgsize / w
            hCal = math.ceil(k * h)
            imgresize = cv2.resize(imgCrop, (imgsize,hCal))
            imgresizeshape = imgresize.shape
            hGap = math.ceil((imgsize - hCal) / 2)
            imgwhite[hGap:hCal + hGap, :] = imgresize

        cv2.imshow("imageCrop", imgCrop)
        cv2.imshow('imagewhite',imgwhite)
        
    cv2.imshow("image",img)
    cv2.waitKey(1)

