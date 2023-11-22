import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os
from PIL import Image, ImageDraw, ImageFont

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 20
imgSize = 300

folder = "Data/na"
counter = 0

# Create the folder if it doesn't exist
if not os.path.exists(folder):
    os.makedirs(folder)

font_path = "NotoSansBengaliUI-Regular.ttf"  # Replace with the path to your Bangla font file
font_size = 40
font_color = (255, 255, 255)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        imgWhite = np.ones((imgSize, imgSize * 2, 3), np.uint8) * 255
        for i, hand in enumerate(hands[:2]):
            x, y, w, h = hand['bbox']
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgCrop.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, i * imgSize + wGap:i * imgSize + wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, i * imgSize:i * imgSize + imgSize] = imgResize

            cv2.imshow(f"ImageCrop {i + 1}", imgCrop)

        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        counter += 1
        imgPil = Image.fromarray(imgWhite)
        draw = ImageDraw.Draw(imgPil)
        font = ImageFont.truetype(font_path, font_size)
        text = f"Image_{time.time()}.jpg"
        text_width, text_height = draw.textsize(text, font=font)
        text_position = (10, 10)
        draw.text(text_position, text, font=font, fill=font_color)
        imgWhite = np.array(imgPil)
        cv2.imwrite(f'{folder}/{text}', imgWhite)
        print(counter)

    if key == 27:  # Press Esc key to exit
        break

cap.release()
cv2.destroyAllWindows()
