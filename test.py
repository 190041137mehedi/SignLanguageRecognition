import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from PIL import ImageFont, ImageDraw, Image

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

print(type(classifier))
print(classifier.summary())
offset = 20
imgSize = 300
font_path = "NotoSansBengaliUI-Regular.ttf"  # Replace with the path to your Bangla font file
font = ImageFont.truetype(font_path, 40)

folder = "Data/ঘ"
counter = 0

classifier.model.sum
labels = ["ক", "খ", "ঘ", "ঙ" , "চ", "ছ", , "ট", "ঠ", "ড", "ঢ","ণ,ন", "ত", "থ", "দ", "ধ", "প", "ফ", "ব,ভ", "ম", "ল", "শ,স,ষ","হ"]

# , "ট", "ঠ", "ড", "ঢ","ণ,ন", "ত", "থ", "দ", "ধ", "প", "ফ", "ব,ভ", "ম", "ল", "শ,স,ষ","হ"

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)

        # Display each character of the label separately
        label = labels[index]
        for i, char in enumerate(label):
            img_pil = Image.fromarray(imgOutput)
            draw = ImageDraw.Draw(img_pil)
            text_width, text_height = draw.textsize(char, font=font)
            draw.text((x + i * 30, y - 26 - text_height), char, font=font, fill=(255, 255, 255))
            imgOutput = np.array(img_pil)

        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()