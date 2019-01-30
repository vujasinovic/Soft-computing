import os

import inline as inline
import matplotlib
import numpy
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def custom_rgb2gray(frame_rgb):  # custom funkcija za prebaivanje u gray, jer se inace gubi plava linija
    frame_gray = np.ndarray((frame_rgb.shape[0], frame_rgb.shape[1]))
    frame_gray = 0.1*frame_rgb[:, :, 1] + 0.1*frame_rgb[:, :, 2]
    frame_gray = frame_gray.astype('uint8')
    return frame_gray


frame_number = 0
cap = cv2.VideoCapture("data/video-2.avi")
cap.set(1, frame_number)

# frames per second
fps = cap.get(cv2.CAP_PROP_FPS)
print("frames per second: %.2f" % fps)

ret_val, frame = cap.read()
frame_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.imshow(frame_orig )
plt.show()

frame_orig_gray = cv2.cvtColor(frame_orig, cv2.COLOR_RGB2GRAY)
plt.imshow(frame_orig_gray, 'gray')
plt.show()

frame_orig_bin = cv2.adaptiveThreshold(frame_orig_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 11)
plt.imshow(frame_orig_bin, 'gray')
plt.show()

frame, contours, hierarchy = cv2.findContours(frame_orig_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

contours_numbers = []
for c in contours:
    center, size, angle = cv2.minAreaRect(c)
    width, height = size
    if 17 < width < 70 and 17 < height < 70:
        contours_numbers.append(c)

frame = frame_orig.copy()
cv2.drawContours(frame, contours_numbers, -1, (255, 0, 0), 2)
plt.imshow(frame)
plt.show()

cap.release()
