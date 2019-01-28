import os

import inline as inline
import matplotlib
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

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


def custom_rgb2gray(frame_rgb):  # custom funkcija za prebaivanje u gray, jer se inace gubi plava linija
    frame_gray = np.ndarray((frame_rgb.shape[0], frame_rgb.shape[1]))
    frame_gray = 0.1*frame_rgb[:, :, 1] + 0.1*frame_rgb[:, :, 2]
    frame_gray = frame_gray.astype('uint8')
    return frame_gray


frame_gray = custom_rgb2gray(frame_orig)
plt.imshow(frame_gray, 'gray')
plt.show()

kernel = np.ones((2, 2))
frame_eroded = cv2.erode(frame_gray, kernel, iterations=1)
frame_open = cv2.dilate(frame_eroded, kernel, iterations=2)  # koristiti jednu iteraciju ako je potrebno
plt.imshow(frame_open, 'gray')
plt.show()

frame_baw = frame_open > 13  # crno bijeli frejm i otklonjen sum
plt.imshow(frame_baw, 'gray')
plt.show()

cap.release()
