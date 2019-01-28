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
cap = cv2.VideoCapture("data/video-1.avi")
cap.set(1, frame_number)

frame_number += 1
ret_val, frame = cap.read()
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print(frame_number)
plt.imshow(frame_gray)
plt.show()

cap.release()
