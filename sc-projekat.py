from __future__ import print_function
import os

import inline as inline
import matplotlib
import numpy
import numpy as np
import cv2
import collections

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

import matplotlib.pyplot as plt


def display_image(image, color=False):
    if color:
        plt.imshow(image)
        plt.show()
    else:
        plt.imshow(image, 'gray')
        plt.show()


def custom_rgb2gray(frame_rgb):  # custom funkcija za prebaivanje u gray, jer se inace gubi plava linija
    frame_gray = np.ndarray((frame_rgb.shape[0], frame_rgb.shape[1]))
    frame_gray = 0.1*frame_rgb[:, :, 1] + 0.1*frame_rgb[:, :, 2]
    frame_gray = frame_gray.astype('uint8')
    return frame_gray


def convert_bgr2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_rgb2gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def adaptive_threshold(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 11)


def extract_number_contours(contours):
    contours_numbers = []
    for c in contours:
        center, size, angle = cv2.minAreaRect(c)
        width, height = size
        if 17 < width < 70 and 17 < height < 70:
            contours_numbers.append(c)
    return contours_numbers


frame_number = 0
cap = cv2.VideoCapture("data/video-2.avi")
cap.set(1, frame_number)

# frames per second
fps = cap.get(cv2.CAP_PROP_FPS)
print("frames per second: %.2f" % fps)


ret_val, frame = cap.read()
frame_orig = convert_bgr2rgb(frame)
display_image(frame_orig, True)

frame_orig_gray = convert_rgb2gray(frame_orig)
display_image(frame_orig_gray, False)

frame_orig_bin = adaptive_threshold(frame_orig_gray)
display_image(frame_orig_bin, False)

frame, ctr, hierarchy = cv2.findContours(frame_orig_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

numbers_only = extract_number_contours(ctr)

frame = frame_orig.copy()
cv2.drawContours(frame, numbers_only, -1, (255, 0, 0), 2)
display_image(frame, True)

cap.release()
