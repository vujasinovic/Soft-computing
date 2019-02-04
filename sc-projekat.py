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


def erode(image):
    kernel = np.ones((2, 2))
    return cv2.erode(image, kernel, iterations=1)


def dilate(image):
    kernel = np.ones((2, 2))
    return cv2.dilate(image, kernel, iterations=1)


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


def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)


def select_roi(image_orig, image_bin):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if area > 100 and h < 100 and h > 15 and w > 20:
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (255, 0, 0), 2)
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]

    return image_orig, sorted_regions


def scale_to_range(image):
    return image/255


def matrix_to_vector(image):
    return image.flatten()


def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))

    return ready_for_ann


def convert_output(alphabet):
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)


frame_number = 0
cap = cv2.VideoCapture("data/video-2.avi")
while cap.isOpened():
    ret_val, frame = cap.read()
    frame_orig = convert_bgr2rgb(frame)
    # display_image(frame_orig, True)

    frame_orig_gray = convert_rgb2gray(frame_orig)
    # display_image(frame_orig_gray, False)
    edges = cv2.Canny(frame_orig_gray, 50, 100, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*a)
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*a)

        cv2.line(frame_orig, (x1, y1), (x2, y2), (0, 0, 255), 2)

    frame_orig_bin = adaptive_threshold(frame_orig_gray)
    frame_orig_bin = erode(frame_orig_bin)
    # display_image(frame_orig_bin, False)

    frame_rec, regions = select_roi(frame_orig, frame_orig_bin)
    cv2.imshow('frame', frame_rec)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
