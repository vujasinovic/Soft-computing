from __future__ import print_function

import cv2
import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
import vectors as vct
import tensorflow as tf
from keras.layers import Conv2D, Dropout, Flatten, MaxPooling2D
from keras.layers.core import Dense
from keras.models import Sequential
from keras.models import load_model

model = load_model('model.h5')

def display_image(image, color=False):
    if color:
        plt.imshow(image)
        plt.show()
    else:
        plt.imshow(image, 'gray')
        plt.show()


def dilate(image):
    kernel = np.ones((2, 2))
    return cv2.dilate(image, kernel, iterations=1)


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


def scale_to_range(image):
    return image / 255


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


def create_masks(hsv):
    lower_blue = np.array([110, 80, 2])
    upper_blue = np.array([130, 255, 255])

    lower_green = np.array([50, 50, 120])
    upper_green = np.array([70, 255, 255])

    green_m = cv2.inRange(hsv, lower_green, upper_green)
    blue_m = cv2.inRange(hsv, lower_blue, upper_blue)

    return green_m, blue_m


def find_line(frame, mask):
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result


def find_green_line_coords(img):
    edges = cv2.Canny(img, 50, 250, apertureSize=3)
    coords = (1000, 0, 0, 1000)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 25, 5, 150)
    if lines is not None:
        for x1, y1, x2, y2 in lines[0]:
            coords = x1, y1, x2, y2
            # print('Green line coords: ')
            # print(coords)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return coords


def find_blue_line_coords(img):
    edges = cv2.Canny(img, 50, 100, apertureSize=3)
    coords = (1000, 0, 0, 1000)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, 100, 10)
    if lines is not None:
        for x1, y1, x2, y2 in lines[0]:
            coords = x1, y1, x2, y2
            # print('Blue line coords: ')
            # print(coords)
            cv2.line(edges, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return coords


def setup_vectors():
    global green_line_start
    global green_line_end
    global green_line_vector
    global green_line_length
    global green_line_unit_vector

    global blue_line_start
    global blue_line_end
    global blue_line_vector
    global blue_line_length
    global blue_line_unit_vector
    green_line_start = (green_x1, green_y1, 0)
    green_line_end = (green_x2, green_y2, 0)
    green_line_vector = vct.vector(green_line_start, green_line_end)
    green_line_length = vct.length(green_line_vector)
    if green_line_length != 0.0:
        green_line_unit_vector = vct.unit(green_line_vector)

    blue_line_start = (blue_x1, blue_y1, 0)
    blue_line_end = (blue_x2, blue_y2, 0)
    blue_line_vector = vct.vector(blue_line_start, blue_line_end)
    blue_line_length = vct.length(blue_line_vector)
    if blue_line_length != 0.0:
        blue_line_unit_vector = vct.unit(blue_line_vector)


def select_roi(image_orig, image_bin):
    global suma

    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    sorted_regions = []
    regions_array = []

    i = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if area > 10 and area < 230 and (w > 15 or h > 15):
            if green_line_length != 0 and blue_line_length != 0:
                dist_green, nearest = vct.pnt2line((x+w, y+h, 0), (green_x1, green_y1, 0), (green_x2, green_y2, 0))
                dist_blue, nearest = vct.pnt2line((x, y, 0), (blue_x1, blue_y1, 0), (blue_x2, blue_y2, 0))

                region = image_bin[y:y + h + 1, x:x + w + 1]
                number = predict_number(region)
                # display_image(region, True)
                if number not in passed_green_array and dist_green < 0.5:
                    passed_green_array.append(number)
                    suma = suma - number;
                    print('PASSED GREEN:', number)
                    print('SUMA: ', suma)

                if number not in passed_blue_array and dist_blue < 0.5:
                    passed_blue_array.append(number)
                    suma = suma + number
                    print('PASSED BLUE:', number)
                    print('SUMA: ', suma)
                else:
                    continue
                # if dist_green < 5:
                #     if number not in passed_green_array:
                #         passed_green_array.append(number)
                #         suma = suma - number
                #         print('PASSED GREEN:', number)
                #         print('SUMA: ', suma)
                #     elif number in passed_green_array:
                #         continue
                # elif number in passed_green_array:
                #     passed_green_array.remove(number)
                # if dist_blue < 5:
                #     if number not in passed_blue_array:
                #         passed_blue_array.append(number)
                #         suma = suma + number
                #         print('PASSED BLUE:', number)
                #         print('SUMA: ', suma)
                #     elif number in passed_blue_array:
                #         continue
                # elif number in passed_blue_array:
                #     passed_blue_array.remove(number)

        area = cv2.contourArea(contour)
        if area > 10 and area < 230 and (w > 14 or h > 14):
            # print(w, h)
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (255, 0, 0), 2)
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = [region[0] for region in regions_array]
    return image_orig, sorted_regions


def predict_number(region):
    black_image = np.ones([28, 28], dtype=np.uint8) * 0
    if region.shape[0] < 25 and region.shape[1] < 25:
        black_image[3:3 + region.shape[0], 3:3 + region.shape[1]] = region
        # display_image(black_image, True)
        result = model.predict(black_image.reshape(1, 28, 28, 1))
        # print(result.argmax())
        return result.argmax()
    elif region.shape[0] > 28 or region.shape[1] > 28:
        return 0
    else:
        black_image[0:0 + region.shape[0], 0:0 + region.shape[1]] = region
        # display_image(black_image, True)
        result = model.predict(black_image.reshape(1, 28, 28, 1))
        # print(result.argmax())
        return result.argmax()


def predict_numbers():
    numbers = []
    frame_rec, regions = select_roi(frame.copy(), frame_orig_bin)
    for region in regions:
        # black_image = cv2.imread('data/black_image.jpg')
        black_image = np.ones([28, 28], dtype=np.uint8) * 0
        if region.shape[0] < 25 and region.shape[1] < 25:
            black_image[3:3 + region.shape[0], 3:3 + region.shape[1]] = region
            # display_image(black_image, True)
            result = model.predict(black_image.reshape(1, 28, 28, 1))
            numbers.append(result.argmax())
        elif region.shape[0] > 28 or region.shape[1] > 28:
            continue
        else:
            black_image[0:0 + region.shape[0], 0:0 + region.shape[1]] = region
            # display_image(black_image, True)
            result = model.predict(black_image.reshape(1, 28, 28, 1))
            numbers.append(result.argmax())

        # print(result)

    # print(numbers)
    return frame_rec


def find_line_coords(blue_line_c):
    global blue_x1
    global blue_x2
    global blue_y1
    global blue_y2

    global green_x1
    global green_x2
    global green_y1
    global green_y2

    blue_x1 = min([bc[0] for bc in blue_line_c])
    blue_y1 = max([bc[1] for bc in blue_line_c])
    blue_x2 = max([bc[2] for bc in blue_line_c])
    blue_y2 = min([bc[3] for bc in blue_line_c])

    green_x1 = min([gc[0] for gc in green_line_c])
    green_y1 = max([gc[1] for gc in green_line_c])
    green_x2 = max([gc[2] for gc in green_line_c])
    green_y2 = min([gc[3] for gc in green_line_c])

    print('b min x1: ', blue_x1)
    print('b max y1: ', blue_y1)
    print('b max x2: ', blue_x2)
    print('b min y2: ', blue_y2)

    print('g min x1: ', green_x1)
    print('g max y1: ', green_y1)
    print('g max x2: ', green_x2)
    print('g min y2: ', green_y2)


suma = 0
frame_number = 0
cap = cv2.VideoCapture("data/video-0.avi")
flag = False
blue_line_c = []
green_line_c = []
blue_x1 = 0
blue_y1 = 0
blue_x2 = 0
blue_y2 = 0
green_x1 = 0
green_x2 = 0
green_y1 = 0
green_y2 = 0
green_line_start = 0
green_line_end = 0
green_line_vector = 0
green_line_length = 0
green_line_unit_vector = 0

blue_line_start = 0
blue_line_end = 0
blue_line_vector = 0
blue_line_length = 0
blue_line_unit_vector = 0

passed_green_array = []
passed_blue_array = []
while cap.isOpened():
    frame_number = frame_number + 1
    if frame_number % 10 == 0:
        passed_green_array = []
        passed_blue_array = []

    ret_val, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    green_mask, blue_mask = create_masks(hsv)

    result_green = find_line(frame, green_mask)
    result_blue = find_line(frame, blue_mask)

    blue_line_rgb = cv2.cvtColor(result_blue, cv2.COLOR_BGR2RGB)
    blue_line_gray = cv2.cvtColor(blue_line_rgb, cv2.COLOR_RGB2GRAY)

    green_line_rgb = cv2.cvtColor(result_green, cv2.COLOR_BGR2RGB)
    green_line_gray = cv2.cvtColor(green_line_rgb, cv2.COLOR_RGB2GRAY)

    if frame_number < 10:
        green_line_coords = find_green_line_coords(result_green)
        blue_line_coords = find_blue_line_coords(result_blue)
        blue_line_c.append(blue_line_coords)
        green_line_c.append(green_line_coords)

    if frame_number == 10:
        find_line_coords(blue_line_c)
        setup_vectors()
        # cv2.imshow('res_blue', result_blue)
        # cv2.imshow('res_green', result_green)

    # cv2.line(frame, (green_x1, green_y1), (green_x2, green_y2), (255, 0, 255), 2)
    # cv2.line(frame, (blue_x1, blue_y1), (blue_x2, blue_y2), (255, 0, 255), 2)
    # cv2.imshow('with lines', frame)
    frame_original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # display_image(frame_orig, True)

    frame_original_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # cv2.imshow('gray', frame_original_gray)
    # display_image(frame_orig_gray, False)

    kernel = np.ones((2, 2))
    frame_orig_bin = cv2.erode(frame_original_gray, np.ones((3, 3)), iterations=2)
    frame_orig_bin = cv2.dilate(frame_original_gray, kernel, iterations=1)
    _, frame_orig_bin = cv2.threshold(frame_orig_bin, 127, 255, cv2.THRESH_BINARY)
    # display_image(frame_orig_bin, False)
    frame_rec = predict_numbers()

    cv2.imshow('frame', frame_rec)
    # cv2.waitKey(2000)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
