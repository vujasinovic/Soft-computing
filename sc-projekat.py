from __future__ import print_function

import cv2
import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Dropout, Flatten, MaxPooling2D
from keras.layers.core import Dense
from keras.models import Sequential

import train


def cnn():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_test', x_test.shape[0])

    # Creating a Sequential Model and adding the layers
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # model.fit(x=x_train, y=y_train, epochs=10)
    # model.evaluate(x_test, y_test)
    return model

def display_image(image, color=False):
    if color:
        plt.imshow(image)
        plt.show()
    else:
        plt.imshow(image, 'gray')
        plt.show()


def erode(image):
    kernel = np.ones((3, 3))
    return cv2.erode(image, kernel, iterations=1)


def dilate(image):
    kernel = np.ones((2, 2))
    return cv2.dilate(image, kernel, iterations=1)


def custom_rgb2gray(frame_rgb):  # custom funkcija za prebaivanje u gray, jer se inace gubi plava linija
    frame_gray = np.ndarray((frame_rgb.shape[0], frame_rgb.shape[1]))
    frame_gray = 0.1 * frame_rgb[:, :, 1] + 0.1 * frame_rgb[:, :, 2]
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


def select_roi(image_orig, image_bin):
    # cv2.imshow('dbg', image_bin)
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_regions = []
    regions_array = []
    i = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if area > 10 and area < 230 and (w > 14 or h > 14):
            # print(w, h)
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([region, (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (255, 0, 0), 2)
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = [region[0] for region in regions_array]

    return image_orig, sorted_regions


def predict_numbers():
    numbers = []
    frame_rec, regions = select_roi(frame_original.copy(), frame_orig_bin)
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

    print(len(numbers))
    return frame_rec


frame_number = 0
cap = cv2.VideoCapture("data/video-2.avi")
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
model = cnn()
while cap.isOpened():
    cv2.waitKey(0)
    frame_number = frame_number + 1
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
        print('b min y1: ', blue_y2)

        print('g min x1: ', green_x1)
        print('g max y1: ', green_y1)
        print('g max x2: ', green_x2)
        print('g min y1: ', green_y2)
        # cv2.imshow('res_blue', result_blue)
        # cv2.imshow('res_green', result_green)

    ########################DRAW LINES##########################################
    cv2.line(frame, (green_x1, green_y1), (green_x2, green_y2), (255, 0, 255), 2)
    cv2.line(frame, (blue_x1, blue_y1), (blue_x2, blue_y2), (255, 0, 255), 2)
    # cv2.imshow('with lines', frame)
    ############################################################################
    frame_original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # display_image(frame_orig, True)

    frame_original_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # cv2.imshow('gray', frame_original_gray)
    # display_image(frame_orig_gray, False)

    # frame_orig_bin = erode(frame_orig_gray)
    frame_orig_bin = dilate(frame_original_gray)
    _, frame_orig_bin = cv2.threshold(frame_orig_bin, 127, 255, cv2.THRESH_BINARY)
    # display_image(frame_orig_bin, False)
    frame_rec = predict_numbers()

    cv2.imshow('frame', frame_rec)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
