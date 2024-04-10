import cv2
import numpy as np

def stack_images(img_array, scale, labels=[]):
    """
    Stacks all images in one window for display.
    """
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_available:
        for x in range(rows):
            for y in range(cols):
                img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                if len(img_array[x][y].shape) == 2: img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(rows):
            img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            if len(img_array[x].shape) == 2: img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver

def reorder(points):
    """
    Reorders points in a specific order.
    """
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points

def find_biggest_contour(contours):
    """
    Finds the biggest contour from a list of contours.
    """
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

def draw_rectangle(img, biggest, thickness):
    """
    Draws a rectangle using the biggest contour points.
    """
    cv2.line(img, tuple(biggest[0][0]), tuple(biggest[1][0]), (0, 255, 0), thickness)
    cv2.line(img, tuple(biggest[0][0]), tuple(biggest[2][0]), (0, 255, 0), thickness)
    cv2.line(img, tuple(biggest[3][0]), tuple(biggest[2][0]), (0, 255, 0), thickness)
    cv2.line(img, tuple(biggest[3][0]), tuple(biggest[1][0]), (0, 255, 0), thickness)
    return img
