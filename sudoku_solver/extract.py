from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_sauvola
from skimage.morphology import disk, dilation, opening
from skimage.measure import find_contours
from skimage.transform import ProjectiveTransform, warp

from sudoku_solver.common import plot_img, plot_contours, plot_corners, plot_digits, resize_img


class GridExtractor(ABC):
    @abstractmethod
    def extract_digits(self, input_img):
        pass


#

from skimage.measure import label, regionprops
from skimage.transform import resize
from skimage import util

def chars74_preprocessing(digit):  

    preprocess_img = rgb2gray(digit)
    
    w, h = preprocess_img.shape[:2]
    win_size = int(w // 1.5)
    if win_size % 2 == 0:
        win_size += 1
    
    # use otsu or something fast here
    binary_img = binarise(preprocess_img, window_size=win_size, dilate=True)

    # remove artifacts connected to image border
    cleared = binary_img #clear_border(binary_img)   
   
    label_image = label(cleared)
   
    def distance(tuple_1, tuple_2):
        return np.abs(tuple_1[0] - tuple_2[0]) + np.abs(tuple_1[1] - tuple_2[1])

    def is_digit(region, center, w, h):        
        return (region.eccentricity < 0.99 and
                distance(region.centroid, center) < w / 2.5 and                
                region.bbox_area < (w*h) / 2 and
                region.bbox_area > (w*h) / 30)

    center = (w / 2, h / 2)    
    props = [r for r in regionprops(label_image) if is_digit(r, center, w, h)]
    region = None
    if len(props) == 0:
        return np.zeros((64, 64)), False # TODO: return black image (+ indicateur ?)
    if len(props) == 1:
        region = props[0]
    else:
        print('multiple region !')
        region = sorted(props, key=lambda r: distance(r.centroid, center))[0]
        #raise Exception('houlala')

    minr, minc, maxr, maxc = region.bbox
    cropped = binary_img[minr:maxr, minc:maxc]
    w_crop, h_crop = cropped.shape[:2]

    before = abs(w_crop - h_crop) // 2
    after = abs(w_crop - h_crop) - before
    
    if w_crop > h_crop:
        pad_width = ((0, 0), (before, after))
    else:
        pad_width = ((before, after), (0, 0))

    PIX_FRAME = 3    
    padded = util.pad(cropped, pad_width, mode='constant')
    padded = resize(padded, (64, 64))
    #padded = util.pad(padded, ((PIX_FRAME, PIX_FRAME), (PIX_FRAME, PIX_FRAME)))  

    # util.invert(padded)
    return padded, True


# Scikit Image

RESIZE_WIDTH = 1000
BLUR_SIGMA = 1.0
BINARY_WINDOW = 25
MORPH_KERNEL = 1
CONTOURS_LEVEL = 0.5

def preprocess(input_img, blur_sigma=BLUR_SIGMA): #TODO => static classes ?
    #if input_img.shape[0] > RESIZE_WIDTH:
    #    resized_img = resize_img(input_img, width=RESIZE_WIDTH)
    #else:
    #    resized_img = input_img    
    gray_img = rgb2gray(input_img)
    if blur_sigma > 0:
        preprocessed_img = gaussian(gray_img, sigma=blur_sigma)
    else:
        preprocessed_img = gray_img
    return preprocessed_img

def binarise(preprocessed_img, window_size=BINARY_WINDOW, dilate=False):
    thresh = threshold_sauvola(preprocessed_img, window_size=window_size)
    binary_img = preprocessed_img < thresh
    if dilate:    
        binary_img = dilation(binary_img, disk(MORPH_KERNEL))       
    return binary_img

def find_biggest_contour(binary_img):
    def contour_area(c): # // pas de fonction cv2.ContourArea
        ll, ur = np.min(c, 0), np.max(c, 0)
        wh = ur - ll
        return (wh[0] * wh[1])  

    contours = find_contours(binary_img, CONTOURS_LEVEL)
    contours.sort(key=lambda c: contour_area(c), reverse=True)
    biggest_contour = contours[0]
    return biggest_contour

def find_corners(grid_contour):
    top_left = sorted(grid_contour, key=lambda p: p[0] + p[1])[0]
    top_right = sorted(grid_contour, key=lambda p: p[0] - p[1])[0]
    bottom_left = sorted(grid_contour, key=lambda p: p[0] - p[1], reverse=True)[0]
    bottom_right = sorted(grid_contour, key=lambda p: p[0] + p[1], reverse=True)[0]

    grid_corners = np.array([top_left, top_right, bottom_left, bottom_right])

    return grid_corners

def perspective_warp(img, grid_corners):    
    top_left, top_right, bottom_left, bottom_right = grid_corners

    top_edge = np.linalg.norm(top_right - top_left)
    bottom_edge = np.linalg.norm(bottom_right - bottom_left)
    left_edge = np.linalg.norm(top_left - bottom_left)
    right_edge = np.linalg.norm(top_right - bottom_right)

    L = int(np.ceil(max([top_edge, bottom_edge, left_edge, right_edge])))
    src = np.flip(grid_corners, 1) # Flip x and y axes
    dst = np.array([[0, 0], [L-1, 0], [0, L-1], [L-1, L-1]])

    tf = ProjectiveTransform()
    tf.estimate(dst, src)
    warped_img = warp(img, tf, output_shape=(L, L))

    return warped_img

def find_digits(warped_img):
    L = warped_img.shape[0]
    side = int(np.ceil(L / 9))
    dd = 0

    digits = []
    for i in range(9):
        this_row = []
        start_row_i = max([i * side - dd, 0])
        stop_row_i = min([(i + 1) * side + dd, L])
        for j in range(9):
            start_col_i = max([j * side - dd, 0])
            stop_col_i = min([(j + 1) * side + dd, L])
            digit = warped_img[start_row_i:stop_row_i, start_col_i:stop_col_i].copy()
            this_row.append(digit)
        digits.append(this_row)

    return digits


class ScikitImageExtractor(GridExtractor):
    def __init__(self, show_steps: bool = False):
        self.show_steps = show_steps

    def extract_digits(self, input_img):
        resized_img = resize_img(input_img, width=RESIZE_WIDTH)

        # 1. Preprocessing
        # We blur image because we don't care of details (digits)
        preprocessed_img = preprocess(resized_img, blur_sigma=BLUR_SIGMA)
        if self.show_steps:
            plot_img(preprocessed_img)

        # 2. Binarization
        # We use dilation because we wan't to make sure the grid sides are all connected
        binary_img = binarise(preprocessed_img, dilate=True)
        if self.show_steps:
            plot_img(binary_img)

        # 3. Find grid
        grid_contour = find_biggest_contour(binary_img)
        if self.show_steps:
            plot_contours(binary_img, [grid_contour, ])

        # 4. Find grid corners
        grid_corners = find_corners(grid_contour)
        if self.show_steps:
            plot_corners(binary_img, grid_corners)

        # 5. Apply perspective warp (bird's-eye view)
        warped_img = perspective_warp(resized_img, grid_corners)
        if self.show_steps:
            plot_img(warped_img)

        # 6. Extract digits
        digits = find_digits(warped_img)    
        if self.show_steps:
            plot_digits(digits)

        return digits

    
