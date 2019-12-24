import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_sauvola
from skimage.morphology import disk, dilation, opening
from skimage.measure import find_contours
from skimage.transform import ProjectiveTransform, warp

from sudoku_solver.utils import plot_img, resize_img

# 
# PLOTTING
#

def plot_contours(img, contours):    
    _, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)    
    plt.axis("off")
    plt.show()

def plot_corners(img, corners):
    _, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.plot(corners[:, 1], corners[:, 0], color='red', marker='o', 
            linestyle='None', markersize=6)    
    ax.axis("off")
    plt.show()

def plot_digits(digits):
    _, ax = plt.subplots()
    for i in range(len(digits)):
        for j in range(len(digits[i])):
            ax = plt.subplot2grid((9, 9), (i, j))
            ax.imshow(digits[i][j], plt.cm.gray)
                            #ax.set_title(str(parsed_img[k][kk]))
            ax.axis('off')
    plt.show()


# 
# PROCESSING
#

# Parameters
RESIZE_WIDTH = 1500
BLUR_SIGMA = 2.0
BINARY_WINDOW = 25
MORPH_KERNEL = 1
CONTOURS_LEVEL = 0.5

def preprocess(input_img):
    #resized_img = resize_img(input_img, width=RESIZE_WIDTH)
    gray_img = rgb2gray(input_img)
    blur_img = gaussian(gray_img, sigma=BLUR_SIGMA)
    return blur_img

def binarise(preprocessed_img):
    thresh = threshold_sauvola(preprocessed_img, window_size=BINARY_WINDOW)
    binary_img = preprocessed_img < thresh    
    dilated_img = dilation(binary_img, disk(MORPH_KERNEL))       
    return dilated_img

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

def extract_grid(input_img):
    preprocessed_img = preprocess(input_img)
    plot_img(preprocessed_img)

    binary_img = binarise(preprocessed_img)
    plot_img(binary_img)

    grid_contour = find_biggest_contour(binary_img)
    plot_contours(binary_img, [grid_contour, ])

    grid_corners = find_corners(grid_contour)
    plot_corners(binary_img, grid_corners)

    warped_img = perspective_warp(preprocessed_img, grid_corners)
    plot_img(warped_img)

    digits = find_digits(warped_img)    
    #plot_digits(digits)

    return digits
