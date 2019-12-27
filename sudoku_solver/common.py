
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.transform import resize

mpl.use('TkAgg')

def load_img(path: str) -> np.ndarray:
    img = io.imread(path)
    return img

def plot_img(img: np.ndarray):
    plt.axis("off")
    plt.imshow(img, cmap = 'gray')
    plt.show()

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


# =========

def plot_comparison(original, filtered, filter_name):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')

def resize_img(img, width = None, height = None):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    output_shape = None
    (w, h) = img.shape[:2]    

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return img

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        output_shape = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        output_shape = (width, int(h * r))

    # resize the image
    resized_img = resize(img, output_shape)

    # return the resized image
    return resized_img

