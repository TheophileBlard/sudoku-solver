from matplotlib import pyplot as plt
import numpy as np
from skimage import io
from skimage.transform import resize

def load_img(path: str) -> np.ndarray:
    img = io.imread(path)
    return img

def plot_img(img: np.ndarray):
    plt.axis("off")
    plt.imshow(img, cmap = 'gray')
    plt.show()

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