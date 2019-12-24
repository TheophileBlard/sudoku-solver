from abc import ABC, abstractmethod

from sudoku_solver.utils import imshow, resize_img

class GridExtractor(ABC):
    @abstractmethod    
    def infer_grid(self, img):
        pass

# Scikit

from skimage.color import rgb2gray
from skimage.filters import gaussian

class ImgProcExtractor(GridExtractor):    
    def infer_grid(self, img):
        resized = resize_img(img, width=500)
        gray = rgb2gray(img)
        blur = gaussian(gray, sigma=2.0)
        imshow(blur)

        return resized, []

        


