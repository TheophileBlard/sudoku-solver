import argparse

from sudoku_solver.extract import ScikitImageExtractor
from sudoku_solver.classify import ScikitClassifier
#from sudoku_solver.solve import Solver

from sudoku_solver.common import load_img, plot_img, plot_digits
from sudoku_solver.data import SudokuGrid


def mnist_preprocessing(digits):
    #TODO: extract the centered blob and scale to 28x28
    return digits    


def launch_pipeline(img):
    #plot_img(img)
    extractor = ScikitImageExtractor(show_steps=False)
    classifier = ScikitClassifier(model_path='models/mnist/forest_clf.joblib')

    img_digits = extractor.extract_digits(img)
    digits = classifier.predict(img_digits, preprocessing_func=mnist_preprocessing)
    grid = SudokuGrid(digits)
    
    print(grid)
    plot_digits(img_digits)



# TODO: prepare a pickle loading for the dataset (img from .jpg, "labels" from .dat)

if __name__ == '__main__':
    #img = load_img('datasets/mathworks/segmentation_data/raw_data/images/0010_05.jpg')
    img = load_img('datasets/bernard/25.jpg')
    launch_pipeline(img)

    

