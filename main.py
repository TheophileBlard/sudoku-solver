import argparse

from sudoku_solver.extract import ScikitImageExtractor
from sudoku_solver.classify import ScikitLearnClassifier
#from sudoku_solver.solve import Solver

from sudoku_solver.common import load_img, plot_img, plot_digits
from sudoku_solver.data import SudokuGrid


from sudoku_solver.extract import chars74_preprocessing

def launch_pipeline(img):
    #plot_img(img)
    extractor = ScikitImageExtractor(show_steps=False)
    classifier = ScikitLearnClassifier(model_path='models/chars74k/forest_clf.joblib')

    raw_digits = extractor.extract_digits(img)
    
    plot_digits(raw_digits)

    for i in range(len(raw_digits)):
        for j in range(len(raw_digits[i])):
            proc_digit, is_digit = chars74_preprocessing(raw_digits[i][j])
            raw_digits[i][j] = proc_digit # TODO: delete
            if is_digit:
                vec = proc_digit.ravel()                
                pred = classifier.predict(vec)
                #print(pred)
                #print(vec)
                #plot_img(proc_digit)
                

    plot_digits(raw_digits)   

# TODO: prepare a pickle loading for the dataset (img from .jpg, "labels" from .dat)

if __name__ == '__main__':
    #img = load_img('datasets/mathworks/segmentation_data/raw_data/images/0010_05.jpg')
    for i in range(14, 15):
        img_path = 'datasets/bernard/{}.jpg'.format(i)
        print(img_path)
        img = load_img(img_path)
        launch_pipeline(img)

    

