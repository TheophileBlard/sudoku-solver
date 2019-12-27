from abc import ABC, abstractmethod
import numpy as np
import joblib


class DigitClassifier(ABC):
    @abstractmethod
    def classify(self, img_digits):
        pass

    def predict(self, img_digits, preprocessing_func=None):
        if preprocessing_func:
            img_digits = preprocessing_func(img_digits)        
        return self.classify(img_digits)


class ScikitClassifier(DigitClassifier):
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        pass
        
    def classify(self, img_digits):
        #TODO: should use save model
        #output = self.model.predict(img_digits[0][0])
        output = np.zeros((9,9))      
        return output.astype(np.uint8)
