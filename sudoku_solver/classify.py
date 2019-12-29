from abc import ABC, abstractmethod
import numpy as np
import joblib


class DigitClassifier(ABC):
    def __init__(self, preproc_fun=None):
        self.preproc_fun = preproc_fun   
    
    def preprocess(self, raw_digit):       
        return self.preproc_fun(raw_digit)

    @abstractmethod
    def predict(self, digit_img):
        pass


class ScikitLearnClassifier(DigitClassifier):
    def __init__(self, model_path, preproc_fun=None):
        super().__init__(preproc_fun)
        data = joblib.load(model_path)        
        self.model = data["model"]
        if 'scaler' in data:
            self.scaler = data["scaler"]      
        
    def predict(self, proc_digit):              
        return self.model.predict([proc_digit.ravel(), ])[0]
