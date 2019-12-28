from abc import ABC, abstractmethod
import numpy as np
import joblib


class DigitClassifier(ABC):
    @abstractmethod
    def classify(self, img_digits):
        pass

    def predict(self, img_digits, preprocessing_func=None):                
        return self.classify(img_digits)


class ScikitLearnClassifier(DigitClassifier):
    def __init__(self, model_path):
        data = joblib.load(model_path)        
        self.model = data["model"]
        if 'scaler' in data:
            self.scaler = data["scaler"]
        
        
    def classify(self, img_digit):        
        #if self.scaler:
        #    img_digit = self.scaler.transform([img_digit, ])[0]        
        output = self.model.predict([img_digit, ])        
        return output
