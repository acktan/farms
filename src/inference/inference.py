from distutils.command.config import config
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

class Inference():
    def __init__(self,
                 config: dict[str, object],
                 model,
                 train_data,
                 future_data) -> None:
        """Infer model performance and predict future data
        """
        self.config = config
        self.model = model
        self.train_data = train_data
        self.future_data = future_data
        
    def predict_future(self):
        yhat = self.model.predict(self.future_data)
        return yhat
    
        
