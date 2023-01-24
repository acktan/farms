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
        self.config = config
        self.model = model
        self.train_data = train_data
        self.future_data = future_data
        
    def get_feature_importance(self):
        dateCols = self.train_data[0:2]
        plt.style.use('default')
        inputCols = self.train_data.columns[2:18].tolist() + dateCols
        plt.figure().set_size_inches(16.5, 8.5)
        plt.bar(range(len(inputCols)), self.model.named_steps['extratreesregressor'].feature_importances_)
        plt.title('Feature importance')
        plt.xticks(range(len(inputCols)), inputCols, rotation='vertical')
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.show()
        
    def 