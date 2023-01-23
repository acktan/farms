import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

class Evaluator():
    """Evaluate model and variable importances"""
    def __init__(self, 
                 config: dict[str, object],):
        self.config = config
        
    def evaluate_model(self,                  
                       X_test: pd.DataFrame,
                       y_test: pd.Series,
                       model):
        """Evaluate model performance
        
        Args:
            X_test: x val dataframe
            y_test: y val dataframe
            model: 
        """
        yhat = model.predict(X_test)
        score = mean_absolute_error(yhat, y_test)
        return score