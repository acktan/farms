import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)

class Evaluator():
    """Evaluate model and variable importances"""
    def __init__(self, 
                 config: dict[str, object],
                 train_data: pd.DataFrame,
                 model: BaseEstimator):
        self.config = config
        self.train_data = train_data
        self.model = model
        
    def evaluate_model(self,                  
                       X_test: pd.DataFrame,
                       y_test: pd.Series,) -> float:
                     
        """Evaluate model performance
        
        Args:
            X_test: x val dataframe
            y_test: y val dataframe
        """
        yhat = self.model.predict(X_test)
        score = mean_absolute_error(yhat, y_test)
        return score
    
    def get_feature_importance(self) -> type[plt.Figure]:
        """Return feature importance of train data
        
        Returns:
            plot with feature importances of training data
        """
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
        
    def compare_test_values(self, 
                            X_test: pd.DataFrame, 
                            y_test: pd.DataFrame) -> type[plt.Figure]:
        """Comparison of real and predicted test values
        
        Args:
            X_test: x val dataframe
            y_test: y val dataframe

        Returns:
            plot comparison of real and predicted test values
        """
        plt.style.use('default')
        plt.figure().set_size_inches(16.5, 8.5)
        yhat = self.model.predict(X_test)
        width = 0.15

        # Plot the first group of bars (yhat_small) at x-coordinates 0, 1, 2, etc.
        plt.bar(np.arange(len(y_test)), yhat, width, label='Prediction (with weather)', yerr=mean_absolute_error(y_test, yhat), color='#a0bd50')
        plt.bar(np.arange(len(y_test)) + width, y_test, width, label='True', color='#5f5d5e')
        plt.legend()
        plt.xlabel('Test Data')
        plt.ylabel('Predictions')
        plt.title('Predictions vs. True Values')