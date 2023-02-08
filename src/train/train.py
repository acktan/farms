import logging
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)

class Train():
    """Train regression model
    
    Args:
        X_train: X training dataframe
        y_train: y training dateframe
    """
    
    def __init__(self, 
                 X_train: pd.DataFrame, 
                 y_train: pd.DataFrame):
        self.X_train = X_train
        self.y_train = y_train
        
    def train(self, 
              model: BaseEstimator, 
              date_columns: list = ['day', 'month']) -> BaseEstimator:
        """Train regression model
        
        Args:
            model: model to be used for regression
            date_columns = date columns to be considered for model
            
        Returns:
            pipe = trained model
        """
        
        preprocessor = ColumnTransformer(
                [
                    ('dateEncode', 'passthrough', date_columns),
                    ('imputer', 'passthrough', self.X_train.columns[2:9]),
                ]
        )

        regressor = model
        pipe = make_pipeline(preprocessor, regressor)

        pipe.fit(self.X_train, self.y_train)
        
        return pipe
        
        