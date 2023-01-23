import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

class Train():
    """Train regression model
    
    Args:
        config: config json file
        X_train: X training dataframe
        y_train: y training dateframe
    """
    
    def __init__(self, 
                 config: dict[str, object], 
                 X_train: pd.DataFrame, 
                 y_train: pd.DataFrame):
        self.config = config
        self.X_train = X_train
        self.y_train = y_train
        
    def train(self, 
              model, 
              date_columns: list = ['day', 'month']):
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
                    ('imputer', 'passthrough', self.X_train.columns[2:18]),
                ]
        )

        regressor = model
        pipe = make_pipeline(preprocessor, regressor)

        pipe.fit(self.X_train, self.y_train)
        
        return pipe
        
        