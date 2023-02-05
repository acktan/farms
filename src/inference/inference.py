import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

class Inference():
    def __init__(self,
                 config: dict[str, object],
                 model: BaseEstimator,
                 train_data: pd.DataFrame,
                 future_data: pd.DataFrame) -> None:
        """Infer model performance and predict future data
        """
        self.config = config
        self.model = model
        self.train_data = train_data
        self.future_data = future_data
        
    def predict_future(self) -> np.array:
        """Predict data using future data
        
        Returns:
            Predicted mushroom growth values
        """
        return self.model.predict(self.future_data)
                

