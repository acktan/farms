import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import typing

class DataLoader():
    """
    Load the training and testing data
    """
    
    def __init__(self, config):
        self.config = config
        self.directory = config['paths']['directory']
        self.input_folder = config['paths']['input_folder']
        
    def column_imputer(df: pd.DataFrame, 
                       cols_to_fill: list) -> pd.DataFrame:
        """Impute missing row data with mean
        
        Args:
            df: dataframe containing mushroom info
            cols_to_fill: columns to impute
            
        Returns:
            Dataframe with imputed values
        
        """
        df['Pre-wetting date'] = df['Pre-wetting date'].fillna(df['Evacuation date (B)'] - timedelta(14))
        diff_we = (df['Pre-wetting date'] - df['Evacuation date (B)']).mean()
        df['Evacuation date (B)'] = df['Evacuation date (B)'].fillna(df['Pre-wetting date'] - diff_we)
        mean = df[cols_to_fill].mean()
        df[cols_to_fill] = df[cols_to_fill].fillna(mean)
        return df