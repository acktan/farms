import pandas as pd
from datetime import timedelta
import re

class DataLoader():
    """
    Load the training and testing data
    """
    
    def __init__(self, config):
        self.config = config
        self.directory = config['paths']['directory']
        self.input_folder = config['paths']['input_folder']
        
    def column_imputer(self, df: pd.DataFrame, 
                       cols_to_fill: dict[str, object]) -> pd.DataFrame:
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
    
    def clean_date_columns(self, 
                           df: pd.DataFrame) -> pd.DataFrame:
        """ Convert date columns to datetime
        
        Args:
            df: dataframe containing mushroom info
        
        Returns:W
            Dataframe with converted date columns
        
        """
        df['Pre-wetting date'] = pd.to_datetime(df['Pre-wetting date'],
                                                format="%d-%b-%y",
                                                errors='coerce')
        df['Evacuation date (B)'] = pd.to_datetime(df['Evacuation date (B)'], 
                                                   format="%d-%b-%y",
                                                   errors='coerce')
        return df
    
    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return train and future data
        
        Args:
            conf: config file
            
        Returns:
            train dataset: data on which to train and validate the model
            future: dataset to predict, with no mushroom data available
        """
        
        path = self.directory + self.input_folder
        dataPath = path + self.conf['paths']['mushroom_data']
        df = pd.read_csv(dataPath)
        
        percentColumns = df.columns[[4, 5, 8, 11, 12]]
        numericalCols = df.columns[2:9]
        targetColumn = ['Kg/bag (White & Brown)']
        df[percentColumns] = df[percentColumns].apply(lambda x : x.str.strip('%').astype(float)/100)
        df[numericalCols] = df[numericalCols].apply(pd.to_numeric, errors='coerce', downcast='float')
        df[targetColumn] = df[targetColumn].apply(pd.to_numeric, errors='coerce', downcast='float')
        df[numericalCols] = df[numericalCols].applymap(lambda x: re.sub("[^0-9]", "", x) if type(x) == str else x)
        df = self.clean_date_columns(df)
        df = self.column_imputer(df, numericalCols)
        futures = df[df['Kg/bag (White & Brown)'].isna()]
        df = df[~df['Kg/bag (White & Brown)'].isna()]

        return df[df.columns[0:9]], futures[futures.columns[0:9]]
        