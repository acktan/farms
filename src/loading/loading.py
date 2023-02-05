import logging
import re
import pandas as pd
from datetime import timedelta

logger = logging.getLogger(__name__)

class DataLoader():
    """
    Load the training and testing data
    """
    
    def __init__(self, 
                 config: dict[str, str]):
        self.config = config
        self.directory = config['paths']['directory']
        self.input_folder = config['paths']['input_folder']
        
    def column_imputer(self, 
                       cols_to_fill: dict[str, object]) -> pd.DataFrame:
        """Impute missing row data with mean
        
        Args:
            df: dataframe containing mushroom info
            cols_to_fill: columns to impute
            
        Returns:
            df:ataframe with imputed values
        
        """
        logger.debug("Started column imputation")
        self.df['Pre-wetting date'] = self.df['Pre-wetting date'].fillna(self.df['Evacuation date (B)'] - timedelta(14))
        diff_we = (self.df['Pre-wetting date'] - self.df['Evacuation date (B)']).mean()
        self.df['Evacuation date (B)'] = self.df['Evacuation date (B)'].fillna(self.df['Pre-wetting date'] - diff_we)
        mean = self.df[cols_to_fill].mean()
        self.df[cols_to_fill] = self.df[cols_to_fill].fillna(mean)
        logger.debug("Completed column imputation")
        return self.df
    
    def clean_date_columns(self) -> pd.DataFrame:
        """ Convert date columns to datetime
        
        Args:
            df: dataframe containing mushroom info
        
        Returns:
            df: dataframe with converted date columns
        
        """
        logger.debug("Started column cleaning")

        self.df['Pre-wetting date'] = pd.to_datetime(self.df['Pre-wetting date'],
                                                format="%d-%b-%y",
                                                errors='coerce')
        self.df['Evacuation date (B)'] = pd.to_datetime(self.df['Evacuation date (B)'], 
                                                   format="%d-%b-%y",
                                                   errors='coerce')
        logger.debug("Completed column cleaning")
        return self.df
    
    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return train and future data
        
        Args:
            conf: config file
            
        Returns:
            train dataset: data on which to train and validate the model
            future: dataset to predict, with no mushroom data available
        """
        
        logger.debug("Collecting data")
        
        path = self.directory + self.input_folder
        dataPath = path + self.conf['paths']['mushroom_data']
        self.df = pd.read_csv(dataPath)
        
        percentColumns = self.df.columns[[4, 5, 8, 11, 12]]
        numericalCols = self.df.columns[2:9]
        targetColumn = ['Kg/bag (White & Brown)']
        self.df[percentColumns] = self.df[percentColumns].apply(lambda x : x.str.strip('%').astype(float)/100)
        self.df[numericalCols] = self.df[numericalCols].apply(pd.to_numeric, errors='coerce', downcast='float')
        self.df[targetColumn] = self.df[targetColumn].apply(pd.to_numeric, errors='coerce', downcast='float')
        self.df[numericalCols] = self.df[numericalCols].applymap(lambda x: re.sub("[^0-9]", "", x) if type(x) == str else x)
        self.df = self.clean_date_columns()
        self.df = self.column_imputer(numericalCols)
        futures = self.df[self.df['Kg/bag (White & Brown)'].isna()]
        self.df = self.df[~self.df['Kg/bag (White & Brown)'].isna()]
        
        logger.debug("Finished collecting data")

        return self.df[self.df.columns[0:9]], futures[futures.columns[0:9]]
    

        