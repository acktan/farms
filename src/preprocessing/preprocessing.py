import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


class Preprocessing():
    """Preprocess dataset
    
    Args:
        df: the cleaned input mushroom data
    Returns:
        train_set: train set
        val_set: validation set
        
    """
    def __init__(self, 
                 df: pd.DataFrame) -> pd.DataFrame:
        self.df = df
    
    def _dateEncode(self, X: pd.DataFrame) -> pd.Dataframe:
        """Convert datetime to cyclical and numerical features
        
        Args:
            X: cleaned input mushroom data
            
        Returns:
            X: cleaned input mushroom data with new date features
        """
        #cyclical encoding of dates
        X = X.copy()
        year_norm = 2 * np.pi * X['Pre-wetting date'].dt.year / X['Pre-wetting date'].dt.year.max()
        month_norm = 2 * np.pi * X['Pre-wetting date'].dt.month / X['Pre-wetting date'].dt.month.max()
        day_norm = 2 * np.pi * X['Pre-wetting date'].dt.day / X['Pre-wetting date'].dt.day.max()
        X.loc[:, 'year_sin'] = np.sin(year_norm)
        X.loc[:, 'year_cos'] = np.cos(year_norm)
        X.loc[:, 'month_sin'] = np.sin(month_norm)
        X.loc[:, 'month_cos'] = np.cos(month_norm)
        X.loc[:, 'day_sin'] = np.sin(day_norm)
        X.loc[:, 'day_cos'] = np.cos(day_norm)
        X.loc[:, 'year'] = X['Pre-wetting date'].dt.year
        X.loc[:, 'month'] = X['Pre-wetting date'].dt.month
        X.loc[:, 'day'] = X['Pre-wetting date'].dt.day
        return X
    
    def get_hist_weather(self) -> pd.DataFrame:
        """Get online historical weather data
        
        Returns:
            df: mushroom dataset with aggregate weather data for time span between pre-wetting date and evacuation date (B)
        """
        
        enddate = self.df['Evacuation date (B)'].max().strftime("%Y-%m-%d")
        response = requests.get('https://archive-api.open-meteo.com/v1/archive?latitude=-1.50&longitude=29.63&start_date=2018-01-13&end_date=' 
                        + enddate 
                        +'&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,surface_pressure,rain,direct_radiation,windspeed_10m,soil_temperature_0_to_7cm,soil_moisture_0_to_7cm')
        weatherData = pd.DataFrame(response.json()['hourly'])
        weatherData['time'] = pd.to_datetime(weatherData['time'], format='%Y-%m-%d')
        weatherData = weatherData.groupby(weatherData.time.dt.date).agg({'temperature_2m':'mean',
                                                   'relativehumidity_2m':'mean',
                                                   'dewpoint_2m':'mean',
                                                   'surface_pressure':'mean',
                                                   'rain':'sum',
                                                   'direct_radiation':'mean',
                                                   'windspeed_10m': 'mean',
                                                   'soil_temperature_0_to_7cm':'mean',
                                                   'soil_moisture_0_to_7cm':'mean'})
        
        arr = pd.DataFrame(columns=weatherData.columns)
        inputData = self.df.loc[:, self.df.columns[0:9]]
        for i in inputData.index:
            startDate = inputData.loc[i, 'Pre-wetting date']
            endDate = inputData.loc[i, 'Evacuation date (B)']
            arr.loc[i] = weatherData.loc[startDate:endDate, :].mean()
            
        df = pd.concat([inputData, arr], axis=1)
        df = pd.concat([df, self.df['Kg/bag (White & Brown)']], axis=1)
        return df
        
    def train_val_split(self, 
                        train_size: float=0.8) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split dataset into train and validation sets
        
        Args:
            train_size: size of the training set
        Returns:
            X_train: X training values
            X_val: X validation values
            y_train: y training values
            y_val: y validation values  
        """
        
        df = self._dateEncode(self.df)
        y = df['Kg/bag (White & Brown)']
        X = df.drop(columns='Kg/bag (White & Brown)')
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_size, random_state=42)
        return X_train, X_val, y_train, y_val
        
        

        
    