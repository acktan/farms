import unittest
from datetime import datetime, timedelta

import pandas as pd

from src.loading import loading


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.config = {
            'paths': {
                'directory': '',
                'input_folder': '../data/',
                'mushroom_data': 'allBatch.csv'
            }
        }
        self.dataloader = loading.DataLoader(self.config)
        self.df = pd.DataFrame({
            'Pre-wetting date': ['1-Mar-22', '2-Mar-22', '3-Mar-22'],
            'Evacuation date (B)': ['14-Mar-22', '15-Mar-22', '16-Mar-22'],
            'Column1': [1, 2, 3],
            'Column2': [4, None, 6]
        })
        self.df, _ = self.dataloader.get_data()
        self.cleaned_df = self.dataloader.clean_date_columns()

    
    def test_column_imputer(self):
        cols_to_fill = ['Column1', 'Column2']
        filled_df = self.dataloader.column_imputer(self.cleaned_df, cols_to_fill)
        self.assertFalse(filled_df.isna().any().any())
    
    def test_clean_date_columns(self):
        expected_df = pd.DataFrame({
            'Pre-wetting date': [
                datetime(2022, 3, 1),
                datetime(2022, 3, 2),
                datetime(2022, 3, 3),
            ],
            'Evacuation date (B)': [
                datetime(2022, 3, 14),
                datetime(2022, 3, 15),
                datetime(2022, 3, 16),
            ],
            'Column1': [1, 2, 3],
            'Column2': [4, None, 6],
        })
        pd.testing.assert_frame_equal(self.cleaned_df, expected_df)
    
