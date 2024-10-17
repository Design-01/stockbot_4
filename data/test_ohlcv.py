import unittest
from datetime import datetime
from data.ohlcv import OHLCV
import pandas as pd

class TestOHLCV(unittest.TestCase):

    def setUp(self):
        self.ohlcv = OHLCV(api_key="YOUR_API_KEY")

    def test_get_end_date(self):
        end_date = self.ohlcv.get_end_date(start_date='2021-01-01', days=5)
        self.assertEqual(end_date, '2021-01-06')

    def test_get_start_date(self):
        start_date = self.ohlcv.get_start_date(end_date='2021-01-06', days=5)
        self.assertEqual(start_date, '2021-01-01')

    def test_get_interval_options(self):
        intervals = self.ohlcv.get_interval_options()
        self.assertIn('1day', intervals)

    def test_get_source_options(self):
        sources = self.ohlcv.get_source_options()
        self.assertIn('twelve_data', sources)

    def test_get_list_of_stored_data(self):
        # This test assumes there is data in the ohlcv_data_store
        data_list = self.ohlcv.get_list_of_stored_data()
        self.assertIsInstance(data_list, pd.DataFrame)

    def test_get_stored_data(self):
        # This test assumes there is stored data for TSLA
        data = self.ohlcv.get_stored_data("twelve_data", "TSLA", "1day", start_date='2021-01-01', end_date='2021-01-06', returnAs='df')
        self.assertIsInstance(data, pd.DataFrame)

    def test_get_live_data(self):
        # This test is a placeholder for live data retrieval
        data = self.ohlcv.get_live_data("twelve_data", "TSLA", "1day", start_date='2021-01-01', end_date='2021-01-06', returnAs='dict')
        self.assertIsInstance(data, dict)

if __name__ == '__main__':
    unittest.main()
