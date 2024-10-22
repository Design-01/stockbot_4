from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from chart.chart import Chart
from strategies.ta import TA
from strategies.signals import Signals
import pandas as pd

@dataclass
class Frame:
    symbol: str
    trading_hours: List[Tuple[str, str]] = field(default_factory=lambda: [("09:30", "16:00")])


    def __post_init__(self):
        self.traders = []
        self.data = pd.DataFrame()
        self.ta = []
        self.sigs = []
        self.chart = None

    #£ Working
    def load_ohlcv(self, ohlcv: pd.DataFrame):
        if self.data.empty:
            self.data = ohlcv
        else:
            print(ohlcv)
            combined_data = pd.concat([self.data, ohlcv])
            # Drop duplicate indexes, keeping the last occurrence
            self.data = combined_data[~combined_data.index.duplicated(keep='last')].sort_index()

    #£ Working
    def setup_chart(self):  
        self.chart = Chart(title=self.symbol, rowHeights=[0.1, 0.1, 0.1, 0.8], height=800, width=800)
        self.chart.add_candles_and_volume(self.data)
        # self.chart.add_trading_hours(self.data, self.trading_hours)

    #£ Working
    def update_data(self, new_data):
        """
        Update the main DataFrame with new data by merging based on the datetime index.
        Ensures OHLCV columns are always first in the returned DataFrame.
        Only overwrites old data if new data is not NaN.
        
        Parameters:
        df (pd.DataFrame): The main DataFrame to update
        new_data (pd.Series or pd.DataFrame): The new data to merge and update
        
        Returns:
        pd.DataFrame: The updated DataFrame with OHLCV columns first
        """
        # Ensure new_data is a DataFrame
        if isinstance(new_data, pd.Series):
            new_data = new_data.to_frame()
        
        # Ensure both DataFrames have a datetime index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Main DataFrame must have a datetime index")
        if not isinstance(new_data.index, pd.DatetimeIndex):
            raise ValueError("New data must have a datetime index")
        
        # Create a copy of the main DataFrame to avoid modifying the original
        updated_df = self.data.copy()
        
        # Update existing columns and add new columns
        for column in new_data.columns:
            if column in updated_df.columns:
                # Update only where new data is not NaN
                mask = new_data.index.isin(updated_df.index) & ~new_data[column].isna()
                updated_df.loc[new_data.index[mask], column] = new_data.loc[mask, column]
            else:
                # Add new column
                updated_df[column] = new_data[column]
        
        # Define the order of OHLCV columns
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Separate OHLCV columns and other columns
        existing_ohlcv = [col for col in ohlcv_columns if col in updated_df.columns]
        other_columns = [col for col in updated_df.columns if col not in ohlcv_columns]
        
        # Reorder columns: OHLCV first, then other columns
        reordered_columns = existing_ohlcv + other_columns
        self.data = updated_df[reordered_columns]
        return self.data

    #£ Working
    def add_ta(self, ta: TA, style: Dict[str, Any] | List[Dict[str, Any]], chart_type: str = "line", row: int = 1):
        # Check for duplicates
        for existing_ta, existing_style, existing_chart_type, existing_row in self.ta:
            if (existing_ta == ta and 
                existing_style == style and 
                existing_chart_type == chart_type and 
                existing_row == row):
                # Duplicate found, do not add
                return
        
        # No duplicates found, add the new TA
        self.ta.append((ta, style, chart_type, row))

    def add_signals(self, signals: Signals, style: Dict[str, Any] | List[Dict[str, Any]], chart_type: str = "line", row: int = 1):
        # Check for duplicates
        for existing_signals, existing_style, existing_chart_type, existing_row in self.sigs:
            if (existing_signals == signals and 
                existing_style == style and 
                existing_chart_type == chart_type and 
                existing_row == row):
                # Duplicate found, do not add
                return
        self.sigs.append((signals, style, chart_type, row))


    def update_ta_data(self):
        """Updates the data for all the technical indicators in the frame"""
        for ta, style, chart_type, row in self.ta:
            self.data = self.update_data(ta.run(self.data))

    def update_signals_data(self):
        """Updates the data for all the signals in the frame"""
        sig_dict = {}
        for signals, style, chart_type, row in self.sigs:
            sig_dict[f'SigL_{signals.name}'] = signals.run('LONG', self.data)
            sig_dict[f'SigS_{signals.name}'] = signals.run('SHORT', self.data)

        self.data = self.update_data(pd.DataFrame(sig_dict, index=[self.data.index[-1]]))
        


    def plot(self, width: int = 1400, height: int = 800, trading_hours: bool = False):
        self.chart.refesh(self.data)
        for indicator, style, chart_type, row in self.ta:
            indicator_data = self.data[indicator.names] # get the data for the indicator which should be updated first 
            self.chart.add_ta(indicator_data, style, chart_type, row)
        for signals, style, chart_type, row in self.sigs:
            sig_data = self.data[[f'SigL_{signals.name}', f'SigS_{signals.name}']]
            self.chart.add_signals(sig_data, style, chart_type, row)
        if trading_hours: self.chart.add_trading_hours(self.data, self.trading_hours)
        self.chart.show(width=width, height=height)


