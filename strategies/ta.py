import pandas as pd
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Union, Dict, Optional, Any

def preprocess_data(func):
    def wrapper(self, data: pd.DataFrame, *args, **kwargs):
        data = self.compute_rows_to_update(data, self.names, self.rowsToUpdate).copy()
        return func(self, data, *args, **kwargs)
    return wrapper




@dataclass
class TA(ABC):
    column: str = None
    name: str = None
    names: List[str] = field(default_factory=list)

    @abstractmethod
    def run(self, data: pd.DataFrame) -> pd.Series | pd.DataFrame:
        pass

    
    def compute_rows_to_update(self, df, column_names, rows_to_update):
            """
            Compute the slice of the DataFrame that needs to be updated based on the columns provided.
            
            Parameters:
            df (pd.DataFrame): The main DataFrame
            column_names (list): The names of the columns to check
            rows_to_update (int): The number of rows to add to the last valid index
            
            Returns:
            pd.DataFrame: The sliced DataFrame that needs to be updated
            """
            if isinstance(column_names, str):
                column_names = [column_names]
            
            last_valid_indices = []
            
            for column_name in column_names:
                if column_name in df.columns:
                    last_valid_index = df[column_name].last_valid_index()
                    if last_valid_index is not None:
                        lookback_index = df.index.get_loc(last_valid_index)
                        last_valid_indices.append(lookback_index)
            
            if not last_valid_indices:
                lookback_index = 0
            else:
                lookback_index = max(min(last_valid_indices) - rows_to_update, 0) # + 1 # added 1 just to be sure

            return df.iloc[-lookback_index:]

    @staticmethod
    def normalize(series: pd.Series, max_value: float) -> pd.Series:
        """Efficient normalization to -100 to 100 range"""
        return (series / max_value).clip(-1, 1) * 100

# class is used for prebatching TA data
@dataclass
class TAData:
    ta: TA
    style: Dict[str, Any] | List[Dict[str, Any]] = field(default_factory=dict)
    chart_type: str = "line"
    row: int = 1
    nameCol: str = ''

@dataclass
class Ffill:
    """Forward fill missing values in a DataFrame"""
    name: str = 'Ffill'
    colToFfill: str = ''

    def __post_init__(self):
        self.name = f"FFILL_{self.colToFfill}"
        self.names = [self.name]
    
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.colToFfill not in data.columns:
            data[self.name] = np.nan
            return data
        
        data[self.name] = data[self.colToFfill].ffill()
        return data
    
@dataclass
class AddColumn:
    """Add new columns to a DataFrame"""
    column: str = ''

    def __post_init__(self):
        self.name = self.column
        self.names = [self.name]
    
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.name] = data[self.column]
        return data


@dataclass
class Levels(TA):
    """Add horizontal levels to a DataFrame"""
    level: str = '' # pre_mkt_high, pre_mkt_low, pre_mkt_close, pre_mkt_open, pre_mkt_volume, regular_high, regular_low, regular_close, regular_open, regular_volume, post_mkt_high, post_mkt_low, post_mkt_close, post_mkt_open, post_mkt_volume, prev_day_high, prev_day_low, prev_day_close, prev_day_open, prev_day_volume

    def __post_init__(self):
        self.name = self.level
        self.names = [self.name]
    
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        # Apply levels to each day
        for date in data.index.normalize().unique():
            day_data = data[data.index.normalize() == date]

            if 'pre_mkt' in self.level:
                df = day_data.between_time('00:00:00', '09:30:00')
                if not df.empty:
                    if   self.level == 'pre_mkt_high':   data.loc[df.index, self.name] = df['high'].cummax()
                    elif self.level == 'pre_mkt_low':    data.loc[df.index, self.name] = df['low'].cummin()
                    elif self.level == 'pre_mkt_close':  data.loc[df.index, self.name] = df['close'].iloc[-1]
                    elif self.level == 'pre_mkt_open':   data.loc[df.index, self.name] = df['open'].iloc[0]
                    elif self.level == 'pre_mkt_volume': data.loc[df.index, self.name] = df['volume'].cumsum()

            if 'intraday' in self.level:
                df = day_data.between_time('09:30:00', '16:00:00')
                if not df.empty:
                    if   self.level == 'intraday_high':   data.loc[df.index, self.name] = df['high'].cummax()
                    elif self.level == 'intraday_low':    data.loc[df.index, self.name] = df['low'].cummin()
                    elif self.level == 'intraday_close':  data.loc[df.index, self.name] = df['close'].iloc[-1]
                    elif self.level == 'intraday_open':   data.loc[df.index, self.name] = df['open'].iloc[0]
                    elif self.level == 'intraday_volume': data.loc[df.index, self.name] = df['volume'].cumsum()

            if 'post_mkt' in self.level:
                df = day_data.between_time('16:00:00', '23:59:59')
                if not df.empty:
                    if   self.level == 'post_mkt_high':   data.loc[df.index, self.name] = df['high'].cummax()
                    elif self.level == 'post_mkt_low':    data.loc[df.index, self.name] = df['low'].cummin()
                    elif self.level == 'post_mkt_close':  data.loc[df.index, self.name] = df['close'].iloc[-1]
                    elif self.level == 'post_mkt_open':   data.loc[df.index, self.name] = df['open'].iloc[0]
                    elif self.level == 'post_mkt_volume': data.loc[df.index, self.name] = df['volume'].cumsum()

        # Apply previous day levels
        if 'prev_day' in self.level:
            for i, date in enumerate(data.index.normalize().unique()[1:], start=1):
                prev_date = data.index.normalize().unique()[i-1]
                prev_day_data = data[data.index.normalize() == prev_date]
                df = prev_day_data.between_time('09:30:00', '16:00:00')
                if not df.empty:
                    if   self.level == 'prev_day_high':   data.loc[data.index.normalize() == date, self.name] = df['high'].max()
                    elif self.level == 'prev_day_low':    data.loc[data.index.normalize() == date, self.name] = df['low'].min()
                    elif self.level == 'prev_day_close':  data.loc[data.index.normalize() == date, self.name] = df['close'].iloc[-1]
                    elif self.level == 'prev_day_open':   data.loc[data.index.normalize() == date, self.name] = df['open'].iloc[0]
                    elif self.level == 'prev_day_volume': data.loc[data.index.normalize() == date, self.name] = df['volume'].sum()

        return data


@dataclass
class MA(TA):
    maCol: str = 'close'
    period: int = 20

    def __post_init__(self):
        self.name = f"MA_{self.maCol[:2]}_{self.period}"
        self.names = f"MA_{self.maCol[:2]}_{self.period}"
        self.rowsToUpdate = self.period 

    @preprocess_data
    def run(self, data: pd.DataFrame) -> pd.Series:
        return data[self.maCol].rolling(window=self.period).mean().rename(self.name)




@dataclass
class VWAP(TA):
    interval: str = 'Session' # 'Session', 'Day', 'Week', 'Month', 'Year'


    def __post_init__(self):
        self.name = f"VWAP"
        self.names = self.name
        self.rowsToUpdate = 10

    @preprocess_data
    def run(self, data: pd.DataFrame) -> pd.Series:
        # # Add datetime index if not present
        # if not isinstance(data.index, pd.DatetimeIndex):
        #     data.index = pd.to_datetime(data.index)
            
        # Create session labels using date
        data['session'] = data.index.date
        
        typical_price = data[self.column]
        volume = data['volume']
        
        # Group by session and calculate VWAP (ie resets every day)
        cumulative_pv = (typical_price * volume).groupby(data['session']).cumsum()
        cumulative_volume = volume.groupby(data['session']).cumsum()
        
        vwap = (cumulative_pv / cumulative_volume).rename(self.name)
        return vwap


@dataclass
class VWAP(TA):
    interval: str = 'Session'  # 'Session', 'Day', 'Week', 'Month', 'Year'

    def __post_init__(self):
        self.validate_inputs()
        self.name = f"VWAP_{self.interval}"
        self.names = self.name
        self.rowsToUpdate = 10

    def _get_period_label(self, index: pd.DatetimeIndex) -> pd.Series:
        """
        Generate period labels based on the specified interval.
        
        Args:
            index (pd.DatetimeIndex): The datetime index of the data
            
        Returns:
            pd.Series: Period labels for grouping
        """
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("Index must be a DatetimeIndex")

        if self.interval.lower() == 'session':
            return pd.Series(index.date, index=index)
        elif self.interval.lower() == 'day':
            return pd.Series(index.date, index=index)
        elif self.interval.lower() == 'week':
            return pd.Series(index.isocalendar().week.astype(str) + 
                           index.isocalendar().year.astype(str), index=index)
        elif self.interval.lower() == 'month':
            return pd.Series(index.strftime('%Y-%m'), index=index)
        elif self.interval.lower() == 'year':
            return pd.Series(index.year, index=index)
        else:
            raise ValueError(f"Invalid interval: {self.interval}. Must be one of: "
                           "'Session', 'Day', 'Week', 'Month', 'Year'")

    @preprocess_data
    def run(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate VWAP for the specified interval.
        
        Args:
            data (pd.DataFrame): DataFrame with price and volume data
            
        Returns:
            pd.Series: VWAP values
        """
        # Ensure we have a datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # Get period labels based on the interval
        period_labels = self._get_period_label(data.index)
        
        typical_price = data[self.column]
        volume = data['volume']
        
        # Calculate cumulative values within each period
        cumulative_pv = (typical_price * volume).groupby(period_labels).cumsum()
        cumulative_volume = volume.groupby(period_labels).cumsum()
        
        # Calculate VWAP
        vwap = (cumulative_pv / cumulative_volume).rename(self.name)
        
        # Handle edge cases where volume is zero
        vwap = vwap.fillna(method='ffill')
        
        return vwap

    def validate_inputs(self) -> None:
        """Validate the input parameters."""
        valid_intervals = {'session', 'day', 'week', 'month', 'year'}
        if self.interval.lower() not in valid_intervals:
            raise ValueError(f"Invalid interval: {self.interval}. "
                           f"Must be one of: {', '.join(valid_intervals)}")


@dataclass
class MACD(TA):
    fast: int = 12
    slow: int = 26
    signal: int = 9
    fastcol: str = 'close'
    slowcol: str = 'close'
    signalcol: str = 'close'

    def __post_init__(self):
        self.macd_name = f"MACD_{self.signalcol[:2]}_{self.fast}_{self.slow}_{self.signal}_MACD"
        self.signal_name = f"MACD_{self.signalcol[:2]}_{self.fast}_{self.slow}_{self.signal}_Signal"
        self.histogram_name = f"MACD_{self.signalcol[:2]}_{self.fast}_{self.slow}_{self.signal}_Histogram"
        self.names = [self.macd_name, self.signal_name, self.histogram_name]
        self.rowsToUpdate = max(self.slow, self.signal, self.fast) 

    @preprocess_data
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        fast_ema = data[self.fastcol].ewm(span=self.fast, adjust=False).mean()
        slow_ema = data[self.slowcol].ewm(span=self.slow, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=self.signal, adjust=False).mean()
        histogram = macd - signal_line
        
        
        return pd.DataFrame({
            self.macd_name: macd,
            self.signal_name: signal_line,
            self.histogram_name: histogram
        })


@dataclass
class RSATRMA(TA):
    "Realtive Strength Average True Range Moving Average "
    comparisonPrefix: str = 'SPY'
    ma: int = 14
    atr: int = 14

    def __post_init__(self):
        self.name_rs = f"RS_{self.comparisonPrefix}_{self.atr}"
        self.name_atr = f"RS_MA_{self.comparisonPrefix}_{self.atr}_{self.ma}"
        self.names = [self.name_rs, self.name_atr]
        self.rowsToUpdate = max(self.atr, self.ma) + 1

    @preprocess_data
    def run(self, data: pd.DataFrame) -> pd.Series:
        # compute market trading range 
        mkt_tr = data[f"{self.comparisonPrefix}_high"] - data[f"{self.comparisonPrefix}_low"]
        mkt_atr = mkt_tr.rolling(window=self.atr).mean()

        #  compute market price change but normalise using the atr
        mkt_change = (data[f"{self.comparisonPrefix}_close"] - data[f"{self.comparisonPrefix}_close"].shift(1)) / mkt_atr

        # compute stock trading range
        stk_tr = data['high'] - data['low']
        stk_atr = stk_tr.rolling(window=self.atr).mean()

        # compute stock price change but normalise using the atr
        stk_change = (data['close'] - data['close'].shift(1)) / stk_atr

        # compute relative strength and relative strength moving average
        data[self.name_rs]  = stk_change - abs(mkt_change) # mkt can be negative so 3 - 2 = 1 ..  but alos want 3- -2 = 1
        data[self.name_atr] = data[self.name_rs].rolling(window=self.ma).mean()

        return data[[self.name_rs, self.name_atr]].round(2)


@dataclass
class HPLP(TA):
    hi_col: str = 'high'
    lo_col: str = 'low'
    span: int = 5

    def __post_init__(self):
        self.name_hp = f"HP_{self.hi_col[:2]}_{self.span}"
        self.name_lp = f"LP_{self.lo_col[:2]}_{self.span}"
        self.names = [self.name_hp, self.name_lp]
        self.rowsToUpdate = 200

    @preprocess_data
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Calculate rolling max and min with min_periods=1
        window = self.span * 2 + 1
        high_max = df[self.hi_col].rolling(window=window, center=True, min_periods=1).max()
        low_min = df[self.lo_col].rolling(window=window, center=True, min_periods=1).min()

        # Identify high and low points
        hi = self.hi_col
        lo = self.lo_col
        df[self.name_hp] = df[hi].where((df[hi] == high_max) & (df[hi].shift(1) != high_max), np.nan)
        df[self.name_lp] = df[lo].where((df[lo] == low_min) & (df[lo].shift(1) != low_min), np.nan)

        # Set the last 3 values to NaN  to avoid lookahead bias
        df.iloc[-3:, df.columns.get_loc(self.name_hp)] = np.nan
        df.iloc[-3:, df.columns.get_loc(self.name_lp)] = np.nan

        return df[[self.name_hp, self.name_lp]]


@dataclass
class LowestHighest:
    hi_col: str = 'high'
    lo_col: str = 'low'
    span: int = 5

    def __post_init__(self):
        self.name_hp = f"HiIst_{self.hi_col[:2]}_{self.span}"
        self.name_lp = f"LoIst_{self.lo_col[:2]}_{self.span}"
        self.names = [self.name_hp, self.name_lp]
        self.rowsToUpdate = 200

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        # df = data.copy()
        # Assign the rolling max and min to new columns
        df[self.name_hp] = df[self.hi_col].rolling(window=self.span, min_periods=1).max()
        df[self.name_lp] = df[self.lo_col].rolling(window=self.span, min_periods=1).min()
        return df[[self.name_hp, self.name_lp]]

    
@dataclass
class ATR(TA):
    hi_col: str = 'high'
    lo_col: str = 'low'
    span: int = 14  # Common default span for ATR

    def __post_init__(self):
        self.name = f"ATR_{self.span}"
        self.names = [self.name]
        self.rowsToUpdate = 200

    @preprocess_data
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Calculate True Range (TR)
        df['TR'] = df[self.hi_col] - df[self.lo_col]

        # Calculate ATR using a rolling mean of the True Range
        df[self.name] = df['TR'].rolling(window=self.span, min_periods=1).mean()

        # Drop the intermediate 'TR' column
        df.drop(columns=['TR'], inplace=True)

        return df[[self.name]]

import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class SupResOld(TA):
    hi_point_col: str = 'HP'
    lo_point_col: str = 'LP'
    pointsAgo: int = 1
    tolerance: float = 0.01
    atr_period: int = 14

    def __post_init__(self):
        self.name_support = f"Sup_{self.pointsAgo}"
        self.name_resistance = f"Res_{self.pointsAgo}"
        self.name_res_upper = f"{self.name_resistance}_Upper"
        self.name_res_lower = f"{self.name_resistance}_Lower"
        self.name_sup_upper = f"{self.name_support}_Upper"
        self.name_sup_lower = f"{self.name_support}_Lower"
        self.names = [self.name_support, self.name_resistance, self.name_res_upper, self.name_res_lower, self.name_sup_upper, self.name_sup_lower]
        self.rowsToUpdate = 200

    def calculate_atr(self, data):
        high = data['high']
        low = data['low']
        close = data['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=self.atr_period).mean()

    def initialize_columns(self, df):
        for col in self.names:
            df[col] = np.nan

    def find_levels(self, all_points, last_close):
        resistance_points = all_points[all_points > last_close]
        resistance_level = resistance_points[self.pointsAgo - 1] if len(resistance_points) >= self.pointsAgo else None

        support_points = all_points[all_points < last_close][::-1]  # Reverse order for support
        support_level = support_points[self.pointsAgo - 1] if len(support_points) >= self.pointsAgo else None

        return resistance_level, support_level

    def update_levels(self, df, level, level_name, upper_name, lower_name, all_points):
        if level is not None:
            level_index = df.index[(df[self.hi_point_col] == level) | (df[self.lo_point_col] == level)][-1]
            df.loc[level_index:, level_name] = level

            atr_at_level = df.loc[level_index, 'ATR']
            upper_bound = level + (atr_at_level * self.tolerance)
            lower_bound = level - (atr_at_level * self.tolerance)

            points_in_range = all_points[(all_points >= lower_bound) & (all_points <= upper_bound)]

            if len(points_in_range) > 0:
                df.loc[level_index:, upper_name] = points_in_range.max()
                df.loc[level_index:, lower_name] = points_in_range.min()
            else:
                df.loc[level_index:, upper_name] = upper_bound
                df.loc[level_index:, lower_name] = lower_bound

    def check_existing_bounds(self, df):
        previous_res_upper = f"Res_{self.pointsAgo - 1}_Upper"
        previous_res_lower = f"Res_{self.pointsAgo - 1}_Lower"
        previous_sup_upper = f"Sup_{self.pointsAgo - 1}_Upper"
        previous_sup_lower = f"Sup_{self.pointsAgo - 1}_Lower"

        last_res_upper = last_res_lower = last_sup_upper = last_sup_lower = None

        if previous_res_upper in df.columns and previous_res_lower in df.columns:
            if not df[previous_res_upper].dropna().empty:
                last_res_upper = df[previous_res_upper].dropna().iloc[-1]
            if not df[previous_res_lower].dropna().empty:
                last_res_lower = df[previous_res_lower].dropna().iloc[-1]

        if previous_sup_upper in df.columns and previous_sup_lower in df.columns:
            if not df[previous_sup_upper].dropna().empty:
                last_sup_upper = df[previous_sup_upper].dropna().iloc[-1]
            if not df[previous_sup_lower].dropna().empty:
                last_sup_lower = df[previous_sup_lower].dropna().iloc[-1]

        return last_res_upper, last_res_lower, last_sup_upper, last_sup_lower

    @preprocess_data
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        last_close = df['close'].iloc[-1]

        # Calculate ATR
        df['ATR'] = self.calculate_atr(df)

        # Combine high points and low points into a single sorted list
        all_points = pd.concat([df[self.hi_point_col], df[self.lo_point_col]]).sort_values().dropna().unique()

        # Check for existing bounds from previous instance
        last_res_upper, last_res_lower, last_sup_upper, last_sup_lower = self.check_existing_bounds(df)

        # Use existing bounds if they exist
        if last_res_upper is not None and last_res_lower is not None:
            last_close = (last_res_upper + last_res_lower) / 2

        if last_sup_upper is not None and last_sup_lower is not None:
            last_close = (last_sup_upper + last_sup_lower) / 2

        # Find resistance and support levels
        resistance_level, support_level = self.find_levels(all_points, last_close)

        # Initialize new columns
        self.initialize_columns(df)

        # Update resistance levels
        self.update_levels(df, resistance_level, self.name_resistance, self.name_res_upper, self.name_res_lower, all_points)

        # Update support levels
        self.update_levels(df, support_level, self.name_support, self.name_sup_upper, self.name_sup_lower, all_points)

        # Remove the temporary ATR column
        df = df.drop('ATR', axis=1)

        return df


@dataclass
class SupRes(TA):
    hi_point_col: str = 'HP'
    lo_point_col: str = 'LP'
    atr_col: str = 'ATR'
    tolerance: float = 0.01
    rowsToUpdate: int = 200
    names: list = field(init=False)
    
    def __post_init__(self):
        self.name_Res_1 = "Res_1"
        self.name_Res_1_Upper = "Res_1_Upper"
        self.name_Res_1_Lower = "Res_1_Lower"
        self.name_Res_2 = "Res_2"
        self.name_Res_2_Upper = "Res_2_Upper"
        self.name_Res_2_Lower = "Res_2_Lower"

        self.name_Sup_1 = "Sup_1"
        self.name_Sup_1_Upper = "Sup_1_Upper"
        self.name_Sup_1_Lower = "Sup_1_Lower"
        self.name_Sup_2 = "Sup_2"
        self.name_Sup_2_Upper = "Sup_2_Upper"
        self.name_Sup_2_Lower = "Sup_2_Lower"
        self.names = [self.name_Res_1, self.name_Res_1_Upper, self.name_Res_1_Lower,
                      self.name_Res_2, self.name_Res_2_Upper, self.name_Res_2_Lower,
                      self.name_Sup_1, self.name_Sup_1_Upper, self.name_Sup_1_Lower,
                      self.name_Sup_2, self.name_Sup_2_Upper, self.name_Sup_2_Lower]

    def run(self, df, startwith='res'):
        start_value = df['close'].iloc[-1]
        
        # Initialize all columns with NaN
        for name in self.names:
            df[name] = np.nan
        
        if startwith == 'res':
            res_levels = self._find_levels(df, start_value, 'res', 2)
            sup_levels = self._find_levels(df, start_value, 'sup', 2)
        else:
            sup_levels = self._find_levels(df, start_value, 'sup', 2)
            res_levels = self._find_levels(df, start_value, 'res', 2)
        
        # Add resistance levels
        for i, (level, upper, lower, idx) in enumerate(res_levels):
            level_name = f"Res_{i+1}"
            upper_name = f"Res_{i+1}_Upper"
            lower_name = f"Res_{i+1}_Lower"
            
            df.loc[df.index >= idx, level_name] = level
            df.loc[df.index >= idx, upper_name] = upper
            df.loc[df.index >= idx, lower_name] = lower
        
        # Add support levels
        for i, (level, upper, lower, idx) in enumerate(sup_levels):
            level_name = f"Sup_{i+1}"
            upper_name = f"Sup_{i+1}_Upper"
            lower_name = f"Sup_{i+1}_Lower"
            
            df.loc[df.index >= idx, level_name] = level
            df.loc[df.index >= idx, upper_name] = upper
            df.loc[df.index >= idx, lower_name] = lower
        
        return df

    def _find_levels(self, df, start_value, level_type, num_levels):
        levels = []
        hp_series = df[self.hi_point_col].dropna()
        lp_series = df[self.lo_point_col].dropna()

        for i in range(num_levels):
            if level_type == 'res':
                hp_candidates = hp_series[hp_series > start_value]
                lp_candidates = lp_series[lp_series > start_value]
            else:
                hp_candidates = hp_series[hp_series < start_value]
                lp_candidates = lp_series[lp_series < start_value]
            
            all_candidates = pd.concat([hp_candidates, lp_candidates]).sort_values()
            # print(f'{all_candidates=}')
            
            if level_type == 'res':
                level = all_candidates.iloc[0] if not all_candidates.empty else np.nan
            else:
                level = all_candidates.iloc[-1] if not all_candidates.empty else np.nan

            # print(f'{level=}')
            
            if pd.isna(level):
                levels.append((np.nan, np.nan, np.nan, df.index[-1]))
                continue
            
            idx = df.index[((df[self.hi_point_col] == level) | (df[self.lo_point_col] == level))].min()
            atr = df.loc[idx, self.atr_col]
            
            # Calculate initial tolerance zone
            upper_bound = level + (self.tolerance * atr)
            lower_bound = level - (self.tolerance * atr)
            
            # Find the highest and lowest points within the tolerance zone
            zone_candidates = all_candidates[(all_candidates >= lower_bound) & (all_candidates <= upper_bound)]
            upper = zone_candidates.max()
            lower = zone_candidates.min()
            
            levels.append((level, upper, lower, idx))
            start_value = upper if level_type == 'res' else lower
        
        return levels
    

@dataclass
class SupResAllRows(TA):
    hi_point_col: str = 'HP'
    lo_point_col: str = 'LP'
    atr_col: str = 'ATR'
    tolerance: float = 0.01
    rowsToUpdate: int = 200
    names: list = field(init=False)
    
    def __post_init__(self):
        self.name_Res_1 = "Res_1"
        self.name_Res_1_Upper = "Res_1_Upper"
        self.name_Res_1_Lower = "Res_1_Lower"
        self.name_Res_2 = "Res_2"
        self.name_Res_2_Upper = "Res_2_Upper"
        self.name_Res_2_Lower = "Res_2_Lower"

        self.name_Sup_1 = "Sup_1"
        self.name_Sup_1_Upper = "Sup_1_Upper"
        self.name_Sup_1_Lower = "Sup_1_Lower"
        self.name_Sup_2 = "Sup_2"
        self.name_Sup_2_Upper = "Sup_2_Upper"
        self.name_Sup_2_Lower = "Sup_2_Lower"
        self.names = [self.name_Res_1, self.name_Res_1_Upper, self.name_Res_1_Lower,
                     self.name_Res_2, self.name_Res_2_Upper, self.name_Res_2_Lower,
                     self.name_Sup_1, self.name_Sup_1_Upper, self.name_Sup_1_Lower,
                     self.name_Sup_2, self.name_Sup_2_Upper, self.name_Sup_2_Lower]

    def run(self, df, startwith='res'):
        # Initialize all columns with NaN
        for name in self.names:
            df[name] = np.nan
            
        # Calculate start index for iteration
        total_rows = len(df)
        start_idx = max(1, total_rows - self.rowsToUpdate)
            
        # Process only the last rowsToUpdate rows
        for i in range(start_idx, total_rows):
            # Get all data up to current point for level calculation
            current_df = df.iloc[:i+1].copy()
            current_close = current_df['close'].iloc[-1]
            
            # Find levels based on all available data up to this point
            if startwith == 'res':
                res_levels = self._find_levels(current_df, current_close, 'res', 2)
                sup_levels = self._find_levels(current_df, current_close, 'sup', 2)
            else:
                sup_levels = self._find_levels(current_df, current_close, 'sup', 2)
                res_levels = self._find_levels(current_df, current_close, 'res', 2)
            
            # Update resistance levels for the current row
            for j, (level, upper, lower, idx) in enumerate(res_levels):
                if pd.notna(level):
                    level_name = f"Res_{j+1}"
                    upper_name = f"Res_{j+1}_Upper"
                    lower_name = f"Res_{j+1}_Lower"
                    
                    df.loc[df.index[i], level_name] = level
                    df.loc[df.index[i], upper_name] = upper
                    df.loc[df.index[i], lower_name] = lower
            
            # Update support levels for the current row
            for j, (level, upper, lower, idx) in enumerate(sup_levels):
                if pd.notna(level):
                    level_name = f"Sup_{j+1}"
                    upper_name = f"Sup_{j+1}_Upper"
                    lower_name = f"Sup_{j+1}_Lower"
                    
                    df.loc[df.index[i], level_name] = level
                    df.loc[df.index[i], upper_name] = upper
                    df.loc[df.index[i], lower_name] = lower
        
        return df

    def _find_levels(self, df, start_value, level_type, num_levels):
        levels = []
        hp_series = df[self.hi_point_col].dropna()
        lp_series = df[self.lo_point_col].dropna()

        for i in range(num_levels):
            if level_type == 'res':
                hp_candidates = hp_series[hp_series > start_value]
                lp_candidates = lp_series[lp_series > start_value]
            else:
                hp_candidates = hp_series[hp_series < start_value]
                lp_candidates = lp_series[lp_series < start_value]
            
            all_candidates = pd.concat([hp_candidates, lp_candidates]).sort_values()
            
            if level_type == 'res':
                level = all_candidates.iloc[0] if not all_candidates.empty else np.nan
            else:
                level = all_candidates.iloc[-1] if not all_candidates.empty else np.nan
            
            if pd.isna(level):
                levels.append((np.nan, np.nan, np.nan, df.index[-1]))
                continue
            
            idx = df.index[((df[self.hi_point_col] == level) | (df[self.lo_point_col] == level))].min()
            atr = df.loc[idx, self.atr_col]
            
            # Calculate initial tolerance zone
            upper_bound = level + (self.tolerance * atr)
            lower_bound = level - (self.tolerance * atr)
            
            # Find the highest and lowest points within the tolerance zone
            zone_candidates = all_candidates[(all_candidates >= lower_bound) & (all_candidates <= upper_bound)]
            upper = zone_candidates.max()
            lower = zone_candidates.min()
            
            levels.append((level, upper, lower, idx))
            start_value = upper if level_type == 'res' else lower
        
        return levels
  

@dataclass
class MansfieldRSI(TA):
    stockCol: str = 'close'  # Column name for stock close price
    marketCol: str = 'index_close'  # Column name for index close price
    span: int = 14  # Default lookback period
    
    def __post_init__(self):
        self.name = f"MRSI_{self.span}_{self.marketCol}"
        self.names = [self.name]
        self.rowsToUpdate = 200

    @preprocess_data
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Step 1: Calculate raw Relative Strength (RS)
        rs = df[self.stockCol] / df[self.marketCol]
        
        # Step 2: Calculate the moving average of RS
        rs_ma = rs.rolling(window=self.span, min_periods=1).mean()
        
        # Step 3: Mansfield RSI = (Normalized RS - 1) * 100
        rs_normalized = rs / rs_ma # extra step in trading view 
        df[self.name] = ((rs_normalized - 1) * 10).fillna(0)       
        
        return df[[self.name]]



@dataclass
class DIR(TA):
    """
    Direction Indicator
    Measures trend direction based on moving average slope, normalized to -100 to +100.
    Positive values indicate upward trend, negative values indicate downward trend.
    Values closer to extremes indicate steeper slopes.
    """
    period: int = 50
    max_slope: float = 0.02  # Maximum expected slope (2% per period)
    
    def __post_init__(self):
        self.name = f"MADIR_{self.column[:2]}_{self.period}"
        self.names = [self.name]
        self.rowsToUpdate = self.period + 1
    
    @staticmethod
    def normalize(series: pd.Series, max_value: float) -> pd.Series:
        """Efficient normalization to -100 to 100 range"""
        return (series / max_value).clip(-1, 1) * 100
    
    @preprocess_data
    def run(self, data: pd.DataFrame) -> pd.Series:
        ma = data[self.column].rolling(window=self.period).mean()
        slope = ma.diff() / ma.shift(1)
        return self.normalize(slope, self.max_slope).rename(self.names)


@dataclass
class ColVal:
    """Checks if a column value is above/below a threshold"""
    column: str

    def __post_init__(self):
        self.name = f"CV_{self.column}"
        self.names = [self.name]
    
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.name] = data[self.column] #! Redundancy but just did this to get it working quickly
        return data[self.name]


@dataclass
class VolAcc(TA):
    """
    Acceleration Indicator
    Measures rate of change in trend direction, normalized to -100 to +100.
    Positive values indicate increasing slope (acceleration up),
    negative values indicate decreasing slope (acceleration down).
    """
    max_accel: float = 0.001  # Maximum expected acceleration (0.1% per period)
    
    def __post_init__(self):
        self.name = f"VolACC"
        self.names = [self.name]
        self.rowsToUpdate = len(self.column) + 1
    
    @staticmethod
    def normalize(series: pd.Series, max_value: float) -> pd.Series:
        """Efficient normalization to -100 to 100 range"""
        return (series / max_value).clip(-1, 1) * 100
    
    @preprocess_data
    def run(self, data: pd.DataFrame) -> pd.Series:
        d = data['volume']
        current_slope = d.diff() / d.shift(1)
        prev_slope = d.shift(1).diff() / d.shift(2)
        acceleration = current_slope - prev_slope
        return self.normalize(acceleration, self.max_accel).rename(self.name)


@dataclass
class ACC(TA):
    """
    Acceleration Indicator using dual moving averages
    Measures acceleration by comparing the change in MA differences over time.
    Positive values indicate increasing difference between MAs,
    negative values indicate decreasing difference between MAs.
    """
    fast_ma: int = 5       # Period for faster moving average
    slow_ma: int = 10      # Period for slower moving average
    max_accel: float = 0.001  # Maximum expected acceleration (0.1% per period)
    
    def __post_init__(self):
        self.name = f"ACC_{self.column}"
        self.names = [self.name]
        self.rowsToUpdate = max(self.fast_ma, self.slow_ma) + 2  # +2 for shift operation
    
    @staticmethod
    def normalize(series: pd.Series, max_value: float) -> pd.Series:
        """Efficient normalization to -100 to 100 range"""
        return (series / max_value).clip(-1, 1) * 100
    
    @preprocess_data
    def run(self, data: pd.DataFrame) -> pd.Series:
        # Calculate two moving averages
        fast_ma = data[self.column].rolling(window=self.fast_ma).mean()
        slow_ma = data[self.column].rolling(window=self.slow_ma).mean()
        
        ma_diff = fast_ma - slow_ma
        current_diff = ma_diff
        return self.normalize(current_diff, self.max_accel).rename(self.name)


@dataclass
class Breaks:
    """Checks if price crosses above/below a metric"""
    price_column: str
    direction: str  # 'above' or 'below'
    metric_column: str

    def __post_init__(self):
        self.name = f"BRK_{self.price_column}_{self.direction[:2]}_{self.metric_column}"
        self.names = [self.name]

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        curr_price = df[self.price_column]
        prev_price = curr_price.shift(1)
        curr_metric = df[self.metric_column]
        prev_metric = curr_metric.shift(1)

        if self.direction == 'above':
            df[self.name] = (prev_price <= prev_metric) & (curr_price > curr_metric)
            return df
        elif self.direction == 'below':
            df[self.name] = (prev_price >= prev_metric) & (curr_price < curr_metric)
            return df
        
        raise ValueError("Direction must be 'above' or 'below'")


@dataclass
class AboveBelow:
    """Checks if price is above/below a metric"""
    value: str | float
    direction: str  # 'above' or 'below'
    metric_column: str

    def __post_init__(self):
        self.name = f"AB_{self.value}_{self.direction[:2]}_{self.metric_column}"
        self.names = [self.name]

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        value =  df[self.value] if isinstance(self.value, str) else self.value
        metric = df[self.metric_column] if isinstance(self.metric_column, str) else self.metric_column

        if self.direction == 'above':
            df[self.name] = value > metric
        elif self.direction == 'below':
            df[self.name] = value < metric
        else:
            raise ValueError("Direction must be 'above' or 'below'")
            
        return df


@dataclass
class PctChange:
    """Calculates percentage change of a column"""
    metric_column: str = 'close'
    period: int = 1

    def __post_init__(self):
        self.name = f"PCT_{self.metric_column}_{self.period}"
        self.names = [self.name]
        self.rowsToUpdate = self.period + 1

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.metric_column not in df.columns:
            raise KeyError(f"Column '{self.metric_column}' not found in DataFrame.")
        df[self.name] = df[self.metric_column].pct_change(periods=self.period) * 100
        return df


@dataclass
class VolDev(TA):
    """Calculates percentage deviation of current volume from its moving average"""
    period: int = 10 # period for the moving average

    def __post_init__(self):
        self.name = f"VDEV_{self.period}"
        self.names = [self.name]
        self.rowsToUpdate = self.period + 1

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # Calculate volume moving average
        volume_ma = df[self.column].rolling(window=self.period).mean()
        
        # Calculate percentage deviation
        # ((current - average) / average) * 100
        df[self.name] = ((df[self.column] - volume_ma) / volume_ma) * 100
            
        return df
    

@dataclass
class VolumeThreshold(TA):
    """
    Volume Threshold Indicator
    Identifies when volume is above a specified percentage threshold
    compared to its moving average.
    Returns 1 when above threshold, 0 when below.
    """
    period: int = 10           # Period for moving average
    threshold: float = 0.8     # 80% above moving average = 1.8
    
    def __post_init__(self):
        self.column = 'volume'
        self.name = f"VOL_THRESH_{self.period}_{int(self.threshold*100)}"
        self.names = [self.name]
        self.rowsToUpdate = self.period + 1
    
    @preprocess_data
    def run(self, data: pd.DataFrame) -> pd.Series:
        # Calculate moving average of volume
        volume_ma = data[self.column].rolling(window=self.period).mean()
        
        # Calculate ratio of current volume to moving average
        volume_ratio = data[self.column] / volume_ma
        
        # Create binary signal: 1 if above threshold, 0 if below
        signal = (volume_ratio > (1 + self.threshold)).astype(int)
        
        return signal.rename(self.name)
    

@dataclass
class TrendDuration(TA):
    """
    Trend Duration Indicator
    Measures the duration of a trend by counting the number of consecutive
    periods in which the trend has been in the same direction.
    return negative number if downtrend, positive number if uptrend
    """

    def __post_init__(self):
        self.name = f"TDUR_{self.column}"
        self.names = [self.name]
        self.rowsToUpdate = 20

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Calculate trend direction
        trend = np.sign(df[self.column].diff())
        
        # Calculate trend duration
        trend_duration = trend.groupby((trend != trend.shift()).cumsum()).cumcount() + 1
        
        # Assign trend duration to the last row
        df[self.name] = trend_duration
        
        # Convert trend duration to negative if in a downtrend
        df.loc[trend < 0, self.name] = -df.loc[trend < 0, self.name]
        
        return df
    
    
def process_ta_filters(df, ta_list, min_score=None):
    """
    Process multiple technical analysis filters and return a dataframe with score summaries.
    
    Parameters:
    frame: The initial frame object that handles technical analysis
    ta_list (list): List of technical analysis filter objects
    min_score (int, optional): Minimum score to filter the results
    
    Returns:
    pandas.DataFrame: DataFrame with filter scores and all-true indicator
    """
    
    # Get the names of all filters
    ta_filter_names = [ta.name for ta in ta_list]

    
    # Calculate the sum of true values for each row
    df['filter_score'] = df[ta_filter_names].sum(axis=1)
    
    # Add column to indicate if all filters are true
    df['all_true'] = df[ta_filter_names].all(axis=1)
    
    # Apply minimum score filter if specified
    if min_score is not None:
        df = df[df['filter_score'] >= min_score]
    
    return df


# Example usage:
# Assume df is your input DataFrame with 'datetime' as index, 'high' and 'low' as columns
# result = cluster_df(df, hp_column='high', lp_column='low', startwith='res', atr=1, tol=0.01)
# print(result)

@dataclass
class SupRes_template:
    hi_point_col: str = 'HP'
    lo_point_col: str = 'LP'
    atr_col: str = 'ATR'
    pointsAgo: int = 1
    tolerance: float = 0.01
    atr_period: int = 14
    rowsToUpdate: int = 200
    names: list = field(init=False)
    
    def __post_init__(self):
        self.name_resistance = f"Res_{self.pointsAgo}"
        self.name_res_upper = f"{self.name_resistance}_Upper"
        self.name_res_lower = f"{self.name_resistance}_Lower"
        self.name_support = f"Sup_{self.pointsAgo}"
        self.name_sup_upper = f"{self.name_support}_Upper"
        self.name_sup_lower = f"{self.name_support}_Lower"
        self.names = [self.name_resistance, self.name_res_upper, self.name_res_lower, 
                      self.name_support, self.name_sup_upper, self.name_sup_lower]
    
    def run(self, df, startwith='res'):
        # FIL CODE HERE 
        return df # return the updated DataFrame with the support and resistance levels and the upper and lower bounds


#! ---- Trend Lines ----




from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.cluster import DBSCAN



@dataclass
class ConsolidationZone(TA):
    hp_column: str = 'HP_hi_10'
    lp_column: str = 'LP_lo_10'
    price_tolerance: float = 0.001
    max_points_between: int = 2
    height_width_ratio: float = 0.5
    name: str = 'CONS'
    atr_column: str = 'ATR'
    limit_zones: int = 5
    """A class that identifies consolidation zones in price data using a multi-step algorithm.
    
    The algorithm identifies consolidation zones through these key steps:
    1. Price Level Clustering: Uses DBSCAN to group similar price levels within a tolerance
    2. Temporal Filtering: Validates point pairs by checking the number of intermediate points
    3. Zone Validation: Ensures zones meet height-to-width ratio criteria using ATR
    4. Zone Extension: Extends zones until MA breaches and multiple candle violations occur
    
    Key Features:
    - Identifies both support and resistance levels
    - Uses adaptive price tolerance based on mean price
    - Considers market volatility through ATR normalization
    - Prevents overlapping zones through leftmost point tracking
    - Supports customizable parameters for different market conditions
    
    Parameters:
        hp_column (str): Column name for high prices, default 'HP_hi_10'
        lp_column (str): Column name for low prices, default 'LP_lo_10'
        price_tolerance (float): Maximum price difference for clustering, as % of mean price
        max_points_between (int): Maximum allowed intermediate points between zone boundaries
        height_width_ratio (float): Maximum allowed height/width ratio for valid zones
        name (str): Prefix for zone column names, default 'RECT'
        atr_column (str): Column name for Average True Range, default 'ATR'
        
    Returns DataFrame with added columns for upper and lower zone boundaries.
    Columns are named as {name}_UPPER_{i} and {name}_LOWER_{i} where i is zone index.
    """
    
    def __post_init__(self):
        self.names = []

    def filter_close_points(self, cluster: List[Tuple[pd.Timestamp, float]]) -> List[List[Tuple[pd.Timestamp, float]]]:
        """Filter points within max_points_between constraint"""
        valid_pairs = []
        for i in range(len(cluster)-1):
            for j in range(i+1, len(cluster)):
                date1, price1 = cluster[i]
                date2, price2 = cluster[j]
                
                # Count points between dates
                points_between = sum(1 for d, _ in cluster 
                                   if date1 < d < date2)
                                   
                if points_between <= self.max_points_between:
                    valid_pairs.append([cluster[i], cluster[j]])
                    
        return valid_pairs

    def cluster_price_levels(self, series: pd.Series) -> List[List[Tuple[pd.Timestamp, float]]]:
        """Cluster similar price levels using DBSCAN then filter by max_points_between"""
        points = series.dropna()
        if len(points) < 2:
            return []
            
        prices = points.values.reshape(-1, 1)
        mean_price = np.mean(prices)
        eps = mean_price * self.price_tolerance
        
        clustering = DBSCAN(eps=eps, min_samples=2).fit(prices)
        
        valid_pairs = []
        for label in set(clustering.labels_):
            if label != -1:
                cluster_mask = clustering.labels_ == label
                cluster_points = list(zip(points.index[cluster_mask], 
                                       points.values[cluster_mask]))
                cluster_points.sort(key=lambda x: x[0])
                
                # Filter points within max_points_between
                valid_pairs.extend(self.filter_close_points(cluster_points))
                
        return valid_pairs

    def find_opposite_extreme(self, data: pd.DataFrame, dates: List[pd.Timestamp], 
                            is_support: bool) -> Dict:
        start_date, end_date = min(dates), max(dates)
        date_range = data.loc[start_date:end_date]
        
        if is_support:
            extreme = date_range['high'].max()
            extreme_date = date_range[date_range['high'] == extreme].index[0]
        else:
            extreme = date_range['low'].min()
            extreme_date = date_range[date_range['low'] == extreme].index[0]
            
        return {'price': extreme, 'date': extreme_date}

    def validate_zone(self, data: pd.DataFrame, dates: List[pd.Timestamp], upper: float, lower: float) -> bool:
        start_date, end_date = min(dates), max(dates)
        width = len(data.loc[start_date:end_date])
        height = upper - lower
        return height / (width * data.loc[end_date, self.atr_column]) <= self.height_width_ratio

    def extend_zone(self, data: pd.DataFrame, zone: Dict) -> Dict:
        ma = data['close'].rolling(window=21).mean()
        
        for direction in ['left', 'right']:
            idx = data.index.get_loc(min(zone['dates']) if direction == 'left' 
                                else max(zone['dates']))
            ma_breach = False
            candle_breach = 0
            
            while 0 < idx < len(data) - 1:
                price = ma.iloc[idx]
                candle = data.iloc[idx]
                
                # Check MA breach
                if not (zone['lower'] <= price <= zone['upper']):
                    ma_breach = True
                
                # Check candle breach
                if candle['high'] > zone['upper'] or candle['low'] < zone['lower']:
                    candle_breach += 1
                
                # Exit if both conditions met
                if ma_breach and candle_breach >= 3:
                    break
                    
                idx = idx - 1 if direction == 'left' else idx + 1
                
            zone[f'{direction}_date'] = data.index[idx]
            
        return zone

    def find_zones(self, data: pd.DataFrame) -> List[Dict]:
        zones = []
        leftmost_allowed = data.index[-1]

        # print(f"1. Finding zones starting from leftmost_allowed: {leftmost_allowed}")    
        
        while leftmost_allowed >= data.index[0]:
            # Add the zone limit check here - if we've hit our limit, break out
            if len(zones) >= self.limit_zones:
                break
            valid_data = data[:leftmost_allowed]
            if len(valid_data) < 3:
                break

            high_pairs = self.cluster_price_levels(valid_data[self.hp_column])
            low_pairs = self.cluster_price_levels(valid_data[self.lp_column])

            # print(f"3. Found {len(high_pairs)} high pairs and {len(low_pairs)} low pairs")
            
            if not high_pairs and not low_pairs:
                # No valid pairs found, move left
                curr_idx = valid_data.index.get_loc(leftmost_allowed)
                if curr_idx > 0:
                    leftmost_allowed = valid_data.index[curr_idx - 1]
                else:
                    break
                continue
            
            all_pairs = [(pair, True) for pair in high_pairs] + \
                       [(pair, False) for pair in low_pairs]
            all_pairs.sort(key=lambda x: min(p[0] for p in x[0]), reverse=False) #!!! reverse=True
            
            zone_found = False
            for pair, is_high in all_pairs:
                # print(f"4. Checking pair {pair} for {['resistance', 'support'][is_high]}")
                # print(f"5. max date: {max(date for date, _ in pair)} > {leftmost_allowed} = {max(date for date, _ in pair) > leftmost_allowed}")
                if max(date for date, _ in pair) > leftmost_allowed:
                    # print(f"6. Skipping pair {pair} for {['resistance', 'support'][is_high]}")
                    continue
                
            dates = [date for point in pair for date, _ in [point]]
            opposite = self.find_opposite_extreme(data, dates, not is_high)
            
            upper = max(opposite['price'], max(price for _, price in pair))
            lower = min(opposite['price'], min(price for _, price in pair))

            
            if self.validate_zone(data, dates + [opposite['date']], upper, lower):
                zone = {
                    'upper': upper,
                    'lower': lower,
                    'dates': dates + [opposite['date']]
                }
                
                zone = self.extend_zone(data, zone)
                zones.append(zone)
                # Update leftmost_allowed to the leftmost point of current zone
                leftmost_allowed = min(zone['left_date'], min(date for date, _ in pair))
                # print(f"7. Found zonne. Now leftmost allowed {leftmost_allowed}")
                zone_found = True
                continue
                
            if not zone_found:
                # If no valid zone found, move leftmost allowed point left
                curr_idx = valid_data.index.get_loc(leftmost_allowed)
                if curr_idx > 0:
                    leftmost_allowed = valid_data.index[curr_idx - 1]
                else:
                    break
                
        return zones

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        self.names = []
        result = data.copy()
        
        zones = self.find_zones(data)

        for i, zone in enumerate(zones):
            upper_name = f"{self.name}_UPPER_{i+1}"
            lower_name = f"{self.name}_LOWER_{i+1}"
            self.names.extend([upper_name, lower_name])
            
            result[upper_name] = np.nan
            result[lower_name] = np.nan
            
            mask = (result.index >= zone['left_date']) & \
                   (result.index <= zone['right_date'])
            result.loc[mask, upper_name] = zone['upper']
            result.loc[mask, lower_name] = zone['lower']
        
        return result
    

@dataclass
class TrendlineDetector_old(TA):
    name: str = 'TREND'
    point_column: str = ''  # Single column for points
    slope_direction: str = 'up'  # 'up' or 'down'
    slope_tolerance: float = 0.1
    min_points: int = 3
    lookback_points: int = 6
    
    
    def __post_init__(self):
        self.names = []
        if self.slope_direction not in ['up', 'down']:
            raise ValueError("slope_direction must be 'up' or 'down'")
        
    def calculate_slope(self, point1: Tuple[pd.Timestamp, float], 
                       point2: Tuple[pd.Timestamp, float]) -> float:
        """Calculate slope between two points using time delta in days"""
        x1 = point1[0].timestamp()
        x2 = point2[0].timestamp()
        y1, y2 = point1[1], point2[1]
        
        # Convert to days for better slope scaling
        time_delta = (x2 - x1) / (24 * 3600)  
        return (y2 - y1) / time_delta if time_delta != 0 else float('inf')
        
    def find_similar_slopes(self, slopes: List[Tuple[float, Tuple]]) -> List[List[Tuple]]:
        """Group points with similar slopes within tolerance"""
        if not slopes:
            return []
            
        groups = []
        current_group = [slopes[0][1]]
        base_slope = slopes[0][0]
        
        for slope, point in slopes[1:]:
            if base_slope == 0:
                similar = abs(slope) <= self.slope_tolerance
            else:
                similar = abs((slope - base_slope) / base_slope) <= self.slope_tolerance
            if similar:
                current_group.append(point)
            else:
                if len(current_group) >= self.min_points:
                    groups.append(current_group)
                current_group = [point]
                base_slope = slope
                
        if len(current_group) >= self.min_points:
            groups.append(current_group)
            
        return groups
        
    def calculate_r_squared(self, points: List[Tuple], slope: float, intercept: float) -> float:
        """Calculate R-squared value for a trendline"""
        x = np.array([(p[0].timestamp() / (24 * 3600)) for p in points])
        y = np.array([p[1] for p in points])
        y_pred = slope * x + intercept
        return 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        
    def validate_trendline(self, points: List[Tuple]) -> Tuple[bool, float, float]:
        """Validate trendline using R-squared and calculate line parameters"""
        x = np.array([(p[0].timestamp() / (24 * 3600)) for p in points])
        y = np.array([p[1] for p in points])
        
        slope, intercept = np.polyfit(x, y, 1)
        r_squared = self.calculate_r_squared(points, slope, intercept)
        
        return r_squared >= 0.8, slope, intercept
        

    def find_trendlines(self, data: pd.DataFrame) -> List[Dict]:
        series = data[self.point_column]
        points = [(idx, val) for idx, val in series.dropna().items()]
        candidate_lines = []
        
        for i, current_point in enumerate(points):
            slopes = []
            for j in range(i + 1, min(i + self.lookback_points + 1, len(points))):
                slope = self.calculate_slope(current_point, points[j])
                # Filter slopes based on direction
                if (self.slope_direction == 'up' and slope > 0) or \
                   (self.slope_direction == 'down' and slope < 0):
                    slopes.append((slope, points[j]))
            
            groups = self.find_similar_slopes(slopes)
            for group in groups:
                all_points = [current_point] + group
                is_valid, slope, intercept = self.validate_trendline(all_points)
                
                if is_valid:
                    candidate_lines.append({
                        'start_date': min(p[0] for p in all_points),
                        'end_date': max(p[0] for p in all_points),
                        'slope': slope,
                        'intercept': intercept,
                        'r_squared': self.calculate_r_squared(all_points, slope, intercept),
                        'points': all_points
                    })
        
        # Remove overlapping lines
        consolidated_lines = []
        for line in sorted(candidate_lines, key=lambda x: x['r_squared'], reverse=True):
            overlapping = False
            for existing in consolidated_lines:
                if (line['start_date'] <= existing['end_date'] and 
                    line['end_date'] >= existing['start_date']):
                    overlapping = True
                    break
            if not overlapping:
                consolidated_lines.append(line)
        
        return consolidated_lines
        
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        self.names = []
        result = data.copy()
        trendlines = self.find_trendlines(data)
        
        for i, trendline in enumerate(trendlines):
            name = f"{self.name}_{i+1}"
            self.names.append(name)
            result[name] = np.nan
            mask = (result.index >= trendline['start_date']) & \
                   (result.index <= trendline['end_date'])
            x_values = np.array([(d.timestamp() / (24 * 3600)) for d in result.index[mask]])
            result.loc[mask, name] = trendline['slope'] * x_values + trendline['intercept']
        
        return result
    
@dataclass
class TrendlineDetector(TA):
    name: str = 'TREND'
    point_column: str = ''  
    slope_direction: str = 'up'  
    slope_tolerance: float = 0.1
    min_points: int = 3
    lookback_points: int = 6
    atr_period: int = 14  
    atr_threshold: float = 1  
    
    def __post_init__(self):
        self.names = []
        if self.slope_direction not in ['up', 'down']:
            raise ValueError("slope_direction must be 'up' or 'down'")
        
    def calculate_slope(self, point1: Tuple[pd.Timestamp, float], 
                       point2: Tuple[pd.Timestamp, float]) -> float:
        x1 = point1[0].timestamp()
        x2 = point2[0].timestamp()
        y1, y2 = point1[1], point2[1]
        time_delta = (x2 - x1) / (24 * 3600)  
        return (y2 - y1) / time_delta if time_delta != 0 else float('inf')

    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'] if 'high' in data.columns else data[self.point_column]
        low = data['low'] if 'low' in data.columns else data[self.point_column]
        close = data[self.point_column].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        return tr.rolling(window=self.atr_period).mean()
        
    def find_similar_slopes(self, slopes: List[Tuple[float, Tuple]]) -> List[List[Tuple]]:
        if not slopes:
            return []
            
        groups = []
        current_group = [slopes[0][1]]
        base_slope = slopes[0][0]
        
        for slope, point in slopes[1:]:
            if base_slope == 0:
                similar = abs(slope) <= self.slope_tolerance
            else:
                similar = abs((slope - base_slope) / base_slope) <= self.slope_tolerance
            if similar:
                current_group.append(point)
            else:
                if len(current_group) >= self.min_points:
                    groups.append(current_group)
                current_group = [point]
                base_slope = slope
                
        if len(current_group) >= self.min_points:
            groups.append(current_group)
            
        return groups
        
    def calculate_r_squared(self, points: List[Tuple], slope: float, intercept: float) -> float:
        x = np.array([(p[0].timestamp() / (24 * 3600)) for p in points])
        y = np.array([p[1] for p in points])
        y_pred = slope * x + intercept
        return 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        
    def validate_trendline(self, points: List[Tuple]) -> Tuple[bool, float, float]:
        x = np.array([(p[0].timestamp() / (24 * 3600)) for p in points])
        y = np.array([p[1] for p in points])
        
        slope, intercept = np.polyfit(x, y, 1)
        r_squared = self.calculate_r_squared(points, slope, intercept)
        
        return r_squared >= 0.8, slope, intercept

    def validate_points_distance(self, points: List[Tuple], slope: float, 
                               intercept: float, atr: pd.Series) -> bool:
        for point in points:
            date, price = point
            x_val = date.timestamp() / (24 * 3600)
            line_price = slope * x_val + intercept
            deviation = abs(price - line_price)
            
            current_atr = atr.get(date, atr.mean())
            # Check deviation in both directions
            if (self.slope_direction == 'up' and (price - line_price) > current_atr * self.atr_threshold) or \
               (self.slope_direction == 'down' and (line_price - price) > current_atr * self.atr_threshold):
                return False
        return True

    def find_trendlines(self, data: pd.DataFrame) -> List[Dict]:
        series = data[self.point_column]
        points = [(idx, val) for idx, val in series.dropna().items()]
        candidate_lines = []
        atr = self.calculate_atr(data)
        
        for i, current_point in enumerate(points):
            slopes = []
            for j in range(i + 1, min(i + self.lookback_points + 1, len(points))):
                slope = self.calculate_slope(current_point, points[j])
                if (self.slope_direction == 'up' and slope > 0) or \
                   (self.slope_direction == 'down' and slope < 0):
                    slopes.append((slope, points[j]))
            
            groups = self.find_similar_slopes(slopes)
            for group in groups:
                all_points = [current_point] + group
                is_valid, slope, intercept = self.validate_trendline(all_points)
                
                if is_valid and self.validate_points_distance(all_points, slope, intercept, atr):
                    candidate_lines.append({
                        'start_date': min(p[0] for p in all_points),
                        'end_date': max(p[0] for p in all_points),
                        'slope': slope,
                        'intercept': intercept,
                        'r_squared': self.calculate_r_squared(all_points, slope, intercept),
                        'points': all_points
                    })
        
        consolidated_lines = []
        for line in sorted(candidate_lines, key=lambda x: x['r_squared'], reverse=True):
            overlapping = False
            for existing in consolidated_lines:
                if (line['start_date'] <= existing['end_date'] and 
                    line['end_date'] >= existing['start_date']):
                    overlapping = True
                    break
            if not overlapping:
                consolidated_lines.append(line)
        
        return consolidated_lines
        
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        self.names = []
        result = data.copy()
        trendlines = self.find_trendlines(data)
        
        for i, trendline in enumerate(trendlines):
            name = f"{self.name}_{i+1}"
            self.names.append(name)
            result[name] = np.nan
            mask = (result.index >= trendline['start_date']) & \
                   (result.index <= trendline['end_date'])
            x_values = np.array([(d.timestamp() / (24 * 3600)) for d in result.index[mask]])
            result.loc[mask, name] = trendline['slope'] * x_values + trendline['intercept']
        
        return result
    

@dataclass
class MicroTrendline_example(TA):
    name: str = 'MTREND'
    pointsCol: str = ''  
    atrCol: str = 'ATR'
    slopeDir: str = 'up'  
    slopeToleranceATR: float = 2
    projectionPeriod: int = 5

    def __post_init__(self):
        self.names = []
        if self.slope_direction not in ['up', 'down']:
            raise ValueError("slope_direction must be 'up' or 'down'")
        
    # -->> other methods here <<--

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        self.names = []
        result = data.copy()
        trendlines = self.find_trendlines(data)
        
        for i, trendline in enumerate(trendlines):
            name = f"{self.name}_{i+1}"
            self.names.append(name)
            result[name] = np.nan
            mask = (result.index >= trendline['start_date']) & \
                   (result.index <= trendline['end_date'])
            x_values = np.array([(d.timestamp() / (24 * 3600)) for d in result.index[mask]])
            result.loc[mask, name] = trendline['slope'] * x_values + trendline['intercept']
        
        return result
    

from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List, Dict, Any


@dataclass
class MicroTrendline:
    name: str = 'MTREND'
    pointsCol: str = ''  # Column containing points
    atrCol: str = 'ATR'
    slopeDir: str = 'down'  
    slopeToleranceATR: float = 2.0
    projectionPeriod: int = 5

    def __post_init__(self):
        self.names = []
        if self.slopeDir not in ['up', 'down']:
            raise ValueError("slopeDir must be 'up' or 'down'")

    def find_micro_extremes(self, data: pd.DataFrame, start_idx: int) -> List[Dict[str, Any]]:
        """Find micro extreme points (highs for downtrend, lows for uptrend) after the last major point."""
        micro_points = []
        
        for i in range(start_idx + 1, len(data) - 1):
            if self.slopeDir == 'down':
                current = data.iloc[i]['high']
                prev = data.iloc[i-1]['high']
                next_val = data.iloc[i+1]['high']
                condition = current > prev and current > next_val
            else:  # uptrend
                current = data.iloc[i]['low']
                prev = data.iloc[i-1]['low']
                next_val = data.iloc[i+1]['low']
                condition = current < prev and current < next_val
            
            if condition:
                micro_points.append({
                    'date': data.index[i],
                    'price': current
                })
                
        return micro_points

    def calculate_best_fit_line(self, points: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate best fit line through given points."""
        x = np.array([(p['date'].timestamp() / (24 * 3600)) for p in points])
        y = np.array([p['price'] for p in points])
        
        slope, intercept = np.polyfit(x, y, 1)
        return {'slope': slope, 'intercept': intercept}

    def find_most_protruding_point(self, data: pd.DataFrame, start_date: pd.Timestamp, 
                                 end_date: pd.Timestamp, line_params: Dict[str, float]) -> Dict[str, Any]:
        """Find the point that most protrudes from the best fit line."""
        mask = (data.index >= start_date) & (data.index <= end_date)
        subset = data[mask]
        
        max_deviation = -np.inf if self.slopeDir == 'down' else np.inf
        max_point = None
        
        for idx, row in subset.iterrows():
            x_val = idx.timestamp() / (24 * 3600)
            projected_price = line_params['slope'] * x_val + line_params['intercept']
            
            if self.slopeDir == 'down':
                price = row['high']
                deviation = price - projected_price
                condition = deviation > max_deviation
            else:
                price = row['low']
                deviation = projected_price - price
                condition = deviation > max_deviation
            
            if condition:
                max_deviation = deviation
                max_point = {'date': idx, 'price': price}
                
        return max_point

    def project_line(self, data: pd.DataFrame, point_b: Dict[str, Any], 
                    point_a: Dict[str, Any]) -> Dict[str, Any]:
        """
        Project a line from point A through point B and beyond until breakthrough.
        For downtrend: point_a should be higher than point_b
        For uptrend: point_a should be lower than point_b
        """
        x1 = point_a['date'].timestamp() / (24 * 3600)
        x2 = point_b['date'].timestamp() / (24 * 3600)
        y1 = point_a['price']
        y2 = point_b['price']
        
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        
        start_idx = data.index.get_loc(point_b['date'])
        consecutive_breaks = 0
        breakthrough_idx = None
        first_break_idx = None
        
        for i in range(start_idx, len(data)):
            x_val = data.index[i].timestamp() / (24 * 3600)
            projected_price = slope * x_val + intercept
            
            if self.slopeDir == 'down':
                break_condition = data.iloc[i]['low'] > projected_price
            else:
                break_condition = data.iloc[i]['high'] < projected_price
            
            if break_condition:
                if consecutive_breaks == 0:
                    first_break_idx = i
                consecutive_breaks += 1
                if consecutive_breaks >= self.projectionPeriod:
                    breakthrough_idx = first_break_idx
                    break
            else:
                consecutive_breaks = 0
                first_break_idx = None
        
        # If we found a breakthrough, extend the line by projectionPeriod bars after the breakthrough
        if breakthrough_idx is not None:
            end_idx = min(breakthrough_idx + self.projectionPeriod, len(data) - 1)
            end_date = data.index[end_idx]
        else:
            end_date = data.index[-1]
        
        return {
            'slope': slope,
            'intercept': intercept,
            'start_date': point_a['date'],
            'end_date': end_date
        }
    
    def check_backward_points(self, data: pd.DataFrame, trendline: Dict[str, Any], tolerance_atr: float) -> Optional[pd.Timestamp]:
        """Check for valid points behind the start of a trendline within tolerance."""
        start_idx = data.index.get_loc(trendline['start_date'])
        
        # Look backward
        for i in range(start_idx - 1, -1, -1):
            x_val = data.index[i].timestamp() / (24 * 3600)
            projected_price = trendline['slope'] * x_val + trendline['intercept']
            current_atr = data.iloc[i][self.atrCol]
            
            if self.slopeDir == 'down':
                price = data.iloc[i]['high']
                if abs(price - projected_price) <= tolerance_atr * current_atr:
                    return data.index[i]
            else:
                price = data.iloc[i]['low']
                if abs(price - projected_price) <= tolerance_atr * current_atr:
                    return data.index[i]
        
        return None

    def find_trendlines(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find trendlines in the data based on slope direction."""
        trendlines = []
        
        valid_points = data[data[self.pointsCol].notna()].index
        if len(valid_points) < 2:
            return trendlines
            
        valid_points = valid_points.tolist()[::-1]
        
        for i in range(len(valid_points) - 1):
            current_idx = valid_points[i]      # More recent point (Point B)
            prev_idx = valid_points[i + 1]     # Previous point (Point A)
            
            point_b = {
                'date': current_idx,
                'price': data.loc[current_idx, self.pointsCol]
            }
            
            point_a = {
                'date': prev_idx,
                'price': data.loc[prev_idx, self.pointsCol]
            }
            
            # Check price difference based on trend direction
            if self.slopeDir == 'down':
                price_diff = point_a['price'] - point_b['price']  # A should be higher than B
            else:
                price_diff = point_b['price'] - point_a['price']  # B should be higher than A
                
            atr_multiple = price_diff / data.loc[point_b['date'], self.atrCol]
            
            if atr_multiple >= self.slopeToleranceATR:
                # Special handling for the most recent high/low point
                if i == 0:
                    start_idx = data.index.get_loc(current_idx)
                    subset_data = data.iloc[start_idx:]
                    
                    # Check if all subsequent bars respect the trend direction
                    if self.slopeDir == 'down':
                        all_lower = all(subset_data['high'] <= point_b['price'])
                    else:
                        all_higher = all(subset_data['low'] >= point_b['price'])
                    
                    if (self.slopeDir == 'down' and all_lower) or (self.slopeDir == 'up' and all_higher):
                        # Find micro extremes (highs for downtrend, lows for uptrend)
                        micro_points = []
                        for j in range(start_idx + 1, len(data) - 1):
                            if self.slopeDir == 'down':
                                current = data.iloc[j]['high']
                                prev = data.iloc[j-1]['high']
                                next_val = data.iloc[j+1]['high']
                                is_extreme = current > prev and current > next_val
                            else:
                                current = data.iloc[j]['low']
                                prev = data.iloc[j-1]['low']
                                next_val = data.iloc[j+1]['low']
                                is_extreme = current < prev and current < next_val
                            
                            if is_extreme:
                                micro_points.append({
                                    'date': data.index[j],
                                    'price': current
                                })
                        
                        if micro_points:
                            # Use best fit line through last point and micro points
                            all_points = [point_b] + micro_points
                            line_params = self.calculate_best_fit_line(all_points)
                            if len(micro_points) >= 2:
                                trendline = {
                                    'slope': line_params['slope'],
                                    'intercept': line_params['intercept'],
                                    'start_date': point_b['date'],
                                    'end_date': data.index[-1]
                                }
                                trendlines.append(trendline)
                                continue
                        
                        # If no micro points found, use all highs/lows if more than 2 bars remain
                        remaining_bars = len(data) - start_idx - 1
                        if remaining_bars >= 2:
                            bar_points = []
                            for j in range(start_idx + 1, len(data)):
                                price = data.iloc[j]['high'] if self.slopeDir == 'down' else data.iloc[j]['low']
                                bar_points.append({
                                    'date': data.index[j],
                                    'price': price
                                })
                            
                            if bar_points:
                                # Initial best fit line
                                all_points = [point_b] + bar_points
                                line_params = self.calculate_best_fit_line(all_points)
                                
                                # Find points that exceed the best fit line
                                exceeding_points = [point_b]  # Always include the starting point
                                for j in range(start_idx + 1, len(data)):
                                    x_val = data.index[j].timestamp() / (24 * 3600)
                                    projected_price = line_params['slope'] * x_val + line_params['intercept']
                                    
                                    if self.slopeDir == 'down':
                                        actual_price = data.iloc[j]['high']
                                        if actual_price > projected_price:
                                            exceeding_points.append({
                                                'date': data.index[j],
                                                'price': actual_price
                                            })
                                    else:  # uptrend
                                        actual_price = data.iloc[j]['low']
                                        if actual_price < projected_price:
                                            exceeding_points.append({
                                                'date': data.index[j],
                                                'price': actual_price
                                            })
                                
                                # Create final best fit line through exceeding points
                                if len(exceeding_points) >= 2:
                                    final_line_params = self.calculate_best_fit_line(exceeding_points)
                                    trendline = {
                                        'slope': final_line_params['slope'],
                                        'intercept': final_line_params['intercept'],
                                        'start_date': point_b['date'],
                                        'end_date': data.index[-1]
                                    }
                                else:
                                    # Fallback to original best fit if not enough exceeding points
                                    trendline = {
                                        'slope': line_params['slope'],
                                        'intercept': line_params['intercept'],
                                        'start_date': point_b['date'],
                                        'end_date': data.index[-1]
                                    }
                                trendlines.append(trendline)
                                continue
                
                trendline = self.project_line(data, point_b, point_a)
                trendlines.append(trendline)
        

        
        return trendlines

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the analysis and return DataFrame with trendlines."""
        self.names = []
        result = data.copy()
        trendlines = self.find_trendlines(data)
        
        for i, trendline in enumerate(trendlines):
            name = f"{self.name}_{self.slopeDir.upper()}_{i+1}"
            self.names.append(name)
            result[name] = np.nan
            mask = (result.index >= trendline['start_date']) & \
                   (result.index <= trendline['end_date'])
            x_values = np.array([(d.timestamp() / (24 * 3600)) for d in result.index[mask]])
            result.loc[mask, name] = trendline['slope'] * x_values + trendline['intercept']
        
        return result


# todo: add the following classes to the strategies/ta.py file
@dataclass
class LineMerge(TA):
    """Takes multiple trendlines and merges them into a single line.
    Checks slope direction and tolerance for merging.
    """

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the analysis and return DataFrame with merged trendlines."""
        pass