import pandas as pd
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np

def preprocess_data(func):
    def wrapper(self, data: pd.DataFrame, *args, **kwargs):
        data = self.compute_rows_to_update(data, self.names, self.rowsToUpdate)
        return func(self, data, *args, **kwargs)
    return wrapper

@dataclass
class TA(ABC):
    column: str = None

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



@dataclass
class MA(TA):
    period: int = 20

    def __post_init__(self):
        self.names = f"MA_{self.column[:2]}_{self.period}"
        self.rowsToUpdate = self.period 

    @preprocess_data
    def run(self, data: pd.DataFrame) -> pd.Series:
        return data[self.column].rolling(window=self.period).mean().rename(self.names)

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
class ATR(TA):
    hi_col: str = 'high'
    lo_col: str = 'low'
    span: int = 14  # Common default span for ATR

    def __post_init__(self):
        self.name_atr = f"ATR_{self.span}"
        self.names = [self.name_atr]
        self.rowsToUpdate = 200

    @preprocess_data
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Calculate True Range (TR)
        df['TR'] = df[self.hi_col] - df[self.lo_col]

        # Calculate ATR using a rolling mean of the True Range
        df[self.name_atr] = df['TR'].rolling(window=self.span, min_periods=1).mean()

        # Drop the intermediate 'TR' column
        df.drop(columns=['TR'], inplace=True)

        return df[[self.name_atr]]
    
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
        last_close = df['close'].iloc[-1]
        
        # Initialize all columns with NaN
        for name in self.names:
            df[name] = np.nan
        
        if startwith == 'res':
            res_levels = self._find_levels(df, last_close, 'res', 2)
            sup_levels = self._find_levels(df, last_close, 'sup', 2)
        else:
            sup_levels = self._find_levels(df, last_close, 'sup', 2)
            res_levels = self._find_levels(df, last_close, 'res', 2)
        
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
