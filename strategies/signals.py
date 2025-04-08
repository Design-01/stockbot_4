from dataclasses import dataclass
from typing import Tuple, Any
import random
import pandas as pd
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Dict, Union, Literal
from collections import defaultdict
from abc import ABC, abstractmethod
from chart.chart import ChartArgs

def trace(fromPrice:float, toPrice:float, priceNow:float):
    """ Determines how far a price has traced from one price to another price expressed as a % of the total dffernce between the fromPrice and toPrice. 
        example --         |  40%  |            |    trace  = 40%
        example --      from      now           to
        example --      100       104           110
        Works for either a pandas series or just floats or any combination of the two type for each arg
        Returns np.array if series args are given
        Returns float if float args are given
    """
    totalDiff  = np.where( abs(fromPrice - toPrice) > 0,  abs(fromPrice - toPrice), np.nan) # remove double scalars warning 
    traceVal   = abs(priceNow - fromPrice)
    tracePcent = traceVal / totalDiff * 100

    switch_up   = (priceNow < fromPrice) & ( fromPrice < toPrice) # price expected to go up but goes lower
    switch_down = (priceNow > fromPrice) & ( fromPrice > toPrice) # price expected to go lower but goes higher
    # if price is switched then make negative 
    tracePcent = np.where(switch_up | switch_down, tracePcent *-1, tracePcent )
    if np.size(tracePcent) == 1: return round(tracePcent.item(), 1) # return as float
    else : return tracePcent.round(1) # return as np array

# def normalize_old(val:float, minVal:float, maxVal:float, roundTo:int=2):
#     """normalizes the value between the min and max."""
#     if minVal == maxVal: return 0 # cannot divide by zero
#     if minVal < maxVal: 
#         if val <= minVal: val = minVal
#         if val >= maxVal: val = maxVal
#     else:
#         if val >= minVal: val = minVal
#         if val <= maxVal: val = maxVal
    
#     r = round((val - minVal) / (maxVal - minVal) * 100, roundTo)
#     # Don't return NaN
#     if r > -100: 
#         return r
#     return 0

def normalize(value: float, min_val: float, max_val: float) -> float:
    if min_val >= max_val:
        raise ValueError('min_val must be less than max_val')
    
    if value == 0:
        return 0

    if value >= 0:
        min_val = max(0, min_val)
        if (max_val - min_val) == 0:
            return 0
        result = (value - min_val) / (max_val - min_val) * 100 
    else:
        max_val = min(0, max_val)
        if (min_val - max_val) == 0:
            return 0
        result = (value - max_val) / (min_val - max_val) * -100 
    
    return max(-100, min(100, result))


def normalize_int(val:float, minVal:float, maxVal:float):
    """normalizes the value between the min and max."""    
    return int(normalize(val, minVal, maxVal))

def is_gap_pivot_crossover(df: pd.DataFrame, pivot_col: str, ls: str) -> bool:
    """
    Check if there is a gap over the most recent pivot at the last bar.
    This function handles finding the most recent pivot and checking if it's been gapped over.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing price data and pivot points
    pivot_col : str
        Name of the column containing pivot points
    ls : str
        'LONG' or 'SHORT' to indicate direction
        
    Returns:
    --------
    bool
        True if the most recent pivot has been gapped over, False otherwise
    """
    if ls not in ['LONG', 'SHORT']:
        raise ValueError("ls parameter must be either 'LONG' or 'SHORT'")
        
    try:
        # Need at least 2 bars
        if len(df) < 2:
            return False
            
        # Get current bar's open and previous bar's close
        current_open = df['open'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        
        # Get data up to but not including current bar
        prior_data = df.iloc[:-1]
        
        # Find the most recent pivot
        prior_pivots = prior_data[pivot_col].dropna()
        if len(prior_pivots) == 0:
            return False
            
        most_recent_pivot = prior_pivots.iloc[-1]
        
        if ls == 'LONG':
            # For longs: prev_close < pivot < current_open (gapped up over pivot)
            return (current_open > prev_close and              # Must be a gap up
                   most_recent_pivot > prev_close and          # Pivot must be above prev close
                   most_recent_pivot < current_open)           # Pivot must be below current open
        elif ls == 'SHORT':
            # For shorts: current_open < pivot < prev_close (gapped down over pivot)
            return (current_open < prev_close and              # Must be a gap down
                   most_recent_pivot < prev_close and          # Pivot must be below prev close
                   most_recent_pivot > current_open)           # Pivot must be above current open
                   
    except (KeyError, IndexError):
        return False

def get_valid_pb(ls, df, pointCol:str, minLen:int=3):
    if minLen < 3:
        raise ValueError("get_valid_pb :: minLen must be at least 3")
    # check if names exists in df
    if not pointCol in df.columns:
        return None
    
    # print(f'{ls} {pointCol=} {toCol=} {atrCol} {minLen} {atrMultiple}')
    # check if has points
    points = df[pointCol].dropna()
    # print(f'length of points: {len(points)}')
    if len(points) < 1:
        return None
    
    # check if window long enough
    w0 = df.loc[points.index[-1]:]
    # print(f'{df.index[-1]} :: lenght of w0: {len(w0)}')
    if len(w0) < minLen:
        return None
    
    if ls == 'LONG':
        #check if high < two previous high ago
        if not w0.high.iat[-1] < w0.high.iat[-3] :
            # print(f'{df.index[-1]} :: not w0.high.iat[-1] < w0.high.iat[-3]')
            return None
        
        return w0
    
    if ls == 'SHORT':       
        #check if low > two previous low ago
        if not w0.low.iat[-1] > w0.low.iat[-3] :
            return None
        
        return w0
    
    return None

#£ Done
@dataclass
class Signals(ABC):
    name: str = ''
    normRange: Tuple[int, int] = (0, 100)
    ls: str = 'LONG'
    lookBack: int = 1
    chartArgs: ChartArgs = None
    invertScoreIfShort: bool = False

    def __post_init__(self):
        self.name = f"{self.ls[0]}_{self.name}"
        self.names = [self.name]

    def add_chart_args(self, chartArgs: ChartArgs):
        self.chartArgs = chartArgs
        return self

    def get_score(self, val):
        if isinstance(val, pd.Series):
            # Apply the function to each element in the series
            # return val.apply(lambda x: 0 if x == 0 else normalize(x, self.normRange[0], self.normRange[1]))
            return val.apply(lambda x: 0 if x == 0 else (np.nan if pd.isna(x) else normalize(x, self.normRange[0], self.normRange[1])))
        else:
            # Handle single value
            if pd.isna(val):
                return np.nan
            if val == 0:
                return 0
            return normalize(val, self.normRange[0], self.normRange[1])

    def get_window(self, df, ls, w=1):
        if len(df) <= 2:
            return None

        subset_1 = df.dropna(subset=[self.maxCol, self.minCol], how='all')[[self.maxCol, self.minCol]]
        if subset_1.empty:
            return None

        self.subset_2 = subset_1 if subset_1.index[-1] != df.index[-1] else subset_1[:-1]
        col = self.maxCol if ls == 'LONG' else self.minCol

        if self.subset_2[col].dropna().empty:
            return None

        idx_of_last_point = self.subset_2[col].dropna().index[-1]
        w0 = df.loc[idx_of_last_point:].copy()

        if len(w0) > 1 and w == 0:
            return w0

        self.subset_3 = self.subset_2[:w0.index[0]].copy()
        if len(self.subset_3) > w + 1:
            return df.loc[self.subset_3.index[-(w + 2)]:self.subset_3.index[-(w + 1)]].copy()

        return None
    
    def return_series(self, index: pd.DatetimeIndex, val: Union[float, pd.Series]):
        if isinstance(val, pd.Series):
            # If single column Series, use existing logic
            if len(val.shape) == 1:
                val.name = self.name
                return val
            # If multiple columns, handle each column
            for col, name in zip(val.columns, self.names):
                val[col].name = name
            return val
        return pd.Series(index=[index], data=val, name=self.name)
    
    @abstractmethod
    def _compute_row(self, df: pd.DataFrame) -> float:
        """This method is to compute each row in the lookback period."""
        pass


    def run(self, df: pd.DataFrame = pd.DataFrame()) -> pd.Series:
        """Generate signal scores for the lookback period."""
        if len(df) < 10:
            return self.return_series(df.index[-1:], self.get_score(0))
        
        lookback = min(self.lookBack, len(df))  # Include all rows in the lookback period
        if lookback < 1:
            return self.return_series(df.index[-1:], self.get_score(0))
        
        result_series = pd.Series(np.nan, index=df.index[-lookback:])
        
        for i in range(lookback):
            current_idx = -(lookback - i) + 1
            if abs(current_idx) > len(df):  # Skip if index out of bounds
                continue

            current_window = df.iloc[: len(df) + current_idx]
            if current_window.empty:
                continue
            
            val = self._compute_row(current_window)
            if not pd.isna(val):
                result_series.iloc[i] = float(val)

        #  if the invertScoreIfShort is True then invert the score if also SHORT.  Allows for example Relative market weakness to be turned int strength if shorting
        score = self.get_score(result_series)*-1 if self.invertScoreIfShort and self.ls=='SHORT' else self.get_score(result_series) 
        return self.return_series(df.index[-lookback:], score)




            

#£ Done
@dataclass
class ExampleClass(Signals):
    """
    Example class to show how to create a new signal class.
    """

    def _compute_row(self, df: pd.DataFrame) -> float:
        """This method is to compute each row in the lookback period."""
        return random.randint(0, 100)


@dataclass
class GetSignalAtPoint(Signals):
    name: str = 'SaP'
    pointCol: str = ''
    sigCol: str = ''
    pointsAgo: int = 1

    def _compute_row(self, df: pd.DataFrame) -> float:
        """
        Gets a Signal at a high Point or a low Point by looking back 
        to the nth most recent point and getting the corresponding signal.
        """
        points = df[self.pointCol].dropna()
        if len(points) < self.pointsAgo:
            return 0.0
        
        index = points.index[-self.pointsAgo]
        
        return df[self.sigCol].loc[index]


@dataclass
class GetMaxSignalSincePoint(Signals):
    name: str = 'MaxSig'
    pointCol: str = ''
    sigCol: str = ''
    pointsAgo: int = 1

    def _compute_row(self, df: pd.DataFrame) -> float:
        """
        Gets a Signal at a high Point or a low Point by looking back 
        to the nth most recent point and getting the corresponding signal.
        """
        points = df[self.pointCol].dropna()
        if len(points) < self.pointsAgo:
            return 0.0
        
        index = points.index[-self.pointsAgo]
        
        return df[self.sigCol].loc[index:].max()


# --------------------------------------------------------------
# ------- P U L L B A C K   S I G N A L S ----------------------
# --------------------------------------------------------------
#£ Done
@dataclass
class PB_PctHLLH(Signals):
    """
    Measures the percentage of pullback bars showing lower highs (LONG) or higher lows (SHORT).
    
    This signal evaluates the quality of a pullback by analyzing the sequence of price bars
    to determine how many maintain the expected structure:
    - For LONG setups: Counts bars with lower highs than the previous bar
    - For SHORT setups: Counts bars with higher lows than the previous bar
    
    A higher percentage indicates a more consistent directional pullback with proper
    price structure, which typically leads to better trade entries.
    
    Parameters
    ----------
    name : str, default 'PB_PctHLLH'
        Base name for the signal. The final signal name will be constructed as
        "{ls[0]}_{name}" where ls[0] is the first character of the trade direction.
    
    pointCol : str, default ''
        Column name containing pivot points used to identify pullbacks.
        Must be set to a valid column name in the DataFrame.
    
    atrCol : str, default ''
        Column name containing ATR (Average True Range) values.
        Used in pullback validation.
    
    atrMultiple : float, default 1.0
        Minimum number of ATRs required between pointCol value and target column.
        Used in pullback validation (handled by get_valid_pb function).
    
    minPbLen : int, default 3
        Minimum number of bars required to consider a valid pullback.
    
    Attributes
    ----------
    names : list
        List containing the signal name, used for identification in the signal system.
    
    prev_val : float
        Stores the previously calculated value to maintain signal persistence
        for one bar when no valid pullback is found.
    
    Methods
    -------
    _use_prev_if_none(val: float) -> float
        Helper method that persists the previous signal value for one bar when
        the current calculation yields no result (None or 0).
        
    _compute_row(df: pd.DataFrame) -> float
        Calculates the percentage of bars in the pullback that maintain the proper
        higher/lower structure based on trade direction.
        Returns a value between 0 and 100.
    """
    name: str = 'PB_PctHLLH'
    pointCol: str = ''
    atrCol: str = ''
    atrMultiple: float = 1.0 # minimum number of ATRs required between pointCol low and toCol
    minPbLen: int = 3
    
    def __post_init__(self):
        self.name = f"{self.ls[0]}_{self.name}"
        self.names = [self.name]   
        self.prev_val = 0.0

    def _use_prev_if_none(self, val):
        """If the value is None or 0, use the previous value.
        The idea is to persista ofr just one bar. """
        if (pd.isna(val) or val == 0.0) and self.prev_val > 0:
            val = self.prev_val
            self.prev_val = 0.0
        else:
            self.prev_val = val
        return val
   

    #$ **kwargs is used to allow any signal arguments to be passed to any run method. This so that the same run method can be used when looping through signals.
    def _compute_row(self, df:pd.DataFrame=pd.DataFrame(), **kwargs):
        """Computes the % of bars that have a lower highs (BULL pullback, so downward)
        Vice versa for BEAR case. So this is only for pullbacks not overall trends. """

        window  = get_valid_pb( ls=self.ls, df=df, pointCol=self.pointCol, minLen=self.minPbLen)
        
        if window is None:
            return self._use_prev_if_none(0.0)
        

        if self.ls == 'LONG': # if there are more than 2 bars in the pullback from the high
            # eg fhp.high < fhp.high.shift() retruns a series of bools. 
            # 2000-01-01 00:13:00    False
            # 2000-01-01 00:14:00     True
            # 2000-01-01 00:15:00    False
            # then [1:] removes the first bar becasue it will always be False
            # 2000-01-01 00:14:00     True
            # 2000-01-01 00:15:00    False
            # then mean() returns the mean of the bools.
            val = (window.high < window.high.shift())[1:].mean() * 100
            return self._use_prev_if_none(val)
    

        if self.ls == 'SHORT' : # if there are more than 2 bars in the pullback from the low
            val = (window.low > window.low.shift())[1:].mean() * 100
            return self._use_prev_if_none(val)
    
        return 0.0


@dataclass
class PB_ASC(Signals):
    """
    Pullback All Same Color (ASC) Signal that measures color consistency in pullbacks.
    
    This class evaluates the quality of a pullback by calculating the percentage of
    bars that have the same color (direction) as expected for the pullback type.
    For a high-quality signal:
    - In LONG setups: Pullback bars should be red (bearish)
    - In SHORT setups: Pullback bars should be green (bullish)
    
    The signal returns a percentage value (0-100) representing how many bars in
    the pullback match the expected color, with 100% indicating perfect consistency.
    
    Parameters
    ----------
    name : str, default 'PB_ASC'
        Base name for the signal. The final signal name will be constructed as
        "{ls[0]}_{name}" where ls[0] is the first character of the trade direction.
    
    pointCol : str, default ''
        Column name containing pivot points used to identify pullbacks.
        Must be set to a valid column name in the DataFrame.
    
    atrMultiple : float, default 1.0
        Minimum number of ATRs required between pointCol low and target column.
        Used in pullback validation (handled by get_valid_pb function).
    
    minPbLen : int, default 3
        Minimum number of bars required to consider a valid pullback.
    
    Attributes
    ----------
    names : list
        List containing the signal name, used for identification in the signal system.
    
    prev_val : float
        Stores the previously calculated value to maintain signal persistence
        for one bar when no valid pullback is found.
    
    Methods
    -------
    _use_prev_if_none(val: float) -> float
        Helper method that persists the previous signal value for one bar when
        the current calculation yields no result (None or 0).
        
    _compute_row(df: pd.DataFrame) -> float
        Calculates the percentage of bars in the pullback that have the correct color
        for the current trade direction (red for LONG, green for SHORT).
        Returns a value between 0 and 100.
    
    Notes
    -----
    - A higher percentage indicates a more consistent directional pullback
    - 100% means all bars in the pullback have the expected color
    - The first bar of the pullback is excluded from the calculation
    - Uses the get_valid_pb() function to identify valid pullbacks
    - Signal persists for one additional bar after a valid pullback disappears
    """
    name: str = 'PB_ASC'
    pointCol: str = ''
    atrMultiple: float = 1.0 # minimum number of ATRs required between pointCol low and toCol
    minPbLen: int = 3
    
    
    def __post_init__(self):
        self.name = f"{self.ls[0]}_{self.name}"
        self.names = [self.name]
        self.prev_val = 0.0

    def _use_prev_if_none(self, val):
        """If the value is None or 0, use the previous value.
        The idea is to persista ofr just one bar. """
        if (pd.isna(val) or val == 0.0) and self.prev_val > 0:
            val = self.prev_val
            self.prev_val = 0.0
        else:
            self.prev_val = val
        return val

    def _compute_row(self, df:pd.DataFrame=pd.DataFrame()):
        window  = get_valid_pb( ls=self.ls, df=df, pointCol=self.pointCol, minLen=self.minPbLen)
        
        if window is None:
            return self._use_prev_if_none(0.0)
        
        total_bars = len(window) -1
        if len(window) > 2:
            if self.ls == 'LONG':
                same_colour_bars = len(window[window['close'] < window['open']]) # red bars
                val =  (same_colour_bars / total_bars) * 100
                return self._use_prev_if_none(val)

            if self.ls == 'SHORT':
                same_colour_bars = len(window[window['close'] > window['open']]) # green bars
                val =  (same_colour_bars / total_bars) * 100
                return self._use_prev_if_none(val)

        return 0.0
    

@dataclass
class PB_CoC_ByCountOpBars(Signals):
    """
    Change of Character (CoC) Signal based on opposite bars count within a pullback.
    
    This class identifies 'Change of Character' setups by analyzing the sequence
    of candles in a pullback, specifically looking for a change in direction
    (green to red or vice versa) at the end of a pullback. The signal strength
    is calculated as the percentage of consecutive opposite-colored candles
    preceding the signal bar.
    
    For LONG signals: Identifies when a pullback with consecutive red candles
                      ends with a green candle (suggesting bullish reversal)
    For SHORT signals: Identifies when a pullback with consecutive green candles
                       ends with a red candle (suggesting bearish reversal)
    
    Parameters
    ----------
    name : str, default 'PB_CoC_OpBars'
        Base name for the signal. The final signal name will be constructed as
        "{ls[0]}_{name}" where ls[0] is the first character of the trade direction.
    
    pointCol : str, default ''
        Column name containing pivot points used to identify pullbacks.
        Must be set to a valid column name in the DataFrame.
    
    minPbLen : int, default 3
        Minimum number of bars required to consider a valid pullback.
        Lower values may produce more signals but with lower quality.
    
    Attributes
    ----------
    names : list
        List containing the signal name, used for identification in the signal system.
    
    prev_window : pd.DataFrame, optional
        Stores the previously identified pullback window to maintain continuity
        across calculations when no new pullback is found.
    
    Methods
    -------
    _compute_row(df: pd.DataFrame) -> float
        Calculates the signal strength based on the proportion of consecutive
        opposite-colored candles in the pullback.
        Returns a percentage value between 0 and 100.
    
    Notes
    -----
    - The signal requires the 'ls' (long/short) parameter to be set in the parent class
    - Uses the get_valid_pb() function to identify pullbacks in price action
    - Requires OHLC data in the input DataFrame
    - The signal is triggered only when the current bar shows a change of character
      (shift from bearish to bullish for LONG, or bullish to bearish for SHORT)
    """
    name: str = 'PB_CoC_OpBars'
    pointCol: str = ''
    minPbLen: int = 3

    
    def __post_init__(self):
        self.name = f"{self.ls[0]}_{self.name}"
        self.names = [self.name]
        self.prev_window = None

    def _compute_row(self, df: pd.DataFrame = pd.DataFrame()):
        window = get_valid_pb(ls=self.ls, df=df, pointCol=self.pointCol, minLen=self.minPbLen)
        if window is None or len(window) <= 1:
            if self.prev_window is not None:
                window = self.prev_window
            else: 
                return 0.0
        else:
            self.prev_window = window
        
        candles = []
        for _, row in window.iterrows():
            candles.append({
                'open': row['open'],
                'close': row['close'],
                'is_green': row['close'] > row['open']
            })
        
        # Reverse to start from most recent
        candles = candles[::-1]
        
        # Check if signal is present (either in last pullback candle or current bar)
        current_bar_is_green = df.close.iat[-1] > df.open.iat[-1]
        prev_bar_is_green = df.close.iat[-2] > df.open.iat[-2]
        prev_bar_is_opposite = prev_bar_is_green != current_bar_is_green
        # last_candle_is_green = candles[0]['is_green'] or (current_bar_is_green and prev_bar_is_opposite)
        last_candle_is_green = current_bar_is_green and prev_bar_is_opposite
        consecutive_count = 0
        
        if self.ls == 'LONG':
            # For LONG, we want the last candle to be green (bullish)
            if not last_candle_is_green:
                return 0.0
            
            # Count consecutive red candles before the last green one
            for i in range(1, len(candles)):
                if candles[i]['is_green']:
                    break
                consecutive_count += 1
                
        elif self.ls == 'SHORT':
            # For SHORT, we want the last candle to be red (bearish)
            if last_candle_is_green:
                return 0.0
            
            # Count consecutive green candles before the last red one
            for i in range(1, len(candles)):
                if not candles[i]['is_green']:
                    break
                consecutive_count += 1
        
        # Calculate as percentage of total bars minus the signal bar
        total_bars = len(candles) - 1  # Exclude the signal bar
        if total_bars > 0:
            return (consecutive_count / total_bars) * 100
        return 0.0
    
    

#£ Done
@dataclass
class PB_Overlap(Signals):
    """
    Measures the quality of a pullback based on the overlap between consecutive bars.
    
    This signal analyzes pullbacks by calculating the percentage overlap between
    consecutive bars, returning a score that indicates how "smooth" the pullback is.
    A perfect pullback is considered one where bars consistently overlap by 50%
    with the previous bar.
    
    The calculation differs based on trade direction:
    - For LONG positions: Measures high of current bar to low of previous bar
    - For SHORT positions: Measures high of previous bar to low of current bar
    
    The signal then averages these overlaps across the pullback and scores based
    on how close the average is to the optimal 50% overlap.
    
    Parameters
    ----------
    name : str, default 'PB_Olap'
        Base name for the signal. The final signal name will be constructed as
        "{ls[0]}_{name}" where ls[0] is the first character of the trade direction.
    
    pointCol : str, default ''
        Column name containing pivot points used to identify pullbacks.
        Must be set to a valid column name in the DataFrame.
    
    minPbLen : int, default 3
        Minimum number of bars required to consider a valid pullback.
    
    Attributes
    ----------
    names : list
        List containing the signal name, used for identification in the signal system.
    
    prev_val : float
        Stores the previously calculated value to maintain signal persistence
        for one bar when no valid pullback is found.
    
    Methods
    -------
    _use_prev_if_none(val: float) -> float
        Helper method that returns the previous value if the current value is None or 0.
        This helps to persist the signal for one additional bar.
        
    _compute_row(df: pd.DataFrame) -> float
        Calculates the overlap quality score for the pullback.
        Returns a value between 0 and 100, where 100 represents a perfect 50% overlap.
    
    Notes
    -----
    - Scores closer to 100 indicate higher quality pullbacks with consistent overlap
    - A score of 100 represents a pullback with exactly 50% average overlap
    - The signal requires price bars with high/low values
    - Uses the get_valid_pb() function to identify pullbacks
    """
    name  : str = 'PB_Olap'
    pointCol: str = ''   
    minPbLen: int = 3 

    def __post_init__(self):
        self.name = f"{self.ls[0]}_{self.name}"
        self.names = [self.name]
        self.prev_val = 0.0

    def _use_prev_if_none(self, val):
        """If the value is None or 0, use the previous value.
        The idea is to persista ofr just one bar. """
        if (pd.isna(val) or val == 0.0) and self.prev_val > 0:
            val = self.prev_val
            self.prev_val = 0.0
        else:
            self.prev_val = val
        return val

    def _compute_row(self, df:pd.DataFrame):
        window  = get_valid_pb( ls=self.ls, df=df, pointCol=self.pointCol, minLen=self.minPbLen)
        
        if window is None:
            return self._use_prev_if_none(0.0)
            
        prev = window.shift(1).copy()
        olap          = window.high - prev.low if self.ls == 'LONG' else prev.high - window.low
        prev_rng      = abs(prev.high - prev.low)
        olap_pct      = olap / prev_rng 
        olap_pct_mean = olap_pct.mean()

        # 150 is to scale the score to be between 0 and 100. playing around with this number will change the sensitivity of the score
        # 100 is the best score as it means the olap_pct is 50% which is the best
        # calculate score based on olap_pct
        optimal_olap_pct = 0.5
        score = 100 - abs(olap_pct_mean - optimal_olap_pct) * 150 
        val = max(score, 0)
        return self._use_prev_if_none(val)


#£ Done
@dataclass
class Trace(Signals):
    name      : str = 'Trace'
    usePoints : bool = True # if True then use points else use fromPriceCol and toPriceCol
    fromCol   : str = ''
    toCol     : str = ''
    optimalRtc: float = None # optimal retracement eg if 50 then 50% is the best retracement
  
    def __post_init__(self):
        self.name = f"{self.ls[0]}_{self.name}"
        self.names = [self.name]

    def compute_from_mid_trace(self, retracement, optimalRetracement):
        """ compute_from_mid_trace is based on retracement. optimal retracement is 50%. 
        Socre decreases by 2 for every 1% away from optimal retracement in either direction. 
        eg 49% or 51% retracement will have a score of 98. 48% or 52% retracement will have a score of 96.
        eg 10% or 90% retracement will have a score of 20. 0% or 100% retracement will have a score of 0.
        Can also work with any optimal values eg 200% or 300% retracement.
        
        """
        
        score = 100 - (2 * abs(retracement - optimalRetracement))
        if score < 0:
            score = 0
        return score

    #$ **kwargs is used to allow any signal arguments to be passed to any run method. This so that the same run method can be used when looping through signals.
    def _compute_row(self, df:pd.DataFrame=pd.DataFrame()):
        """trace high 1 ago to low 1 ago and compare  """

        def last_change_index(df, col):
            changes = df[col] != df[col].shift(1)
            last_change_index = changes[::-1].idxmax()
            return last_change_index

        if not df.empty:
            fromPrice = 0
            toPrice = 0

            # only want to get values for the direction the trace. 
            # it will return values on the bouce back as well as the pullback if we don't do this
            # get the recnt change in prices of the HPs and LPs
            from_last_change = last_change_index(df, self.fromCol)
            to_last_change = last_change_index(df, self.toCol)

            if from_last_change < to_last_change:
                return 0

            # long W1 is the move up from the low of W1 to the high of W1 which is the most recent HP point (the start of the pullback represented by fromHP) 
            if self.ls == 'LONG':
                fromPrice = df[self.fromCol].iat[-1]
                toPrice   = df[self.toCol].iat[-1]

            # short W1 is the move down from the high of W1 to the low of W1 which is the most recent LP point (the start of the pullback represented by fromLP)
            elif self.ls == 'SHORT': 
                fromPrice = df[self.fromCol].iat[-1]
                toPrice   = df[self.toCol].iat[-1]

            priceNow  = df.close.iat[-1]

            # avoid div by zero
            if fromPrice != toPrice:
                t = trace(fromPrice, toPrice, priceNow)
                if self.optimalRtc:
                    return self.compute_from_mid_trace(t, self.optimalRtc)
                
        return 0


@dataclass
class Lower:
    name     : str = 'Lower'
    col : str = ''
    span: int = 1
    allLower: bool = False

    def __post_init__(self):
        self.name = f"{self.name}_{self.col}_{self.span}"
        self.names = [self.name]  

    def _compute_row(self, df:pd.DataFrame=pd.DataFrame()):
        """Computes the % of bars that have a lower highs (BULL pullback, so downward)
        Vice versa for BEAR case. So this is only for pullbacks not overall trends. """

        if not df.empty:
            if len(df) > self.span:
                if self.allLower:
                    window = df[self.col].iloc[-self.span:]
                    mask = window[self.col] < window[self.col].shift(1)
                    return mask.all() 
                else:
                    return df[self.col].iloc[-1] < df[self.col].iloc[-self.span]
        return 0  
    

@dataclass
class Higher:
    name     : str = 'Higher'
    col : str = ''
    span: int = 1
    allHigher: bool = False

    def __post_init__(self):
        self.name = f"{self.name}_{self.col}_{self.span}"
        self.names = [self.name]  

    def _compute_row(self, df:pd.DataFrame=pd.DataFrame()):
        """Computes the % of bars that have a lower highs (BULL pullback, so downward)
        Vice versa for BEAR case. So this is only for pullbacks not overall trends. """

        if not df.empty:
            if len(df) > self.span:
                if self.allHigher:
                    window = df[self.col].iloc[-self.span:]
                    mask = window[self.col] > window[self.col].shift(1)
                    return mask.all()
                else:
                    return df[self.col].iloc[-1] > df[self.col].iloc[-self.span]
        return 0
    

@dataclass
class ColourWithLS:
    """Chceks if the current bar is the same colour as the trade direction."""
    name: str = 'ColourWithLS'
    ls: str = 'LONG'

    def __post_init__(self):
        self.name = f"{self.ls[0]}_{self.name}"
        self.names = [self.name]

    def _compute_row(self, df:pd.DataFrame=pd.DataFrame()):
        if not df.empty:
            if self.ls == 'LONG':
                return df['open'].iat[-1] > df['close'].iat[-1]
            elif self.ls == 'SHORT':
                return df['open'].iat[-1] < df['close'].iat[-1]
        return 0

# --------------------------------------------------------------
# ------- R E V E R S A L   S I G N A L S ----------------------
# --------------------------------------------------------------

@dataclass
class TouchWithBar(Signals):
    """
    TouchWithBar is a signal class that computes a normalized score (0-100) based on how close
    price is to a specified level, scaled by ATR. Different scales are used for approaching vs
    overshooting the level.

    Attributes:
        name (str): The name of the signal. Default is 'Touch'.
        valCol (str): The column name containing the level to touch.
        atrCol (int): The column index containing ATR values.
        direction (str): The direction of approach ('up' or 'down').
        toTouchAtrScale (float): Maximum ATR distance when approaching (score=0 at this distance).
        pastTouchAtrScale (float): Maximum ATR distance when overshooting (score=0 at this distance).
    """
    name: str = 'Touch'
    valCol: str = ''
    atrCol: int = 1
    direction: str = 'down'
    toTouchAtrScale: float = 10.0    # Max ATR distance for approaching
    pastTouchAtrScale: float = 2.0    # Max ATR distance for overshooting
    
    def __post_init__(self):
        """Post-initialization to set up the signal name."""
        self.name = f"{self.name}_{self.direction}_{self.valCol}"
        self.names = [self.name]

    def _normalize_score(self, atr_distance: float, max_atr: float) -> float:
        """
        Normalize ATR distance to a score between 0 and 100.
        
        Args:
            atr_distance (float): Distance from level in ATR units
            max_atr (float): Maximum ATR distance (score will be 0 beyond this)
            
        Returns:
            float: Normalized score between 0 and 100
        """
        abs_distance = abs(atr_distance)
        if abs_distance >= max_atr:
            return 0.0
        
        return (1 - (abs_distance / max_atr)) * 100

    def _compute_row(self, df: pd.DataFrame) -> float:
        """
        Compute the score for the current row based on price's distance from the level.
        
        Args:
            df (pd.DataFrame): DataFrame containing price and ATR data
            
        Returns:
            float: Score between 0 and 100
        """
        if len(df) < 10:
            return 0.0
        
        if self.direction == 'down':
            level = df[self.valCol].iat[-1]
            bar_low = df.low.iat[-1]
            bar_close = df.close.iat[-1]
            atr = df[self.atrCol].iat[-1]

            # Perfect touch (low crosses but close recovers)
            if bar_low <= level <= bar_close:
                return 100.0

            # Calculate ATR distance
            if bar_low > level:
                # Approaching from above
                atr_distance = (bar_low - level) / atr
                return self._normalize_score(atr_distance, self.toTouchAtrScale)
            else:
                # Overshooting
                atr_distance = (level - bar_close) / atr
                return self._normalize_score(atr_distance, self.pastTouchAtrScale)
        
        elif self.direction == 'up':
            level = df[self.valCol].iat[-1]
            bar_high = df.high.iat[-1]
            bar_close = df.close.iat[-1]
            atr = df[self.atrCol].iat[-1]

            # Perfect touch (high crosses but close recovers)
            if bar_high >= level >= bar_close:
                return 100.0

            # Calculate ATR distance
            if bar_high < level:
                # Approaching from below
                atr_distance = (level - bar_high) / atr
                return self._normalize_score(atr_distance, self.toTouchAtrScale)
            else:
                # Overshooting
                atr_distance = (bar_close - level) / atr
                return self._normalize_score(atr_distance, self.pastTouchAtrScale)

        return 0.0


@dataclass
class Retest(Signals):
    """
    Identifies price retests of specific levels within an ATR-based range.
    
    This class detects when current price action (high or low depending on direction)
    is testing a previously established level. It counts how many points or bars from the 
    specified value column fall within a defined ATR range (are touched) of the current price.
    So how may points or bars are touched. 
    
    Parameters
    ----------
    name : str, default 'Retest'
        Base name for the signal. The final signal name will be constructed as 
        "{name}_{direction}_{valCol}".
    
    direction : str, default 'down'
        Direction of the retest:
        - 'down': Uses the low price to test for support retests
        - 'up': Uses the high price to test for resistance retests
    
    atrCol : str, default ''
        Column name in the DataFrame containing ATR values. Used to define the
        range for determining valid retests.
    
    valCol : str, default ''
        Column name containing the values/levels to check for retests.
        Often contains support/resistance levels or pivot points.
    
    withinAtrRange : Tuple[float, float], default (-0.15, 0.15)
        Range multiplier for ATR to define the retest zone:
        - First value: Range below current price (negative for below)
        - Second value: Range above current price (positive for above)
        
        Example: (-0.15, 0.15) creates a zone from 0.15 ATR below to 0.15 ATR
        above the current price.
    
    rollingLen : int, default 10
        Number of most recent non-NaN values to examine from valCol.
        Note: This looks back for the nth previous values, automatically
        dropping any NaN values. When using a points column, this effectively
        looks back for the nth number of points regardless of their bar position.
    
    Methods
    -------
    _compute_row(df: pd.DataFrame) -> float
        Calculates the number of points within the ATR-defined range.
        Returns the count as a float value.
    
    Examples
    --------
    >>> # Create a retest signal for detecting support retests
    >>> support_retest = Retest(
    ...     direction='down',
    ...     atrCol='ATR_14',
    ...     valCol='Support_Levels',
    ...     withinAtrRange=(-0.2, 0.1),
    ...     rollingLen=15
    ... )
    ...
    >>> # Apply to DataFrame
    >>> df['Support_Retest'] = df.apply(support_retest, axis=1)
    """
    name: str = 'Retest'
    direction: str = 'down'
    atrCol: str = ''
    valCol: str = ''
    withinAtrRange: Tuple[float, float] = (-0.15, 0.15)  # (range below, range above)
    rollingLen: int = 10
    
    def __post_init__(self):
        # Create unique identifier name by combining base name, direction, and value column
        self.name = f"{self.name}_{self.direction}_{self.valCol}"
        self.names = [self.name]

    def _compute_row(self, df: pd.DataFrame) -> float:
        # Ensure we have enough data to perform the calculation
        if len(df) < 10:
            return 0.0

        # Determine reference price based on direction (low for down retests, high for up retests)
        bar_value = df.low.iat[-1] if self.direction == 'down' else df.high.iat[-1]
        
        # Get current ATR value
        atr = df[self.atrCol].iat[-1]
        
        # Calculate min and max boundaries of the retest zone using ATR multipliers
        min_level = bar_value + (atr * self.withinAtrRange[0])  # Lower bound
        max_level = bar_value + (atr * self.withinAtrRange[1])  # Upper bound

        # Get the most recent non-NaN values from the value column
        # This automatically handles sparse data like pivot points
        vals = df[self.valCol].dropna().iloc[-self.rollingLen:]
        
        # Return 0 if no values are available
        if len(vals) < 1:
            return 0.0

        # Count how many points fall within the retest zone
        points_in_range = vals[(vals >= min_level) & (vals <= max_level)]
        return len(points_in_range)





#£ Done
@dataclass
class BarSW(Signals):
    name: str = 'BarSW'  # Bar Strength Weakness 
    atrCol: str = ''

    def __post_init__(self):
        self.name = f"{self.name}"
        self.names = [self.name]

    def _compute_row(self, df: pd.DataFrame):
        # Extract the most recent open, high, low, and close values
        open = df['open'].iloc[-1]
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        atr = df[self.atrCol].iloc[-1]

        # top = high - max(open, close)
        top = (high - max(open, close)) / 2 # give tails less weight
        body = (close - open) 
        # bot = min(open, close) - low
        bot = (min(open, close) - low) / 2 # give tails less weight

        score = (bot - top + body) / atr
        # print(f"{df.index[-1]} open: {open}, close: {close}, score: {score} ... ({top=} - {bot=} + {body=}) / ({high=} - {low=})")

        return score
 


#£ Done
@dataclass
class PullbackNear(Signals):
    name: str = 'PBN'
    fromCol: str = ''  # col which the price pullback is near to
    toCol: str = ''    # col which the points are in
    optimalRetracement: float = 99
    atrCol: str = ''   # column name for ATR values
    atrMultiple: float = 2.0  # minimum number of ATRs required between fromCol and toCol

    def __post_init__(self):
        self.name = f"Sig{self.ls[0]}_{self.name}_{self.toCol[:3]}"
        self.names = [self.name]

    def pb_score(self, retracement):
        """Calculate score based on retracement back to a value eg MA21. Optimal retracement is 95%.
        Score decreases by 10 for every 1% away from optimal retracement above 95%.
        eg 91% retracement will have a score of 85. 92% retracement will have a score of 80.
        eg 80% retracement will have a score of 80. 70% retracement will have a score of 70.
        eg 50% retracement will have a score of 50. 0% retracement will have a score of 0.
        """
        optimal_retracement = self.optimalRetracement
        if retracement <= optimal_retracement:
            return max(retracement, 0)
        
        score = optimal_retracement - (retracement - optimal_retracement) * 5
        return max(score, 0)

    def _compute_row(self, df: pd.DataFrame):
        """How near is the priceNow to the MA from the pullback high at start (bull case) to the low at the end.
        Also validates that the distance between fromCol and toCol is at least n ATRs."""
        
        if self.fromCol not in df.columns or self.toCol not in df.columns:
            return 0.0
        
        points = df[self.fromCol].dropna()
        if len(points) < 2:
            return 0.0
        
        w0 = df.loc[points.index[-1]:]
        if len(w0) < 3:
            return 0.0

        priceNow = df.close.iat[-1]
        current_atr = df[self.atrCol].iat[-1]

        if self.ls == 'LONG':
            # check high bar has cleared the toCol value
            if not w0.low.iat[-1] > df[self.toCol].iat[-1]:
                return 0.0
            # Get the high point and the corresponding toCol value
            high_point = w0.high.iat[0]
            to_col_value = df[self.toCol].iat[-1]
            
            # Check if the distance between high point and toCol is at least n ATRs
            distance = abs(high_point - to_col_value)
            if distance < (self.atrMultiple * current_atr):
                return 0.0
                
            val = trace(high_point, df[self.toCol].iat[-1], priceNow)
            return self.pb_score(val)

        elif self.ls == 'SHORT':
            # check low bar has cleared the toCol value
            if not w0.high.iat[-1] < df[self.toCol].iat[-1]:
                return 0.0
            # Get the low point and the corresponding toCol value
            low_point = w0.low.iat[0]
            to_col_value = df[self.toCol].iat[-1]
            
            # Check if the distance between low point and toCol is at least n ATRs
            distance = abs(low_point - to_col_value)
            if distance < (self.atrMultiple * current_atr):
                return 0.0
                
            val = trace(low_point, df[self.toCol].iat[-1], priceNow)
            return self.pb_score(val)

        return 0






#!!! --------->>>  Not implemented yet.  Needs to be checked  <<<-----------





    


#! Not implemented yet.  Needs to be checked
@dataclass
class HigherLowPointsLowerHighPoints:
    colname : str   
    lpCol   : str
    hpCol   : str

    def __post_init__(self):
        self.columns = [self.colname]
        self.val = 0.0

    
    def run(self, longshort: str, df: pd.DataFrame) -> float:
        """If longshort == 'LONG' then checks if the last hp > than previous hp and last lp > previous lp. Vice versa for 'SHORT'.
        if 0 are True then val = 0.0
        if 1 is  True then val = 0.5
        if 2 are True then val = 1.0        
        """
        
        hp = df[self.hpCol].dropna().values
        lp = df[self.lpCol].dropna().values
        
        if len(hp) < 2 or len(lp) < 2:
            self.val = 0.0
            return self.val

        last_hp, prev_hp = hp[-1], hp[-2]
        last_lp, prev_lp = lp[-1], lp[-2]

        if longshort == 'LONG':
            hp_condition = last_hp > prev_hp
            lp_condition = last_lp > prev_lp
            self.val = (hp_condition + lp_condition) / 2.0
            return self.val
        
        if longshort == 'SHORT':
            hp_condition = last_hp < prev_hp
            lp_condition = last_lp < prev_lp
            self.val = (hp_condition + lp_condition) / 2.0
            return self.val
        
        self.val = 0.0
        return self.val
    
#! Not implemented yet.  Needs to be checked
@dataclass
class TrendlinneRightDirection:
    colname  : str
    trendCol : str

    def __post_init__(self):
        self.val = 0.0

    def run(self, longshort:str, df: pd.DataFrame) -> float:
        """If longshort == 'LONG' then checks the if the last trend > 0. Vice versa for 'SHORT'.
        if True then val  = 1.0
        if False then val = 0.0
        """ 

        if df.empty or len(df) < 2:
            self.val = 0.0
            return self.val

        if longshort == 'LONG':
            self.val = df[self.trendCol].iloc[-1] >= -0.001 # allow for small negative values
            return self.val
        
        
        if longshort == 'SHORT':
            self.val = df[self.trendCol].iloc[-1] <= 0.001 # allow for small positive values
            return self.val
        
        self.val = 0.0 # always set as self.val may have been set in previous run
        return self.val



#£ Done
@dataclass
class ChangeOfColour(Signals):
    """
    Returns the ratio of the max consecutive colours to the total number of bars in the move.
    Excludes the last bar as it is the bar that changed colour.
    if the last bar is the same colour as the longshort direction then the then a the ratio is returned.
    """
    
    def __post_init__(self):
        self.columns = [self.colname]     
        self.df = pd.DataFrame()

    #$ **kwargs is used to allow any signal arguments to be passed to any run method. This so that the same run method can be used when looping through signals.
    def run(self, longshort:str='', fromHP:pd.DataFrame=pd.DataFrame(), fromLP:pd.DataFrame=pd.DataFrame(), **kwargs):

        self.df = fromHP if longshort == 'LONG' else fromLP
        if len(self.df) < 3 : 
            return 0.0                                
        
        # check if any reason not to proceed
        colors = list(self.df.close > self.df.open) # True if green, False if red

        """
                Example of self.df colours get converted to a list of True or False wich can be checked against 0 or 1 meaning red or green
                _____________________________________________________________________________________
                2000-01-03 16:45:00    0 (R)  group_with 3 consecutive colours          |
                2000-01-03 16:50:00    0 (R)  group_with 3 consecutive colours          |
                2000-01-03 16:55:00    0 (R)  group_with 3 consecutive colours          |
                2000-01-03 17:00:00    1 (G)                                        bars within the move to get the ratio of the max consecutive colours
                2000-01-03 17:05:00    0 (R)                                            |
                2000-01-03 17:10:00    1 (G)                                            | 
                2000-01-03 17:15:00    0 (R)  group_with 4 consecutive colours          |                
                2000-01-03 17:20:00    0 (R)  group_with 4 consecutive colours          |
                2000-01-03 17:25:00    0 (R)  group_with 4 consecutive colours          |
                2000-01-03 17:30:00    0 (R)  group_with 4 consecutive colours          |
                                            ---------------------------------------------------------
                2000-01-03 17:35:00    1 (G)  last colour is ignored

                so the ratio of the max consecutive colours is 4 / 10 = 40%
        """

        if colors[-1] == 0 and longshort == 'LONG' : return 0.0 # a pull back expects Red (0)  if LONG,  so if the coloiur is R then it is still in the pull back
        if colors[-1] == 1 and longshort == 'SHORT': return 0.0 # a pull back expects Green if SHORT, so if the coloiur is G then it is still in the pull back
        if colors[-1] == colors[-2] : return 0.0                # check if the last colour is a change from the previous colour if not then return 0
        
        # aftre checing it is a change of colour and in the right direction
        max_count = 0
        count = 1  # Start at 1 to account for the current color
        
        for i in range(1, len(colors)):
            if colors[i] == colors[i-1]:
                count += 1
            else:
                max_count = max(max_count, count)
                count = 1  # Reset count for the new color sequence
        
        # Check last sequence
        self.val =  max(max_count, count) / (len(colors) - 1) * 100



@dataclass
class RelativeStrengthWeakness(Signals):
    rsiCol : str = ''
 
    def __post_init__(self):
        self.columns = [self.colname]   

    #$ **kwargs is used to allow any signal arguments to be passed to any run method. This so that the same run method can be used when looping through signals. 
    def run(self, longshort:str='', df:pd.DataFrame=pd.DataFrame(), **kwargs):

        if longshort == 'LONG':
            self.val = df[self.rsiCol].iat[-1]

        elif longshort == 'SHORT':
            self.val = df[self.rsiCol].iat[-1] *-1

#$ ----------------------------------------------------------------------------
#$ ------------ G A P S -------------------------------------------------------
#$ ----------------------------------------------------------------------------


#$ ------ Gaps Helper Functions -----------------------------------
@dataclass
class GapBase:
    def is_gap(self, df, ls:Literal['LONG', 'SHORT']) -> bool:
        """Determine if there is a gap based on the given direction (LONG or SHORT)."""
        if ls == "LONG":
            this_lo = df.low.iat[-1]
            prev_hi = df.high.iat[-2]
            return this_lo > prev_hi
        elif ls == "SHORT":
            this_hi = df.high.iat[-1]
            prev_lo = df.low.iat[-2]
            return this_hi < prev_lo

    def get_gap_bounds_and_cancel_price(self, df, ls:Literal['LONG', 'SHORT']):
        if ls == 'LONG':
            upper_bound = df.close.iat[-1]  # Current bar close as ceiling
            lower_bound = df.close.iat[-2]  # Previous bar close as floor
            cancel_price = df.low.iat[-2]   # Previous bar high as invalidation level
        elif ls == 'SHORT':
            upper_bound = df.close.iat[-2]  # Previous bar close as ceiling
            lower_bound = df.close.iat[-1]  # Current bar close as floor
            cancel_price = df.high.iat[-2]  # Previous bar high as invalidation level
        
        return upper_bound, lower_bound, cancel_price
    
    def has_crossed_over_pivot(self, df, pointCol: str, ls:Literal['LONG', 'SHORT']) -> bool:
        """Check if the current bar has gapped over a pivot point."""
        pivots = df[pointCol].iloc[:-2].dropna()
        if len(pivots) == 0:
            return False
        
        prior_piv = pivots.iat[-1]

        if ls == 'LONG':
            return df.close.iat[-1] > prior_piv and df.high.iat[-2] < prior_piv
        elif ls == 'SHORT':
            return df.close.iat[-1] < prior_piv and df.low.iat[-2] > prior_piv

# Used for checking if this works correctly. 
@dataclass
class IsGappedOverPivot(Signals):
    """
    Simple signal class to verify gap-over-pivot conditions.
    Returns 1.0 when a pivot is gapped over, 0.0 otherwise.
    """
    name: str = 'GapPiv'
    pointCol: str = 'pivot'

    def __post_init__(self):
        self.name = f"{self.name}_{self.pointCol}"
        self.names = [self.name]    

    def _compute_row(self, df: pd.DataFrame) -> float:
        # Get data up to current bar
        return is_gap_pivot_crossover(df, self.pointCol, self.ls)

    

#£ Done
@dataclass
class GappedPivots(Signals, GapBase):
    """
    Computes the ratio or count of pivots that fall within the gap between
    previous close and current open.
    
    Parameters:
    -----------
    pointCol : str : Column name containing pivot points
    span : int: How far back to look for pivots
    ls : str 'LONG' or 'SHORT' to indicate direction
    lookBack : int Number of recent bars to analyze
    """
    name: str = 'GPivs'
    pointCol: str = ''
    spanPivots: int = 20
    ls: str = 'LONG'  # Added default value

    def __post_init__(self):
        self.name = f"{self.ls[0]}_{self.name}"
        self.names = [self.name]

    def _compute_row(self, df: pd.DataFrame) -> tuple[int, int]:
        """Count pivots that fall within the gap range."""

        if not self.is_gap(df, self.ls):
            return 0.0
        
        pivots = df[self.pointCol].iloc[:-2].dropna()
        
        if len(pivots) == 0:
            return 0
        
        # Use an integer for slicing, not a Series
        span_pivs = min(self.spanPivots, len(pivots))
        recent_pivots = pivots.iloc[-span_pivs:]

        current_open = df['open'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        
        # Only count pivots that fall within the gap range
        if self.ls == 'LONG' and current_open > prev_close:
            return sum((pivot > prev_close) & (pivot < current_open) for pivot in recent_pivots)
        elif self.ls == 'SHORT' and current_open < prev_close:
            return sum((pivot < prev_close) & (pivot > current_open) for pivot in recent_pivots)

        return 0



#£ Done
@dataclass
class GappedWRBs(Signals, GapBase):
    """
    Measures the shock value of price gaps by calculating a cumulative score based on the strength/weakness
    of prior price bars that have been "gapped over" by the current bar.
    
    This class evaluates price action after a gap (up for LONG signals, down for SHORT signals) and
    calculates a score by summing the strength/weakness values of relevant historical bars that meet
    specific criteria. The calculation gives higher scores to gaps that move beyond multiple significant
    prior bars with strong price action in the opposite direction.
    
    Attributes:
        name (str): Base name for the signal, defaults to 'GapWRBs'
        bswCol (str): Column name in the dataframe containing bar strength/weakness values
        ls (str): Signal direction, either 'LONG' or 'SHORT' (inherited from Signals parent class)
        names (list): List containing the formatted signal name
    
    Notes:
        - The class expects a pandas DataFrame with OHLC data and a bar strength/weakness column
        - The calculation only applies when there is a true gap (current low > previous high for LONG,
          current high < previous low for SHORT)
        - For performance reasons, the calculation stops evaluating prior bars when certain boundary
          conditions are met
    """
    name: str = 'GapWRBs'
    bswCol: str = ''

    def __post_init__(self):
        """
        Initialize the signal name by prepending the first letter of the signal direction ('L' or 'S')
        to the base name.
        """
        self.name = f"{self.ls[0]}_{self.name}"
        self.names = [self.name]

    def _compute_row(self, df: pd.DataFrame) -> float:
        """
        Calculates the shock value score for a price gap by analyzing prior bars.
        
        For LONG signals:
            1. Confirms a true gap exists (current bar's low > previous bar's high)
            2. Establishes boundaries for evaluation:
               - upper_bound: Current bar's close price
               - lower_bound: Previous bar's close price
               - cancel_price: Previous bar's low price
            3. Iterates backward through prior bars and adds to the score when:
               - The bar's high is below the upper_bound (hasn't exceeded current close)
               - The bar's high is above the cancel_price (hasn't gone too low)
               - The bar's high is above the lower_bound (is relevant to the gap zone)
               - The bar is a down bar (close < open), representing counter-trend strength
            4. Breaks iteration when a bar's high exceeds upper_bound or falls below cancel_price
        
        For SHORT signals:
            1. Confirms a true gap exists (current bar's high < previous bar's low)
            2. Establishes boundaries for evaluation:
               - upper_bound: Previous bar's close price
               - lower_bound: Current bar's close price
               - cancel_price: Previous bar's high price
            3. Iterates backward through prior bars and adds to the score when:
               - The bar's low is above the lower_bound (hasn't gone below current close)
               - The bar's low is below the cancel_price (hasn't gone too high)
               - The bar's low is below the upper_bound (is relevant to the gap zone)
               - The bar is an up bar (close > open), representing counter-trend strength
            4. Breaks iteration when a bar's low falls below lower_bound or exceeds cancel_price
        
        Args:
            df (pd.DataFrame): DataFrame containing OHLC data and the bar strength/weakness column.
                               The most recent bar is assumed to be at the end of the DataFrame.
        
        Returns:
            float: The cumulative score representing the shock value of the gap. Returns 0.0 if 
                  there is no valid gap or if no prior bars meet the criteria.
        """

        score_sum = 0.0

        if not self.is_gap(df, self.ls):
            return 0.0
        
        upper_bound, lower_bound, cancel_price = self.get_gap_bounds_and_cancel_price(df, self.ls)

        if self.ls == 'LONG':

            # Iterate through bars backward from the second-to-last bar
            for i in range(len(df) - 2, -1, -1):
                # Stop if bar high exceeds the upper bound (current close)
                if df.high.iat[i] > upper_bound:
                    break

                # Stop if bar high is below the cancel price (too low to be relevant)
                if df.high.iat[i] < cancel_price:
                    break

                # Skip bars where high is below lower bound (not relevant to gap zone)
                if df.high.iat[i] < lower_bound:
                    continue

                # Only count down bars (close < open) as they represent counter-trend strength
                if df.close.iat[i] < df.open.iat[i]:
                    score_sum += abs(df[self.bswCol].iat[i])

        elif self.ls == 'SHORT':
            
            # Iterate through bars backward from the second-to-last bar
            for i in range(len(df) - 2, -1, -1):
                # Stop if bar low falls below the lower bound (current close)
                if df.low.iat[i] < lower_bound:
                    break

                # Stop if bar low exceeds the cancel price (too high to be relevant)
                if df.low.iat[i] > cancel_price:
                    break

                # Skip bars where low is above upper bound (not relevant to gap zone)
                if df.low.iat[i] > upper_bound:
                    continue

                # Only count up bars (close > open) as they represent counter-trend strength
                if df.close.iat[i] > df.open.iat[i]:
                    score_sum += abs(df[self.bswCol].iat[i])
            
        return score_sum
        


#£Done
@dataclass
class GappedPastPivot(Signals, GapBase):
    """
    Assess the quality of gaps past pivot points by evaluating gap size relative to ATR.
    Uses a diminishing returns approach for oversized gaps.
    """
    name: str = 'GPP'
    atrCol: str = ''
    pointCol: str = ''
    maxAtrMultiple: int = 1  # Number of ATR past pivot before score starts diminishing

    def _compute_row(self, df: pd.DataFrame) -> float:
        """
        Calculate a score based on how far price has gapped over the most recent pivot.
        Score diminishes as price moves further from the pivot, reaching 0 at max_atr_multiple.
            
        Returns:
        --------
        float
            Score between 0 and 100, with higher scores indicating better gaps
        """
        # First check if there's a valid gap over pivot
        if not self.is_gap(df, self.ls):
            return 0.0
        
        if not self.has_crossed_over_pivot(df, self.pointCol, self.ls):
            return 0.0
            
        try:
            # Get relevant values
            current_price = df['close'].iloc[-1]
            pivot_value = df[self.pointCol].dropna().iloc[-1]  # Most recent pivot
            atr = df[self.atrCol].iloc[-1]
            
            # Calculate gap size in ATR terms
            if self.ls == 'LONG':
                gap_points = current_price - pivot_value
            else:
                gap_points = pivot_value - current_price
                
            gap_atr_ratio = gap_points / atr
            
            # Calculate score using exponential decay
            # This will create a more gradual decline in score as gap increases
            decay_factor = -3 * (gap_atr_ratio / self.maxAtrMultiple)
            score = 100 * math.exp(decay_factor)
            
            # Ensure score is between 0 and 100
            return max(0, min(100, score))
            
        except (KeyError, IndexError):
            return 0.0



@dataclass
class GapSize(Signals):
    name: str = 'GapSz'
    atrCol: str = ''

    def __post_init__(self):
        self.names = [self.name]

    def _compute_row(self, df: pd.DataFrame) -> float:
        """Calculate the size of the gap relative to ATR."""
        if len(df) < 10:
            return 0.0
        current_open = df['open'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        current_atr = df[self.atrCol].iloc[-1]
        
        if current_atr == 0:
            return 0.0
        
        # * 100 make this a % value
        # 50 would mean the gap is 50% of ATR
        # 120 would mean the gap is 120% of ATR
        gap = current_open - prev_close
        return gap / current_atr * 100  

# -----------------------------------------------------------------------
# ---- S E N T I M E N T ------------------------------------------------
# -----------------------------------------------------------------------

@dataclass
class SentimentGap(Signals):
    """Measure the sentiment of a gap based on the relationship between the gap size and price change."""
    name: str = 'SnmtGap'


    def __post_init__(self):
        self.name = f"Sig_{self.name}"
        self.names = [self.name]
    
    def _compute_row(self, df: pd.DataFrame) -> float:
        raw_val = (df['open'].iat[-1] - df['close'].iat[-2]) / df['close'].iat[-2] * 100

        sentiment = max(-100, min(100, raw_val))
        return sentiment


@dataclass
class SenitmentBar(Signals):
    """Measure the sentiment of a gap based on the relationship between the gap size and price change."""
    name: str = 'SnmtBar'

    def __post_init__(self):
        self.name = f"Sig_{self.name}"
        self.names = [self.name]
    
    def _compute_row(self, df: pd.DataFrame) -> float:
        raw_val =  (df['close'].iat[-1] - df['open'].iat[-1]) / df['open'].iat[-1] * 100

        sentiment = max(-100, min(100, raw_val))
        return sentiment


@dataclass
class SentimentVolume(Signals):
    """
    Detects a volume spike in a pandas dataframe with a 'volume' column.
    Returns the percent change between the current volume and the rolling average volume over 'volMA' periods.
    """
    name         : str = 'SnmtVol'
    volMACol     : str = ''
    atrCol       : str = ''


    def __post_init__(self):
        self.name = f"Sig_{self.name}"
        self.names = [self.name]

    def _compute_row(self, df:pd.DataFrame):
        current_volume = df['volume'].iat[-1]
        vol_ma = df[self.volMACol].iat[-1]
        current_atr = df[self.atrCol].iat[-1]
        
        # Calculate volume change percentage
        vol_change = ((current_volume - vol_ma) / vol_ma) * 100
        
        # Calculate price change relative to ATR
        price_change = df['close'].iat[-1] - df['open'].iat[-1]
        normalized_price_change = price_change / current_atr
        
        # Weight volume change by normalized price movement
        return vol_change * normalized_price_change


@dataclass
class RSI(Signals):
    """
    Calculates the Relative Strength Index (RSI) of a given column.
    """
    name: str = 'RSI'
    rsiLookBack: int = 14

    def __post_init__(self):
        self.name = f"Sig_{self.name}_{self.rsiLookBack}"
        self.names = [self.name]

    def _compute_row(self, df: pd.DataFrame) -> float:
        series = df['close'].iloc[-self.rsiLookBack:]
        delta = series.diff()
        
        # Use RMA instead of simple mean
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/self.rsiLookBack).mean().iloc[-1]
        avg_loss = loss.ewm(alpha=1/self.rsiLookBack).mean().iloc[-1]
        
        # Handle division by zero
        if avg_loss == 0:
            return 100
        if avg_gain == 0:
            return 0
            
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


@dataclass
class SentimentMAvsPrice(Signals):
    """
    Detects a volume spike in a pandas dataframe with a 'volume' column.
    Returns the percent change between the current volume and the rolling average volume over 'volMA' periods.
    """
    name         : str = 'SnmtMAP'
    maCol        : str = ''
    atrCol       : str = ''


    def __post_init__(self):
        self.name = f"Sig_{self.name}_{self.maCol}"
        self.names = [self.name]

    def _compute_row(self, df:pd.DataFrame):
        current_price = df['close'].iat[-1]
        ma_price = df[self.maCol].iat[-1]
        current_atr = df[self.atrCol].iat[-1]

        # Calculate price change percentage
        price_change = ((current_price - ma_price) / ma_price) * 100

        # Calculate price change relative to ATR
        normalized_price_change = price_change / current_atr

        return normalized_price_change




# -----------------------------------------------------------------------
# ---- V O L U M E ------------------------------------------------------
# -----------------------------------------------------------------------
#£ Done
@dataclass
class VolumeSpike(Signals):
    """
    Detects directional volume spikes normalized by ATR.
    Returns volume intensity weighted by price movement relative to ATR:
    - Positive: High volume with strong upward price movement
    - Negative: High volume with strong downward price movement
    """
    name         : str = 'VolSpike'
    volMACol     : str = ''

    def __post_init__(self):
        self.names = [self.name]

    def _compute_row(self, df: pd.DataFrame):
        current_volume = df['volume'].iat[-1]
        vol_ma = df[self.volMACol].iat[-1]
        
        # Check if vol_ma is zero to avoid division by zero
        if vol_ma == 0:
            return 0  # or return 0, or handle it as needed
        
        # Calculate the percent change between the current volume and the rolling average volume
        return ((current_volume - vol_ma) / vol_ma) * 100


@dataclass
class VolumeROC(Signals):
    """
    Calculates the Rate of Change (ROC) of volume between consecutive bars.
    Returns the acceleration of volume changes over the lookback period.
    """
    name: str = 'VolROC'

    def __post_init__(self):
        self.names = [self.name]

    
    def _compute_row(self, df: pd.DataFrame):
        """
        Calculate volume ROC acceleration over the lookback period.
        """
        # Get the volume series for the lookback period
        vol1 = df['volume'].iat[-1]
        vol2 = df['volume'].iat[-2]
        
        # Check if vol_ma is zero to avoid division by zero
        if vol2 == 0:
            return 0  # or return 0, or handle it as needed
        
        # Calculate ROC
        return ((vol1 - vol2) / vol2) * 100


@dataclass
class ROC(Signals):
    """
    Calculates the Rate of Change (ROC) of values between bars over a looks back preiod .
    """
    name: str = 'ROC'
    metricCol: str = ''
    rocLookBack: int = 10

    def __post_init__(self):
        self.name = f"Sig{self.ls[0]}_{self.name}_{self.metricCol}"
        self.names = [self.name]

    def _compute_row(self, df: pd.DataFrame):
        """
        Calculate volume ROC acceleration over the lookback period.
        """
        # Get the volume series for the lookback period
        val1 = df[self.metricCol].iat[-1]
        val2 = df[self.metricCol].iat[-self.rocLookBack]
        
        # Calculate ROC
        return ((val1 - val2) / val2) * 100


@dataclass
class PctDiff(Signals):
    """
    Calculates the percentage difference between two values.
    """
    name: str = 'PctDiff'
    metricCol1: str = ''
    metricCol2: str = ''
    
    def __post_init__(self):
        self.name = f"{self.name}_{self.metricCol2}"
        self.names = [self.name]

    def _compute_row(self, df: pd.DataFrame):
        """
        Calculate the percentage difference between two values.
        """
        val1 = df[self.metricCol1].iat[-1]
        val2 = df[self.metricCol2].iat[-1]

        if pd.isna(val1) or pd.isna(val2) or val2 == 0:
            return 0
        
        diff = val2 - val1
        
        # Calculate percentage difference
        return (diff / 100) * 100

# -----------------------------------------------------------------------
# ---- P R I C E --------------------------------------------------------
# -----------------------------------------------------------------------


@dataclass
class RoomToMove(Signals):
    """Calculate the available room for price movement relative to a target level, measured in ATR multiples.

    This signal evaluates how much space exists between the current price and a target level (e.g., pivot point
    or resistance/support level), normalized by the Average True Range (ATR). The measurement helps determine if
    there's sufficient room for a potential trade in either direction.

    Behavior:
    - For LONG positions:
        * If price > target: Returns unlimitedVal (unlimited room to move up)
        * If price < target: Returns (target - price) / ATR
    - For SHORT positions:
        * If price < target: Returns unlimitedVal (unlimited room to move down)
        * If price > target: Returns (price - target) / ATR
    - Returns 0 if target or ATR values are missing (NaN)

    Parameters:
    ----------
    name : str, default 'RTM'
        Base name for the signal
    tgetCol : str
        Column name containing target prices (e.g., pivot points, resistance/support levels)
    atrCol : str
        Column name containing ATR values
    unlimitedVal : int, default 10
        Value to return when price has moved beyond target, indicating unlimited room to move

    Example:
    --------
    LONG position scenario:
    - Current price = 100
    - Target level = 110
    - ATR = 5
    - Result = (110 - 100) / 5 = 2 ATR multiples room to move

    If price = 115 (above target):
    - Result = unlimitedVal (indicating unlimited room to move)
    """
    name: str = 'RTM'
    tgetCol: str = ''
    atrCol: str = ''
    unlimitedVal: int = 10

    def __post_init__(self):
        
        self.name = f"{self.name}_{self.ls[0]}_{self.tgetCol}"
        self.names = [self.name]

    def _compute_row(self, df: pd.DataFrame = pd.DataFrame(), **kwargs):
        if len(df) <= 1:
            return 0

        close_price = df.close.iat[-1]
        target_price = df[self.tgetCol].iat[-1]
        atr = df[self.atrCol].iat[-1]

        # Check if we have valid values
        if not pd.notna(atr):
            return 0
        
        if not pd.notna(target_price):
            return self.unlimitedVal       

        if self.ls == 'LONG':
            # If price is already above target, unlimited room to move
            if close_price > target_price:
                return self.unlimitedVal
            # Calculate room to move in ATR multiples
            return (target_price - close_price) / atr

        elif self.ls == 'SHORT':
            # If price is already below target, unlimited room to move
            if close_price < target_price:
                return self.unlimitedVal
            # Calculate room to move in ATR multiples
            return (close_price - target_price) / atr

        return 0
    



@dataclass
class RoomToMoveCustomValues(Signals):
    """This signal calculates the room to move based on the current price and the last pivot point.
    The room to move is calculated as the distance between the current price and the last pivot point that is higher than the current price (if LONG).
    The signal returns a value for the number atr muliples the current price is away from the last pivot point.
    Account the points column having nan values.

    Example:
    If the current price is 2 atr multiples away from the last pivot point, the signal will return 2.
    price  = 100
    pivot  = 110
    atr    = 5
    room   = 2
    """

    val   : float = 0.0

    def run_long(self, tget:float, priceNow:float, atr:float):
        if tget is not None:
            self.val = (tget - priceNow) / atr
        else:
            self.val = 2

    def run_short(self, tget:float, priceNow:float, atr:float):
        if tget is not None:
            self.val = (priceNow - tget) / atr
        else:
            self.val = 2


@dataclass
class PriceProximity(Signals):
    """ Price Proximity class to calculate the score based on the price proximity to the nearest increment.
    # if long then use the low of the body as to check if the price is giving support to the body
    Args: 
        scoreTable (dict): A dictionary containing the score table.
        
    Returns:
        value:  The score based on the price proximity to the nearest increment.
    """
    scoreTable = {   
            10   : {1: 100, 10: 100, 100: 100, 1000: 100},
            5    : {1: 100, 10: 100, 100: 100, 1000: 100},
            1    : {1: 100, 10: 100, 100: 100, 1000: 75 },
            0.5  : {1: 100, 10: 75,  100: 75,  1000: 50 },
            0.1  : {1: 100, 10: 50,  100: 50,  1000: 10 },
            0.05 : {1: 75,  10: 25,  100: 10,  1000: 5  },
        }
    
    def __post_init__(self):
        self.scorekey = list(self.scoreTable.keys())
        self.prices = np.array(list(next(iter(self.scoreTable.values())).keys()))
        self.increment_lookup = {decimal: inc for inc in self.scorekey for decimal in range(int(inc * 100), 100, max(1, int(10 * inc)))}
        self.price_lookup = {price: self.prices[np.argmin(np.abs(self.prices - price))] for price in range(1, 10001)}
        self.df = pd.DataFrame()

    def get_score_table_as_df(self):
        # label rows as inceremnts and column as rounded prices
        df = pd.DataFrame(self.scoreTable).T
        df.index.name = 'Increment'
        df.columns.name = 'Price'
        return df
    
    def get_scorekey(self, price):
        for inc in self.scorekey:
            val  = round(price / inc, 4)
            if val % 1 == 0:
                return inc
        return 0

    def run(self, longshort:str='', df:pd.DataFrame=pd.DataFrame(), **kwargs):
        if len(df) < 2:
            self.val = 0
            return self.val
        self.df = df.copy()
        
        def get_price():
            if longshort == 'LONG':
                return min(self.df['close'].iat[-1], self.df['open'].iat[-1])
                print(f'{longshort=}, {price=}')    
            elif longshort == 'SHORT':
                return max(self.df['close'].iat[-1], self.df['open'].iat[-1])
            return 0
        
        if longshort == 'LONG':
            if not self.df.high.iat[-2] > self.df.high.iat[-1]: # check if lower high
                self.val = 0
                return self.val
                
        elif longshort == 'SHORT':
            if self.df.low.iat[-2] < self.df.low.iat[-1]:
                self.val = 0
                return self.val

        price = get_price()
        inc = self.get_scorekey(price)
        if inc != 0:
            rounded_price = self.price_lookup.get(int(price), 1000)
            self.val      = self.scoreTable.get(inc, {1000: 0})[rounded_price]
            # print(f'{price=}, {inc=}, {rounded_price=}, {self.val=}')
            return self.val
                
        self.val = 0
        return self.val
    
@dataclass
class BarRange(Signals):
    """ returns a value between 0 and 100 based on the ratio of the bar range to the ATR.
        The ratio is limited to the range of normBest to normWorst before being normalized to the range [0, 100].
    
    Args:
        colname   : str = 'nrb'  - the column name to use for the signal
        atrCol    : str = 'ATR'  - the column name of the ATR or ABR to use for the average range
        barsAgo   : int  = None  - the number of bars ago to calculate the signal for.  None means the current bar.
        normBest : float = 0.5   - the best ratio value to normalize to
        normWorst: float = 1.0   - the worst ratio value to normalize to
        body     : bool  = False - if True then the body range is used (eg open to close) otherwise the high to low range is used.
        wideOrNarrow : str = 'narrow' - 
                    if 'narrow' then the ratio is limited to the range of normBest to normWorst before being normalized to the range [0, 100].
                    if 'wide' then the ratio is limited to the range of normWorst to normBest before being normalized to the range [0, 100].
    """

    colname   : str = 'nrb'
    atrCol    : str = 'ATR'
    barsAgo   : int  = None
    normBest : float = 0.5 # 
    normWorst: float = 1.0 # 
    body     : bool  = False
    wideOrNarrow : str = 'narrow'

    def __post_init__(self):
        self.columns = [self.colname] 
        if self.body:
            if 'ABR' not in self.atrCol:
                raise ValueError('When body is True the atrCol must be the ABR column name.  eg "ABR20"')
        else:
            if 'ATR' not in self.atrCol:
                raise ValueError('When body is False the atrCol must be the ATR column name.  eg "ATR20"')

    #$ **kwargs is used to allow any signal arguments to be passed to any run method. This so that the same run method can be used when looping through signals.

    def run(self, df:pd.DataFrame, longshort:str, fromHP:pd.DataFrame=pd.DataFrame(), fromLP:pd.DataFrame=pd.DataFrame(), **kwargs): 


        if self.wideOrNarrow == 'narrow':
            # if the last move is too short then return 0
            if longshort == 'LONG':
                if len(fromHP) - self.barsAgo < 1:
                    self.val = 0
                    return self.val
                
                high         = fromHP.high.iat[-self.barsAgo-1]   # the high of the last bar (barsago)
                other_highs  = fromHP.high.iloc[:-self.barsAgo-2] # the slice up to the last bar (barsago)

                if high > other_highs.min():
                    self.val = 0
                    return self.val


            elif longshort == 'SHORT':
                if len(fromLP) - self.barsAgo < 1:
                    self.val = 0
                    return self.val

                low        = fromLP.low.iat[-self.barsAgo-1]  # the low of the last bar (barsago)
                other_lows = fromLP.low.iloc[:self.barsAgo-2] # the slice up to the last bar (barsago)

                if low < other_lows.min():
                    self.val = 0
                    return self.val
            
        

        idx = self.barsAgo-1 if self.barsAgo is not None else -1    
            
        atr = df[self.atrCol].iloc[idx]
 
        
        if atr == 0:
            self.val
            return self.val
        
        barRange = df['high'].iloc[idx] - df['low'].iloc[idx] if not self.body else abs(df['open'].iloc[idx] - df['close'].iloc[idx])
        ratio    = barRange / atr

        if self.wideOrNarrow == 'narrow':
            # Ensure the ratio is within the range of normBest to normWorst
            ratioLimited = max(self.normBest, min(ratio, self.normWorst))

            # Normalize the ratio to the range [0, 100]
            self.val = np.interp(ratioLimited, [self.normBest, self.normWorst], [100, 0])


        if self.wideOrNarrow == 'wide':
            # Ensure the ratio is within the range of normBest to normWorst
            ratioLimited = max(self.normWorst, min(ratio, self.normBest))

            # Normalize the ratio to the range [0, 100]
            self.val = np.interp(ratioLimited, [self.normWorst, self.normBest], [0, 100])

        return self.val

    def reset(self):
        self.val = 0
          
#$ ------- Volitility ---------------
@dataclass
class Acceleration(Signals):
    """ Acceleration class to calculate the score based on the acceleration of the price movement.
    The acceleration is calculated as the difference between the current price and the previous price divided by the previous price.
    The signal returns a value for the acceleration of the price movement.
    """
    colname : str = 'acc'
    val     : float = 0.0
    accCol  : str = 'acc'

    def run(self, longshort:str='', df:pd.DataFrame=pd.DataFrame(), **kwargs):
        self.val = df[self.accCol].iat[-1]
        return self.val


# --------------------------------------------------------------------
# ---- R E V E R S A L   S i g n a l s -------------------------------
# --------------------------------------------------------------------

@dataclass
class ReversalIntoResistance(Signals):
    """ Reversal Into Resistance assumes that the price movement into resistance is a sign of a reversal.
    """
    pass

@dataclass
class ReversalOverExtended(Signals):
    """ Reversal Over Extended assumes that 8 consecutive green bars all above 21EMA is over extended and is likely to reverse.
    counts consecutive bars above the 21EMA
    """
    pass

@dataclass
class ReversalFromMTop(Signals):
    """ Reversal From M Top assumes that if an M top is formed then the price is likely to reverse.
    """
    pass

@dataclass
class ReversalFromWBottom(Signals):
    """ Reversal From W Bottom assumes that if a W bottom is formed then the price is likely to reverse.
    """
    pass

@dataclass
class ReversalHugeGap(Signals):
    """ Reversal Huge Gap assumes that a huge gap is a sign of a reversal.
    """
    pass

@dataclass
class ReversalParabolic(Signals):
    """ Reversal Parabolic assumes that a parabolic move is a sign of a reversal.
    """
    pass


#$ 1) Trend - Must be in a stage 2 uptrend or coming from a double bottom retest/transition. (W pattern)
#$ 2) HLLH - 2 or more lower highs (LH)
#$ 3) Overlap - ‘Sequential’ pullback with less than a 50% overlap on any bar. (A 45° (degree) angle of retracement is ideal. Don’t want it ‘too’ steep.)
#$ Volume: Ending (EV), Igniting (IV), Resting (RV)
#$ Support Areas: ‘Level 1 & 2’ (1S), (1R), (2S), (2R)       
#$ Multiple Timeframes in Alignment (MTFA)
#$ Market Timing (MT)

# --------------------------------------------------------------------
# ---- M U L T I   S i g n a l s -------------------------------------
# --------------------------------------------------------------------
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Tuple, Union
import pandas as pd
import numpy as np

@dataclass
class MultiSignals(ABC):
    """
    Optimized base class for indicators that generate multiple related signals.
    Processes data in bulk rather than row by row for better performance.
    """
    name: str = ''
    normRange: Tuple[int, int] = (0, 100)
    ls: str = 'LONG'
    lookBack: int = 1
    columnStartsWith: str = ''
    chartArgs: ChartArgs = None
    invertScoreIfShort: bool = False

    def __post_init__(self):
        """Initialize with empty signal names - will be populated during setup."""
        self.names = []
        self.source_columns = []
        self.column_mapping = {}

    def add_chart_args(self, chartArgs: ChartArgs):
        self.chartArgs = chartArgs
        return self

    def setup_columns(self, df: pd.DataFrame):
        """Set up column mappings and signal names."""
        self.source_columns = [col for col in df.columns if col.startswith(self.columnStartsWith)]
        self.names = []
        self.column_mapping = {}
        
        for col in self.source_columns:
            signal_name = f"{self.name}_{col}"
            self.names.append(signal_name)
            self.column_mapping[col] = signal_name

    # def get_score(self, val):
    #     """Normalize values efficiently using vectorized operations."""
    #     def normalize_vec(x):
    #             if x is None:
    #                 return np.nan  # or another suitable default value, like np.nan
    #             normalized = (x - self.normRange[0]) / (self.normRange[1] - self.normRange[0]) * 100
    #             clamped = np.clip(normalized, 0, 100)  # Clamp the values between 0 and 100
    #             return np.round(clamped, 2)  # Added rounding to match single-value function
        
    #     # if norm range is ste to None, return the value as is
    #     if self.normRange is None:
    #         return val


    #     if isinstance(val, (pd.Series, pd.DataFrame)):
    #         return val.apply(normalize_vec)
    #     return normalize_vec(val)
    
    def get_score(self, val):
        if isinstance(val, pd.Series):
            # Apply the function to each element in the series
            # return val.apply(lambda x: 0 if x == 0 else normalize(x, self.normRange[0], self.normRange[1]))
            return val.apply(lambda x: 0 if x == 0 else (np.nan if pd.isna(x) else normalize(x, self.normRange[0], self.normRange[1])))
        else:
            # Handle single value
            if pd.isna(val):
                return np.nan
            if val == 0:
                return 0
            return normalize(val, self.normRange[0], self.normRange[1])

    @abstractmethod
    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Child classes implement this to compute all signals for the given window.
        Should return DataFrame with columns for each signal.
        """
        pass

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all signals using efficient bulk processing.
        """
        if not self.names:
            self.setup_columns(df)

        if len(df) < 10:
            return pd.DataFrame(np.nan, index=[df.index[-1]], columns=self.names)
        
        lookBack = min(self.lookBack, len(df))

        # Get the window we need to process
        window = df.iloc[-lookBack:]
        
        # Compute signals for the entire window at once
        signals = self.compute_signals(window)
        
        # Ensure we have all expected columns
        for name in self.names:
            if name not in signals.columns:
                signals[name] = np.nan


        # Normalize the results
        for col in signals.columns:
            s1 = self.get_score(signals[col])
            #  if the invertScoreIfShort is True then invert the score if also SHORT.  Allows for example Relative market weakness to be turned int strength if shorting
            signals[col] = s1*-1 if self.invertScoreIfShort and self.ls=='SHORT' else s1


        # # Normalize the results
        # for col in signals.columns:
        #     signals[col] = self.get_score(signals[col])

        return signals
#$ -------  Cosnsolidation and Trend Signals ---------------

@dataclass
class Score(MultiSignals):
    name: str = ''
    ls: str = ''
    sigs: List[Signals] = field(default_factory=list)
    cols : Union[str, List[str]] = field(default_factory=list)   
    scoreType: Literal["mean", "sum", "min", "max", "any", "all"] = ''
    operator: Literal[">", "<", ">=", "<=", "=="] = ''
    threshold: Union[float, str] = 0 # Can be a value or column name

    def __post_init__(self):
        # Ensure the name is unique
        self.name = f"{self.name}_{self.scoreType}"
        self.name_passed = f"{self.name}_{self.operator}_{self.threshold}"
        self.names = [self.name, self.name_passed]
        self._filtered_cols = None
        
        # Convert columns to list if it's a string
        if isinstance(self.cols, str):
            self.columns = [self.cols]
        
        # Ensure threshold is a string if it's not a float
        if not isinstance(self.threshold, (float, int)):
            self.threshold = str(self.threshold)
        
        # Initialize the score value
        self.val = 0.0

    def _get_filtered_columns(self, df: pd.DataFrame) -> List[str]:
        """Get and cache filtered columns to avoid recomputation."""
        if self._filtered_cols is None:
            if self.cols and self.sigs:
                raise ValueError("Score::Cannot provide both sigs and cols")
            
            # Use signal names if provided, otherwise use column names
            if self.sigs:
                self._filtered_cols = [sig.name for sig in self.sigs]
            else:
                self._filtered_cols = self.cols if self.cols else list(df.columns)
            
        return self._filtered_cols

    def _compute_row(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute the score based on the last row of the dataframe.
        
        Args:
            df: The pandas DataFrame to compute the score from
                (already pre-sliced to include the relevant rows)
        
        Returns:
            A pandas Series with the computed value and pass/fail status
        """
        # Always use the last row of the dataframe
        last_row = df.iloc[-1]
        
        # Get the values from the column(s)
        values = last_row[self._get_filtered_columns(df)]

        # Apply the score type
        if self.scoreType == "sum":
            computed_value = np.sum(values)
        elif self.scoreType == "mean":
            computed_value = np.mean(values)
        elif self.scoreType == "min":
            computed_value = np.min(values)
        elif self.scoreType == "max":
            computed_value = np.max(values)
        elif self.scoreType == "any":
            computed_value = np.any(values)
        elif self.scoreType == "all":
            computed_value = np.all(values)

        # apply score normalization
        if self.normRange:
            computed_value = self.get_score(computed_value)

        # Invert the score if short
        if self.invertScoreIfShort and self.ls == 'SHORT':
            computed_value *= -1
        
        # Get the threshold value
        if isinstance(self.threshold, str):
            threshold_value = last_row[self.threshold]
        else:
            threshold_value = self.threshold
        
        # Apply the comparison operator
        if self.operator == ">":
            passed = computed_value > threshold_value
        elif self.operator == "<":
            passed = computed_value < threshold_value
        elif self.operator == ">=":
            passed = computed_value >= threshold_value
        elif self.operator == "<=":
            passed = computed_value <= threshold_value
        elif self.operator == "==":
            passed = computed_value == threshold_value
        
        # Create a Series with the computed value and pass status
        # The second column includes the operator and threshold in the name
        return pd.Series({ self.name: round(computed_value,2), self.name_passed: passed})
        

    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        results = pd.DataFrame(index=df.index, columns=self.names)
        for i in range(len(df)):
            results.iloc[i] = self._compute_row(df.iloc[:i+1])
        return results

@dataclass
class CountTouches(MultiSignals):
    """
    Optimized version of CountTouches that processes data more efficiently
    and properly handles support/resistance touches based on specific rules.

    Touch Detection Rules:
    ---------------------
    For Support Lines:
    1. Distance Rule:
        - The low of the bar must be within [tolerance * ATR] distance of the support line
        - The high of the bar should not be below [tolerance * ATR] distance of the support line
        - Distance is measured in absolute terms using abs(price - line)
    
    2. Bar Position Rule:
        Either:
        - The line must be within the bar's range (high > line > low)
        OR
        - The bar must be very close to the line (within tolerance/2 * ATR)
    
    3. Price Distribution Rule:
        - After N consecutive bars below the line, we need at least N bars
          above the line before counting new touches
        - This ensures proper recovery after a support break
        - A bar is considered "below" if its high is below the line
        - A bar is considered "above" if its low is above the line

    For Resistance Lines:
    1. Distance Rule:
        - The high of the bar must be within [tolerance * ATR] distance of the resistance line
        - The low of the bar should not be above [tolerance * ATR] distance of the resistance line
        - Distance is measured in absolute terms using abs(price - line)
    
    2. Bar Position Rule:
        Either:
        - The line must be within the bar's range (high > line > low)
        OR
        - The bar must be very close to the line (within tolerance/2 * ATR)
    
    3. Price Distribution Rule:
        - Same as support but inverted
        - After N consecutive bars above the line, we need at least N bars
          below the line before counting new touches
        - This ensures proper recovery after a resistance break

    Parameters:
    ----------
    name : str
        Name prefix for the generated signals
    columnStartsWith : str
        Prefix for columns to process
    touchTolerance : float
        Multiplier for ATR to determine acceptable distance from line
        e.g., if touchTolerance = 2 and ATR = 0.5, touches within 1.0 units are valid
    atrCol : str
        Name of the ATR column in the dataframe
    ls : str
        Signal type identifier
    supOrRes : Literal['sup', 'res']
        Whether to look for support ('sup') or resistance ('res') touches

    Implementation Details:
    --------------------
    - Uses a 3-bar window for analysis (setup bar, touch bar, follow-through bar)
    - ATR tolerance is dynamically calculated based on current ATR value
    - Maintains running count of touches for each line
    - Processes data incrementally to support streaming/real-time updates
    """
    name: str = 'CTouch'
    columnStartsWith: str = ''
    touchTolerance: float = 0.0
    atrCol: str = 'ATR'
    ls: str = 'LONG'
    supOrRes: Literal['sup', 'res'] = 'sup'

    def _check_price_distribution(self, df_slice: pd.DataFrame, line_vals: pd.Series) -> bool:
        """
        Check price distribution with the rule that after N bars below the line,
        we need at least N bars above the line before counting new touches.
        """
        highs = df_slice['high'].values
        lows = df_slice['low'].values
        line = line_vals.values
        
        # Count consecutive bars below the line leading up to this point
        bars_below = 0
        i = len(highs) - 1
        while i >= 0 and highs[i] < line[i]:
            bars_below += 1
            i -= 1
            
        if bars_below == 0:  # If no bars below, distribution is valid
            return True
            
        # Now count how many bars are above the line since the last bar below
        bars_above = 0
        while i >= 0 and lows[i] > line[i]:
            bars_above += 1
            i -= 1
            
        # Need at least as many bars above as we had below
        return bars_above >= bars_below

    def _is_valid_touch_sequence(self, df_slice: pd.DataFrame, line_vals: pd.Series, tolerance: float) -> bool:
        """
        Check if we have a valid touch sequence based on the rules:
        For support:
        1. Low within tolerance distance of the line
        2. High not below the line
        3. Line is within the bar or very close
        4. Price distribution condition met
        """
        if line_vals.isna().any():
            return False

        # Get relevant price data for current bar
        high = df_slice['high'].iloc[1]  # Touch bar
        low = df_slice['low'].iloc[1]
        line = line_vals.iloc[1]
        
        distance_to_line = abs(low - line)
        
        if self.supOrRes == 'sup':
            # Check if the low is within tolerance distance of the line
            low_within_tolerance = distance_to_line <= tolerance
            
            # High should not be significantly below the line
            high_not_below = high >= (line - tolerance)
            
            # Either the line is within the bar or very close
            line_within_range = (high > line and low < line) or distance_to_line <= tolerance/2
            
            # Check price distribution
            valid_distribution = self._check_price_distribution(df_slice, line_vals)
            
            return low_within_tolerance and high_not_below and line_within_range and valid_distribution
            
        else:  # resistance
            # For resistance, we check the high against the line
            high_within_tolerance = abs(high - line) <= tolerance
            
            # Low should not be significantly above the line
            low_not_above = low <= (line + tolerance)
            
            # Either the line is within the bar or very close
            line_within_range = (high > line and low < line) or abs(high - line) <= tolerance/2
            
            # Check price distribution
            valid_distribution = self._check_price_distribution(df_slice, line_vals)
            
            return high_within_tolerance and low_not_above and line_within_range and valid_distribution

    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cumulative touch counts for all lines efficiently.
        Returns DataFrame with running counts for each line.
        """
        # Get ATR-based tolerance
        current_atr = df[self.atrCol].iloc[-1]
        if pd.isna(current_atr):
            current_atr = df[self.atrCol].mean()
        tolerance = current_atr * self.touchTolerance

        # Initialize results DataFrame
        results = pd.DataFrame(index=df.index, columns=self.names)
        
        # Get the count from previous data if available
        prev_counts = {col: 0 for col in self.names}
        if hasattr(self, '_prev_touch_counts'):
            prev_counts = self._prev_touch_counts
        
        # Process each trend line
        for src_col in self.source_columns:
            signal_name = self.column_mapping[src_col]
            touch_count = prev_counts[signal_name]
            
            # Need enough bars to check for distribution pattern
            if len(df) >= 3:
                # Look for touches in rolling windows
                for i in range(1, len(df) - 1):
                    df_slice = df.iloc[i-1:i+2]  # Get 3 bars for analysis
                    line_vals = df[src_col].iloc[i-1:i+2]
                    
                    # Check if trend line exists (not NaN) in current window
                    if line_vals.isna().any():
                        touch_count = 0  # Reset count when trend line disappears
                        results.loc[df_slice.index[1]:, signal_name] = 0
                        continue
                    
                    if self._is_valid_touch_sequence(df_slice, line_vals, tolerance):
                        touch_count += 1
                        # Set the count from the touch bar onwards
                        results.loc[df_slice.index[1]:, signal_name] = touch_count
            
            # Store final count for next run
            prev_counts[signal_name] = touch_count
            
        # Store counts for next computation
        self._prev_touch_counts = prev_counts
        
        return results
    

@dataclass
class LineLengths(MultiSignals):
    """
    LineLength class to calculate scores based on trend line lengths.
    Longer lines are more significant and thus score higher.
    Length is measured as the number of bars the line has existed.
    """
    name: str = 'LLen'
    columnStartsWith: str = ''
    minBars: int = 10  # Minimum bars for a valid line
    
    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute length-based scores for all trend lines in the window.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the lookback window of data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with raw (unnormalized) lengths for each line
        """
        # Initialize results with the same index as input window
        results = pd.DataFrame(index=df.index, columns=self.names)
        
        # Process each trend line
        for src_col in self.source_columns:
            signal_name = self.column_mapping[src_col]
            
            # For each point in the window, calculate the length up to that point
            lengths = []
            for i in range(len(df)):
                # Get data up to current point
                current_slice = df[src_col].iloc[:i+1]
                
                # Count backwards from current point until we hit NaN
                length = 0
                for val in reversed(current_slice):
                    if pd.isna(val):
                        break
                    length += 1
                    
                # Only count if we meet minimum length requirement
                lengths.append(length if length >= self.minBars else 0)
            
            # Assign the lengths to results
            results[signal_name] = lengths
            
        return results


@dataclass
class ConsolidationShape(MultiSignals):
    """
    A signal class that analyzes the shape characteristics of price consolidation periods.
    It calculates a ratio between the width (time duration) and height (price range) of 
    consolidation areas, normalized by ATR. Higher scores indicate "tighter" consolidations,
    which are periods where price stays within a narrow range over a longer time period.

    Key Concepts:
    -------------
    - Width: Number of bars in the consolidation period
    - Height: Difference between upper and lower consolidation boundaries
    - ATR Normalization: Height is normalized by ATR to make comparisons meaningful
      across different price ranges and volatility regimes
    - Score = width / (height/ATR): Higher scores indicate tighter consolidations

    Implementation Details:
    ----------------------
    1. Column Setup:
       - Overrides the base MultiSignals setup_columns method because consolidations
         require paired columns (UPPER/LOWER) rather than single columns
       - Uses the base class's column discovery mechanism to find UPPER columns,
         then matches them with corresponding LOWER columns
       - Creates simplified numerical signal names (e.g., ConsShape_1) instead of
         using full column names to maintain clarity

    2. Signal Computation:
       - Processes each consolidation pair separately
       - Only calculates scores during valid consolidation periods (when both
         UPPER and LOWER values exist)
       - Returns NaN for periods outside consolidations
       - Requires a minimum number of bars (minBars) to consider a consolidation valid
       - Normalizes the height by ATR to make scores comparable across different
         price ranges and volatility conditions

    Usage Notes:
    ------------
    - Higher scores indicate "better" consolidations (longer duration relative to height)
    - NaN values in output indicate periods where no valid consolidation exists
    - The score is dynamic and will change as the consolidation develops
    - Multiple consolidation areas can be tracked simultaneously (e.g., ConsShape_1, ConsShape_2)

    Parameters:
    -----------
    name : str, default 'ConsShape'
        Base name for the signal outputs
    consUpperCol : str, default ''
        Column name prefix for upper consolidation boundaries
    consLowerCol : str, default ''
        Column name prefix for lower consolidation boundaries
    atrCol : str, default 'ATR'
        Name of the ATR column used for normalization
    minBars : int, default 5
        Minimum number of bars required for a valid consolidation

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns named {name}_1, {name}_2, etc., containing
        shape scores for each consolidation period. Values are NaN outside
        of valid consolidation periods.
    """
    name: str = 'ConsShape'
    consUpperCol: str = ''
    consLowerCol: str = ''
    atrCol: str = 'ATR'  # Default ATR column name
    minBars: int = 5     # Minimum bars needed for valid consolidation

    def __post_init__(self):
        """Initialize with consolidation column pairs."""
        super().__post_init__()
        # Extract the base names without the suffix
        if self.consUpperCol.endswith('_1'):
            self.base_name = self.consUpperCol[:-2]
        elif self.consUpperCol.endswith('_2'):
            self.base_name = self.consUpperCol[:-2]
        else:
            self.base_name = self.consUpperCol
            
    def __post_init__(self):
        """Initialize pairs of consolidation columns."""
        super().__post_init__()
        self.columnStartsWith = 'CONS_UPPER'  # This helps the base class find relevant columns
        self.column_pairs = []  # We'll populate this in setup_columns
        
    def setup_columns(self, df: pd.DataFrame):
        """Set up column mappings for consolidation pairs."""
        # Use base class to find upper columns
        super().setup_columns(df)
        
        # Reset our specific attributes
        self.names = []
        self.column_pairs = []
        
        # For each upper column found by base class
        for upper_col in self.source_columns:
            # Find corresponding lower column
            lower_col = upper_col.replace('UPPER', 'LOWER')
            if lower_col in df.columns:
                # Create numbered signal names instead of using column names
                signal_name = f"{self.name}_{len(self.names) + 1}"
                self.names.append(signal_name)
                self.column_pairs.append((upper_col, lower_col))

    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute shape-based scores for all consolidation areas in the window.
        A tighter consolidation (smaller height/width ratio) scores higher.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the lookback window of data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with raw (unnormalized) shape scores for each consolidation
        """
        # Initialize results DataFrame with NaN
        results = pd.DataFrame(index=df.index, columns=self.names)
        
        # Process each consolidation pair
        for (upper_col, lower_col), signal_name in zip(self.column_pairs, self.names):
            scores = []
            
            # For each point in the window
            for i in range(len(df)):
                current_row = df.iloc[i]
                
                # If current row has no consolidation, append NaN
                if pd.isna(current_row[upper_col]) or pd.isna(current_row[lower_col]):
                    scores.append(np.nan)
                    continue
                
                # Find the current consolidation window
                current_window = df.iloc[:i+1]
                mask = current_window[[upper_col, lower_col]].notna().all(axis=1)
                cons_window = current_window[mask]
                
                # If we don't have enough bars in this consolidation, append NaN
                if len(cons_window) < self.minBars:
                    scores.append(np.nan)
                    continue
                
                # Get the last complete consolidation window
                last_nan_idx = cons_window.index[-1]
                window_start = cons_window.index[0]
                cons_window = df.loc[window_start:last_nan_idx]
                
                # Calculate consolidation metrics for current window only
                height = cons_window[upper_col].iloc[-1] - cons_window[lower_col].iloc[-1]
                width = len(cons_window)  # Number of bars in current consolidation
                
                # Get ATR for scale normalization
                atr = cons_window[self.atrCol].iloc[-1] if self.atrCol in cons_window else 1
                
                # Calculate shape score using width / (height/atr)
                if width > 0 and height > 0:
                    shape_score = width / (height / atr)
                else:
                    shape_score = np.nan
                    
                scores.append(shape_score)
            
            # Assign scores to results
            results[signal_name] = scores
            
        return results


@dataclass
class ConsolidationPosition(MultiSignals):
    """
    ConsolidationPosition class to calculate the score based on the consolidation position.
    The score is based on how far the consolidation zone is positioned within the overall
    price range of the stock. Consolidations near all-time highs or all-time lows score 
    highest (100), while consolidations in the middle of the overall range score lowest (0).
    When scoring, we consider:
    - For consolidations above the middle: distance of upper boundary to overall high
    - For consolidations below the middle: distance of lower boundary to overall low
    """
    name: str = 'ConsPos'
    consUpperCol: str = ''  # Column name prefix for upper consolidation boundaries
    consLowerCol: str = ''  # Column name prefix for lower consolidation boundaries
    priceCol: str = 'close'  # Column to use for overall price range calculation
    
    def __post_init__(self):
        """Initialize pairs of consolidation columns."""
        super().__post_init__()
        self.columnStartsWith = 'CONS_UPPER'
        self.column_pairs = []
        # Set normalization range for scores to be between 0 and 1
        self.normRange = (0, 1)
        
    def setup_columns(self, df: pd.DataFrame):
        """Set up column mappings for consolidation pairs."""
        super().setup_columns(df)
        self.names = []
        self.column_pairs = []
        
        for upper_col in self.source_columns:
            lower_col = upper_col.replace('UPPER', 'LOWER')
            if lower_col in df.columns:
                signal_name = f"{self.name}_{len(self.names) + 1}"
                self.names.append(signal_name)
                self.column_pairs.append((upper_col, lower_col))
                
    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute position-based scores for consolidation areas relative to the overall price range.
        Raw scores are between 0 and 1, which will be normalized to 0-100 by the base class.
        """
        results = pd.DataFrame(index=df.index, columns=self.names)
        
        # Calculate overall price range for the entire dataset
        overall_high = df[self.priceCol].max()
        overall_low = df[self.priceCol].min()
        overall_range = overall_high - overall_low
        range_midpoint = overall_low + (overall_range / 2)
        
        for (upper_col, lower_col), signal_name in zip(self.column_pairs, self.names):
            upper_first_idx = df[upper_col].first_valid_index()
            lower_first_idx = df[lower_col].first_valid_index()
            
            if upper_first_idx is not None and lower_first_idx is not None:
                cons_upper = df.loc[upper_first_idx, upper_col]
                cons_lower = df.loc[lower_first_idx, lower_col]
                cons_mid = (cons_upper + cons_lower) / 2
                
                # Determine if consolidation is in upper or lower half
                in_upper_half = cons_mid > range_midpoint
                
                if in_upper_half:
                    # For upper half, normalize distance from midpoint to high
                    distance = max(0, min(cons_upper - range_midpoint, overall_high - range_midpoint))
                    denominator = overall_high - range_midpoint
                    if denominator != 0:
                        position_score = distance / denominator
                    else:
                        position_score = 0  # or some other appropriate value or handling
                else:
                    # For lower half, normalize distance from low to midpoint
                    distance = max(0, min(range_midpoint - cons_lower, range_midpoint - overall_low))
                    denominator = range_midpoint - overall_low
                    if denominator != 0:
                        position_score = distance / denominator
                    else:
                        position_score = 0  # or some other appropriate value or handling
                
                # Create mask for valid consolidation periods
                mask = (~pd.isna(df[upper_col])) & (~pd.isna(df[lower_col]))
                
                # Apply score only where consolidation exists, now outputting 0-1 range
                results[signal_name] = np.where(mask, position_score, np.nan)
            else:
                results[signal_name] = np.nan
            
        return results


@dataclass
class ConsolidationPreMove(MultiSignals):
    """
    ConsolidationPreMove class to calculate the score based on the price move before entering into a consolidation period.
    The pre move values are based on the MoveCol (eg 50MA) and the ATR.
    1. get the last pivot of the MA before entering the consolidation zone.
    2. detremine if the pivoit is above the upper bound or below the lower bound. if not then pass
    3. if the pivot is above then calculate the move height from the MA pivot down to the upper bound. opposite if the MA pivot is below. the height vales is in ATR multiples.
    4. calculate the move duration in bars. the duration is from the MA pivot to the point where it crosses the upper or lower bound price level even if it cross the 
    price level before or after the zone it is still valid.  it is the price level cross over point that is important. 
    5. calculate the score based on the height, duration, and steepness of the move. both steepness and druation add to increasing the score.

    Notes: 
    The consUpperCol and consLowerCol are the bounds for each consolidation zone in the dataframe.
    The values in the coluns will be the same for the entire consolidation period. the other values outside of the consilidation period will be NaN. 
    The output will be a score column for each consolidation periods (pair of upper and lower bounds).
    The output filled values wil match the same rows as the consolidation rows from the consUpperCol and consLowerCol columns.
    """
    name: str = 'ConsPreMove'
    consUpperCol: str = ''
    consLowerCol: str = ''
    maCol: str = ''
    atrCol: str = ''
    columnStartsWith: str = 'CONS_UPPER'
    
    def setup_columns(self, df: pd.DataFrame):
        """Set up column mappings for consolidation pairs."""
        upper_columns = [col for col in df.columns if col.startswith(self.columnStartsWith)]
        self.names = []
        self.column_pairs = []
        
        for upper_col in upper_columns:
            lower_col = upper_col.replace('UPPER', 'LOWER')
            if lower_col in df.columns:
                signal_name = f"{self.name}_{upper_col.split('_')[-1]}"
                self.names.append(signal_name)
                self.column_pairs.append((upper_col, lower_col))

    def find_last_pivot(self, ma_series: pd.Series) -> Tuple[pd.Timestamp, float]:
        """
        Find the last pivot point (peak or trough) in the MA series.
        Returns (pivot_timestamp, pivot_price)
        """
        if len(ma_series) < 3:
            return ma_series.index[-1], ma_series.iloc[-1]

        # Calculate differences between consecutive points
        diffs = ma_series.diff()
        
        # Find where the direction changes (sign changes in differences)
        sign_changes = np.sign(diffs).diff()
        
        # Get indices of pivot points (where sign changes)
        pivot_indices = ma_series.index[abs(sign_changes) > 0]
        
        if len(pivot_indices) == 0:
            return ma_series.index[-1], ma_series.iloc[-1]
            
        # Return the last pivot point

        # if no pivit found then use the start of the ma 
        if len(pivot_indices) == 0:
            return ma_series.index[0], ma_series.iloc[0]
        
        last_pivot_idx = pivot_indices[-1]
        return last_pivot_idx, ma_series[last_pivot_idx]

    def find_crossover_point(self, ma_series: pd.Series, upper_bound: float, lower_bound: float) -> Tuple[pd.Timestamp, float]:
        """
        Find the point where the MA series crosses the consolidation bounds.
        Returns (crossover_timestamp, crossover_price)
        """
        if ma_series.empty:
            return None, None

        # Create series indicating whether price is outside bounds
        above_upper = ma_series > upper_bound
        below_lower = ma_series < lower_bound
        outside_bounds = above_upper | below_lower

        # print(f'above_upper: {above_upper}')

        # Find the first point where price moves outside bounds
        if not outside_bounds.any():
            return None, None

        crossover_idx = outside_bounds[outside_bounds].index[-1]
        return crossover_idx, ma_series[crossover_idx]

    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute pre-move scores for each consolidation area.
        """
        if df.empty or len(self.column_pairs) == 0:
            return pd.DataFrame(np.nan, index=df.index, columns=self.names)

        results = pd.DataFrame(np.nan, index=df.index, columns=self.names)
        
        for (upper_col, lower_col), signal_name in zip(self.column_pairs, self.names):
            # Skip if required columns are missing
            if not all(col in df.columns for col in [upper_col, lower_col, self.maCol, self.atrCol]):
                continue

            # Get consolidation bounds
            upper_bound = df[upper_col].dropna().iloc[0] if not df[upper_col].dropna().empty else None
            lower_bound = df[lower_col].dropna().iloc[0] if not df[lower_col].dropna().empty else None
            
            if upper_bound is None or lower_bound is None:
                continue

            # Get the consolidation zone start and end
            cons_start = df[upper_col].dropna().index[0]
            cons_end = df[upper_col].dropna().index[-1]
            
            # Get MA series up to consolidation start and up to the end
            ma_before_cons = df[self.maCol].loc[:cons_start]
            
            # Find last pivot before consolidation
            pivot_time, pivot_price = self.find_last_pivot(ma_before_cons)
            # print(f'pivot_time: {pivot_time}, pivot_price: {pivot_price}')
            if pivot_time is None:
                continue

            # Find crossover point with consolidation bounds
            if pivot_time > cons_start:
                continue

            ma_from_pivot = df[self.maCol].loc[pivot_time:cons_end]
            crossover_time, crossover_price = self.find_crossover_point(ma_from_pivot, upper_bound, lower_bound)
            # print(f'crossover_time: {crossover_time}, crossover_price: {crossover_price}')

            # Calculate score components
            if crossover_time is None or crossover_price is None:
                continue

            premove_height = abs(pivot_price - crossover_price) / df[self.atrCol].loc[pivot_time]
            premove_duration = len(ma_from_pivot)

            
            # Calculate score using the premove_height and premove_duration.  
            steepness = premove_height / premove_duration
            final_score = steepness * premove_height * np.log1p(premove_duration)

            # Create a series of zeros for the full index
            score_series = pd.Series(np.nan, index=df.index)

            # Create a mask for the consolidation period
            cons_mask = (~pd.isna(df[upper_col])) & (~pd.isna(df[lower_col]))
            # Assign the score only during consolidation period
            
            # Assign scores to result DataFrame
            score_series[cons_mask] = self.get_score(final_score)
            results[signal_name] = score_series[cons_mask]

        return results


@dataclass
class BuySetup(MultiSignals):
    name: str = 'BuySetup'
    bswCol: str = ''
    retestCol: str = ''
    minCount: int = 3
    minBSW: float = 0.5
    minRetest: float = 0.5
    
    
    def __post_init__(self):
        self.normRange = None # No normalization for this signal
        self.name = f"{self.ls[0]}_{self.name}"
        self.name_sig_count = f"{self.name}_SigCount"
        self.name_entry = f"{self.name}_Entry"
        self.name_stop = f"{self.name}_Stop"
        self.name_buy = f"{self.name}_isBuy"
        self.name_fail = f"{self.name}_isFail"
        self.name_lhs = f"{self.name}_lhs"
        self.name_lls = f"{self.name}_lls"
        self.name_red1 = f"{self.name}_red1"
        self.name_red2 = f"{self.name}_red2"
        self.name_hh = f"{self.name}_hh"
        self.name_hl = f"{self.name}_hl"
        self.name_cc = f"{self.name}_cc"
        self.name_sw = f"{self.name}_sw"
        self.name_dr = f"{self.name}_dr"

        self.names = [
            self.name, 
            self.name_sig_count, 
            self.name_entry, 
            self.name_stop, 
            self.name_buy, 
            self.name_fail,
            self.name_lhs,
            self.name_lls,
            self.name_red2,
            self.name_red1,
            self.name_hh,
            self.name_hl,
            self.name_cc,
            self.name_sw,
            self.name_dr
        ]
        
        self.reversal_signal_count = 0
        self.entry_price = None
        self.stop_price = None
        self.is_buy = 0
        self.is_fail = 0
        self.step = 0
        self.hh = 0
        self.hl = 0
        self.cc = 0
        self.sw = 0
        self.dr = 0
        self.lhs = 0
        self.lls = 0
        self.red1 = 0
        self.red2 = 0

    def reset(self):
        self.reversal_signal_count = 0
        self.entry_price = None
        self.stop_price = None
        self.is_buy = 0
        self.is_fail = 0
        self.step = 0
        self.hh = 0
        self.hl = 0
        self.cc = 0
        self.sw = 0
        self.dr = 0
        self.lhs = 0
        self.lls = 0
        self.red1 = 0
        self.red2 = 0

    def get_default_series(self) -> pd.Series:
        return pd.Series({
            self.name: 0,
            self.name_sig_count: 0,
            self.name_entry: None,
            self.name_stop: None,
            self.name_buy: 0,
            self.name_fail: 0,
            self.name_lhs: 0,
            self.name_lls: 0,
            self.name_red1: 0,
            self.name_red2: 0,
            self.name_hh: 0,
            self.name_hl: 0,
            self.name_cc: 0,
            self.name_sw: 0,
            self.name_dr: 0
        })

    def _compute_row(self, df: pd.DataFrame) -> pd.Series:
        # Check if was a trigger or cancel last round
        if self.is_buy or self.is_fail:
            self.reset()
            return self.get_default_series()
        
        if len(df) < 3:
            return self.get_default_series()

        # Has 2 x lower highs and 2 x lower lows
        self.lhs = df.high.iat[-1] < df.high.iat[-2] and df.high.iat[-2] < df.high.iat[-3]
        self.lls = df.low.iat[-1] < df.low.iat[-2] and df.low.iat[-2] < df.low.iat[-3]
        self.red1 = df.open.iat[-2] > df.close.iat[-1]
        self.red2 = df.open.iat[-3] > df.close.iat[-2]

        if self.lhs and self.lls and self.red1 and self.red2:
            self.step = 1

        if self.step == 1:
            # Get the reversal signals for every run of the function
            self.hh = df.high.iat[-1] > df.high.iat[-2]            # Has a higher high
            self.hl = df.low.iat[-1] > df.low.iat[-2]              # Has a higher low
            self.cc = df.open.iat[-1] < df.close.iat[-1]           # Has a bullish candle
            self.sw = df[self.bswCol].iat[-1] >= self.minBSW       # Has a bullish strength/weakness
            self.dr = df[self.retestCol].iat[-1] >= self.minRetest # Has a double retest
            
            self.reversal_signal_count += sum([self.hh, self.hl, self.cc, self.sw, self.dr])

            # Set entry and stop if not set
            if self.reversal_signal_count >= self.minCount:
                self.entry_price = df.high.iat[-2]
                self.stop_price = min(df.low.iat[-1], df.low.iat[-2])
                self.step = 2

        # Buy Setup checking if entry price is set
        if self.step == 2:
            # Breaks entry price
            if df.high.iat[-1] > self.entry_price:
                self.is_buy = True
            elif df.low.iat[-1] < self.stop_price:
                self.is_fail = True
            elif df.high.iat[-1] < df.high.iat[-2]:
                self.entry_price = df.high.iat[-2]

        # normRagnge disabled for this signal so manually apply the normalization
        return pd.Series({
            self.name: self.is_buy * 100,
            self.name_sig_count: self.reversal_signal_count,
            self.name_entry: self.entry_price,
            self.name_stop: self.stop_price,
            self.name_buy: self.is_buy * 100,
            self.name_fail: self.is_fail * 100,
            self.name_lhs: self.lhs * 100,
            self.name_lls: self.lls * 100,
            self.name_red1: self.red1 * 100,
            self.name_red2: self.red2 * 100,
            self.name_hh: self.hh * 100,
            self.name_hl: self.hl * 100,
            self.name_cc: self.cc * 100,
            self.name_sw: self.sw * 100,
            self.name_dr: self.dr * 100
        })

    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        results = pd.DataFrame(index=df.index, columns=self.names)
        for i in range(len(df)):
            results.iloc[i] = self._compute_row(df.iloc[:i+1])
        return results
    
    

# --------------------------------------------------------------------
# ----- E V E N T   S I G N A L S ------------------------------------
# --------------------------------------------------------------------

@dataclass
class IsValid(Signals):
    """Checks if a given event is valid based on a condition."""
    name: str = 'IS_VALID'
    colsToValidate: list = None
    threshold: float = 0.0
    validType: str = 'any' # 'any' or 'all'

    def __post_init__(self):
        # self.name = f"IS_VALID_{self.event_column}"
        self.names = [self.name]

    def _compute_row(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 2:
            return False
        if self.validType == 'any':
            return (df[self.colsToValidate].iloc[-1]  > self.threshold).any()
        elif self.validType == 'all':
            return (df[self.colsToValidate].iloc[-1]  > self.threshold).all()
        else:
            raise ValueError("Valid type must be 'any' or 'all'")




@dataclass
class Validate(Signals):
    """
        A class for performing value and time-based comparisons on DataFrame values.
        Inherits from Signals class to provide signal generation capabilities.

        The class supports multiple comparison types:
        1. Direct value comparisons ('>', '<', '==')
        2. Pivot-based comparisons ('p>p', 'p<p', '^p', 'vp', '>p', '<p')
        3. Cross-point detections ('^', 'v')
        4. Range validation ('><')
        5. Time-based comparisons ('t>t', 't<t', 't==t', 't>=t', 't<=t', 't!=t')

        Parameters
        ----------
        val1 : str | float | tuple
            First value for comparison. Can be:
            - str: Column name (uses index -1)
            - float: Direct value
            - tuple(str, int): (column_name, index)
            For time-based comparisons, this will always use df.index[-1]
        
        val2 : str | float | tuple
            Second value for comparison. Can be:
            - str: Column name (uses index -1) or time string (e.g., '10:55') for time comparisons
            - float: Direct value
            - tuple(str, int): (column_name, index)
            - tuple(float, float): Range bounds for '><' operator
        
        operator : str
            Comparison operator. Valid options:
            - '>', '<', '==': Direct comparisons
            - 'p>p', 'p<p': Pivot-to-pivot comparisons
            - '^p', 'vp': Cross-up/down through pivot
            - '>p', '<p': Value to pivot comparisons
            - '^', 'v': Cross-up/down through value
            - '><': Range check
            - 't>t', 't<t', 't==t', 't>=t', 't<=t', 't!=t': Time comparisons
        
        ls : str, default 'LONG'
            Signal direction indicator. Used in signal name generation.

        Methods
        -------
        _get_value(df: pd.DataFrame, val_spec, prev: bool = False) -> float
            Extracts value from DataFrame based on specification.
        
        _get_pivot(df: pd.DataFrame, val_spec) -> float
            Gets pivot value from DataFrame, handling NaN values.
        
        _compute_row(df: pd.DataFrame) -> bool
            Computes the comparison result for the current row.

        Examples
        --------
        # Compare if current value of 'CLOSE' is greater than VWAP
        >>> validator = Validate('CLOSE',  '>', 'VWAP_session', ls='LONG')
        
        # Check if current value crossed above pivot point
        >>> validator = Validate('CLOSE', ^p', ('HP_hi_10', -2), ' ls='LONG')
        
        # Check if current time is after 10:55
        >>> validator = Validate(None, 't>t', '10:55', ls='LONG')
        
        # Check if value is within range
        >>> validator = Validate('RSI', '><',  (30, 70), ls='LONG')

        Notes
        -----
        - For pivot-based comparisons, the pivot value is extracted from historical data
        - Time comparisons always use the latest index timestamp (df.index[-1])
        - The class automatically generates a signal name based on parameters
        - NaN values are handled gracefully in pivot calculations
        - Minimum of 2 rows required in DataFrame for comparisons involving previous values

        Raises
        ------
        ValueError
            If an invalid operator is provided
    """ 
    val1: str | float | tuple = ''
    operator: str = ''
    val2: str | float | tuple = ''
    ls: str = 'LONG'
    

    def __post_init__(self):
        self.name = f"VAL_{self.ls[:1]}_{self.val1}_{self.operator}_{self.val2}"
        self.names = [self.name]
        if self.operator not in ['>', '<', 'p>p', 'p<p', '^p', 'vp', '>p', '<p', '^', 'v', '><', '==',
                        't>t', 't<t', 't==t', 't>=t', 't<=t', 't!=t']:
            raise ValueError("Invalid operator: must be one of '>', '<', 'p>p', 'p<p', '^p', 'vp', '>p', '<p', '^', 'v', '><', '==' ")
        self.normRange = (0,1)

    def _get_value(self, df: pd.DataFrame, val_spec, prev:bool=False) -> float:
        """Get value from dataframe based on specification."""
        offset = 1 if prev else 0 
        if isinstance(val_spec, tuple): 
            if isinstance(val_spec[0], str): return df[val_spec[0]].iat[val_spec[1]-offset]          
            return val_spec
        elif isinstance(val_spec, str): return df[val_spec].iat[-1-offset]
        return val_spec

    def _get_pivot(self, df: pd.DataFrame, val_spec) -> float:
        """Get pivot value from dataframe, handling NaN values."""
        col = val_spec[0] if isinstance(val_spec, tuple) else val_spec
        idx = val_spec[1] if isinstance(val_spec, tuple) else -1
        valid_points = df[col].dropna()
        if len(valid_points) < abs(idx): return None
        return None if valid_points.empty else valid_points.iloc[idx]

    def _compute_row(self, df: pd.DataFrame) -> bool:
        """Compute the comparison result for a single row."""
        if len(df) < 2: return False

        if self.operator in ['t>t', 't<t', 't==t', 't>=t', 't<=t', 't!=t']:
            # Get current timestamp from index
            t1 = df.index[-1].time()
            
            # Convert comparison time string to time object
            t2 = pd.to_datetime(self.val2).time()
            
            # Perform time-based comparisons
            if self.operator == 't>t': return t1 > t2
            if self.operator == 't<t': return t1 < t2
            if self.operator == 't==t': return t1 == t2
            if self.operator == 't>=t': return t1 >= t2
            if self.operator == 't<=t': return t1 <= t2
            if self.operator == 't!=t': return t1 != t2


        v1_as_pivot = ['p>p', 'p<p']
        v2_as_pivot = ['p>p', 'p<p', '^p', 'vp', '>p', '<p']

        v1 = self._get_value(df, self.val1) if self.operator not in v1_as_pivot else self._get_pivot(df, self.val1)
        v2 = self._get_value(df, self.val2) if self.operator not in v2_as_pivot else self._get_pivot(df, self.val2)

        if pd.isna(v1) or pd.isna(v2): return False

        if self.operator == 'p>p': return v1 > v2
        if self.operator == 'p<p': return v1 < v2
        if self.operator == '>p' : return v1 > v2
        if self.operator == '>'  : return v1 > v2
        if self.operator == '<p' : return v1 < v2
        if self.operator == '<'  : return v1 < v2
        if self.operator == '==' : return v1 == v2

        if self.operator == '><': return self.val2[0] <= v1 <= self.val2[1] if isinstance(self.val2, tuple) else False


        v1_prev = self._get_value(df, self.val1, prev=True)
        v2_prev = self._get_value(df, self.val2, prev=True) if self.operator not in v2_as_pivot else self._get_pivot(df, self.val2)
        
        if pd.isna(v1_prev) or pd.isna(v2_prev): return False

        if self.operator == '^p': return v1_prev < v2_prev and v1 >= v2_prev
        if self.operator == 'vp': return v1_prev > v2_prev and v1 <= v2_prev
        if self.operator == '^' : return v1_prev < v2_prev and v1 >= v2_prev
        if self.operator == 'v' : return v1_prev > v2_prev and v1 <= v2_prev


        raise ValueError("Invalid operator")


@dataclass
class ValidatePoints(Signals):
    pnt1: str = ''
    pnt2: str = ''
    operator: str = ''
    pnt1idx: int = 0
    pnt2idx: int = 0
    ls: str = 'LONG'

    def __post_init__(self):
        self.name = f"VAL_{self.ls[:1]}_{self.pnt1}_{self.operator}_{self.pnt2}"
        self.names = [self.name]


@dataclass
class IsMATrending(Signals):
    """Checks if price is trending up or down compared the MA"""
    maCol: str = ''

    def __post_init__(self):
        self.name = f"TREND_{self.maCol}"
        self.names = [self.name]

    def _compute_row(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 2:
            return False
        
        if self.ls == 'LONG':
            ma_is_up = df[self.maCol].iloc[-1] > df[self.maCol].iloc[-2] 
            price_above_ma = df['low'].iloc[-1] > df[self.maCol].iloc[-1]
            return ma_is_up and price_above_ma
        
        elif self.ls == 'SHORT':
            ma_is_down = df[self.maCol].iloc[-1] < df[self.maCol].iloc[-2] 
            price_below_ma = df['high'].iloc[-1] < df[self.maCol].iloc[-1]
            return ma_is_down and price_below_ma

import re

@dataclass
class IsPointsTrending(Signals):
    """Checks if price is trending up or down compared the MA"""
    hpCol: str = ''
    lpCol: str = ''

    def __post_init__(self):
        point_span = re.search(r'\d+$', self.hpCol).group() # gets the last val from the points column eg HP_h_3 = 3
        self.name = f"TREND_{self.ls}_{point_span}"
        self.names = [self.name]

    def _compute_row(self, df: pd.DataFrame) -> pd.DataFrame:       
        hps = df[self.hpCol].dropna()
        lps = df[self.lpCol].dropna()

        if len(hps) < 4 and len(lps) < 4:
            return False
        
        if self.ls == 'LONG':
            hp_is_up = hps.iloc[-1] > hps.iloc[-2] 
            lp_is_up = lps.iloc[-1] > lps.iloc[-2]
            return hp_is_up and lp_is_up
        
        elif self.ls == 'SHORT':
            hp_is_down = hps.iloc[-1] < hps.iloc[-2] 
            lp_is_down = lps.iloc[-1] < lps.iloc[-2]
            return hp_is_down and lp_is_down


@dataclass
class BreaksPivot(Signals):
    """Checks if price crosses above/below a metric"""
    pointCol: str = ''
    direction: str = '' # 'above' or 'below'

    def __post_init__(self):
        self.name = f"BRK_{self.direction[:2]}_{self.pointCol}"
        self.names = [self.name]

    def _compute_row(self, df: pd.DataFrame) -> pd.DataFrame:
        point = df[self.pointCol].dropna()
        if len(point) < 1:
            return False

        if self.direction == 'above':
            return df['close'].iloc[-1] > point.iloc[-1] 
        
        elif self.direction == 'below':
            return df['close'].iloc[-1] < point.iloc[-1]
        
        else:
            raise ValueError("Direction must be 'above' or 'below'")

@dataclass
class IsPullbackBounce(Signals):
    name: str = 'PBB'
    pointCol: str = ''
    supResCol: str = ''

    def __post_init__(self):
        self.name = f"{self.name}_{self.pointCol}_{self.supResCol}"
        self.names = [self.name]
        self.normRange = (0,1)

    def _compute_row(self, df: pd.DataFrame) -> float:
        """
        Gets a Signal at a high Point or a low Point by looking back 
        to the nth most recent point and confirming if:
        Long example:
        1. recent hp low > current support (proves hp clears current support levels)
        2. since recent HP the lowest point < current support
        3. close > support (price bounced off support)
        """
        points = df[self.pointCol].dropna()
        if len(points) < 1:
            return 0.0
        
        pnt_idx = points.index[-1]

        if self.ls == 'LONG':
            # recent hp low > current support (proves hp clears current support levels)
            recent_hp_low = df['low'].loc[pnt_idx]
            sup = df[self.supResCol].iloc[-1]
            hp_bar_cleard_sup = recent_hp_low > sup

            # since recent HP the lowest point < current support
            lowest = df['low'].loc[pnt_idx:].min()
            hp_lowest_below_sup = lowest < sup

            # close > support (price bounced off support)
            close = df['close'].iloc[-1]
            close_above_sup = close > sup

            return hp_bar_cleard_sup and hp_lowest_below_sup and close_above_sup
        
        elif self.ls == 'SHORT':
            # recent lp high < current resistance (proves lp clears current resistance levels)
            recent_lp_high = df['high'].loc[pnt_idx]
            res = df[self.supResCol].iloc[-1]
            lp_bar_cleard_res = recent_lp_high < res

            # since recent LP the highest point > current resistance
            highest = df['high'].loc[pnt_idx:].max()
            lp_highest_above_res = highest > res

            # close < resistance (price bounced off resistance)
            close = df['close'].iloc[-1]
            close_below_res = close < res

            return lp_bar_cleard_res and lp_highest_above_res and close_below_res
        
        return False
    
#! not working.maybe redundent
@dataclass
class IsConsolidationBreakout(Signals):
    """
    Detects breakouts from consolidation zones, using provided pivot points column.
    Only considers breakouts after the last pivot point within consolidation duration.
    """
    valToCheck: str = 'close'  # Price column to check
    pointCol: str = ''  # Column containing pivot points (highs for SHORT, lows for LONG)
    consColumns: list[str] = field(default_factory=list)  # List of consolidation columns
    extendPeriods: int = 0  # Number of periods to extend consolidation check
    
    def __post_init__(self):
        self.name = f"CONS_BRK_{self.valToCheck}"
        super().__post_init__()

    
    def _compute_row(self, df: pd.DataFrame) -> float:
        """
        Compute breakout signal for the current window.
        Only considers breakouts after the last pivot point within consolidation duration.
        Can check for breakouts beyond consolidation end using extendPeriods.
        """
        if len(df) < 2:
            return np.nan
            
        curr_price = df[self.valToCheck].iloc[-1]
        prev_price = df[self.valToCheck].iloc[-2]
        
        # Check each consolidation column until we find a valid one
        for cons_col in self.consColumns:
            if cons_col not in df.columns:
                continue

            # see if consilidation period exists
            last_cons_idx = df[cons_col].last_valid_index()
            if last_cons_idx is None:
                continue


            # Check now is within checking period 
            # checking period is from the last pivot point within the consolidation period 
            # to the end of of the consolidation plus the extendPeriods

            # Cheeck if past the end of the checking period
            end_checking_period = df[cons_col].last_valid_index() + self.extendPeriods #! dtaetime index vs int index
            if df.index[-1] > end_checking_period:
                continue
            
            # Find the last pivot point within the consolidation period and make sure we are not before it 
            cons_start = df[cons_col].first_valid_index()
            last_point = df.loc[cons_start:][self.pointCol].last_valid_index()

            # consolidation period
            cons_start = df[cons_col].first_valid_index()
            cons_end = df[cons_col].last_valid_index()
            if cons_start is None or cons_end is None:
                continue

            # Find the last pivot point within the consolidation period
            last_point = df[self.pointCol].loc[:cons_end].last_valid_index()

            # check if the last point is within the consolidation period
            if last_point < cons_start and last_point > cons_end:
                continue

                

                
            cons_level = df[cons_col].loc[last_cons_idx]
            
            if self.ls == 'LONG':
                # Check if last low point was within consolidation
                was_inside = pivot_point <= cons_level
                # Check for breakout
                breaks_out = (prev_price <= cons_level and curr_price > cons_level)
                
                if was_inside and breaks_out:
                    return 1.0
                    
            else:  # SHORT
                # Check if last high point was within consolidation
                was_inside = pivot_point >= cons_level
                # Check for breakout
                breaks_out = (prev_price >= cons_level and curr_price < cons_level)
                
                if was_inside and breaks_out:
                    return 1.0
        
        return 0.0






# --------------------------------------------------------------------
# ----- S T R A T E G Y   S I G N A L S ------------------------------
# --------------------------------------------------------------------

@dataclass
class TBP(Signals):
    name: str = 'TBP' # Three Bar Play
    atrCol: str = ''
    barsToPlay: int = 3

    def __post_init__(self):
        self.name = f"{self.name}_{self.barsToPlay}"
        self.names = [self.name]

    def _compute_row(self, df: pd.DataFrame):
        if len(df) < self.barsToPlay:
            return 0
            
        bars = df.iloc[-self.barsToPlay:]
        ranges = bars['high'] - bars['low']
        bodies = abs(bars['close'] - bars['open'])
        atr = df[self.atrCol].iloc[-1]
        
        # Bar 1 (igniting bar) scoring
        bar1_range = ranges.iloc[0]
        bar1_body = bodies.iloc[0]
        bar1_score = (bar1_range / atr) * (bar1_body / bar1_range)
        
        # Get consolidation bars (between igniting and trigger)
        consol_bars = ranges.iloc[1:-1]
        
        if self.ls == 'LONG':
            # For longs, check if bars are in upper 50% of igniting bar's range
            bar1_midpoint = bars['low'].iloc[0] + (bar1_range / 2)
            consol_lows = bars['low'].iloc[1:-1]
            zone_score = sum(consol_lows > bar1_midpoint) / len(consol_lows)
            # Check high alignment
            highs = bars['high'].iloc[:-1]
            alignment_score = 1 - (highs.std() / bar1_range)
        else:  # SHORT
            # For shorts, check if bars are in lower 50% of igniting bar's range
            bar1_midpoint = bars['high'].iloc[0] - (bar1_range / 2)
            consol_highs = bars['high'].iloc[1:-1]
            zone_score = sum(consol_highs < bar1_midpoint) / len(consol_highs)
            # Check low alignment
            lows = bars['low'].iloc[:-1]
            alignment_score = 1 - (lows.std() / bar1_range)
        
        # Tightness scoring (same for both long/short)
        tightness_score = 1 - (consol_bars.mean() / bar1_range)
        
        score = (
            bar1_score * 0.4 +
            zone_score * 0.3 +
            tightness_score * 0.3
        ) * 100

        return score if not pd.isna(score) else 0
    

@dataclass
class TurnBar(Signals):
    name: str = 'TURNBAR'
    atrCol: str = 'atr'  # Column name for Average True Range
    
    def __post_init__(self):
        self.name = f"{self.name}_{self.ls}"
        self.names = [self.name]
    
    def _compute_row(self, df: pd.DataFrame = pd.DataFrame()):
            
        score = 0
        if self.ls == 'LONG':
            # Get initial down move
            down_move = df.high.iat[-2] - df.low.iat[-2]
            move_strength = min(down_move / df[self.atrCol].iat[-2] / 2, 1) * 50
            
            # Check recovery
            recovery = df.high.iat[-1] - df.low.iat[-2]
            total_move = df.high.iat[-2] - df.low.iat[-2]
            recovery_score = min(recovery / total_move, 1) * 50
            
            score = move_strength + recovery_score
            
        elif self.ls == 'SHORT':
            # Get initial up move
            up_move = df.high.iat[-2] - df.low.iat[-2]
            move_strength = min(up_move / df[self.atrCol].iat[-2] / 2, 1) * 50
            
            # Check reversal
            reversal = df.high.iat[-2] - df.low.iat[-1]
            total_move = df.high.iat[-2] - df.low.iat[-2]
            reversal_score = min(reversal / total_move, 1) * 50
            
            score = move_strength + reversal_score
            
        return round(score, 2)


@dataclass
class Condition:
    step: Union[int, str]
    name: str
    val1: Union[str, float, int]
    operator: str
    val2: Union[str, float, int]
    is_met: bool = False
    startFromStep: int = 1
    
    def _get_value(self, row: pd.Series, val: Union[str, float, int]) -> float:
        """Extract value from either a row column or direct value"""
        if isinstance(val, str):
            return row[val]
        return val
    
    def _compare(self, val1: float, val2: float) -> bool:
        """Compare two values based on the operator"""
        if   self.operator == '>':  return val1 >  val2
        elif self.operator == '<':  return val1 <  val2
        elif self.operator == '>=': return val1 >= val2
        elif self.operator == '<=': return val1 <= val2
        elif self.operator == '==': return val1 == val2
        return False
    
    def reset(self):
        self.is_met = False
    
    def evaluate(self, row: pd.Series) -> bool:
        if self.is_met: return True
        val1 = self._get_value(row, self.val1)
        val2 = self._get_value(row, self.val2)
        self.is_met = self._compare(val1, val2)
        return self.is_met


@dataclass
class Strategy_old(MultiSignals):
    name: str
    
    def __post_init__(self):
        # Initialize MultiSignals parent class
        self.normRange = None # No normalization for this signal
        self.name = f'Stgy_{self.name}'
        self.name_conditions_met = f"{self.name}_ConditionsMet"
        self.name_steps_passed = f"{self.name}_StepsPassed"
        self.name_action = f"{self.name}_Action"
        self.name_pct_complete = f"{self.name}_PctComplete"

        self.names = [
            self.name_conditions_met, 
            self.name_steps_passed, 
            self.name_action, 
            self.name_pct_complete
            ]
        
        self.steps = {}
        self.current_step = 1  # Start at step 1 instead of 0
        self.total_steps = 0
        self.steps_passed = 0  # Added counter for steps passed
        self.results = pd.Series({
            self.name_conditions_met: 0, 
            self.name_steps_passed: 0, 
            self.name_action: 'WAIT',
            self.name_pct_complete: 0
        })

    def _new_step(self, step:int):
        self.steps[step] = {'pass_ifs': [], 'reset_ifs': []}
        self.total_steps = max(self.total_steps, step)


    def pass_if(self, step:int, scoreCol:str, operator:str, threshold:float|int):
        if step not in self.steps: self._new_step(step)
        cond_name = f"{self.name}_Step{step}_PassIf{scoreCol}_{operator}_{threshold}"
        self.steps[step]['pass_ifs'] += [Condition(step, cond_name, scoreCol, operator, threshold)]
        self.results[cond_name] = False
        # Add condition name to names list if not already there
        if cond_name not in self.names:
            self.names.append(cond_name)      

    def reset_if(self, step:int, scoreCol:str, operator:str, threshold:float|int, startFromStep:int):
        if step not in self.steps: raise ValueError(f"Strategy :: Step {step} does not exist")
        cond_name = f"{self.name}_Step{step}_ResetIf{scoreCol}_{operator}_{threshold}"
        self.steps[step]['reset_ifs'] += [Condition(step, cond_name, scoreCol, operator, threshold, False, startFromStep)]
        self.results[cond_name] = False
        # Add condition name to names list if not already there
        if cond_name not in self.names:
            self.names.append(cond_name)
    
    def _check_step_conditions(self, condType: str, step: int, row: pd.Series):
        """Check if all conditions of a specific type are met for a step"""
        # If the step doesn't exist or there are no conditions of this type, return False
        if step not in self.steps or not self.steps[step][condType]:
            return False
        
        # Start with assumption that all conditions are met
        all_met = True
        
        # Evaluate ALL conditions and update results
        for cond in self.steps[step][condType]:
            is_met = cond.evaluate(row)
            self.results[cond.name] = is_met
            
            # Track if any condition fails, but continue evaluating all conditions
            if not is_met:
                all_met = False
        
        return all_met
    
    def _reset_from_step(self, step:int):
        self.current_step = step
        for s in self.steps:
            if s >= step:
                # Reset both pass and reset conditions
                for cond in self.steps[s]['pass_ifs']:
                    cond.reset()
                    self.results[cond.name] = False
                for cond in self.steps[s]['reset_ifs']:
                    cond.reset()
                    self.results[cond.name] = False

    
    def _get_min_start_step(self, step: int) -> int:
        if not self.steps[step]['reset_ifs']:
            return step
        # Filter to only get startFromSteps of conditions that were met
        met_conditions = [cond for cond in self.steps[step]['reset_ifs'] if cond.is_met]
        if not met_conditions:
            return step
        return min([cond.startFromStep for cond in met_conditions])
    
    def _update_metrics(self):
        """Update metrics in the results Series"""
        # Count total conditions met
        met_conditions = 0
        
        for step in self.steps:
            for cond in self.steps[step]['pass_ifs']:
                if cond.is_met:
                    met_conditions += 1
        
        # Update metrics
        self.results[self.name_conditions_met] = met_conditions
        self.results[self.name_steps_passed] = self.current_step - 1
        
        # Calculate percentage complete (based on steps)
        if self.total_steps > 0:
            self.results[self.name_pct_complete] = round((self.current_step - 1) / self.total_steps * 100, 1)
    
    def _compute_row(self, df: pd.DataFrame) -> pd.Series:
        if len(df) == 0:
            return self.results
        
        row = df.iloc[-1]
        reset_occurred = False
        triggered_reset_condition = None

        # First check if we need to reset due to reset conditions
        # Only check reset conditions for the current and previous steps
        for step in range(1, self.current_step + 1):
            if step in self.steps and self._check_step_conditions('reset_ifs', step, row):
                reset_step = self._get_min_start_step(step)
                
                # Save which reset condition was triggered before resetting
                for cond in self.steps[step]['reset_ifs']:
                    if cond.is_met:
                        triggered_reset_condition = cond.name
                
                self._reset_from_step(reset_step)
                reset_occurred = True
                # If we've reset, update metrics and exit
                self._update_metrics()
                
                # Re-set the triggered condition to True after reset
                if triggered_reset_condition:
                    self.results[triggered_reset_condition] = True
                    
                return self.results
        
        # Check if current step's pass conditions are met
        if self._check_step_conditions('pass_ifs', self.current_step, row):
            self.current_step += 1
            if self.current_step > self.total_steps:
                self.results[self.name_action] = 'BUY'
        
        # Update metrics
        self._update_metrics()

        return self.results
    
    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        results = pd.DataFrame(index=df.index, columns=self.names)
        for i in range(len(df)):
            results.iloc[i] = self._compute_row(df.iloc[:i+1])
        return results

@dataclass
class Strategy(MultiSignals):
    name: str
    chartArgs: List[ChartArgs] = field(default_factory=list)


    def __post_init__(self):
        self.names = [self.name, f"{self.name}_pctComplete", f"{self.name}_passed", f"{self.name}_meanScore"]
        self.steps = []
        self.rows = [] #  list of dicts for each row
        self.pctComplete = 0
        self.meanScoreOfSteps = 0
        self.current_step = 1
        self.passed = False
        self.scoreList = []
        self.chartArgs = []
        self.count_steps_passed = 0

    
    def add_chart_args(self, meanScoreArgs:ChartArgs, pctCompleteArgs:ChartArgs, scoreArgs:ChartArgs, failArgs:ChartArgs, scoreSubItemArgs:ChartArgs):
        meanScoreArgs.name = f"{self.name}_meanScore"
        meanScoreArgs.columns = [meanScoreArgs.name]
        pctCompleteArgs.name = f"{self.name}_pctComplete"
        pctCompleteArgs.columns = [pctCompleteArgs.name]
        scoreArgs.name = f"{self.name}_score"
        scoreArgs.columns = []
        failArgs.name = f"{self.name}_fail"
        failArgs.columns = []
        scoreSubItemArgs.name = f"{self.name}_subItem"
        scoreSubItemArgs.columns = []
        self.chartArgs = [meanScoreArgs, pctCompleteArgs, scoreArgs, failArgs, scoreSubItemArgs]
        return self
    
    def get_name(self, step:int, objName):
        name = f"{self.name}_{step}_{objName}"
        if name not in self.names:  # Avoid duplicates
            self.names.append(name)
        return name

    def add_step(self, scoreObj:Score, failObj:Score, ifFailStartFromStep:int):
        self.steps.append({'scoreObj': scoreObj, 'failObj': failObj, 'ifFailStartFromStep': ifFailStartFromStep})
        step = len(self.steps)
        self.chartArgs[2].columns += [self.get_name(step, scoreObj.name)]
        self.chartArgs[3].columns += [self.get_name(step, failObj.name)]
        self.chartArgs[4].columns += [self.get_name(step, sig.name) for sig in scoreObj.sigs]
        return self
        
    
    def add_row_item(self, df, obj):
        name = f"{self.name}_{self.current_step}_{obj.name}"
        self.rows[-1][name] = df[obj.name].iat[-1] 
        return self
    
    def add_row_metrics(self):
        self.rows[-1][f"{self.name}_passed"] = self.passed
        self.rows[-1][f"{self.name}_pctComplete"] = self.pctComplete
        self.rows[-1][f"{self.name}_meanScore"] = self.meanScoreOfSteps
        return self
    
    def update_mean_score(self, df, scoreObj):
        score = df[scoreObj.name].iloc[-1]
        self.scoreList += [score]
        self.meanScoreOfSteps = sum(self.scoreList) / len(self.scoreList)

    def update_pct_complete(self):
        self.pctComplete = 0 if self.count_steps_passed == 0 else (self.count_steps_passed / len(self.steps)) * 100

    def reset(self, startFromStep:int=1):
        self.current_step = startFromStep
        self.count_steps_passed = startFromStep - 1
        self.scoreList = self.scoreList[:startFromStep-1] # keep the scores from the failed step
        self.meanScoreOfSteps = 0 if len(self.scoreList) == 0 else sum(self.scoreList) / len(self.scoreList)
        self.update_pct_complete()
        self.passed = False
  


    def _compute_row(self, df: pd.DataFrame) -> pd.Series:
        if self.passed > 0:
            self.reset(1) # reset the strategy if it has passed

        self.rows.append({}) # add a new row to the list of rows

        for i, step in enumerate(self.steps):

            if i + 1 >= self.current_step:
                scoreObj = step['scoreObj']
                failObj = step['failObj']
                scorePassed  = df[scoreObj.name_passed].iloc[-1]
                failed = df[failObj.name_passed].iloc[-1]

                if pd.isna(scorePassed):
                    break

                if failed > 0:
                    self.reset(step['ifFailStartFromStep'])
                    break

                if scorePassed > 0:
                    self.count_steps_passed += 1
                    self.update_mean_score(df, scoreObj)
                    self.add_row_item(df, scoreObj)
                    self.add_row_item(df, failObj)
                    
                    for sig in scoreObj.sigs:
                        self.add_row_item(df, sig)

                    if self.current_step == len(self.steps):
                        self.passed = True
                    else:
                        self.current_step += 1

        self.update_pct_complete()
        self.add_row_metrics()
      
        # Create a Series with NaN values for all expected columns
        result = pd.Series(index=self.names, dtype='float64')
        
        # Fill in the values that we have
        for key, value in self.rows[-1].items():
            if key in result.index:
                result[key] = value
        
        return result

    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        results = pd.DataFrame(index=df.index, columns=self.names)
        for i in range(len(df)):
            results.iloc[i] = self._compute_row(df.iloc[:i+1])
        return results



