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

def normalize(val:float, minVal:float, maxVal:float, roundTo:int=2):
    """normalizes the value between the min and max."""
    if minVal == maxVal: return 0 # cannot divide by zero
    if minVal < maxVal: 
        if val <= minVal: val = minVal
        if val >= maxVal: val = maxVal
    else:
        if val >= minVal: val = minVal
        if val <= maxVal: val = maxVal
    
    r = round((val - minVal) / (maxVal - minVal) * 100, roundTo)
    # Don't return NaN
    if r > -1: 
        return r
    return 0

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

def get_valid_pb(ls, df, pointCol:str, toCol:str, atrCol:str, minLen:int=3, atrMultiple:int=1):
    # check if names exists in df
    if not all([pointCol in df.columns, toCol in df.columns, atrCol in df.columns]):
        return None
    
    # print(f'{ls} {pointCol=} {toCol=} {atrCol} {minLen} {atrMultiple}')
    # check if has points
    points = df[pointCol].dropna()
    # print(f'length of points: {len(points)}')
    if len(points) < 2:
        return None
    
    # check if window long enough
    w0 = df.loc[points.index[-1]:]
    # print(f'lenght of w0: {len(w0)}')
    if len(w0) < minLen:
        return None
    
    # important to check if the value are ok at the point and not after the point
    toVal = w0[toCol].iat[0]
    atr = w0[atrCol].iat[0]

    # print(f'{toVal=} {atr=}')

    if any([pd.isna(toVal), pd.isna(atr)]):
        return None
    
    if ls == 'LONG':
        hp_bar_low = w0.low.iat[0]

        #check if high < two previous high ago
        if not w0.high.iat[-1] < w0.high.iat[-3] :
            return None
        
        # Check if the distance between high point and toCol is at least n ATRs
        distance = hp_bar_low - toVal
        if distance <= 0:
            return None
        
        if distance < (atrMultiple * atr):
            return None

        # print(f'{atr=} {hp_bar_low=} {toVal=} {distance=} ')
        return w0
    
    if ls == 'SHORT':       
        lp_bar_high = w0.high.iat[0]
        
        # Check if the distance between high point and toCol is at least n ATRs
        distance = toVal - lp_bar_high
        if distance <= 0:
            return None
        
        if distance < (atrMultiple * atr):
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

    def __post_init__(self):
        self.name = f"Sig{self.ls[0]}_{self.name}"
        self.names = [self.name]

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
                        
    # @abstractmethod
    def run(self, df: pd.DataFrame = pd.DataFrame()) -> pd.Series:
        """Generate signal scores for the lookback period."""
        if len(df) < 10:
            return self.return_series(df.index[-1], self.get_score(0))
        
        lookback = min(self.lookBack, len(df)) - 1

        # this then gets populated with the results of the computation
        result_series = pd.Series(np.nan, index=df.index[-lookback:])
        
        for i in range(lookback):
            current_idx = -(lookback - i)
            if abs(current_idx) >= len(df) :
                continue
            
            #! This is where the computation is done. 
            #! Each sub class must have a _compute_row method that does the computation
            current_window = df.iloc[:current_idx+1]
            if current_window.empty:
                continue
            val = self._compute_row(current_window)
            if not pd.isna(val):
                result_series.iloc[i] = float(val)

        return self.return_series(df.index[-self.lookBack:], self.get_score(result_series))


@dataclass
class Score(Signals):
    cols: List[str] = field(default_factory=list)
    weight: float = 1.0
    scoreType: str = 'mean'  # 'mean', 'sum', 'max', 'min'
    containsString: str = ''
    containsAllStrings: List[str] = field(default_factory=list)
    rawName: str = ''
    
    def __post_init__(self):
        """Initialize the Score class and validate inputs."""
        if self.scoreType not in ['mean', 'sum', 'max', 'min']:
            raise ValueError(f"Invalid scoreType: {self.scoreType}")
            
        self.name = self.rawName if self.rawName else f"Score_{self.name}"
        self.names = [self.name]
        self._filtered_cols = None  # Cache for filtered columns
    
    def _get_filtered_columns(self, df: pd.DataFrame) -> List[str]:
        """Get and cache filtered columns to avoid recomputation."""
        if self._filtered_cols is None:
            cols = self.cols if self.cols else list(df.columns)
            
            if self.containsString:
                cols = [col for col in cols if self.containsString in col]
            
            if self.containsAllStrings:
                cols = [col for col in cols if all(s in col for s in self.containsAllStrings)]
                
            self._filtered_cols = cols
            
        return self._filtered_cols
    
    def _compute_row(self, df: pd.DataFrame) -> float:
        """Compute score for the current window of data."""
        # Get filtered columns (using cached version if available)
        filtered_cols = self._get_filtered_columns(df)
        
        if len(filtered_cols) == 0:
            return np.nan
        
        # Get rows to score
        rows_to_score = df[filtered_cols].iloc[-1:] 
        
        # Handle empty or all-NaN case
        if rows_to_score.isna().all().all():
            return np.nan
        
        # Fill NaN with 0 before calculating
        rows_to_score = rows_to_score.fillna(0)
        
        # Compute the score based on type
        if self.scoreType == 'mean':
            val = rows_to_score.mean().mean()
        elif self.scoreType == 'sum':
            val = rows_to_score.sum().sum()
        elif self.scoreType == 'max':
            val = rows_to_score.max().max()
        elif self.scoreType == 'min':
            # For min, we might want to exclude the zeros that were NaN
            val = rows_to_score.replace(0, np.inf).min().min()
        else:
            val = np.nan
        
        if pd.isna(val):
            return np.nan
            
        return val * self.weight
    
    def reset_cache(self):
        """Reset the filtered columns cache if needed (e.g., if columns change)."""
        self._filtered_cols = None
            

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
class PBPctHLLH(Signals):
    """Computes the % of bars that have a lower highs (BULL pullback, so downward)
    Vice versa for BEAR case. So this is only for pullbacks not overall trends. """
    name: str = 'PB_HLLH'
    pointCol: str = ''
    toCol  : str = ''
    atrCol: str = ''
    atrMultiple: float = 1.0 # minimum number of ATRs required between pointCol low and toCol
    
    def __post_init__(self):
        self.name = f"Sig{self.ls[0]}_{self.name}"
        self.names = [self.name]   
   

    #$ **kwargs is used to allow any signal arguments to be passed to any run method. This so that the same run method can be used when looping through signals.
    def _compute_row(self, df:pd.DataFrame=pd.DataFrame(), **kwargs):
        """Computes the % of bars that have a lower highs (BULL pullback, so downward)
        Vice versa for BEAR case. So this is only for pullbacks not overall trends. """

        window  = get_valid_pb(
            ls=self.ls, 
            df=df, 
            pointCol=self.pointCol, 
            toCol=self.toCol, 
            atrCol=self.atrCol, 
            minLen=3, 
            atrMultiple=1)
        
        if window is None:
            return 0.0

        if self.ls == 'LONG': # if there are more than 2 bars in the pullback from the high
            # eg fhp.high < fhp.high.shift() retruns a series of bools. 
            # 2000-01-01 00:13:00    False
            # 2000-01-01 00:14:00     True
            # 2000-01-01 00:15:00    False
            # then [1:] removes the first bar becasue it will always be False
            # 2000-01-01 00:14:00     True
            # 2000-01-01 00:15:00    False
            # then mean() returns the mean of the bools.
            return (window.high < window.high.shift())[1:].mean() * 100
    

        if self.ls == 'SHORT' and len(window) > 2: # if there are more than 2 bars in the pullback from the low
            return (window.low > window.low.shift())[1:].mean() * 100
    
        return 0.0


@dataclass
class PBAllSameColour(Signals):
    """Retruns the ration of how many bars are of the same colour as the longshort direction. 
    This class is the check the pullback is all in the same direction. 
    eg if long then all the bars in the pullback are red.
    eg if short then all the bars in the pullback are green.
    """
    name: str = 'PB_ASClr'
    pointCol: str = ''
    toCol  : str = ''
    atrCol: str = ''
    atrMultiple: float = 1.0 # minimum number of ATRs required between pointCol low and toCol
    
    
    def __post_init__(self):
        self.name = f"Sig{self.ls[0]}_{self.name}"
        self.names = [self.name]

    # def run(self,longshort:str='', fromHP:pd.DataFrame=pd.DataFrame(), fromLP:pd.DataFrame=pd.DataFrame(), **kwargs):
    def _compute_row(self, df:pd.DataFrame=pd.DataFrame()):
        window  = get_valid_pb(
            ls=self.ls, 
            df=df, 
            pointCol=self.pointCol, 
            toCol=self.toCol, 
            atrCol=self.atrCol, 
            minLen=3, 
            atrMultiple=1)
        
        if window is None:
            return 0.0
        
        total_bars = len(window) -1
        if len(window) > 2:
            if self.ls == 'LONG':
                same_colour_bars = len(window[window['close'] < window['open']]) # red bars
                return  (same_colour_bars / total_bars) * 100

            if self.ls == 'SHORT':
                same_colour_bars = len(window[window['close'] > window['open']]) # green bars
                return  (same_colour_bars / total_bars) * 100

        return 0.0


#£ Done
@dataclass
class PBOverlap(Signals):
    """Computes the overlap as % from this high to prev low (BULL, so pullback is down) as ratio to prev bar range .
    Then gets the mean % of all overlaps. Works very well to give a good guide for a smooth pullback
    if the mean % is 50%. so the nearer to 50% the better """
    name  : str = 'Olap'
    pointCol: str = ''
    toCol  : str = ''
    atrCol: str = ''
    atrMultiple: float = 1.0 # minimum number of ATRs required between pointCol low and toCol
    

    def __post_init__(self):
        self.name = f"Sig{self.ls[0]}_{self.name}"
        self.names = [self.name]

    def _compute_row(self, df:pd.DataFrame):
        window  = get_valid_pb(
            ls=self.ls, 
            df=df, 
            pointCol=self.pointCol, 
            toCol=self.toCol, 
            atrCol=self.atrCol, 
            minLen=3, 
            atrMultiple=1)
        
        if window is None:
            return 0.0
            
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
        return max(score, 0)


#£ Done
@dataclass
class Trace(Signals):
    name      : str = 'Rtcmt'
    usePoints : bool = True # if True then use points else use fromPriceCol and toPriceCol
    fromCol   : str = ''
    toCol     : str = ''
    optimalRtc: float = None # optimal retracement eg if 50 then 50% is the best retracement
  
    def __post_init__(self):
        self.name = f"Sig{self.ls[0]}_{self.name}"
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


# --------------------------------------------------------------
# ------- R E V E R S A L   S I G N A L S ----------------------
# --------------------------------------------------------------

#£ Done
@dataclass
class Tail(Signals):
    name: str = 'Tail'    
    tailExceedsNthBarsAgo: int = 0

    def _compute_row(self, df:pd.DataFrame):
        """Top Tail / Bottom Tail is the ratio of the top and the bottom. 
        The top is low of the body to the high.
        The bottom is the high of the body to the low. 
              ___
          | 
         _|_  Top   ___
        |   |  
        |   |
        |___| ___   Bottom
          |
          |         ___
        """   
        
        if len(df) < self.tailExceedsNthBarsAgo+2:
            return 0
    
        x = df.iloc[-1] # get the last bar 

        # top of body is the max of open and close
        top = max(x.open, x.close)

        # bottom of body is the min of open and close
        bottom = min(x.open, x.close)

        # body length is the difference between open and close
        body_len = abs(x.open - x.close)

        # get the length of the top of body to low and bottom of body to high
        # 0.05 is 5% of body length. This is to avoid div by zero and to give a 
        # min value to keep extream values in check as this is used to divide by.
        top_len    = max(x.high - bottom, body_len * 0.05) 
        bottom_len = max(top - x.low, body_len * 0.05)

        # ratio of top to bottom
        # 25 is to help scale the value to be between 0 and 100
        # 100 is the best value as it means the top is 4 times the bottom
        # lots of visula checks have been done to make sure 25 is a good value
        
        if self.ls == 'LONG': 
            # get the lowest low of the last n bars
            lowest  = min(df.low.iloc[-self.tailExceedsNthBarsAgo-1:-2])
            if top_len > 0 and df.low.iat[-1] <= lowest: 
                return  round(bottom_len / top_len *25, 2) # gives higher number the longer the bottom is (signals a bullish reversal)
  
        else: 
            # get the highest high of the last n bars
            highest = max(df.high.iloc[-self.tailExceedsNthBarsAgo-1:-2])
            if bottom_len > 0 and df.high.iat[-1] >= highest:
                return round(top_len / bottom_len *25, 2) # gives higher number the longer the top is (signals a bearish reversal)

        return 0


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

#$ ------- Gaps ---------------
# Used for checking if this works correctly. 
@dataclass
class IsGappedOverPivot(Signals):
    """
    Simple signal class to verify gap-over-pivot conditions.
    Returns 1.0 when a pivot is gapped over, 0.0 otherwise.
    """
    name: str = 'GapPiv'
    pointCol: str = 'pivot'

    def _compute_row(self, df: pd.DataFrame) -> float:
        # Get data up to current bar
        return is_gap_pivot_crossover(df, self.pointCol, self.ls)

    

#£ Done
@dataclass
class GappedPivots(Signals):
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
    runType: str = 'pct'
    span: int = 20

    def __post_init__(self):
        self.name = f"Sig{self.ls[0]}_{self.name}"
        self.names = [self.name]

    def _compute_row(self, df: pd.DataFrame) -> tuple[int, int]:
        """Count pivots that fall within the gap range."""

        if not is_gap_pivot_crossover(df, self.pointCol, self.ls):
            return 0
        
        pivots = df[self.pointCol].iloc[:-2].dropna()
        
        if len(pivots) == 0:
            return 0

        current_open = df['open'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        
        # Only count pivots that fall within the gap range
        if self.ls == 'LONG' and current_open > prev_close:
            return sum((pivot > prev_close) & (pivot < current_open) for pivot in pivots)
        elif self.ls == 'SHORT' and current_open < prev_close:
            return sum((pivot < prev_close) & (pivot > current_open) for pivot in pivots)

        return 0



#£ Done
@dataclass
class GappedRetracement(Signals):
    name: str = 'GRtc'
    atrCol: str = ''
    pointCol: str = ''

    def _compute_row(self, df: pd.DataFrame) -> float:
        """
        Measures the shock value of a gap pivot crossover by calculating the retracement 
        relative to ATR. A larger retracement indicates a more significant gap move.
        """
        # First check if we have a valid gap pivot crossover
        has_crossover = is_gap_pivot_crossover(df, self.pointCol, self.ls)
        if not has_crossover:
            return 0.0

        try:
            # Get previous close and most recent pivot
            prev_close = df['close'].iloc[-2]
            prior_pivots = df.iloc[:-1][self.pointCol].dropna()
            
            if len(prior_pivots) == 0:
                return 0.0
                
            pivot_value = prior_pivots.iloc[-1]
            atr = df[self.atrCol].iloc[-1]
            
            # Calculate retracement size relative to ATR
            if self.ls == 'LONG':
                # For longs: distance from pivot to previous close
                retracement = pivot_value - prev_close
            else:
                # For shorts: distance from previous close to pivot
                retracement = prev_close - pivot_value
                
            # Convert to percentage of ATR
            shock_value = (retracement / atr) * 100
            return shock_value
                
        except (IndexError, KeyError):
            return 0.0




#£Done
@dataclass
class GappedPastPivot(Signals):
    """
    Assess the quality of gaps past pivot points by evaluating gap size relative to ATR.
    Uses a diminishing returns approach for oversized gaps.
    """
    name: str = 'GPP'
    atrCol: str = ''
    pointCol: str = ''
    maxAtrMultiple: int = 10  # Number of ATR past pivot before score starts diminishing

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
        if not is_gap_pivot_crossover(df, self.pointCol, self.ls):
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



#£Done
@dataclass
class GapSize(Signals):
    """Measure the size of price gaps relative to the previous close."""
    name: str = 'GSiz'
    atrCol: str = ''
    pointCol: str = ''  # Column name for pivot points
        
    def _compute_row(self, df: pd.DataFrame) -> float:
        """
        Calculate the gap score relative to ATR for a confirmed pivot crossover.

        Returns:
        --------
        float
            Gap score as a percentage of ATR
        """
        # First check if there's a valid gap over pivot
        if not is_gap_pivot_crossover(df, self.pointCol, self.ls):
            return 0.0
        
        current_open = df['open'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        current_atr = df[self.atrCol].iloc[-1]
        
        if current_atr == 0:
            return 0.0
            
        if self.ls == 'LONG':
            gap_points = current_open - prev_close
        else:  # SHORT
            gap_points = prev_close - current_open
            
        return gap_points / current_atr * 100
        


# -----------------------------------------------------------------------
# ---- V O L U M E ------------------------------------------------------
# -----------------------------------------------------------------------
#£ Done
@dataclass
class VolumeSpike(Signals):
    """
    Detects a volume spike in a pandas dataframe with a 'volume' column.
    Returns the percent change between the current volume and the rolling average volume over 'volMA' periods.
    """
    name         : str = 'VolSpike'
    volMACol     : str = ''

    def _compute_row(self, df:pd.DataFrame):
        current_volume = df['volume'].iat[-1]
        vol_ma = df[self.volMACol].iat[-1]
        
        # Calculate the percent change between the current volume and the rolling average volume
        return ((current_volume - vol_ma) / vol_ma) * 100


@dataclass
class VolumeROC(Signals):
    """
    Calculates the Rate of Change (ROC) of volume between consecutive bars.
    Returns the acceleration of volume changes over the lookback period.
    """
    name: str = 'VolROC'

    
    def _compute_row(self, df: pd.DataFrame):
        """
        Calculate volume ROC acceleration over the lookback period.
        """
        # Get the volume series for the lookback period
        vol1 = df['volume'].iat[-1]
        vol2 = df['volume'].iat[-2]
        
        # Calculate ROC
        return ((vol1 - vol2) / vol2) * 100


#$ ------- Price ---------------
@dataclass
class RoomToMove(Signals):
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
    name : str = 'RTM'
    tgetCol : str = ''
    atrCol: str = ''

    def __post_init__(self):
        self.name = f"Sig{self.ls[0]}_{self.name}{self.tgetCol}"
        self.names = [self.name]

    def _compute_row(self, df:pd.DataFrame=pd.DataFrame(), **kwargs):

        val = 0

        if len(df) > 1:
            if self.ls == 'LONG':
                if df[self.tgetCol].iat[-1] is not None:
                    return (df[self.tgetCol].iat[-1] - df.close.iat[-1]) / df[self.atrCol].iat[-1]
                
                # if there is no target then return 2. meaning it has room to move so give an arbitrary high number
                else:
                    return 2

            elif self.ls == 'SHORT':
                if df[self.tgetCol].iat[-1] is not None:
                    return (df.close.iat[-1] - df[self.tgetCol].iat[-1]) / df[self.atrCol].iat[-1]

                # if there is no target then return 2. meaning it has room to move so give an arbitrary high number
                else:
                    return 2
   
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

    def __post_init__(self):
        """Initialize with empty signal names - will be populated during setup."""
        self.names = []
        self.source_columns = []
        self.column_mapping = {}

    def setup_columns(self, df: pd.DataFrame):
        """Set up column mappings and signal names."""
        self.source_columns = [col for col in df.columns if col.startswith(self.columnStartsWith)]
        self.names = []
        self.column_mapping = {}
        
        for col in self.source_columns:
            signal_name = f"{self.name}_{col}"
            self.names.append(signal_name)
            self.column_mapping[col] = signal_name

    def get_score(self, val):
        """Normalize values efficiently using vectorized operations."""
        def normalize_vec(x):
                normalized = (x - self.normRange[0]) / (self.normRange[1] - self.normRange[0]) * 100
                clamped = np.clip(normalized, 0, 100)  # Clamp the values between 0 and 100
                return np.round(clamped, 2)  # Added rounding to match single-value function
                
        if isinstance(val, (pd.Series, pd.DataFrame)):
            return val.apply(normalize_vec)
        return normalize_vec(val)

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
            return pd.DataFrame(0, index=[df.index[-1]], columns=self.names)
        
        lookBack = min(self.lookBack, len(df))

        # Get the window we need to process
        window = df.iloc[-lookBack:]
        
        # Compute signals for the entire window at once
        signals = self.compute_signals(window)
        
        # Ensure we have all expected columns
        for name in self.names:
            if name not in signals.columns:
                signals[name] = 0
                
        # Normalize the results
        for col in signals.columns:
            signals[col] = self.get_score(signals[col])

        return signals
#$ -------  Cosnsolidation and Trend Signals ---------------



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



# --------------------------------------------------------------------
# ----- E V E N T   S I G N A L S ------------------------------------
# --------------------------------------------------------------------
@dataclass
class Breaks(Signals):
    """Checks if price crosses above/below a metric"""
    price_column: str = ''
    direction: str = '' # 'above' or 'below'
    metric_column: str = ''

    def __post_init__(self):
        self.name = f"BRK_{self.price_column}_{self.direction[:2]}_{self.metric_column}"
        self.names = [self.name]

    def _compute_row(self, df: pd.DataFrame) -> pd.DataFrame:
        
        curr_price = df[self.price_column].iloc[-1]
        prev_price = df[self.price_column].iloc[-2]
        curr_metric = df[self.metric_column].iloc[-1]
        prev_metric = df[self.metric_column].iloc[-2]

        if self.direction == 'above':
            return (prev_price <= prev_metric) & (curr_price > curr_metric)
        elif self.direction == 'below':
            return (prev_price >= prev_metric) & (curr_price < curr_metric)
        
        else:
            raise ValueError("Direction must be 'above' or 'below'")
  
        


@dataclass
class AboveBelow(Signals):
    """Checks if price is above/below a metric"""
    value: str | float = ''
    direction: str  = ''# 'above' or 'below'
    metric_column: str = ''

    def __post_init__(self):
        self.name = f"AB_{self.value}_{self.direction[:2]}_{self.metric_column}"
        self.names = [self.name]

    def _compute_row(self, df: pd.DataFrame) -> pd.DataFrame:
        value =  df[self.value].iat[-1] if isinstance(self.value, str) else self.value
        metric = df[self.metric_column].iat[-1] if isinstance(self.metric_column, str) else self.metric_column

        if self.direction == 'above':
            return  value > metric
        elif self.direction == 'below':
            return  value < metric
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
class Strategy:
    name: str = ''
    ls: str = 'LONG'
    lookBack: int = 20

    def __post_init__(self):
        self.name = f'Stgy_{self.name}'
        self.event_tracker = {}
        self.steps = set()
        self.current_step = 1
        self.triggered_events = set()
        
        # Track conditions by type
        self.reset_conditions = set()
        self.reset_after_validation_conditions = set()
        self.true_counts = {}  # Track periods true for each reset-after condition
        
        # Initialize names list - will be populated as conditions are added
        self.names = []

    def add_event(self, step: int, name: str, valToCheck: Union[str, float], checkIf: str, colThreshold: str):
        """Add an event that happens once and triggers a validation."""
        condition_name = f'{self.name}_{step}-{name}'
        self.event_tracker[condition_name] = {
            'step': step,
            'type': 'event',
            'status': 'pending',
            'valToCheck': valToCheck,
            'checkIf': checkIf,
            'colThreshold': colThreshold,
            'isTrue': 0
        }
        self.steps.add(step)
        self.names.append(condition_name)

    def add_validation(self, step: int, name: str, valToCheck: Union[str, float], checkIf: str, colThreshold: str):
        """Add a validation that is ongoing and reversible."""
        condition_name = f'{self.name}_{step}-{name}'
        self.event_tracker[condition_name] = {
            'step': step,
            'type': 'validation',
            'status': 'pending',
            'valToCheck': valToCheck,
            'checkIf': checkIf,
            'colThreshold': colThreshold,
            'isTrue': 0
        }
        self.steps.add(step)
        self.names.append(condition_name)

    def add_reset(self, name: str, valToCheck: Union[str, float], checkIf: str, colThreshold: str):
        """Add a reset condition that cancels the strategy if triggered."""
        condition_name = f'{self.name}_R_{name}'
        self.event_tracker[condition_name] = {
            'type': 'reset',
            'valToCheck': valToCheck,
            'checkIf': checkIf,
            'colThreshold': colThreshold,
            'isTrue': 0
        }
        self.reset_conditions.add(condition_name)
        self.names.append(condition_name)

    def add_reset_after_validation(self, name: str, valToCheck: Union[str, float], 
                                 checkIf: str, colThreshold: str, periods: int = 5):
        """Add a condition that resets the strategy after being valid for N periods."""
        condition_name = f'{self.name}_{name}'
        self.event_tracker[condition_name] = {
            'type': 'reset_after_validation',
            'valToCheck': valToCheck,
            'checkIf': checkIf,
            'colThreshold': colThreshold,
            'isTrue': 0,
            'periods': periods
        }
        self.reset_after_validation_conditions.add(condition_name)
        self.true_counts[condition_name] = 0
        self.names.append(condition_name)

    def evaluate_condition(self, row: pd.Series, condition: dict) -> bool:
        """Evaluate a single condition against a row of data."""
        if 'valToCheck' not in condition or 'colThreshold' not in row:
            return False
        val = condition['valToCheck']
        threshold = condition['colThreshold']
        
        actual_valToCheck = float(row[val]) if isinstance(val, str) else val
        actual_threshold = float(row[threshold]) if isinstance(threshold, str) else threshold
        # print(f"{row.name} - valToCheck: {actual_valToCheck} {type(actual_valToCheck)} {condition['checkIf']} {actual_threshold} = {actual_valToCheck > actual_threshold}" )
    
        if condition['checkIf'] == '>':
            return actual_valToCheck > actual_threshold
        elif condition['checkIf'] == '<':
            return actual_valToCheck < actual_threshold
        elif condition['checkIf'] == '>=':
            return actual_valToCheck >= actual_threshold
        elif condition['checkIf'] == '<=':
            return actual_valToCheck <= actual_threshold
        return False

    def reset_strategy(self):
        """Reset all strategy state."""
        self.current_step = 1
        self.triggered_events.clear()
        for event in self.event_tracker.values():
            event['isTrue'] = 0
            event['status'] = 'pending'
        for condition in self.true_counts:
            self.true_counts[condition] = 0

    def check_step_completion(self, step: int) -> float:
        """Calculate completion percentage for a given step."""
        step_items = {k: v for k, v in self.event_tracker.items() 
                     if v.get('step') == step}
        if not step_items:
            return 0.0
        
        completed = 0
        for name, condition in step_items.items():
            if condition['type'] == 'event' and name in self.triggered_events:
                completed += 1
            elif condition['type'] == 'validation' and condition['isTrue'] == 1:
                completed += 1
                
        return completed / len(step_items) if step_items else 0.0

    def _compute_row(self, df: pd.DataFrame) -> pd.Series:
        """Compute strategy values for the current row."""
        if len(df) < 2:
            return pd.Series(0, index=self.names)
            
        current_row = df.iloc[-1]
        results = {}
        
        # Check reset conditions first
        for name in self.reset_conditions:
            condition = self.event_tracker[name]
            if self.evaluate_condition(current_row, condition):
                condition['isTrue'] = 1
                results[name] = 100  # Signal the reset condition that triggered
                self.reset_strategy()
                return pd.Series(results)
            condition['isTrue'] = 0
            results[name] = 0
        
        # Process events and validations for current step
        step_items = {k: v for k, v in self.event_tracker.items() 
                     if v.get('step', 0) == self.current_step}
        
        # Update events (permanent until reset)
        for name, condition in step_items.items():
            if condition['type'] == 'event':
                if name in self.triggered_events:
                    condition['isTrue'] = 1
                    results[name] = 100
                elif self.evaluate_condition(current_row, condition):
                    # print(f"{current_row.name} Event Triggered: {name}")
                    self.triggered_events.add(name)
                    condition['isTrue'] = 1
                    condition['status'] = 'completed'
                    results[name] = 100
                else:
                    results[name] = 0

        # Update validations (can change each row)
        for name, condition in step_items.items():
            if condition['type'] == 'validation':
                condition['isTrue'] = 1 if self.evaluate_condition(current_row, condition) else 0
                results[name] = 100 if condition['isTrue'] else 0
        
        # Calculate step completions
        for step in sorted(self.steps):
            completion = self.check_step_completion(step)
            results[f'{self.name}_step{step}'] = completion * 100
            
            # Progress to next step if current is complete
            if step == self.current_step and completion == 1.0:
                if step < max(self.steps):
                    self.current_step += 1
        
        # Calculate total completion
        if self.steps:
            total_completion = sum(self.check_step_completion(step) 
                                 for step in range(1, self.current_step + 1)) / len(self.steps)
            results[f'{self.name}_total'] = total_completion * 100
        else:
            results[f'{self.name}_total'] = 0
            
        # Handle reset after validation conditions
        for name in self.reset_after_validation_conditions:
            condition = self.event_tracker[name]
            if self.evaluate_condition(current_row, condition):
                condition['isTrue'] = 1
                self.true_counts[name] += 1
                if self.true_counts[name] >= condition['periods']:
                    results[name] = 100  # Signal which reset-after condition triggered
                    self.reset_strategy()
                    return pd.Series(results)
            else:
                condition['isTrue'] = 0
                self.true_counts[name] = 0
            results[name] = 100 if condition['isTrue'] else 0
                
        return pd.Series(results)
    

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process strategy over lookback period maintaining proper state progression."""
        if len(df) < 10:
            return pd.DataFrame(0, index=df.index, columns=self.names)

        # Add step and total columns if not already in names
        for step in sorted(self.steps):
            step_col = f'{self.name}_step{step}'
            if step_col not in self.names:
                self.names.append(step_col)
        total_col = f'{self.name}_total'
        if total_col not in self.names:
            self.names.append(total_col)

        # Initialize results DataFrame
        results = pd.DataFrame(0, index=df.index, columns=self.names)
        
        # Reset strategy state at start of run
        self.reset_strategy()
        
        # Process each row in the lookback period
        lookBack = min(self.lookBack, len(df)) - 1
        lookback_indices = df.index[-lookBack:]
        
        for i, idx in enumerate(lookback_indices):
            current_window = df.loc[:idx]
            if current_window.empty:
                continue
                
            row_results = self._compute_row(current_window)
            
            # Ensure row_results has all the columns in the correct order
            row_results = row_results.reindex(results.columns, fill_value=0)
            
            # Use the index location for setting values
            results.loc[idx] = row_results
            
        return results