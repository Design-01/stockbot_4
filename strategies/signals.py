from dataclasses import dataclass
from typing import Tuple, Any

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Union
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




#£ Done
@dataclass
class Signals(ABC):
    name: str = ''
    normRange: Tuple[int, int] = (0, 100)
    ls: str = 'LONG'
    lookBack: int = 20

    def __post_init__(self):
        self.name = f"Sig{self.ls[0]}_{self.name}"
        self.names = [self.name]

    def get_score(self, val):
        if isinstance(val, pd.Series):
            # Apply the function to each element in the series
            return val.apply(lambda x: 0 if x == 0 else normalize(x, self.normRange[0], self.normRange[1]))
        else:
            # Handle single value
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
    
    def return_series(self, index:pd.DatetimeIndex, val:float):
        if isinstance(val, pd.Series):
            val.name = self.name
            return val
        return pd.Series(index=[index], data=val, name=self.name)
    
    @abstractmethod
    def _compute_row(self, df: pd.DataFrame) -> float:
        """This method is to compute each row in the lookback period."""
        pass
                        
    # @abstractmethod
    def run(self, df: pd.DataFrame = pd.DataFrame()) -> pd.Series:
        """Generate signal scores for the lookback period."""
        if len(df) <= self.lookBack:
            return self.return_series(df.index[-1], self.get_score(0))

        # this then gets populated with the results of the computation
        result_series = pd.Series(0.0, index=df.index[-self.lookBack:])
        
        for i in range(self.lookBack):
            current_idx = -(self.lookBack - i)
            if abs(current_idx) >= len(df) :
                continue
            
            #! This is where the computation is done. 
            #! Each sub class must have a _compute_row method that does the computation
            current_window = df.iloc[:current_idx+1]
            if current_window.empty:
                continue
            val = self._compute_row(current_window)
            result_series.iloc[i] = float(val)

        return self.return_series(df.index[-self.lookBack:], self.get_score(result_series))


@dataclass
class Score(Signals):
    cols: List[str] = field(default_factory=list)
    weight: float = 1.0
    scoreType:str = 'mean' # 'mean', 'sum', 'max', 'min'
    
    def _compute_row(self, df: pd.DataFrame) -> float:
        """This method is to compute each row in the lookback period."""
        if len(self.cols) == 0:
            return 0.0
        
        val = 0.0

        last_row = df[self.cols].iloc[-1]

        if self.scoreType == 'mean':
            val = last_row.mean()
        elif self.scoreType == 'sum':
            val = last_row.sum()
        elif self.scoreType == 'max':
            val = last_row.max()
        elif self.scoreType == 'min':
            val = last_row.min()
        else:
            val = 0.0

        score = self.get_score(val) * self.weight
        return score

      
import random

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

@dataclass
class PullbackBounce(Signals):
    name: str = 'PBB'
    pointCol: str = ''
    supResCol: str = ''

    def _compute_row(self, df: pd.DataFrame) -> float:
        """
        Gets a Signal at a high Point or a low Point by looking back 
        to the nth most recent point and getting the corresponding signal.
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
    pullbackCol     : str = '' # col which the price pullback is near to
    maxCol          : str = '' # col which the price is near to : used for get window
    minCol          : str = '' # col which the price is near to : used for get window
    optimalRetracement: float = 99


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
        
        score = optimal_retracement - (retracement - optimal_retracement) * 10
        return max(score, 0)

    def _compute_row(self, df:pd.DataFrame):
        """How near is the priceNow to the MA from the pullback high at start (bull case) to the low at the end """
        #! even though the lower of two recent bars is chosen as the low point the MA is always the current bar. so the distance will change evethough it may be compared to the same low point.
        if len(df) < 3:
            return 0
        
        w0 = self.get_window(df, self.ls, 0)
        if w0.empty or len(w0) < 2:
            return 0
        
        priceNow = df.close.iat[-1]

        if self.ls == 'LONG':
            val = trace(w0.high.iat[0], df[self.pullbackCol].iat[-1], priceNow)
            return self.pb_score(val)

        elif self.ls == 'SHORT':
            val = trace(w0.low.iat[0], df[self.pullbackCol].iat[-1], priceNow)
            return self.pb_score(val)

        return 0


#£ Done
@dataclass
class Overlap(Signals):
    name  : str = 'Olap'
    maxCol: str = '' # col which the price is near to : used for get window
    minCol: str = '' # col which the price is near to : used for get window

    def _compute_row(self, df:pd.DataFrame):
        """Computes the overlap as % from this high to prev low (BULL, so pullback is down) as ratio to prev bar range .
            Then gets the mean % of all overlaps. Works very well to give a good guide for a smooth pullback
            if the mean % is 50%. so the nearer to 50% the better """

        # from recent high point (fromHP) or from recent low point (fromLP) to the end of the df
        w0 = self.get_window(df, self.ls, 0)
        
        if  len(w0) <= 2:
            return 0
        
        # check if in downward pullback the high of the current bar is lower than the low of the prior bar etc
        if self.ls == 'LONG' and not w0.high.iat[-3] > w0.high.iat[-1]: 
            return 0
        if self.ls == 'SHORT'and not w0.low.iat[-3]  < w0.low.iat[-1]:
            return 0
            
        prev = w0.shift(1).copy()
        olap          = w0.high - prev.low if self.ls == 'LONG' else prev.high - w0.low
        prev_rng      = abs(prev.high - prev.low)
        olap_pct      = olap / prev_rng 
        olap_pct_mean = olap_pct.mean()

        # 150 is to scale the score to be between 0 and 100. playing around with this number will change the sensitivity of the score
        # 100 is the best score as it means the olap_pct is 50% which is the best
        # calculate score based on olap_pct
        optimal_olap_pct = 0.5
        score = 100 - abs(olap_pct_mean - optimal_olap_pct) * 150 
        return max(score, 0)



#!!! --------->>>  Not implemented yet.  Needs to be checked  <<<-----------

#£ Done
@dataclass
class Trace(Signals):
    usePoints    : bool = True # if True then use points else use fromPriceCol and toPriceCol
    fromLongCol  : str = ''
    fromShortCol : str = ''
    toLongCol    : str = ''
    toShortCol   : str = ''
    optimalRtc: float = None # optimal retracement eg if 50 then 50% is the best retracement
  
    def __post_init__(self):
        self.columns = [self.colname] + ['fromPrice', 'toPrice', 'priceNow', 'fromIdx', 'toIdx']  

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
    def run(self, longshort:str='', df:pd.DataFrame=pd.DataFrame(), longW1:pd.DataFrame=pd.DataFrame(), shortW1:pd.DataFrame=pd.DataFrame(), **kwargs):
        """trace high 1 ago to low 1 ago and compare  """

        if not df.empty:
            fromPrice = 0
            toPrice = 0
            # long W1 is the move up from the low of W1 to the high of W1 which is the most recent HP point (the start of the pullback represented by fromHP) 
            if longshort == 'LONG': #and not longW1.empty:
                # fromPrice = longW1.high.iat[-1] 
                # toPrice   = longW1.low.iat[0]
                fromPrice = df[self.fromLongCol].iat[-1]
                toPrice   = df[self.toLongCol].iat[-1]

            # short W1 is the move down from the high of W1 to the low of W1 which is the most recent LP point (the start of the pullback represented by fromLP)
            elif longshort == 'SHORT':  #and not shortW1.empty:
                # fromPrice = shortW1.low.iat[-1]
                # toPrice   = shortW1.high.iat[0]
                fromPrice = df[self.fromShortCol].iat[-1]
                toPrice   = df[self.toShortCol].iat[-1]

            priceNow  = df.close.iat[-1]

            
            # avoid div by zero
            if fromPrice != toPrice:
                t = trace(fromPrice, toPrice, priceNow)
                if self.optimalRtc:
                    self.val = self.compute_from_mid_trace(t, self.optimalRtc)
                else: 
                    self.val = t
            # display((f'{longshort} {df.index[-1]}-- fromPrice: {fromPrice}, toPrice: {toPrice}, priceNow: {priceNow}, trace: {self.val}'))
        else:
            self.val = 0

    def reset(self):
        self.val = 0

#£ Done
@dataclass
class HigherLowsLowerHighs(Signals):
    
    def __post_init__(self):
        self.columns = [self.colname]   
   

    #$ **kwargs is used to allow any signal arguments to be passed to any run method. This so that the same run method can be used when looping through signals.
    def run(self, longshort:str='', fromHP:pd.DataFrame=pd.DataFrame(), fromLP:pd.DataFrame=pd.DataFrame(), **kwargs):
        """Computes the % of bars that have a lower highs (BULL pullback, so downward)
        Vice versa for BEAR case. So this is only for pullbacks not overall trends. """

        if longshort == 'LONG' and len(fromHP) > 2: # if there are more than 2 bars in the pullback from the high
            # eg fhp.high < fhp.high.shift() retruns a series of bools. 
            # 2000-01-01 00:13:00    False
            # 2000-01-01 00:14:00     True
            # 2000-01-01 00:15:00    False
            # then [1:] removes the first bar becasue it will always be False
            # 2000-01-01 00:14:00     True
            # 2000-01-01 00:15:00    False
            # then mean() returns the mean of the bools.
            self.val = (fromHP.high < fromHP.high.shift())[1:].mean()

        elif longshort == 'SHORT' and len(fromLP) > 2:
            self.val = (fromLP.low > fromLP.low.shift())[1:].mean()
        
        else: 
            self.val = 0
            
        self.val *= 100

    


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


@dataclass
class AllSameColour(Signals):
    """Retruns the ration of how many bars are of the same colour as the longshort direction. 
    This class is the check the pullback is all in the same direction. 
    eg if long then all the bars in the pullback are red.
    eg if short then all the bars in the pullback are green.
    """
    
    def __post_init__(self):
        self.columns = [self.colname]     
        self.df = pd.DataFrame()

    # def run(self,longshort:str='', fromHP:pd.DataFrame=pd.DataFrame(), fromLP:pd.DataFrame=pd.DataFrame(), **kwargs):
    def run(self,longshort:str='', **kwargs):
        df = kwargs.get('fromHP') if longshort == 'LONG' else kwargs.get('fromLP')
        
        total_bars = len(df)
        if len(df) > 2:
            if longshort == 'LONG':
                same_colour_bars = len(df[df['close'] < df['open']])
                self.val =  (same_colour_bars / total_bars) * 100

            if longshort == 'SHORT':
                same_colour_bars = len(df[df['close'] > df['open']])
                self.val =  (same_colour_bars / total_bars) * 100

        else:
            self.val = 0

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
    name: str = 'GPiv'
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


import math

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
        



#$ ------- Volume ---------------
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


#$ -------  Reversal Signals ---------------



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









