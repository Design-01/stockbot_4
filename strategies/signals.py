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


#£ Done
@dataclass
class Signals(ABC):
    name: str = ''
    maxCol: str = 'high'
    minCol: str = 'low'
    normRange: Tuple[int, int] = (0, 100)



    def get_score(self, val:float):
        # test for NaN and set to 0 if so
        if val == 0: return 0
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
                        
    @abstractmethod
    def run(self, longshort:str='', df:pd.DataFrame=pd.DataFrame()):
        pass


#£ Done
@dataclass
class Tail(Signals):
    name: str = 'Tail'    
    tailExceedsNthBarsAgo: int = 0

    def __post_init__(self):
        self.x = None
        if self.tailExceedsNthBarsAgo < 2 :
            self.tailExceedsNthBarsAgo = 3


    #$ **kwargs is used to allow any signal arguments to be passed to any run method. This so that the same run method can be used when looping through signals.
    def run(self, longshort:str='', df:pd.DataFrame=pd.DataFrame()):
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
    
        self.x = df.iloc[-1] # get the last bar 

        # top of body is the max of open and close
        top = max(self.x.open, self.x.close)

        # bottom of body is the min of open and close
        bottom = min(self.x.open, self.x.close)

        # body length is the difference between open and close
        body_len = abs(self.x.open - self.x.close)

        # get the length of the top of body to low and bottom of body to high
        # 0.05 is 5% of body length. This is to avoid div by zero and to give a 
        # min value to keep extream values in check as this is used to divide by.
        top_len    = max(self.x.high - bottom, body_len * 0.05) 
        bottom_len = max(top - self.x.low, body_len * 0.05)

        # ratio of top to bottom
        # 25 is to help scale the value to be between 0 and 100
        # 100 is the best value as it means the top is 4 times the bottom
        # lots of visula checks have been done to make sure 25 is a good value
        
        if longshort == 'LONG': 
            # get the lowest low of the last n bars
            lowest  = min(df.low.iloc[-self.tailExceedsNthBarsAgo-1:-2])
            if top_len > 0 and df.low.iat[-1] <= lowest: 
                return round(bottom_len / top_len *25, 2) # gives higher number the longer the bottom is (signals a bullish reversal)

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
    maxCol: str = 'high'
    minCol: str = 'low'
    longPullbackCol     : str = '' # col which the price pullback is near to
    shortPullbackCol    : str = '' # col which the price pullback is near to


    def get_score(self, retracement):
        """Calculate score based on retracement. Optimal retracement is 95%.
        Score decreases by 10 for every 1% away from optimal retracement above 95%.
        eg 91% retracement will have a score of 85. 92% retracement will have a score of 80.
        eg 80% retracement will have a score of 80. 70% retracement will have a score of 70.
        eg 50% retracement will have a score of 50. 0% retracement will have a score of 0.
        """
        optimal_retracement = 99 # Was 95% but changed to 99%
        if retracement <= optimal_retracement:
            return max(retracement, 0)
        
        score = optimal_retracement - (retracement - optimal_retracement) * 10
        return max(score, 0)


    #$ **kwargs is used to allow any signal arguments to be passed to any run method. This so that the same run method can be used when looping through signals.
    def run(self, longshort:str='', df:pd.DataFrame=pd.DataFrame()):
        """How near is the priceNow to the MA from the pullback high at start (bull case) to the low at the end """

        #! even though the lower of two recent bars is chosen as the low point the MA is always the current bar. so the distance will change evethough it may be compared to the same low point.
        
        if len(df) < 3:
            return 0
        
        w0 = self.get_window(df, longshort, 0)
        if w0.empty or len(w0) < 2:
            return 0
        
        priceNow = df.close.iat[-1]

        if longshort == 'LONG':
            return self.get_score(trace(w0.high.iat[0], df[self.longPullbackCol].iat[-1], priceNow))
                   
        elif longshort == 'SHORT':
            return self.get_score(trace(w0.low.iat[0], df[self.shortPullbackCol].iat[-1], priceNow))


#£ Done
@dataclass
class Overlap(Signals):
    name: str = 'Olap'

    #$ **kwargs is used to allow any signal arguments to be passed to any run method. This so that the same run method can be used when looping through signals.
    def run(self, longshort:str='', df:pd.DataFrame=pd.DataFrame(), **kwargs):
        """Computes the overlap as % from this high to prev low (BULL, so pullback is down) as ratio to prev bar range .
            Then gets the mean % of all overlaps. Works very well to give a good guide for a smooth pullback
            if the mean % is 50%. so the nearer to 50% the better """

        # from recent high point (fromHP) or from recent low point (fromLP) to the end of the df
        w0 = self.get_window(df, longshort, 0)
        
        if  len(w0) <= 2:
            return 0
        
        # check if in downward pullback the high of the current bar is lower than the low of the prior bar etc
        if longshort == 'LONG' and not w0.high.iat[-3] > w0.high.iat[-1]: 
            return 0
        if longshort == 'SHORT'and not w0.low.iat[-3]  < w0.low.iat[-1]:
            return 0
            
        prev = w0.shift(1).copy()
        olap          = w0.high - prev.low if longshort == 'LONG' else prev.high - w0.low
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
                trace = sbu.trace(fromPrice, toPrice, priceNow)
                if self.optimalRtc:
                    self.val = self.compute_from_mid_trace(trace, self.optimalRtc)
                else: 
                    self.val = trace
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
#£ Done
@dataclass
class GappedPivots(Signals):
    """Computes the number of pivots that have been gapped over as a ratio to the total number of pivots.
    This means that as the candels progress the ratio will decrease as more pivots appear on the chart over. 
    This gives more potency to earlier gapped pivots."""
    colname : str = ''
    hpCol   : str = ''
    lpCol   : str = ''
    mustGap : bool = False 
    runType : str = 'pct' 
    span    : int = 20 

    """Args:
        colname (str) : The name of the column to store the signal values in.
        hpCol (str)   : The name of the column containing the high pivots.
        lpCol (str)   : The name of the column containing the low pivots.
        mustGap (bool): If True, only count pivots if there is a gap between the current bar and the previous bar. If False, count pivots from prev close to current close.
        runType (str) : If 'pct', return the ratio of pivots gapped over to total pivots. If 'count', return the number of pivots gapped over.
        span (int)    : How many bars to look back to find pivots.
    """


    def __post_init__(self):
        self.columns = [self.colname]
        self.pivots = object
        self.min_val = 0
        self.max_val = 0
        self.piv = object
        self.piv_no_nans = object

    def run(self, longshort:str='', df:pd.DataFrame=pd.DataFrame(), **kwargs):
        
        if len(df) > 3:

            if longshort == 'LONG':
                #! cut last row so do not count last HP. This is incase Point is added to very last bar
                self.pivots = df[self.hpCol][self.span:-2].dropna()  
                if self.mustGap:
                    self.max_val = df.low.iat[-1]
                    self.min_val = df.high.iat[-2]
                else:
                    self.max_val = df.open.iat[-1]
                    self.min_val = df.high.iat[-2]
               
            if longshort == 'SHORT':
                #! cut last row so do not count last HP. This is incase Point is added to very last bar
                self.pivots = df[self.lpCol][self.span:-2].dropna() 
                if self.mustGap:
                    self.min_val = df.high.iat[-1]
                    self.max_val = df.low.iat[-2]
                else:
                    self.min_val = df.open.iat[-1]
                    self.max_val = df.low.iat[-2]
            
            if self.max_val > self.min_val:
                is_greater_than_min = self.pivots >= self.min_val # if the pivot is greater than the current bar close then it has been gapped over
                is_less_than_max    = self.pivots <= self.max_val # if the pivot is less than the prior bar close then it has been gapped over

                # drop nan values from piv
                self.piv = np.where( is_greater_than_min & is_less_than_max, self.pivots, np.nan)
                self.piv_no_nans = self.piv[~np.isnan(self.piv)]
                self.val    = len(self.piv_no_nans)

                if self.runType == 'pct':
                    # ratio of pivots gapped over to total pivots
                    # 100 is the best value as it means all pivots have been gapped over
                    #$ this will give a high score at the start of the trade as there will be less pivots to gap over meaning the ratio will be higher.
                    #$ this seems to be ok as earlier trades are more important than later trades. 
                    if len(self.pivots) > 0:
                        self.val = self.val / len(self.pivots) * 100
                    
                    self.log_vals(ls=longshort, val=self.val)

                elif self.runType == 'count':
                    self.val = len(self.piv_no_nans)
                    self.log_vals(ls=longshort, val=self.val)
                
            else:
                self.val = 0

        else:
            self.val = 0

    def reset(self):
        self.val = 0
        self.pivots = object
        self.min_val = 0
        self.max_val = 0
        self.piv = object
        self.piv_no_nans = object


#! Not yet integrated, but tested and working
@dataclass
class GappedBarQuality(Signals):
    """Assess the quality of the gap by assessing the previous bar that has been gapped over.
    If this is a long trade then gaping over a wide range red bar is good but gaping over a 
    wide range green bar is bad, a narrow range bar has less impact, so return the points based upon this assessment"""
    atrMultiple : int = 2
    atrCol      : str = ''
    val    : float = 0.0

    def run(self, longshort:str='', df:pd.DataFrame=pd.DataFrame(), **kwargs):
        if len(df) > 1: 

            atr = df[self.atrCol].iat[-1]

            if longshort == 'LONG':
                has_upper_gap = df.low.iat[-1] > df.high.iat[-2]
                prev_is_red   = df.close.iat[-2] < df.open.iat[-2]

                if has_upper_gap:
                    if prev_is_red:
                        bar_range = df.high.iat[-2] - df.close.iat[-2]
                        self.val =   bar_range / (atr * self.atrMultiple) * 100
                    else:
                        bar_range = df.high.iat[-2] - df.open.iat[-2]
                        self.val =  - bar_range / (atr * self.atrMultiple) * 100 # divide by 2 as the prev bar is green so the gap is not as good so reduce the points

            elif longshort == 'SHORT':
                has_lower_gap = df.high.iat[-1] < df.low.iat[-2]
                prev_is_green = df.close.iat[-2] > df.open.iat[-2]

                if has_lower_gap:
                    if prev_is_green:
                        bar_range = df.close.iat[-2] - df.low.iat[-2]
                        self.val  =  bar_range / (atr * self.atrMultiple) * 100
                    else:
                        bar_range = df.open.iat[-2] - df.low.iat[-2]
                        self.val  = - bar_range / (atr * self.atrMultiple * 2) * 100 # divide by 2 as the prev bar is green so the gap is not as good so reduce the points


#! Not yet integrated, but tested and working
@dataclass
class GappedBarCount(Signals):
    """Assess the quality of the gap by assessing the previous bar that has been gapped over.
    """
    val        : float = 0.0
  
    def get_last_window(self, df:pd.DataFrame=pd.DataFrame()):
        """ get the last window of bars that have a MajorHP and MajorLP """
        hplp_no_nans = df[['MajorHP', 'MajorLP']][:-2].dropna(how='all')

        if len(hplp_no_nans) > 0:
            last_point_index = hplp_no_nans.index[-1]
            last_window      = df.loc[last_point_index:]

            return last_window

    def run(self, longshort:str='', df:pd.DataFrame=pd.DataFrame(), **kwargs):
        if len(df) > 1: 

                if longshort == 'LONG':
                    has_upper_gap = df.low.iat[-1] > df.high.iat[-2]
                    if has_upper_gap:
                        last_window = self.get_last_window(df)
                        if last_window is not None:
                            within_gap = (df.low.iat[-1] > last_window.high) & (df.open.iat[-2] < last_window.high)
                            valid_bars = last_window[within_gap]
                            self.val = valid_bars.shape[0]
        
                if longshort == 'SHORT':
                    has_lower_gap = df.high.iat[-1] < df.low.iat[-2]
                    if has_lower_gap:
                        last_window = self.get_last_window(df)
                        if last_window is not None:
                            within_gap = (df.high.iat[-1] < last_window.low) & (df.open.iat[-2] > last_window.low)
                            valid_bars = last_window[within_gap]
                            self.val = valid_bars.shape[0]

#! Not yet integrated, but tested and working
@dataclass
class GappedPastPivot(Signals):
    """Assess the quality of the gap by assessing the previous bar that has been gapped over.
    """
    atrCol     : str = ''
    val        : float = 0.0



    def get_last_hp(self, df:pd.DataFrame=pd.DataFrame()):
        hp_no_nans = df['MajorHP'][:-2].dropna(how='all')
        if len(hp_no_nans) > 0:
            return hp_no_nans.iat[-1]
    
    def get_last_lp(self, df:pd.DataFrame=pd.DataFrame()):
        lp_no_nans = df['MajorLP'][:-2].dropna(how='all')
        if len(lp_no_nans) > 0:
            return lp_no_nans.iat[-1]
        
    def run(self, longshort:str='', df:pd.DataFrame=pd.DataFrame(), **kwargs):
        if len(df) > 1: 

            if longshort == 'LONG':
                has_upper_gap = df.low.iat[-1] > df.high.iat[-2]
                if has_upper_gap:
                    last_hp = self.get_last_hp(df)
                    if last_hp is not None and not df.high.iat[-2] > last_hp:
                        gap_point_to_low = df.low.iat[-1] - last_hp
                        gap_atr_ratio    = gap_point_to_low / df[self.atrCol].iat[-1] * 100
                        self.val = gap_atr_ratio

            if longshort == 'SHORT':
                has_lower_gap = df.high.iat[-1] < df.low.iat[-2]
                if has_lower_gap:
                    last_lp = self.get_last_lp(df)
                    if last_lp is not None and not df.low.iat[-2] < last_lp:
                        gap_point_to_high = last_lp - df.high.iat[-1]
                        gap_atr_ratio     = gap_point_to_high / df[self.atrCol].iat[-1] * 100
                        self.val = gap_atr_ratio


#$ ------- Volume ---------------
#£ Done
@dataclass
class VolumeSpike(Signals):
    """
    Detects a volume spike in a pandas dataframe with a 'volume' column.
    Returns the percent change between the current volume and the rolling average volume over 'volMA' periods.
    """
    volMACol     : str = ''
    volRatioBest : float = 2.0

    """
    Args:
        volMA (int)          : The number of periods to use for the rolling average volume.
        volRatioBest (float) : The ratio of current volume to rolling average volume that is considered the best.
    """

    def __post_init__(self):
        self.columns = [self.colname]

    def run(self, df, **kwargs):
        current_volume = df['volume'].iloc[-1]
        
        # Calculate the percent change between the current volume and the rolling average volume
        percent_change = ((current_volume - df[self.volMACol].iloc[-1]) / df[self.volMACol].iloc[-1]) * 100

        # Check the value is not NaN
        if pd.isna(percent_change):
            percent_change = 0

        self.val = max(percent_change, 0)
        return self.val

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
    tgetLCol : str = ''
    tgetSCol : str = ''
    atrCol: str = ''
    val   : float = 0.0

    def run(self, longshort:str='', df:pd.DataFrame=pd.DataFrame(), **kwargs):
        if len(df) > 1:
            if longshort == 'LONG':
                if df[self.tgetLCol].iat[-1] is not None:
                    self.val = (df[self.tgetLCol].iat[-1] - df.close.iat[-1]) / df[self.atrCol].iat[-1]
                
                # if there is no pivot point then return 2. meaning it has room to move so give an arbitrary high number
                else:
                    self.val = 2

            elif longshort == 'SHORT':
                if df[self.tgetSCol].iat[-1] is not None:
                    self.val = (df.close.iat[-1] - df[self.tgetSCol].iat[-1]) / df[self.atrCol].iat[-1]

                # if there is no pivot point then return 2. meaning it has room to move so give an arbitrary high number
                else:
                    self.val = 2

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









