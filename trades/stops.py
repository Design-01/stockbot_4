from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import trade_log as tl

# Abstract base class for stop loss strategies
class StopStrategy(ABC):
    @abstractmethod
    def calculate(self, df: pd.DataFrame, trade_log: tl.TradeDetails) -> float:
        """Calculate stop loss price based on the strategy"""
        pass

# Concrete stop strategies
class StopPrevBar(StopStrategy):
    def calculate(self, df: pd.DataFrame, trade_log: tl.TradeDetails) -> float:
        """Use previous bar's low/high as stop"""
        if trade_log.entry_time is None:
            return 0.0
        
        entry_idx = df.index.get_loc(trade_log.entry_time)
        if entry_idx > 0:
            if trade_log.direction == "LONG":
                return df.iloc[entry_idx - 1]['low']
            return df.iloc[entry_idx - 1]['high']
        return 0.0

class StopPriorPiv:
    def __init__(self, n_bars: int = 3):
        """
        Initialize the StopPriorPiv class.
        
        Args:
            n_bars (int): Number of bars to consider on each side for pivot point detection.
                         Default is 3 bars before and after the potential pivot point.
        """
        self.n_bars = n_bars
        
    def _find_pivots(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Find all pivot highs and lows in the dataset.
        
        Args:
            df (pd.DataFrame): DataFrame containing price data with 'high' and 'low' columns
            
        Returns:
            Tuple[pd.Series, pd.Series]: Boolean series indicating pivot highs and lows
        """
        # Create rolling windows for comparison
        high_window = df['high'].rolling(window=2*self.n_bars + 1, center=True)
        low_window = df['low'].rolling(window=2*self.n_bars + 1, center=True)
        
        # Find pivot highs (middle point higher than all others in window)
        pivot_highs = pd.Series(False, index=df.index)
        pivot_lows = pd.Series(False, index=df.index)
        
        # For pivot highs
        for i in range(self.n_bars, len(df) - self.n_bars):
            window_highs = df['high'].iloc[i-self.n_bars:i+self.n_bars+1]
            if df['high'].iloc[i] == max(window_highs):
                pivot_highs.iloc[i] = True
                
        # For pivot lows
        for i in range(self.n_bars, len(df) - self.n_bars):
            window_lows = df['low'].iloc[i-self.n_bars:i+self.n_bars+1]
            if df['low'].iloc[i] == min(window_lows):
                pivot_lows.iloc[i] = True
        
        return pivot_highs, pivot_lows

    def _validate_pivots(self, df: pd.DataFrame, trade_time: datetime, direction: str,
                        pivot_highs: pd.Series, pivot_lows: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Validate pivots based on trade direction and subsequent price action.
        
        Args:
            df (pd.DataFrame): Price data
            trade_time (datetime): Current trade time
            direction (str): Trade direction ('LONG' or 'SHORT')
            pivot_highs (pd.Series): Series of pivot highs
            pivot_lows (pd.Series): Series of pivot lows
            
        Returns:
            Tuple[pd.Series, pd.Series]: Validated pivot highs and lows
        """
        valid_highs = pivot_highs.copy()
        valid_lows = pivot_lows.copy()
        
        # Get indices where pivots are True
        high_indices = valid_highs[valid_highs].index
        low_indices = valid_lows[valid_lows].index
        
        if direction == 'LONG':
            # For each pivot high
            for high_idx in high_indices:
                if high_idx > trade_time:
                    continue
                current_high = df.loc[high_idx, 'high']
                # Invalidate previous lows that haven't been confirmed
                for low_idx in low_indices:
                    if low_idx >= high_idx:
                        continue
                    if low_idx <= trade_time and df.loc[low_idx:high_idx, 'high'].max() < current_high:
                        valid_lows.loc[low_idx] = False
                        
        elif direction == 'SHORT':
            # For each pivot low
            for low_idx in low_indices:
                if low_idx > trade_time:
                    continue
                current_low = df.loc[low_idx, 'low']
                # Invalidate previous highs that haven't been confirmed
                for high_idx in high_indices:
                    if high_idx >= low_idx:
                        continue
                    if high_idx <= trade_time and df.loc[high_idx:low_idx, 'low'].min() > current_low:
                        valid_highs.loc[high_idx] = False
        
        return valid_highs, valid_lows

    def calculate(self, df: pd.DataFrame, trade_log: tl.TradeDetails) -> Optional[float]:
        """
        Calculate the prior pivot point to be used as a stop loss, ensuring it doesn't exceed current price.
        
        Args:
            df (pd.DataFrame): DataFrame containing price data with 'high' and 'low' columns
            trade_log: Trade log object containing trade information
            
        Returns:
            Optional[float]: Price level of the relevant prior pivot, or None if no valid pivot found
        """
        # Find all potential pivot points
        pivot_highs, pivot_lows = self._find_pivots(df)
        
        # Get data up to trade time
        df_subset = df[df.index <= trade_log.chart_time].copy()
        
        if df_subset.empty:
            return None
            
        # Get current price
        current_price = trade_log.price_now if trade_log.price_now is not None else df_subset['close'].iloc[-1]
        
        # Validate pivots based on subsequent price action
        valid_highs, valid_lows = self._validate_pivots(
            df, trade_log.chart_time, trade_log.direction, pivot_highs, pivot_lows
        )
        
        # Find the most recent valid pivot based on direction
        if trade_log.direction == 'LONG':
            # For long trades, look for the most recent valid pivot low that's below current price
            valid_pivot_times = valid_lows[valid_lows & (df.index <= trade_log.chart_time)].index
            
            # Filter out pivots that would result in immediate stop out
            for pivot_time in reversed(valid_pivot_times):
                pivot_price = df.loc[pivot_time, 'low']
                if pivot_price < current_price:
                    # print(f"Found pivot price: {pivot_price} at {pivot_time}")
                    return pivot_price
                
        elif trade_log.direction == 'SHORT':
            # For short trades, look for the most recent valid pivot high that's above current price
            valid_pivot_times = valid_highs[valid_highs & (df.index <= trade_log.chart_time)].index
            
            # Filter out pivots that would result in immediate stop out
            for pivot_time in reversed(valid_pivot_times):
                pivot_price = df.loc[pivot_time, 'high']
                if pivot_price > current_price:
                    return pivot_price
        
        return None
    
class StopGapDiff:
    def __init__(self, lag: int = 1):
        """
        Initialize the StopGapDiff class for calculating stop losses based on price gaps.
        
        Args:
            lag (int): Number of bars to lag behind the current bar. Default is 1 to use the previous
                      completed bar instead of the current bar. Must be >= 1.
        """
        if lag < 1:
            raise ValueError("Lag must be at least 1 to ensure using completed bars")
        self.lag = lag
        
    def calculate(self, df: pd.DataFrame, trade_log: tl.TradeDetails) -> Optional[float]:
        """
        Calculate the stop loss price based on the gap difference method.
        
        Args:
            df (pd.DataFrame): DataFrame containing price data with 'high' and 'low' columns
            trade_log (TradeDetails): Trade log object containing entry information and direction
            
        Returns:
            float: Calculated stop loss price or None if no valid stop can be calculated
        """
        # Get the current bar index based on the entry time
        current_idx = df.index.get_loc(trade_log.chart_time)
        
        # Apply lag to shift which bar we consider as "current"
        reference_idx = current_idx - self.lag
        
        # Check if we have enough bars
        if reference_idx < 0:
            return None
            
        if trade_log.direction == 'LONG':
            return self._calculate_long_stop(df, reference_idx)
        elif trade_log.direction == 'SHORT':
            return self._calculate_short_stop(df, reference_idx)
        else:
            raise ValueError(f"Invalid direction: {trade_log.direction}")
    
    def _calculate_long_stop(self, df: pd.DataFrame, reference_idx: int) -> Optional[float]:
        """
        Calculate stop loss for a long position
        
        Args:
            df (pd.DataFrame): Price DataFrame
            reference_idx (int): Index of the reference bar (lagged from current)
            
        Returns:
            Optional[float]: Stop loss price or None if no valid stop found
        """
        # Get the reference bar's low (lagged from current)
        reference_low = df.iloc[reference_idx]['low']
        
        # Look backwards from the reference bar for the first high that's lower than reference low
        for i in range(reference_idx - 1, -1, -1):
            prev_high = df.iloc[i]['high']
            if prev_high < reference_low:
                # Found the reference high point
                # Calculate midpoint between reference low and previous high
                stop_price = (reference_low + prev_high) / 2
                return stop_price
                
        return None  # No valid stop loss found
    
    def _calculate_short_stop(self, df: pd.DataFrame, reference_idx: int) -> Optional[float]:
        """
        Calculate stop loss for a short position
        
        Args:
            df (pd.DataFrame): Price DataFrame
            reference_idx (int): Index of the reference bar (lagged from current)
            
        Returns:
            Optional[float]: Stop loss price or None if no valid stop found
        """
        # Get the reference bar's high (lagged from current)
        reference_high = df.iloc[reference_idx]['high']
        
        # Look backwards from the reference bar for the first low that's higher than reference high
        for i in range(reference_idx - 1, -1, -1):
            prev_low = df.iloc[i]['low']
            if prev_low > reference_high:
                # Found the reference low point
                # Calculate midpoint between reference high and previous low
                stop_price = (reference_high + prev_low) / 2
                return stop_price
                
        return None  # No valid stop loss found

class StopMA(StopStrategy):
    def __init__(self, period: int):
        self.period = period
        
    def calculate(self, df: pd.DataFrame, trade_log: tl.TradeDetails) -> float:
        """Use moving average as stop"""
        if trade_log.entry_time is None:
            return 0.0
            
        df['ma'] = df['close'].rolling(window=self.period).mean()
        entry_idx = df.index.get_loc(trade_log.entry_time)
        return df.iloc[entry_idx]['ma']

# Stop condition protocols
class StopCondition(Protocol):
    def is_valid(self, df: pd.DataFrame, trade_log: tl.TradeDetails) -> bool:
        """Check if condition is met"""
        pass

class CondDuration:
    def __init__(self, bars: int):
        self.bars = bars
        
    def is_valid(self, df: pd.DataFrame, trade_log: tl.TradeDetails) -> bool:
        """Check if trade duration exceeds specified bars"""
        # print("Checking duration")
        if trade_log.entry_time is None:
            return False
            
        entry_idx = df.index.get_loc(trade_log.entry_time)
        current_idx = len(df) - 1
        is_met = (current_idx - entry_idx) >= self.bars
        # print(f"Duration id met: {is_met}")
        return (current_idx - entry_idx) >= self.bars

class CondRRatio:
    def __init__(self, ratio: float):
        self.ratio = ratio
        
    def is_valid(self, df: pd.DataFrame, trade_log: tl.TradeDetails) -> bool:
        """Check if risk-reward ratio exceeds specified value"""
        if trade_log.entry_time is None or trade_log.av_entry_price is None:
            return False
            
        current_price = df.iloc[-1]['close']
        entry_price = trade_log.av_entry_price
        stop_price = trade_log.stop_price if trade_log.stop_price else entry_price
        
        if trade_log.direction == "LONG":
            reward = current_price - entry_price
            risk = entry_price - stop_price
        else:
            reward = entry_price - current_price
            risk = stop_price - entry_price
            
        return (risk != 0) and (reward / risk >= self.ratio)

class StopLoss:
    def __init__(
        self,
        init: StopStrategy,
        trail1: Optional[StopStrategy] = None,
        trail2: Optional[StopStrategy] = None,
        cond1: Optional[StopCondition] = None,
        cond2: Optional[StopCondition] = None
    ):
        self.init = init
        self.trail1 = trail1
        self.trail2 = trail2
        self.cond1 = cond1
        self.cond2 = cond2
        self.current_stop = None

    def get_price(self, df: pd.DataFrame, trade_log: tl.TradeDetails) -> float:
        """Calculate the current stop loss price based on conditions and strategies"""
        if trade_log.entry_time is None:
            return 0.0

        # Initialize stop if not set
        if self.current_stop is None:
            self.current_stop = self.init.calculate(df, trade_log)
            return self.current_stop

        # Check conditions and apply trailing stops
        new_stop = self.current_stop
        
        if self.trail1 and self.cond1 and self.cond1.is_valid(df, trade_log):
            trail1_price = self.trail1.calculate(df, trade_log)
            if trail1_price is None:
                return new_stop
            if trade_log.direction == "LONG":
                new_stop = max(new_stop, trail1_price)
            else:
                new_stop = min(new_stop, trail1_price)

        if self.trail2 and self.cond2 and self.cond2.is_valid(df, trade_log):
            trail2_price = self.trail2.calculate(df, trade_log)
            if trail2_price is None:
                return new_stop
            if trade_log.direction == "LONG":
                new_stop = max(new_stop, trail2_price)
            else:
                new_stop = min(new_stop, trail2_price)

        self.current_stop = new_stop
        return new_stop