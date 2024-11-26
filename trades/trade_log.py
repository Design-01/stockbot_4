from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, List, ClassVar, Dict
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
import os

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, ClassVar, Tuple
import pandas as pd

@dataclass
class TradeDetails:
    # Class variable to track the current ID across all instances
    _current_trade_number: ClassVar[int] = 1
    
    # Basic Trade Info (required fields)
    symbol: str
    barsize: str
    
    # Auto-generated fields (not included in init)
    trade_number: int = field(init=False)
    log_id: str = field(init=False)
    
    # Optional fields with defaults
    direction: str = "LONG"
    status: str = "PENDING"
    is_active: bool = True
    
    # Time Related
    chart_time: datetime = field(default_factory=datetime.now)
    real_time: datetime = field(default_factory=datetime.now)
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    duration: Optional[float] = None
    
    # Entry Related
    entry_name: Optional[str] = None
    av_entry_price: Optional[float] = None
    ib_entry_id: Optional[str] = None
    entry_filled: Tuple[int, int] = (0, 0)
    
    # Exit Related
    exit_name: Optional[str] = None
    av_exit_price: Optional[float] = None
    ib_exit_id: Optional[str] = None
    exit_filled: Tuple[int, int] = (0, 0)
    exit_type: Optional[str] = None
    
    # Position Information
    position: int = 0
    value: float = 0.0
    close_price: Optional[float] = None
    
    # Risk Management
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_name: Optional[str] = None
    target_name: Optional[str] = None
    risk_reward: float = 0.0
    target_risk_reward: float = 0.0
    
    # Performance Metrics
    unrealized_pl: float = 0.0
    realized_pl: float = 0.0
    
    # New fields for enhanced functionality
    precision: int = 2
    risk_percentage: float = 0.01  # 1% risk per trade
    account_size: float = 100000.0  # Default account size
    commission: float = 0.0
    slippage: float = 0.0
    total_cost: float = 0.0  # commission + slippage
    
    # IB Order IDs
    stop_order_id: Optional[str] = None
    target_order_id: Optional[str] = None

    
    def __post_init__(self):
        # Generate trade number and log ID
        self.trade_number = TradeDetails._current_trade_number
        TradeDetails._current_trade_number += 1
        self.log_id = f"{self.symbol}_{self.trade_number}"
    
    
    def to_dict(self) -> Dict[str, Any]:
        """Returns all fields as a dictionary"""
        return self.__dict__
    
    def to_dataframe(self) -> pd.DataFrame:
        """Returns trade details as a single-row DataFrame"""
        return pd.DataFrame([self.to_dict()])
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeDetails':
        """Creates a TradeDetails instance from a dictionary"""
        return cls(**{k: v for k, v in data.items() 
                     if k in cls.__dataclass_fields__})

class TradeLogManager:
    def __init__(self):
        self.trades: List[TradeDetails] = []
        
    def log_trade(self, trade: TradeDetails) -> None:
        """Add a new trade to the trade log."""
        self.trades.append(trade)

    def _organize_trades(self) -> None:
        """
        Organize trades by removing exact duplicates but keeping different versions
        of the same trade (same ID, different values) sorted by chart_time.
        """
        # First, group trades by log_id
        trade_groups: Dict[str, List[TradeDetails]] = {}
        for trade in self.trades:
            if trade.log_id not in trade_groups:
                trade_groups[trade.log_id] = []
            trade_groups[trade.log_id].append(trade)
        
        # For each group, sort by chart_time and remove exact duplicates
        organized_trades: List[TradeDetails] = []
        for trade_group in trade_groups.values():
            # Sort by chart_time
            sorted_trades = sorted(trade_group, key=lambda x: x.chart_time)
            
            # Remove exact duplicates while preserving order
            unique_trades = []
            for trade in sorted_trades:
                # Only add if not an exact duplicate of the last trade
                if not unique_trades or not trade.is_exact_duplicate(unique_trades[-1]):
                    unique_trades.append(trade)
            
            organized_trades.extend(unique_trades)
        
        # Sort all trades by trade_number and then chart_time
        organized_trades.sort(key=lambda x: (x.trade_number, x.chart_time))
        self.trades = organized_trades

    def save_csv(self, filepath: str, append: bool = True) -> None:
        """
        Save trade logs to a CSV file.
        
        Args:
            filepath: Path to save the CSV file
            append: If True, append to existing file, if False, overwrite
        """
        # If appending and file exists, load existing trades first
        if append and os.path.exists(filepath):
            existing_manager = TradeLogManager()
            existing_manager.load_csv(filepath)
            
            # Add current trades to existing trades
            for trade in self.trades:
                existing_manager.trades.append(trade)
            
            # Organize trades (remove exact duplicates, sort by chart_time)
            existing_manager._organize_trades()
            
            # Get DataFrame from combined trades
            df = existing_manager.get_df()
        else:
            self._organize_trades()
            df = self.get_df()
        
        # Convert datetime columns to string format
        datetime_columns = ['chart_time', 'real_time', 'entry_time', 'exit_time']
        for col in datetime_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x.isoformat() if pd.notnull(x) else '')
        
        # Save to CSV
        df.to_csv(filepath, index=False)

    def load_csv(self, filepath: str) -> None:
        """Load trade logs from a CSV file."""
        if not os.path.exists(filepath):
            return
            
        df = pd.read_csv(filepath)
        
        # Clear existing trades
        self.trades.clear()
        
        # Convert string dates back to datetime objects
        datetime_columns = ['chart_time', 'real_time', 'entry_time', 'exit_time']
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert string tuples back to actual tuples
        if 'entry_filled' in df.columns:
            df['entry_filled'] = df['entry_filled'].apply(eval)
        if 'exit_filled' in df.columns:
            df['exit_filled'] = df['exit_filled'].apply(eval)
        
        # Convert DataFrame rows back to TradeLog objects
        highest_trade_number = 0
        for _, row in df.iterrows():
            trade_dict = row.to_dict()
            if 'trade_number' in trade_dict:
                highest_trade_number = max(highest_trade_number, trade_dict['trade_number'])
            trade = TradeDetails.from_dict(trade_dict)
            self.trades.append(trade)
        
        # Update the TradeLog class counter
        if highest_trade_number > 0:
            TradeDetails.set_trade_number(highest_trade_number + 1)
        
        # Organize trades
        self._organize_trades()

    def get_trade_history(self, log_id: str) -> List[TradeDetails]:
        """
        Get all versions of a specific trade sorted by chart_time.
        """
        trade_versions = [trade for trade in self.trades if trade.log_id == log_id]
        return sorted(trade_versions, key=lambda x: x.chart_time)

    def get_df(self) -> pd.DataFrame:
        """Convert all trades to a pandas DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        # Convert trades to dictionaries
        trade_dicts = [asdict(trade) for trade in self.trades]
        
        # Create DataFrame
        df = pd.DataFrame(trade_dicts)
        
        # Convert tuple columns to strings for better storage
        if not df.empty:
            if 'entry_filled' in df.columns:
                df['entry_filled'] = df['entry_filled'].apply(str)
            if 'exit_filled' in df.columns:
                df['exit_filled'] = df['exit_filled'].apply(str)
        
        return df