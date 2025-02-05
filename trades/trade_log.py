from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
import pickle
from pathlib import Path
from typing import ClassVar, Tuple
import pandas as pd


@dataclass
class LogMarketTA:
    """Logs market conditions and sector conditions from DataFrame analysis"""
    symbol: str = ''
    barsize: str = ''
    chart_time: datetime = field(default_factory=datetime.now)
    real_time: datetime = field(default_factory=datetime.now)
    log_id: str = field(init=False)
    conditions: dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.log_id = f"{self.symbol}_{self.barsize}_{self.chart_time}"

@dataclass
class LogProfitLoss:
    """Simple container for P&L metrics"""
    entry_price: float = 0.0
    exit_price: float = 0.0
    position_size: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    risk_reward_ratio: float = 0.0
    realized_pl: float = 0.0
    realized_r_multiple: float = 0.0


@dataclass
class LogDiary:
    """Simple trade diary entry"""
    real_entry_time: datetime = field(default_factory=datetime.now)
    symbol: str = ''
    strategy_name: str = ''
    entry_reason: str = ''
    exit_reason: str = ''
    condidence: float = 0 # 0-10
    notes: str = ''


@dataclass
class TradeLog:
    """Main trade logging class that combines all components"""
    trade_id: str = None
    entry_time: datetime = None
    exit_time: datetime = None
    symbol: str = ''
    barSize: str = ''
    strategy_name: str = ''
    market_conditions: LogMarketTA = None
    sector_conditions: LogMarketTA = None
    stock_conditions: LogMarketTA = None
    entry_strategy: Dict[str, Any] = None
    exit_strategy: Dict[str, Any] = None
    pnl: LogProfitLoss = None
    notes: LogDiary = None
    status: str = "open"  # open, closed, cancelled
    
    def __post_init__(self):
        self.trade_id = f"{self.symbol}_{self.barSize}_{datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}"
    
    def close_trade(self, exit_price: float, exit_time: datetime = None):
        """Close the trade with exit details"""
        self.exit_time = exit_time or datetime.now()
        self.pnl.exit_price = exit_price
        self.status = "closed"

class TradeLogger:
    def __init__(self, base_path: str = "trade_logs"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def log_trade(self, trade: TradeLog) -> None:
        """Save trade object as pickle"""
        pickle_path = self.base_path / f"{trade.trade_id}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(trade, f)
    
    def load_trade(self, trade_id: str) -> Optional[TradeLog]:
        """Load a specific trade from pickle"""
        pickle_path = self.base_path / f"{trade_id}.pkl"
        if not pickle_path.exists():
            return None
        
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    
    def list_trades(self) -> list:
        """List all trade IDs"""
        return [f.stem for f in self.base_path.glob('*.pkl')]
    
# Example of proper trade updating
def example_trade_update():
    logger = TradeLogger()
    
    # First create and log a new trade
    market_conditions = LogMarketTA(
        symbol="SPY",
        barsize="5min",
        conditions={'spx_above_ma': True, 'market_trend': 'bullish'}
    )
    
    sector_conditions = LogMarketTA(
        symbol="XLK",
        barsize="5min",
        conditions={'sector_strength': 0.85}
    )
    
    stock_conditions = LogMarketTA(
        symbol="TSLA",
        barsize="5min",
        conditions={'rsi': 65.5}
    )
    
    # Create initial trade
    pnl = LogProfitLoss(
        entry_price=250.75,
        stop_loss=245.00,
        target_price=260.00,
        position_size=100
    )
    
    trade = TradeLog(
        symbol="TSLA",
        strategy_name="Breakout_Strategy",
        market_conditions=market_conditions,
        sector_conditions=sector_conditions,
        stock_conditions=stock_conditions,
        entry_strategy={"type": "breakout", "level": 250.00},
        exit_strategy={"type": "trailing_stop", "percentage": 2.0},
        pnl=pnl
    )
    
    # Save initial trade
    logger.log_trade(trade)
    trade_id = trade.trade_id  # Store the ID for later use
    
    # Later, when we want to close the trade:
    
    # 1. First load the existing trade
    trade = logger.load_trade(trade_id)
    if trade is None:
        raise ValueError(f"Trade {trade_id} not found")
    
    # 2. Update the trade
    trade.close_trade(exit_price=258.50)
    trade.pnl.realized_pl = 775.00  # Your calculated P&L
    trade.pnl.realized_r_multiple = 1.5  # Your calculated R-multiple
    
    # 3. Save the updated trade back to pickle
    logger.log_trade(trade)
    
    # We can verify the update worked
    updated_trade = logger.load_trade(trade_id)
    print(f"Trade status: {updated_trade.status}")
    print(f"Exit price: {updated_trade.pnl.exit_price}")
    print(f"Realized P&L: {updated_trade.pnl.realized_pl}")

# ----------------------------------------------------
# ------- Trade Details Dataclass --------------------
# ----------------------------------------------------

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

