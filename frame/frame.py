from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from chart.chart import Chart
from strategies.ta import TA
import pandas as pd

@dataclass
class Frame:
    symbol: str
    trading_hours: List[Tuple[str, str]] = field(default_factory=lambda: [("09:30", "16:00")])


    def __post_init__(self):
        self.traders = []
        self.data = pd.DataFrame()
        self.ta = {}
        self.chart = None

    #£ Working
    def load_ohlcv(self, ohlcv: pd.DataFrame):
        if self.data.empty:
            self.data = ohlcv
        else:
            self.data = pd.concat([self.data, ohlcv])

    #£ Working
    def setup_chart(self):  
        self.chart = Chart(title=self.symbol, rowHeights=[0.2, 0.1, 0.8], height=800, width=800)
        self.chart.add_candles_and_volume(self.data)
        # self.chart.add_trading_hours(self.data, self.trading_hours)

    #! NOT Working
    def add_ta(self, ta: TA, style: Dict[str, Any] | List[Dict[str, Any]], chart_type: str = "line", row: int = 1):
        self.ta[ta.__class__.__name__] = (ta, style, chart_type, row)

    def run_ta(self) -> str | List[str]:
        for name, (ta, style, chart_type, row) in self.ta.items():
            ta.run(self.data)
            self.data.loc[:, ta.names] = ta.data




    def update_data(self, new_data: Any):
        self.dm.update_data(new_data)
        self._update_chart()

    def _update_chart(self):
        self.chart.refesh(self.data)
        # for indicator, style, chart_type, row in self.ta:
            # indicator_data = self.data[indicator.names]
            # self.chart.add_ta(indicator_data, style, chart_type, row)

    def plot(self, width: int = 1400, height: int = 800, trading_hours: bool = False):
        self._update_chart()
        if trading_hours: self.chart.add_trading_hours(self.data, self.trading_hours)
        self.chart.show(width=width, height=height)


    def plot_refresh(self, width: int = 1400, height: int = 800):
        self.chart.refesh(self.data)
        self._update_chart()
        self.chart.show(width=width, height=height)
