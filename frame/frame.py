from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from data.data_manager import DataManager
from chart.chart import Chart
from strategies.ta import Indicator

@dataclass
class Frame:
    data: Any
    symbol: str
    trading_hours: List[Tuple[str, str]]
    indicators: List[Tuple[Indicator, Dict[str, Any], str, int]] = field(default_factory=list)

    def __post_init__(self):
        self.dm = DataManager(self.data)
        self.chart = Chart(title=self.symbol, rowHeights=[0.2, 0.1, 0.8], height=800, width=800)
        self.chart.add_candles_and_volume(self.dm.data)
        self.chart.add_trading_hours(self.dm.data, self.trading_hours)

    def add_ta(self, indicator: Indicator, style: Dict[str, Any] | List[Dict[str, Any]], chart_type: str = "line", row: int = 1):
        self.dm.add_ta(indicator)
        self.indicators.append((indicator, style, chart_type, row))

    def update_data(self, new_data: Any):
        self.dm.update_data(new_data)
        self._update_chart()

    def _update_chart(self):
        self.chart.refesh(self.dm.data)
        for indicator, style, chart_type, row in self.indicators:
            indicator_data = self.dm.data[indicator.names]
            self.chart.add_ta(indicator_data, style, chart_type, row)

    def plot(self, width: int = 1400, height: int = 800):
        self._update_chart()
        self.chart.show(width=width, height=height)

    def plot_refresh(self, width: int = 1400, height: int = 800):
        self.chart.refesh(self.dm.data)
        self._update_chart()
        self.chart.show(width=width, height=height)
