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
    indicators: List[Tuple[Indicator, Dict[str, Any]]] = field(default_factory=list)

    def __post_init__(self):
        self.dm = DataManager(self.data)
        self.chart = Chart(self.dm, title=self.symbol, rowHeights=[0.2, 0.1, 0.7], height=800, width=800)
        self.chart.add_candles_and_volume()
        self.chart.add_trading_hours(self.trading_hours)

    def add_ta(self, indicator: Indicator, style: Dict[str, Any]):
        self.dm.add_ta(indicator)
        self.indicators.append((indicator, style))

    def update_data(self, new_data: Any):
        self.dm.update_data(new_data)
        self._update_chart()

    def _update_chart(self):
        self.chart.update_chart()
        for indicator, style in self.indicators:
            chart_type = "macd" if indicator.__class__.__name__ == "MACD" else "line"
            self.chart.add_ta(indicator, style, chart_type)

    def plot(self):
        self._update_chart()
        self.chart.show(width=1400)

    def plot_refresh(self):
        self.chart.refesh()
        self._update_chart()
        self.chart.show(width=1400)
