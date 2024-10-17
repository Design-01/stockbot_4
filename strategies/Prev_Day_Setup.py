import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
from typing import List, Tuple

class PreviousDaySetupStrategy:
    def __init__(self, risk_multiple: float = 2.0):
        self.risk_multiple = risk_multiple

    def get_previous_day_levels(self, daily_data: pd.DataFrame) -> Tuple[float, float]:
        return daily_data.iloc[-2]['high'], daily_data.iloc[-2]['low']

    def get_first_5min_levels(self, intraday_data: pd.DataFrame) -> Tuple[float, float]:
        first_5min = intraday_data.iloc[:5]
        return first_5min['high'].max(), first_5min['low'].min()

    def check_long_entry(self, price: float, five_min_high: float, previous_day_high: float) -> bool:
        return price > five_min_high  # Removed the condition: and price < previous_day_high

    def calculate_target(self, entry: float, stop_loss: float) -> float:
        risk = entry - stop_loss
        return entry + (risk * self.risk_multiple)

    def calculate_stop_loss(self, prices: pd.DataFrame, entry_index: int, breakout_index: int) -> float:
        relevant_prices = prices.iloc[breakout_index+1:entry_index]
        return relevant_prices['low'].min()

    def execute_trade(self, intraday_data: pd.DataFrame, daily_data: pd.DataFrame) -> List[dict]:
        prev_day_high, prev_day_low = self.get_previous_day_levels(daily_data)
        five_min_high, five_min_low = self.get_first_5min_levels(intraday_data)
        
        trades = []
        in_trade = False
        entry_price = 0
        stop_loss = 0
        breakout_index = None
        
        pullback_detector = PullbackAndBounceDetecter()
        
        for i, row in intraday_data.iterrows():
            if not in_trade:
                if self.check_long_entry(row['close'], five_min_high, prev_day_high):
                    print(f"Potential entry at {i}: Close {row['close']}")
                    if pullback_detector.detect(intraday_data.loc[:i], five_min_high):
                        print(f"Trade entered at {i}: Price {row['close']}")
                        entry_price = row['close']
                        stop_loss = self.calculate_stop_loss(intraday_data, i, breakout_index)
                        target = self.calculate_target(entry_price, stop_loss)
                        in_trade = True
                        trades.append({
                            'entry_time': i,
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'target': target
                        })
            else:
                if row['low'] <= stop_loss:
                    trades[-1]['exit_time'] = i
                    trades[-1]['exit_price'] = stop_loss
                    trades[-1]['profit'] = stop_loss - entry_price
                    in_trade = False
                    breakout_index = None
                elif row['high'] >= target:
                    trades[-1]['exit_time'] = i
                    trades[-1]['exit_price'] = target
                    trades[-1]['profit'] = target - entry_price
                    in_trade = False
                    breakout_index = None
                elif row['close'] > prev_day_high:
                    new_target = self.calculate_target(row['close'], stop_loss)
                    trades[-1]['target'] = new_target
        
        print(f"Total trades executed: {len(trades)}")
        return trades

class PullbackAndBounceDetecter:
    def detect(self, prices: pd.DataFrame, level: float) -> bool:
        if len(prices) < 3:
            return False
        
        # Find the first candle that closed above the level
        breakout_index = prices[prices['close'] > level].index[0]
        
        # Check for pullback
        pullback_prices = prices.loc[breakout_index:]
        pullback = (pullback_prices['low'] <= level).any()
        
        if not pullback:
            return False
        
        # Find the lowest point of the pullback
        lowest_point = pullback_prices['low'].idxmin()
        
        # Check for bounce
        bounce_prices = prices.loc[lowest_point:]
        if len(bounce_prices) < 2:
            return False
        
        bounce = (bounce_prices['close'] > level).any() and (bounce_prices['high'] > bounce_prices.iloc[0]['high']).any()
        
        return bounce

class Backtester:
    def __init__(self, strategy: PreviousDaySetupStrategy):
        self.strategy = strategy

    def run(self, intraday_data: pd.DataFrame, daily_data: pd.DataFrame) -> pd.DataFrame:
        trades = self.strategy.execute_trade(intraday_data, daily_data)
        return pd.DataFrame(trades)

    def calculate_metrics(self, trades: pd.DataFrame) -> dict:
        if trades.empty:
            return {'total_trades': 0, 'win_rate': 0, 'average_profit': 0, 'profit_factor': 0}

        winning_trades = trades[trades['profit'] > 0]
        losing_trades = trades[trades['profit'] <= 0]

        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        average_profit = trades['profit'].mean()
        profit_factor = abs(winning_trades['profit'].sum() / losing_trades['profit'].sum()) if losing_trades['profit'].sum() != 0 else float('inf')

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'average_profit': average_profit,
            'profit_factor': profit_factor
        }

# Example usage:
# strategy = PreviousDaySetupStrategy(risk_multiple=2.0)
# backtester = Backtester(strategy)
# results = backtester.run(intraday_data, daily_data)
# metrics = backtester.calculate_metrics(results)
# print(metrics)

class AnimatedTradingVisualization:
    def __init__(self, intraday_data: pd.DataFrame, trades: pd.DataFrame, strategy: PreviousDaySetupStrategy,
                 entry_time_col='entry_time', entry_price_col='entry_price',
                 exit_time_col='exit_time', exit_price_col='exit_price',
                 profit_col='profit'):
        self.intraday_data = intraday_data
        self.trades = trades
        self.strategy = strategy
        self.entry_time_col = entry_time_col
        self.entry_price_col = entry_price_col
        self.exit_time_col = exit_time_col
        self.exit_price_col = exit_price_col
        self.profit_col = profit_col
        self.fig = self._create_base_figure()

    def _create_base_figure(self):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.1, 
                            row_heights=[0.7, 0.3])

        # Add candlestick chart with lowercase column names
        fig.add_trace(go.Candlestick(x=self.intraday_data.index,
                                     open=self.intraday_data['open'],
                                     high=self.intraday_data['high'],
                                     low=self.intraday_data['low'],
                                     close=self.intraday_data['close'],
                                     name='Price'),
                      row=1, col=1)

        # Add previous day high and low
        prev_day_high, prev_day_low = self.strategy.get_previous_day_levels(self.intraday_data)
        fig.add_hline(y=prev_day_high, line_dash="dash", line_color="red", annotation_text="Previous Day High", row=1, col=1)
        fig.add_hline(y=prev_day_low, line_dash="dash", line_color="green", annotation_text="Previous Day Low", row=1, col=1)

        # Add 5-minute high and low
        five_min_high, five_min_low = self.strategy.get_first_5min_levels(self.intraday_data)
        fig.add_hline(y=five_min_high, line_dash="dot", line_color="orange", annotation_text="5-min High", row=1, col=1)
        fig.add_hline(y=five_min_low, line_dash="dot", line_color="purple", annotation_text="5-min Low", row=1, col=1)

        # Update layout
        fig.update_layout(title='Animated Trading Strategy Visualization',
                          xaxis_title='Time',
                          yaxis_title='Price',
                          xaxis2_title='Time',
                          yaxis2_title='Cumulative P&L',
                          height=800)

        fig.update_xaxes(rangeslider_visible=False)

        return fig

    def animate(self, step_size: int = 10):
        frames = []
        for i in range(0, len(self.intraday_data), step_size):
            frame_data = self.intraday_data.iloc[:i+1]
            frame_trades = self.trades[self.trades[self.entry_time_col] <= frame_data.index[-1]]

            frame = go.Frame(
                data=[
                    go.Candlestick(x=frame_data.index,
                                   open=frame_data['open'],
                                   high=frame_data['high'],
                                   low=frame_data['low'],
                                   close=frame_data['close'],
                                   name='Price'),
                    go.Scatter(x=frame_trades[self.entry_time_col], y=frame_trades[self.entry_price_col],
                               mode='markers', marker_symbol='triangle-up', marker_size=10, marker_color='green',
                               name='Buy'),
                    go.Scatter(x=frame_trades[self.exit_time_col], y=frame_trades[self.exit_price_col],
                               mode='markers', marker_symbol='triangle-down', marker_size=10, marker_color='red',
                               name='Sell'),
                    go.Scatter(x=frame_trades[self.exit_time_col], y=frame_trades[self.profit_col].cumsum(),
                               mode='lines+markers', name='Cumulative P&L')
                ],
                traces=[0, 1, 2, 3],
                name=f'frame{i}'
            )
            frames.append(frame)

        self.fig.frames = frames

        # Add play and pause buttons
        self.fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="Play",
                             method="animate",
                             args=[None, {"frame": {"duration": 100, "redraw": True},
                                          "fromcurrent": True,
                                          "transition": {"duration": 0}}]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None], {"frame": {"duration": 0, "redraw": True},
                                            "mode": "immediate",
                                            "transition": {"duration": 0}}])
                    ]
                )
            ]
        )

        # Add slider
        self.fig.update_layout(
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Time: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 100, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [f"frame{k}"],
                            {"frame": {"duration": 100, "redraw": True},
                             "mode": "immediate",
                             "transition": {"duration": 0}}
                        ],
                        "label": str(self.intraday_data.index[k]),
                        "method": "animate"
                    }
                    for k in range(0, len(self.intraday_data), step_size)
                ]
            }]
        )

    def show(self):
        self.fig.show()

# Example usage:
# intraday_data = pd.read_csv('intraday_data.csv', index_col='timestamp', parse_dates=True)
# strategy = PreviousDaySetupStrategy(risk_multiple=2.0)
# backtester = Backtester(strategy)
# trades = backtester.run(intraday_data, daily_data)
# viz = AnimatedTradingVisualization(intraday_data, trades, strategy)
# viz.animate(step_size=10)
# viz.show()