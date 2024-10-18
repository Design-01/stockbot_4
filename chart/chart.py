import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Tuple, Dict, Any
import datetime




class Chart:

    def __init__(self, title: str = "Candlestick Chart", rowHeights:List[float] = [0.6, 0.2, 0.1, 0.1], height:int = 800, width:int = 800):
        self.title = title
        self.rowHeights = rowHeights
        self.height = height
        self.width = width
        self.fig = None
        self.indicators: Dict[str, Dict[str, Any]] = {}
        self.set_fig()

    def set_fig(self):
        self.fig = make_subplots(rows=len(self.rowHeights), cols=1, shared_xaxes=True, 
                vertical_spacing=0.02,
                row_width=self.rowHeights)

    def get_volume_colours(self, df: pd.DataFrame):
        clrred = 'rgba(255, 59, 59, 0.8)'
        clrgrn = 'rgba(0, 255, 0, 0.8)'
        return [clrred if df['open'].iat[x] >= df['close'].iat[x] else clrgrn for x in range(len(df))]

    def add_volume(self, df: pd.DataFrame):
        self.fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["volume"],
                name='Volume',
                marker=dict(color=self.get_volume_colours(df)),
                opacity=0.8
            ),
            row=2, col=1
        )
        return self.fig

    def add_candlestick(self, df: pd.DataFrame):
        self.fig.add_trace(
            go.Candlestick(
                x    = df.index,
                open = df['open'],
                high = df['high'],
                low  = df['low'],
                close= df['close'],
                increasing=dict(line=dict(width=1, color='rgba(0, 255, 0, 0.7)'), fillcolor='rgba(0, 255, 0, 0.3)'),
                decreasing=dict(line=dict(width=1, color='rgba(255, 59, 59, 0.7)'), fillcolor='rgba(255, 59, 59, 0.3)'),
                name='Candlestick',  
            ),
            row=1, col=1
        )

        # Update x-axis to treat x values as categories
        self.fig.update_xaxes(type='category', row=1, col=1)
        return self.fig
    
    def add_day_dividers(self, df: pd.DataFrame):
        # Add vertical lines at the start of each new day
        previous_date = None
        for timestamp in df.index:
            current_date = timestamp.date()
            if current_date != previous_date:
                self.fig.add_shape(
                    type="line",
                    x0=timestamp,
                    y0=0,
                    x1=timestamp,
                    y1=1,
                    xref="x",
                    yref="paper",
                    line=dict(
                        color="DarkGrey",
                        width=1,
                        dash="dash",
                    ),
                )
                previous_date = current_date

    def add_layout_and_format(self):


        self.fig.update_layout(
            title=self.title,
            xaxis_rangeslider_visible=False,
            xaxis=dict(
                showgrid=False,
                showline=True,
                tickmode='auto', 
                nticks=10, 
                showticklabels=True,
                rangeslider=dict(visible=False)
            ),
            yaxis=dict(
                title="Price",
                gridcolor="rgba(255, 255, 255, 0.1)",
                showgrid=True
            ),
            xaxis_rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(step="all")
                ]),
                font=dict(color='rgb(179, 179, 179)'),
                bgcolor='rgb(60, 60, 60)',
                activecolor='rgb(83, 100, 105)'
            ),
            plot_bgcolor='rgb(0,0,0)', # black
            paper_bgcolor='rgb(0,0,0)', # black
            font=dict(color='white'),
            bargap=0.1,
            boxgroupgap=0,
            height=self.height,
            width=self.width
        )

        self.fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgb(50, 50, 50)')
        self.fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgb(50, 50, 50)')
        return self.fig

    def add_candles_and_volume(self, df: pd.DataFrame):
        self.add_candlestick(df)
        self.add_volume(df)
        self.add_layout_and_format()
        self.add_day_dividers(df)
        return self.fig
    
    def add_trading_hours(self, df: pd.DataFrame, times: List[Tuple[str]]):
        def find_matching_datetimes(df, time_ranges):
            time_ranges = [(datetime.datetime.strptime(start, '%H:%M').time(), datetime.datetime.strptime(end, '%H:%M').time()) for start, end in time_ranges]
            dates = pd.Series(df.index.date)
            times = pd.Series(df.index.time)
            
            matching_datetimes = []
            for date in dates.unique():
                day_times = times[dates == date]
                day_index = df.index[dates == date]
                for start, end in time_ranges:
                    matching_times = day_index[(day_times >= start) & (day_times <= end)]
                    if not matching_times.empty:
                        matching_datetimes.append((min(matching_times), max(matching_times)))
            return matching_datetimes

        trading_hours = find_matching_datetimes(df, times)

        min_time = df.index.min()
        max_time = df.index.max()
        line_alpha = 0.2
        shape_alpha = 0.05
        shape_colour = "LightSkyBlue"

        trading_hours.sort()

        # Adjust trading hours to the DataFrame's time range
        trading_hours = [(max(start, min_time), min(end, max_time)) for start, end in trading_hours]

        if not trading_hours:
            return  # No valid trading hours after adjustment, exit the function

        self.fig.add_shape(
            type="rect",
            xref="x", yref="paper",
            x0=min_time, y0=0,
            x1=trading_hours[0][0], y1=1,
            fillcolor=shape_colour,
            opacity=shape_alpha,
            layer="below",
            line_width=0,
        )

        for i in range(len(trading_hours) - 1):
            self.fig.add_shape(
                type="rect",
                xref="x", yref="paper",
                x0=trading_hours[i][1], y0=0,
                x1=trading_hours[i+1][0], y1=1,
                fillcolor=shape_colour,
                opacity=shape_alpha,
                layer="below",
                line_width=0,
            )

        self.fig.add_shape(
            type="rect",
            xref="x", yref="paper",
            x0=trading_hours[-1][1], y0=0,
            x1=max_time, y1=1,
            fillcolor=shape_colour,
            opacity=shape_alpha,
            layer="below",
            line_width=0,
        )

        for start, end in trading_hours:
            for time in [start, end]:
                self.fig.add_shape(
                    type="line",
                    xref="x", yref="paper",
                    x0=time, y0=0,
                    x1=time, y1=1,
                    line=dict(
                        color=f"rgba(128, 128, 128, {line_alpha})",
                        width=1,
                        dash="dash",
                    )
                )
    
    def show(self, width:int = 800, height:int = 800):
        needs_update = False
        if width:
            self.width = width
            needs_update = True
        if height:
            self.height = height
            needs_update = True
        if needs_update: 
            self.fig.update_layout(height=self.height, width=self.width)
        self.fig.show()

    def refesh(self, df: pd.DataFrame):
        self.set_fig()
        self.add_candles_and_volume(df)

    def add_ta(self, data: pd.Series | pd.DataFrame, style: Dict[str, Any] | list[Dict[str, Any]], chart_type: str, row:int=1) -> None:
        """Adds ta's to the chart

        args:
        data: pd.Series | pd.DataFrame: The data to be added to the chart
        style: Dict[str, Any]: The style of the data
        chart_type: str: The type of chart to be added
        """

        # make styles consistent
        style = [style] if not isinstance(style, list) else style

        if chart_type == 'line':
            if isinstance(data, pd.Series):
                self.fig.add_trace(go.Scatter(x=data.index, y=data, name=data.name, line=style[0]), row=row, col=1)
            elif isinstance(data, pd.DataFrame):
                for column, stl in zip(data.columns, style):
                    self.fig.add_trace(go.Scatter(x=data.index, y=data[column], name=column, line=stl), row=row, col=1)

        # Add MACD subplot if provided
        if chart_type == 'macd' and isinstance(data, pd.DataFrame):
            macd_col = [col for col in data.columns if col.endswith('MACD')][0]
            signal_col = [col for col in data.columns if col.endswith('Signal')][0]
            hist_col = [col for col in data.columns if col.endswith('Histogram')][0]
            
            hist_style = {k: v for k, v in style[2].items() if k != 'width'}  # Remove 'width' from style
            self.fig.add_trace(go.Bar(x=data.index, y=data[hist_col], name=hist_col, marker=hist_style), row=3, col=1)
            self.fig.add_trace(go.Scatter(x=data.index, y=data[macd_col], name=macd_col, line=style[0]), row=3, col=1)
            self.fig.add_trace(go.Scatter(x=data.index, y=data[signal_col], name=signal_col, line=style[1]), row=3, col=1)
            
            self.fig.update_layout()  # Adjust the height to accommodate the new subplot
        
        if chart_type == 'points' and isinstance(data, pd.DataFrame):
            for column, stl in zip(data.columns, style):
                # Ensure 'size' and 'opacity' are set in the marker style if not provided
                stl.setdefault('size', 10)  # Default size if not provided
                stl.setdefault('color', 'blue')  # Default color if not provided
                stl.setdefault('opacity', 0.8)  # Default opacity if not provided

                self.fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[column],
                    name=column,
                    mode='markers',
                    marker=stl
                ), row=row, col=1)

        if chart_type == 'support_resistance':
            self.add_support_resistance(data, style)

    def add_support_resistance(self, data, style):
        support_col = [col for col in data.columns if col.endswith('Support')][0]
        resistance_col = [col for col in data.columns if col.endswith('Resistance')][0]
        # support_band_col = 'Support_Band'
        # resistance_band_col = 'Resistance_Band'

        # Extract the most recent support and resistance levels
        recent_support = data[support_col].iloc[-1]
        recent_resistance = data[resistance_col].iloc[-1]
        # support_band = data[support_band_col].iloc[-1]
        # resistance_band = data[resistance_band_col].iloc[-1]

        print("Recent support level:", recent_support)
        print("Recent resistance level:", recent_resistance)
        # print("Support band levels:", support_band)
        # print("Resistance band levels:", resistance_band)

        # Add the most recent support and resistance levels as lines
        self.fig.add_shape(
            type="line",
            x0=data.index[0], x1=data.index[-1], y0=recent_support, y1=recent_support,
            line=dict(color=style[0]['color'], width=style[0]['width'], dash="dash"),
            row=1, col=1
        )
        self.fig.add_shape(
            type="line",
            x0=data.index[0], x1=data.index[-1], y0=recent_resistance, y1=recent_resistance,
            line=dict(color=style[1]['color'], width=style[1]['width'], dash="dash"),
            row=1, col=1
        )

        # # Add shaded areas for the support and resistance bands
        # self.fig.add_shape(
        #     type="rect",
        #     x0=data.index[0], x1=data.index[-1], y0=min(support_band), y1=max(support_band),
        #     fillcolor=style[0]['color'], opacity=0.2, line_width=0,
        #     row=1, col=1
        # )
        # self.fig.add_shape(
        #     type="rect",
        #     x0=data.index[0], x1=data.index[-1], y0=min(resistance_band), y1=max(resistance_band),
        #     fillcolor=style[1]['color'], opacity=0.2, line_width=0,
        #     row=1, col=1
        # )

        print("Finished adding support and resistance levels")