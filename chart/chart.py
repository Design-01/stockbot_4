import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Tuple, Dict, Any
import datetime
from plotly.colors import hex_to_rgb
import numpy as np
import plotly.io as pio
from dataclasses import dataclass, field


# used to store the chart arguments for Chart.add_multi_ta
@dataclass
class ChartArgs:
    style: Dict[str, Any] | list[Dict[str, Any]] = field(default_factory=dict)
    chartType: str = 'line'
    row: int = 1
    nameCol: pd.Series = None
    columns: List[str] = None

class Chart:

    def __init__(self, title: str = "Candlestick Chart", rowHeights:List[float] = [0.6, 0.2, 0.1, 0.1], height:int = 800, width:int = 800):
        self.title = title
        self.rowHeights = rowHeights
        self.height = height
        self.width = width
        self.fig = None
        self.ta = []
        self.set_fig()
    

    def set_fig(self):
        self.fig = make_subplots(
            rows=len(self.rowHeights), 
            cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.02,
            row_width=self.rowHeights)
    
        
    def store_ta(self, ta_list):
        self.ta = ta_list

    def save_chart(self, filename, format='png'):
        """
        Save a Plotly figure as an image file
        
        Parameters:
        fig: plotly figure object
        filename: str, path where to save the file (without extension)
        format: str, either 'png' or 'pdf'
        """
        if format.lower() not in ['png', 'pdf']:
            raise ValueError("Format must be either 'png' or 'pdf'")
            
        # Add file extension if not present
        if not filename.lower().endswith(f'.{format}'):
            filename = f"{filename}.{format}"
        
        # Save the figure
        if format.lower() == 'png':
            pio.write_image(self.fig, filename)
        else:
            pio.write_image(self.fig, filename, format='pdf')
        
        print(f"Chart saved as {filename}")

    def save_chart_region(self, x_start, x_end, y_min=None, y_max=None, filename='zoomed_chart.png',
                        x_padding='1D', y_padding_pct=1.0, plot:bool = False):
        """
        Saves a zoomed region of a plotly chart as an image with proper padding.
        
        Parameters:
        x_start: start point for x-axis zoom (can be datetime or index)
        x_end: end point for x-axis zoom (can be datetime or index)
        y_min: minimum y value (optional - will auto-calculate if None)
        y_max: maximum y value (optional - will auto-calculate if None)
        filename: output filename for the image
        x_padding: string for time padding (e.g., '1D' for 1 day, '12H' for 12 hours)
        y_padding_pct: percentage padding for y-axis (1.0 = 1%)
        """
        from datetime import datetime
        import pandas as pd
        
        # Create a copy of the figure to avoid modifying original
        zoomed_fig = go.Figure(self.fig)
        
        # Handle datetime types
        try:
            # Convert to pandas datetime if string
            if isinstance(x_start, str):
                x_start = pd.to_datetime(x_start)
                x_end = pd.to_datetime(x_end)
            
            # Check if datetime or Timestamp
            if isinstance(x_start, (pd.Timestamp, datetime)):
                padding_td = pd.Timedelta(x_padding)
                x_start_padded = x_start - padding_td
                x_end_padded = x_end + padding_td
            else:
                # For numeric indices, use integer padding
                x_start_padded = x_start - 1
                x_end_padded = x_end + 1
                
        except Exception as e:
            print(f"Error handling dates: {e}. Falling back to integer padding.")
            x_start_padded = x_start - 1
            x_end_padded = x_end + 1
        
        # For candlestick charts, extract OHLC data within the range
        highs = []
        lows = []
        
        for trace in self.fig.data:
            if isinstance(trace, go.Candlestick):
                # Get indices within the date range (including padding)
                mask = [(x >= x_start_padded) and (x <= x_end_padded) for x in trace.x]
                
                # Extract high and low values within range
                range_highs = [h for h, m in zip(trace.high, mask) if m]
                range_lows = [l for l, m in zip(trace.low, mask) if m]
                
                if range_highs and range_lows:
                    highs.extend(range_highs)
                    lows.extend(range_lows)
        
        # Calculate y-axis range if not provided
        if y_min is None or y_max is None:
            if highs and lows:
                price_range = max(highs) - min(lows)
                padding = price_range * (y_padding_pct / 100)
                y_min = min(lows) - padding if y_min is None else y_min
                y_max = max(highs) + padding if y_max is None else y_max
            else:
                raise ValueError("No data found in the specified date range")
        else:
            # Apply padding to provided y values
            price_range = y_max - y_min
            padding = price_range * (y_padding_pct / 100)
            y_min = y_min - padding
            y_max = y_max + padding
        
        # Update the layout with new ranges
        zoomed_fig.update_layout(
            xaxis_range=[x_start_padded, x_end_padded],
            yaxis_range=[y_min, y_max],
            yaxis_autorange=False,
            xaxis_autorange=False
        )
        
        # Save the zoomed figure
        zoomed_fig.write_image(filename)
        print(f"Zoomed chart saved as {filename}")
        if plot:
            zoomed_fig.show()
        return zoomed_fig

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
        return self.fig
    
    def add_day_dividers(self, df: pd.DataFrame):
        """
        Add vertical lines at the start of each new day, but only if the data is intraday.
        
        Args:
            df (pd.DataFrame): DataFrame with a datetime index
        """
        # Check if data is intraday by examining the time differences
        time_diff = df.index.to_series().diff().median()
        
        # If median time difference is >= 1 day, it's daily data or longer
        if time_diff >= pd.Timedelta(days=1):
            return  # Exit function for daily or longer intervals
            
        # Proceed with adding dividers for intraday data
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

    def add_horizontal_lines(self, row_heights, total_height):
        """
        Add horizontal lines to separate subplots in a Plotly figure.

        Parameters:
        fig (plotly.graph_objects.Figure): The Plotly figure to modify.
        row_heights (list of float): List of relative heights for each subplot.
        total_height (int): Total height of the figure.

        Returns:
        None
        """
        # Calculate the height of each subplot
        subplot_heights = [total_height * h for h in row_heights]

        # Calculate the Y offset for each subplot
        y_offsets = np.cumsum([0] + subplot_heights[:-1])

        # # Add horizontal lines to separate subplots
        # for y in y_offsets[1:]:  # Skip the first offset (0)
        #     self.fig.add_shape(
        #         type="line",
        #         x0=0,
        #         x1=1,
        #         y0=y / total_height,
        #         y1=y / total_height,
        #         xref='paper',
        #         yref='paper',
        #         line=dict(color="yellow", width=2)
        #     )

        # Add one line to separate the bottom two plots
        y =  y_offsets[1]  # Skip the first offset (0) which is the very bottom line
        self.fig.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=y / total_height,
            y1=y / total_height,
            xref='paper',
            yref='paper',
            line=dict(color="white", width=2)
        )

    def add_layout_and_format(self, df: pd.DataFrame= pd.DataFrame()):


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

        # ---- Format the x-axis ----
        # When not formatted Plotly creates a regular date axis so it ends up with large 
        # gaps between the dates. This function will create a more readable date axis.
        def get_smart_ticks(df, num_ticks=10, date_format='%b %d %H:%M'):
            total_bars = len(df)
            step = max(total_bars // num_ticks, 1)
            
            # Get indices at regular intervals
            tick_positions = list(range(0, total_bars, step))
            
            # Format the dates
            tick_dates = [d.strftime(date_format) for d in df.index[tick_positions]]
    
            return tick_positions, tick_dates

        tick_positions, tick_dates = get_smart_ticks(df, num_ticks=5)

        self.fig.update_xaxes(
            type='category',
            tickmode='array',
            ticktext=tick_dates,
            tickvals=tick_positions,
            tickangle=0,  # Angle the dates for better readability
            showgrid=True, gridwidth=1, gridcolor='rgb(50, 50, 50)'
        )
        # --------------------------------
        # self.fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgb(50, 50, 50)')
        self.fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgb(50, 50, 50)')
        self.add_horizontal_lines(self.rowHeights, self.height)
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
        self.add_layout_and_format(df)

    def add_ta(self, data: pd.Series | pd.DataFrame, style: Dict[str, Any] | list[Dict[str, Any]], chartType: str, row:int=1, nameCol:pd.Series=None, columns:List[str]=None) -> None:
        """Adds ta's to the chart

        args:
        data: pd.Series | pd.DataFrame: The data to be added to the chart
        style: Dict[str, Any]: The style of the data
        chartType: str: The type of chart to be added
        """

        if style == {}: return

        # make styles consistent
        style = [style] if not isinstance(style, list) else style

        columns = data.columns if columns is None else columns

        if chartType == 'line':
            if isinstance(data, pd.Series):
                self.fig.add_trace(go.Scatter(x=data.index, y=data, name=data.name, line=style[0]), row=row, col=1)
            elif isinstance(data, pd.DataFrame):
                for column, stl in zip(columns, style):
                    self.fig.add_trace(go.Scatter(x=data.index, y=data[column], name=column, line=stl), row=row, col=1)

        if chartType == 'lines+markers':
            # print(f"Adding nameCol type: {type(nameCol)} nameCol: {nameCol},  {chartType} to chart")
            
            if isinstance(data, pd.Series):
            #     labels = ''
            #     if isinstance(nameCol, pd.Series):
            #         print(f"Adding nameCol ... isinstance pd.Series... type: {type(nameCol)} nameCol: {nameCol},  {chartType} to chart")
            #         labels = nameCol.to_list()
            #         print(labels)
                self.fig.add_trace(go.Scatter(
                    x=data.index, 
                    y=data, 
                    name=data.name, 
                    line=style[0], 
                    mode=chartType,
                    text='test'
                    ), row=row, col=1)
                
            elif isinstance(data, pd.DataFrame):
                for column, stl in zip(columns, style):
                    self.fig.add_trace(go.Scatter(x=data.index, y=data[column], name=column, line=stl, mode=chartType), row=row, col=1)



        # Add MACD subplot if provided
        if chartType == 'macd' and isinstance(data, pd.DataFrame):
            macd_col = [col for col in columns if col.endswith('MACD')][0]
            signal_col = [col for col in columns if col.endswith('Signal')][0]
            hist_col = [col for col in columns if col.endswith('Histogram')][0]
            
            # self.fig.add_trace(go.Bar(x=data.index, y=data[hist_col], name=hist_col, marker=hist_style), row=3, col=1)
            self.fig.add_trace(go.Scatter(x=data.index, y=data[macd_col], name=macd_col, line=style[0]), row=3, col=1)
            self.fig.add_trace(go.Scatter(x=data.index, y=data[signal_col], name=signal_col, line=style[1]), row=3, col=1)
            self.fig.add_trace(go.Bar(x=data.index, y=data[hist_col], name=hist_col, marker=style[2]), row=3, col=1)
            
            self.fig.update_layout()  # Adjust the height to accommodate the new subplot
        
        if chartType == 'points' and isinstance(data, pd.DataFrame):
            for column, stl in zip(columns, style):
                # Ensure 'size' and 'opacity' are set in the marker style if not provided
                stl.setdefault('size', 10)  # Default size if not provided
                stl.setdefault('color', 'blue')  # Default color if not provided
                stl.setdefault('opacity', 0.8)  # Default opacity if not provided

                labels = ''
                if isinstance(nameCol, pd.Series):
                    labels = nameCol.to_list()


                self.fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[column],
                    name=column,
                    mode='markers',
                    marker=stl,
                    text=labels
                ), row=row, col=1)

        if chartType == 'support_resistance':
            self.add_support_resistance(data, style)

        if chartType.upper() in ['CONS', 'RECT']:
            self.add_rectangle(data, style, chartType.upper())
        
        if chartType == 'trendlines':
            trend_cols = [col for col in columns if 'TREND' in col]
            for col in trend_cols:
                mask = ~data[col].isna()
                if mask.any():
                    self.fig.add_trace(go.Scatter(
                        x=data.index[mask],
                        y=data[col][mask],
                        name=col,
                        line=style[0]  # Use the same style for all trend lines
                    ), row=row, col=1)


    def add_multi_ta(self, data: pd.DataFrame, chartArgs: List[ChartArgs]) -> None:
        """
        Adds multiple technical analysis indicators to the chart.

        Args:
        data (pd.DataFrame): DataFrame containing technical analysis indicators
        chartArgs (List[ChartArgs]): List of ChartArgs objects containing indicator details
        """
        for chartArg in chartArgs:
            self.add_ta(data, chartArg.style, chartArg.chartType, chartArg.row, chartArg.nameCol)

        

    def add_support_resistance(self, data: pd.DataFrame, style: List[Dict[str, Any]]) -> None:
        """
        Adds support and resistance levels to the chart with upper and lower bounds.

        Args:
        data (pd.DataFrame): DataFrame containing support and resistance levels with bounds
        style (List[Dict[str, Any]]): List of style dictionaries for support and resistance
        """

        def create_traces(level_col, upper_col, lower_col, style):
            """
            Creates Plotly traces for support/resistance levels with properly bounded shaded areas.
            
            This function generates a set of traces that visualize support/resistance levels with:
            1. A main level line (solid)
            2. Upper and lower bound lines (dotted)
            3. Shaded areas between upper and lower bounds
            
            The shading is carefully constructed to only appear where both upper and lower bounds
            have valid data (non-NaN values). This prevents the shading from extending beyond
            where the support/resistance lines actually exist.
            
            The function breaks the data into continuous segments and creates separate fill
            areas for each segment to ensure proper visualization when lines have gaps.
            
            Args:
                level_col (str): Column name for the main support/resistance level
                upper_col (str): Column name for the upper bound
                lower_col (str): Column name for the lower bound
                style (dict): Dictionary with styling parameters including:
                    - dash (str): Line style for main level ('solid', 'dash', 'dot', etc.)
                    - width (int): Line width in pixels
                    - fillcolour (str): Color for shaded area (RGBA format)
                    - main_line_colour (str): Color for main level line (RGBA format)
                    - zone_edge_colour (str): Color for upper/lower bound lines (RGBA format)
            
            Returns:
                list: List of Plotly go.Scatter traces ready to be added to a figure
            """
            # Extract styling parameters from the style dictionary with defaults
            dash = style.get('dash', 'solid')                            # Line style for main level
            width = style.get('width', 2)                                # Line width in pixels
            fillcolour = style.get('fillcolour', 'rgba(0, 0, 255, 0.1)') # Color for shaded area
            main_line_colour = style.get('main_line_colour', 'rgba(0, 0, 255, 0.8)')  # Main line color
            zone_edge_colour = style.get('zone_edge_colour', 'rgba(0, 0, 255, 0.2)')  # Edge line color

            traces = []
            
            # PART 1: IDENTIFY CONTINUOUS SEGMENTS
            # We need to find where both upper and lower bounds have valid data to properly create fills
            mask = ~(data[upper_col].isna() | data[lower_col].isna())  # True where both lines have values
            
            # Find the start and end indices of each continuous segment
            segment_starts = []  # Will hold starting indices of segments
            segment_ends = []    # Will hold ending indices of segments
            
            in_segment = False   # Tracks whether we're currently in a valid segment
            for i, valid in enumerate(mask):
                if valid and not in_segment:
                    # We just entered a valid segment
                    segment_starts.append(i)
                    in_segment = True
                elif not valid and in_segment:
                    # We just exited a valid segment
                    segment_ends.append(i - 1)
                    in_segment = False
            
            # If the last segment continues until the end of data, add the last index
            if in_segment:
                segment_ends.append(len(mask) - 1)
            
            # PART 2: CREATE FILL AREAS FOR EACH SEGMENT
            # Iterate through each identified segment and create a separate fill area
            for start, end in zip(segment_starts, segment_ends):
                # Extract the relevant slice of data for this segment
                segment_indices = data.index[start:end+1]
                upper_values = data[upper_col].iloc[start:end+1].tolist()
                lower_values = data[lower_col].iloc[start:end+1].tolist()
                
                # Create the x and y coordinates for the fill polygon by:
                # 1. Going forward along the upper bound (left to right)
                # 2. Then going backward along the lower bound (right to left)
                # This creates a closed path that Plotly can fill with 'toself'
                x_fill = segment_indices.tolist() + segment_indices.tolist()[::-1]
                y_fill = upper_values + lower_values[::-1]
                
                # Add the fill area for this segment
                traces.append(go.Scatter(
                    x=x_fill,
                    y=y_fill,
                    fill='toself',                        # Fill the path formed by x and y
                    fillcolor=fillcolour,                 # Use the specified fill color
                    line=dict(color='rgba(0, 0, 0, 0)'),  # Transparent line (no visible border)
                    hoverinfo='skip',                     # Disable hover tooltips on the fill
                    showlegend=False,                     # Don't show in legend
                    name=f"{level_col} Zone Segment"      # Naming for debugging purposes
                ))
            
            # PART 3: ADD THE ACTUAL LINES
            # These will be drawn on top of the fill areas
            
            # Main level line (will usually be in the middle of the filled area)
            traces.append(go.Scatter(
                x=data.index, 
                y=data[level_col], 
                name=level_col,                           # Name that will appear in legend
                line=dict(color=main_line_colour, width=width, dash=dash)  # Styling
            ))
            
            # Upper bound (dotted line)
            traces.append(go.Scatter(
                x=data.index, 
                y=data[upper_col], 
                name=f"{level_col} Upper",                # Name that will appear in legend
                line=dict(color=zone_edge_colour, width=width, dash='dot')  # Always dotted
            ))
            
            # Lower bound (dotted line)
            traces.append(go.Scatter(
                x=data.index, 
                y=data[lower_col], 
                name=f"{level_col} Lower",                # Name that will appear in legend
                line=dict(color=zone_edge_colour, width=width, dash='dot')  # Always dotted
            ))
            
            return traces

        def add_traces(cols, style):
            if len(cols) == 0:
                return
            # print(f"Adding support/resistance traces: {cols}")
            traces = create_traces(cols[0], cols[1], cols[2], style)
            for trace in traces:
                self.fig.add_trace(trace, row=1, col=1)

        if not style: return
        support_style, resistance_style = style

        # Support traces
        add_traces([col for col in data.columns if col.startswith('Sup_1')], support_style)
        add_traces([col for col in data.columns if col.startswith('Sup_2')], support_style)

        # Resistance traces
        add_traces([col for col in data.columns if col.startswith('Res_1')], resistance_style)
        add_traces([col for col in data.columns if col.startswith('Res_2')], resistance_style)
   
    def add_rectangle(self, data: pd.DataFrame, style: List[Dict[str, Any]], chartType='') -> None:
        """
        Adds support and resistance levels to the chart with upper and lower bounds.

        Args:
        data (pd.DataFrame): DataFrame containing support and resistance levels with bounds
        style (List[Dict[str, Any]]): List of style dictionaries for support and resistance
        """
        if not style:
            return
            
        support_style, resistance_style = style

        def create_traces(upper_col, lower_col, style, name):
            color = style.get('color', 'blue')
            dash = style.get('dash', 'solid')
            width = style.get('width', 2)
            fillcolour = style.get('fillcolour', 'rgba(0, 0, 255, 0.1)')
            
            # Upper bound (dashed)
            yield go.Scatter(
                x=data.index, 
                y=data[upper_col], 
                name=f"{name} {upper_col.split('_')[-1]} Upper",
                line=dict(color=color, width=width-1, dash='dash')
            )
            
            # Lower bound (dashed)
            yield go.Scatter(
                x=data.index, 
                y=data[lower_col], 
                name=f"{name} {lower_col.split('_')[-1]} Lower",
                line=dict(color=color, width=width-1, dash='dash')
            )
            
            # Filter out NaN values for the fill to work correctly
            mask = data[upper_col].notna() & data[lower_col].notna()
            x_data = data.index[mask]
            upper_data = data[upper_col][mask]
            lower_data = data[lower_col][mask]
            
            # Shaded area between upper and lower bounds
            yield go.Scatter(
                x=x_data.tolist() + x_data.tolist()[::-1],
                y=upper_data.tolist() + lower_data.tolist()[::-1],
                fill='toself',
                fillcolor=fillcolour,
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name=f"{name} {upper_col.split('_')[-1]} Zone"
            )

        # Get all rectangle columns
        name = chartType.upper()
        rect_cols = [col for col in data.columns if col.startswith(name)]
        rect_pairs = []
        
        # Group upper and lower bounds
        for i in range(1, len(rect_cols)//2 + 1):
            upper = f"{name}_UPPER_{i}"
            lower = f"{name}_LOWER_{i}"
            if upper in rect_cols and lower in rect_cols:
                rect_pairs.append((upper, lower))
        
        # Add traces for each rectangle
        for upper_col, lower_col in rect_pairs:
            for trace in create_traces(upper_col, lower_col, support_style, name):
                self.fig.add_trace(trace, row=1, col=1)

    def add_line(self, data: pd.Series, style: Dict[str, Any], row:int=1) -> None:
        """Adds a line to the chart

        args:
        data: pd.Series: The data to be added to the chart
        style: Dict[str, Any]: The style of the data
        """

        if style == {}: return
        self.fig.add_trace(go.Scatter(x=data.index, y=data, name=data.name, line=style), row=row, col=1)

    


