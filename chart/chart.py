import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Tuple, Dict, Any, Literal, Union
import datetime
import numpy as np
import plotly.io as pio
from dataclasses import dataclass, field
import plotly.colors as pc
from matplotlib.colors import CSS4_COLORS

from chart.chart_args import PlotArgs


def get_rgba(color, opacity=1.0):
    """
    Convert a color name or hex to rgba format for plotly with custom opacity.
    
    Parameters:
    -----------
    color : str
        Color name (e.g., 'green') or hex code (e.g., '#00FF00')
    opacity : float
        Opacity value between 0 and 1
        
    Returns:
    --------
    str
        rgba string in format 'rgba(r,g,b,a)' that plotly can use
    """
    
    # Handle color name to hex conversion
    if not color.startswith('#'):
        # Try to get color from matplotlib's CSS4 colors
        try:
            color = CSS4_COLORS[color.lower()]
        except KeyError:
            # Default to a standard color if not found
            color = '#000000'  # Default to black
    
    # Convert hex to RGB
    rgb = pc.hex_to_rgb(color)
    
    # Return as rgba string
    return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})'






class Chart:

    def __init__(self, title: str = "Candlestick Chart", rowHeights:List[float] = [0.6, 0.2, 0.1, 0.1], height:int = 800, width:int = 800):
        self.title = title
        self.rowHeights = rowHeights
        self.height = height
        self.width = width
        self.fig = None
        self.plot_function_map = {}
        self.ta = []
        self.set_fig()
        self._register_plot_functions()
    

    def set_fig(self):
        self.fig = make_subplots(
            rows=len(self.rowHeights), 
            cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.02,
            row_width=self.rowHeights)
    
    def _register_plot_functions(self):
        """Register the mapping of plot types to their handler functions"""
        self.plot_function_map = {
            'lines': self.add_line,
            'lines+markers': self.add_line,  # Consolidated to use the same handler
            'lines+markers+text': self.add_line,  # Consolidated to use the same handler
            'zone': self.add_zone,
            'points': self.add_points,
            'points+text': self.add_points,  # Consolidated to use the same handler
            'buysell': self.add_points,
        }
        
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

    # =====  Adding plots to the chart ======   

    def add_line(self, df:pd.DataFrame, plot_args:PlotArgs):
        """
        Add line traces to a plotly figure based on the provided ChartArgItem.
        Will handle both single values and lists for styling attributes.
        
        This method handles 'line', 'line+marker', and 'line+marker+label' plot types.
        """
        # print(f"Chart.add_line :: {plot_args=}")
        row = plot_args.plotRow
        col = plot_args.plotCol
        idx = df.index
        mode = plot_args.plotType
        
        # Convert all attributes to lists for consistent handling
        columns = [plot_args.dataCols] if not isinstance(plot_args.dataCols, list) else plot_args.dataCols
        columns = [col for col in columns if col in df.columns]  # Filter out invalid columns
        if not columns:
            print(f"Chart.add_line :: No valid columns for {plot_args.name=} found in {plot_args.dataCols=} , {plot_args=}") 
            return
        
        # Convert other style attributes to lists (or empty lists if None)
        names = [plot_args.name] if not isinstance(plot_args.name, list) else plot_args.name if plot_args.name else []
        labels = [plot_args.textCols] if not isinstance(plot_args.textCols, list) else plot_args.textCols if plot_args.textCols else []
        colours = [plot_args.colours] if not isinstance(plot_args.colours, list) else plot_args.colours if plot_args.colours else []
        opacities = [plot_args.opacities] if not isinstance(plot_args.opacities, list) else plot_args.opacities if plot_args.opacities else []
        dashes = [plot_args.dashes] if not isinstance(plot_args.dashes, list) else plot_args.dashes if plot_args.dashes else []
        line_widths = [plot_args.lineWidths] if not isinstance(plot_args.lineWidths, list) else plot_args.lineWidths if plot_args.lineWidths else []
        marker_sizes = [plot_args.markerSizes] if not isinstance(plot_args.markerSizes, list) else plot_args.markerSizes if plot_args.markerSizes else []
        text_positions = [plot_args.textPositions] if not isinstance(plot_args.textPositions, list) else plot_args.textPositions if plot_args.textPositions else []
        
        # Add a trace for each column
        for i, column in enumerate(columns):
            # Get data for the current column
            data = df[column]
            
            # Create the line style dictionary
            line_style = {}
            
            # Get the current color and opacity
            current_color = colours[i % len(colours)] if colours else None
            current_opacity = opacities[i % len(opacities)] if opacities else 1.0
            
            # Apply color with opacity if both are available
            if current_color:
                # Use get_rgba to handle the opacity in the color
                line_style["color"] = get_rgba(current_color, current_opacity) if current_opacity is not None else current_color
                
            # Apply line width if available
            if line_widths:
                line_style["width"] = line_widths[i % len(line_widths)]
                
            # Apply dash style if available
            if dashes:
                line_style["dash"] = dashes[i % len(dashes)]
            
            # Create marker style - only used if we need markers
            marker_style = {}
            if 'marker' in mode and current_color:
                marker_style["color"] = line_style.get("color")
            
            if 'marker' in mode and marker_sizes:
                marker_style["size"] = marker_sizes[i % len(marker_sizes)]
            
            # Get label data if we're using text
            text_data = None
            if 'text' in mode:
                # Get the label data if specified
                label_column = labels[i % len(labels)] if i < len(labels) and labels else None
                text_data = df[label_column].astype(str) if label_column and label_column in df.columns else None
            
            # Get text position if specified
            text_position = "top center"  # Default position
            if text_positions and i < len(text_positions) and text_positions[i] is not None:
                text_position = text_positions[i]
            
            # Get trace name (use provided name, label, or column name)
            trace_name = None
            if names and i < len(names) and names[i] is not None:
                trace_name = names[i]
            elif labels and i < len(labels) and labels[i] is not None:
                trace_name = labels[i]
            else:
                trace_name = column
                
            # Add the trace with all configurations
            self.fig.add_trace(
                go.Scatter(
                    x=idx,
                    y=data,
                    name=trace_name,
                    text=text_data,
                    textposition=text_position if text_data is not None else None,
                    line=line_style,
                    marker=marker_style if marker_style else None,
                    mode=mode  # Set mode based on plot type
                ),
                row=row,
                col=col
            )
        
    def add_points(self, df: pd.DataFrame, plot_args: PlotArgs):
        """
        Add points (markers) to the plot, optionally with labels.
        Handles 'points', 'points+text', and 'buy_sell' plot types.
        """
        row = plot_args.plotRow
        col = plot_args.plotCol
        idx = df.index
        plot_type = plot_args.plotType
        
        # Convert all attributes to lists for consistent handling
        columns = [plot_args.dataCols] if not isinstance(plot_args.dataCols, list) else plot_args.dataCols
        
        # Convert other style attributes to lists
        names = [plot_args.name] if not isinstance(plot_args.name, list) else plot_args.name if plot_args.name else []
        labels = [plot_args.textCols] if not isinstance(plot_args.textCols, list) else plot_args.textCols if plot_args.textCols else []
        colours = [plot_args.colours] if not isinstance(plot_args.colours, list) else plot_args.colours if plot_args.colours else []
        opacities = [plot_args.opacities] if not isinstance(plot_args.opacities, list) else plot_args.opacities if plot_args.opacities else []
        marker_sizes = [plot_args.markerSizes] if not isinstance(plot_args.markerSizes, list) else plot_args.markerSizes if plot_args.markerSizes else []
        text_positions = [plot_args.textPositions] if not isinstance(plot_args.textPositions, list) else plot_args.textPositions if plot_args.textPositions else []
        text_sizes = [plot_args.textSizes] if not isinstance(plot_args.textSizes, list) else plot_args.textSizes if plot_args.textSizes else []
        
        # Add support for marker symbols
        marker_symbols = [plot_args.markerSymbols] if not isinstance(plot_args.markerSymbols, list) else plot_args.markerSymbols if plot_args.markerSymbols else []
        
        # Determine the mode based on plot type
        mode = 'markers'
        if plot_type in ['points+text', 'buy_sell']:
            mode = 'markers+text'
        
        # Add a trace for each column
        for i, column in enumerate(columns):
            # Get data for the current column
            data = df[column]
            
            # Get current styling
            current_color = colours[i % len(colours)] if colours else None
            current_opacity = opacities[i % len(opacities)] if opacities else 1.0
            
            # Create marker style
            marker_style = {}
            if current_color:
                marker_style["color"] = get_rgba(current_color, current_opacity) if current_opacity is not None else current_color
            
            if marker_sizes:
                marker_style["size"] = marker_sizes[i % len(marker_sizes)]
            
            # Add marker symbol if specified
            if marker_symbols:
                marker_style["symbol"] = marker_symbols[i % len(marker_symbols)]
            
            # Get label data if we're using text
            text_data = None
            if 'text' in mode:
                # Get the label data if specified
                label_column = labels[i % len(labels)] if i < len(labels) and labels else None
                text_data = df[label_column].astype(str) if label_column and label_column in df.columns else None
            
            # Get text position if specified
            text_position = "top center"  # Default position
            if text_positions and i < len(text_positions) and text_positions[i] is not None:
                text_position = text_positions[i]
            
            # Get trace name
            trace_name = names[i % len(names)] if names else column
            
            # Setup text font properties
            text_font = {}
            # Use the same color as the marker for text
            if current_color:
                text_font["color"] = get_rgba(current_color, current_opacity) if current_opacity is not None else current_color
            
            # Apply text size if specified
            if text_sizes:
                text_font["size"] = text_sizes[i % len(text_sizes)]
            
            # Add the trace with appropriate mode
            self.fig.add_trace(
                go.Scatter(
                    x=idx,
                    y=data,
                    name=trace_name,
                    marker=marker_style,
                    text=text_data,
                    textposition=text_position if text_data is not None else None,
                    mode=mode,
                    textfont=text_font
                ),
                row=row,
                col=col
            )

    def add_zone(self, df: pd.DataFrame, plot_args: PlotArgs):
            """
            Creates Plotly traces for support/resistance levels with properly bounded shaded areas.
            
            This function generates a set of traces that visualize support/resistance levels with:
            1. Upper and lower bound lines (dotted)
            2. Shaded areas between upper and lower bounds
            
            The colContains parameter is expected to contain patterns for identifying pairs of columns.
            """
            row = plot_args.plotRow
            col = plot_args.plotCol
            
            # Check if we have colContains defined
            if not plot_args.colContains:
                print("Zone plot requires colContains to identify column pairs")
                return
                
            # Convert colContains to a list if it's not already
            col_patterns = [plot_args.colContains] if not isinstance(plot_args.colContains, list) else plot_args.colContains
            columns = [plot_args.dataCols] if not isinstance(plot_args.dataCols, list) else plot_args.dataCols
            
            # Check if we have at least one pattern
            if not col_patterns:
                print("Zone plot requires at least one pattern in colContains")
                return
                
            # Get styling attributes as lists
            colours = [plot_args.colours] if not isinstance(plot_args.colours, list) else plot_args.colours if plot_args.colours else []
            fill_colours = [plot_args.fillColours] if not isinstance(plot_args.fillColours, list) else plot_args.fillColours if plot_args.fillColours else []
            fill_opacities = [plot_args.fillOpacites] if not isinstance(plot_args.fillOpacites, list) else plot_args.fillOpacites if plot_args.fillOpacites else []
            dashes = [plot_args.dashes] if not isinstance(plot_args.dashes, list) else plot_args.dashes if plot_args.dashes else []
            line_widths = [plot_args.lineWidths] if not isinstance(plot_args.lineWidths, list) else plot_args.lineWidths if plot_args.lineWidths else []
            
            # Process each pattern to create zones
            for i, pattern in enumerate(col_patterns):
                # Find columns matching this pattern
                matching_cols = [col for col in columns if pattern in col]
                
                # Skip if we don't have enough columns
                if len(matching_cols) < 2:
                    print(f"Pattern '{pattern}' doesn't match at least 2 columns, skipping")
                    continue
                    
                # Sort columns so we have a consistent order
                matching_cols = sorted(matching_cols)
                
                # Get current styling for this zone
                dash = dashes[i % len(dashes)] if dashes else 'dot'
                width = line_widths[i % len(line_widths)] if line_widths else 2
                
                # Get current zone edge color
                zone_edge_colour = colours[i % len(colours)] if colours else 'rgba(0, 0, 0, 0.5)'
                
                # Get current fill color and opacity
                fill_colour = fill_colours[i % len(fill_colours)] if fill_colours else 'rgba(0, 0, 255, 0.1)'
                fill_opacity = fill_opacities[i % len(fill_opacities)] if fill_opacities else 0.1
                
                # Convert fill color to rgba with opacity
                fill_rgba = get_rgba(fill_colour, fill_opacity) if isinstance(fill_colour, str) else 'rgba(0, 0, 255, 0.1)'
                
                # Get name for this zone
                name = f"{pattern} Zone" if pattern else f"Zone {i+1}"
                
                # Process each pair of columns as a zone
                for j in range(0, len(matching_cols) - 1, 2):
                    # Get upper and lower columns for this pair
                    upper_col = matching_cols[j]
                    lower_col = matching_cols[j+1]
                    
                    # Create the zone for this pair of columns
                    self._create_zone_for_column_pair(df, upper_col, lower_col, name, zone_edge_colour, fill_rgba, dash, width, row, col)
    
    def _create_zone_for_column_pair(self, df, upper_col, lower_col, name, edge_color, fill_color, dash, width, row, col):
        """
        Helper method to create a zone between two columns.
        This handles the segment identification and trace creation for a single pair of columns.
        """
        # IDENTIFY CONTINUOUS SEGMENTS
        # Find where both upper and lower bounds have valid data
        mask = ~(df[upper_col].isna() | df[lower_col].isna())
        
        # Find the start and end indices of each continuous segment
        segment_starts = []
        segment_ends = []
        
        in_segment = False
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
        
        # CREATE FILL AREAS FOR EACH SEGMENT
        traces = []
        for start, end in zip(segment_starts, segment_ends):
            # Extract the relevant slice of data for this segment
            segment_indices = df.index[start:end+1]
            upper_values = df[upper_col].iloc[start:end+1].tolist()
            lower_values = df[lower_col].iloc[start:end+1].tolist()
            
            # Create the x and y coordinates for the fill polygon by:
            # 1. Going forward along the upper bound (left to right)
            # 2. Then going backward along the lower bound (right to left)
            # This creates a closed path that Plotly can fill with 'toself'
            x_fill = segment_indices.tolist() + segment_indices.tolist()[::-1]
            y_fill = upper_values + lower_values[::-1]
            
            # Add the fill area for this segment
            self.fig.add_trace(
                go.Scatter(
                    x=x_fill,
                    y=y_fill,
                    fill='toself',
                    fillcolor=fill_color,
                    line=dict(color='rgba(0, 0, 0, 0)'),  # Transparent line
                    hoverinfo='skip',
                    showlegend=False,
                    name=f"{name} Segment"
                ),
                row=row,
                col=col
            )
        
        # ADD THE UPPER AND LOWER BOUND LINES
        # Upper bound line
        self.fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df[upper_col], 
                name=f"{name} Upper",
                line=dict(color=edge_color, width=width, dash=dash)
            ),
            row=row,
            col=col
        )
        
        # Lower bound line
        self.fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df[lower_col], 
                name=f"{name} Lower",
                line=dict(color=edge_color, width=width, dash=dash)
            ),
            row=row,
            col=col
        )

    def add_ta_plots(self, data: pd.DataFrame, plot_args: PlotArgs):
        plot_args = plot_args if isinstance(plot_args, list) else [plot_args]
        for pa in plot_args:
            # Check if the plot argument is a valid PlotArgs instance
            if not isinstance(pa, PlotArgs):
                raise ValueError(f"Invalid plot argument: {pa}")

            if pa.plotType == '':
                continue
            
            # print(f"add_ta_plots :: {pa.plotType} : {pa}")
            self.plot_function_map[pa.plotType](data, pa)
        return self.fig
       
        

