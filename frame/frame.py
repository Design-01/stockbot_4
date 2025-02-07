from typing import Any, Dict, List, Tuple, Union, Optional
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
from chart.chart import Chart  # Use relative import for Chart
import strategies.ta as ta
from strategies.ta import TA
from strategies.signals import Signals

@dataclass
class Frame:
    symbol: str
    data: pd.DataFrame = pd.DataFrame()
    trading_hours: List[Tuple[str, str]] = field(default_factory=lambda: [("09:30", "16:00")])
    run_ta_on_load: bool = True
    rowHeights: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.2, 0.6])
    name: str = None

    def __post_init__(self):
        self.traders = []
        self.ta = []
        self.chart = None
        # Backtesting attributes
        self.backtest_data = pd.DataFrame()
        self.snapshots = []  # list of dfs for each snapshot
        self._backtest_start_idx = None
        self._backtest_end_idx = None
        self._current_backtest_idx = None
        self._save_snapshots = False

        self.load_ohlcv(self.data)

    def load_ohlcv(self, ohlcv: pd.DataFrame):
        if self.data.empty:
            self.data = ohlcv
        else:
            combined_data = pd.concat([self.data, ohlcv])
            # Drop duplicate indexes, keeping the last occurrence
            self.data = combined_data[~combined_data.index.duplicated(keep='last')].sort_index()

    def setup_chart(self):
        title = f"{self.symbol} ({self.name})" if self.name else self.symbol  
        self.chart = Chart(title=title, rowHeights=self.rowHeights, height=800, width=800)
        self.chart.add_candles_and_volume(self.data)
        # self.chart.add_trading_hours(self.data, self.trading_hours)

    def update_data(self, new_data):
        """
        Update the main DataFrame with new data by merging based on the datetime index.
        Ensures OHLCV columns are always first in the returned DataFrame.
        Removed! : Only overwrites old data if new data is not NaN.
        
        Parameters:
        df (pd.DataFrame): The main DataFrame to update
        new_data (pd.Series or pd.DataFrame): The new data to merge and update
        
        Returns:
        pd.DataFrame: The updated DataFrame with OHLCV columns first
        """
        # Ensure new_data is a DataFrame
        if isinstance(new_data, pd.Series):
            new_data = new_data.to_frame()
        
        # Ensure both DataFrames have a datetime index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Main DataFrame must have a datetime index")
        if not isinstance(new_data.index, pd.DatetimeIndex):
            raise ValueError("New data must have a datetime index")
        
        # Create a copy of the main DataFrame to avoid modifying the original
        updated_df = self.data.copy()
        
        # Update existing columns and add new columns
        for column in new_data.columns:
            if column in updated_df.columns:
                # Update only where new data is not NaN
                # mask = new_data.index.isin(updated_df.index) & ~new_data[column].isna() # ! This line is wrong
                mask = new_data.index.isin(updated_df.index)
                updated_df.loc[new_data.index[mask], column] = new_data.loc[mask, column]
            else:
                # Add new column
                updated_df[column] = new_data[column]
        
        # Define the order of OHLCV columns
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Separate OHLCV columns and other columns
        existing_ohlcv = [col for col in ohlcv_columns if col in updated_df.columns]
        other_columns = [col for col in updated_df.columns if col not in ohlcv_columns]
        
        # Reorder columns: OHLCV first, then other columns
        reordered_columns = existing_ohlcv + other_columns
        self.data = updated_df[reordered_columns]
        return self.data

    def add_ta(self, ta: TA, style: Dict[str, Any] | List[Dict[str, Any]] = {}, chart_type: str = "line", row: int = 1, nameCol:str=None):
        # Check for duplicates
        for existing_ta, existing_style, existing_chart_type, existing_row, existing_nameCol in self.ta:
            if (existing_ta == ta and 
                existing_style == style and 
                existing_chart_type == chart_type and 
                existing_row == row):
                # Duplicate found, do not add
                return ta
        
        # No duplicates found, add the new TA
        if self.run_ta_on_load:
            self.update_data(ta.run(self.data))
        
        self.ta.append((ta, style, chart_type, row, nameCol))
        # returning the ta object to allow it to be assigend and therefor access the name vaialbe in the ta object
        # eg 
        # ta1 = frame.add_ta(ta('close'), style, chart_type, row, nameCol)
        # ta2 = frame.add_ta(ta(ta1.name), style, chart_type, row, nameCol)
        return ta

    def add_ta_batch(self, taList:list[ta.TAData], forceRun:bool=False):
        for ta in taList:
            self.add_ta(ta.ta, ta.style, ta.chart_type, ta.row, ta.nameCol)

    def update_ta_data(self):
        """Updates the data for all the technical indicators in the frame"""
        for ta, style, chart_type, row in self.ta:
            self.data = self.update_data(ta.run(self.data))
    
    def import_data(self, import_df, importCols: list[str] = None, colsContain: list[str] = None, ffillAutoLimit=True, ffillManualLimit=None, prefix='_', merge_to_backtest: bool = False):
        """
        Import and merge high timeframe data into the existing low timeframe data.

        Parameters:
        import_df (pd.DataFrame): The high timeframe DataFrame to import.
        importCols (Union[str, List[str]]): The column(s) to import from the high timeframe DataFrame.
                                            If a single string is provided, it will be converted to a list.
        colsContain (Union[str, List[str]]): The text(s) to search for within the column names of the high timeframe DataFrame.
                                                If a single string is provided, it will be converted to a list.
        ffillAutoLimit (bool): If True, automatically determine the forward fill limit based on the time delta between rows.
                        If False, use the ffillManualLimit value.
        ffillManualLimit (Optional[int]): The maximum number of rows to forward fill if ffillAutoLimit is False.
                                    If None, a default value of 4 will be used.
        prefix (str): The prefix to add to the column names from the high timeframe DataFrame when merging.
        merge_to_backtest (bool): If True, merge the data into the backtest_data attribute.
                                If False, merge the data into the data attribute.

        Returns:
        None: The method updates the data or backtest_data attribute of the Frame instance in place.

        Notes:
        - The method first filters the columns of import_df based on the importCols and colsContain parameters.
        - It then renames the filtered columns with the specified prefix.
        - The method merges the filtered and renamed columns into the low timeframe DataFrame (self_df).
        - The merged columns are forward filled based on the determined or specified fill limit.
        - The updated DataFrame is assigned back to either the data or backtest_data attribute of the Frame instance.
        """
        if isinstance(importCols, str):
            importCols = [importCols]
        if isinstance(colsContain, str):
            colsContain = [colsContain]

        self_df = self.backtest_data if merge_to_backtest else self.data

        filtered_columns = []
        if importCols:
            filtered_columns.extend([col for col in import_df.columns if col in importCols])
        if colsContain:
            filtered_columns.extend([col for col in import_df.columns if any(target in col for target in colsContain)])

        filtered_columns = list(set(filtered_columns))  # Remove duplicates

        column_mapping = {col: f"{prefix}{col}" for col in filtered_columns}
        self_df = self_df.drop(columns=list(column_mapping.values()), errors='ignore')

        if ffillAutoLimit:
            low_tf_delta = pd.Series(self_df.index).diff().mode()[0]
            high_tf_delta = pd.Series(import_df.index).diff().mode()[0]
            fill_limit = max(1, int((high_tf_delta / low_tf_delta) - 1))
        else:
            fill_limit = max(1, ffillManualLimit if ffillManualLimit is not None else 4)

        # Pre-rename columns before merge
        import_df_subset = import_df[filtered_columns].rename(columns=column_mapping)
        df_merged = pd.merge_asof(self_df, import_df_subset, left_index=True, right_index=True, direction='backward')

        # Forward fill the new columns
        for col in column_mapping.values():
            df_merged[col] = df_merged[col].fillna(method='ffill', limit=fill_limit)

        if merge_to_backtest:
            self.backtest_data = df_merged
        else:
            self.data = df_merged
        


        # Example usage:
        # Auto detect limit
        # result = merge_timeframes(df_1min, df_5min, ['close', 'volume'])

        # Manual limit
        # result = merge_timeframes(df_1min, df_5min, 'close', auto_limit=False, ffillManualLimit=4)

    def plot(self, width: int = 1400, height: int = 800, trading_hours: bool = False, 
        show: bool = True, snapshot_data: pd.DataFrame = None, use_backtest_data: bool = False,
        animate: bool = False):

        """Plot the frame data with technical indicators."""

        if animate and self.snapshots:
            if self.chart is None:
                self.setup_chart()
            self.chart.enable_animation()
            self.chart.create_frames_from_snapshots(self.snapshots)

        # Determine which data to use for plotting
        original_data = None
        if snapshot_data is not None:
            original_data = self.data
            self.data = snapshot_data
        elif use_backtest_data and hasattr(self, 'backtest_data') and not self.backtest_data.empty:
            original_data = self.data
            self.data = self.backtest_data
            
        # Ensure chart is initialized
        if self.chart is None:
            self.setup_chart()
            
        # Use existing plot logic
        self.chart.refesh(self.data)
        
        for indicator, style, chart_type, row, nameCol in self.ta:
            if style == {}: 
                continue
                
            # Handle names whether they're in a list or string
            names = indicator.names
            if isinstance(names, str):
                names = [names]
            elif not isinstance(names, list):
                names = list(names)
                
            available_columns = [name for name in names if name in self.data.columns]
            if available_columns:
                indicator_data = self.data[available_columns]
                nameData = self.data[nameCol] if nameCol in self.data.columns else None
                self.chart.add_ta(indicator_data, style, chart_type, row, nameData)
        
        if trading_hours:
            self.chart.add_trading_hours(self.data, self.trading_hours)
        
        if show:
            self.chart.show(width=width, height=height)

        # Restore original data if we temporarily changed it
        if original_data is not None:
            self.data = original_data

       # ---------------- Backtesting Methods ----------------

    def backtest_setup(self, start: str | int, end: str | int, save_snapshots: bool = False):
        """Create a slice of the data for backtesting."""
        if self.data.empty:
            raise ValueError("No data loaded for backtesting")

        # Convert datetime strings to index positions if necessary
        if isinstance(start, str):
            start = self.data.index.get_indexer([pd.to_datetime(start)])[0]
        if isinstance(end, str):
            end = self.data.index.get_indexer([pd.to_datetime(end)])[0]

        # Handle negative indices
        if start < 0:
            start = len(self.data) + start
        if end < 0:
            end = len(self.data) + end

        # Validate indices
        if not (0 <= start < len(self.data) and 0 <= end < len(self.data)):
            raise ValueError("Start and end indices must be within data range")
        if start >= end:
            raise ValueError("Start index must be less than end index")

        # Store control parameters
        self._backtest_start_idx = start
        self._backtest_end_idx = end
        self._current_backtest_idx = start
        self._save_snapshots = save_snapshots

        # Initialize backtest data with historical data up to start index
        self.backtest_data = self.data.iloc[:start].copy()
        self.snapshots = []  # Clear previous snapshots
        
        # Run initial TA on the starting slice
        temp_data = self.data  # Store original data
        self.data = self.backtest_data  # Set data to backtest slice
        self.update_ta_data()  # Run TA on initial slice
        self.backtest_data = self.data.copy()  # Store result
        self.data = temp_data  # Restore original data

        # Take initial snapshot if enabled
        if save_snapshots:
            self.snapshots.append({
                'index': self._current_backtest_idx,
                'date': self.data.index[self._current_backtest_idx],
                'data': self.backtest_data.copy()
            })

    def backtest_next_row(self, importData:pd.DataFrame=None, prefix:str='') -> bool:
        """Move to the next row in the backtest data."""
        if self._current_backtest_idx is None:
            raise ValueError("Backtest not initialized. Call backtest_setup first.")
        
        if self._current_backtest_idx >= self._backtest_end_idx:
            return False

        # Add next row to backtest data
        next_row = self.data.iloc[self._current_backtest_idx:self._current_backtest_idx + 1]
        self.backtest_data = pd.concat([self.backtest_data, next_row])

        if importData is not None:
            self.import_data(importData, importData.columns, merge_to_backtest=True, prefix=prefix)
        
        # Run TA on the updated backtest data
        temp_data = self.data  # Store original data
        self.data = self.backtest_data  # Set data to backtest data
        self.update_ta_data()  # Run TA
        self.backtest_data = self.data.copy()  # Store result
        self.data = temp_data  # Restore original data
        
        # Save snapshot if enabled
        if self._save_snapshots:
            self.snapshots.append({
                'index': self._current_backtest_idx,
                'date': self.data.index[self._current_backtest_idx],
                'data': self.backtest_data.copy()
            })
        
        self._current_backtest_idx += 1
        return True

    def backtest_run(self, callback=None, update_main_data: bool = False, clear_snapshots: bool = True):
        """
        Run the backtest for all rows in the specified range.
        
        Args:
            callback: Optional function to call after each step with signature:
                    callback(current_idx, current_date, backtest_data)
            update_main_data: Whether to update self.data with final backtest state
        """
        try:
            from IPython.display import clear_output, display
            jupyter_available = True
        except ImportError:
            jupyter_available = False
            print("Warning: IPython not available. Running without progress display.")

        if self._current_backtest_idx is None:
            raise ValueError("Backtest not initialized. Call backtest_setup first.")

        total_rows = self._backtest_end_idx - self._backtest_start_idx
        processed_rows = 0

        while self.backtest_next_row():
            processed_rows += 1
            current_date = self.data.index[self._current_backtest_idx - 1]
            
            if jupyter_available:
                if clear_snapshots: 
                    clear_output(wait=True)
                print(f"Tested rows {processed_rows}/{total_rows} - Date: {current_date} (location {self._current_backtest_idx - 1})")
            
            if callback:
                callback(
                    self._current_backtest_idx - 1,
                    current_date,
                    self.backtest_data
                )

    def get_snapshot(self, identifier: Union[str, int, datetime], plot: bool = False,
                        width: int = 1400, height: int = 800, trading_hours: bool = False) -> Optional[pd.DataFrame]:
        """
        Retrieve and optionally plot a specific snapshot.
        
        Args:
            identifier: Can be:
                    - datetime string or datetime object for date lookup
                    - integer for position in snapshots list
                    - negative integer for position from end
            plot: Whether to plot the snapshot
            width: Width of the chart if plotting
            height: Height of the chart if plotting
            trading_hours: Whether to show trading hours if plotting
        
        Returns:
            Optional[pd.DataFrame]: The snapshot data if found, None otherwise
        """
        try:
            from IPython.display import clear_output
            jupyter_available = True
        except ImportError:
            jupyter_available = False

        snapshot_data = None

        if isinstance(identifier, (str, datetime)):
            # Convert string to datetime if necessary
            if isinstance(identifier, str):
                identifier = pd.to_datetime(identifier)
            
            # Find closest snapshot by date
            for i, snapshot in enumerate(self.snapshots):
                if snapshot['date'] == identifier:
                    snapshot_data = snapshot['data']
                    if jupyter_available:
                        clear_output(wait=True)
                    print(f"Viewing snapshot {i + 1}/{len(self.snapshots)} - Date: {snapshot['date']} (location {snapshot['index']})")
                    break
        
        elif isinstance(identifier, int):
            # Handle negative indices
            if identifier < 0:
                identifier = len(self.snapshots) + identifier
            
            # Return snapshot at position if valid
            if 0 <= identifier < len(self.snapshots):
                snapshot_data = self.snapshots[identifier]['data']
                if jupyter_available:
                    clear_output(wait=True)
                print(f"Viewing snapshot {identifier + 1}/{len(self.snapshots)} - Date: {self.snapshots[identifier]['date']} (location {self.snapshots[identifier]['index']})")

        # Plot the snapshot if requested and data was found
        if plot and snapshot_data is not None:
            self.plot(
                width=width,
                height=height,
                trading_hours=trading_hours,
                show=True,
                snapshot_data=snapshot_data
            )

        return snapshot_data
    
    def run_snapshots(self, start: int | str = 0, end: int | str = -1, plot: bool = True,
                    width: int = 1400, height: int = 800, trading_hours: bool = False,
                    display_df: bool = False, sleep_time: float = 0.5) -> List[pd.DataFrame]:
        """
        Iterate through snapshots between start and end indices, optionally plotting each one.
        
        Args:
            start: Starting snapshot index or datetime string
            end: Ending snapshot index or datetime string (inclusive)
            plot: Whether to plot each snapshot
            width: Width of plot if plotting
            height: Height of plot if plotting
            trading_hours: Whether to show trading hours in plot
            display_df: Whether to display the DataFrame alongside the plot
            sleep_time: Time to pause between snapshots in seconds
            
        Returns:
            List of DataFrames for each snapshot viewed
        """
        try:
            from IPython.display import clear_output, display
            import time
            jupyter_available = True
        except ImportError:
            jupyter_available = False
            print("Warning: IPython not available. Running without clear_output functionality.")
        
        if not self.snapshots:
            raise ValueError("No snapshots available. Run backtest with save_snapshots=True first.")
        
        # Convert datetime strings to indices if necessary
        if isinstance(start, str):
            start_date = pd.to_datetime(start)
            start = next((i for i, snap in enumerate(self.snapshots) 
                        if pd.to_datetime(snap['date']) >= start_date), 0)
        
        if isinstance(end, str):
            end_date = pd.to_datetime(end)
            end = len(self.snapshots) - 1 - next((i for i, snap in enumerate(reversed(self.snapshots))
                                                if pd.to_datetime(snap['date']) <= end_date), 0)
        
        # Handle negative indices
        if start < 0:
            start = len(self.snapshots) + start
        if end < 0:
            end = len(self.snapshots) + end
        
        # Validate indices
        start = max(0, min(start, len(self.snapshots) - 1))
        end = max(0, min(end, len(self.snapshots) - 1))
        
        if start > end:
            raise ValueError("Start index must be less than or equal to end index")
        
        viewed_snapshots = []
        
        for i in range(start, end + 1):
            if jupyter_available:
                clear_output(wait=True)
            
            snapshot_data = self.get_snapshot(i, plot=plot, width=width, 
                                            height=height, trading_hours=trading_hours)
            
            if snapshot_data is not None:
                viewed_snapshots.append(snapshot_data)
                
                if display_df and jupyter_available:
                    print(f"\nSnapshot {i} - Date: {self.snapshots[i]['date']}")
                    display(snapshot_data)
            
            if sleep_time > 0 and i < end:
                time.sleep(sleep_time)
        
        # return viewed_snapshots



    def get_current_backtest_state(self) -> dict:
        """
        Get the current state of the backtest.
        
        Returns:
            dict: Current backtest state information
        """
        return {
            'start_index': self._backtest_start_idx,
            'end_index': self._backtest_end_idx,
            'current_index': self._current_backtest_idx,
            'save_snapshots': self._save_snapshots,
            'num_snapshots': len(self.snapshots),
            'is_running': self._current_backtest_idx is not None and 
                         self._current_backtest_idx < self._backtest_end_idx
        }
