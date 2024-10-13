import pandas as pd
from typing import Optional
from datetime import datetime

class Backtester:
    def __init__(self, data_manager, trader):
        self.data_manager = data_manager
        self.trader = trader
        self.backtest_results = []

    def run(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None):
        data = self.data_manager.get_data()
        
        start_idx = data.index[data.index >= start_date][0] if start_date else data.index[0]
        end_idx = data.index[data.index <= end_date][-1] if end_date else data.index[-1]
        
        backtest_data = data.loc[start_idx:end_idx]
        
        for timestamp, row in backtest_data.iterrows():
            # Update DataManager with new data
            self.data_manager.update_data(row)
            
            # Query Trader for actions and results
            trader_report = self.trader.process_data(self.data_manager)
            
            # Log the report
            self._log_report(timestamp, trader_report)

        self._generate_results()

    def _log_report(self, timestamp, report):
        report['timestamp'] = timestamp
        self.backtest_results.append(report)

    def _generate_results(self):
        results_df = pd.DataFrame(self.backtest_results)
        results_df.set_index('timestamp', inplace=True)
        
        # Add backtest data to DataManager
        self.data_manager.add_backtest_data('results', results_df)

        return results_df

    def get_results(self):
        return self.data_manager.get_backtest_data('results')