from ib_insync import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class IBMarketAnalyzer:
    def __init__(self, ib: IB):
        self.ib = ib
        self.market_signals = {}
        self.sector_signals = {}
        self.spy_contract = None
        self.sector_etfs = {}
        
    def set_sectors(self):

        # Create SPY contract
        self.spy_contract = Stock('SPY', 'SMART', 'USD')
        
        # Define major sector ETF contracts
        self.sector_etfs = {
            'Technology': Stock('XLK', 'SMART', 'USD'),
            'Healthcare': Stock('XLV', 'SMART', 'USD'),
            'Financials': Stock('XLF', 'SMART', 'USD'),
            'Energy': Stock('XLE', 'SMART', 'USD'),
            'Consumer_Discretionary': Stock('XLY', 'SMART', 'USD'),
            'Consumer_Staples': Stock('XLP', 'SMART', 'USD'),
            'Industrials': Stock('XLI', 'SMART', 'USD'),
            'Materials': Stock('XLB', 'SMART', 'USD'),
            'Utilities': Stock('XLU', 'SMART', 'USD'),
            'Real_Estate': Stock('XLRE', 'SMART', 'USD')
        }

            
    def get_historical_data(self, contract, duration='1 Y', bar_size='1 day'):
            """
            Fetch historical data from IB
            """
            # Qualify the contracts first
            self.ib.qualifyContracts(contract)
            
            # Format end time in UTC
            end_time = datetime.now().strftime('%Y%m%d-%H:%M:%S')
            
            try:
                bars = self.ib.reqHistoricalData(
                    contract,
                    endDateTime=end_time,
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow='TRADES',
                    useRTH=True,
                    formatDate=1,
                    timeout=10
                )
                
                if bars:
                    df = util.df(bars)
                    df.set_index('date', inplace=True)
                    return df
                return None
                
            except Exception as e:
                print(f"Error fetching data for {contract.symbol}: {str(e)}")
                return None

    def analyze_market_direction(self, lookback_periods={'short': 20, 'medium': 50, 'long': 200}):
        """
        Analyze overall market direction using SPY data from IB
        """
        if not self.spy_contract:
            raise ValueError("IB connection not initialized. Call connect_ib() first.")
            
        df = self.get_historical_data(self.spy_contract)
        if df is None:
            raise ValueError("Failed to fetch SPY data from IB")
            
        # Calculate moving averages
        for period_name, period in lookback_periods.items():
            df[f'MA_{period}'] = df['close'].rolling(window=period).mean()
        
        # Get current price and moving averages
        current_price = df['close'].iloc[-1]
        mas = {period_name: df[f'MA_{period}'].iloc[-1] 
               for period_name, period in lookback_periods.items()}
        
        # Calculate momentum (Rate of Change)
        df['ROC_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate VWAP
        df['VWAP'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
        
        self.market_signals = {
            'trend': {
                'short_term': 'Bullish' if current_price > mas['short'] else 'Bearish',
                'medium_term': 'Bullish' if current_price > mas['medium'] else 'Bearish',
                'long_term': 'Bullish' if current_price > mas['long'] else 'Bearish'
            },
            'momentum': {
                'roc': df['ROC_10'].iloc[-1],
                'rsi': df['RSI'].iloc[-1]
            },
            'price_levels': {
                'current_price': current_price,
                'moving_averages': mas,
                'vwap': df['VWAP'].iloc[-1]
            },
            'volume': {
                'current': df['volume'].iloc[-1],
                'avg_20d': df['volume'].rolling(20).mean().iloc[-1]
            }
        }
        
        return self.market_signals
    
    def analyze_sector_strength(self, lookback_period=20):
        """
        Analyze relative strength of different market sectors using IB data
        """
        sector_performance = {}
        spy_data = self.get_historical_data(self.spy_contract)
        spy_return = ((spy_data['close'].iloc[-1] - spy_data['close'].iloc[-lookback_period]) / 
                     spy_data['close'].iloc[-lookback_period]) * 100
        
        for sector_name, contract in self.sector_etfs.items():
            df = self.get_historical_data(contract)
            if df is None:
                continue
                
            # Calculate period returns
            current_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-lookback_period]
            period_return = ((current_price - prev_price) / prev_price) * 100
            
            # Calculate relative strength vs SPY
            relative_strength = period_return - spy_return
            
            # Calculate momentum
            df['ROC_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
            
            # Calculate volume trend
            recent_volume = df['volume'].tail(lookback_period).mean()
            previous_volume = df['volume'].tail(lookback_period * 2).head(lookback_period).mean()
            volume_trend = 'Increasing' if recent_volume > previous_volume else 'Decreasing'
            
            sector_performance[sector_name] = {
                'period_return': period_return,
                'relative_strength': relative_strength,
                'momentum': df['ROC_10'].iloc[-1],
                'volume_trend': volume_trend
            }
        
        # Rank sectors by relative strength
        ranked_sectors = dict(sorted(sector_performance.items(), 
                                   key=lambda x: x[1]['relative_strength'], 
                                   reverse=True))
        
        self.sector_signals = {
            'ranked_performance': ranked_sectors,
            'top_sectors': list(ranked_sectors.keys())[:3],
            'weakest_sectors': list(ranked_sectors.keys())[-3:],
            'market_return': spy_return
        }
        
        return self.sector_signals
    
    def get_trading_signals(self):
        """
        Combine market and sector analysis to generate trading signals
        """
        if not self.market_signals or not self.sector_signals:
            return "Please run market and sector analysis first"
        
        # Count bullish vs bearish signals
        trend_signals = self.market_signals['trend'].values()
        bullish_count = sum(1 for signal in trend_signals if signal == 'Bullish')
        bearish_count = sum(1 for signal in trend_signals if signal == 'Bearish')
        
        # Determine overall market condition
        if bullish_count >= 2:
            market_condition = 'Bullish'
        elif bearish_count >= 2:
            market_condition = 'Bearish'
        else:
            market_condition = 'Mixed'
            
        # Get strongest sectors with positive relative strength
        strong_sectors = [sector for sector in self.sector_signals['top_sectors'] 
                         if self.sector_signals['ranked_performance'][sector]['relative_strength'] > 0]
        
        return {
            'overall_market': market_condition,
            'market_momentum': {
                'direction': 'Positive' if self.market_signals['momentum']['roc'] > 0 else 'Negative',
                'rsi_level': self.market_signals['momentum']['rsi']
            },
            'strong_sectors': strong_sectors,
            'weak_sectors': self.sector_signals['weakest_sectors'],
            'recommendation': self._generate_recommendation(market_condition, strong_sectors)
        }
    
    def _generate_recommendation(self, market_condition, strong_sectors):
        """
        Generate specific trading recommendations based on market conditions
        """
        if market_condition == 'Bullish':
            if strong_sectors:
                return f"Market showing strength. Focus on {', '.join(strong_sectors[:2])} sectors. " \
                       f"RSI at {self.market_signals['momentum']['rsi']:.1f}"
            return "Market bullish but sectors showing weakness. Proceed with caution."
        elif market_condition == 'Bearish':
            return f"Defensive positioning recommended. Consider reducing exposure. " \
                   f"Weakest sectors: {', '.join(self.sector_signals['weakest_sectors'][:2])}"
        else:
            return "Mixed signals - maintain balanced exposure and wait for clearer direction"
    
    def disconnect(self):
        """
        Disconnect from IB
        """
        self.ib.disconnect()


    def create_analysis_report(self):
        """
        Creates a comprehensive DataFrame containing ALL metrics for market and sectors
        """
        if not self.market_signals or not self.sector_signals:
            print("Running market and sector analysis first...")
            self.analyze_market_direction()
            self.analyze_sector_strength()

        # Create detailed market overview data with descriptions
        market_data = {
            'Metric': [
                'SPY Current Price',
                'SPY 20-Day MA',
                'SPY 50-Day MA',
                'SPY 200-Day MA',
                'SPY VWAP',
                'Price vs 20MA (%)',
                'Price vs 50MA (%)',
                'Price vs 200MA (%)',
                'Price vs VWAP (%)',
                'SPY RSI',
                'SPY ROC (10-day)',
                'RSI Trend',
                'ROC Trend',
                'Current Volume',
                '20-Day Avg Volume',
                'Volume Ratio (Current/Avg)',
                'Volume Trend',
                'Short-term Trend',
                'Medium-term Trend',
                'Long-term Trend',
                'Trend Strength',
                'Market Condition',
                'Market Return (%)'
            ],
            'Description': [
                'Current market price of SPY ETF',
                '20-day simple moving average - short-term trend indicator',
                '50-day simple moving average - medium-term trend indicator',
                '200-day simple moving average - long-term trend indicator',
                'Volume Weighted Average Price - shows average price based on volume',
                'Percentage difference between current price and 20-day MA',
                'Percentage difference between current price and 50-day MA',
                'Percentage difference between current price and 200-day MA',
                'Percentage difference between current price and VWAP',
                'Relative Strength Index (14-day) - momentum indicator (0-100)',
                'Rate of Change over 10 days - momentum measurement',
                'RSI interpretation (Overbought >70, Oversold <30)',
                'Direction of price change over 10-day period',
                'Today\'s trading volume',
                'Average daily volume over the last 20 trading days',
                'Ratio of current volume to 20-day average volume',
                'Volume comparison to 20-day average',
                'Trend based on 20-day MA comparison',
                'Trend based on 50-day MA comparison',
                'Trend based on 200-day MA comparison',
                'Number of bullish trends out of three timeframes',
                'Overall market assessment based on trends',
                'Market return over analysis period'
            ],
            'Value': [
                round(self.market_signals['price_levels']['current_price'], 2),
                round(self.market_signals['price_levels']['moving_averages']['short'], 2),
                round(self.market_signals['price_levels']['moving_averages']['medium'], 2),
                round(self.market_signals['price_levels']['moving_averages']['long'], 2),
                round(self.market_signals['price_levels']['vwap'], 2),
                round(((self.market_signals['price_levels']['current_price'] / 
                    self.market_signals['price_levels']['moving_averages']['short'] - 1) * 100), 2),
                round(((self.market_signals['price_levels']['current_price'] / 
                    self.market_signals['price_levels']['moving_averages']['medium'] - 1) * 100), 2),
                round(((self.market_signals['price_levels']['current_price'] / 
                    self.market_signals['price_levels']['moving_averages']['long'] - 1) * 100), 2),
                round(((self.market_signals['price_levels']['current_price'] / 
                    self.market_signals['price_levels']['vwap'] - 1) * 100), 2),
                round(self.market_signals['momentum']['rsi'], 2),
                round(self.market_signals['momentum']['roc'], 2),
                'Overbought' if self.market_signals['momentum']['rsi'] > 70 else 
                'Oversold' if self.market_signals['momentum']['rsi'] < 30 else 'Neutral',
                'Positive' if self.market_signals['momentum']['roc'] > 0 else 'Negative',
                int(self.market_signals['volume']['current']),
                int(self.market_signals['volume']['avg_20d']),
                round(self.market_signals['volume']['current'] / 
                    self.market_signals['volume']['avg_20d'], 2),
                'Above Average' if (self.market_signals['volume']['current'] > 
                                self.market_signals['volume']['avg_20d']) else 'Below Average',
                self.market_signals['trend']['short_term'],
                self.market_signals['trend']['medium_term'],
                self.market_signals['trend']['long_term'],
                f"{sum(1 for x in self.market_signals['trend'].values() if x == 'Bullish')}/3",
                self.get_trading_signals()['overall_market'],
                round(self.sector_signals['market_return'], 2)
            ],
            'Interpretation': [
                'Current market price point',
                'Price above 20MA is bullish short-term',
                'Price above 50MA is bullish medium-term',
                'Price above 200MA is bullish long-term',
                'Price above VWAP indicates buying pressure',
                'Positive % indicates price strength vs 20MA',
                'Positive % indicates price strength vs 50MA',
                'Positive % indicates price strength vs 200MA',
                'Positive % indicates current buying pressure',
                'RSI > 70 overbought, < 30 oversold',
                'Positive values indicate upward momentum',
                'Current RSI trend interpretation',
                'Current price momentum interpretation',
                'Today\'s trading activity level',
                'Normal trading activity baseline',
                '> 1.0 indicates above-average volume',
                'Volume trend interpretation',
                'Short-term trend direction',
                'Medium-term trend direction',
                'Long-term trend direction',
                'Number of bullish trends (more = stronger)',
                'Overall market trend assessment',
                'Market performance over period'
            ]
        }
        
        # Create DataFrame with all columns
        market_df = pd.DataFrame(market_data)
        
        # Set display options for better visualization
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        # Create detailed sector analysis DataFrame
        sector_data = []
        for sector, metrics in self.sector_signals['ranked_performance'].items():
            sector_data.append({
                'Sector': sector,
                'Return (%)': round(metrics['period_return'], 2),
                'Relative Strength': round(metrics['relative_strength'], 2),
                'Momentum (ROC)': round(metrics['momentum'], 2),
                'Volume Trend': metrics['volume_trend'],
                'Outperforming SPY': 'Yes' if metrics['relative_strength'] > 0 else 'No',
                'Momentum Direction': 'Positive' if metrics['momentum'] > 0 else 'Negative',
                'Relative Strength Rank': len(sector_data) + 1,
                'Return Rank': None,  # Will be filled later
                'Momentum Rank': None,  # Will be filled later
                'Overall Score': None,  # Will be filled later
            })
        
        sector_df = pd.DataFrame(sector_data)
        
        # Add rankings
        sector_df['Return Rank'] = sector_df['Return (%)'].rank(ascending=False)
        sector_df['Momentum Rank'] = sector_df['Momentum (ROC)'].rank(ascending=False)
        
        # Calculate overall score (lower is better)
        sector_df['Overall Score'] = (sector_df['Relative Strength Rank'] + 
                                    sector_df['Return Rank'] + 
                                    sector_df['Momentum Rank']) / 3
        
        # Sort by Overall Score
        sector_df = sector_df.sort_values('Overall Score')
        
        # Format DataFrames
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        
        print("\n=== Detailed Market Overview ===")
        print(market_df.to_string(index=False))
        
        print("\n=== Detailed Sector Analysis ===")
        print(sector_df.to_string(index=False))
        
        print("\n=== Trading Recommendation ===")
        print(self.get_trading_signals()['recommendation'])
        
        return market_df, sector_df

    def filter_sectors(self, sector_df, **kwargs):
        """
        Filter sector data based on multiple criteria
        
        Parameters:
        sector_df: DataFrame from create_analysis_report
        **kwargs: Filtering criteria, such as:
            - min_return: Minimum return percentage
            - min_rs: Minimum relative strength
            - min_momentum: Minimum momentum
            - volume_trend: 'Increasing' or 'Decreasing'
            - outperforming_spy: True/False
            - top_n: Number of top sectors to return
            - sort_by: Column to sort by
            - ascending: Sort order
        
        Returns:
        DataFrame: Filtered sector data
        """
        filtered_df = sector_df.copy()
        
        if 'min_return' in kwargs:
            filtered_df = filtered_df[filtered_df['Return (%)'] >= kwargs['min_return']]
            
        if 'min_rs' in kwargs:
            filtered_df = filtered_df[filtered_df['Relative Strength'] >= kwargs['min_rs']]
            
        if 'min_momentum' in kwargs:
            filtered_df = filtered_df[filtered_df['Momentum (ROC)'] >= kwargs['min_momentum']]
            
        if 'volume_trend' in kwargs:
            filtered_df = filtered_df[filtered_df['Volume Trend'] == kwargs['volume_trend']]
            
        if 'outperforming_spy' in kwargs:
            filtered_df = filtered_df[filtered_df['Outperforming SPY'] == 
                                    ('Yes' if kwargs['outperforming_spy'] else 'No')]
            
        if 'sort_by' in kwargs:
            filtered_df = filtered_df.sort_values(kwargs['sort_by'], 
                                                ascending=kwargs.get('ascending', False))
            
        if 'top_n' in kwargs:
            filtered_df = filtered_df.head(kwargs['top_n'])
            
        return filtered_df