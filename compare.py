import pandas as pd
import numpy as np

class SectorAnalysis:
    def __init__(self, etf_df, market_df):
        self.etf_df = etf_df
        self.market_df = market_df
        self.result_df = pd.DataFrame(index=self.etf_df.index)
    
    def _moving_average(self, df, period, column='close'):
        return df[column].rolling(window=period).mean()
    
    def compute_moving_averages(self, short=50, long=200):
        self.result_df['etf_50_ma'] = self._moving_average(self.etf_df, short)
        self.result_df['etf_200_ma'] = self._moving_average(self.etf_df, long)
        self.result_df['market_50_ma'] = self._moving_average(self.market_df, short)
        self.result_df['market_200_ma'] = self._moving_average(self.market_df, long)
    
    def _mansfield_rsi(self, security_close, index_close, period=200):
        # Calculate RP (Relative Performance)
        rp = (security_close / index_close) * 100
        
        # Calculate SMA of RP
        rp_sma = rp.rolling(window=period).mean()
        
        # Calculate MRS (Mansfield Relative Strength)
        mrs = ((rp / rp_sma) - 1) * 100
        
        return mrs
    
    def compute_mansfield_rsi(self, period=200):
        self.result_df['mansfield_rsi'] = self._mansfield_rsi(
            self.etf_df['close'], 
            self.market_df['close'], 
            period
        )
    
    def compute_ma_roc_ratio(self, ma_period=50):
        """
        Compute the ratio of MA percent changes between current and previous bar
        """
        # Calculate moving averages
        etf_ma = self._moving_average(self.etf_df, ma_period)
        market_ma = self._moving_average(self.market_df, ma_period)
        
        # Calculate percent change for current bar vs previous bar
        etf_ma_roc = etf_ma.pct_change()
        market_ma_roc = market_ma.pct_change()
        
        # Calculate ratio (add small epsilon to avoid division by zero)
        epsilon = 1e-10
        ratio = (etf_ma_roc + epsilon) / (market_ma_roc + epsilon)
        
        # Handle extreme values
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        ratio = ratio.fillna(1)  # Neutral value when undefined
        
        # Normalize to -10 to +10 scale
        centered = ratio - 1
        self.result_df['ma_roc_ratio'] = 10 * (2 / (1 + np.exp(-centered)) - 1)
    
    def compute_combined_score(self):
        """
        Compute a combined score that stays within -10 to +10 range
        """
        self.result_df['combined_score'] = (
            self.result_df['mansfield_rsi'] + 
            self.result_df['ma_roc_ratio']
        ) / 2
    
    def compute_all(self, mansfield_period=200, roc_period=50, ma_short=50, ma_long=200):
        """
        Compute all metrics with configurable periods
        
        Parameters:
        -----------
        mansfield_period : int, default 200
            Period for Mansfield RSI calculation
        roc_period : int, default 50
            Period for MA ROC ratio calculation
        ma_short : int, default 50
            Short period for moving averages
        ma_long : int, default 200 
            Long period for moving averages
        """
        self.compute_moving_averages(short=ma_short, long=ma_long)
        self.compute_mansfield_rsi(period=mansfield_period)
        self.compute_ma_roc_ratio(ma_period=roc_period)
        self.compute_combined_score()
    
    def get_today(self, verbose=False):
        latest = self.result_df.iloc[-1]
        
        today_metrics = pd.Series({
            'mansfield_rsi': latest['mansfield_rsi'],
            'ma_roc_ratio': latest['ma_roc_ratio'],
            'combined_score': latest['combined_score']
        })
        
        if verbose:
            print("\n=== Today's Market Analysis ===\n")
            print(f"Mansfield RSI: {latest['mansfield_rsi']:.2f}")
            print(f"MA ROC Ratio: {latest['ma_roc_ratio']:.2f}")
            print(f"Combined Score: {latest['combined_score']:.2f}\n")
            
            score = latest['combined_score']
            if score > 7.5:
                print("🚀 STRONGLY BULLISH")
            elif score > 2.5:
                print("📈 BULLISH")
            elif score > -2.5:
                print("↔️ NEUTRAL")
            elif score > -7.5:
                print("📉 BEARISH")
            else:
                print("🐻 STRONGLY BEARISH")
        
        return today_metrics
    
    def get_df(self):
        return self.result_df.dropna()