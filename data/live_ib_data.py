from IPython.display import display, clear_output
import pandas as pd
from ib_insync import Stock, IB, util


class LiveData:

    def __init__(self, ib:IB):
        self.ib = ib
        self.tickers = []
        self.contracts = {}
        self.bar_log = {}
        self.tick_data = {}
        self.show = False
        self.live_bars_active = False
        self.live_tick_active = False
        self.last_bar_timestamp = None
    

    def onLiveBarUpdate(self, bars, hasNewBar):
        df = util.df(bars)
        self.bar_log[bars.contract.symbol] = df

        if self.show:
            clear_output(wait=False)
            # if hasNewBar: display(bars.contract) # eg Stock(symbol='TSLA', exchange='SMART', currency='USD')
            display(bars.contract.symbol)
            display(df.tail())

    def setup_tickers(self, tickers:list):
        if  isinstance(tickers, str): tickers = [tickers] # allow for tickers to be a single string
        self.tickers = tickers
        self.setup_contracts()
        return tickers

    def setup_contracts(self):
        if self.tickers == []: raise ValueError('SBIB :: Error : No ticker symbols are set up. Use setup_tickers to give list of symbols')
        self.contracts = { tk : None for tk in self.tickers }
        for k, v in self.contracts.items():
            contract = Stock(k, 'SMART', 'USD') 
            self.ib.qualifyContracts(contract) # NOTE is blocking - https://ib-insync.readthedocs.io/api.html#ib_insync.ib.IB.qualifyContracts
            self.contracts[k] = contract
    
    def reqLiveBars(self, show:bool=False):
        """Activates live realtime bars for the tickers. 
        If no tickers given then uses self.tickers set in initial

        Args:
            tickers (list, optional): list of tickers eg ['TSLA', 'AMD']. Defaults to [].
            show (bool, optional): if true will display the dfs as they update. Defaults to False.
        """
        # if self.tickers == []: raise ValueError('SBIB :: Error : No ticker symbols are set up. Use setup_tickers to give list of symbols')
        self.show = show 
        self.bar_log = { tk : None for tk in self.tickers }
        
        for k, contract in self.contracts.items():
            bars = self.ib.reqRealTimeBars(contract, 5, 'TRADES', False) # 'MIDPOINT' 
            bars.updateEvent += self.onLiveBarUpdate
        self.live_bars_active = True
    
    def reqLiveTicks(self, show:bool=False):
        if self.tickers == []: raise ValueError('SBIB :: Error : No ticker symbols are set up. Use setup_tickers to give list of symbols')
        self.show = show 
        
        self.tick_data = { tk : None for tk in self.tickers }
        for k, contract in self.contracts.items():
            self.tick_data[k] = self.ib.reqMktData(contract, '', False, False)
        self.live_tick_active = True

    def get_live_price(self, symbol:str):
        if not self.live_tick_active: raise RuntimeError(f'SBIB :: Error : Streaming Tick Data not Active. Try self.reqLiveTicks() ')
        return self.tick_data[symbol].marketPrice()

    def get_tick_data(self, symbol:str, data:str=''):
        tick = self.tick_data[symbol]
        if data != '':
            values = {
                "bid         : {tick.bid} ",        
                "bidSize     : {tick.bidSize} ",    
                "ask         : {tick.ask} ",         
                "askSize     : {tick.askSize} ",     
                "last        : {tick.last} ",     
                "lastSize    : {tick.lastSize} ",    
                "prevBid     : {tick.prevBid} ",     
                "prevBidSize : {tick.prevBidSize} ",
                "volume      : {tick.volume} ",     
                "close       : {tick.close} ",    
                "halted      : {tick.halted} "}
            return values[data]
        return tick
    
    def format_bars_df(self, df, tz:str=''):
        """ Checks 'open_' to 'open and 'time to 'date' and resets index and only returns OHLCV.
        tz : timezone to convert the index to. eg 'US/Eastern'. If empty, makes index timezone unaware while preserving original values. """
        
        if df is not None:
            amend_cols = {}

            if 'open_' in df.columns: amend_cols['open_'] = 'open'
            if 'time' in df.columns: amend_cols['time'] = 'date'
            df = df.rename(columns=amend_cols).set_index('date')

            # CHECK IF INDEX IS JUST A DATE (IF SO THEN DON'T TRY TO ADD TZ) OR IS A DATETIME, IN WHICH CASE ADD TZ 
            if isinstance(df.index, pd.DatetimeIndex):
                if tz != '':
                    if df.index.tz is not None:
                        df = df.tz_convert(tz)
                    else:
                        df = df.tz_localize(tz)
                elif df.index.tz is not None:
                    # Just strip off timezone info without changing the timestamp values
                    df.index = pd.DatetimeIndex([t.replace(tzinfo=None) for t in df.index])
            
            return df[['open', 'high', 'low', 'close', 'volume']].copy()
        # raise RuntimeError("SBIB :: format_bars_df Error : 'time' or 'date' not in columns. Cannot convert format ")

    def update_bar(self, df, symbol:str):
        if self.last_bar_timestamp != self.bar_log[symbol].index[-1]: # IF NEW BAR THEN DONT ADD PRICE, JUST GIVE NEW BAR
            self.last_bar_timestamp = self.bar_log[symbol].index[-1]
            return df
        price  = self.get_live_price(symbol)
        df.high.iat[-1]  = max(df.high.iat[-1], price)
        df.low.iat[-1]   = min(df.low.iat[-1], price)
        df.close.iat[-1] = price
        return df

    def get_live_bar_df(self, ticker:str, format:bool=False, bsize:str='', tickUpdate:bool=True, tz:str=''):
        """ origonal columns are : time	endTime	open_	high	low	close	volume	wap	count.
        resets index from number to datetime using the time column. 
        only uses columns: 'open_', 'high', 'low', 'close', 'volume'.
        tz : timezone to convert the index to. eg 'US/Eastern' """

        if ticker not in self.bar_log: raise ValueError(f"LiveData Error :: {ticker} not Found")
        if not self.live_bars_active: raise RuntimeError(f'LiveData :: Error : Streaming Bars Data not Active. Try self.reqLiveBars() ')
        
        df  = self.bar_log[ticker]
        if df is None: raise ValueError(f"SBIB.get_live_bar_df :: Error :: no live bar data available") 
        
        if format: 
            df = self.format_bars_df( df, tz=tz)
            if bsize != '': 

                bsize = bsize.replace(' ', '').replace('s', '') # eg '2 mins' becomes '2min'

                # IF RESAMPLE, THEN ONLY COLUMNS IN AGG WILL BE RETURNED 
                df = df.resample(bsize).agg({
                    'open' : 'first',
                    'high'  : 'max',
                    'low'   : 'min',
                    'close' : 'last',
                    'volume': 'mean'})

        if tickUpdate:
            df = self.update_bar(df, ticker)
        return df
    

"""Example Usage:

from ib_insync import util, IB
from data.live_ib_data import LiveData
util.startLoop()

ib = IB()
ib.connect('127.0.0.1', 7496, clientId=16)

ld = LiveData(ib)

ld.setup_tickers(['TSLA', 'AAPL', 'AMD'])
ld.setup_contracts()
ld.reqLiveBars(show=False)


IF show=True then will display the bars as they update. you will be able to run other cells in the notebook.
but they will get overwritten with the live update so this is only useful for debugging.

To get the live bar data for a ticker use the get_live_bar_df method.

The format parameter will convert the columns to OHLCV and the time to a datetime index.
fromat=False:  time	endTime	open_	high	low	close	volume	wap	count 
format=True:   open	high low close volume

tz='US/Eastern' : will convert the index to this timezone.

bsize='' : will resample the data.

tickUpdate=True : will update the last bar with the current price.

# ! Waring : if you run the ceel in quick succession you will get a error. just wait a second and rerun
#! RuntimeError: SBIB :: Error : Streaming Tick Data not Active. Try self.reqLiveTicks() 


df = ld.get_live_bar_df(ticker='AMD', format=True, bsize='5 mins', tickUpdate=True) #.iloc[-10:]
df

        """