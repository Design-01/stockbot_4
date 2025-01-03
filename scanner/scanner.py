import pandas as pd
from ib_insync import IB, ScannerSubscription, TagValue 
import stock
from strategies import ta

class StockbotScanner:
    def __init__(self, ib):
        self.ib = ib
        self.scan_results_df = None
        self.daily_stockx = []
        self.fund_reults_df = pd.DataFrame()
        self.ta_results_df = pd.DataFrame()

    def scan(self, 
            scan_code:str ='TOP_PERC_GAIN',
            price: tuple[float, float] = (1, 100),
            volume: int = 100_000,
            change_perc: float = 4,
            market_cap: tuple[float, float] = (100, 10000), # value is in millions
            location='STK.US.MAJOR',
            ):
        
        # set tags for scanner
        tags = [
            TagValue('priceAbove', price[0]),      
            TagValue('priceBelow', price[1]),
            TagValue('volumeAbove', volume),   
            TagValue('changePercAbove', change_perc),
            TagValue('marketCapAbove1e6', market_cap[0]), # value is in millions
            TagValue('marketCapBelow1e6', market_cap[1]) # value is in millions
        ]
        
        # Request scanner data
        sub = ScannerSubscription(
            instrument='STK',
            locationCode=location,
            scanCode=scan_code
        )

        print(f'Scanning {location} for {scan_code} ..')
        print(f'price: {price}, volume: {volume}, changePerc: {change_perc}, marketCap: {market_cap}')

        data = self.ib.reqScannerData(sub, [], tags)
        print(f'{location} .. len(data): {len(data)}')
        return data
    
    def multiscan(self, 
            scan_code: str = 'TOP_PERC_GAIN',
            price: tuple[float, float] = (1, 100),
            volume: int = 100_000,
            change_perc: float = 4,
            market_cap: float = 100, # millions
            limit_each_cap: int = 100):
        
        market_caps = [(market_cap, market_cap*10), (market_cap*10, market_cap*100), (market_cap*100, market_cap*1000)]
        rows = []

        # uses market_caps to scan for each market cap range.
        # this is so more stcoks can be scanned than just 50 at a time
        for cap in market_caps:
            scanData = self.scan(scan_code, price, volume, change_perc, cap)
            cap_str = f'{cap[0]}M-{cap[1]}M'

            limit = limit_each_cap
            
            for data in scanData:
                contract = data.contractDetails.contract
                row = {
                    'Rank': data.rank,
                    'Symbol': contract.symbol,
                    'Market Cap Range': cap,
                }
                rows.append(row)

                limit -= 1
                if limit == 0:
                    break
                
            self.ib.sleep(60)
        
        if not rows:
            return None
        
        # Create DataFrame and set column order
        df = pd.DataFrame(rows)
        
        # Define column order with rank and market_cap_range first
        columns = ['Rank', 'Symbol', 'Market Cap Range' ]
        
        # Store DataFrame in instance variable and return it
        self.scan_results_df = df[columns].sort_values('Rank')
        return df
    
    def update_scan_results(self, allowed_etfs, ib, limit=100):
        
        # copy the scan results to fund results so it has the correct first set of columns 
        self.fund_results_df = self.scan_results_df.copy() 
        self.ta_results_df = self.scan_results_df.copy()

        for s in self.scan_results_df['Symbol']:
            print(f"Processing symbol: {s}")
            sx = stock.StockXDaily(ib, s)
            self.daily_stockx.append(sx)

            # Find the index of the row with the symbol
            index = self.scan_results_df[self.scan_results_df['Symbol'] == s].index[0]

            print("Requesting fundamentals...")
            sx.req_fundamentals(max_days_old=1)
            if sx.fundamentals is not None:
                results_df = sx.get_funadmentals_validation_results(allowed_etfs)
                # Update the row with the validation results
                for key, value in results_df.items():
                    self.scan_results_df.loc[index, key] = value
                    self.fund_results_df.loc[index, key] = value

            print("Requesting OHLCV data...")
            sx.req_ohlcv()
            if not sx.frame.data.empty:
                results_df = sx.get_TA_validation_results()
                # Update the row with the validation results
                for key, value in results_df.items():
                    self.scan_results_df.loc[index, key] = value
                    self.ta_results_df.loc[index, key] = value

            limit -= 1
            print(f"Remaining limit: {limit}")
            if limit == 0:
                break

        return self.scan_results_df

# Usage
# stockbot = StockbotScanner(ib)
# stockbot.multiscan(scan_code='TOP_PERC_GAIN', price=(1, 100), volume=100_000, change_perc=4, market_cap=100)
# stockbot.scan_results_df
