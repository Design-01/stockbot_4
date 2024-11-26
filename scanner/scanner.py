import pandas as pd
from ib_insync import IB, ScannerSubscription, TagValue 


class StockbotScanner:
    def __init__(self, ib):
        self.ib = ib
        self.scan_results_df = None

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
            market_cap: float = 100):
        
        market_caps = [(market_cap, market_cap*10), (market_cap*10, market_cap*100), (market_cap*100, market_cap*1000)]
        rows = []

        for cap in market_caps:
            scanData = self.scan(scan_code, price, volume, change_perc, cap)
            cap_str = f'{cap[0]}M-{cap[1]}M'
            
            for data in scanData:
                contract = data.contractDetails.contract
                row = {
                    'rank': data.rank,
                    'market_cap_range': cap,
                    'symbol': contract.symbol,
                    'exchange': contract.exchange,
                    'currency': contract.currency,
                    'primaryExchange': contract.primaryExchange,
                }
                rows.append(row)
                
            self.ib.sleep(60)
        
        # Create DataFrame and set column order
        df = pd.DataFrame(rows)
        
        # Define column order with rank and market_cap_range first
        columns = ['rank', 'market_cap_range', 'symbol', 'exchange', 'currency', 'primaryExchange']
        
        # Reorder columns and sort by rank
        df = df[columns].sort_values('rank')
        
        # Store DataFrame in instance variable and return it
        self.scan_results_df = df
        return df

# Usage
# stockbot = StockbotScanner(ib)
# stockbot.multiscan(scan_code='TOP_PERC_GAIN', price=(1, 100), volume=100_000, change_perc=4, market_cap=100)
# stockbot.scan_results_df