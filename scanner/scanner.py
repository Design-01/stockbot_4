import pandas as pd
from ib_insync import IB, ScannerSubscription, TagValue 
import stock
from strategies import ta

class StockbotScanner:
    def __init__(self, ib):
        self.ib = ib
        self.scan_results_df = None
        self.daily_stockx = []
        self.fund_reults_df = None
        self.ta_results_df = None

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
    
    def update_scan_results(self, scan_results_df, allowed_etfs, ib, send_email_if_any=True, send_email_if_all=True):
            for s in scan_results_df['symbol']:
                sx = stock.StockXDaily(ib, s)
                self.daily_stockx.append(sx)
                sx.req_fundamentals(max_days_old=1)
                sx.req_ohlcv()
                fund_results = sx.get_funadmentals_validation_results(allowed_etfs)
                ta_results = sx.get_TA_validation_results()

                self.fund_results_df = fund_results
                self.ta_results_df = ta_results

                fundametals_passed = fund_results['Fundamentals Passed']
                ta_passed = ta_results['TA Passed']

                # Find the index of the row with the symbol
                index = scan_results_df[scan_results_df['symbol'] == s].index[0]

                # Update the row with the validation results
                for key, value in fund_results.items():
                    scan_results_df.loc[index, key] = value
                
                for key, value in ta_results.items():
                    scan_results_df.loc[index, key] = value

                def format_section(section_dict, section_name):
                    lines = [f"\n{section_name}:"]
                    # Get all items except the "Passed" summary
                    regular_items = {k: v for k, v in section_dict.items() 
                                if not k.endswith('Passed')}
                    
                    # Format regular items
                    if regular_items:
                        # First item gets extra indentation
                        first_key = list(regular_items.keys())[0]
                        lines.append(f"        {first_key}: {regular_items[first_key]}")
                        
                        # Rest of the items
                        for key in list(regular_items.keys())[1:]:
                            lines.append(f"    {key}: {regular_items[key]}")
                    
                    # Add the "Passed" status at the end with extra indentation
                    passed_key = next((k for k in section_dict.keys() if k.endswith('Passed')), None)
                    if passed_key:
                        lines.append(f"        {passed_key}: {section_dict[passed_key]}")
                    
                    return "\n".join(lines)

                # Create the email body
                if send_email_if_any and any([fundametals_passed, ta_passed]):
                    sx.validation_fundamentals_report(asDF=True, save_image=True)
                    sx.frame.setup_chart()
                    sx.frame.plot(show=False)
                    sx.save_chart()
                    sx.save_zoomed_chart(show=False)
                    
                    body = f"{s} passed one or more validation tests"
                    body += format_section(fund_results, "Fundamentals")
                    body += format_section(ta_results, "Technical Analysis")
                    
                    sx.email_report(body=body)

                elif send_email_if_all and all([fundametals_passed, ta_passed]):
                    body = f"{s} passed all validation tests"
                    body += format_section(fund_results, "Fundamentals")
                    body += format_section(ta_results, "Technical Analysis")
                    
                    sx.email_report(body=body)

            return scan_results_df

# Usage
# stockbot = StockbotScanner(ib)
# stockbot.multiscan(scan_code='TOP_PERC_GAIN', price=(1, 100), volume=100_000, change_perc=4, market_cap=100)
# stockbot.scan_results_df