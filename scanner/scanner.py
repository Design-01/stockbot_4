import pandas as pd
from ib_insync import IB, ScannerSubscription, TagValue 
import stock
from strategies import ta

class StockbotScanner:
    def __init__(self, ib):
        self.ib = ib
        self.results = []
        
    def scan(self, 
            scanCode:str ='TOP_PERC_GAIN',
            priceRange: tuple[float, float] = (1, 100),
            avgVolumeAbove: int = 100_000,
            changePercent: float = 4,  # Changed from changePercAbove
            marketCapRange: tuple[float, float] = (100, 10000), # value is in millions
            location='STK.US.MAJOR',
            ):
        
        # Store scan parameters
        params = {
            'scanCode': scanCode,
            'priceRange': priceRange,
            'avgVolumeAbove': avgVolumeAbove,
            'changePercent': changePercent,
            'marketCapRange': marketCapRange,
            'location': location
        }
        
        # set tags for scanner based on whether we're looking for gains or losses
        tags = [
            TagValue('priceAbove', priceRange[0]),      
            TagValue('priceBelow', priceRange[1]),
            TagValue('avgVolumeAbove', avgVolumeAbove),
            TagValue('marketCapAbove1e6', marketCapRange[0]),
            TagValue('marketCapBelow1e6', marketCapRange[1])
        ]
        
        # Add the appropriate tag based on whether we're looking for gains or losses
        if changePercent >= 0:
            tags.append(TagValue('changePercAbove', abs(changePercent)))
        else:
            tags.append(TagValue('changePercBelow', -abs(changePercent)))
        
        # Request scanner data
        sub = ScannerSubscription(
            instrument='STK',
            locationCode=location,
            scanCode=scanCode
        )

        print(f'Scanning {location} for {scanCode} ..')
        print(f'price range: {priceRange}, av volume: {avgVolumeAbove}, changePercent: {changePercent}, marketCap: {marketCapRange}')


        data = self.ib.reqScannerData(sub, [], tags)
        
        # Store results with parameters embedded
        for item in data:
            # Attach parameters to each result item
            item.scan_params = params
            
        self.results.extend(data)
        
        print(f'{location} .. len(data): {len(data)}')
        print('-----------------------------------------------------------------------------------')
        return data
    
    def get_results(self):
        data = []
        for d in self.results:
            # Extract the parameters that were attached to each result
            params = getattr(d, 'scan_params', {})
            
            entry = {
                'symbol': d.contractDetails.contract.symbol, 
                'rank': d.rank,
                **params  # Unpack all parameters into the result
            }
            data.append(entry)
        return pd.DataFrame(data)
    
    def save_to_csv(self, folder_path):
        """
        Saves the scan results to a CSV file in the specified folder.
        
        Args:
            folder_path (str): The folder path where the CSV file will be saved.
        
        Returns:
            str: The full path of the saved CSV file.
        """
        import os
        from datetime import datetime
        
        # Get the results as a DataFrame
        results_df = self.get_results()
        
        # If there are no results, print a message and return
        if results_df.empty:
            print("No results to save.")
            return None
        
        # Create the folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)

        current_date = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        filename = f"Scan_Results_{current_date}.csv"
        file_path = os.path.join(folder_path, filename)
        results_df.to_csv(file_path, index=False)
        
        print(f"Results saved to: {file_path}")
        return file_path
