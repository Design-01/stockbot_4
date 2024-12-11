import pandas as pd
import os

def get_etf_from_sector_code(classification, sector_code, printout=False):
    file_names = {
        'NAICS': 'NAICS_mapped.csv',
        'TRBC': 'TRBC_mapped.csv',
        'SIC': 'SIC_US_mapped.csv',
        'GICS' : 'GICS_mapped.csv'
    }

    code_column = {
        'NAICS': 'Sub-Industry Code',
        'TRBC': 'Hierarchical ID',
        'SIC': 'SIC',
        'GICS': 'Sub-Industry Code'
    }

    sector_column = {
        'NAICS': 'Sub-Industry Name',
        'TRBC': 'Sub-Industry Name',
        'SIC': 'Description',
        'GICS': 'Sub-Industry Name'
    }

    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the absolute path to the CSV file
    csv_file_path = os.path.join(current_dir, file_names[classification])

    df = pd.read_csv(csv_file_path)

    sector_code = int(sector_code)
    
    matching_row = df[df[code_column[classification]] == int(sector_code)]
    
    if len(matching_row) > 0:
        return matching_row['etf_ticker'].iloc[0]
    else:
        return None

# Example usage:

# print(get_etf_from_sector_code('TRBC', 5310101014)) 
# print(get_etf_from_sector_code('NAICS', 335312)) 
# print(get_etf_from_sector_code('SIC_US', 3711)) 