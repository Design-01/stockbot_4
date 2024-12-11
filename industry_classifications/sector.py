import pandas as pd

def get_etf_from_sector_code(classification, sector_code, printout=False):
    file_names = {
        'NIACS': 'NIACS_mapped.csv',
        'TRBC': 'TRBC_mapped.csv',
        'SIC_US': 'SIC_US_mapped.csv',
        'GICS' : 'GICS_mapped.csv'
    }

    code_column = {
        'NIACS': 'Sub-Industry Code',
        'TRBC': 'Hierarchical ID',
        'SIC_US': 'SIC',
        'GICS': 'Sub-Industry Code'
    }

    sector_column = {
        'NIACS': 'Sub-Industry Name',
        'TRBC': 'Sub-Industry Name',
        'SIC_US': 'Description',
        'GICS': 'Sub-Industry Name'
    }

    df = pd.read_csv(file_names[classification])
    etf    = df[df[code_column[classification]]   == sector_code]['etf_ticker'].values[0]
    sector = df[df[code_column[classification]] == sector_code][sector_column[classification]].values[0]
    if  printout: 
        print(f"ETF for {classification} sector {sector} ({sector_code}): {etf}")
    return etf

# Example usage:

# print(get_etf_from_sector_code('TRBC', 5310101014)) 
# print(get_etf_from_sector_code('NIACS', 335312)) 
# print(get_etf_from_sector_code('SIC_US', 3711)) 