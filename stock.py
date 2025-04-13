from dataclasses import dataclass
from typing import List, Dict, Union, Tuple, Optional
from datetime import datetime
import pandas as pd
from ib_insync import IB, Stock
import xml.etree.ElementTree as ET
import pickle
import os
from my_ib_utils import IBRateLimiter
from pathlib import Path
from dataframe_image import export
import plotly.graph_objects as go
from typing import Dict, Any, List

from data import historical_data as hd
from frame.frame import Frame
from strategies import ta
from industry_classifications.sector import get_etf_from_sector_code
import emails.email_client as email_client
from project_paths import get_project_path
from chart.chart_args import ChartArgs
from strategies.preset_strats import TAPresets1D, TAPresets1H, TAPresets5M2M1M



def parse_xml_value(ratio_element) -> Union[float, str, datetime]:
    """Extract value from a Ratio XML element, handling different data types"""
    field_type = ratio_element.get('Type', 'N')  # Default to numeric if no type specified
    
    # Get the raw value (either from text or nested Value element)
    if ratio_element.text and ratio_element.text.strip():
        raw_value = ratio_element.text.strip()
    else:
        value_elem = ratio_element.find('.//Value')
        if value_elem is not None and value_elem.text:
            raw_value = value_elem.text.strip()
        else:
            return 0.0 if field_type == 'N' else ''
    
    # Parse based on type
    try:
        if field_type == 'N':  # Numeric
            return float(raw_value)
        elif field_type == 'D':  # Date
            return raw_value  # Keep as string for now
        else:  # Default to string for unknown types
            return raw_value
    except ValueError:
        return 0.0 if field_type == 'N' else raw_value

def get_ratio_value(data: Dict[str, Union[float, str]], field: str, default: float = 0.0) -> float:
    """Safely get numeric value from data dictionary"""
    try:
        value = data.get(field, default)
        return float(value) if value != '' else default
    except (ValueError, TypeError):
        return default


@dataclass
class ForecastMetrics:
    """Forecast data from analyst estimates"""
    consensus_recommendation: float  # ConsRecom
    target_price: float             # TargetPrice
    projected_growth_rate: float    # ProjIGrowthRate
    projected_pe: float             # ProjPE
    projected_sales: float          # ProjSales
    projected_sales_growth: float   # ProjSalesQ
    projected_eps: float            # ProjEPS
    projected_eps_q: float          # ProjEPSQ
    projected_profit: float         # ProjProfit
    projected_operating_margin: float  # ProjOPS


@dataclass
class StockIndustries:
    type: str = '' # eg 'GICS', 'NAICS', 'SIC', 'TRBC
    order: int = 0
    code: str = ''
    description: str = ''
    sector_etf: str = ''

@dataclass
class StockFundamentals:
    request_date: str = "" # Date of the request
    # Sector and industry fields
    industry: str = ""
    category: str = ""
    subcategory: str = ""
    primary_etf: str = ""
    secondary_etf: str = ""

    # Stock static information
    currency: str = ''
    longName: str = ''
    timeZoneId: str = ''
    tradingHours: tuple = ('', '') # start and end time eg ('0400', '2000')
    liquidHours: tuple = ('', '') # start and end time eg ('0400', '2000')
    
    # Price and Volume Metrics
    current_price: float = 0.0
    high_52week: float = 0.0
    low_52week: float = 0.0
    pricing_date: str = ""
    volume_10day_avg: float = 0.0
    enterprise_value: float = 0.0
    
    # Income Statement Metrics
    market_cap: float = 0.0
    revenue_ttm: float = 0.0
    ebitda: float = 0.0
    net_income_ttm: float = 0.0
    
    # Per Share Metrics
    eps_ttm: float = 0.0
    revenue_per_share: float = 0.0
    book_value_per_share: float = 0.0
    cash_per_share: float = 0.0
    cash_flow_per_share: float = 0.0
    dividend_per_share: float = 0.0
    
    # Margin Metrics
    gross_margin: float = 0.0
    operating_margin: float = 0.0
    net_profit_margin: float = 0.0
    
    # Growth Metrics
    revenue_growth_rate: float = 0.0
    eps_growth_rate: float = 0.0
    
    # Valuation Metrics
    pe_ratio: float = 0.0
    price_to_book: float = 0.0
    price_to_sales: float = 0.0
    
    # Company Information
    employee_count: int = 0
    
    # Forecast Data
    forecast: Optional[ForecastMetrics] = None

    # industry classifications
    industrys: Optional[List[StockIndustries]] = None
    
    # Computed Metrics
    price_to_10day_avg: float = 0.0
    volume_vs_10day_avg_pct: float = 0.0
    distance_from_52wk_high_pct: float = 0.0
    distance_from_52wk_low_pct: float = 0.0

    def compute_derived_metrics(self, current_volume: float = 0):
        """Compute additional metrics based on available data"""
        if self.current_price and self.high_52week:
            self.distance_from_52wk_high_pct = ((self.high_52week - self.current_price) / self.high_52week) * 100
            
        if self.current_price and self.low_52week:
            self.distance_from_52wk_low_pct = ((self.current_price - self.low_52week) / self.low_52week) * 100
            
        if current_volume and self.volume_10day_avg:
            self.volume_vs_10day_avg_pct = ((current_volume - self.volume_10day_avg) / self.volume_10day_avg) * 100


def extract_trading_hours(trading_hours_str):
    """Exmaple of trading_hours_str: '20241211:0400-20241211:2000;20241212:0400-20241212:2000;20241213:0400-20241213:2000;20241214:CLOSED;20241215:CLOSED;20241216:0400-20241216:2000',
    return ('0400', '2000')
    """
    for day in trading_hours_str.split(';'):
        if 'CLOSED' not in day:
            dtime = day.split('-')
            return (dtime[0].split(':')[1], dtime[1].split(':')[1])

def summarize_sector_etfs(df):
    """
    Summarizes sector ETFs with weighted scores based on the `order` column.
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns ['type', 'order', 'code', 'description', 'sector_etf'].
        
    Returns:
        dict: Dictionary with unique ETF tickers as keys and aggregated weighted scores as values.
    """
    # Calculate weights as 1/order
    df['weight'] = 1 / df['order']
    
    # Normalize weights within each `type`
    df['normalized_weight'] = df.groupby('type')['weight'].transform(lambda x: x / x.sum())
    
    # Aggregate normalized weights by `sector_etf`
    aggregated_weights = df.groupby('sector_etf')['normalized_weight'].sum()
    
    # Normalize aggregated weights to percentages (sum to 1)
    total_weight = aggregated_weights.sum()
    percentages = (aggregated_weights / total_weight).to_dict()
    
    # Round percentages to 2 decimal places
    rounded_percentages = {k: round(v, 2) for k, v in percentages.items()}
    
    # Convert to list of tuples and sort by percentage in descending order
    sorted_percentages = sorted(rounded_percentages.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_percentages

def get_fundamentals_from_file(symbol: str, max_days_old: int = 1) -> Optional[StockFundamentals]:
    """
    Retrieve fundamental data for a stock symbol if it exists and is not too old.
    If max_days_old is 0, returns the data regardless of age.
    """
    filename = get_project_path('data', 'fundamental_data_store', f'{symbol.upper()}_fundamentals.pkl')
    
    if not os.path.exists(filename):
        return None
    
    try:
        with open(filename, 'rb') as f:
            data: StockFundamentals = pickle.load(f)
        
        # Skip age check if max_days_old is 0
        if max_days_old > 0:
            stored_date = datetime.strptime(data.request_date, '%Y-%m-%d %H:%M:%S')
            days_old = (datetime.now() - stored_date).days
            
            if days_old > max_days_old:
                return None
                
        return data
        
    except Exception as e:
        print(f"Error loading fundamentals for {symbol}: {str(e)}")
        return None

def get_stock_fundamentals(ib, ticker: str, current_volume: float = 0, max_days_old=0) -> StockFundamentals:
    """Retrieve comprehensive stock information including all available ratios."""
    rate_limiter = IBRateLimiter(ib)
    contract = Stock(ticker, 'SMART', 'USD')
    rate_limiter.wait()
    details = ib.reqContractDetails(contract)
    if not details:
        raise ValueError(f"No contract details found for {ticker}")
    
    # try from file first if within max_days_old
    if max_days_old != 0:
        file_data = get_fundamentals_from_file(ticker, max_days_old)
        if file_data:
            print(f"Using cached data for {ticker} (age: {file_data.request_date})")
            return file_data
    
    rate_limiter.wait()
    fundamental_data = ib.reqFundamentalData(contract, 'ReportSnapshot')
    if not fundamental_data:
        return None
    
    root = ET.fromstring(fundamental_data)

    # Initialize data dictionary
    data = {}
    
    # Process all ratio elements
    for ratio in root.findall('.//Ratio'):
        field_name = ratio.get('FieldName')
        if field_name:
            data[field_name] = parse_xml_value(ratio)
    
    # Process forecast data
    forecast_data = root.find('.//ForecastData')
    forecast = None
    if forecast_data is not None:
        try:
            forecast = ForecastMetrics(
                consensus_recommendation=get_ratio_value(data, 'ConsRecom'),
                target_price=get_ratio_value(data, 'TargetPrice'),
                projected_growth_rate=get_ratio_value(data, 'ProjIGrowthRate'),
                projected_pe=get_ratio_value(data, 'ProjPE'),
                projected_sales=get_ratio_value(data, 'ProjSales'),
                projected_sales_growth=get_ratio_value(data, 'ProjSalesQ'),
                projected_eps=get_ratio_value(data, 'ProjEPS'),
                projected_eps_q=get_ratio_value(data, 'ProjEPSQ'),
                projected_profit=get_ratio_value(data, 'ProjProfit'),
                projected_operating_margin=get_ratio_value(data, 'ProjOPS')
            )
        except Exception as e:
            print(f"Warning: Could not parse forecast data: {e}")
            forecast = None

    industries = []
    
    for industry in root.findall('.//Industry'):
        industry_code_type = industry.get('type', '')
        industry_order = int(industry.get('order', '0'))
        industry_code = industry.get('code', '')
        industry_description = industry.text.strip() if industry.text else ''
        
        industries.append(StockIndustries(
            type = industry_code_type,
            order = industry_order,
            code = industry_code,
            description = industry_description,
            sector_etf = get_etf_from_sector_code(industry_code_type, industry_code)
        ))

    list_of_etfs = summarize_sector_etfs(pd.DataFrame([i.__dict__ for i in industries]))
    
    stock_info = StockFundamentals(
        request_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        # Industry Classification
        industry = details[0].industry,
        category = details[0].category,
        subcategory = details[0].subcategory,

        # etf
        primary_etf = list_of_etfs[0] if len(list_of_etfs) > 0 else None,
        secondary_etf = list_of_etfs[1] if len(list_of_etfs) > 1 else None,
        
        # Stock static information
        currency= details[0].contract.currency,
        longName = details[0].longName,
        timeZoneId = details[0].timeZoneId,
        tradingHours = extract_trading_hours(details[0].tradingHours),
        liquidHours = extract_trading_hours(details[0].liquidHours),
        
        # Price and Volume
        current_price=get_ratio_value(data, 'NPRICE'),
        high_52week=get_ratio_value(data, 'NHIG'),
        low_52week=get_ratio_value(data, 'NLOW'),
        pricing_date=data.get('PDATE', ''),
        volume_10day_avg=get_ratio_value(data, 'VOL10DAVG'),
        enterprise_value=get_ratio_value(data, 'EV'),
        
        # Income Statement
        market_cap=get_ratio_value(data, 'MKTCAP'),
        revenue_ttm=get_ratio_value(data, 'TTMREV'),
        ebitda=get_ratio_value(data, 'TTMEBITD'),
        net_income_ttm=get_ratio_value(data, 'TTMNIAC'),
        
        # Per Share
        eps_ttm=get_ratio_value(data, 'TTMEPSXCLX'),
        revenue_per_share=get_ratio_value(data, 'TTMREVPS'),
        book_value_per_share=get_ratio_value(data, 'QBVPS'),
        cash_per_share=get_ratio_value(data, 'QCSHPS'),
        cash_flow_per_share=get_ratio_value(data, 'TTMCFSHR'),
        dividend_per_share=get_ratio_value(data, 'TTMDIVSHR'),
        
        # Margins
        gross_margin=get_ratio_value(data, 'TTMGROSMGN'),
        operating_margin=get_ratio_value(data, 'TTMOPMGN'),
        net_profit_margin=get_ratio_value(data, 'TTMNPMGN'),
        
        # Growth
        revenue_growth_rate=get_ratio_value(data, 'TTMREVCHG'),
        eps_growth_rate=get_ratio_value(data, 'TTMEPSCHG'),
        
        # Valuation
        pe_ratio=get_ratio_value(data, 'PEEXCLXOR'),
        price_to_book=get_ratio_value(data, 'PRICE2BK'),
        price_to_sales=get_ratio_value(data, 'TMPR2REV'),
        
        # Company Info
        employee_count=int(get_ratio_value(data, 'Employees')),
        
        # Forecast
        forecast=forecast,

        # industry classifications
        industrys = industries
    )
    
    # Compute additional metrics
    stock_info.compute_derived_metrics(current_volume)

    # Save data to file
    save_fundamentals(ticker, stock_info)
    
    return stock_info

def save_fundamentals(symbol: str, data: StockFundamentals) -> None:
    """
    Save fundamental data for a stock symbol to a file.
    
    Args:
        symbol: Stock symbol
        data: StockFundamentals data class instance
    """
    file_path = get_project_path('data', 'fundamental_data_store', f'{symbol.upper()}_fundamentals.pkl')
    
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Error saving fundamentals for {symbol} at {file_path}: {str(e)}")
        print(f"File exists: {file_path.exists()}")
        print(f"Directory exists: {file_path.parent.exists()}")
        raise


class StockXDaily:
    def __init__(self, ib:IB, symbol):
        self.ib = ib
        self.symbol = symbol
        self.fundamentals: StockFundamentals = None
        self.fundamentals_validation_results = {}
        self.frame = Frame(self.symbol)
        self.frame_validation_results = {}
        self.frame_score_results = {}
        self.dayscores = None
        
        # self.image_path = {
        #     'valid_fundamentals_df': f'{self.symbol}_fundamentals.png',
        #     'valid_ta_df': f'{self.symbol}_ta.png',
        #     'chart': f'{self.symbol}.png',
        #     'zoomed_chart': f'{self.symbol}_zoomed.png',
        # }
        self.image_path = {
            'valid_fundamentals_df': get_project_path('data', 'fundamental_data_store', f'{self.symbol}_fundamentals.png'),
            'valid_ta_df':           get_project_path('data', 'fundamental_data_store', f'{self.symbol}_ta.png'),
            'chart':                 get_project_path('data', 'fundamental_data_store', f'{self.symbol}.png'),
            'zoomed_chart':          get_project_path('data', 'fundamental_data_store', f'{self.symbol}_zoomed.png'),
        }

    def req_fundamentals(self, max_days_old=0):
        if not self.fundamentals:
            self.fundamentals = get_stock_fundamentals(self.ib, self.symbol, max_days_old=max_days_old)
        return self.fundamentals
    
    def validate_fundamental(self, key: str, validation_type: str, value: Union[float, int, str, List, Tuple], description:str='') -> bool:
        """
        Validates a fundamental value against a specified condition.
        
        Args:
            key (str): The fundamental metric to validate (e.g., 'pe_ratio', 'primary_etf')
            validation_type (str): Type of validation ('isin', '>', '<', '>=', '<=', '==', '!=', 'rng')
            value: Value to compare against. For 'rng', provide tuple of (min, max)
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        if self.fundamentals is None:
            raise ValueError("Fundamentals data not initialized")
            
        # Check if the fundamental exists
        if not hasattr(self.fundamentals, key):
            raise KeyError(f"Fundamental '{key}' not found")
            
        # Get the fundamental value
        fundamental_value = getattr(self.fundamentals, key)
        
        # Handle None/null values
        if fundamental_value is None:
            result = False
        else:
            try:
                # Handle special case for ETF tuples
                # if key in ['primary_etf', 'secondary_etf']:
                if key in ['primary_etf']:
                    if validation_type == 'isin':
                        result = fundamental_value[0] in value
                    else:
                        # For numeric comparisons, use the weight value
                        try:
                            fval = float(fundamental_value[1])
                            compare_value = float(value)
                        except (ValueError, TypeError):
                            raise ValueError("Cannot compare ETF weight with non-numeric value")
                        
                        if validation_type == '>':
                            result = fval > compare_value
                        elif validation_type == '<':
                            result = fval < compare_value
                        elif validation_type == '>=':
                            result = fval >= compare_value
                        elif validation_type == '<=':
                            result = fval <= compare_value
                        elif validation_type == '==':
                            result = fval == compare_value
                        elif validation_type == '!=':
                            result = fval != compare_value
                        elif validation_type == 'rng':
                            min_val, max_val = float(value[0]), float(value[1])
                            result = min_val <= fval <= max_val
                        else:
                            raise ValueError(f"Invalid validation type: {validation_type}")
                
                # Handle other cases
                elif validation_type == 'isin':
                    if not isinstance(value, (list, tuple)):
                        raise ValueError("Value must be a list or tuple for 'isin' validation")
                    result = fundamental_value in value
                
                elif validation_type == 'rng':
                    if not isinstance(value, (list, tuple)) or len(value) != 2:
                        raise ValueError("Value must be a tuple/list of (min, max) for range validation")
                    try:
                        fval = float(fundamental_value)
                        min_val, max_val = float(value[0]), float(value[1])
                        result = min_val <= fval <= max_val
                    except (ValueError, TypeError):
                        raise ValueError("Value and range bounds must be numeric")
                
                else:  # Numeric comparisons
                    try:
                        fval = float(fundamental_value)
                        compare_value = float(value)
                    except (ValueError, TypeError):
                        raise ValueError("Both fundamental value and comparison value must be numeric")
                    
                    if validation_type == '>':
                        result = fval > compare_value
                    elif validation_type == '<':
                        result = fval < compare_value
                    elif validation_type == '>=':
                        result = fval >= compare_value
                    elif validation_type == '<=':
                        result = fval <= compare_value
                    elif validation_type == '==':
                        result = fval == compare_value
                    elif validation_type == '!=':
                        result = fval != compare_value
                    else:
                        raise ValueError(f"Invalid validation type: {validation_type}")
            
            except Exception as e:
                result = False
                print(f"Validation error for {key}: {str(e)}")
        
        # Store the validation result
        validation_id = f"{key}_{validation_type}_{str(value)}"
        self.fundamentals_validation_results[validation_id] = {
            'key': key,
            'description': description,
            'validation_type': validation_type,
            'comparison_value': value,
            'actual_value': fundamental_value,
            'passed': result
        }
        
        return result
    
    def validation_fundamentals_report(self, asDF: bool = True, save_image:bool = False) -> Union[pd.DataFrame, Dict]:
        """
        Returns a report of all validations performed.
        
        Args:
            asDF (bool): If True, returns pandas DataFrame, otherwise returns dictionary
            
        Returns:
            Union[pd.DataFrame, Dict]: Validation results
        """
        if asDF:
            df = pd.DataFrame.from_dict(self.fundamentals_validation_results, orient='index')
            if save_image:
                self.save_df_as_image(df, self.image_path['valid_fundamentals_df'])
            return df
        return self.fundamentals_validation_results
    
    def get_funadmentals_validation_results(self, allowed_etfs: List[str]) -> Dict: 
        return {
            'Sector1 Valid': self.validate_fundamental('primary_etf', 'isin', allowed_etfs, description='Stocks primary sector ETF is allowed'),
            # 'Sector2 Valid': self.validate_fundamental('secondary_etf', 'isin', allowed_etfs, description='Stocks primary sector ETF is allowed'),
            'Market Cap > 300M': self.validate_fundamental('market_cap', '>=', 300, description='Market cap is greater than 300M'),
            'Vol 10DayMA > 300K': self.validate_fundamental('volume_10day_avg', '>=', 0.3, description='Volume is greater than 300k'),
            'Fundamentals Passed': self.validation_fundamentals_has_passed()
        }
    
    def get_TA_validation_results(self) -> Dict:
        """Returns a dictionary of technical analysis validation results. Each key represents a validation check and its value is a boolean result."""
        return {
            'Close > $1'       : self.validate_TA(ta.ColVal('close'),          ta.AboveBelow('CV_close', 'above', 1),         description='close price is above 1'),
            'Above 200MA'      : self.validate_TA(ta.MA('close', 200),         ta.AboveBelow('close', 'above', 'MA_cl_200'),  description='close price is above 200 MA',      style={'dash': 'solid', 'color': 'cyan', 'width': 3}, row=1),
            'Above 150MA'      : self.validate_TA(ta.MA('close', 150),         ta.AboveBelow('close', 'above', 'MA_cl_150'),  description='close price is above 150 MA',      style={'dash': 'solid', 'color': 'pink', 'width': 2}, row=1),
            'Breaks Above 50MA': self.validate_TA(ta.MA('close', 50),          ta.Breaks('close', 'above', 'MA_cl_50'),       description='close price breaks above 50 MA',   style={'dash': 'solid', 'color': 'purple', 'width': 2}, row=1),
            '50MA Slope > 0'   : self.validate_TA(ta.PctChange('MA_cl_50', 1), ta.AboveBelow('PCT_MA_cl_50_1', 'above', 0),   description='pct change of 50 MA is above 0'),
            'Gap Up > 4%'      : self.validate_TA(ta.PctChange('close', 1),    ta.AboveBelow('PCT_close_1', 'above', 4),      description='pct change of close is above 4 (4% Gap)'),
            'Volume > 50K'     : self.validate_TA(ta.ColVal('volume'),         ta.AboveBelow('CV_volume', 'above', 50_000),   description='volume is above 50k'),
            'Volume Above 10MA': self.validate_TA(ta.MA('volume', 10),         ta.Breaks('volume', 'above', 'MA_vo_10'),      description='volume breaks above 10 MA',              style={'dash': 'solid', 'color': 'pink', 'width': 1}, row=2),
            'Volume Dev > 80%' : self.validate_TA(ta.VolDev('volume', 10),     ta.AboveBelow('VDEV_10', 'above', 80),         description='volume is above 80% of 10 MA Deviation', style={'dash': 'solid', 'color': 'pink', 'width': 1}, row=3, ),
            'TA Passed'        : self.validation_TA_has_passed()
        }
    
    def validation_TA_report(self, asDF: bool = True, save_image:bool = False) -> Union[pd.DataFrame, Dict]:
        """
        Returns a report of all validations performed.
        
        Args:
            asDF (bool): If True, returns pandas DataFrame, otherwise returns dictionary
            
        Returns:
            Union[pd.DataFrame, Dict]: Validation results
        """
        if asDF:
            df = pd.DataFrame.from_dict(self.frame_validation_results, orient='index')
            if save_image:
                self.save_df_as_image(df, self.image_path['valid_ta_df'])
            return df
        return self.frame_validation_results
    
    def validation_fundamentals_has_passed(self) -> float:
        """
        Calculates the ratio of passed validations.
        
        Returns:
            float: Ratio of passed validations to total validations
        """
        if not self.fundamentals_validation_results:
            return 0.0
        
        total_validations = len(self.fundamentals_validation_results)
        passed_validations = sum(1 for result in self.fundamentals_validation_results.values() if result['passed'])
        
        result = passed_validations / total_validations if total_validations > 0 else 0.0
        return round(result, 2)

    def validation_TA_has_passed(self) -> float:
        """
        Calculates the ratio of passed validations.
        
        Args:
            maxFails (int): Maximum number of allowed failures (not used in ratio calculation)
            
        Returns:
            float: Ratio of passed validations to total validations, rounded to two decimal places
        """
        if not self.frame_validation_results:
            return 0.0
        
        total_validations = len(self.frame_validation_results)
        passed_validations = sum(1 for result in self.frame_validation_results.values() if result['passed'])
        
        ratio = passed_validations / total_validations if total_validations > 0 else 0.0
        return round(ratio, 2)


    def req_ohlcv(self, start_date:str="52 weeksAgo", end_date:str='now'):
        if self.frame.data.empty:
             self.frame.load_ohlcv(hd.get_hist_data(self.symbol, start_date, end_date, '1 day'))
        return self.frame.data
    
    def add_TA(self, ta_indicator, style={}, row=None):
        default_row = 2 if row is None else row 
        self.frame.add_ta(ta_indicator, style, row=default_row)
        self.frame.update_ta_data()
        return self.frame.data

    def validate_TA(self, ta_indicator, ta_validator, style={}, row=None, description=None):
        """
        Validates technical analysis indicators against specified validation criteria.
        
        Args:
            ta_indicator: The technical indicator to be added (e.g., MA, VolDev)
            ta_validator: The validation method to test the indicator (e.g., Breaks, AboveBelow)
            style (dict, optional): Plotting style parameters. If None, uses default style.
        
        Returns:
            bool: Result of the technical analysis validation
        """
        # Default style if none provided
        default_row = 2 if row is None else row 
        
        # Add the technical indicator to the frame
        self.frame.add_ta(ta_indicator, style, row=default_row)
        self.frame.add_ta(ta_validator)
        
        # Create a unique identifier for this validation
        validation_id = f"{ta_indicator.name}_{ta_validator.name}"
        
        # Initialize technical validation results dictionary if it doesn't exist
        if not hasattr(self, 'technical_validation_results'):
            self.technical_validation_results = {}
        
        # Store the validation result
        self.frame_validation_results[validation_id] = {
            'description': description,
            'indicator': ta_indicator.name,
            'validator': ta_validator.name,
            'validation_type': type(ta_validator).__name__,  # e.g., 'Breaks' or 'AboveBelow'
            'indicator_value': self.frame.data[ta_indicator.name].iloc[-1],
            'passed': self.frame.data[ta_validator.name].iloc[-1],
            'timestamp': self.frame.data.index[-1],  # Store the timestamp of the validation
        }
        
        return self.frame_validation_results[validation_id]['passed']
    
    def score_TA(self, signal, scorer, style={}, row=None, description=None):
        """
        Scores technical analysis indicators against specified criteria.
        
        Args:
            ta_indicator: The technical indicator to be added (e.g., MA, VolDev)
            ta_validator: The validation method to test the indicator (e.g., Breaks, AboveBelow)
            style (dict, optional): Plotting style parameters. If None, uses default style.
        
        Returns:
            float: Score of the technical analysis
        """
        # Default style if none provided
        default_row = 2 if row is None else row 
        
        # Add the technical indicator to the frame
        self.frame.add_signals(signal, style, row=default_row)
        
        # Load and update the data
        self.frame.update_signals_data()
        
        # Add the validator
        self.frame.add_signals(scorer, style, row=default_row)
        
        # # Update data again to include validator results
        # self.frame.load_ohlcv(self.frame.data)

        
        # Create a unique identifier for this validation
        validation_id = f"{signal.name}_{scorer.name}"
        
        # Initialize technical validation results dictionary if it doesn't exist
        if not hasattr(self, 'technical_validation_results'):
            self.technical_validation_results = {}
        
        # Calculate the score (example: difference between indicator value and validator value)
        signal_value = self.frame.data[signal.name].iloc[-1]
        score_value = self.frame.data[scorer.name].iloc[-1]
        score = abs(signal_value - score_value)  # Example scoring logic
        
        # Store the score result
        self.frame_score_results[validation_id] = {
            'description': description,
            'signal': signal.name,
            'scorer': scorer.name,
            'validation_type': type(scorer).__name__,  # e.g., 'Breaks' or 'AboveBelow'
            'signal_value': signal_value,
            'score_value': score_value,
            'score': score,
            'timestamp': self.frame.data.index[-1],  # Store the timestamp of the validation
        }
        
        return self.frame_score_results[validation_id]['score']
    
    def save_chart(self, width=1400, height=800):
        self.frame.chart.save_chart(self.image_path['chart']) # saves the chart to a file

    def save_zoomed_chart(self, width=1400, height=800, path=None, show:bool = False):
        path = self.image_path['zoomed_chart'] if path is None else path
        start_date = self.frame.data.index[-50]  # 50 bars from the end
        end_date = self.frame.data.index[-1]     # to the latest bar
        self.frame.chart.save_chart_region( start_date, end_date, x_padding='1D', y_padding_pct=0.1, filename=path, plot=show) # saves the chart to a file

    def save_df_as_image(self, df, image_path=None):
        """
        Format DataFrame for display and save as image with dark theme and subtle grid
        
        Parameters:
        df: pandas DataFrame
        image_path: path to save the image
        """
        # Create a copy to avoid modifying the original
        display_df = df.copy()
        
        # Format numeric values
        for col in display_df.columns:
            if col == 'indicator_value':
                display_df[col] = display_df[col].apply(lambda x: f"{float(x):.2f}" if isinstance(x, (int, float)) else x)
            elif col == 'timestamp':
                display_df[col] = pd.to_datetime(display_df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(display_df.columns),
                align='left',
                font=dict(size=12, color='#A9A9A9'),
                height=40,
                fill=dict(color='#2F2F2F'),
                line=dict(color='#1A1A1A', width=1)  # Very dark grey lines
            ),
            cells=dict(
                values=[display_df[col] for col in display_df.columns],
                align='left',
                font=dict(size=11, color='#A9A9A9'),
                height=35,
                fill=dict(color='#000000'),
                line=dict(color='#1A1A1A', width=1)  # Very dark grey lines
            )
        )])

        fig.update_layout(
            width=1200,
            height=len(display_df) * 35 + 100,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='black',
            plot_bgcolor='black'
        )
        fig.write_image(image_path)
        return image_path
    
    def email_report(self, subject:str = None, body:str = None ):
        """
        Send an email report with optional attachments.
        """        
        # Send email with attachments
        subject = subject if subject else f'STOCKBOT Alert: {self.symbol} Daily Analysis'
        body    = body if body else f"Attached is the daily analysis for {self.symbol}"

        email_client.send_outlook_email(
            subject = subject,
            body = body,
            recipients = ['pary888@gmail.com'],
            image_paths = [path for path in self.image_path.values()],
            is_html = False)
        
        print(f"Email sent to for {self.symbol}")

from data import historical_data as hd
import compare
from typing import Union, Tuple
import pandas as pd
import numpy as np



# ----------------------------------------------------------
# ------- S T O C K X  -------------------------------------
# ----------------------------------------------------------
import pandas as pd
from enum import Enum
from IPython.display import display
from dataclasses import dataclass, field
from typing import Dict, Any, List
from ib_insync import *

from data.random_data import RandomOHLCV
from frame.frame import Frame
from data import historical_data as hd
import strategies.ta as ta
import strategies.signals as sig
import stock_fundamentals
import strategies.preset_strats as ps
from trades.price_x import EntryX, StopX, TargetX, TrailX
from trades.tradex import TraderX
import time
from data.live_ib_data import LiveData
from data.historical_data import HistoricalData
import my_ib_utils


@dataclass
class TAData:
    ta: ta.TA
    style: Dict[str, Any] | List[Dict[str, Any]] = field(default_factory=dict)
    chartType: str = "line"
    row: int = 1

class SignalStatus:
    DAILY_TA = "DAILY_TA"
    PRE_MARKET = "PRE_MARKET"
    IN_TRADE  = "IN_TRADE"
    PENDING   = "PENDING"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"
    INACTIVE  = "INACTIVE"

@dataclass
class StockStatsDaily:
    # Stock information
    symbol: str = ''
    ls: str = '' # LONG or SHORT
    snapshotTime: datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    status: SignalStatus = SignalStatus.DAILY_TA

    # Basic fundamentals
    price: float = 0.0
    marketCap: float = 0.0
    currentPrice: float = 0.0
    peRatio: float = 0.0
    vol10dayAvg: float = 0.0

    # Daily stats
    vol_TODC : float = 0.0 # Volume Time of Day Change
    priceChgPct: float = 0.0
    breaksAbove10DayAvg: bool = False
    breaksAbove50MA: bool = False
    breaksAbove200MA: bool = False
    breaksBelow50MA: bool = False
    breaksBelow200MA: bool = False

    # signals
    sigGappedWRBs: float = 0.0
    sigGappedPivs: float = 0.0
    sigGappedPastPiv: float = 0.0
    sigRTM: float = 0.0
    sigRS: float = 0.0
    sigVolume: float = 0.0

    # Scores
    scoreVol : float = 0.0
    scoreGaps : float = 0.0
    scoreRTM : float = 0.0
    score_1D : float = 0.0

    # validation results
    validAv_1D : float = 0.0

@dataclass
class StockStats:
    # Stock information
    symbol: str = ''

    # Core signal information
    status: SignalStatus = SignalStatus.INACTIVE
    status_why: str = ""  # Internal notes explaining current status

    # Score information
    score_1D: float = 0.0
    score_vol_TODC: float = 0.0
    score_5M: float = 0.0
    score_2M: float = 0.0
    score_AVG: float = 0.0
    score_best_barsize: str = ''
    
    # Performance metrics
    pnl_realized: float = 0.0
    pnl_unrealized: float = 0.0
    trades_today: int = 0
    win_rate: float = 0.0  # Percentage of winning trades
    av_Rratio: float = 0.0  # Average R ratio per trade
    
    # Timestamps
    last_updated: datetime = field(default_factory=datetime.now)
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None

    def update_score_av(self, scores:list[str]=None):
        scores = scores if scores else [self.score_1D, self.score_vol_TODC, self.score_5M, self.score_2M] 
        self.score_AVG = round(sum(scores) / len(scores), 1)   
        
@dataclass
class StockScoreCols:
    vol_TODC : str = ''
    D1 : str = ''
    M5 : str = ''
    M2 : str = ''
    M1 : str = ''
    time : str = ''
    level_premkt_gt : str = ''
    touches : str = ''
    reset_if_breaks : str = ''
    pullback : str = ''
    retest : str = ''
    bsw : str = ''
    buysetup : str = ''
    rtm : str = ''
    buy : str = '' 
    gaps : str = ''

@dataclass
class StockX:
    ib: IB = None
    symbol: str = ''
    ls: str = '' # LONG or SHORT
    intradaySizes: List[str] = field(default_factory=list)
    tradeSizes: List[str] = field(default_factory=list)
    riskAmount: float = 100.00
    outsideRth: bool = False
    isMarket: bool = False # to identify if this stock is used for main market eg SPY or QQQ


    def __post_init__(self):
        self.fundamentals = stock_fundamentals.Fundamentals(self.ib, self.symbol)
        self.frames = {}
        self.stats = StockStats(self.symbol)
        self.stats_daily = StockStatsDaily(self.symbol)
        self.trader = TraderX(self.ib, self.symbol)
        self.livedata = LiveData(self.ib)
        self.historicaldata = HistoricalData(self.ib)   
        self.spy = None
        self.score_cols = StockScoreCols()  # a way of managing the various score columns produced and sharing accorss the different methods
        self.isMarket = True if self.symbol in ['SPY', 'QQQ'] else False
    # ------ Get Methods ----------------

    def get_frame(self, timeframe:str):
        if timeframe not in self.frames:
            return None
        return self.frames[timeframe]
    
    def get_frame_data(self, timeframe:str, colsContains:List[str]=None):
        if timeframe not in self.frames:
            return None
        if colsContains:
            cols  = [col for col in self.frames[timeframe].data.columns if colsContains in col]
            return self.frames[timeframe].data[cols]
        return self.frames[timeframe].data
    
    def get_score_status_by_item(self, item_name: str, dataType:str) -> float:
        """
        Returns the score corresponding to the given item name from the DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame containing 'item' and 'score' columns.
        item_name (str): The name of the item to look up.

        Returns:
        float: The score corresponding to the item name.
        """
        df = self.get_status_df()
        row = df.loc[(df['item'] == item_name) & (df['dataType'] == dataType)]
        if not row.empty:
            return row['score'].iat[0]
        else:
            raise ValueError(f"Item '{item_name}' not found in the DataFrame")

    def get_stats(self) -> StockStats:
        return self.stats

    def get_today_ohlcv(self):
        contract = Stock(self.symbol, 'SMART', 'USD')
        
        # Check if there's already an active market data request for this contract
        existing_tickers = [ticker for ticker in self.ib.tickers() if ticker.contract == contract]
        
        if existing_tickers:
            # If there's an existing ticker, cancel it first
            self.ib.cancelMktData(existing_tickers[0].contract)
        
        # Request new market data
        ticker = self.ib.reqMktData(contract)
        self.ib.sleep(2)
        return ticker.open, ticker.high, ticker.low, ticker.last, ticker.volume

    #!------ Check Methods ---------------- Work in Progress

    def day_score_passed(self, min_score:float = 0.5) -> bool:
        return self.stats.score_1D >= min_score
    
    def pre_market_passed(self, min_score:float = 0.5) -> bool:
        return self.stats.score_5M >= min_score

    # ------- Sector  -------------------

    def req_fundamentals(self, max_days_old=0):
        self.fundamentals.req_fundamentals(max_days_old)

    def sector_ETF_is_allowed(self, allowed_etfs: List[str]) -> bool:
        """
        Determines if a stock should be traded based on its ETF composition and allowed ETFs.
        
        The method uses the following criteria:
        1. If both primary and secondary ETFs are in allowed list - return True
        2. If only primary ETF is allowed and its weight > 0.5 - return True
        3. If only secondary ETF is allowed and combined non-allowed ETF weight < 0.5 - return True
        4. Otherwise - return False
        
        Args:
            allowed_etfs (List[str]): List of ETF ticker symbols that are allowed for trading
            
        Returns:
            bool: True if the stock meets the ETF criteria for trading, False otherwise
            
        Example:
            If allowed_etfs = ['XLY', 'XLK'] and stock has:
            - primary_etf = ('XLY', 0.56)
            - secondary_etf = ('XLI', 0.44)
            Returns True because primary ETF is allowed and weight > 0.5
        """
        # Get ETF information from fundamentals
        primary_etf, primary_weight = self.fundamentals.fundamentals.primary_etf
        secondary_etf, secondary_weight = self.fundamentals.fundamentals.secondary_etf
        
        # Check if ETFs are in allowed list
        primary_allowed = primary_etf in allowed_etfs
        secondary_allowed = secondary_etf in allowed_etfs
        
        # Case 1: Both ETFs are allowed
        if primary_allowed and secondary_allowed:
            return True
        
        # Case 2: Only primary ETF is allowed but has dominant weight
        if primary_allowed and not secondary_allowed:
            return primary_weight > 0.5
        
        # Case 3: Only secondary ETF is allowed
        if secondary_allowed and not primary_allowed:
            return primary_weight < 0.5  # Same as secondary_weight > 0.5
        
        # Case 4: Neither ETF is allowed
        return False
    
    # ------- Set up -------------------

    def set_ls(self, ls:str):
        if ls not in ['LONG', 'SHORT']:
            raise ValueError(f"Invalid ls value: {ls}")
        self.ls = ls
        self.trader.set_ls(ls)

    #! Not Used
    def get_ta_preset(self, barSize:str):
        if barSize == '1 day': return TAPresets1D()
        if barSize == '1 hour': return TAPresets1H()
        if barSize in ['5 mins', '2 mins', '1 min']: return TAPresets5M2M1M()


    def setup_frame(self, timeframe, dataType:str='random', duration:str="3 D", endDateTime:str='now', isIntradayFrame:bool=False, isTradeFrame:bool=False, isDayFrame:bool=False, taPresets:TAPresets1D | TAPresets1H | TAPresets5M2M1M=None):
        if isIntradayFrame:
            self.intradaySizes.append(timeframe)
        if isTradeFrame:
            self.tradeSizes.append(timeframe)
        name = timeframe          
        
        self.frames[name] = Frame(self.symbol, name=name, rowHeights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.5], taPresets=taPresets)
        
        if dataType == 'random':
            df =  RandomOHLCV( 
            freq      = timeframe, 
            head_max  = 0.3, 
            tail_max  = 0.3, 
            start     = '2024',           
            open_val  = 100.00,           
            periods   = 400, 
            open_rng  = (-0.4, 0.4), 
            close_rng = (-0.4, 0.4), 
            vol_rng   = (-1, 1),
            volatility_rng  = (0, 0.02),
            volatility_dur  = 3,
            volatility_freq = 50).get_dataframe()

            self.frames[timeframe].load_ohlcv(df)

        elif dataType == 'ohlcv':

            df = self.historicaldata.get_data(self.symbol, timeframe, endDateTime,  durationStr=duration, print_info=True)
            self.frames[name].load_ohlcv(df)

        elif dataType == 'tick':
            # todo: implement tick data
            pass

        if isDayFrame:
            df_last = self.frames[name].data.index[-1].date()
            today = datetime.now().date()
            if today > df_last:
                if my_ib_utils.is_market_day(today=True):
                    self.frames[name].data = self.add_today_row_live_data(self.frames[name].data)

    def add_today_row_live_data(self, df):
        today_date = datetime.now().strftime('%Y-%m-%d')
        df.loc[today_date] = np.nan
        op, hi, lo, cl, vol = self.get_today_ohlcv()
        df.loc[today_date, ['open', 'high', 'low', 'close', 'volume']] = op, hi, lo, cl, vol
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d') 
        return df

    def setup_all_frames(self, dataType:str='ohlcv', end_date="now", force_download:bool=False):
        self.setup_frame('1 day', dataType, start_date="52 weeksAgo", end_date=end_date, force_download=force_download)
        barsize_start_dates = {
            '1 week': "200 daysAgo",
            '1 day': "52 daysAgo",
            '4 hours': "6 weeksAgo",
            '1 hour': "3 weeksAgo",
            '5 mins': "3 daysAgo"
        }
        for barsize in self.intradaySizes:
            sd = barsize_start_dates[barsize]
            self.setup_frame(barsize, dataType, start_date=sd, end_date=end_date, force_download=force_download)

    def setup_TA_intraday(self, lookBack, atrSpan, sigRow=3, validationRow=4):

        # ----- S E T U P -----
        for barsize in self.intradaySizes:
            f = self.frames[barsize]
            print(f'Running {self.ls} on {barsize}')
            ps.TA_TA(f, lookBack, atrSpan, pointsSpan=10)
            ps.SCORE_TA_Volume(f, lookBack, volMA=10, TArow=sigRow, scoreRow=validationRow)
            ps.TA_Levels(f)

            
        # ----- I M P O R T -----
        self.import_all_HTF_data()


        # # ----- R U N -----
        for barsize in self.tradeSizes:
            f = self.frames[barsize]

            # -- Validations --  
            if self.ls == 'LONG':
                # setup required scores and valicadations for the strategy
                self.score_cols.time = ps.SCORE_VALID_time_of_day(f, self.ls, lookBack, sigRow, validationRow) #! > 9:35 will not include 9:35
                self.score_cols.level_premkt_gt = ps.SCORE_VALID_Levels_premkt_and_5minbar(f, self.ls, lookBack, sigRow, validationRow)
                self.score_cols.touches = ps.SCORE_TA_touches(f, self.ls, lookBack, atrSpan, direction='down', toTouchAtrScale=10, pastTouchAtrScale=2, TArow=3, scoreRow=4) #! > 50 is good touch
                self.score_cols.reset_if_breaks = ps.SCORE_VALID_reset_if_breaks(f, self.ls, lookBack, sigRow, validationRow) #! > 1 is bad
                self.score_cols.pullback = ps.SCORE_TA_Pullback(f, self.ls, lookBack, atrSpan, minPbLen=4, TArow=3, scoreRow=4) #! added persisnace to the pullback signals 
                self.score_cols.retest = ps.SCORE_TA_retest(f, self.ls, lookBack, atrSpan, TArow=3, scoreRow=4) #! maybe needs some adjusting 
                self.score_cols.bsw  = ps.SCORE_TA_Bar_StrengthWeakness(f, self.ls, lookBack, atrSpan, TArow=3, scoreRow=4) #! works but not sure what this is telling me
                self.score_cols.buysetup = ps.SCORE_VALID_BuySetup(f, self.ls, self.score_cols.bsw, self.score_cols.retest, lookBack, TArow=3, scoreRow=4) #! works 
                self.score_cols.rtm = ps.SCORE_TA_RTM(f, self.ls, lookBack, atrSpan, TArow=3, scoreRow=4)
                self.score_cols.buy = ps.SCORE_VALID_buy(f, self.ls, lookBack, sigRow, validationRow)


                # resets if new HP is formed
                strat = sig.Strategy('PB', lookBack=lookBack)

                """
                -- premarket higher volume than average
                -- time > 9:35 (wait for first 5 mins to play out)
                -- breaks premkt high
                -- price moves above first 5 min bar high
                -- price pulls back 
                    a. to max 50% of the first 5 min bar
                    b. 2 or more lower highs (LH)
                    c. sequential pullback with less than 50% overlap on any bar
                    d. touches a support level (prev day high, this day low, daily Res 1 lower )
                -- bullish bar completes (BSW ..  bot tail or CoC)
                -- buy signal is confirmed (RTM, RS, break prev bar high)
                """

                # validates various metris. each step must be fully validated before moving to the next step


                # PreMkt - get validated once for the day and then used for all steps
                # volume is already run in the daily setup
                strat.pass_if(step=1, scoreCol=self.score_cols.time,   operator='>', threshold=1)
                
                # Step 1) - Moves up
                strat.pass_if(step=2, scoreCol=self.score_cols.level_premkt_gt, operator='>', threshold=1)

                # Step 2) - Pulls back and touches a level
                strat.pass_if(step=3, scoreCol=self.score_cols.touches, operator='>', threshold=1)

                # Step 3) - Pullback Quality
                strat.pass_if(step=4, scoreCol=self.score_cols.pullback, operator='>', threshold=1)

                # Step 4) - Buy Signals
                strat.pass_if(step=5, scoreCol=self.score_cols.buysetup, operator='>', threshold=1)
                strat.pass_if(step=5, scoreCol=self.score_cols.buy,      operator='>', threshold=1)
                strat.pass_if(step=5, scoreCol=self.score_cols.rtm,      operator='>', threshold=1)


                # resets all steps if any of the following events are true. can be applied to a step or all steps
                """
                -- Buysetup fails
                -- price breaks below the first 5 min bar low
                -- price breaks below days lows 
                """
                strat.reset_if(step=2, scoreCol='L_BuySetup_isFail',  operator='>', threshold=1, startFromStep=2)
                strat.reset_if(step=5, scoreCol=self.score_cols.reset_if_breaks, operator='>', threshold=1, startFromStep=2)

                """
                Retruns:
                -- current step: the step that is being evaluated
                -- steps passed: the number of steps that have been passed
                -- conditions met: the number of conditions that have been met
                -- action: 'BUY' or 'SELL'
                """

                f.add_multi_ta(strat, [
                    ChartArgs({'dash': 'solid', 'color': 'cyan', 'width':5},     chartType='lines+markers', row=5, columns=[strat.name_pct_complete])
                ],
                runOnLoad=False)
        
        self.stats.status = SignalStatus.PRE_MARKET
        self.stats.status_why = f"Intraday TA is set up {self.intradaySizes}"

    # ------- Imports -------------------

    def import_market_data(self, df:pd.DataFrame, timeframe:str, prefix:str='imported_'):
        self.frames[timeframe].import_data(df, importCols=['open', 'high', 'low', 'close', 'volume'], prefix=prefix)

    def import_all_market_data(self, mktStockX:object):
        """takes a different StockX object such as SPY and imports all its data (OHLCV) into this StockX object.
        Time frames are match from object to object and the data is imported with a prefix of the symbol of the imported StockX object.

        Args:
            mktStockX (StockX): StockX object must match the same timeframes as this object.
        """
        # check if timeframes match
        if self.frames.keys() != mktStockX.frames.keys():
            raise ValueError("StockX::Timeframes do not match. When importing market data the imported StockX must have the same barsizes as this StockX object.")

        for barsize in self.frames.keys():
            mktDF = mktStockX.frames[barsize].data
            self.import_market_data(mktDF, barsize, f"{mktStockX.symbol}_")

    def import_HTF_data(self, fromBarsize:str, toBarsize:str, fromCols:List[str], prefix:str='imported_'):
        """Import the 1 Hour data to the 5 min data etc """
        prefix  = fromBarsize.replace(' ', '_') + '_'
        self.frames[toBarsize].import_data(self.frames[fromBarsize].data, importCols=fromCols, prefix=prefix)

    def import_all_HTF_data(self):
        # only imports if the HTF is availble in the frames. so most of the time it will only import 1 day, 1 hour to 5 mins
        import_map = {
                '1 min'  : ['1 day', '4 hours', '1 hour', '15 mins', '5 mins'], 
                '2 mins' : ['1 day', '4 hours', '1 hour', '15 mins', '5 mins'],
                '3 mins' : ['1 day', '4 hours', '1 hour', '15 mins'],
                '5 mins' : ['1 day', '4 hours', '1 hour', '15 mins'],
                '15 mins': ['1 day', '4 hours', '1 hour'],
                '1 hour' : ['1 day', '4 hours'],
            }
        
        column_map = {
            '1 day'  : ['Res_1_Lower', 'Sup_1_Upper', 'MA_cl_50', 'MA_cl_200'],
            '4 hours': ['Res_1_Lower', 'Sup_1_Upper',],
            '1 hour' : ['Res_1_Lower', 'Sup_1_Upper',],
            '15 mins': ['Res_1_Lower', 'Sup_1_Upper',],
            '5 mins' : ['Res_1_Lower', 'Sup_1_Upper',],
        }
        
        for barsize in self.intradaySizes:
            f = self.get_frame(barsize)
            for fromBarsize in import_map[barsize]:
                if self.get_frame(fromBarsize):
                    from_f = self.get_frame(fromBarsize)
                    # print(f'Importing {fromBarsize} to {barsize}')
                    importCols = column_map[fromBarsize]
                    f.import_data(from_f.data, importCols=importCols, prefix=fromBarsize+'_')
    
    #! -------- Run --------------------- Work in progress

    def RUN_DAILY(self, ls:str='', spy:object=None, isMarket:bool=False, displayCharts:bool=False, printStats:bool=False, forceDownload:bool=False):
        print(f"------------ StockX::RUN_DAILY: {self.symbol} {self.ls}------------------------------------------------------------------------")

        self.set_ls(ls)
        if isMarket:
            self.setup_frame('1 day', 'ohlcv', duration="200 D", endDateTime='now', isDayFrame=True,      taPresets=TAPresets1D(ls=self.ls, isSpy=True))
            self.setup_frame('1 hour', 'ohlcv', duration="15 D", endDateTime='now', isIntradayFrame=True, taPresets=TAPresets1H(ls=self.ls, isSpy=True))
            self.frames['1 day'].run_ta()
            self.frames['1 hour'].run_ta()

        else:
            self.setup_frame('1 day', 'ohlcv', duration="200 D", endDateTime='now', isDayFrame=True,      taPresets=TAPresets1D(ls=self.ls, lookBack=100))
            self.setup_frame('1 hour', 'ohlcv', duration="15 D", endDateTime='now', isIntradayFrame=True, taPresets=TAPresets1H(ls=self.ls, lookBack=100, volChgPctThreshold=50))
            self.import_all_market_data(spy)
            self.frames['1 day'].run_ta()
            self.frames['1 hour'].run_ta()
            

    def RUN_DAILY_old(self, spy:object=None, isMarket:bool=False, displayCharts:bool=False, printStats:bool=False, forceDownload:bool=False):
        print(f"------------ StockX::RUN_DAILY: {self.symbol} {self.ls}------------------------------------------------------------------------")
        self.setup_frame('1 day', 'ohlcv', duration="200 D", endDateTime='now', force_download=False, isDayFrame=True)
        self.setup_frame('1 hour', 'ohlcv', duration="15 D", endDateTime='now', force_download=False, isIntradayFrame=True)
        if isMarket:
            return

        f_D1 = self.frames['1 day']
        f_H1 = self.frames['1 hour']
        self.import_all_market_data(spy)

        # args
        lookBack = 30
        atrSpan = 14
        sigRow = 3
        validationRow = 4
        
        # basic TA
        ps.TA_TA(f_D1, lookBack, atrSpan, pointsSpan=10, isDaily=True)
        ps.TA_TA(f_H1, lookBack, atrSpan, pointsSpan=10, isDaily=False)
        ps.TA_Levels(f_H1) # not require for pre market but saves running in intraday 
        self.import_all_HTF_data()

        # Scores
        pointCol = 'HP_hi_10' if self.ls == 'LONG' else 'LP_lo_10'
        self.score_cols.vol_TODC = ps.TA_Daily_Volume_Change(f_H1, lookBackDays=10) # lookcBackDays specific to VolumeTimeOfDayChangePct. (not the same as lookBack)
        self.score_cols.D1       = ps.TA_Daily(f_D1, self.ls, pointCol=pointCol, atrSpan=atrSpan, lookBack=lookBack, TArow=3, scoreRow=4)
        self.score_cols.gaps     = ps.SCORE_Gaps(f_D1, ls=self.ls, pointCol=pointCol, atrSpan=atrSpan, lookBack=lookBack, TArow=3, scoreRow=4)
        self.score_cols.rtm      = ps.SCORE_TA_RTM_DAILY(f_D1, ls=self.ls, atrSpan=atrSpan, lookBack=lookBack, TArow=3, scoreRow=4)

        self.stats.status = SignalStatus.DAILY_TA
        self.stats.status_why = "Daily TA is set up"

        if displayCharts:
            self.spy.frames['1 day'].plot()  
            print(f'Score: 1 day             --  {self.stats.score_1D}')   
            print(f'Score: Pre Market Volume --  {self.stats.score_preMkt_vol}') 
            print(f'Score: preMkt Average    --  {self.stats.score_AVG}')
            self.frames['1 day'].plot()
            self.frames['1 hour'].plot()

    def set_daily_stats(self, displayCharts:bool=False, printStats:bool=False, forceDownload:bool=False, incFundamentals:bool=True):
        # setup the daily stats
        self.stats_daily = StockStatsDaily(self.symbol, self.ls)  # auto sets the time to now
        self.stats_daily.price = self.get_frame_data('1 day').iloc[-1]['close']

        # get the latest fundamentals
        if incFundamentals:
            self.req_fundamentals(max_days_old=10)
        if self.fundamentals.fundamentals is None:
            print(f"StockX {self.symbol}::set_daily_stats: Fundamentals are not found.")
        else:
            self.stats_daily.marketCap = self.fundamentals.fundamentals.market_cap
            self.stats_daily.currentPrice = self.fundamentals.fundamentals.current_price
            self.stats_daily.peRatio = self.fundamentals.fundamentals.pe_ratio
            self.stats_daily.vol10dayAvg = self.fundamentals.fundamentals.volume_10day_avg
   
        # set the stats
        self.stats_daily.scoreVol = round(self.get_frame_data('1 hour').iloc[-1][self.score_cols.vol_TODC], 2)
        self.stats_daily.scoreGaps = round(self.get_frame_data('1 day').iloc[-1][self.score_cols.gaps], 2)
        self.stats_daily.scoreRTM = round(self.get_frame_data('1 day').iloc[-1][self.score_cols.rtm], 2)
        self.stats_daily.score_1D = round(sum([self.stats_daily.scoreVol, self.stats_daily.scoreGaps, self.stats_daily.scoreRTM]) / 3, 2)
        self.stats_daily.validAv_1D = round(self.get_frame_data('1 day').iloc[-1][self.score_cols.D1], 2)

        prev_day = self.get_frame_data('1 day').iloc[-2]
        today = self.get_frame_data('1 day').iloc[-1]
        close_tday = self.get_frame_data('1 hour').iloc[-1]['close']
        close_yday = prev_day['close']
        ma50_yday = prev_day['MA_cl_50']
        ma200_yday = prev_day['MA_cl_200']
        ma50_today = today['MA_cl_50']
        ma200_today = today['MA_cl_200']

        self.stats_daily.breaksAbove50MA = close_yday < ma50_yday and close_tday > ma50_today
        self.stats_daily.breaksAbove200MA = close_yday < ma200_yday and close_tday > ma200_today
        self.stats_daily.breaksBelow50MA =  close_yday > ma50_yday and close_tday < ma50_today
        self.stats_daily.breaksBelow200MA = close_yday > ma200_yday and close_tday < ma200_today

        self.stats_daily.priceChgPct = round((close_tday - close_yday) / close_yday, 2) * 100


        if printStats:
            self.stats = self.get_stats()
            display(pd.DataFrame(self.stats_daily.__dict__, index=[0]))


    def RUN_FUNDAMENTALS(self, maxDaysOld:int=10, allowedETFs:List[str]=['XLY', 'XLK', 'XLC']):
        # runs on load 
        self.req_fundamentals(max_days_old=maxDaysOld)
        self.sector_ETF_is_allowed(allowedETFs)
        
    def RUN_SETUP(self, spy:object, dataType:str='ohlcv', endDate:str='now', forceDownload:bool=False):
        self.spy = spy
        self.setup_all_frames(dataType, endDate, forceDownload)
        self.spy.setup_all_frames(dataType, endDate, forceDownload)
        self.import_all_market_data(spy)

    def RUN_PRE_MARKET(self, ls:str='LONG', timeToRun:str='9:35', minPreMarketScore:float=0.5, displayCharts:bool=False, printStats:bool=False):
        self.set_ls(ls)
        self.setup_TA_PreMarket(lookBack=100, atrSpan=14)
        self.setup_TA_PreMarket(lookBack=100, atrSpan=14, isSpy=True)

        self.stats.score_preMkt_vol = self.get_frame_data('1 hour').iloc[-1][self.score_cols.preMkt_vol]
        self.stats.score_1D     = self.get_frame_data('1 day').iloc[-1][self.score_cols.D1]
        self.stats.update_score_av([self.stats.score_1D, self.stats.score_preMkt_vol])


        if printStats:
            self.stats = self.get_stats()
            display(pd.DataFrame(self.stats.__dict__, index=[0]))

        if displayCharts:
            self.spy.frames['1 day'].plot()  #!to Fix -- spy not plotting TA
            print(f'Score: 1 day             --  {self.stats.score_1D}')   
            print(f'Score: Pre Market Volume --  {self.stats.score_preMkt_vol}') 
            print(f'Score: preMkt Average    --  {self.stats.score_AVG}')
            self.frames['1 day'].plot()
            self.frames['1 hour'].plot()

    def RUN_INTRADAY(self, updateEverySeconds:int=1, myTradingTimes:List[Tuple[str, str]]=None, maxTrades:int=3, displayCharts:bool=False, logTrades:bool=False):
        """
        1. Runs all frames to look for a buy signal
        2. If a buy signal is found it will place the order
        3. follow up with the trade
        4. log the trade
        5. start again but check trade limits"""
        self.setup_TA_intraday(lookBack=100, atrSpan=14)
        if displayCharts:
            print(f'Score: 1 day             --  {self.stats.score_1D}')   
            print(f'Score: Pre Market Volume --  {self.stats.score_preMkt_vol}') 
            print(f'Score: preMkt Average    --  {self.stats.score_AVG}')
            print('----------------------------------------------')
            print(f'Score: 5 mins  --  {self.stats.score_5M}')
            self.frames['5 mins'].plot()
        
        pass

    #! ------- Update ------------------- Work in progress

    def reqLiveBars(self):
        # request the tick data from the data source
        self.livedata.setup_tickers([self.symbol])
        self.livedata.reqLiveBars(show=False)
        self.ib.sleep(3)

    def updateLiveOHLCV(self, timeframes:List[str]=['5 mins'], spy:object=None):  
        # add the ohlcv to the data frame 
        for barsize in timeframes:
            ohlcv  = self.livedata.get_live_bar_df(ticker=self.symbol, format=True, bsize=barsize)
            display(ohlcv)
            self.frames[barsize].load_ohlcv(ohlcv)
            self.import_all_market_data(spy)
            self.frames[barsize].update_ta_data()
            return self.frames[barsize].data
    
    

    #! ------- Trading ------------------- Work in progress

    def setup_trader(self):
        #! below is copied from TEST_order.ipynb

        self.trader.add_entry(entryx=EntryX(orderType='STP', longPriceCol='high', shortPriceCol='low', barsAgo=1, offsetPct=0.00, limitOffsetPct=0.001))
        # self.trader.add_entry(entryx=EntryX(orderType='MKT'))

        trailInitType = 'rrr'
        self.trader.add_stop_and_target(qtyPct=25,
                                targetx=TargetX(longPriceCol='Res_1', shortPriceCol='Sup_1', offsetVal=0.50, barsAgo=1, rrIfNoTarget=2),
                                initStop=StopX(longPriceCol='LoIst_lo_3', shortPriceCol='HiIst_hi_3', offsetVal=0.50, barsAgo=1),
                                trailingStopPrices= [
                                        TrailX(initType=trailInitType, initTrigVal=1, barsAgo=2, longPriceCol='FFILL_LP_lo_3', shortPriceCol='FFILL_HP_hi_3', offsetVal=0.01),
                                        TrailX(initType=trailInitType, initTrigVal=2, barsAgo=2, longPriceCol='MA_cl_50',      shortPriceCol='MA_cl_50',      offsetVal=0.01),
                                        TrailX(initType=trailInitType, initTrigVal=3, barsAgo=2, longPriceCol='MA_cl_21',      shortPriceCol='MA_cl_21',      offsetVal=0.01)
                            ])

        self.trader.add_stop(qtyPct=75, 
                            initStop=StopX(name='Stop1', longPriceCol='LoIst_lo_3', shortPriceCol='HiIst_hi_3', offsetVal=0.50, barsAgo=1),
                            trailingStopPrices= [
                                TrailX(name='Stop1', initType=trailInitType, initTrigVal=1, barsAgo=2, longPriceCol='FFILL_LP_lo_3', shortPriceCol='FFILL_HP_hi_3', offsetVal=0.01),
                                TrailX(name='Stop2', initType=trailInitType, initTrigVal=2, barsAgo=2, longPriceCol='MA_cl_50',      shortPriceCol='MA_cl_50',      offsetVal=0.01),
                                TrailX(name='Stop3', initType=trailInitType, initTrigVal=3, barsAgo=2, longPriceCol='MA_cl_21',      shortPriceCol='MA_cl_21',      offsetVal=0.01)
                            ])

    def START_TRADE(self, tf:str):
        f = self.frames[tf]
        self.trader.set_orders(f.data, self.riskAmount, self.outsideRth)
        self.trader.place_orders(delay_between_orders=1)

    # ------- Backtest -------------------
            
    def run_backtest(self, start: str | int, end: str | int, htf_imports: Dict[str, list] = None, save_snapshots: bool = False):
        """
        Run backtest with higher timeframe data importing.
        
        Args:
            start: Start datetime or index
            end: End datetime or index
            htf_imports: Dict mapping timeframe to columns to import:
                {'4H': ['close', 'volume']}
            save_snapshots: Whether to save snapshots
        """
        print(f"\nInitializing backtest with parameters:")
        print(f"Start: {start}, End: {end}")
        
        smallest_tf_frame = None
        smallest_frequency = pd.Timedelta.max
        
        for frame_name, frame in self.frames.items():
            if not frame.data.empty:
                frequencies = pd.Series(frame.data.index[1:] - frame.data.index[:-1]).mode()
                if not frequencies.empty:
                    current_frequency = frequencies.iloc[0]
                    if current_frequency < smallest_frequency:
                        smallest_frequency = current_frequency
                        smallest_tf_frame = frame
                        smallest_tf_name = frame_name
        
        if smallest_tf_frame is None:
            raise ValueError("No valid timeframes found")
        
        # Initialize backtests
        smallest_tf_frame.backtest_setup(start, end, save_snapshots)
        start_time = smallest_tf_frame.data.index[smallest_tf_frame._backtest_start_idx]
        end_time = smallest_tf_frame.data.index[smallest_tf_frame._backtest_end_idx]
        
        for frame_name, frame in self.frames.items():
            if frame_name != smallest_tf_name:
                start_idx = frame.data.index.get_indexer([start_time])[0]
                end_idx = frame.data.index.get_indexer([end_time])[0]
                frame.backtest_setup(start_idx, end_idx, save_snapshots)
        
        running = True
        while running:
            # Import HTF data before running next row
            if htf_imports:
                for htf_name, columns in htf_imports.items():
                    if htf_name in self.frames:
                        htf_frame = self.frames[htf_name]
                        smallest_tf_frame.import_data(
                            htf_frame.backtest_data,
                            columns,
                            merge_to_backtest=True
                        )
            
            # Update smallest timeframe with imported data
            if not smallest_tf_frame.backtest_next_row():
                running = False
                continue
                
            current_time = smallest_tf_frame.backtest_data.index[-1]
            
            # Update other timeframes when needed
            for frame_name, frame in self.frames.items():
                if frame_name == smallest_tf_name:
                    continue
                    
                if current_time > frame.backtest_data.index[-1]:
                    # Import HTF data before running next row for intermediate timeframes
                    if htf_imports:
                        for htf_name, columns in htf_imports.items():
                            if htf_name in self.frames:
                                htf_frame = self.frames[htf_name]
                                frame.import_data(
                                    htf_frame.backtest_data,
                                    columns,
                                    merge_to_backtest=True
                                )
                                
                    frame.backtest_next_row()

    # ------- Helpers -------------------

    def display_frame_data(self, timeframe:str, column_contains:str=None):
        if timeframe not in self.frames:
            print(f"Timeframe '{timeframe}' not found in StockX frames")
            return
        cols = self.frames[timeframe].data.columns if not column_contains else [col for col in self.frames[timeframe].data.columns if column_contains in col]
        display(self.frames[timeframe].data[cols])
    
    def display_columns(self, timeframe:str, contains:str=None):
        cols = self.frames[timeframe].data.columns
        if contains:
            cols = [col for col in cols if contains in col]
        display(cols)
        
    def display_stats(self):
        display(self.stats)    
