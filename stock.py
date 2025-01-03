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

from data import historical_data as hd
from frame.frame import Frame
from strategies import ta
from industry_classifications.sector import get_etf_from_sector_code
import emails.email_client as email_client
from project_paths import get_project_path



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

def analyze_sector(self,
    etf_symbol: str,
    lookback_period: str = "52 weeksAgo",
    mansfield_period: int = 200,
    roc_period: int = 200,
    ma_short: int = 50,
    ma_long: int = 200,
    market_symbol: str = "SPY",
    return_full_df: bool = False,
    verbose: bool = False
) -> Union[Tuple[float, float, float], pd.DataFrame]:
    """
    Analyze a sector ETF against the market using Mansfield RSI and ROC ratio.
    
    Parameters:
    -----------
    etf_symbol : str
        Symbol of the sector ETF to analyze (e.g., 'XLF', 'XLE', etc.)
    lookback_period : str, default "52 weeksAgo"
        Historical data lookback period
    mansfield_period : int, default 200
        Period for Mansfield RSI calculation
    roc_period : int, default 200
        Period for Rate of Change ratio calculation
    ma_short : int, default 50
        Short-term moving average period
    ma_long : int, default 200
        Long-term moving average period
    market_symbol : str, default "SPY"
        Symbol to use as market benchmark
    return_full_df : bool, default False
        If True, returns full DataFrame; if False, returns latest values only
    verbose : bool, default False
        If True, prints analysis summary
        
    Returns:
    --------
    Union[Tuple[float, float, float], pd.DataFrame]
        If return_full_df=False:
            Returns (mansfield_rsi, ma_roc_ratio, combined_score)
        If return_full_df=True:
            Returns complete DataFrame with all calculations
    """
    # Get historical data
    etf_data = hd.get_hist_data(etf_symbol, lookback_period, 'now', '1 day')
    market_data = hd.get_hist_data(market_symbol, lookback_period, 'now', '1 day')
    
    # Initialize and run analysis
    analysis = compare.SectorAnalysis(etf_data, market_data)
    analysis.compute_all(
        mansfield_period=mansfield_period,
        roc_period=roc_period,
        ma_short=ma_short,
        ma_long=ma_long
    )
    
    # Get results
    result_df = analysis.get_df()
    today_metrics = analysis.get_today(verbose=verbose)
    
    if return_full_df:
        return result_df
    else:
        return (
            today_metrics['mansfield_rsi'],
            today_metrics['ma_roc_ratio'],
            today_metrics['combined_score']
        )

# ----------------------------------------------------------
# ------- S T O C K X  -------------------------------------
# ----------------------------------------------------------
from frame.frame import Frame
from dataclasses import dataclass, field
from data import historical_data as hd
from ib_insync import *
from typing import Dict, Any, List
import pandas as pd
from data.random_data import RandomOHLCV
import strategies.ta as ta
import strategies.signals as sig
import stock_fundamentals
import strategies.preset_strats as ps

@dataclass
class TAData:
    ta: ta.TA
    style: Dict[str, Any] | List[Dict[str, Any]] = field(default_factory=dict)
    chart_type: str = "line"
    row: int = 1

@dataclass
class StatusLog:
    item: str
    dataType: str
    dataRecieved: bool = False
    dataValidated: bool = False
    dataRows: int = 0
    status: str = 'Not Started'
    score: float = 0.0

@dataclass
class StockX:
    ib: IB = None
    symbol: str = ''

    def __post_init__(self):
        self.fundamentals = stock_fundamentals.Fundamentals(self.ib, self.symbol)
        self.frames = {}
        self.status = [] # list of StatusLog

    def get_status_df(self):
        return pd.DataFrame([s.__dict__ for s in self.status])
        
    def set_up_frame(self, timeframe, dataType:str='random', start_date:str="52 weeksAgo", end_date:str='now'):
        name = f"{timeframe}_{dataType[:3]}" if dataType in ['primary_etf', 'secondary_etf', 'mkt'] else timeframe
        for status in self.status:
            if status.item == timeframe and status.dataType == dataType:
                return
            
        
        self.frames[name] = Frame(self.symbol, run_ta_on_load=True, rowHeights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
        
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
             self.frames[name].load_ohlcv(hd.get_hist_data(self.symbol, start_date, end_date, timeframe))

        elif dataType == 'tick':
            # todo: implement tick data
            pass

        elif dataType == 'mkt':
            self.frames[name].load_ohlcv(hd.get_hist_data('SPY', start_date, end_date, timeframe))

        elif dataType == 'primary_etf':
            eft_symbol = self.fundamentals.fundamentals.primary_etf
            if eft_symbol is not None:
                self.frames[name].load_ohlcv(hd.get_hist_data(eft_symbol[0], start_date, end_date, timeframe))
            else:
                print(f"Primary ETF not found for {self.symbol}")

        elif dataType == 'secondary_etf':
            eft_symbol = self.fundamentals.fundamentals.secondary_etf
            if eft_symbol is not None:
                self.frames[name].load_ohlcv(hd.get_hist_data(eft_symbol[0], start_date, end_date, timeframe))
            else:
                print(f"Secondary ETF not found for {self.symbol}")

        data_validated = not self.frames[name].data.empty
        len_df = len(self.frames[name].data)
        self.status += [StatusLog(timeframe, dataType, True, data_validated, len_df, 'Setup', 0.0)]

    def req_fundamentals(self, max_days_old=0, allowedETFs: List[str] = []):
        self.fundamentals.req_fundamentals(max_days_old)
        etf_is_allowed = self.fundamentals.validate_fundamental('primary_etf', 'isin', allowedETFs, description='Stocks primary sector ETF is allowed')
        self.status += [StatusLog('Fundamentals', 'Fundamentals', True, etf_is_allowed, 1, 'Complete', self.fundamentals.validation_fundamentals_has_passed())]

    def sector_ETF_is_allowed(self):
        return self.status[0].dataValidated
    
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
    
    def run_daily_frame(self, lookback:int=1):
        """setup_framees must be run first. Ech Frame is set up with the arg run_ta_on_load=True. 
        This means every time a frame is loaded with data, the ta is run on the data automatically.
        Therefore the items below are automatically run on the data."""
        f_day = self.frames['1 day']
        mktDF = self.frames['1 day_mkt'].data
        etfDF = self.frames['1 day_pri'].data
        # preset ta signals and scorers
        ps.import_to_daily_df(f_day, mktDF, etfDF, RSIRow=3) #! test mansfield on real ccharts and compare to trading view
        ps.require_ta_for_all(f_day)
        ps.ma_ta(f_day, [50, 150, 200])
        ps.volume_ta(f_day, ls='LONG', ma=10, scoreRow=4, lookBack=lookback)
        ps.consolidation_ta(f_day, atrSpan=50, maSpan=50, lookBack=lookback, scoreRow=4)
        ps.STRATEGY_daily_consolidation_bo(f_day, lookBack=lookback, scoreRow=5) # only works if consolidation has been run
        ps.STRATEGY_pullback_to_cons(f_day, ls='LONG', lookBack=lookback, scoreRow=5) # only works if consolidation has been run
 
        score = f_day.data['PBX_ALL_Scores'].iat[-1]
        self.status += [StatusLog('1 day', 'ohlcv', True, True, len(f_day.data), 'Complete', score)]

    def run_intraday_frames(self, lookback:int=1, plot:bool=False):
        frames_run = []
        if '1 hour' in self.frames:
            f_1hr = self.frames['1 hour']
            ps.require_ta_for_all(f_1hr)
            ps.ma_ta(f_1hr, [50, 150, 200])
            frames_run.append('1 hour')

        if '5 mins' in self.frames:
            f_5min = self.frames['5 mins']
            ps.require_ta_for_all(f_5min)
            ps.ma_ta(f_5min, [8, 21, 50])
            frames_run.append('5 mins')

        if '2 mins' in self.frames:
            f_5min = self.frames['2 mins']
            ps.require_ta_for_all(f_5min)
            ps.ma_ta(f_5min, [8, 21])
            frames_run.append('2 mins')
        
        print(f"Ran intraday frames: {frames_run}")

        if plot:
            for frame in frames_run:
                print(f"Plotting {frame}")
                self.frames[frame].plot()
      


    def req_tick_data(self, timeframe):
        # request the tick data from the data source
        pass

    def add_ohlcv(self, timeframe, ohlcv):
        # add the ohlcv to the data frame 
        # can be used to add market data or other data not just the open high low close
        pass

    def add_rows(self, timeframe, rows):
        # add the rows to the data frame.  eg if new data such as market data is added
        pass


            
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


