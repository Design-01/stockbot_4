from dataclasses import dataclass
from typing import List, Dict, Union, Tuple, Optional
from datetime import datetime
import pandas as pd
from ib_insync import IB, Stock
import xml.etree.ElementTree as ET
import pickle
import os

from data import historical_data as hd
from frame.frame import Frame
from strategies import ta
from industry_classifications.sector import get_etf_from_sector_code


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
    
    Args:
        symbol: Stock symbol
        max_days_old: Maximum age of data in days. Use 0 to ignore age check.
    
    Returns:
        StockFundamentals if valid data exists, None otherwise
    """
    filename = f"data//fundamental_data_store//{symbol.upper()}_fundamentals.pkl"
    
    # Check if file exists
    if not os.path.exists(filename):
        return None
    
    try:
        # Load the data
        with open(filename, 'rb') as f:
            data: StockFundamentals = pickle.load(f)
        
            
        # Convert stored string date to datetime for comparison
        stored_date = datetime.strptime(data.request_date, '%Y-%m-%d %H:%M:%S')
        days_old = (datetime.now() - stored_date).days
        if days_old > max_days_old:
            return None
            
        return data
        
    except (pickle.UnpicklingError, EOFError, AttributeError):
        # Handle corrupted file or incompatible data structure
        return None

def get_stock_fundamentals(ib, ticker: str, current_volume: float = 0, max_days_old=0) -> StockFundamentals:
    """Retrieve comprehensive stock information including all available ratios."""
    contract = Stock(ticker, 'SMART', 'USD')
    details = ib.reqContractDetails(contract)
    if not details:
        raise ValueError(f"No contract details found for {ticker}")
    
    # try from file first if within max_days_old
    if max_days_old != 0:
        file_data = get_fundamentals_from_file(ticker, max_days_old)
        if file_data:
            print(f"Using cached data for {ticker} (age: {file_data.request_date})")
            return file_data
    
    fundamental_data = ib.reqFundamentalData(contract, 'ReportSnapshot')
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
    filename = f"data//fundamental_data_store//{symbol.upper()}_fundamentals.pkl"
    
    # Save data using pickle
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


class StockXDaily:
    def __init__(self, ib:IB, symbol):
        self.ib = ib
        self.symbol = symbol
        self.fundamentals: StockFundamentals = None
        self.fundamentals_validation_results = {}
        self.frame = Frame(self.symbol)
        self.frame_validation_results = {}
        self.dayscores = None

    def req_fundamentals(self, max_days_old=0):
        if not self.fundamentals:
            self.fundamentals = get_stock_fundamentals(self.ib, self.symbol, max_days_old=max_days_old)
        return self.fundamentals
    
    def validate_fundamental(self, key: str, validation_type: str, value: Union[float, int, str, List, Tuple]) -> bool:
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
                if key in ['primary_etf', 'secondary_etf']:
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
            'validation_type': validation_type,
            'comparison_value': value,
            'actual_value': fundamental_value,
            'passed': result
        }
        
        return result
    
    def validation_fundamentals_report(self, asDF: bool = True) -> Union[pd.DataFrame, Dict]:
        """
        Returns a report of all validations performed.
        
        Args:
            asDF (bool): If True, returns pandas DataFrame, otherwise returns dictionary
            
        Returns:
            Union[pd.DataFrame, Dict]: Validation results
        """
        if asDF:
            return pd.DataFrame.from_dict(self.fundamentals_validation_results, orient='index')
        return self.fundamentals_validation_results
    
    def validation_TA_report(self, asDF: bool = True) -> Union[pd.DataFrame, Dict]:
        """
        Returns a report of all validations performed.
        
        Args:
            asDF (bool): If True, returns pandas DataFrame, otherwise returns dictionary
            
        Returns:
            Union[pd.DataFrame, Dict]: Validation results
        """
        if asDF:
            return pd.DataFrame.from_dict(self.frame_validation_results, orient='index')
        return self.frame_validation_results
    
    def validation_fundamentals_has_passed(self, maxFails: int = 0) -> bool:
        """
        Checks if validations have passed within the maximum failure threshold.
        
        Args:
            maxFails (int): Maximum number of allowed failures
            
        Returns:
            bool: True if number of failures is <= maxFails
        """
        if not self.fundamentals_validation_results:
            return False
            
        failures = sum(1 for result in self.fundamentals_validation_results.values() if not result['passed'])
        return failures <= maxFails

    def validation_TA_has_passed(self, maxFails: int = 0) -> bool:
        """
        Checks if validations have passed within the maximum failure threshold.
        
        Args:
            maxFails (int): Maximum number of allowed failures
            
        Returns:
            bool: True if number of failures is <= maxFails
        """
        if not self.frame_validation_results:
            return False
            
        failures = sum(1 for result in self.frame_validation_results.values() if not result['passed'])
        return failures <= maxFails


    def req_ohlcv(self, start_date:str="52 weeksAgo", end_date:str='now'):
        if self.frame.data.empty:
             self.frame.load_ohlcv(hd.get_hist_data(self.symbol, start_date, end_date, '1 day'))
        return self.frame.data
    
    def validate_TA(self, ta_indicator, ta_validator, style=None):
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
        default_style = {
            'dash': 'solid',
            'color': 'cyan',
            'width': 1
        }
        
        # Use provided style or default
        plot_style = style if style is not None else default_style
        
        # Add the technical indicator to the frame
        self.frame.add_ta(ta_indicator, plot_style)
        
        # Load and update the data
        self.frame.update_ta_data()
        
        # Add the validator
        self.frame.add_ta(ta_validator)
        
        # Update data again to include validator results
        self.frame.load_ohlcv(self.frame.data)
        self.frame.update_ta_data()
        
        # Create a unique identifier for this validation
        validation_id = f"{ta_indicator.name}_{ta_validator.name}"
        
        # Initialize technical validation results dictionary if it doesn't exist
        if not hasattr(self, 'technical_validation_results'):
            self.technical_validation_results = {}
        
        # Store the validation result
        self.frame_validation_results[validation_id] = {
            'indicator': ta_indicator.name,
            'validator': ta_validator.name,
            'validation_type': type(ta_validator).__name__,  # e.g., 'Breaks' or 'AboveBelow'
            'indicator_value': self.frame.data[ta_indicator.name].iloc[-1],
            'passed': self.frame.data[ta_validator.name].iloc[-1],
            'timestamp': self.frame.data.index[-1]  # Store the timestamp of the validation
        }
        
        return self.frame_validation_results[validation_id]
    
    def set_default_daily_ta(self):
        # TA may add many several columns of data to the dataframe 
        self.frame.add_ta(ta.MA('close', 200), {'dash': 'solid', 'color': 'cyan', 'width': 2})
        self.frame.add_ta(ta.MA('close', 50), {'dash': 'solid', 'color': 'purple', 'width': 2})
        self.frame.add_ta(ta.MA('volume', 10), {'dash': 'solid', 'color': 'cyan', 'width': 1}, row=2)
        self.frame.add_ta(ta.ACC('close', 5, 10, 10), {'dash': 'solid', 'color': 'grey', 'width': 1}, row=3)
        self.frame.add_ta(ta.VolDev('volume', 10), {'dash': 'solid', 'color': 'pink', 'width': 1}, row=3)
        self.frame.add_ta(ta.TrendDuration('MA_cl_50'), {'dash': 'solid', 'color': 'green', 'width': 1}, row=4)
        
        # Load and update the data in the frame
        self.frame.load_ohlcv(self.daydata)
        self.frame.update_ta_data()
        return self.frame.data

    def compute_daily_filters_with_scores(self):

        # add additional TA data that requires the first set of data to be loaded
        ta_filters = [
            ta.Breaks('close', 'MA_cl_200', True), # BreaksMA = 200
            ta.Breaks('close', 'MA_cl_50', True), # BreaksMA = 50
            ta.AboveBelow('close', 'MA_cl_50', True), # > MA50
            ta.AboveBelow('close', 'MA_cl_200', True), # > MA200
            ta.AboveBelow(3, 'TDUR_MA_cl_50', True), # MA50isRisingNthDays
            ta.AboveBelow(80, 'VDEV_10', True) # > 80% above MA10
        ]

        # add the additional TA data to the frame
        for t in ta_filters:
            self.frame.add_ta(t)

        self.frame.load_ohlcv(self.frame.data)
        self.frame.update_ta_data()


        # Get the names of all filters
        ta_filter_names = [ta.name for ta in ta_filters]

        
        # Calculate the sum of true values for each row
        self.frame.data['filter_score'] = self.frame.data[ta_filter_names].sum(axis=1)
        
        # Add column to indicate if all filters are true
        self.frame.data['all_true'] = self.frame.data[ta_filter_names].all(axis=1)

        self.frame.load_ohlcv(self.frame.data)
        self.frame.update_ta_data()

        self.dayscores = self.frame.data[ta_filter_names + ['filter_score', 'all_true']]

        return self.dayscores
    
    def get_market_sector_etf(self):
        
        return get_etf_from_sector_code('GICS', 45102010)


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


