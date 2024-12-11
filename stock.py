from dataclasses import dataclass
from typing import List, Dict, Union, Tuple, Optional
from datetime import datetime
import pandas as pd
from ib_insync import IB, Stock
import xml.etree.ElementTree as ET

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


def get_stock_fundamentals(ib, ticker: str, current_volume: float = 0) -> StockFundamentals:
    """Retrieve comprehensive stock information including all available ratios."""
    contract = Stock(ticker, 'SMART', 'USD')
    details = ib.reqContractDetails(contract)
    if not details:
        raise ValueError(f"No contract details found for {ticker}")
    
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
    
    return stock_info



class StockX:
    def __init__(self, ib:IB, symbol):
        self.ib = ib
        self.symbol = symbol
        self.fundamentals: StockFundamentals = None
        self.daydata: pd.DataFrame = None
        self.frame = Frame(self.symbol)
        self.dayscores = None

    def get_fundamentals(self):
        if not self.fundamentals:
            self.fundamentals = get_stock_fundamentals(self.ib, self.symbol)
        return self.fundamentals
    
    def get_daily_data(self, start_date:str="52 weeksAgo", end_date:str='now'):
        if not self.daydata:
            self.daydata =  hd.get_hist_data(self.symbol, start_date, end_date, '1 day')
        return self.daydata
    
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


