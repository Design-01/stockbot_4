from dataclasses import dataclass
from typing import List, Dict, Union, Tuple, Optional
from datetime import datetime
import pandas as pd
from ib_insync import IB, Stock
import xml.etree.ElementTree as ET


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
class StockFundamentals:
    # Sector and industry fields
    sector: str 
    industry: str
    sub_industry: str

    # Timestamp and Metadata
    latest_available_date: str
    price_currency: str
    reporting_currency: str
    exchange_rate: float
    
    # Price and Volume Metrics
    current_price: float
    high_52week: float
    low_52week: float
    pricing_date: str
    volume_10day_avg: float
    enterprise_value: float
    
    # Income Statement Metrics
    market_cap: float
    revenue_ttm: float
    ebitda: float
    net_income_ttm: float
    
    # Per Share Metrics
    eps_ttm: float
    revenue_per_share: float
    book_value_per_share: float
    cash_per_share: float
    cash_flow_per_share: float
    dividend_per_share: float
    
    # Margin Metrics
    gross_margin: float
    operating_margin: float
    net_profit_margin: float
    
    # Growth Metrics
    revenue_growth_rate: float
    eps_growth_rate: float
    
    # Valuation Metrics
    pe_ratio: float
    price_to_book: float
    price_to_sales: float
    
    # Company Information
    employee_count: int
    
    # Forecast Data
    forecast: Optional[ForecastMetrics] = None
    
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

def extract_industry_info(details) -> tuple:
    """
    Extract sector, industry, and sub-industry information from contract details.
    Returns default values if information is not available.
    """
    if not details:
        return "Unclassified", "Unclassified", "Unclassified"
    
    detail = details[0]
    
    # Try to get industry category info from contract details
    sector = getattr(detail.contract, 'industry', '') or ''
    industry = getattr(detail.contract, 'category', '') or ''
    sub_industry = getattr(detail.contract, 'subcategory', '') or ''
    
    # If any field is empty, try to parse from industry string
    if not all([sector, industry, sub_industry]):
        industry_str = getattr(detail, 'industryName', '') or ''
        if industry_str:
            parts = industry_str.split(' - ')
            if len(parts) >= 1:
                sector = sector or parts[0]
            if len(parts) >= 2:
                industry = industry or parts[1]
            if len(parts) >= 3:
                sub_industry = sub_industry or parts[2]
    
    # Provide default values if still empty
    return (
        sector or "Unclassified",
        industry or "Unclassified",
        sub_industry or "Unclassified"
    )

def get_stock_fundamentals(ib, ticker: str, current_volume: float = 0) -> StockFundamentals:
    """Retrieve comprehensive stock information including all available ratios."""
    contract = Stock(ticker, 'SMART', 'USD')
    details = ib.reqContractDetails(contract)
    if not details:
        raise ValueError(f"No contract details found for {ticker}")
    
    # Extract sector, industry, and sub-industry information
    sector, industry, sub_industry = extract_industry_info(details)
    
    fundamental_data = ib.reqFundamentalData(contract, 'ReportSnapshot')
    root = ET.fromstring(fundamental_data)
    
    # Get root level attributes
    ratios = root.find('.//Ratios')
    price_currency = ratios.get('PriceCurrency', 'USD')
    reporting_currency = ratios.get('ReportingCurrency', 'USD')
    exchange_rate = float(ratios.get('ExchangeRate', '1.0'))
    latest_date = ratios.get('LatestAvailableDate', '')
    
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
    
    stock_info = StockFundamentals(
        # Industry Classification
        sector=sector,
        industry=industry,
        sub_industry=sub_industry,
        
        # Metadata
        latest_available_date=latest_date,
        price_currency=price_currency,
        reporting_currency=reporting_currency,
        exchange_rate=exchange_rate,
        
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
        forecast=forecast
    )
    
    # Compute additional metrics
    stock_info.compute_derived_metrics(current_volume)
    
    return stock_info


from data import historical_data as hd
from frame.frame import Frame
from strategies import ta

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


