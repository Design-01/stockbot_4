from dataclasses import dataclass
from typing import List, Dict, Union, Tuple, Optional
from datetime import datetime
import pandas as pd
from ib_insync import IB
from fundamentals import get_stock_info, StockInfo, ForecastMetrics

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
class StockInfo:
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

def get_stock_info(ib, ticker: str, current_volume: float = 0) -> StockInfo:
    """Retrieve comprehensive stock information including all available ratios."""
    contract = Stock(ticker, 'SMART', 'USD')
    details = ib.reqContractDetails(contract)
    if not details:
        raise ValueError(f"No contract details found for {ticker}")
    
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
    
    stock_info = StockInfo(
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
#---------------------------------------------------------

import my_ib_utils

class Fundamentals:
    def __init__(self, ib: IB, rate_limiter: Optional[my_ib_utils.IBRateLimiter] = None):
        """
        Initialize Fundamentals with an existing IB connection and optional rate limiter
        
        Args:
            ib: Active IB connection instance
            rate_limiter: Optional IBRateLimiter instance
        """
        self.ib = ib
        self.rate_limiter = rate_limiter or my_ib_utils.IBRateLimiter(ib)
        self.stock_data: Dict[str, StockInfo] = {}

    def _get_single_stock_info(self, symbol: str) -> StockInfo:
        """Get fundamental data for a single symbol with rate limiting"""
        info = get_stock_info(self.ib, symbol)
        self.rate_limiter.wait()
        return info

    def get_fundamentals(self, symbols: List[str]) -> Dict[str, StockInfo]:
        """
        Fetch fundamental data for multiple stock symbols
        
        Args:
            symbols: List of stock ticker symbols
            
        Returns:
            Dictionary mapping symbols to their fundamental data
        """
        try:
            for symbol in symbols:
                stock_info = self._get_single_stock_info(symbol)
                self.stock_data[symbol] = stock_info
            return self.stock_data
        except Exception as e:
            print(f"Error fetching fundamentals: {e}")
            return {}

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all stored stock data to a pandas DataFrame"""
        if not self.stock_data:
            return pd.DataFrame()

        flat_data = []
        for symbol, info in self.stock_data.items():
            data_dict = info.__dict__.copy()
            forecast_dict = data_dict.pop('forecast').__dict__
            forecast_dict = {f'forecast_{k}': v for k, v in forecast_dict.items()}
            data_dict.update(forecast_dict)
            data_dict['symbol'] = symbol
            flat_data.append(data_dict)

        return pd.DataFrame(flat_data).set_index('symbol')

    def filter_stocks(self, criteria: Dict[str, Union[Tuple[float, float], float]]) -> pd.DataFrame:
        """Filter stocks based on specified criteria"""
        df = self.to_dataframe()
        if df.empty:
            return df

        for metric, value in criteria.items():
            if metric not in df.columns:
                print(f"Warning: Metric '{metric}' not found in data")
                continue

            if isinstance(value, tuple):
                min_val, max_val = value
                if min_val is not None:
                    df = df[df[metric] >= min_val]
                if max_val is not None:
                    df = df[df[metric] <= max_val]
            else:
                df = df[df[metric] == value]

        return df

    def get_summary_metrics(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Get key summary metrics for specified symbols"""
        df = self.to_dataframe()
        if df.empty:
            return df

        if symbols:
            df = df.loc[symbols]

        key_metrics = [
            'current_price', 'pe_ratio', 'market_cap', 'dividend_per_share',
            'eps_ttm', 'revenue_ttm', 'gross_margin', 'net_profit_margin',
            'forecast_target_price', 'forecast_consensus_recommendation'
        ]

        return df[key_metrics]