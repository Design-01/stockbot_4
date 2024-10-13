from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
from abc import ABC, abstractmethod
import requests
from ib_insync import IB, Stock, Contract

class Bias(Enum):
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0

class DataSource(ABC):
    @abstractmethod
    def get_stock_data(self, symbol: str) -> Dict:
        pass

    @abstractmethod
    def get_market_data(self) -> Dict:
        pass

class TwelveDataSource(DataSource):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"

    def get_stock_data(self, symbol: str) -> Dict:
        endpoint = f"{self.base_url}/quote"
        params = {
            "symbol": symbol,
            "apikey": self.api_key
        }
        response = requests.get(endpoint, params=params)
        data = response.json()
        
        return {
            "symbol": data["symbol"],
            "current_price": float(data["close"]),
            "previous_close": float(data["previous_close"]),
            "volume": int(data["volume"]),
            "average_volume": int(data["average_volume"]),
            "atr": float(data.get("atr", 0)),  # You might need to make a separate API call for ATR
            "news": []  # Twelve Data doesn't provide news, you'd need another source for this
        }

    def get_market_data(self) -> Dict:
        # For simplicity, we'll just get S&P 500 data as a proxy for market data
        endpoint = f"{self.base_url}/quote"
        params = {
            "symbol": "SPY",
            "apikey": self.api_key
        }
        response = requests.get(endpoint, params=params)
        data = response.json()
        
        return {
            "index_futures": {"S&P 500": float(data["close"])},
            "sector_performance": {}  # You'd need to implement sector performance separately
        }

class IBDataSource(DataSource):
    def __init__(self):
        self.ib = IB()
        self.ib.connect('127.0.0.1', 7497, clientId=1)  # Adjust these parameters as needed

    def get_stock_data(self, symbol: str) -> Dict:
        contract = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)
        
        [ticker] = self.ib.reqTickers(contract)
        
        return {
            "symbol": symbol,
            "current_price": ticker.last,
            "previous_close": ticker.close,
            "volume": ticker.volume,
            "average_volume": 0,  # IB doesn't provide this directly, you'd need to calculate it
            "atr": 0,  # You'd need to calculate this separately
            "news": []  # You'd need to implement news retrieval separately
        }

    def get_market_data(self) -> Dict:
        spy_contract = Stock('SPY', 'SMART', 'USD')
        self.ib.qualifyContracts(spy_contract)
        
        [spy_ticker] = self.ib.reqTickers(spy_contract)
        
        return {
            "index_futures": {"S&P 500": spy_ticker.last},
            "sector_performance": {}  # You'd need to implement sector performance separately
        }

@dataclass
class StockData:
    symbol: str
    current_price: float
    previous_close: float
    volume: int
    average_volume: int
    atr: float
    news: List[str] = field(default_factory=list)
    
    @property
    def price_change(self):
        return self.current_price - self.previous_close
    
    @property
    def price_change_percent(self):
        return (self.price_change / self.previous_close) * 100

@dataclass
class MarketData:
    index_futures: Dict[str, float]
    sector_performance: Dict[str, float]

@dataclass
class AnalysisResult:
    stock: StockData
    stock_bias: Bias
    market_bias: Bias
    momentum_score: float
    breakout_score: float
    volume_score: float
    gap_score: float
    room_to_move_score: float
    total_score: float = field(init=False)
    
    def __post_init__(self):
        self.total_score = sum([
            self.momentum_score,
            self.breakout_score,
            self.volume_score,
            self.gap_score,
            self.room_to_move_score
        ])

class PreMarketAnalyzer:
    def __init__(self, data_source: DataSource):
        self.data_source = data_source
    
    def analyze_stock(self, symbol: str, market_data: MarketData) -> AnalysisResult:
        stock_data = StockData(**self.data_source.get_stock_data(symbol))
        
        stock_bias = self._calculate_stock_bias(stock_data)
        market_bias = self._calculate_market_bias(market_data)
        
        momentum_score = self._calculate_momentum_score(stock_data)
        breakout_score = self._calculate_breakout_score(stock_data)
        volume_score = self._calculate_volume_score(stock_data)
        gap_score = self._calculate_gap_score(stock_data)
        room_to_move_score = self._calculate_room_to_move_score(stock_data)
        
        return AnalysisResult(
            stock=stock_data,
            stock_bias=stock_bias,
            market_bias=market_bias,
            momentum_score=momentum_score,
            breakout_score=breakout_score,
            volume_score=volume_score,
            gap_score=gap_score,
            room_to_move_score=room_to_move_score
        )
    
    def _calculate_stock_bias(self, stock_data: StockData) -> Bias:
        if stock_data.price_change_percent > 1:
            return Bias.BULLISH
        elif stock_data.price_change_percent < -1:
            return Bias.BEARISH
        else:
            return Bias.NEUTRAL
    
    def _calculate_market_bias(self, market_data: MarketData) -> Bias:
        sp500_change = market_data.index_futures.get("S&P 500", 0) - 1  # Assuming 1 is the baseline
        if sp500_change > 0.5:
            return Bias.BULLISH
        elif sp500_change < -0.5:
            return Bias.BEARISH
        else:
            return Bias.NEUTRAL
    
    def _calculate_momentum_score(self, stock_data: StockData) -> float:
        return min(max(stock_data.price_change_percent, -10), 10)  # Scale between -10 and 10
    
    def _calculate_breakout_score(self, stock_data: StockData) -> float:
        # This is a simplified example. In practice, you'd want to look at recent price levels
        if abs(stock_data.price_change_percent) > 5:
            return 10
        else:
            return 0
    
    def _calculate_volume_score(self, stock_data: StockData) -> float:
        volume_ratio = stock_data.volume / stock_data.average_volume if stock_data.average_volume else 1
        return min(volume_ratio * 5, 10)  # Scale up to 10
    
    def _calculate_gap_score(self, stock_data: StockData) -> float:
        gap_percent = abs(stock_data.price_change_percent)
        return min(gap_percent, 10)  # Scale up to 10
    
    def _calculate_room_to_move_score(self, stock_data: StockData) -> float:
        # This is a simplified example. In practice, you'd want to look at recent price levels and resistance/support
        atr_ratio = stock_data.atr / stock_data.current_price
        return min(atr_ratio * 100, 10)  # Scale up to 10

class PreMarketScanner:
    def __init__(self, analyzer: PreMarketAnalyzer, symbols: List[str]):
        self.analyzer = analyzer
        self.symbols = symbols
    
    def scan(self) -> List[AnalysisResult]:
        market_data = MarketData(**self.analyzer.data_source.get_market_data())
        results = []
        
        for symbol in self.symbols:
            result = self.analyzer.analyze_stock(symbol, market_data)
            results.append(result)
        
        return sorted(results, key=lambda x: x.total_score, reverse=True)

class ScanResultScorer:
    def __init__(self, weight_momentum: float = 1.0, weight_breakout: float = 1.0,
                 weight_volume: float = 1.0, weight_gap: float = 1.0,
                 weight_room_to_move: float = 1.0):
        self.weights = {
            'momentum': weight_momentum,
            'breakout': weight_breakout,
            'volume': weight_volume,
            'gap': weight_gap,
            'room_to_move': weight_room_to_move
        }
    
    def score(self, result: AnalysisResult) -> float:
        weighted_score = (
            result.momentum_score * self.weights['momentum'] +
            result.breakout_score * self.weights['breakout'] +
            result.volume_score * self.weights['volume'] +
            result.gap_score * self.weights['gap'] +
            result.room_to_move_score * self.weights['room_to_move']
        )
        return weighted_score
    
    def rank_results(self, results: List[AnalysisResult]) -> List[AnalysisResult]:
        return sorted(results, key=self.score, reverse=True)

# Example usage:
# twelve_data_source = TwelveDataSource(api_key="your_api_key_here")
# ib_data_source = IBDataSource()
# 
# analyzer_twelve = PreMarketAnalyzer(twelve_data_source)
# analyzer_ib = PreMarketAnalyzer(ib_data_source)
# 
# scanner_twelve = PreMarketScanner(analyzer_twelve, ['AAPL', 'GOOGL', 'MSFT', 'AMZN'])
# scanner_ib = PreMarketScanner(analyzer_ib, ['AAPL', 'GOOGL', 'MSFT', 'AMZN'])
# 
# results_twelve = scanner_twelve.scan()
# results_ib = scanner_ib.scan()
# 
# scorer = ScanResultScorer(weight_momentum=1.2, weight_volume=1.5)  # Emphasize momentum and volume
# ranked_results_twelve = scorer.rank_results(results_twelve)
# ranked_results_ib = scorer.rank_results(results_ib)
# 
# for result in ranked_results_twelve:
#     print(f"{result.stock.symbol}: Total Score = {scorer.score(result)}")