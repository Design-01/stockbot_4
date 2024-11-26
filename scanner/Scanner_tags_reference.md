# Interactive Brokers Scanner Tags Reference

## Basic Market Scanner Codes (scanCode)

### Price/Volume Based
- `TOP_PERC_GAIN` - Top Percentage Gainers
- `TOP_PERC_LOSE` - Top Percentage Losers
- `MOST_ACTIVE` - Most Active
- `TOP_VOLUME_RATE` - Highest Volume Rate
- `HOT_BY_VOLUME` - Hot Stocks by Volume
- `TOP_PRICE_RANGE` - Top Price Range
- `HOT_BY_PRICE` - Hot by Price
- `TOP_TRADE_RATE` - Highest Trade Rate
- `TOP_TRADE_COUNT` - Most Trades Today

### Technical Analysis
- `HIGH_VS_52W_HL` - Near 52 Week High
- `LOW_VS_52W_HL` - Near 52 Week Low
- `HIGH_VS_13W_HL` - High vs 13 Week
- `LOW_VS_13W_HL` - Low vs 13 Week
- `HIGH_VS_26W_HL` - High vs 26 Week
- `LOW_VS_26W_HL` - Low vs 26 Week
- `HIGH_GROWTH_RATE` - Highest Growth Rate
- `TOP_RSI` - Highest RSI
- `BOTTOM_RSI` - Lowest RSI
- `HIGH_SYNTH_BID_REV_NAT` - Highest Synthetic Bid Rev
- `MOST_ACTIVE_USD` - Most Active in USD
- `HALTED` - Halted Stocks

### Value/Fundamentals
- `TOP_OPEN_PERC_GAIN` - Top Open Price Gainers
- `TOP_OPEN_PERC_LOSE` - Top Open Price Losers
- `HIGH_DIVIDEND_YIELD` - Highest Dividend Yield
- `TOP_ROE` - Highest Return on Equity
- `TOP_ROA` - Highest Return on Assets
- `PRICE_RANGE_USD` - Price Range in USD
- `READY_TO_ORDER` - Ready to Order

## Scanner Tag Parameters

### Price and Volume Tags
```
priceAbove=<value>
priceBelow=<value>
volumeAbove=<value>
avgVolumeAbove=<value>
avgVolumeBelow=<value>
tradeRateAbove=<trades_per_min>
```

### Market Cap Tags
```
marketCapAbove1=<value>  # in millions
marketCapBelow1=<value>  # in millions
```

### Movement and Volatility Tags
```
movementAbove=<percentage>
movementBelow=<percentage>
volatilityAbove=<value>
volatilityBelow=<value>
gapAbove=<percentage>
gapBelow=<percentage>
```

### Financial Metric Tags
```
peRatioAbove=<value>
peRatioBelow=<value>
pbRatioAbove=<value>
pbRatioBelow=<value>
divYieldAbove=<percentage>
divYieldBelow=<percentage>
epsGrowthAbove=<percentage>
revenueGrowthAbove=<percentage>
netProfitMarginAbove=<percentage>
debtToEquityBelow=<value>
```

### Technical Indicator Tags
```
rsiAbove=<value>
rsiBelow=<value>
macdAbove=<value>
macdBelow=<value>
stochKAbove=<value>
stochKBelow=<value>
stochDAbove=<value>
stochDBelow=<value>
```

### Options-Related Tags
```
optVolumeAbove=<value>
impliedVolatilityAbove=<percentage>
impliedVolatilityBelow=<percentage>
optOpenInterestAbove=<value>
optVolumeRateAbove=<value>
```

### Industry/Sector Tags
```
industryCode=<code>
sectorCode=<code>
moodyRatingAbove=<rating>
spRatingAbove=<rating>
```

### Exchange and Location Tags
```
locationCode=STK.US.MAJOR  # US Major Exchanges
locationCode=STK.US.MINOR  # US Minor Exchanges
locationCode=STK.HK.SEHK   # Hong Kong
locationCode=STK.EU        # European Stocks
```

## Example Usage in Code:

```python
from ib_insync import *

def create_custom_scanner(ib):
    # Create scanner subscription
    sub = ScannerSubscription(
        instrument='STK',
        locationCode='STK.US.MAJOR',
        scanCode='TOP_PERC_GAIN'
    )
    
    # Define tag values
    tag_values = [
        ('priceAbove', '10'),
        ('priceBelow', '100'),
        ('marketCapAbove1', '1000'),  # $1B minimum
        ('avgVolumeAbove', '100000'),
        ('peRatioAbove', '0'),
        ('peRatioBelow', '20'),
        ('gapAbove', '2'),
        ('movementAbove', '5')
    ]
    
    return ib.reqScannerData(sub, [], tag_values)
```

## Important Notes:

1. Not all combinations of tags work together
2. Some tags may be restricted based on your IB account type and permissions
3. Performance impact increases with the number of filters
4. Some tags may require market data subscriptions
5. Real-time scanning may have different available tags than end-of-day scanning
6. Tag availability may vary by exchange and instrument type

## Tips for Testing Tags:

1. Start with basic scans and add filters incrementally
2. Monitor API responses for error messages about invalid parameters
3. Use the TWS API documentation to verify tag support
4. Test during market hours for most accurate results
5. Consider using smaller universes (e.g., specific exchanges) when testing new tag combinations