# IB Scanner API Documentation

## Basic Usage Example

```python
from ib_insync import ScannerSubscription, TagValue

# Create a scanner subscription
sub = ScannerSubscription(
    instrument='STK',
    locationCode='STK.US.MAJOR',
    scanCode='TOP_PERC_GAIN'
)

# Add filter criteria using TagValue objects
tagValues = [
    # Double values must be passed as strings
    TagValue('priceAbove', '25.50'),      # scanner.filter.DoubleField
    TagValue('changePercAbove', '5.5'),   # scanner.filter.DoubleField
    TagValue('volumeAbove', '1000000'),   # scanner.filter.IntField
    TagValue('marketCapAbove', '1e6'),    # scanner.filter.DoubleField
    TagValue('moodyRatingAbove', 'AAA')   # scanner.filter.ComboField
]

# Request scanner data
scanData = ib.reqScannerData(sub, [], tagValues)
```

## Important Notes About Data Types

When using TagValue filters, all values must be passed as strings, but they should conform to the expected format:

- `scanner.filter.DoubleField`: Pass numbers as strings (e.g., '25.50', '100.0')
- `scanner.filter.IntField`: Pass integers as strings (e.g., '1000000', '100')
- `scanner.filter.DateField`: Pass dates as 'YYYYMMDD' strings (e.g., '20240101')
- `scanner.filter.ComboField`: Pass the exact code as string (e.g., 'AAA' for ratings)


## Available TagValue Filters


### AFTERHRSCHANGE

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| afterHoursChangeAbove | DoubleField | After-Hours Change Above | `TagValue('afterHoursChangeAbove', '100.0')` |
| afterHoursChangeBelow | DoubleField | After-Hours Change Below | `TagValue('afterHoursChangeBelow', '100.0')` |

### AFTERHRSCHANGEPERC

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| afterHoursChangePercAbove | DoubleField | After-Hours Change (%) Above | `TagValue('afterHoursChangePercAbove', '5.5')` |
| afterHoursChangePercBelow | DoubleField | After-Hours Change (%) Below | `TagValue('afterHoursChangePercBelow', '5.5')` |

### AV1MOCHNG

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etf1moChngAbove | DoubleField | 1mo Change (AltaVista) Above | `TagValue('etf1moChngAbove', '100.0')` |
| etf1moChngBelow | DoubleField | 1mo Change (AltaVista) Below | `TagValue('etf1moChngBelow', '100.0')` |

### AV3MOCHNG

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etf3moChngAbove | DoubleField | 3mo Change (AltaVista) Above | `TagValue('etf3moChngAbove', '100.0')` |
| etf3moChngBelow | DoubleField | 3mo Change (AltaVista) Below | `TagValue('etf3moChngBelow', '100.0')` |

### AV5YREPS

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etf5yrEPSAbove | DoubleField | 5yr EPS Growth (AltaVista) Above | `TagValue('etf5yrEPSAbove', '100.0')` |
| etf5yrEPSBelow | DoubleField | 5yr EPS Growth (AltaVista) Below | `TagValue('etf5yrEPSBelow', '100.0')` |

### AVALTAR

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfALTARAbove | DoubleField | Altar Score (AltaVista) Above | `TagValue('etfALTARAbove', '100.0')` |
| etfALTARBelow | DoubleField | Altar Score (AltaVista) Below | `TagValue('etfALTARBelow', '100.0')` |

### AVASSETS

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfAssetsAbove | DoubleField | Assets Under Management (AltaVista) Above | `TagValue('etfAssetsAbove', '100.0')` |
| etfAssetsBelow | DoubleField | Assets Under Management (AltaVista) Below | `TagValue('etfAssetsBelow', '100.0')` |

### AVASSETTURNS

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfAssetTurnsAbove | DoubleField | Asset Turnover Ratio (AltaVista) Above | `TagValue('etfAssetTurnsAbove', '100.0')` |
| etfAssetTurnsBelow | DoubleField | Asset Turnover Ratio (AltaVista) Below | `TagValue('etfAssetTurnsBelow', '100.0')` |

### AVAVGALTAR

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfAvgALTARAbove | DoubleField | Avg Altar Score (AltaVista) Above | `TagValue('etfAvgALTARAbove', '100.0')` |
| etfAvgALTARBelow | DoubleField | Avg Altar Score (AltaVista) Below | `TagValue('etfAvgALTARBelow', '100.0')` |

### AVBETASPX

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfBetaSPXAbove | DoubleField | Beta S&P500 (AltaVista) Above | `TagValue('etfBetaSPXAbove', '100.0')` |
| etfBetaSPXBelow | DoubleField | Beta S&P500 (AltaVista) Below | `TagValue('etfBetaSPXBelow', '100.0')` |

### AVBIDASKPCT

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfBidAskPctAbove | DoubleField | Average Bid Ask Spread (AltaVista) Above | `TagValue('etfBidAskPctAbove', '100.0')` |
| etfBidAskPctBelow | DoubleField | Average Bid Ask Spread (AltaVista) Below | `TagValue('etfBidAskPctBelow', '100.0')` |

### AVCOMP_COUNT

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfComponent_CountAbove | DoubleField | Number of Components (AltaVista) Above | `TagValue('etfComponent_CountAbove', '100.0')` |
| etfComponent_CountBelow | DoubleField | Number of Components (AltaVista) Below | `TagValue('etfComponent_CountBelow', '100.0')` |

### AVCONVEX_WGTAVG

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfCONVEX_WgtAvgAbove | DoubleField | Convexity Weighted Average (AltaVista) Above | `TagValue('etfCONVEX_WgtAvgAbove', '100.0')` |
| etfCONVEX_WgtAvgBelow | DoubleField | Convexity Weighted Average (AltaVista) Below | `TagValue('etfCONVEX_WgtAvgBelow', '100.0')` |

### AVCPN_WGTAVG

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfCPN_WgtAvgAbove | DoubleField | Coupon Weighted Average (AltaVista) Above | `TagValue('etfCPN_WgtAvgAbove', '100.0')` |
| etfCPN_WgtAvgBelow | DoubleField | Coupon Weighted Average (AltaVista) Below | `TagValue('etfCPN_WgtAvgBelow', '100.0')` |

### AVCURYLD_WGTAVG

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfCurYld_WgtAvgAbove | DoubleField | Current Yield Weighted Average (AltaVista) Above | `TagValue('etfCurYld_WgtAvgAbove', '100.0')` |
| etfCurYld_WgtAvgBelow | DoubleField | Current Yield Weighted Average (AltaVista) Below | `TagValue('etfCurYld_WgtAvgBelow', '100.0')` |

### AVDEV

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfDevAbove | DoubleField | Developed Market Exposure (AltaVista) Above | `TagValue('etfDevAbove', '100.0')` |
| etfDevBelow | DoubleField | Developed Market Exposure (AltaVista) Below | `TagValue('etfDevBelow', '100.0')` |

### AVDUR_WGTAVG

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfDUR_WgtAvgAbove | DoubleField | Macaulay Duration Weighted Average (AltaVista) Above | `TagValue('etfDUR_WgtAvgAbove', '100.0')` |
| etfDUR_WgtAvgBelow | DoubleField | Macaulay Duration Weighted Average (AltaVista) Below | `TagValue('etfDUR_WgtAvgBelow', '100.0')` |

### AVEMG

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfEmgAbove | DoubleField | Emerging Market Exposure (AltaVista) Above | `TagValue('etfEmgAbove', '100.0')` |
| etfEmgBelow | DoubleField | Emerging Market Exposure (AltaVista) Below | `TagValue('etfEmgBelow', '100.0')` |

### AVEXPENSE

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfExpenseAbove | DoubleField | Expense (AltaVista) Above | `TagValue('etfExpenseAbove', '100.0')` |
| etfExpenseBelow | DoubleField | Expense (AltaVista) Below | `TagValue('etfExpenseBelow', '100.0')` |

### AVFINUM_DISTINCT

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfNum_DistinctAbove | DoubleField | Number of Issuers (AltaVista) Above | `TagValue('etfNum_DistinctAbove', '100.0')` |
| etfNum_DistinctBelow | DoubleField | Number of Issuers (AltaVista) Below | `TagValue('etfNum_DistinctBelow', '100.0')` |

### AVFWD_PCF

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfFwd_PCFAbove | DoubleField | Forward Price/Cash Flow (AltaVista) Above | `TagValue('etfFwd_PCFAbove', '100.0')` |
| etfFwd_PCFBelow | DoubleField | Forward Price/Cash Flow (AltaVista) Below | `TagValue('etfFwd_PCFBelow', '100.0')` |

### AVFWD_PE

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfFwd_PEAbove | DoubleField | Forward Price/Earnings (AltaVista) Above | `TagValue('etfFwd_PEAbove', '100.0')` |
| etfFwd_PEBelow | DoubleField | Forward Price/Earnings (AltaVista) Below | `TagValue('etfFwd_PEBelow', '100.0')` |

### AVFWD_YLD

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfFwd_YldAbove | DoubleField | Forward Dividend Yield (AltaVista) Above | `TagValue('etfFwd_YldAbove', '100.0')` |
| etfFwd_YldBelow | DoubleField | Forward Dividend Yield (AltaVista) Below | `TagValue('etfFwd_YldBelow', '100.0')` |

### AVFYCURBVPS

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| currYrETFFYBVPerShareAbove | DoubleField | Curr Year Book Value Per Share (AltaVista) Above | `TagValue('currYrETFFYBVPerShareAbove', '100.0')` |
| currYrETFFYBVPerShareBelow | DoubleField | Curr Year Book Value Per Share (AltaVista) Below | `TagValue('currYrETFFYBVPerShareBelow', '100.0')` |

### AVFYCURDPG

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| currYrETFFYDPSAbove | DoubleField | Curr Year Dividends Per Share (AltaVista) Above | `TagValue('currYrETFFYDPSAbove', '100.0')` |
| currYrETFFYDPSBelow | DoubleField | Curr Year Dividends Per Share (AltaVista) Below | `TagValue('currYrETFFYDPSBelow', '100.0')` |

### AVFYCURDY

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| currYrETFFYDividendYieldAbove | DoubleField | Curr Year Dividend Yield (AltaVista) Above | `TagValue('currYrETFFYDividendYieldAbove', '100.0')` |
| currYrETFFYDividendYieldBelow | DoubleField | Curr Year Dividend Yield (AltaVista) Below | `TagValue('currYrETFFYDividendYieldBelow', '100.0')` |

### AVFYCUREPS

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| currYrETFFYEPSAbove | DoubleField | Curr Year Earnings Per Share (AltaVista) Above | `TagValue('currYrETFFYEPSAbove', '100.0')` |
| currYrETFFYEPSBelow | DoubleField | Curr Year Earnings Per Share (AltaVista) Below | `TagValue('currYrETFFYEPSBelow', '100.0')` |

### AVFYCURNET

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| currYrETFFYNetMarginAbove | DoubleField | Curr Year Net Margins Per Share (AltaVista) Above | `TagValue('currYrETFFYNetMarginAbove', '100.0')` |
| currYrETFFYNetMarginBelow | DoubleField | Curr Year Net Margins Per Share (AltaVista) Below | `TagValue('currYrETFFYNetMarginBelow', '100.0')` |

### AVFYCURPBV

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| currYrETFFYPriceToBookValueAbove | DoubleField | Curr Year Price/Book Value Ratio (AltaVista) Above | `TagValue('currYrETFFYPriceToBookValueAbove', '25.50')` |
| currYrETFFYPriceToBookValueBelow | DoubleField | Curr Year Price/Book Value Ratio (AltaVista) Below | `TagValue('currYrETFFYPriceToBookValueBelow', '25.50')` |

### AVFYCURPCF

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| currYrETFFYPriceToCashFlowAbove | DoubleField | Curr Year Price/Cash Flow Ratio (AltaVista) Above | `TagValue('currYrETFFYPriceToCashFlowAbove', '25.50')` |
| currYrETFFYPriceToCashFlowBelow | DoubleField | Curr Year Price/Cash Flow Ratio (AltaVista) Below | `TagValue('currYrETFFYPriceToCashFlowBelow', '25.50')` |

### AVFYCURPE

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| currYrETFFYPriceToEarningsAbove | DoubleField | Curr Year Price/Earnings Ratio (AltaVista) Above | `TagValue('currYrETFFYPriceToEarningsAbove', '25.50')` |
| currYrETFFYPriceToEarningsBelow | DoubleField | Curr Year Price/Earnings Ratio (AltaVista) Below | `TagValue('currYrETFFYPriceToEarningsBelow', '25.50')` |

### AVFYCURPEG

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| currYrETFFYPriceToGrowthAbove | DoubleField | Curr Year Price/Growth Ratio (AltaVista) Above | `TagValue('currYrETFFYPriceToGrowthAbove', '25.50')` |
| currYrETFFYPriceToGrowthBelow | DoubleField | Curr Year Price/Growth Ratio (AltaVista) Below | `TagValue('currYrETFFYPriceToGrowthBelow', '25.50')` |

### AVFYCURPS

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| currYrETFFYPriceToSalesAbove | DoubleField | Curr Year Price/Sales Ratio (AltaVista) Above | `TagValue('currYrETFFYPriceToSalesAbove', '25.50')` |
| currYrETFFYPriceToSalesBelow | DoubleField | Curr Year Price/Sales Ratio (AltaVista) Below | `TagValue('currYrETFFYPriceToSalesBelow', '25.50')` |

### AVFYCURROE

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| currYrETFFYRoEAbove | DoubleField | Curr Year Return on Equity (AltaVista) Above | `TagValue('currYrETFFYRoEAbove', '100.0')` |
| currYrETFFYRoEBelow | DoubleField | Curr Year Return on Equity (AltaVista) Below | `TagValue('currYrETFFYRoEBelow', '100.0')` |

### AVFYCURSALYOY

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| currYrETFFYSPSGrowthAbove | DoubleField | Curr Year Sales Per Share Growth (AltaVista) Above | `TagValue('currYrETFFYSPSGrowthAbove', '100.0')` |
| currYrETFFYSPSGrowthBelow | DoubleField | Curr Year Sales Per Share Growth (AltaVista) Below | `TagValue('currYrETFFYSPSGrowthBelow', '100.0')` |

### AVFYCURSPS

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| currYrETFFYSPSAbove | DoubleField | Curr Year Sales Per Share (AltaVista) Above | `TagValue('currYrETFFYSPSAbove', '100.0')` |
| currYrETFFYSPSBelow | DoubleField | Curr Year Sales Per Share (AltaVista) Below | `TagValue('currYrETFFYSPSBelow', '100.0')` |

### AVFYCURYOY

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| currYrETFFYEPSGrowthAbove | DoubleField | Curr Year Earnings Per Share Growth (AltaVista) Above | `TagValue('currYrETFFYEPSGrowthAbove', '100.0')` |
| currYrETFFYEPSGrowthBelow | DoubleField | Curr Year Earnings Per Share Growth (AltaVista) Below | `TagValue('currYrETFFYEPSGrowthBelow', '100.0')` |

### AVFYNXTBVPS

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| nextYrETFFYBVPerShareAbove | DoubleField | Next Year Book Value Per Share (AltaVista) Above | `TagValue('nextYrETFFYBVPerShareAbove', '100.0')` |
| nextYrETFFYBVPerShareBelow | DoubleField | Next Year Book Value Per Share (AltaVista) Below | `TagValue('nextYrETFFYBVPerShareBelow', '100.0')` |

### AVFYNXTDPG

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| nextYrETFFYDPSAbove | DoubleField | Next Year Dividends Per Share (AltaVista) Above | `TagValue('nextYrETFFYDPSAbove', '100.0')` |
| nextYrETFFYDPSBelow | DoubleField | Next Year Dividends Per Share (AltaVista) Below | `TagValue('nextYrETFFYDPSBelow', '100.0')` |

### AVFYNXTDY

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| nextYrETFFYDividendYieldAbove | DoubleField | Next Year Dividend Yield (AltaVista) Above | `TagValue('nextYrETFFYDividendYieldAbove', '100.0')` |
| nextYrETFFYDividendYieldBelow | DoubleField | Next Year Dividend Yield (AltaVista) Below | `TagValue('nextYrETFFYDividendYieldBelow', '100.0')` |

### AVFYNXTEPS

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| nextYrETFFYEPSAbove | DoubleField | Next Year Earnings Per Share (AltaVista) Above | `TagValue('nextYrETFFYEPSAbove', '100.0')` |
| nextYrETFFYEPSBelow | DoubleField | Next Year Earnings Per Share (AltaVista) Below | `TagValue('nextYrETFFYEPSBelow', '100.0')` |

### AVFYNXTNET

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| nextYrETFFYNetMarginAbove | DoubleField | Next Year Net Margins Per Share (AltaVista) Above | `TagValue('nextYrETFFYNetMarginAbove', '100.0')` |
| nextYrETFFYNetMarginBelow | DoubleField | Next Year Net Margins Per Share (AltaVista) Below | `TagValue('nextYrETFFYNetMarginBelow', '100.0')` |

### AVFYNXTPBV

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| nextYrETFFYPriceToBookValueAbove | DoubleField | Next Year Price/Book Value Ratio (AltaVista) Above | `TagValue('nextYrETFFYPriceToBookValueAbove', '25.50')` |
| nextYrETFFYPriceToBookValueBelow | DoubleField | Next Year Price/Book Value Ratio (AltaVista) Below | `TagValue('nextYrETFFYPriceToBookValueBelow', '25.50')` |

### AVFYNXTPCF

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| nextYrETFFYPriceToCashFlowAbove | DoubleField | Next Year Price/Cash Flow Ratio (AltaVista) Above | `TagValue('nextYrETFFYPriceToCashFlowAbove', '25.50')` |
| nextYrETFFYPriceToCashFlowBelow | DoubleField | Next Year Price/Cash Flow Ratio (AltaVista) Below | `TagValue('nextYrETFFYPriceToCashFlowBelow', '25.50')` |

### AVFYNXTPE

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| nextYrETFFYPriceToEarningsAbove | DoubleField | Next Year Price/Earnings Ratio (AltaVista) Above | `TagValue('nextYrETFFYPriceToEarningsAbove', '25.50')` |
| nextYrETFFYPriceToEarningsBelow | DoubleField | Next Year Price/Earnings Ratio (AltaVista) Below | `TagValue('nextYrETFFYPriceToEarningsBelow', '25.50')` |

### AVFYNXTPEG

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| nextYrETFFYPriceToGrowthAbove | DoubleField | Next Year Price/Growth Ratio (AltaVista) Above | `TagValue('nextYrETFFYPriceToGrowthAbove', '25.50')` |
| nextYrETFFYPriceToGrowthBelow | DoubleField | Next Year Price/Growth Ratio (AltaVista) Below | `TagValue('nextYrETFFYPriceToGrowthBelow', '25.50')` |

### AVFYNXTPS

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| nextYrETFFYPriceToSalesAbove | DoubleField | Next Year Price/Sales Ratio (AltaVista) Above | `TagValue('nextYrETFFYPriceToSalesAbove', '25.50')` |
| nextYrETFFYPriceToSalesBelow | DoubleField | Next Year Price/Sales Ratio (AltaVista) Below | `TagValue('nextYrETFFYPriceToSalesBelow', '25.50')` |

### AVFYNXTROE

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| nextYrETFFYRoEAbove | DoubleField | Next Year Return on Equity (AltaVista) Above | `TagValue('nextYrETFFYRoEAbove', '100.0')` |
| nextYrETFFYRoEBelow | DoubleField | Next Year Return on Equity (AltaVista) Below | `TagValue('nextYrETFFYRoEBelow', '100.0')` |

### AVFYNXTSALYOY

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| nextYrETFFYSPSGrowthAbove | DoubleField | Next Year Sales Per Share Growth (AltaVista) Above | `TagValue('nextYrETFFYSPSGrowthAbove', '100.0')` |
| nextYrETFFYSPSGrowthBelow | DoubleField | Next Year Sales Per Share Growth (AltaVista) Below | `TagValue('nextYrETFFYSPSGrowthBelow', '100.0')` |

### AVFYNXTSPS

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| nextYrETFFYSPSAbove | DoubleField | Next Year Sales Per Share (AltaVista) Above | `TagValue('nextYrETFFYSPSAbove', '100.0')` |
| nextYrETFFYSPSBelow | DoubleField | Next Year Sales Per Share (AltaVista) Below | `TagValue('nextYrETFFYSPSBelow', '100.0')` |

### AVFYNXTYOY

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| nextYrETFFYEPSGrowthAbove | DoubleField | Next Year Earnings Per Share Growth (AltaVista) Above | `TagValue('nextYrETFFYEPSGrowthAbove', '100.0')` |
| nextYrETFFYEPSGrowthBelow | DoubleField | Next Year Earnings Per Share Growth (AltaVista) Below | `TagValue('nextYrETFFYEPSGrowthBelow', '100.0')` |

### AVFYPRVBVPS

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| prevYrETFFYBVPerShareAbove | DoubleField | Prev Year Book Value Per Share (AltaVista) Above | `TagValue('prevYrETFFYBVPerShareAbove', '100.0')` |
| prevYrETFFYBVPerShareBelow | DoubleField | Prev Year Book Value Per Share (AltaVista) Below | `TagValue('prevYrETFFYBVPerShareBelow', '100.0')` |

### AVFYPRVDPG

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| prevYrETFFYDPSAbove | DoubleField | Prev Year Dividends Per Share (AltaVista) Above | `TagValue('prevYrETFFYDPSAbove', '100.0')` |
| prevYrETFFYDPSBelow | DoubleField | Prev Year Dividends Per Share (AltaVista) Below | `TagValue('prevYrETFFYDPSBelow', '100.0')` |

### AVFYPRVDY

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| prevYrETFFYDividendYieldAbove | DoubleField | Prev Year Dividend Yield (AltaVista) Above | `TagValue('prevYrETFFYDividendYieldAbove', '100.0')` |
| prevYrETFFYDividendYieldBelow | DoubleField | Prev Year Dividend Yield (AltaVista) Below | `TagValue('prevYrETFFYDividendYieldBelow', '100.0')` |

### AVFYPRVEPS

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| prevYrETFFYEPSAbove | DoubleField | Prev Year Earnings Per Share (AltaVista) Above | `TagValue('prevYrETFFYEPSAbove', '100.0')` |
| prevYrETFFYEPSBelow | DoubleField | Prev Year Earnings Per Share (AltaVista) Below | `TagValue('prevYrETFFYEPSBelow', '100.0')` |

### AVFYPRVNET

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| prevYrETFFYNetMarginAbove | DoubleField | Prev Year Net Margins Per Share (AltaVista) Above | `TagValue('prevYrETFFYNetMarginAbove', '100.0')` |
| prevYrETFFYNetMarginBelow | DoubleField | Prev Year Net Margins Per Share (AltaVista) Below | `TagValue('prevYrETFFYNetMarginBelow', '100.0')` |

### AVFYPRVPBV

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| prevYrETFFYPriceToBookValueAbove | DoubleField | Prev Year Price/Book Value Ratio (AltaVista) Above | `TagValue('prevYrETFFYPriceToBookValueAbove', '25.50')` |
| prevYrETFFYPriceToBookValueBelow | DoubleField | Prev Year Price/Book Value Ratio (AltaVista) Below | `TagValue('prevYrETFFYPriceToBookValueBelow', '25.50')` |

### AVFYPRVPCF

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| prevYrETFFYPriceToCashFlowAbove | DoubleField | Prev Year Price/Cash Flow Ratio (AltaVista) Above | `TagValue('prevYrETFFYPriceToCashFlowAbove', '25.50')` |
| prevYrETFFYPriceToCashFlowBelow | DoubleField | Prev Year Price/Cash Flow Ratio (AltaVista) Below | `TagValue('prevYrETFFYPriceToCashFlowBelow', '25.50')` |

### AVFYPRVPE

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| prevYrETFFYPriceToEarningsAbove | DoubleField | Prev Year Price/Earnings Ratio (AltaVista) Above | `TagValue('prevYrETFFYPriceToEarningsAbove', '25.50')` |
| prevYrETFFYPriceToEarningsBelow | DoubleField | Prev Year Price/Earnings Ratio (AltaVista) Below | `TagValue('prevYrETFFYPriceToEarningsBelow', '25.50')` |

### AVFYPRVPEG

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| prevYrETFFYPriceToGrowthAbove | DoubleField | Prev Year Price/Growth Ratio (AltaVista) Above | `TagValue('prevYrETFFYPriceToGrowthAbove', '25.50')` |
| prevYrETFFYPriceToGrowthBelow | DoubleField | Prev Year Price/Growth Ratio (AltaVista) Below | `TagValue('prevYrETFFYPriceToGrowthBelow', '25.50')` |

### AVFYPRVPS

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| prevYrETFFYPriceToSalesAbove | DoubleField | Prev Year Price/Sales Ratio (AltaVista) Above | `TagValue('prevYrETFFYPriceToSalesAbove', '25.50')` |
| prevYrETFFYPriceToSalesBelow | DoubleField | Prev Year Price/Sales Ratio (AltaVista) Below | `TagValue('prevYrETFFYPriceToSalesBelow', '25.50')` |

### AVFYPRVROE

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| prevYrETFFYRoEAbove | DoubleField | Prev Year Return on Equity (AltaVista) Above | `TagValue('prevYrETFFYRoEAbove', '100.0')` |
| prevYrETFFYRoEBelow | DoubleField | Prev Year Return on Equity (AltaVista) Below | `TagValue('prevYrETFFYRoEBelow', '100.0')` |

### AVFYPRVSALYOY

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| prevYrETFFYSPSGrowthAbove | DoubleField | Prev Year Sales Per Share Growth (AltaVista) Above | `TagValue('prevYrETFFYSPSGrowthAbove', '100.0')` |
| prevYrETFFYSPSGrowthBelow | DoubleField | Prev Year Sales Per Share Growth (AltaVista) Below | `TagValue('prevYrETFFYSPSGrowthBelow', '100.0')` |

### AVFYPRVSPS

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| prevYrETFFYSPSAbove | DoubleField | Prev Year Sales Per Share (AltaVista) Above | `TagValue('prevYrETFFYSPSAbove', '100.0')` |
| prevYrETFFYSPSBelow | DoubleField | Prev Year Sales Per Share (AltaVista) Below | `TagValue('prevYrETFFYSPSBelow', '100.0')` |

### AVFYPRVYOY

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| prevYrETFFYEPSGrowthAbove | DoubleField | Prev Year Earnings Per Share Growth (AltaVista) Above | `TagValue('prevYrETFFYEPSGrowthAbove', '100.0')` |
| prevYrETFFYEPSGrowthBelow | DoubleField | Prev Year Earnings Per Share Growth (AltaVista) Below | `TagValue('prevYrETFFYEPSGrowthBelow', '100.0')` |

### AVGPRICETARGET

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| avgPriceTargetAbove | DoubleField | Avg Analyst Price Target Above | `TagValue('avgPriceTargetAbove', '25.50')` |
| avgPriceTargetBelow | DoubleField | Avg Analyst Price Target Below | `TagValue('avgPriceTargetBelow', '25.50')` |

### AVGRATING

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| avgRatingAbove | DoubleField | Avg Analyst Rating Above | `TagValue('avgRatingAbove', '100.0')` |
| avgRatingBelow | DoubleField | Avg Analyst Rating Below | `TagValue('avgRatingBelow', '100.0')` |

### AVGTARGET2PRICERATIO

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| avgAnalystTarget2PriceRatioAbove | DoubleField | Avg Analyst Target/Price Ratio Above | `TagValue('avgAnalystTarget2PriceRatioAbove', '25.50')` |
| avgAnalystTarget2PriceRatioBelow | DoubleField | Avg Analyst Target/Price Ratio Below | `TagValue('avgAnalystTarget2PriceRatioBelow', '25.50')` |

### AVGVOLUME

Category: High/Low/Volume

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| avgVolumeAbove | IntField | Avg Volume Above | `TagValue('avgVolumeAbove', '1000000')` |
| avgVolumeBelow | IntField | Avg Volume Below | `TagValue('avgVolumeBelow', '1000000')` |

### AVGVOLUME_USD

Category: High/Low/Volume

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| avgUsdVolumeAbove | DoubleField | Avg Volume ($) Above | `TagValue('avgUsdVolumeAbove', '100.0')` |
| avgUsdVolumeBelow | DoubleField | Avg Volume ($) Below | `TagValue('avgUsdVolumeBelow', '100.0')` |

### AVLEVERAGE

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfLeverageAbove | DoubleField | Leverage (AltaVista) Above | `TagValue('etfLeverageAbove', '100.0')` |
| etfLeverageBelow | DoubleField | Leverage (AltaVista) Below | `TagValue('etfLeverageBelow', '100.0')` |

### AVLTG

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfLTGAbove | DoubleField | Long Term Growth (AltaVista) Above | `TagValue('etfLTGAbove', '100.0')` |
| etfLTGBelow | DoubleField | Long Term Growth (AltaVista) Below | `TagValue('etfLTGBelow', '100.0')` |

### AVMATURITY_WGTAVG

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfMaturity_WgtAvgAbove | DoubleField | Maturity Weighted Average (AltaVista) Above | `TagValue('etfMaturity_WgtAvgAbove', '100.0')` |
| etfMaturity_WgtAvgBelow | DoubleField | Maturity Weighted Average (AltaVista) Below | `TagValue('etfMaturity_WgtAvgBelow', '100.0')` |

### AVMOD_DUR_WGTAVG

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfMOD_DUR_WgtAvgAbove | DoubleField | Modified Duration Weighted Average (AltaVista) Above | `TagValue('etfMOD_DUR_WgtAvgAbove', '100.0')` |
| etfMOD_DUR_WgtAvgBelow | DoubleField | Modified Duration Weighted Average (AltaVista) Below | `TagValue('etfMOD_DUR_WgtAvgBelow', '100.0')` |

### AVMOODYRATING

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfMoodyRatingAbove | ComboField | Average Moody Rating (AltaVista) Above | `TagValue('etfMoodyRatingAbove', 'AAA')` |

Allowed values for etfMoodyRatingAbove:
- `AAA`: AAA
- `AA1`: AA1
- `AA2`: AA2
- `AA3`: AA3
- `A1`: A1
- `A2`: A2
- `A3`: A3
- `BAA1`: BAA1
- `BAA2`: BAA2
- `BAA3`: BAA3
- `BA1`: BA1
- `BA2`: BA2
- `BA3`: BA3
- `B1`: B1
- `B2`: B2
- `B3`: B3
- `CAA1`: CAA1
- `CAA2`: CAA2
- `CAA3`: CAA3
- `CA`: CA
- `C`: C
- `NR`: NR

| etfMoodyRatingBelow | ComboField | Average Moody Rating (AltaVista) Below | `TagValue('etfMoodyRatingBelow', 'AAA')` |

Allowed values for etfMoodyRatingBelow:
- `AAA`: AAA
- `AA1`: AA1
- `AA2`: AA2
- `AA3`: AA3
- `A1`: A1
- `A2`: A2
- `A3`: A3
- `BAA1`: BAA1
- `BAA2`: BAA2
- `BAA3`: BAA3
- `BA1`: BA1
- `BA2`: BA2
- `BA3`: BA3
- `B1`: B1
- `B2`: B2
- `B3`: B3
- `CAA1`: CAA1
- `CAA2`: CAA2
- `CAA3`: CAA3
- `CA`: CA
- `C`: C
- `NR`: NR


### AVPAYOUT

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfPayoutAbove | DoubleField | Payout (AltaVista) Above | `TagValue('etfPayoutAbove', '100.0')` |
| etfPayoutBelow | DoubleField | Payout (AltaVista) Below | `TagValue('etfPayoutBelow', '100.0')` |

### AVPCT_AT_MTY

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfPct_at_MtyAbove | DoubleField | Portion With Standard Maturity (AltaVista) Above | `TagValue('etfPct_at_MtyAbove', '100.0')` |
| etfPct_at_MtyBelow | DoubleField | Portion With Standard Maturity (AltaVista) Below | `TagValue('etfPct_at_MtyBelow', '100.0')` |

### AVPCT_FIXED

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfPct_FixedAbove | DoubleField | Portion With Fixed Interest Rate (AltaVista) Above | `TagValue('etfPct_FixedAbove', '100.0')` |
| etfPct_FixedBelow | DoubleField | Portion With Fixed Interest Rate (AltaVista) Below | `TagValue('etfPct_FixedBelow', '100.0')` |

### AVRSI

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfRSIAbove | DoubleField | 30-day RSI (AltaVista) Above | `TagValue('etfRSIAbove', '100.0')` |
| etfRSIBelow | DoubleField | 30-day RSI (AltaVista) Below | `TagValue('etfRSIBelow', '100.0')` |

### AVSHT_INT

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfSht_IntAbove | DoubleField | Short Interest (AltaVista) Above | `TagValue('etfSht_IntAbove', '100.0')` |
| etfSht_IntBelow | DoubleField | Short Interest  (AltaVista) Below | `TagValue('etfSht_IntBelow', '100.0')` |

### AVSPRATING

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfSPRatingAbove | ComboField | Average S&P Rating (AltaVista) Above | `TagValue('etfSPRatingAbove', 'AAA')` |

Allowed values for etfSPRatingAbove:
- `AAA`: AAA
- `AA+`: AA+
- `AA`: AA
- `AA-`: AA-
- `A+`: A+
- `A`: A
- `A-`: A-
- `BBB+`: BBB+
- `BBB`: BBB
- `BBB-`: BBB-
- `BB+`: BB+
- `BB`: BB
- `BB-`: BB-
- `B+`: B+
- `B`: B
- `B-`: B-
- `CCC+`: CCC+
- `CCC`: CCC
- `CCC-`: CCC-
- `CC+`: CC+
- `CC`: CC
- `CC-`: CC-
- `C+`: C+
- `C`: C
- `C-`: C-
- `D`: D
- `NR`: NR

| etfSPRatingBelow | ComboField | Average S&P Rating (AltaVista) Below | `TagValue('etfSPRatingBelow', 'AAA')` |

Allowed values for etfSPRatingBelow:
- `AAA`: AAA
- `AA+`: AA+
- `AA`: AA
- `AA-`: AA-
- `A+`: A+
- `A`: A
- `A-`: A-
- `BBB+`: BBB+
- `BBB`: BBB
- `BBB-`: BBB-
- `BB+`: BB+
- `BB`: BB
- `BB-`: BB-
- `B+`: B+
- `B`: B
- `B-`: B-
- `CCC+`: CCC+
- `CCC`: CCC
- `CCC-`: CCC-
- `CC+`: CC+
- `CC`: CC
- `CC-`: CC-
- `C+`: C+
- `C`: C
- `C-`: C-
- `D`: D
- `NR`: NR


### AVTR10YR

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfTR10yrAbove | DoubleField | 10yr Total Return (AltaVista) Above | `TagValue('etfTR10yrAbove', '100.0')` |
| etfTR10yrBelow | DoubleField | 10yr Total Return (AltaVista) Below | `TagValue('etfTR10yrBelow', '100.0')` |

### AVTR1YR

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfTR1yrAbove | DoubleField | 1yr Total Return (AltaVista) Above | `TagValue('etfTR1yrAbove', '100.0')` |
| etfTR1yrBelow | DoubleField | 1yr Total Return (AltaVista) Below | `TagValue('etfTR1yrBelow', '100.0')` |

### AVTR5YR

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfTR5yrAbove | DoubleField | 5yr Total Return (AltaVista) Above | `TagValue('etfTR5yrAbove', '100.0')` |
| etfTR5yrBelow | DoubleField | 5yr Total Return (AltaVista) Below | `TagValue('etfTR5yrBelow', '100.0')` |

### AVTRACKINGDIFFPCT

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfTrackingDiffPctAbove | DoubleField | 1yr Tracking Difference (AltaVista) Above | `TagValue('etfTrackingDiffPctAbove', '100.0')` |
| etfTrackingDiffPctBelow | DoubleField | 1yr Tracking Difference (AltaVista) Below | `TagValue('etfTrackingDiffPctBelow', '100.0')` |

### AVTRINCEP

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfTRIncepAbove | DoubleField | Total Return Since Inception (AltaVista) Above | `TagValue('etfTRIncepAbove', '100.0')` |
| etfTRIncepBelow | DoubleField | Total Return Since Inception (AltaVista) Below | `TagValue('etfTRIncepBelow', '100.0')` |

### AVTRYTD

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfTRytdAbove | DoubleField | Year-to-Date Total Return (AltaVista) Above | `TagValue('etfTRytdAbove', '100.0')` |
| etfTRytdBelow | DoubleField | Year-to-Date Total Return (AltaVista) Below | `TagValue('etfTRytdBelow', '100.0')` |

### AVYTM_WGTAVG

Category: Alta Vista

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| etfYTM_WgtAvgAbove | DoubleField | Yield-To-Maturity Weighted Average (AltaVista) Above | `TagValue('etfYTM_WgtAvgAbove', '100.0')` |
| etfYTM_WgtAvgBelow | DoubleField | Yield-To-Maturity Weighted Average (AltaVista) Below | `TagValue('etfYTM_WgtAvgBelow', '100.0')` |

### BOND_AMT_OUTSTANDING

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondAmtOutstandingAbove | DoubleField | Amt. Outstanding Above | `TagValue('bondAmtOutstandingAbove', '100.0')` |
| bondAmtOutstandingBelow | DoubleField | Amt. Outstanding Below | `TagValue('bondAmtOutstandingBelow', '100.0')` |

### BOND_ASK

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondAskAbove | DoubleField | Price Above | `TagValue('bondAskAbove', '100.0')` |
| bondAskBelow | DoubleField | Price Below | `TagValue('bondAskBelow', '100.0')` |

### BOND_ASK_SZ_VALUE

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondAskSizeValueAbove | DoubleField | Quantity Above | `TagValue('bondAskSizeValueAbove', '100.0')` |
| bondAskSizeValueBelow | DoubleField | Quantity Below | `TagValue('bondAskSizeValueBelow', '100.0')` |

### BOND_ASK_YIELD

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondAskYieldAbove | DoubleField | Yield Above | `TagValue('bondAskYieldAbove', '100.0')` |
| bondAskYieldBelow | DoubleField | Yield Below | `TagValue('bondAskYieldBelow', '100.0')` |

### BOND_BID

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondBidAbove | DoubleField | Price Above | `TagValue('bondBidAbove', '100.0')` |
| bondBidBelow | DoubleField | Price Below | `TagValue('bondBidBelow', '100.0')` |

### BOND_BID_OR_ASK

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondBidOrAskAbove | DoubleField | Price Above | `TagValue('bondBidOrAskAbove', '100.0')` |
| bondBidOrAskBelow | DoubleField | Price Below | `TagValue('bondBidOrAskBelow', '100.0')` |

### BOND_BID_OR_ASK_SZ_VALUE

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondBidOrAskSizeValueAbove | DoubleField | Quantity Above | `TagValue('bondBidOrAskSizeValueAbove', '100.0')` |
| bondBidOrAskSizeValueBelow | DoubleField | Quantity Below | `TagValue('bondBidOrAskSizeValueBelow', '100.0')` |

### BOND_BID_OR_ASK_YIELD

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondBidOrAskYieldAbove | DoubleField | Yield Above | `TagValue('bondBidOrAskYieldAbove', '100.0')` |
| bondBidOrAskYieldBelow | DoubleField | Yield Below | `TagValue('bondBidOrAskYieldBelow', '100.0')` |

### BOND_BID_SZ_VALUE

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondBidSizeValueAbove | DoubleField | Quantity Above | `TagValue('bondBidSizeValueAbove', '100.0')` |
| bondBidSizeValueBelow | DoubleField | Quantity Below | `TagValue('bondBidSizeValueBelow', '100.0')` |

### BOND_BID_YIELD

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondBidYieldAbove | DoubleField | Yield Above | `TagValue('bondBidYieldAbove', '100.0')` |
| bondBidYieldBelow | DoubleField | Yield Below | `TagValue('bondBidYieldBelow', '100.0')` |

### BOND_CALL_PROT

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondNextCallDateAbove | DateField | Call Protection Above | `TagValue('bondNextCallDateAbove', '20240101')` |
| bondNextCallDateBelow | DateField | Call Protection Below | `TagValue('bondNextCallDateBelow', '20240101')` |

### BOND_CONVEXITY

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondConvexityAbove | DoubleField | Convexity Above | `TagValue('bondConvexityAbove', '100.0')` |
| bondConvexityBelow | DoubleField | Convexity Below | `TagValue('bondConvexityBelow', '100.0')` |

### BOND_DEBT_2_BOOK_RATIO

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondDebt2BookRatioAbove | DoubleField | Debt/Book Above | `TagValue('bondDebt2BookRatioAbove', '100.0')` |
| bondDebt2BookRatioBelow | DoubleField | Debt/Book Below | `TagValue('bondDebt2BookRatioBelow', '100.0')` |

### BOND_DEBT_2_EQUITY_RATIO

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondDebt2EquityRatioAbove | DoubleField | Debt/Equity Above | `TagValue('bondDebt2EquityRatioAbove', '100.0')` |
| bondDebt2EquityRatioBelow | DoubleField | Debt/Equity Below | `TagValue('bondDebt2EquityRatioBelow', '100.0')` |

### BOND_DEBT_2_TAN_BOOK_RATIO

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondDebt2TanBookRatioAbove | DoubleField | Debt/Tang. Book Above | `TagValue('bondDebt2TanBookRatioAbove', '100.0')` |
| bondDebt2TanBookRatioBelow | DoubleField | Debt/Tang. Book Below | `TagValue('bondDebt2TanBookRatioBelow', '100.0')` |

### BOND_DEBT_OUTSTANDING

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondDebtOutstandingAbove | DoubleField | Debt Outstanding Above | `TagValue('bondDebtOutstandingAbove', '100.0')` |
| bondDebtOutstandingBelow | DoubleField | Debt Outstanding Below | `TagValue('bondDebtOutstandingBelow', '100.0')` |

### BOND_DEBT_OUTSTANDING_MUNI

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondDebtOutstandingMuniAbove | DoubleField | Debt Outstanding Above | `TagValue('bondDebtOutstandingMuniAbove', '100.0')` |
| bondDebtOutstandingMuniBelow | DoubleField | Debt Outstanding Below | `TagValue('bondDebtOutstandingMuniBelow', '100.0')` |

### BOND_DURATION

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondDurationAbove | DoubleField | Duration (%) Above | `TagValue('bondDurationAbove', '100.0')` |
| bondDurationBelow | DoubleField | Duration (%) Below | `TagValue('bondDurationBelow', '100.0')` |

### BOND_EQUITY_2_BOOK_RATIO

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondEquity2BookRatioAbove | DoubleField | Equity/Book Above | `TagValue('bondEquity2BookRatioAbove', '100.0')` |
| bondEquity2BookRatioBelow | DoubleField | Equity/Book Below | `TagValue('bondEquity2BookRatioBelow', '100.0')` |

### BOND_EQUITY_2_TAN_BOOK_RATIO

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondEquity2TanBookRatioAbove | DoubleField | Equity/Tang. Book Above | `TagValue('bondEquity2TanBookRatioAbove', '100.0')` |
| bondEquity2TanBookRatioBelow | DoubleField | Equity/Tang. Book Below | `TagValue('bondEquity2TanBookRatioBelow', '100.0')` |

### BOND_INCREMENT_SIZE

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondIncrementSizeAbove | IntField | Increment Size Above | `TagValue('bondIncrementSizeAbove', '100')` |
| bondIncrementSizeBelow | IntField | Increment Size Below | `TagValue('bondIncrementSizeBelow', '100')` |

### BOND_INITIAL_SIZE

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondInitialSizeAbove | IntField | Initial Size Above | `TagValue('bondInitialSizeAbove', '100')` |
| bondInitialSizeBelow | IntField | Initial Size Below | `TagValue('bondInitialSizeBelow', '100')` |

### BOND_NET_ASK

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondNetAskAbove | DoubleField | Price Above | `TagValue('bondNetAskAbove', '100.0')` |
| bondNetAskBelow | DoubleField | Price Below | `TagValue('bondNetAskBelow', '100.0')` |

### BOND_NET_ASK_SZ_VALUE

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondNetAskSizeValueAbove | DoubleField | Quantity Above | `TagValue('bondNetAskSizeValueAbove', '100.0')` |
| bondNetAskSizeValueBelow | DoubleField | Quantity Below | `TagValue('bondNetAskSizeValueBelow', '100.0')` |

### BOND_NET_ASK_YIELD

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondNetAskYieldAbove | DoubleField | Yield Above | `TagValue('bondNetAskYieldAbove', '100.0')` |
| bondNetAskYieldBelow | DoubleField | Yield Below | `TagValue('bondNetAskYieldBelow', '100.0')` |

### BOND_NET_BID

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondNetBidAbove | DoubleField | Price Above | `TagValue('bondNetBidAbove', '100.0')` |
| bondNetBidBelow | DoubleField | Price Below | `TagValue('bondNetBidBelow', '100.0')` |

### BOND_NET_BID_OR_ASK

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondNetBidOrAskAbove | DoubleField | Price Above | `TagValue('bondNetBidOrAskAbove', '100.0')` |
| bondNetBidOrAskBelow | DoubleField | Price Below | `TagValue('bondNetBidOrAskBelow', '100.0')` |

### BOND_NET_BID_OR_ASK_SZ_VALUE

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondNetBidOrAskSizeValueAbove | DoubleField | Quantity Above | `TagValue('bondNetBidOrAskSizeValueAbove', '100.0')` |
| bondNetBidOrAskSizeValueBelow | DoubleField | Quantity Below | `TagValue('bondNetBidOrAskSizeValueBelow', '100.0')` |

### BOND_NET_BID_OR_ASK_YIELD

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondNetBidOrAskYieldAbove | DoubleField | Yield Above | `TagValue('bondNetBidOrAskYieldAbove', '100.0')` |
| bondNetBidOrAskYieldBelow | DoubleField | Yield Below | `TagValue('bondNetBidOrAskYieldBelow', '100.0')` |

### BOND_NET_BID_SZ_VALUE

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondNetBidSizeValueAbove | DoubleField | Quantity Above | `TagValue('bondNetBidSizeValueAbove', '100.0')` |
| bondNetBidSizeValueBelow | DoubleField | Quantity Below | `TagValue('bondNetBidSizeValueBelow', '100.0')` |

### BOND_NET_BID_YIELD

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondNetBidYieldAbove | DoubleField | Yield Above | `TagValue('bondNetBidYieldAbove', '100.0')` |
| bondNetBidYieldBelow | DoubleField | Yield Below | `TagValue('bondNetBidYieldBelow', '100.0')` |

### BOND_NET_SPREAD

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondNetSpreadAbove | DoubleField | Spread Above | `TagValue('bondNetSpreadAbove', '100.0')` |
| bondNetSpreadBelow | DoubleField | Spread Below | `TagValue('bondNetSpreadBelow', '100.0')` |

### BOND_SPREAD

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondSpreadAbove | DoubleField | Spread Above | `TagValue('bondSpreadAbove', '100.0')` |
| bondSpreadBelow | DoubleField | Spread Below | `TagValue('bondSpreadBelow', '100.0')` |

### BOND_STK_MKTCAP

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| bondStkMarketCapAbove | DoubleField | Equity Cap. Above | `TagValue('bondStkMarketCapAbove', '100.0')` |
| bondStkMarketCapBelow | DoubleField | Equity Cap. Below | `TagValue('bondStkMarketCapBelow', '100.0')` |

### CHANGEOPENPERC

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| changeOpenPercAbove | DoubleField | Change Since Open (%) Above | `TagValue('changeOpenPercAbove', '5.5')` |
| changeOpenPercBelow | DoubleField | Change Since Open (%) Below | `TagValue('changeOpenPercBelow', '5.5')` |

### CHANGEPERC

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| changePercAbove | DoubleField | Change (%) Above | `TagValue('changePercAbove', '5.5')` |
| changePercBelow | DoubleField | Change (%) Below | `TagValue('changePercBelow', '5.5')` |

### CPNRATE

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| couponRateAbove | DoubleField | Coupon Rate Above | `TagValue('couponRateAbove', '100.0')` |
| couponRateBelow | DoubleField | Coupon Rate Below | `TagValue('couponRateBelow', '100.0')` |

### DIVIB

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| dividendFrdAbove | DoubleField | Dividends Above (%) | `TagValue('dividendFrdAbove', '100.0')` |
| dividendFrdBelow | DoubleField | Dividends Below (%) | `TagValue('dividendFrdBelow', '100.0')` |

### DIVYIELDIB

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| dividendYieldFrdAbove | DoubleField | Dividend Yield Above (%) | `TagValue('dividendYieldFrdAbove', '100.0')` |
| dividendYieldFrdBelow | DoubleField | Dividend Yield Below (%) | `TagValue('dividendYieldFrdBelow', '100.0')` |

### EMA_100

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| curEMA100Above | DoubleField | EMA(100) Above | `TagValue('curEMA100Above', '100.0')` |
| curEMA100Below | DoubleField | EMA(100) Below | `TagValue('curEMA100Below', '100.0')` |

### EMA_20

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| curEMA20Above | DoubleField | EMA(20) Above | `TagValue('curEMA20Above', '100.0')` |
| curEMA20Below | DoubleField | EMA(20) Below | `TagValue('curEMA20Below', '100.0')` |

### EMA_200

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| curEMA200Above | DoubleField | EMA(200) Above | `TagValue('curEMA200Above', '100.0')` |
| curEMA200Below | DoubleField | EMA(200) Below | `TagValue('curEMA200Below', '100.0')` |

### EMA_50

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| curEMA50Above | DoubleField | EMA(50) Above | `TagValue('curEMA50Above', '100.0')` |
| curEMA50Below | DoubleField | EMA(50) Below | `TagValue('curEMA50Below', '100.0')` |

### EPS_CHANGE_TTM

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| epsChangeTTMAbove | DoubleField | EPS Chg TTM % (Refinitiv) Above | `TagValue('epsChangeTTMAbove', '100.0')` |
| epsChangeTTMBelow | DoubleField | EPS Chg TTM % (Refinitiv) Below | `TagValue('epsChangeTTMBelow', '100.0')` |

### ESG_COMBINED_SCORE

Category: ESG - Combined Scores

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| esgCombinedScoreAbove | DoubleField | ESG Combined Score Above | `TagValue('esgCombinedScoreAbove', '100.0')` |
| esgCombinedScoreBelow | DoubleField | ESG Combined Score Below | `TagValue('esgCombinedScoreBelow', '100.0')` |

### ESG_COMMUNITY_SCORE

Category: ESG - Pillar Scores

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| esgCommunityScoreAbove | DoubleField | ESG Community Score Above | `TagValue('esgCommunityScoreAbove', '100.0')` |
| esgCommunityScoreBelow | DoubleField | ESG Community Score Below | `TagValue('esgCommunityScoreBelow', '100.0')` |

### ESG_CONTROVERSIES_SCORE

Category: ESG - Pillar Scores

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| esgControversiesScoreAbove | DoubleField | ESG Controversies Score Above | `TagValue('esgControversiesScoreAbove', '100.0')` |
| esgControversiesScoreBelow | DoubleField | ESG Controversies Score Below | `TagValue('esgControversiesScoreBelow', '100.0')` |

### ESG_CORP_GOV_PILLAR_SCORE

Category: ESG Corporate Governance Score

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| esgCorpGovPillarScoreAbove | DoubleField | ESG Corporate Governance Score Above | `TagValue('esgCorpGovPillarScoreAbove', '100.0')` |
| esgCorpGovPillarScoreBelow | DoubleField | ESG Corporate Governance Score Below | `TagValue('esgCorpGovPillarScoreBelow', '100.0')` |

### ESG_CSR_STRATEGY_SCORE

Category: ESG - Pillar Scores

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| esgStrategyScoreAbove | DoubleField | ESG CSR Strategy Score Above | `TagValue('esgStrategyScoreAbove', '100.0')` |
| esgStrategyScoreBelow | DoubleField | ESG CSR Strategy Score Below | `TagValue('esgStrategyScoreBelow', '100.0')` |

### ESG_EMISSIONS_SCORE

Category: ESG - Pillar Scores

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| esgEmissionsScoreAbove | DoubleField | ESG Emissions Score Above | `TagValue('esgEmissionsScoreAbove', '100.0')` |
| esgEmissionsScoreBelow | DoubleField | ESG Emissions Score Below | `TagValue('esgEmissionsScoreBelow', '100.0')` |

### ESG_ENV_INNOVATION_SCORE

Category: ESG - Pillar Scores

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| esgEnvInvScoreAbove | DoubleField | ESG Environmental Innovation Score Above | `TagValue('esgEnvInvScoreAbove', '100.0')` |
| esgEnvInvScoreBelow | DoubleField | ESG Environmental Innovation Score Below | `TagValue('esgEnvInvScoreBelow', '100.0')` |

### ESG_ENV_PILLAR_SCORE

Category: ESG Environmental Score

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| esgEnvPillarScoreAbove | DoubleField | ESG Environmental Score Above | `TagValue('esgEnvPillarScoreAbove', '100.0')` |
| esgEnvPillarScoreBelow | DoubleField | ESG Environmental Score Below | `TagValue('esgEnvPillarScoreBelow', '100.0')` |

### ESG_HUMAN_RIGHTS_SCORE

Category: ESG - Pillar Scores

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| esgHrScoreAbove | DoubleField | ESG Human Rights Score Above | `TagValue('esgHrScoreAbove', '100.0')` |
| esgHrScoreBelow | DoubleField | ESG Human Rights Score Below | `TagValue('esgHrScoreBelow', '100.0')` |

### ESG_MANAGEMENT_SCORE

Category: ESG - Pillar Scores

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| esgManagementScoreAbove | DoubleField | ESG Management Score Above | `TagValue('esgManagementScoreAbove', '100.0')` |
| esgManagementScoreBelow | DoubleField | ESG Management Score Below | `TagValue('esgManagementScoreBelow', '100.0')` |

### ESG_PRODUCT_RESPONSIBILITY_SCORE

Category: ESG - Pillar Scores

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| esgProdRespScoreAbove | DoubleField | ESG Product Responsibility Score Above | `TagValue('esgProdRespScoreAbove', '100.0')` |
| esgProdRespScoreBelow | DoubleField | ESG Product Responsibility Score Below | `TagValue('esgProdRespScoreBelow', '100.0')` |

### ESG_RESOURCE_USE_SCORE

Category: ESG - Pillar Scores

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| esgResourceUseScoreAbove | DoubleField | ESG Resource Use Score Above | `TagValue('esgResourceUseScoreAbove', '100.0')` |
| esgResourceUseScoreBelow | DoubleField | ESG Resource Use Score Below | `TagValue('esgResourceUseScoreBelow', '100.0')` |

### ESG_SCORE

Category: ESG - Combined Scores

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| esgScoreAbove | DoubleField | ESG Score Above | `TagValue('esgScoreAbove', '100.0')` |
| esgScoreBelow | DoubleField | ESG Score Below | `TagValue('esgScoreBelow', '100.0')` |

### ESG_SHAREHOLDERS_SCORE

Category: ESG - Pillar Scores

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| esgShareholdersScoreAbove | DoubleField | ESG Shareholders Score Above | `TagValue('esgShareholdersScoreAbove', '100.0')` |
| esgShareholdersScoreBelow | DoubleField | ESG Shareholders Score Below | `TagValue('esgShareholdersScoreBelow', '100.0')` |

### ESG_SOCIAL_PILLAR_SCORE

Category: ESG Social Score

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| esgSocialPillarScoreAbove | DoubleField | ESG Social Score Above | `TagValue('esgSocialPillarScoreAbove', '100.0')` |
| esgSocialPillarScoreBelow | DoubleField | ESG Social Score Below | `TagValue('esgSocialPillarScoreBelow', '100.0')` |

### ESG_WORKFORCE_SCORE

Category: ESG - Pillar Scores

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| esgWorkforceScoreAbove | DoubleField | ESG Workforce Score Above | `TagValue('esgWorkforceScoreAbove', '100.0')` |
| esgWorkforceScoreBelow | DoubleField | ESG Workforce Score Below | `TagValue('esgWorkforceScoreBelow', '100.0')` |

### FEERATE

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| feeRateAbove | DoubleField | Fee Rate Above | `TagValue('feeRateAbove', '100.0')` |
| feeRateBelow | DoubleField | Fee Rate Below | `TagValue('feeRateBelow', '100.0')` |

### FIRSTTRADEDATE

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| firstTradeDateAbove | DateField | First Trade Date Above | `TagValue('firstTradeDateAbove', '20240101')` |
| firstTradeDateBelow | DateField | First Trade Date Below | `TagValue('firstTradeDateBelow', '20240101')` |

### GROWTHRATE

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| minGrowthRate | DoubleField | Growth Rate Above (Refinitiv) | `TagValue('minGrowthRate', '100.0')` |
| maxGrowthRate | DoubleField | Growth Rate Below (Refinitiv) | `TagValue('maxGrowthRate', '100.0')` |

### HISTDIVIB

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| histDividendFrdAbove | DoubleField | Dividends TTM Above (%) | `TagValue('histDividendFrdAbove', '100.0')` |
| histDividendFrdBelow | DoubleField | Dividends TTM Below (%) | `TagValue('histDividendFrdBelow', '100.0')` |

### HISTDIVYIELDIB

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| histDividendFrdYieldAbove | DoubleField | Dividend Yield TTM Above (%) | `TagValue('histDividendFrdYieldAbove', '100.0')` |
| histDividendFrdYieldBelow | DoubleField | Dividend Yield TTM Below (%) | `TagValue('histDividendFrdYieldBelow', '100.0')` |

### HV_PERCENTILE13

Category: Options

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| HVPercntl13wAbove | DoubleField | 13 Week HV Percentile Above | `TagValue('HVPercntl13wAbove', '5.5')` |
| HVPercntl13wBelow | DoubleField | 13 Week HV Percentile Below | `TagValue('HVPercntl13wBelow', '5.5')` |

### HV_PERCENTILE26

Category: Options

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| HVPercntl26wAbove | DoubleField | 26 Week HV Percentile Above | `TagValue('HVPercntl26wAbove', '5.5')` |
| HVPercntl26wBelow | DoubleField | 26 Week HV Percentile Below | `TagValue('HVPercntl26wBelow', '5.5')` |

### HV_PERCENTILE52

Category: Options

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| HVPercntl52wAbove | DoubleField | 52 Week HV Percentile Above | `TagValue('HVPercntl52wAbove', '5.5')` |
| HVPercntl52wBelow | DoubleField | 52 Week HV Percentile Below | `TagValue('HVPercntl52wBelow', '5.5')` |

### HV_RANK13

Category: Options

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| hvRank13wAbove | DoubleField | 13 Week HV Rank Above | `TagValue('hvRank13wAbove', '100.0')` |
| hvRank13wBelow | DoubleField | 13 Week HV Rank Below | `TagValue('hvRank13wBelow', '100.0')` |

### HV_RANK26

Category: Options

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| HVRank26wAbove | DoubleField | 26 Week HV Rank Above | `TagValue('HVRank26wAbove', '100.0')` |
| HVRank26wBelow | DoubleField | 26 Week HV Rank Below | `TagValue('HVRank26wBelow', '100.0')` |

### HV_RANK52

Category: Options

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| HVRank52wAbove | DoubleField | 52 Week HV Rank Above | `TagValue('HVRank52wAbove', '100.0')` |
| HVRank52wBelow | DoubleField | 52 Week HV Rank Below | `TagValue('HVRank52wBelow', '100.0')` |

### IMBALANCE

Category: Auction

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| imbalanceAbove | DoubleField | Imbalance Above | `TagValue('imbalanceAbove', '100.0')` |
| imbalanceBelow | DoubleField | Imbalance Below | `TagValue('imbalanceBelow', '100.0')` |

### IMBALANCEADVRATIOPERC

Category: Auction

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| displayImbalanceAdvRatioAbove | DoubleField | Imbalance Adv Ratio (%) Above | `TagValue('displayImbalanceAdvRatioAbove', '100.0')` |
| displayImbalanceAdvRatioBelow | DoubleField | Imbalance Adv Ratio (%) Below | `TagValue('displayImbalanceAdvRatioBelow', '100.0')` |

### IMPVOLAT

Category: Options

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| impVolatAbove | DoubleField | Opt Imp Vol Above | `TagValue('impVolatAbove', '100.0')` |
| impVolatBelow | DoubleField | Opt Imp Vol Below | `TagValue('impVolatBelow', '100.0')` |

### IMPVOLATCHANGEPERC

Category: Options

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| impVolatChangePercAbove | DoubleField | Opt Imp Vol Change (%) Above | `TagValue('impVolatChangePercAbove', '5.5')` |
| impVolatChangePercBelow | DoubleField | Opt Imp Vol Change (%) Below | `TagValue('impVolatChangePercBelow', '5.5')` |

### IMPVOLATOVERHIST

Category: Options

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| impVolatOverHistAbove | DoubleField | Opt Imp Vol Over Hist Above | `TagValue('impVolatOverHistAbove', '100.0')` |
| impVolatOverHistBelow | DoubleField | Opt Imp Vol Over Hist Below | `TagValue('impVolatOverHistBelow', '100.0')` |

### INSIDEROFFLOATPERC

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| ihInsiderOfFloatPercAbove | DoubleField | Insider Shares as % of Float Above | `TagValue('ihInsiderOfFloatPercAbove', '5.5')` |
| ihInsiderOfFloatPercBelow | DoubleField | Insider Shares as % of Float Below | `TagValue('ihInsiderOfFloatPercBelow', '5.5')` |

### INSTITUTIONALOFFLOATPERC

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| iiInstitutionalOfFloatPercAbove | DoubleField | Institutional Shares as % of Float Above | `TagValue('iiInstitutionalOfFloatPercAbove', '5.5')` |
| iiInstitutionalOfFloatPercBelow | DoubleField | Institutional Shares as % of Float Below | `TagValue('iiInstitutionalOfFloatPercBelow', '5.5')` |

### IV_PERCENTILE13

Category: Options

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| ivPercntl13wAbove | DoubleField | 13 Week IV Percentile Above | `TagValue('ivPercntl13wAbove', '5.5')` |
| ivPercntl13wBelow | DoubleField | 13 Week IV Percentile Below | `TagValue('ivPercntl13wBelow', '5.5')` |

### IV_PERCENTILE26

Category: Options

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| ivPercntl26wAbove | DoubleField | 26 Week IV Percentile Above | `TagValue('ivPercntl26wAbove', '5.5')` |
| ivPercntl26wBelow | DoubleField | 26 Week IV Percentile Below | `TagValue('ivPercntl26wBelow', '5.5')` |

### IV_PERCENTILE52

Category: Options

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| ivPercntl52wAbove | DoubleField | 52 Week IV Percentile Above | `TagValue('ivPercntl52wAbove', '5.5')` |
| ivPercntl52wBelow | DoubleField | 52 Week IV Percentile Below | `TagValue('ivPercntl52wBelow', '5.5')` |

### IV_RANK13

Category: Options

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| ivRank13wAbove | DoubleField | 13 Week IV Rank Above | `TagValue('ivRank13wAbove', '100.0')` |
| ivRank13wBelow | DoubleField | 13 Week IV Rank Below | `TagValue('ivRank13wBelow', '100.0')` |

### IV_RANK26

Category: Options

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| ivRank26wAbove | DoubleField | 26 Week IV Rank Above | `TagValue('ivRank26wAbove', '100.0')` |
| ivRank26wBelow | DoubleField | 26 Week IV Rank Below | `TagValue('ivRank26wBelow', '100.0')` |

### IV_RANK52

Category: Options

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| ivRank52wAbove | DoubleField | 52 Week IV Rank Above | `TagValue('ivRank52wAbove', '100.0')` |
| ivRank52wBelow | DoubleField | 52 Week IV Rank Below | `TagValue('ivRank52wBelow', '100.0')` |

### LIPPER_ALPHA_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperAlpha1yrAbove | DoubleField | Alpha 1Yr Above | `TagValue('lipperAlpha1yrAbove', '100.0')` |
| lipperAlpha1yrBelow | DoubleField | Alpha 1Yr Below | `TagValue('lipperAlpha1yrBelow', '100.0')` |

### LIPPER_AVG_COUPON

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperAverageCouponAbove | DoubleField | Average Coupon Above | `TagValue('lipperAverageCouponAbove', '100.0')` |
| lipperAverageCouponBelow | DoubleField | Average Coupon Below | `TagValue('lipperAverageCouponBelow', '100.0')` |

### LIPPER_AVG_FIN_COMP_ZSCORE

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperAverageFinalCompositeZScoreAbove | DoubleField | Average Final Composite Z-Score Above | `TagValue('lipperAverageFinalCompositeZScoreAbove', '100.0')` |
| lipperAverageFinalCompositeZScoreBelow | DoubleField | Average Final Composite Z-Score Below | `TagValue('lipperAverageFinalCompositeZScoreBelow', '100.0')` |

### LIPPER_AVG_LOSS_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperAverageLoss1yrAbove | DoubleField | Average Loss 1Yr Above | `TagValue('lipperAverageLoss1yrAbove', '100.0')` |
| lipperAverageLoss1yrBelow | DoubleField | Average Loss 1Yr Below | `TagValue('lipperAverageLoss1yrBelow', '100.0')` |

### LIPPER_AVG_RETURN_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperAverageReturn1yrAbove | DoubleField | Average Return 1Yr Above | `TagValue('lipperAverageReturn1yrAbove', '100.0')` |
| lipperAverageReturn1yrBelow | DoubleField | Average Return 1Yr Below | `TagValue('lipperAverageReturn1yrBelow', '100.0')` |

### LIPPER_BEAR_BETA_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperBearBeta1yrAbove | DoubleField | Bear Beta 1Yr Above | `TagValue('lipperBearBeta1yrAbove', '100.0')` |
| lipperBearBeta1yrBelow | DoubleField | Bear Beta 1Yr Below | `TagValue('lipperBearBeta1yrBelow', '100.0')` |

### LIPPER_BETA_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperBeta1yrAbove | DoubleField | Beta 1Yr Above | `TagValue('lipperBeta1yrAbove', '100.0')` |
| lipperBeta1yrBelow | DoubleField | Beta 1Yr Below | `TagValue('lipperBeta1yrBelow', '100.0')` |

### LIPPER_BULL_BETA_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperBullBeta1yrAbove | DoubleField | Bull Beta 1Yr Above | `TagValue('lipperBullBeta1yrAbove', '100.0')` |
| lipperBullBeta1yrBelow | DoubleField | Bull Beta 1Yr Below | `TagValue('lipperBullBeta1yrBelow', '100.0')` |

### LIPPER_CALC_AVG_QUALITY

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperCalculatedAverageQualityAbove | DoubleField | Calculated Average Quality Above | `TagValue('lipperCalculatedAverageQualityAbove', '5.5')` |
| lipperCalculatedAverageQualityBelow | DoubleField | Calculated Average Quality Below | `TagValue('lipperCalculatedAverageQualityBelow', '5.5')` |

### LIPPER_COMP_ZSCORE

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperLatestCompositeZScoreAbove | DoubleField | Composite Z-Score Latest Above | `TagValue('lipperLatestCompositeZScoreAbove', '100.0')` |
| lipperLatestCompositeZScoreBelow | DoubleField | Composite Z-Score Latest Below | `TagValue('lipperLatestCompositeZScoreBelow', '100.0')` |

### LIPPER_CORREL_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperCorrelation1yrAbove | DoubleField | Correlation 1Yr Above | `TagValue('lipperCorrelation1yrAbove', '5.5')` |
| lipperCorrelation1yrBelow | DoubleField | Correlation 1Yr Below | `TagValue('lipperCorrelation1yrBelow', '5.5')` |

### LIPPER_COVAR_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperCovariance1yrAbove | DoubleField | CoVariance 1Yr Above | `TagValue('lipperCovariance1yrAbove', '5.5')` |
| lipperCovariance1yrBelow | DoubleField | CoVariance 1Yr Below | `TagValue('lipperCovariance1yrBelow', '5.5')` |

### LIPPER_DIST_YLD_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperDistributionYield1yrValueAbove | DoubleField | Distribution Yield 1Yr Above | `TagValue('lipperDistributionYield1yrValueAbove', '100.0')` |
| lipperDistributionYield1yrValueBelow | DoubleField | Distribution Yield 1Yr Below | `TagValue('lipperDistributionYield1yrValueBelow', '100.0')` |

### LIPPER_DIV_YIELD_WGT_AVG

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperDividendYieldWAvgAbove | DoubleField | Dividend Yield Weighted Average Above | `TagValue('lipperDividendYieldWAvgAbove', '100.0')` |
| lipperDividendYieldWAvgBelow | DoubleField | Dividend Yield Weighted Average Below | `TagValue('lipperDividendYieldWAvgBelow', '100.0')` |

### LIPPER_DOWN_DEV_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperDownsideDeviationPopulation1yrAbove | DoubleField | Downside Deviation 1Yr Above | `TagValue('lipperDownsideDeviationPopulation1yrAbove', '100.0')` |
| lipperDownsideDeviationPopulation1yrBelow | DoubleField | Downside Deviation 1Yr Below | `TagValue('lipperDownsideDeviationPopulation1yrBelow', '100.0')` |

### LIPPER_DPS_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperDividendPerShare1yrAbove | DoubleField | Dividend Per Share 1Yr Above | `TagValue('lipperDividendPerShare1yrAbove', '100.0')` |
| lipperDividendPerShare1yrBelow | DoubleField | Dividend Per Share 1Yr Below | `TagValue('lipperDividendPerShare1yrBelow', '100.0')` |

### LIPPER_DPS_3YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperDividendPerShare3yrAbove | DoubleField | Dividend Per Share 3Yr Above | `TagValue('lipperDividendPerShare3yrAbove', '100.0')` |
| lipperDividendPerShare3yrBelow | DoubleField | Dividend Per Share 3Yr Below | `TagValue('lipperDividendPerShare3yrBelow', '100.0')` |

### LIPPER_EBIT_2_INT

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperEBIT2InterestAbove | DoubleField | EBIT to Interest Above | `TagValue('lipperEBIT2InterestAbove', '100.0')` |
| lipperEBIT2InterestBelow | DoubleField | EBIT to Interest Below | `TagValue('lipperEBIT2InterestBelow', '100.0')` |

### LIPPER_EFF_MATURITY

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperEffectiveMaturityAbove | DoubleField | Effective Maturity Above | `TagValue('lipperEffectiveMaturityAbove', '100.0')` |
| lipperEffectiveMaturityBelow | DoubleField | Effective Maturity Below | `TagValue('lipperEffectiveMaturityBelow', '100.0')` |

### LIPPER_EPS_GRWTH_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperEPSGrowth1yrAbove | DoubleField | EPS Growth - 1Yr Above | `TagValue('lipperEPSGrowth1yrAbove', '100.0')` |
| lipperEPSGrowth1yrBelow | DoubleField | EPS Growth - 1Yr Below | `TagValue('lipperEPSGrowth1yrBelow', '100.0')` |

### LIPPER_EPS_GRWTH_3YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperEPSGrowth3yrAbove | DoubleField | EPS Growth - 3Yr Above | `TagValue('lipperEPSGrowth3yrAbove', '100.0')` |
| lipperEPSGrowth3yrBelow | DoubleField | EPS Growth - 3Yr Below | `TagValue('lipperEPSGrowth3yrBelow', '100.0')` |

### LIPPER_EPS_GRWTH_5YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperEPSGrowth5yrAbove | DoubleField | EPS Growth - 5Yr Above | `TagValue('lipperEPSGrowth5yrAbove', '100.0')` |
| lipperEPSGrowth5yrBelow | DoubleField | EPS Growth - 5Yr Below | `TagValue('lipperEPSGrowth5yrBelow', '100.0')` |

### LIPPER_GRWTH_ANN_10YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperAnnualizedPerformance10yrValueAbove | DoubleField | Annualized Performance 10Yr Above | `TagValue('lipperAnnualizedPerformance10yrValueAbove', '100.0')` |
| lipperAnnualizedPerformance10yrValueBelow | DoubleField | Annualized Performance 10Yr Below | `TagValue('lipperAnnualizedPerformance10yrValueBelow', '100.0')` |

### LIPPER_GRWTH_ANN_3YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperAnnualizedPerformance3yrValueAbove | DoubleField | Annualized Performance 3Yr Above | `TagValue('lipperAnnualizedPerformance3yrValueAbove', '100.0')` |
| lipperAnnualizedPerformance3yrValueBelow | DoubleField | Annualized Performance 3Yr Below | `TagValue('lipperAnnualizedPerformance3yrValueBelow', '100.0')` |

### LIPPER_GRWTH_ANN_5YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperAnnualizedPerformance5yrValueAbove | DoubleField | Annualized Performance 5Yr Above | `TagValue('lipperAnnualizedPerformance5yrValueAbove', '100.0')` |
| lipperAnnualizedPerformance5yrValueBelow | DoubleField | Annualized Performance 5Yr Below | `TagValue('lipperAnnualizedPerformance5yrValueBelow', '100.0')` |

### LIPPER_GRWTH_CUM

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperPercentageGrowthCumulativeValueAbove | DoubleField | Percentage Growth Cumulative Above | `TagValue('lipperPercentageGrowthCumulativeValueAbove', '5.5')` |
| lipperPercentageGrowthCumulativeValueBelow | DoubleField | Percentage Growth Cumulative Below | `TagValue('lipperPercentageGrowthCumulativeValueBelow', '5.5')` |

### LIPPER_INFO_RATIO_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperInformationRatio1yrAbove | DoubleField | Information Ratio 1Yr Above | `TagValue('lipperInformationRatio1yrAbove', '100.0')` |
| lipperInformationRatio1yrBelow | DoubleField | Information Ratio 1Yr Below | `TagValue('lipperInformationRatio1yrBelow', '100.0')` |

### LIPPER_LEVERAGE_RATIO

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperFundLeverageRatioAbove | DoubleField | Fund Leverage Ratio Above | `TagValue('lipperFundLeverageRatioAbove', '100.0')` |
| lipperFundLeverageRatioBelow | DoubleField | Fund Leverage Ratio Below | `TagValue('lipperFundLeverageRatioBelow', '100.0')` |

### LIPPER_LT_DEBT_2_SE

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperLTDebt2ShareholdersEquityAbove | DoubleField | LT Debt / Shareholders Equity Above | `TagValue('lipperLTDebt2ShareholdersEquityAbove', '100.0')` |
| lipperLTDebt2ShareholdersEquityBelow | DoubleField | LT Debt / Shareholders Equity Below | `TagValue('lipperLTDebt2ShareholdersEquityBelow', '100.0')` |

### LIPPER_MAX_DRAW_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperMaximumDrawdown1yrAbove | DoubleField | Max Drawdown 1Yr Above | `TagValue('lipperMaximumDrawdown1yrAbove', '100.0')` |
| lipperMaximumDrawdown1yrBelow | DoubleField | Max Drawdown 1Yr Below | `TagValue('lipperMaximumDrawdown1yrBelow', '100.0')` |

### LIPPER_MAX_GAIN_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperMaxGain1yrAbove | DoubleField | Max Gain 1Yr Above | `TagValue('lipperMaxGain1yrAbove', '100.0')` |
| lipperMaxGain1yrBelow | DoubleField | Max Gain 1Yr Below | `TagValue('lipperMaxGain1yrBelow', '100.0')` |

### LIPPER_MAX_LOSS_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperMaxLoss1yrAbove | DoubleField | Max Loss 1Yr Above | `TagValue('lipperMaxLoss1yrAbove', '100.0')` |
| lipperMaxLoss1yrBelow | DoubleField | Max Loss 1Yr Below | `TagValue('lipperMaxLoss1yrBelow', '100.0')` |

### LIPPER_MKT_CAP_AVG

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperMktCapAvgLatestAbove | DoubleField | Market Capitalisation - Avg. Latest Above | `TagValue('lipperMktCapAvgLatestAbove', '100.0')` |
| lipperMktCapAvgLatestBelow | DoubleField | Market Capitalisation - Avg. Latest Below | `TagValue('lipperMktCapAvgLatestBelow', '100.0')` |

### LIPPER_NOM_MATURITY

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperNominalMaturityAbove | DoubleField | Nominal Maturity Above | `TagValue('lipperNominalMaturityAbove', '100.0')` |
| lipperNominalMaturityBelow | DoubleField | Nominal Maturity Below | `TagValue('lipperNominalMaturityBelow', '100.0')` |

### LIPPER_NUM_OF_SEC

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperNumberOfSecuritiesAbove | DoubleField | Number of Securities Above | `TagValue('lipperNumberOfSecuritiesAbove', '100.0')` |
| lipperNumberOfSecuritiesBelow | DoubleField | Number of Securities Below | `TagValue('lipperNumberOfSecuritiesBelow', '100.0')` |

### LIPPER_OP_CASH_FLOW_GRWTH_RATE_3YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperOperCashFlowGrowthRate3yrAbove | DoubleField | Operating Cash Flow - Growth Rate 3Yr Above | `TagValue('lipperOperCashFlowGrowthRate3yrAbove', '5.5')` |
| lipperOperCashFlowGrowthRate3yrBelow | DoubleField | Operating Cash Flow - Growth Rate 3Yr Below | `TagValue('lipperOperCashFlowGrowthRate3yrBelow', '5.5')` |

### LIPPER_PAYOUT_RATIO

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperDividendPayoutRatioAbove | DoubleField | Dividend Payout Ratio Above | `TagValue('lipperDividendPayoutRatioAbove', '100.0')` |
| lipperDividendPayoutRatioBelow | DoubleField | Dividend Payout Ratio Below | `TagValue('lipperDividendPayoutRatioBelow', '100.0')` |

### LIPPER_PAYOUT_RATIO_5YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperDividendPayoutRatio5yrAbove | DoubleField | Dividend Payout Ratio - 5Yr Above | `TagValue('lipperDividendPayoutRatio5yrAbove', '100.0')` |
| lipperDividendPayoutRatio5yrBelow | DoubleField | Dividend Payout Ratio - 5Yr Below | `TagValue('lipperDividendPayoutRatio5yrBelow', '100.0')` |

### LIPPER_PB_ZSCORE

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperLatestPrice2BookZScoreAbove | DoubleField | Price to Book Z-Score Latest Above | `TagValue('lipperLatestPrice2BookZScoreAbove', '25.50')` |
| lipperLatestPrice2BookZScoreBelow | DoubleField | Price to Book Z-Score Latest Below | `TagValue('lipperLatestPrice2BookZScoreBelow', '25.50')` |

### LIPPER_PCT_CHANGE

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperPctChangeAbove | DoubleField | Pct Change Above | `TagValue('lipperPctChangeAbove', '100.0')` |
| lipperPctChangeBelow | DoubleField | Pct Change Below | `TagValue('lipperPctChangeBelow', '100.0')` |

### LIPPER_PE_ZSCORE

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperLatestPrice2EarningsZScoreAbove | DoubleField | Price to Earnings Z-Score Latest Above | `TagValue('lipperLatestPrice2EarningsZScoreAbove', '25.50')` |
| lipperLatestPrice2EarningsZScoreBelow | DoubleField | Price to Earnings Z-Score Latest Below | `TagValue('lipperLatestPrice2EarningsZScoreBelow', '25.50')` |

### LIPPER_POS_PERIODS_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperPositivePeriods1yrAbove | DoubleField | Positive Periods 1Yr Above | `TagValue('lipperPositivePeriods1yrAbove', '100.0')` |
| lipperPositivePeriods1yrBelow | DoubleField | Positive Periods 1Yr Below | `TagValue('lipperPositivePeriods1yrBelow', '100.0')` |

### LIPPER_PRICE_2_BOOK

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperPrice2BookRatioAbove | DoubleField | Price/Book Above | `TagValue('lipperPrice2BookRatioAbove', '25.50')` |
| lipperPrice2BookRatioBelow | DoubleField | Price/Book Below | `TagValue('lipperPrice2BookRatioBelow', '25.50')` |

### LIPPER_PRICE_2_BOOK_LATEST

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperLatestPrice2BookRatioAbove | DoubleField | Price/Book Ratio Latest Above | `TagValue('lipperLatestPrice2BookRatioAbove', '25.50')` |
| lipperLatestPrice2BookRatioBelow | DoubleField | Price/Book Ratio Latest Below | `TagValue('lipperLatestPrice2BookRatioBelow', '25.50')` |

### LIPPER_PRICE_2_CASH

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperPrice2CashAbove | DoubleField | Price to Cash Above | `TagValue('lipperPrice2CashAbove', '25.50')` |
| lipperPrice2CashBelow | DoubleField | Price to Cash Below | `TagValue('lipperPrice2CashBelow', '25.50')` |

### LIPPER_PRICE_2_DIV

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperPrice2DividendAbove | DoubleField | Price to Dividend Above | `TagValue('lipperPrice2DividendAbove', '25.50')` |
| lipperPrice2DividendBelow | DoubleField | Price to Dividend Below | `TagValue('lipperPrice2DividendBelow', '25.50')` |

### LIPPER_PRICE_2_EARNINGS

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperPrice2EarningsRatioAbove | DoubleField | Price/Earnings Ratio Above | `TagValue('lipperPrice2EarningsRatioAbove', '25.50')` |
| lipperPrice2EarningsRatioBelow | DoubleField | Price/Earnings Ratio Below | `TagValue('lipperPrice2EarningsRatioBelow', '25.50')` |

### LIPPER_PRICE_2_EARNINGS_LATEST

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperLatestPrice2EarningsRatioAbove | DoubleField | Price/Earnings Ratio Latest Above | `TagValue('lipperLatestPrice2EarningsRatioAbove', '25.50')` |
| lipperLatestPrice2EarningsRatioBelow | DoubleField | Price/Earnings Ratio Latest Below | `TagValue('lipperLatestPrice2EarningsRatioBelow', '25.50')` |

### LIPPER_PRICE_2_SALES

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperPrice2SalesRatioAbove | DoubleField | Price/Sales Ratio Latest Above | `TagValue('lipperPrice2SalesRatioAbove', '25.50')` |
| lipperPrice2SalesRatioBelow | DoubleField | Price/Sales Ratio Latest Below | `TagValue('lipperPrice2SalesRatioBelow', '25.50')` |

### LIPPER_PRICE_2_SALES_LATEST

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperLatestPrice2SalesRatioAbove | DoubleField | Price/Sales Ratio Latest Above | `TagValue('lipperLatestPrice2SalesRatioAbove', '25.50')` |
| lipperLatestPrice2SalesRatioBelow | DoubleField | Price/Sales Ratio Latest Below | `TagValue('lipperLatestPrice2SalesRatioBelow', '25.50')` |

### LIPPER_PROJ_YIELD

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperProjectedYieldValueAbove | DoubleField | Projected Yield Value Above | `TagValue('lipperProjectedYieldValueAbove', '100.0')` |
| lipperProjectedYieldValueBelow | DoubleField | Projected Yield Value Below | `TagValue('lipperProjectedYieldValueBelow', '100.0')` |

### LIPPER_PS_ZSCORE

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperLatestPrice2SalesZScoreAbove | DoubleField | Price/Sales Z-Score Latest Above | `TagValue('lipperLatestPrice2SalesZScoreAbove', '25.50')` |
| lipperLatestPrice2SalesZScoreBelow | DoubleField | Price/Sales Z-Score Latest Below | `TagValue('lipperLatestPrice2SalesZScoreBelow', '25.50')` |

### LIPPER_REL_STRENGTH

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperRelativeStrengthAbove | DoubleField | Relative Strength Above | `TagValue('lipperRelativeStrengthAbove', '100.0')` |
| lipperRelativeStrengthBelow | DoubleField | Relative Strength Below | `TagValue('lipperRelativeStrengthBelow', '100.0')` |

### LIPPER_RET_ON_CAPITAL

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperReturnOnCapitalAbove | DoubleField | Return on Capital Above | `TagValue('lipperReturnOnCapitalAbove', '100.0')` |
| lipperReturnOnCapitalBelow | DoubleField | Return on Capital Below | `TagValue('lipperReturnOnCapitalBelow', '100.0')` |

### LIPPER_RET_ON_CAPITAL_3YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperReturnOnCapital3yrAbove | DoubleField | Return on Capital 3Yr Above | `TagValue('lipperReturnOnCapital3yrAbove', '100.0')` |
| lipperReturnOnCapital3yrBelow | DoubleField | Return on Capital 3Yr Below | `TagValue('lipperReturnOnCapital3yrBelow', '100.0')` |

### LIPPER_RET_RISK_RATIO_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperReturnRiskRatio1yrAbove | DoubleField | Return Risk Ratio 1Yr Above | `TagValue('lipperReturnRiskRatio1yrAbove', '100.0')` |
| lipperReturnRiskRatio1yrBelow | DoubleField | Return Risk Ratio 1Yr Below | `TagValue('lipperReturnRiskRatio1yrBelow', '100.0')` |

### LIPPER_ROA_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperReturnOnAssets1yrAbove | DoubleField | Return on Assets 1Yr Above | `TagValue('lipperReturnOnAssets1yrAbove', '100.0')` |
| lipperReturnOnAssets1yrBelow | DoubleField | Return on Assets 1Yr Below | `TagValue('lipperReturnOnAssets1yrBelow', '100.0')` |

### LIPPER_ROA_3YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperReturnOnAssets3yrAbove | DoubleField | Return on Assets 3Yr Above | `TagValue('lipperReturnOnAssets3yrAbove', '100.0')` |
| lipperReturnOnAssets3yrBelow | DoubleField | Return on Assets 3Yr Below | `TagValue('lipperReturnOnAssets3yrBelow', '100.0')` |

### LIPPER_ROE_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperReturnOnEquity1yrAbove | DoubleField | Return on Equity 1Yr Above | `TagValue('lipperReturnOnEquity1yrAbove', '100.0')` |
| lipperReturnOnEquity1yrBelow | DoubleField | Return on Equity 1Yr Below | `TagValue('lipperReturnOnEquity1yrBelow', '100.0')` |

### LIPPER_ROE_3YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperReturnOnEquity3yrAbove | DoubleField | Return on Equity 3Yr Above | `TagValue('lipperReturnOnEquity3yrAbove', '100.0')` |
| lipperReturnOnEquity3yrBelow | DoubleField | Return on Equity 3Yr Below | `TagValue('lipperReturnOnEquity3yrBelow', '100.0')` |

### LIPPER_ROE_WGT_AVG_LATEST

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperLatestReturnOnEquityWAvgAbove | DoubleField | Return on Equity Weighted Average Latest Above | `TagValue('lipperLatestReturnOnEquityWAvgAbove', '100.0')` |
| lipperLatestReturnOnEquityWAvgBelow | DoubleField | Return on Equity Weighted Average Latest Below | `TagValue('lipperLatestReturnOnEquityWAvgBelow', '100.0')` |

### LIPPER_ROE_ZSCORE

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperReturnOnEquityZScoreAbove | DoubleField | Return on Equity Z-Score Latest Above | `TagValue('lipperReturnOnEquityZScoreAbove', '100.0')` |
| lipperReturnOnEquityZScoreBelow | DoubleField | Return on Equity Z-Score Latest Below | `TagValue('lipperReturnOnEquityZScoreBelow', '100.0')` |

### LIPPER_ROI_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperReturnOnInvestment1yrAbove | DoubleField | Return on Investment 1Yr Above | `TagValue('lipperReturnOnInvestment1yrAbove', '100.0')` |
| lipperReturnOnInvestment1yrBelow | DoubleField | Return on Investment 1Yr Below | `TagValue('lipperReturnOnInvestment1yrBelow', '100.0')` |

### LIPPER_ROI_3YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperReturnOnInvestment3yrAbove | DoubleField | Return on Investment 3Yr Above | `TagValue('lipperReturnOnInvestment3yrAbove', '100.0')` |
| lipperReturnOnInvestment3yrBelow | DoubleField | Return on Investment 3Yr Below | `TagValue('lipperReturnOnInvestment3yrBelow', '100.0')` |

### LIPPER_RSQ_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperRSquared1yrAbove | DoubleField | R-Squared 1Yr Above | `TagValue('lipperRSquared1yrAbove', '100.0')` |
| lipperRSquared1yrBelow | DoubleField | R-Squared 1Yr Below | `TagValue('lipperRSquared1yrBelow', '100.0')` |

### LIPPER_RSQ_ADJ_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperAdjustedRSquared1yrAbove | DoubleField | R-Squared Adjusted 1Yr Above | `TagValue('lipperAdjustedRSquared1yrAbove', '100.0')` |
| lipperAdjustedRSquared1yrBelow | DoubleField | R-Squared Adjusted 1Yr Below | `TagValue('lipperAdjustedRSquared1yrBelow', '100.0')` |

### LIPPER_SALES_2_TOTAL_ASSETS

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperSales2TotalAssetsAbove | DoubleField | Sales to Total Assets Above | `TagValue('lipperSales2TotalAssetsAbove', '100.0')` |
| lipperSales2TotalAssetsBelow | DoubleField | Sales to Total Assets Below | `TagValue('lipperSales2TotalAssetsBelow', '100.0')` |

### LIPPER_SALES_GRWTH_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperSalesGrowth1yrAbove | DoubleField | Sales Growth 1Yr Above | `TagValue('lipperSalesGrowth1yrAbove', '100.0')` |
| lipperSalesGrowth1yrBelow | DoubleField | Sales Growth 1Yr Below | `TagValue('lipperSalesGrowth1yrBelow', '100.0')` |

### LIPPER_SALES_GRWTH_3YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperSalesGrowth3yrAbove | DoubleField | Sales Growth 3Yr Above | `TagValue('lipperSalesGrowth3yrAbove', '100.0')` |
| lipperSalesGrowth3yrBelow | DoubleField | Sales Growth 3Yr Below | `TagValue('lipperSalesGrowth3yrBelow', '100.0')` |

### LIPPER_SALES_GRWTH_5YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperSalesGrowth5yrAbove | DoubleField | Sales Growth 5Yr Above | `TagValue('lipperSalesGrowth5yrAbove', '100.0')` |
| lipperSalesGrowth5yrBelow | DoubleField | Sales Growth 5Yr Below | `TagValue('lipperSalesGrowth5yrBelow', '100.0')` |

### LIPPER_SEMI_DEV_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperSemiDeviation1yrAbove | DoubleField | Semi Deviation 1Yr Above | `TagValue('lipperSemiDeviation1yrAbove', '100.0')` |
| lipperSemiDeviation1yrBelow | DoubleField | Semi Deviation 1Yr Below | `TagValue('lipperSemiDeviation1yrBelow', '100.0')` |

### LIPPER_SEMI_VAR_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperSemiVariance1yrAbove | DoubleField | Semi Variance 1Yr Above | `TagValue('lipperSemiVariance1yrAbove', '100.0')` |
| lipperSemiVariance1yrBelow | DoubleField | Semi Variance 1Yr Below | `TagValue('lipperSemiVariance1yrBelow', '100.0')` |

### LIPPER_SHARPE_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperSharpeRatio1yrAbove | DoubleField | Sharpe Ratio 1Yr Above | `TagValue('lipperSharpeRatio1yrAbove', '100.0')` |
| lipperSharpeRatio1yrBelow | DoubleField | Sharpe Ratio 1Yr Below | `TagValue('lipperSharpeRatio1yrBelow', '100.0')` |

### LIPPER_SORTINO_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperSortino1yrAbove | DoubleField | Sortino Ratio 1Yr Above | `TagValue('lipperSortino1yrAbove', '100.0')` |
| lipperSortino1yrBelow | DoubleField | Sortino Ratio 1Yr Below | `TagValue('lipperSortino1yrBelow', '100.0')` |

### LIPPER_SPS_GRWTH_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperSalesPerShareGrowth1yrAbove | DoubleField | Sales Per Share Growth 1 Year Above | `TagValue('lipperSalesPerShareGrowth1yrAbove', '100.0')` |
| lipperSalesPerShareGrowth1yrBelow | DoubleField | Sales Per Share Growth 1 Year Below | `TagValue('lipperSalesPerShareGrowth1yrBelow', '100.0')` |

### LIPPER_SPS_GRWTH_3YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperSalesPerShareGrowth3yrAbove | DoubleField | Sales Per Share Growth 3 Year Above | `TagValue('lipperSalesPerShareGrowth3yrAbove', '100.0')` |
| lipperSalesPerShareGrowth3yrBelow | DoubleField | Sales Per Share Growth 3 Year Below | `TagValue('lipperSalesPerShareGrowth3yrBelow', '100.0')` |

### LIPPER_SPS_GRWTH_3YR_LATEST

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperLatestSalesPerShareGrowth3yrAbove | DoubleField | Sales Per Share Growth 3Yr Latest Above | `TagValue('lipperLatestSalesPerShareGrowth3yrAbove', '100.0')` |
| lipperLatestSalesPerShareGrowth3yrBelow | DoubleField | Sales Per Share Growth 3Yr Latest Below | `TagValue('lipperLatestSalesPerShareGrowth3yrBelow', '100.0')` |

### LIPPER_SPS_GRWTH_ZSCORE

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperLatestSPSGrowthZScoreAbove | DoubleField | SPS Growth Z-Score Latest Above | `TagValue('lipperLatestSPSGrowthZScoreAbove', '100.0')` |
| lipperLatestSPSGrowthZScoreBelow | DoubleField | SPS Growth Z-Score Latest Below | `TagValue('lipperLatestSPSGrowthZScoreBelow', '100.0')` |

### LIPPER_SRRI_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperSRRI1yrAbove | DoubleField | Synthetic Risk and Reward Indicator 1Yr Above | `TagValue('lipperSRRI1yrAbove', '100.0')` |
| lipperSRRI1yrBelow | DoubleField | Synthetic Risk and Reward Indicator 1Yr Below | `TagValue('lipperSRRI1yrBelow', '100.0')` |

### LIPPER_STD_DEV_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperStandardDeviation1yrAbove | DoubleField | Standard Deviation 1Yr Above | `TagValue('lipperStandardDeviation1yrAbove', '100.0')` |
| lipperStandardDeviation1yrBelow | DoubleField | Standard Deviation 1Yr Below | `TagValue('lipperStandardDeviation1yrBelow', '100.0')` |

### LIPPER_TOT_ASSETS_2_TOT_EQ

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperTotalAssets2TotalEquityAbove | DoubleField | Total Assets / Total Equity Above | `TagValue('lipperTotalAssets2TotalEquityAbove', '100.0')` |
| lipperTotalAssets2TotalEquityBelow | DoubleField | Total Assets / Total Equity Below | `TagValue('lipperTotalAssets2TotalEquityBelow', '100.0')` |

### LIPPER_TOT_DEBT_2_TOT_CAP

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperTotalDebt2TotalCapitalAbove | DoubleField | Total Debt / Total Capital Above | `TagValue('lipperTotalDebt2TotalCapitalAbove', '100.0')` |
| lipperTotalDebt2TotalCapitalBelow | DoubleField | Total Debt / Total Capital Below | `TagValue('lipperTotalDebt2TotalCapitalBelow', '100.0')` |

### LIPPER_TOT_DEBT_2_TOT_EQ

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperTotalDebt2TotalEquityAbove | DoubleField | Total Debt / Total Equity Above | `TagValue('lipperTotalDebt2TotalEquityAbove', '100.0')` |
| lipperTotalDebt2TotalEquityBelow | DoubleField | Total Debt / Total Equity Below | `TagValue('lipperTotalDebt2TotalEquityBelow', '100.0')` |

### LIPPER_TOT_EXP_RATIO

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperTotalExpenseRatioAbove | DoubleField | Total Expense Ratio Above | `TagValue('lipperTotalExpenseRatioAbove', '100.0')` |
| lipperTotalExpenseRatioBelow | DoubleField | Total Expense Ratio Below | `TagValue('lipperTotalExpenseRatioBelow', '100.0')` |

### LIPPER_TOT_NET_ASST

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperTotalNetAssetsAbove | DoubleField | Total Net Assets Above | `TagValue('lipperTotalNetAssetsAbove', '100.0')` |
| lipperTotalNetAssetsBelow | DoubleField | Total Net Assets Below | `TagValue('lipperTotalNetAssetsBelow', '100.0')` |

### LIPPER_TRACKING_ERR_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperTrackingError1yrAbove | DoubleField | Tracking Error 1Yr Above | `TagValue('lipperTrackingError1yrAbove', '100.0')` |
| lipperTrackingError1yrBelow | DoubleField | Tracking Error 1Yr Below | `TagValue('lipperTrackingError1yrBelow', '100.0')` |

### LIPPER_TREYNOR_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperTreynorRatio1yrAbove | DoubleField | Treynor Ratio 1Yr Above | `TagValue('lipperTreynorRatio1yrAbove', '100.0')` |
| lipperTreynorRatio1yrBelow | DoubleField | Treynor Ratio 1Yr Below | `TagValue('lipperTreynorRatio1yrBelow', '100.0')` |

### LIPPER_VARIANCE_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperVariance1yrAbove | DoubleField | Variance 1Yr Above | `TagValue('lipperVariance1yrAbove', '100.0')` |
| lipperVariance1yrBelow | DoubleField | Variance 1Yr Below | `TagValue('lipperVariance1yrBelow', '100.0')` |

### LIPPER_VAR_NORMAL_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperValueAtRiskNormal1yrAbove | DoubleField | Value At Risk Normal 1Yr Above | `TagValue('lipperValueAtRiskNormal1yrAbove', '100.0')` |
| lipperValueAtRiskNormal1yrBelow | DoubleField | Value At Risk Normal 1Yr Below | `TagValue('lipperValueAtRiskNormal1yrBelow', '100.0')` |

### LIPPER_VAR_NORMAL_ETL_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperValueAtRiskNormalETL1yrAbove | DoubleField | Value At Risk Normal End Tail Loss 1Yr Above | `TagValue('lipperValueAtRiskNormalETL1yrAbove', '100.0')` |
| lipperValueAtRiskNormalETL1yrBelow | DoubleField | Value At Risk Normal End Tail Loss 1Yr Below | `TagValue('lipperValueAtRiskNormalETL1yrBelow', '100.0')` |

### LIPPER_VAR_QUANTILE_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperValueAtRiskQuantile1yrAbove | DoubleField | Value At Risk Quantile 1Yr Above | `TagValue('lipperValueAtRiskQuantile1yrAbove', '100.0')` |
| lipperValueAtRiskQuantile1yrBelow | DoubleField | Value At Risk Quantile 1Yr Below | `TagValue('lipperValueAtRiskQuantile1yrBelow', '100.0')` |

### LIPPER_VAR_QUANTILE_ETL_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperValueAtRiskQuantileETL1yrAbove | DoubleField | Value At Risk Quantile End Tail Loss 1Yr Above | `TagValue('lipperValueAtRiskQuantileETL1yrAbove', '100.0')` |
| lipperValueAtRiskQuantileETL1yrBelow | DoubleField | Value At Risk Quantile End Tail Loss 1Yr Below | `TagValue('lipperValueAtRiskQuantileETL1yrBelow', '100.0')` |

### LIPPER_WGT_FIN_COMP_ZSCORE

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperWeightedFinalCompositeZScoreAbove | DoubleField | Weighted Final Composite Z-Score Above | `TagValue('lipperWeightedFinalCompositeZScoreAbove', '100.0')` |
| lipperWeightedFinalCompositeZScoreBelow | DoubleField | Weighted Final Composite Z-Score Below | `TagValue('lipperWeightedFinalCompositeZScoreBelow', '100.0')` |

### LIPPER_YIELD_1YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperYieldValueAbove | DoubleField | Yield 1Yr Above | `TagValue('lipperYieldValueAbove', '100.0')` |
| lipperYieldValueBelow | DoubleField | Yield 1Yr Below | `TagValue('lipperYieldValueBelow', '100.0')` |

### LIPPER_YIELD_TO_MATURITY

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperYieldToMaturityAbove | DoubleField | Yield to Maturity Above | `TagValue('lipperYieldToMaturityAbove', '100.0')` |
| lipperYieldToMaturityBelow | DoubleField | Yield to Maturity Below | `TagValue('lipperYieldToMaturityBelow', '100.0')` |

### LIPPER_YIELD_ZSCORE

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lipperDividendYieldZScoreAbove | DoubleField | Dividend Yield Z-Score Latest Above | `TagValue('lipperDividendYieldZScoreAbove', '100.0')` |
| lipperDividendYieldZScoreBelow | DoubleField | Dividend Yield Z-Score Latest Below | `TagValue('lipperDividendYieldZScoreBelow', '100.0')` |

### MACD

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| curMACDAbove | DoubleField | MACD Above | `TagValue('curMACDAbove', '100.0')` |
| curMACDBelow | DoubleField | MACD Below | `TagValue('curMACDBelow', '100.0')` |

### MACD_HISTOGRAM

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| curMACDDistAbove | DoubleField | MACD Histogram Above | `TagValue('curMACDDistAbove', '100.0')` |
| curMACDDistBelow | DoubleField | MACD Histogram Below | `TagValue('curMACDDistBelow', '100.0')` |

### MACD_SIGNAL

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| curMACDSignalAbove | DoubleField | MACD Signal Above | `TagValue('curMACDSignalAbove', '100.0')` |
| curMACDSignalBelow | DoubleField | MACD Signal Below | `TagValue('curMACDSignalBelow', '100.0')` |

### MATDATE

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| maturityDateAbove | DateField | Maturity Date Above | `TagValue('maturityDateAbove', '20240101')` |
| maturityDateBelow | DateField | Maturity Date Below | `TagValue('maturityDateBelow', '20240101')` |

### MF_LDR_CONSIS_RET_SCR_10YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| mfConsistentReturnScore10yrAbove | IntField | Consistent Return Score 10yr Above | `TagValue('mfConsistentReturnScore10yrAbove', '100')` |
| mfConsistentReturnScore10yrBelow | IntField | Consistent Return Score 10yr Below | `TagValue('mfConsistentReturnScore10yrBelow', '100')` |

### MF_LDR_CONSIS_RET_SCR_3YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| mfConsistentReturnScore3yrAbove | IntField | Consistent Return Score 3yr Above | `TagValue('mfConsistentReturnScore3yrAbove', '100')` |
| mfConsistentReturnScore3yrBelow | IntField | Consistent Return Score 3yr Below | `TagValue('mfConsistentReturnScore3yrBelow', '100')` |

### MF_LDR_CONSIS_RET_SCR_5YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| mfConsistentReturnScore5yrAbove | IntField | Consistent Return Score 5yr Above | `TagValue('mfConsistentReturnScore5yrAbove', '100')` |
| mfConsistentReturnScore5yrBelow | IntField | Consistent Return Score 5yr Below | `TagValue('mfConsistentReturnScore5yrBelow', '100')` |

### MF_LDR_CONSIS_RET_SCR_ALL

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| mfConsistentReturnScoreOverallAbove | IntField | Consistent Return Score Overall Above | `TagValue('mfConsistentReturnScoreOverallAbove', '100')` |
| mfConsistentReturnScoreOverallBelow | IntField | Consistent Return Score Overall Below | `TagValue('mfConsistentReturnScoreOverallBelow', '100')` |

### MF_LDR_EXP_SCR_10YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| mfExpenseScore10yrAbove | IntField | Expense Score 10yr Above | `TagValue('mfExpenseScore10yrAbove', '100')` |
| mfExpenseScore10yrBelow | IntField | Expense Score 10yr Below | `TagValue('mfExpenseScore10yrBelow', '100')` |

### MF_LDR_EXP_SCR_3YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| mfExpenseScore3yrAbove | IntField | Expense Score 3yr Above | `TagValue('mfExpenseScore3yrAbove', '100')` |
| mfExpenseScore3yrBelow | IntField | Expense Score 3yr Below | `TagValue('mfExpenseScore3yrBelow', '100')` |

### MF_LDR_EXP_SCR_5YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| mfExpenseScore5yrAbove | IntField | Expense Score 5yr Above | `TagValue('mfExpenseScore5yrAbove', '100')` |
| mfExpenseScore5yrBelow | IntField | Expense Score 5yr Below | `TagValue('mfExpenseScore5yrBelow', '100')` |

### MF_LDR_EXP_SCR_ALL

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| mfExpenseScoreOverallAbove | IntField | Expense Score Overall Above | `TagValue('mfExpenseScoreOverallAbove', '100')` |
| mfExpenseScoreOverallBelow | IntField | Expense Score Overall Below | `TagValue('mfExpenseScoreOverallBelow', '100')` |

### MF_LDR_PRESERV_SCR_10YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| mfPreservationScore10yrAbove | IntField | Preservation Score 10yr Above | `TagValue('mfPreservationScore10yrAbove', '100')` |
| mfPreservationScore10yrBelow | IntField | Preservation Score 10yr Below | `TagValue('mfPreservationScore10yrBelow', '100')` |

### MF_LDR_PRESERV_SCR_3YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| mfPreservationScore3yrAbove | IntField | Preservation Score 3yr Above | `TagValue('mfPreservationScore3yrAbove', '100')` |
| mfPreservationScore3yrBelow | IntField | Preservation Score 3yr Below | `TagValue('mfPreservationScore3yrBelow', '100')` |

### MF_LDR_PRESERV_SCR_5YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| mfPreservationScore5yrAbove | IntField | Preservation Score 5yr Above | `TagValue('mfPreservationScore5yrAbove', '100')` |
| mfPreservationScore5yrBelow | IntField | Preservation Score 5yr Below | `TagValue('mfPreservationScore5yrBelow', '100')` |

### MF_LDR_PRESERV_SCR_ALL

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| mfPreservationScoreOverallAbove | IntField | Preservation Score Overall Above | `TagValue('mfPreservationScoreOverallAbove', '100')` |
| mfPreservationScoreOverallBelow | IntField | Preservation Score Overall Below | `TagValue('mfPreservationScoreOverallBelow', '100')` |

### MF_LDR_TAX_EFF_SCR_10YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| mfTaxEfficiencyScore10yrAbove | IntField | Tax Efficiency Score 10yr Above | `TagValue('mfTaxEfficiencyScore10yrAbove', '100')` |
| mfTaxEfficiencyScore10yrBelow | IntField | Tax Efficiency Score 10yr Below | `TagValue('mfTaxEfficiencyScore10yrBelow', '100')` |

### MF_LDR_TAX_EFF_SCR_3YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| mfTaxEfficiencyScore3yrAbove | IntField | Tax Efficiency Score 3yr Above | `TagValue('mfTaxEfficiencyScore3yrAbove', '100')` |
| mfTaxEfficiencyScore3yrBelow | IntField | Tax Efficiency Score 3yr Below | `TagValue('mfTaxEfficiencyScore3yrBelow', '100')` |

### MF_LDR_TAX_EFF_SCR_5YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| mfTaxEfficiencyScore5yrAbove | IntField | Tax Efficiency Score 5yr Above | `TagValue('mfTaxEfficiencyScore5yrAbove', '100')` |
| mfTaxEfficiencyScore5yrBelow | IntField | Tax Efficiency Score 5yr Below | `TagValue('mfTaxEfficiencyScore5yrBelow', '100')` |

### MF_LDR_TAX_EFF_SCR_ALL

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| mfTaxEfficiencyScoreOverallAbove | IntField | Tax Efficiency Score Overall Above | `TagValue('mfTaxEfficiencyScoreOverallAbove', '100')` |
| mfTaxEfficiencyScoreOverallBelow | IntField | Tax Efficiency Score Overall Below | `TagValue('mfTaxEfficiencyScoreOverallBelow', '100')` |

### MF_LDR_TOT_RET_SCR_10YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| mfTotalReturnScore10yrAbove | IntField | Total Return Score 10yr Above | `TagValue('mfTotalReturnScore10yrAbove', '100')` |
| mfTotalReturnScore10yrBelow | IntField | Total Return Score 10yr Below | `TagValue('mfTotalReturnScore10yrBelow', '100')` |

### MF_LDR_TOT_RET_SCR_3YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| mfTotalReturnScore3yrAbove | IntField | Total Return Score 3yr Above | `TagValue('mfTotalReturnScore3yrAbove', '100')` |
| mfTotalReturnScore3yrBelow | IntField | Total Return Score 3yr Below | `TagValue('mfTotalReturnScore3yrBelow', '100')` |

### MF_LDR_TOT_RET_SCR_5YR

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| mfTotalReturnScore5yrAbove | IntField | Total Return Score 5yr Above | `TagValue('mfTotalReturnScore5yrAbove', '100')` |
| mfTotalReturnScore5yrBelow | IntField | Total Return Score 5yr Below | `TagValue('mfTotalReturnScore5yrBelow', '100')` |

### MF_LDR_TOT_RET_SCR_ALL

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| mfTotalReturnScoreOverallAbove | IntField | Total Return Score Overall Above | `TagValue('mfTotalReturnScoreOverallAbove', '100')` |
| mfTotalReturnScoreOverallBelow | IntField | Total Return Score Overall Below | `TagValue('mfTotalReturnScoreOverallBelow', '100')` |

### MF_PRICE_CHG_VAL

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| mfPriceChangeValueAbove | DoubleField | Price Change Value Above | `TagValue('mfPriceChangeValueAbove', '25.50')` |
| mfPriceChangeValueBelow | DoubleField | Price Change Value Below | `TagValue('mfPriceChangeValueBelow', '25.50')` |

### MKTCAP

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| marketCapAbove1e6 | DoubleField | Capitalization Above | `TagValue('marketCapAbove1e6', '100.0')` |
| marketCapBelow1e6 | DoubleField | Capitalization Below | `TagValue('marketCapBelow1e6', '100.0')` |

### MOODY

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| moodyRatingAbove | ComboField | Moody Rating Above | `TagValue('moodyRatingAbove', 'AAA')` |

Allowed values for moodyRatingAbove:
- `AAA`: AAA
- `AA1`: AA1
- `AA2`: AA2
- `AA3`: AA3
- `A1`: A1
- `A2`: A2
- `A3`: A3
- `BAA1`: BAA1
- `BAA2`: BAA2
- `BAA3`: BAA3
- `BA1`: BA1
- `BA2`: BA2
- `BA3`: BA3
- `B1`: B1
- `B2`: B2
- `B3`: B3
- `CAA1`: CAA1
- `CAA2`: CAA2
- `CAA3`: CAA3
- `CA`: CA
- `C`: C
- `NR`: NR

| moodyRatingBelow | ComboField | Moody Rating Below | `TagValue('moodyRatingBelow', 'AAA')` |

Allowed values for moodyRatingBelow:
- `AAA`: AAA
- `AA1`: AA1
- `AA2`: AA2
- `AA3`: AA3
- `A1`: A1
- `A2`: A2
- `A3`: A3
- `BAA1`: BAA1
- `BAA2`: BAA2
- `BAA3`: BAA3
- `BA1`: BA1
- `BA2`: BA2
- `BA3`: BA3
- `B1`: B1
- `B2`: B2
- `B3`: B3
- `CAA1`: CAA1
- `CAA2`: CAA2
- `CAA3`: CAA3
- `CA`: CA
- `C`: C
- `NR`: NR


### NET_PROFIT_MARGIN_TTM

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| netProfitMarginTTMAbove | DoubleField | Net Profit Margin(%) (Refinitiv) Above | `TagValue('netProfitMarginTTMAbove', '100.0')` |
| netProfitMarginTTMBelow | DoubleField | Net Profit Margin(%) (Refinitiv) Below | `TagValue('netProfitMarginTTMBelow', '100.0')` |

### NEXTDIVAMOUNT

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| dividendNextAmountAbove | DoubleField | Next Dividend per Share Above | `TagValue('dividendNextAmountAbove', '100.0')` |
| dividendNextAmountBelow | DoubleField | Next Dividend per Share Below | `TagValue('dividendNextAmountBelow', '100.0')` |

### NEXTDIVDATE

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| dividendNextDateAbove | DateField | Next Dividend Date Above | `TagValue('dividendNextDateAbove', '20240101')` |
| dividendNextDateBelow | DateField | Next Dividend Date Below | `TagValue('dividendNextDateBelow', '20240101')` |

### NUMPRICETARGETS

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| numPriceTargetsAbove | IntField | # of Analyst Price Targets Above | `TagValue('numPriceTargetsAbove', '100')` |
| numPriceTargetsBelow | IntField | # of Analyst Price Targets Below | `TagValue('numPriceTargetsBelow', '100')` |

### NUMRATINGS

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| numRatingsAbove | IntField | # of Analyst Ratings Above | `TagValue('numRatingsAbove', '100')` |
| numRatingsBelow | IntField | # of Analyst Ratings Below | `TagValue('numRatingsBelow', '100')` |

### NUMSHARESINSIDER

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| ihNumSharesInsiderAbove | DoubleField | # Shares held by Insider Above | `TagValue('ihNumSharesInsiderAbove', '100.0')` |
| ihNumSharesInsiderBelow | DoubleField | # Shares held by Insider Below | `TagValue('ihNumSharesInsiderBelow', '100.0')` |

### NUMSHARESINSTITUTIONAL

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| iiNumSharesInstitutionalAbove | DoubleField | # Shares held by Institutions Above | `TagValue('iiNumSharesInstitutionalAbove', '100.0')` |
| iiNumSharesInstitutionalBelow | DoubleField | # Shares held by Institutions Below | `TagValue('iiNumSharesInstitutionalBelow', '100.0')` |

### OPENGAPPERC

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| openGapPercAbove | DoubleField | Gap (%) Above | `TagValue('openGapPercAbove', '5.5')` |
| openGapPercBelow | DoubleField | Gap (%) Below | `TagValue('openGapPercBelow', '5.5')` |

### OPERATING_MARGIN_TTM

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| operatingMarginTTMAbove | DoubleField | Operating Margin (Refinitiv) Above | `TagValue('operatingMarginTTMAbove', '100.0')` |
| operatingMarginTTMBelow | DoubleField | Operating Margin (Refinitiv) Below | `TagValue('operatingMarginTTMBelow', '100.0')` |

### OPTOI

Category: Options

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| optOpenInterestAbove | DoubleField | Opt Open Interest Above | `TagValue('optOpenInterestAbove', '100.0')` |
| optOpenInterestBelow | DoubleField | Opt Open Interest Below | `TagValue('optOpenInterestBelow', '100.0')` |

### OPTVOLUME

Category: Options

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| optVolumeAbove | DoubleField | Option Volume Above | `TagValue('optVolumeAbove', '100.0')` |
| optVolumeBelow | DoubleField | Option Volume Below | `TagValue('optVolumeBelow', '100.0')` |

### OPTVOLUMEPCRATIO

Category: Options

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| optVolumePCRatioAbove | DoubleField | Opt Volume P/C Ratio Above | `TagValue('optVolumePCRatioAbove', '100.0')` |
| optVolumePCRatioBelow | DoubleField | Opt Volume P/C Ratio Below | `TagValue('optVolumePCRatioBelow', '100.0')` |

### PAYOUT_RATIO_TTM

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| payoutRatioTTMAbove | DoubleField | Payout Ratio TTM (Refinitiv) Above | `TagValue('payoutRatioTTMAbove', '100.0')` |
| payoutRatioTTMBelow | DoubleField | Payout Ratio TTM (Refinitiv) Below | `TagValue('payoutRatioTTMBelow', '100.0')` |

### PERATIO

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| minPeRatio | DoubleField | P/E Ratio Above (Refinitiv) | `TagValue('minPeRatio', '100.0')` |
| maxPeRatio | DoubleField | P/E Ratio Below (Refinitiv) | `TagValue('maxPeRatio', '100.0')` |

### PPO

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| curPPOAbove | DoubleField | PPO Above | `TagValue('curPPOAbove', '100.0')` |
| curPPOBelow | DoubleField | PPO Below | `TagValue('curPPOBelow', '100.0')` |

### PPO_HISTOGRAM

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| curPPODistAbove | DoubleField | PPO Histogram Above | `TagValue('curPPODistAbove', '100.0')` |
| curPPODistBelow | DoubleField | PPO Histogram Below | `TagValue('curPPODistBelow', '100.0')` |

### PPO_SIGNAL

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| curPPOSignalAbove | DoubleField | PPO Signal Above | `TagValue('curPPOSignalAbove', '100.0')` |
| curPPOSignalBelow | DoubleField | PPO Signal Below | `TagValue('curPPOSignalBelow', '100.0')` |

### PRICE

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| priceAbove | DoubleField | Price Above | `TagValue('priceAbove', '25.50')` |
| priceBelow | DoubleField | Price Below | `TagValue('priceBelow', '25.50')` |

### PRICE2BK

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| minPrice2Bk | DoubleField | Price/Book Ratio Above (Refinitiv) | `TagValue('minPrice2Bk', '25.50')` |
| maxPrice2Bk | DoubleField | Price/Book Ratio Below (Refinitiv) | `TagValue('maxPrice2Bk', '25.50')` |

### PRICE2TANBK

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| minPrice2TanBk | DoubleField | Price/Tang. Book Above (Refinitiv) | `TagValue('minPrice2TanBk', '25.50')` |
| maxPrice2TanBk | DoubleField | Price/Tang. Book Below (Refinitiv) | `TagValue('maxPrice2TanBk', '25.50')` |

### PRICERANGE

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| priceRangeAbove | DoubleField | Price Range Above | `TagValue('priceRangeAbove', '25.50')` |
| priceRangeBelow | DoubleField | Price Range Below | `TagValue('priceRangeBelow', '25.50')` |

### PRICE_2_CASH_TTM

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| price2CashTTMAbove | DoubleField | Price To Cash Per Share (Refinitiv) Above | `TagValue('price2CashTTMAbove', '25.50')` |
| price2CashTTMBelow | DoubleField | Price To Cash Per Share (Refinitiv) Below | `TagValue('price2CashTTMBelow', '25.50')` |

### PRICE_USD

Category: Prices

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| usdPriceAbove | DoubleField | Price ($) Above | `TagValue('usdPriceAbove', '25.50')` |
| usdPriceBelow | DoubleField | Price ($) Below | `TagValue('usdPriceBelow', '25.50')` |

### PRICE_VS_EMA_100

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lastVsEMAChangeRatio100Above | DoubleField | Price vs EMA(100) Change(%) Above | `TagValue('lastVsEMAChangeRatio100Above', '100.0')` |
| lastVsEMAChangeRatio100Below | DoubleField | Price vs EMA(100) Change(%) Below | `TagValue('lastVsEMAChangeRatio100Below', '100.0')` |

### PRICE_VS_EMA_20

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lastVsEMAChangeRatio20Above | DoubleField | Price vs EMA(20) Change(%) Above | `TagValue('lastVsEMAChangeRatio20Above', '100.0')` |
| lastVsEMAChangeRatio20Below | DoubleField | Price vs EMA(20) Change(%) Below | `TagValue('lastVsEMAChangeRatio20Below', '100.0')` |

### PRICE_VS_EMA_200

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lastVsEMAChangeRatio200Above | DoubleField | Price vs EMA(200) Change(%) Above | `TagValue('lastVsEMAChangeRatio200Above', '100.0')` |
| lastVsEMAChangeRatio200Below | DoubleField | Price vs EMA(200) Change(%) Below | `TagValue('lastVsEMAChangeRatio200Below', '100.0')` |

### PRICE_VS_EMA_50

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| lastVsEMAChangeRatio50Above | DoubleField | Price vs EMA(50) Change(%) Above | `TagValue('lastVsEMAChangeRatio50Above', '100.0')` |
| lastVsEMAChangeRatio50Below | DoubleField | Price vs EMA(50) Change(%) Below | `TagValue('lastVsEMAChangeRatio50Below', '100.0')` |

### QUICKRATIO

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| minQuickRatio | DoubleField | Quick Ratio Above (Refinitiv) | `TagValue('minQuickRatio', '100.0')` |
| maxQuickRatio | DoubleField | Quick Ratio Below (Refinitiv) | `TagValue('maxQuickRatio', '100.0')` |

### RCGITENDDATE

Category: Recognia

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| rcgIntermediateTermTechnicalDateAbove | DateField | Intermediate Term Technical Event Date (Recognia) Above | `TagValue('rcgIntermediateTermTechnicalDateAbove', '20240101')` |
| rcgIntermediateTermTechnicalDateBelow | DateField | Intermediate Term Technical Event Date (Recognia) Below | `TagValue('rcgIntermediateTermTechnicalDateBelow', '20240101')` |

### RCGITIVALUE

Category: Recognia

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| rcgIntermediateTermEventScoreAbove | DoubleField | Intermediate Term Event Score (Recognia) Above | `TagValue('rcgIntermediateTermEventScoreAbove', '100.0')` |
| rcgIntermediateTermEventScoreBelow | DoubleField | Intermediate Term Event Score (Recognia) Below | `TagValue('rcgIntermediateTermEventScoreBelow', '100.0')` |

### RCGLTENDDATE

Category: Recognia

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| rcgLongTermTechnicalDateAbove | DateField | Long Term Technical Event Date (Recognia) Above | `TagValue('rcgLongTermTechnicalDateAbove', '20240101')` |
| rcgLongTermTechnicalDateBelow | DateField | Long Term Technical Event Date (Recognia) Below | `TagValue('rcgLongTermTechnicalDateBelow', '20240101')` |

### RCGLTIVALUE

Category: Recognia

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| rcgLongTermEventScoreAbove | DoubleField | Long Term Event Score (Recognia) Above | `TagValue('rcgLongTermEventScoreAbove', '100.0')` |
| rcgLongTermEventScoreBelow | DoubleField | Long Term Event Score (Recognia) Below | `TagValue('rcgLongTermEventScoreBelow', '100.0')` |

### RCGSTENDDATE

Category: Recognia

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| rcgShortTermTechnicalDateAbove | DateField | Short Term Technical Event Date (Recognia) Above | `TagValue('rcgShortTermTechnicalDateAbove', '20240101')` |
| rcgShortTermTechnicalDateBelow | DateField | Short Term Technical Event Date (Recognia) Below | `TagValue('rcgShortTermTechnicalDateBelow', '20240101')` |

### RCGSTIVALUE

Category: Recognia

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| rcgShortTermEventScoreAbove | DoubleField | Short Term Event Score (Recognia) Above | `TagValue('rcgShortTermEventScoreAbove', '100.0')` |
| rcgShortTermEventScoreBelow | DoubleField | Short Term Event Score (Recognia) Below | `TagValue('rcgShortTermEventScoreBelow', '100.0')` |

### REGIMBALANCE

Category: Auction

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| regulatoryImbalanceAbove | DoubleField | Regulatory Imbalance Above | `TagValue('regulatoryImbalanceAbove', '100.0')` |
| regulatoryImbalanceBelow | DoubleField | Regulatory Imbalance Below | `TagValue('regulatoryImbalanceBelow', '100.0')` |

### REGIMBALANCEADVRATIOPERC

Category: Auction

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| displayRegulatoryImbAdvRatioAbove | DoubleField | Regulatory Imbalance Adv Ratio (%) Above | `TagValue('displayRegulatoryImbAdvRatioAbove', '100.0')` |
| displayRegulatoryImbAdvRatioBelow | DoubleField | Regulatory Imbalance Adv Ratio (%) Below | `TagValue('displayRegulatoryImbAdvRatioBelow', '100.0')` |

### RETEQUITY

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| minRetnOnEq | DoubleField | Return On Equity Above (Refinitiv) | `TagValue('minRetnOnEq', '100.0')` |
| maxRetnOnEq | DoubleField | Return On Equity Below (Refinitiv) | `TagValue('maxRetnOnEq', '100.0')` |

### RETURN_ON_INVESTMENT_TTM

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| returnOnInvestmentTTMAbove | DoubleField | Return On Investment (Refinitiv) Above | `TagValue('returnOnInvestmentTTMAbove', '100.0')` |
| returnOnInvestmentTTMBelow | DoubleField | Return On Investment (Refinitiv) Below | `TagValue('returnOnInvestmentTTMBelow', '100.0')` |

### REV_CHANGE

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| revChangeAbove | DoubleField | Rev Change(%) (Refinitiv) Above | `TagValue('revChangeAbove', '100.0')` |
| revChangeBelow | DoubleField | Rev Change(%) (Refinitiv) Below | `TagValue('revChangeBelow', '100.0')` |

### REV_GROWTH_RATE_5Y

Category: Fundamentals

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| revGrowthRate5YAbove | DoubleField | Revenue Growth Rate 5Y (Refinitiv) Above | `TagValue('revGrowthRate5YAbove', '100.0')` |
| revGrowthRate5YBelow | DoubleField | Revenue Growth Rate 5Y (Refinitiv) Below | `TagValue('revGrowthRate5YBelow', '100.0')` |

### SCHANGE

Category: Social Sentiment

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| socialSentimentScoreChangeAbove | DoubleField | Social Sentiment Score Change Above | `TagValue('socialSentimentScoreChangeAbove', '100.0')` |
| socialSentimentScoreChangeBelow | DoubleField | Social Sentiment Score Change Below | `TagValue('socialSentimentScoreChangeBelow', '100.0')` |

### SHORTABLESHARES

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| sharesAvailableManyAbove | IntField | Shares Available Above | `TagValue('sharesAvailableManyAbove', '100')` |
| sharesAvailableManyBelow | IntField | Shares Available Below | `TagValue('sharesAvailableManyBelow', '100')` |

### SP

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| spRatingAbove | ComboField | S&P Rating Above | `TagValue('spRatingAbove', 'AAA')` |

Allowed values for spRatingAbove:
- `AAA`: AAA
- `AA+`: AA+
- `AA`: AA
- `AA-`: AA-
- `A+`: A+
- `A`: A
- `A-`: A-
- `BBB+`: BBB+
- `BBB`: BBB
- `BBB-`: BBB-
- `BB+`: BB+
- `BB`: BB
- `BB-`: BB-
- `B+`: B+
- `B`: B
- `B-`: B-
- `CCC+`: CCC+
- `CCC`: CCC
- `CCC-`: CCC-
- `CC+`: CC+
- `CC`: CC
- `CC-`: CC-
- `C+`: C+
- `C`: C
- `C-`: C-
- `D`: D
- `NR`: NR

| spRatingBelow | ComboField | S&P Rating Below | `TagValue('spRatingBelow', 'AAA')` |

Allowed values for spRatingBelow:
- `AAA`: AAA
- `AA+`: AA+
- `AA`: AA
- `AA-`: AA-
- `A+`: A+
- `A`: A
- `A-`: A-
- `BBB+`: BBB+
- `BBB`: BBB
- `BBB-`: BBB-
- `BB+`: BB+
- `BB`: BB
- `BB-`: BB-
- `B+`: B+
- `B`: B
- `B-`: B-
- `CCC+`: CCC+
- `CCC`: CCC
- `CCC-`: CCC-
- `CC+`: CC+
- `CC`: CC
- `CC-`: CC-
- `C+`: C+
- `C`: C
- `C-`: C-
- `D`: D
- `NR`: NR


### SSCORE

Category: Social Sentiment

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| socialSentimentScoreAbove | DoubleField | Social Sentiment Score Above | `TagValue('socialSentimentScoreAbove', '100.0')` |
| socialSentimentScoreBelow | DoubleField | Social Sentiment Score Below | `TagValue('socialSentimentScoreBelow', '100.0')` |

### STVOLUME_10MIN

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| stVolume10minAbove | IntField | 10min Volume Above | `TagValue('stVolume10minAbove', '1000000')` |
| stVolume10minBelow | IntField | 10min Volume Below | `TagValue('stVolume10minBelow', '1000000')` |

### STVOLUME_3MIN

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| stVolume3minAbove | IntField | 3min Volume Above | `TagValue('stVolume3minAbove', '1000000')` |
| stVolume3minBelow | IntField | 3min Volume Below | `TagValue('stVolume3minBelow', '1000000')` |

### STVOLUME_5MIN

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| stVolume5minAbove | IntField | 5min Volume Above | `TagValue('stVolume5minAbove', '1000000')` |
| stVolume5minBelow | IntField | 5min Volume Below | `TagValue('stVolume5minBelow', '1000000')` |

### SVCHANGE

Category: Social Sentiment

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| tweetVolumeScoreChangeAbove | DoubleField | Tweet Volume Score Change Above | `TagValue('tweetVolumeScoreChangeAbove', '100.0')` |
| tweetVolumeScoreChangeBelow | DoubleField | Tweet Volume Score Change Below | `TagValue('tweetVolumeScoreChangeBelow', '100.0')` |

### SVSCORE

Category: Social Sentiment

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| tweetVolumeScoreAbove | DoubleField | Tweet Volume Score Above | `TagValue('tweetVolumeScoreAbove', '100.0')` |
| tweetVolumeScoreBelow | DoubleField | Tweet Volume Score Below | `TagValue('tweetVolumeScoreBelow', '100.0')` |

### TRADECOUNT

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| tradeCountAbove | IntField | Trade Count Above | `TagValue('tradeCountAbove', '100')` |
| tradeCountBelow | IntField | Trade Count Below | `TagValue('tradeCountBelow', '100')` |

### TRADERATE

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| tradeRateAbove | DoubleField | Trade Rate Above | `TagValue('tradeRateAbove', '100.0')` |
| tradeRateBelow | DoubleField | Trade Rate Below | `TagValue('tradeRateBelow', '100.0')` |

### UTILIZATION

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| utilizationAbove | DoubleField | Utilization Above | `TagValue('utilizationAbove', '100.0')` |
| utilizationBelow | DoubleField | Utilization Below | `TagValue('utilizationBelow', '100.0')` |

### VOLUMERATE

Category: General

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| volumeRateAbove | DoubleField | Volume Rate Above | `TagValue('volumeRateAbove', '100.0')` |
| volumeRateBelow | DoubleField | Volume Rate Below | `TagValue('volumeRateBelow', '100.0')` |

### VOLUME_USD

Category: High/Low/Volume

| Filter Code | Field Type | Description | Example Usage |
|-------------|------------|-------------|---------------|
| usdVolumeAbove | DoubleField | Volume ($) Above | `TagValue('usdVolumeAbove', '100.0')` |
| usdVolumeBelow | DoubleField | Volume ($) Below | `TagValue('usdVolumeBelow', '100.0')` |

## Available Location Codes

Use these codes with the `locationCode` parameter:

| Location Code | Usage Example |
|--------------|----------------|
| BOND.AGNCY.US | `locationCode='BOND.AGNCY.US'` |
| BOND.CD.US | `locationCode='BOND.CD.US'` |
| BOND.EU.EBS | `locationCode='BOND.EU.EBS'` |
| BOND.EU.EURONEXT | `locationCode='BOND.EU.EURONEXT'` |
| BOND.GOVT.EU.EBS | `locationCode='BOND.GOVT.EU.EBS'` |
| BOND.GOVT.EU.EURONEXT | `locationCode='BOND.GOVT.EU.EURONEXT'` |
| BOND.GOVT.HK.SEHK | `locationCode='BOND.GOVT.HK.SEHK'` |
| BOND.GOVT.NON-US | `locationCode='BOND.GOVT.NON-US'` |
| BOND.GOVT.US | `locationCode='BOND.GOVT.US'` |
| BOND.GOVT.US.NON-US | `locationCode='BOND.GOVT.US.NON-US'` |
| BOND.MUNI.US | `locationCode='BOND.MUNI.US'` |
| BOND.US | `locationCode='BOND.US'` |
| BOND.WW | `locationCode='BOND.WW'` |
| ETF.EQ.ARCA | `locationCode='ETF.EQ.ARCA'` |
| ETF.EQ.BATS | `locationCode='ETF.EQ.BATS'` |
| ETF.EQ.NASDAQ.NMS | `locationCode='ETF.EQ.NASDAQ.NMS'` |
| ETF.EQ.US | `locationCode='ETF.EQ.US'` |
| ETF.EQ.US.MAJOR | `locationCode='ETF.EQ.US.MAJOR'` |
| ETF.FI.ARCA | `locationCode='ETF.FI.ARCA'` |
| ETF.FI.BATS | `locationCode='ETF.FI.BATS'` |
| ETF.FI.NASDAQ.NMS | `locationCode='ETF.FI.NASDAQ.NMS'` |
| ETF.FI.US | `locationCode='ETF.FI.US'` |
| ETF.FI.US.MAJOR | `locationCode='ETF.FI.US.MAJOR'` |
| FUND.ALL | `locationCode='FUND.ALL'` |
| FUND.NON-US | `locationCode='FUND.NON-US'` |
| FUND.US | `locationCode='FUND.US'` |
| FUT.CBOT | `locationCode='FUT.CBOT'` |
| FUT.CFE | `locationCode='FUT.CFE'` |
| FUT.CME | `locationCode='FUT.CME'` |
| FUT.COMEX | `locationCode='FUT.COMEX'` |
| FUT.ENDEX | `locationCode='FUT.ENDEX'` |
| FUT.EU | `locationCode='FUT.EU'` |
| FUT.EU.BELFOX | `locationCode='FUT.EU.BELFOX'` |
| FUT.EU.CEDX | `locationCode='FUT.EU.CEDX'` |
| FUT.EU.EUREX | `locationCode='FUT.EU.EUREX'` |
| FUT.EU.FTA | `locationCode='FUT.EU.FTA'` |
| FUT.EU.ICEEU | `locationCode='FUT.EU.ICEEU'` |
| FUT.EU.IDEM | `locationCode='FUT.EU.IDEM'` |
| FUT.EU.LMEOTC | `locationCode='FUT.EU.LMEOTC'` |
| FUT.EU.MEFFRV | `locationCode='FUT.EU.MEFFRV'` |
| FUT.EU.MONEP | `locationCode='FUT.EU.MONEP'` |
| FUT.EU.OMS | `locationCode='FUT.EU.OMS'` |
| FUT.EU.UK | `locationCode='FUT.EU.UK'` |
| FUT.HK | `locationCode='FUT.HK'` |
| FUT.HK.BURSAMY | `locationCode='FUT.HK.BURSAMY'` |
| FUT.HK.HKFE | `locationCode='FUT.HK.HKFE'` |
| FUT.HK.KSE | `locationCode='FUT.HK.KSE'` |
| FUT.HK.NSE | `locationCode='FUT.HK.NSE'` |
| FUT.HK.OSE_JPN | `locationCode='FUT.HK.OSE_JPN'` |
| FUT.HK.SGX | `locationCode='FUT.HK.SGX'` |
| FUT.HK.SNFE | `locationCode='FUT.HK.SNFE'` |
| FUT.ICECRYPTO | `locationCode='FUT.ICECRYPTO'` |
| FUT.IPE | `locationCode='FUT.IPE'` |
| FUT.NA | `locationCode='FUT.NA'` |
| FUT.NA.CDE | `locationCode='FUT.NA.CDE'` |
| FUT.NA.MEXDER | `locationCode='FUT.NA.MEXDER'` |
| FUT.NYBOT | `locationCode='FUT.NYBOT'` |
| FUT.NYMEX | `locationCode='FUT.NYMEX'` |
| FUT.NYSELIFFE | `locationCode='FUT.NYSELIFFE'` |
| FUT.US | `locationCode='FUT.US'` |
| IND.EU | `locationCode='IND.EU'` |
| IND.EU.BELFOX | `locationCode='IND.EU.BELFOX'` |
| IND.EU.EUREX | `locationCode='IND.EU.EUREX'` |
| IND.EU.FTA | `locationCode='IND.EU.FTA'` |
| IND.EU.ICEEU | `locationCode='IND.EU.ICEEU'` |
| IND.EU.MONEP | `locationCode='IND.EU.MONEP'` |
| IND.HK | `locationCode='IND.HK'` |
| IND.HK.HKFE | `locationCode='IND.HK.HKFE'` |
| IND.HK.KSE | `locationCode='IND.HK.KSE'` |
| IND.HK.NSE | `locationCode='IND.HK.NSE'` |
| IND.HK.OSE_JPN | `locationCode='IND.HK.OSE_JPN'` |
| IND.HK.SGX | `locationCode='IND.HK.SGX'` |
| IND.HK.SNFE | `locationCode='IND.HK.SNFE'` |
| IND.US | `locationCode='IND.US'` |
| NATCOMB | `locationCode='NATCOMB'` |
| NATCOMB.CME | `locationCode='NATCOMB.CME'` |
| NATCOMB.OPT.AMEX | `locationCode='NATCOMB.OPT.AMEX'` |
| NATCOMB.OPT.CBOE | `locationCode='NATCOMB.OPT.CBOE'` |
| NATCOMB.OPT.ISE | `locationCode='NATCOMB.OPT.ISE'` |
| NATCOMB.OPT.PHLX | `locationCode='NATCOMB.OPT.PHLX'` |
| NATCOMB.OPT.PSE | `locationCode='NATCOMB.OPT.PSE'` |
| NATCOMB.OPT.US | `locationCode='NATCOMB.OPT.US'` |
| SLB.PREBORROW | `locationCode='SLB.PREBORROW'` |
| SSF.EU | `locationCode='SSF.EU'` |
| SSF.EU.EUREX | `locationCode='SSF.EU.EUREX'` |
| SSF.EU.ICEEU | `locationCode='SSF.EU.ICEEU'` |
| SSF.EU.IDEM | `locationCode='SSF.EU.IDEM'` |
| SSF.EU.MEFFRV | `locationCode='SSF.EU.MEFFRV'` |
| SSF.EU.OMS | `locationCode='SSF.EU.OMS'` |
| SSF.HK | `locationCode='SSF.HK'` |
| SSF.HK.HKFE | `locationCode='SSF.HK.HKFE'` |
| SSF.HK.KSE | `locationCode='SSF.HK.KSE'` |
| SSF.HK.NSE | `locationCode='SSF.HK.NSE'` |
| SSF.HK.SGX | `locationCode='SSF.HK.SGX'` |
| SSF.NA | `locationCode='SSF.NA'` |
| SSF.NA.MEXDER | `locationCode='SSF.NA.MEXDER'` |
| STK.AMEX | `locationCode='STK.AMEX'` |
| STK.ARCA | `locationCode='STK.ARCA'` |
| STK.BATS | `locationCode='STK.BATS'` |
| STK.EU | `locationCode='STK.EU'` |
| STK.EU.AEB | `locationCode='STK.EU.AEB'` |
| STK.EU.BM | `locationCode='STK.EU.BM'` |
| STK.EU.BVL | `locationCode='STK.EU.BVL'` |
| STK.EU.BVME | `locationCode='STK.EU.BVME'` |
| STK.EU.CPH | `locationCode='STK.EU.CPH'` |
| STK.EU.EBS | `locationCode='STK.EU.EBS'` |
| STK.EU.HEX | `locationCode='STK.EU.HEX'` |
| STK.EU.IBIS | `locationCode='STK.EU.IBIS'` |
| STK.EU.IBIS-ETF | `locationCode='STK.EU.IBIS-ETF'` |
| STK.EU.IBIS-EUSTARS | `locationCode='STK.EU.IBIS-EUSTARS'` |
| STK.EU.IBIS-NEWX | `locationCode='STK.EU.IBIS-NEWX'` |
| STK.EU.IBIS-USSTARS | `locationCode='STK.EU.IBIS-USSTARS'` |
| STK.EU.IBIS-XETRA | `locationCode='STK.EU.IBIS-XETRA'` |
| STK.EU.LSE | `locationCode='STK.EU.LSE'` |
| STK.EU.MOEX | `locationCode='STK.EU.MOEX'` |
| STK.EU.OSE | `locationCode='STK.EU.OSE'` |
| STK.EU.OTHER | `locationCode='STK.EU.OTHER'` |
| STK.EU.PRA | `locationCode='STK.EU.PRA'` |
| STK.EU.SBF | `locationCode='STK.EU.SBF'` |
| STK.EU.SFB | `locationCode='STK.EU.SFB'` |
| STK.EU.VSE | `locationCode='STK.EU.VSE'` |
| STK.HK | `locationCode='STK.HK'` |
| STK.HK.ASX | `locationCode='STK.HK.ASX'` |
| STK.HK.NSE | `locationCode='STK.HK.NSE'` |
| STK.HK.SEHK | `locationCode='STK.HK.SEHK'` |
| STK.HK.SEHKNTL | `locationCode='STK.HK.SEHKNTL'` |
| STK.HK.SEHKSTAR | `locationCode='STK.HK.SEHKSTAR'` |
| STK.HK.SEHKSZSE | `locationCode='STK.HK.SEHKSZSE'` |
| STK.HK.SGX | `locationCode='STK.HK.SGX'` |
| STK.HK.TSE_JPN | `locationCode='STK.HK.TSE_JPN'` |
| STK.HK.TWSE | `locationCode='STK.HK.TWSE'` |
| STK.ME | `locationCode='STK.ME'` |
| STK.ME.TADAWUL | `locationCode='STK.ME.TADAWUL'` |
| STK.ME.TASE | `locationCode='STK.ME.TASE'` |
| STK.NA | `locationCode='STK.NA'` |
| STK.NA.CANADA | `locationCode='STK.NA.CANADA'` |
| STK.NA.MEXI | `locationCode='STK.NA.MEXI'` |
| STK.NA.TSE | `locationCode='STK.NA.TSE'` |
| STK.NA.VENTURE | `locationCode='STK.NA.VENTURE'` |
| STK.NASDAQ | `locationCode='STK.NASDAQ'` |
| STK.NASDAQ.NMS | `locationCode='STK.NASDAQ.NMS'` |
| STK.NASDAQ.SCM | `locationCode='STK.NASDAQ.SCM'` |
| STK.NYSE | `locationCode='STK.NYSE'` |
| STK.PINK | `locationCode='STK.PINK'` |
| STK.US | `locationCode='STK.US'` |
| STK.US.MAJOR | `locationCode='STK.US.MAJOR'` |
| STK.US.MINOR | `locationCode='STK.US.MINOR'` |