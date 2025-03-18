
import pandas as pd
import strategies.ta as ta
import strategies.signals as sig
from  strategies.ta import TAData
from frame import Frame
from data.random_data import RandomOHLCV
from dataclasses import dataclass
from typing import Dict, Any, List
from chart.chart import ChartArgs 

#------------------------------------------------------------
# ----------  H E L P E R S  --------------------------------
#------------------------------------------------------------

def batch_add_ta(f, ta:list, style: Dict[str, Any] | List[Dict[str, Any]] = {}, chart_type: str = "line", row: int = 1, nameCol:str=None):
    for t in ta:
        f.add_ta(t, style=style, chart_type=chart_type, row=row)

def add_score(f, validations:List[sig.Validate], name:str, weight:int=1, scoreType='mean', lookBack:int=10, row:int=3):
    cols=[v.name for v in validations]
    score_obj = sig.Score(name=name, cols=cols, scoreType=scoreType, weight=weight, lookBack=lookBack)
    f.add_ta(score_obj, {'dash': 'solid', 'color': 'magenta', 'width': 3}, chart_type='line', row=row)
    return score_obj.name

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------

def SIG_sentiment(f, lookBack, atr:int=20, volMA:int=10, rsiPeriod:int=14, scoreRow:int=3):
    f.add_ta(sig.SentimentGap(normRange=(-5,5), lookBack=lookBack), {'dash': 'solid', 'color': 'yellow', 'width': 1}, chart_type='line', row=scoreRow)
    f.add_ta(sig.SenitmentBar(normRange=(-5,5), lookBack=lookBack), {'dash': 'solid', 'color': 'cyan', 'width': 1}, chart_type='line', row=scoreRow)
    f.add_ta(sig.SentimentVolume(volMACol=f'MA_vo_{volMA}', atrCol=f'ATR_{atr}',  normRange=(-100,100), lookBack=lookBack), {'dash': 'solid', 'color': 'yellow', 'width': 1}, chart_type='line', row=scoreRow)
    f.add_ta(sig.RSI(rsiLookBack=rsiPeriod,  normRange=(0,100), lookBack=lookBack), {'dash': 'solid', 'color': 'yellow', 'width': 1}, chart_type='line', row=scoreRow)
    f.add_ta(sig.SentimentMAvsPrice(normRange=(-5,5), atrCol=f'ATR_{atr}', maCol='MA_cl_50', lookBack=lookBack), {'dash': 'solid', 'color': 'yellow', 'width': 1}, chart_type='line', row=scoreRow)
    f.add_ta(sig.SentimentMAvsPrice(normRange=(-5,5), atrCol=f'ATR_{atr}', maCol='MA_cl_200', lookBack=lookBack), {'dash': 'solid', 'color': 'yellow', 'width': 1}, chart_type='line', row=scoreRow)
    f.add_ta(sig.Score(name='SNMT', cols=['Sig_SnmtGap', 'Sig_SnmtBar', 'Sig_SnmtVol', f'Sig_RSI_{rsiPeriod}', 'Sig_SnmtMAP_MA_cl_50', 'Sig_SnmtMAP_MA_cl_200'], scoreType='mean',  weight=1, lookBack=lookBack), {'dash': 'solid', 'color': 'magenta', 'width': 3}, chart_type='line', row=scoreRow)


def TA_atr_hplp_supres_volma(f, pointsSpan:int=10, atrSpan:int=50, volMA:int=10, supresRowsToUpdate:int=10):
    f.add_ta(ta.MA('volume', volMA), {'dash': 'solid', 'color': 'yellow', 'width': 2}, row=2)
    f.add_ta(ta.ATR(span=atrSpan), {'dash': 'solid', 'color': 'cyan', 'width': 1}, row=3, chart_type='')
    f.add_ta(ta.HPLP(hi_col='high', lo_col='low', span=pointsSpan), [{'color': 'green', 'size': 10}, {'color': 'red', 'size': 10}], chart_type = 'points'),
    # f.add_ta(ta.SupRes(hi_point_col=f'HP_hi_{pointsSpan}', lo_point_col=f'LP_lo_{pointsSpan}', atr_col=f'ATR_{atrSpan}', tolerance=1),
    #         [{'dash': 'solid', 'color': 'green', 'fillcolour': "rgba(0, 255, 0, 0.1)", 'width': 2}, # support # green = rgba(0, 255, 0, 0.1)
    #         {'dash': 'solid', 'color': 'red', 'fillcolour': "rgba(255, 0, 0, 0.1)", 'width': 2}], # resistance # red = rgba(255, 0, 0, 0.1)
    #         chart_type = 'support_resistance')
    # f.add_ta(ta.SupResAllRows(hi_point_col=f'HP_hi_{pointsSpan}', lo_point_col=f'LP_lo_{pointsSpan}', atr_col=f'ATR_{atrSpan}', tolerance=1, rowsToUpdate=supresRowsToUpdate),
    #         [{'dash': 'solid', 'color': 'green', 'fillcolour': "rgba(0, 255, 0, 0.1)", 'width': 1}, # support # green = rgba(0, 255, 0, 0.1)
    #         {'dash': 'solid', 'color': 'red', 'fillcolour': "rgba(255, 0, 0, 0.1)", 'width': 1}], # resistance # red = rgba(255, 0, 0, 0.1)
    #         chart_type = 'support_resistance')
    

def SIG_is_trending(f, ls='LONG', ma=50, lookBack:int=100, scoreRow:int=6):
    f.add_ta(sig.IsMATrending(maCol=f'MA_cl_{ma}', ls=ls, lookBack=lookBack, normRange=(0,1)), {'dash': 'solid', 'color': 'green', 'width': 2}, chart_type='line', row=scoreRow)
    f.add_ta(sig.IsPointsTrending(hpCol='HP_hi_3', lpCol='LP_lo_3', ls=ls, lookBack=lookBack, normRange=(0,1)), {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=scoreRow)
    f.add_ta(sig.IsPointsTrending(hpCol='HP_hi_10', lpCol='LP_lo_10', ls=ls, lookBack=lookBack, normRange=(0,1)), {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=scoreRow)
    f.add_ta(sig.AboveBelow(value='close', direction='above', metric_column=f'MA_cl_{ma}', ls=ls, lookBack=lookBack, normRange=(0,1)), {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=scoreRow)
    
    f.add_ta(sig.Score(name='Trend', cols=[f'TREND_MA_cl_{ma}', 'TREND_LONG_3', 'TREND_LONG_10', 'AB_close_ab_MA_cl_50'], scoreType='mean',  weight=1, lookBack=lookBack, normRange=(0,100)), {'dash': 'solid', 'color': 'magenta', 'width': 3}, chart_type='line', row=scoreRow)
        

def SIG_volume(f, ls:str='LONG', ma:int=10, lookBack:int=100, scoreRow:int=3, chartType:str='line'):
    volMACol = f'MA_vo_{ma}'
    l_or_s = ls[0]
    f.add_ta_batch([
        TAData(ta.MA('volume', ma), {'dash': 'solid', 'color': 'yellow', 'width': 2}, row=2),
        TAData(sig.VolumeSpike(ls=ls, normRange=(0, 200), volMACol=volMACol, lookBack=lookBack), {'dash': 'solid', 'color': 'blue', 'width': 1}, chart_type=chartType, row=scoreRow),
        TAData(sig.VolumeROC(ls=ls, normRange=(0, 300), lookBack=lookBack), {'dash': 'solid', 'color': 'red', 'width': 1}, chart_type=chartType, row=scoreRow),
        TAData(sig.Score(name=f'{l_or_s}_Vol', cols=[f'Sig{l_or_s}_VolSpike', f'Sig{l_or_s}_VolROC'], scoreType='max', weight=1, lookBack=lookBack), {'dash': 'solid', 'color': 'magenta', 'width': 3}, chart_type=chartType, row=scoreRow)
    ])


def SIG_is_breaking_out(f, ls:str='LONG', lookBack:int=100, scoreRow:int=6, chartType:str='line'):
    f.add_ta(sig.Breakout(f, price_column='close', direction='above', normRange=(0,1), resCols=['Res_1_Upper', 'Res_2_Upper'], lookBack=lookBack, shift=1), {'dash': 'solid', 'color': 'green', 'width': 1}, chart_type='line', row=scoreRow)
    f.add_ta(sig.BreaksPivot(f, pointCol='HP_hi_3', direction='above', normRange=(0,1), lookBack=lookBack), {'dash': 'solid', 'color': 'lime', 'width': 1}, chart_type='line', row=scoreRow)
    f.add_ta(sig.BreaksPivot(f, pointCol='HP_hi_10', direction='above', normRange=(0,1), lookBack=lookBack), {'dash': 'solid', 'color': 'cyan', 'width': 1}, chart_type='line', row=scoreRow)
    f.add_ta(sig.Score(name='Breakout', cols=['BRKOUT_close_ab', 'BRK_ab_HP_hi_3', 'BRK_ab_HP_hi_10'], scoreType='mean',  weight=1, lookBack=lookBack, normRange=(0,100)), {'dash': 'solid', 'color': 'magenta', 'width': 3}, chart_type='line', row=scoreRow)


def SIG_has_room_to_move(f, ls:str='LONG', atr:int=20, lookBack:int=100, scoreRow:int=7, chartType:str='line'):
    f.add_ta(sig.RoomToMove(ls=ls, tgetCol='Res_1_Lower', atrCol=f'ATR_{atr}', normRange=(0,10), lookBack=lookBack), {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=scoreRow)
    f.add_ta(sig.RoomToMove(ls=ls, tgetCol='Res_2_Lower', atrCol=f'ATR_{atr}', normRange=(0,10), lookBack=lookBack), {'dash': 'solid', 'color': 'cyan', 'width': 2}, chart_type='line', row=scoreRow)
    f.add_ta(sig.Score(name='RoomToMove', cols=['SigL_RTM_Res_1_Lower', 'SigL_RTM_Res_2_Lower'], scoreType='mean',  weight=1, lookBack=lookBack, normRange=(0,100)), {'dash': 'solid', 'color': 'magenta', 'width': 3}, chart_type='line', row=scoreRow)


def import_to_daily_df(f, spy:pd.DataFrame=None, etf:pd.DataFrame=None, RSIRow:int=4):
    if spy is not None:
        f.import_data(spy, has_columns=['close'], prefix='SPY_')
        f.add_ta(ta.MansfieldRSI(stockCol='close', marketCol='SPY_close', span=14), {'dash': 'solid', 'color': 'yellow', 'width': 1}, row=RSIRow)
    if etf is not None:
        f.import_data(etf, has_columns=['close'], prefix='ETF_')
        f.add_ta(ta.MansfieldRSI(stockCol='close', marketCol='ETF_close', span=14), {'dash': 'solid', 'color': 'magenta', 'width': 1}, row=RSIRow)


def import_to_minute_df(f, daily:pd.DataFrame=None, hr4:pd.DataFrame=None, hr1:pd.DataFrame=None):
    if daily is not None: f.import_data(daily, columns_contain=['Sup', 'Res'], prefix='DAILY_' )
    if hr4   is not None: f.import_data(hr4,   columns_contain=['Sup', 'Res'], prefix='HR4_' )
    if hr1   is not None: f.import_data(hr1,   columns_contain=['Sup', 'Res'], prefix='HR1_' )


def ma_ta(f, periods: list[int], ma_col: str = 'close', row: int = 1):
    ma_colour_map = {
        200: {'colour': 'darkslateblue', 'size': 5},
        150: {'colour': 'cornflowerblue', 'size': 4},
        50: {'colour': 'darkmagenta', 'size': 3},
        21: {'colour': 'hotpink', 'size': 2},
        13: {'colour': 'deepskyblue', 'size': 1},
        8: {'colour': 'khaki', 'size': 1}}
    
    for period in periods:
        if period in ma_colour_map:
            f.add_ta(ta.MA(maCol=ma_col, period=period), 
                    {'dash': 'solid', 
                     'color': ma_colour_map[period]['colour'], 
                     'width': ma_colour_map[period]['size']}, 
                    row=row)



def gaps_ta(f, ls:str='LONG', pointCol:str='HP_hi_10', atrCol:str='ATR_50', lookBack:int=1, row:int=4, chartType:str='lines+markers'):
    l_or_s = ls[0]
    f.add_ta_batch([
        TAData(sig.IsGappedOverPivot(ls=ls, normRange=(0,1), pointCol=pointCol, lookBack=lookBack), {'dash': 'solid', 'color': 'blue', 'width': 1}, chart_type=chartType, row=row),
        TAData(sig.GappedPivots(ls=ls, normRange=(0, 3), pointCol=pointCol, span=400, lookBack=lookBack), {'dash': 'solid', 'color': 'orange', 'width': 1}, chart_type=chartType, row=row),
        TAData(sig.GappedRetracement(ls=ls, normRange=(0,100), pointCol=pointCol, atrCol=atrCol, lookBack=lookBack), {'dash': 'solid', 'color': 'magenta', 'width': 1}, chart_type=chartType, row=row),
        TAData(sig.GappedPastPivot(ls=ls, normRange=(0,100), atrCol=atrCol, pointCol=pointCol, lookBack=lookBack, maxAtrMultiple=10), {'dash': 'solid', 'color': 'red', 'width': 1}, chart_type=chartType, row=row),
        TAData(sig.GapSize(ls=ls, normRange=(0,300), pointCol=pointCol, atrCol=atrCol, lookBack=lookBack), {'dash': 'solid', 'color': 'red', 'width': 1}, chart_type=chartType, row=row),
        TAData(sig.Score(name=f'{l_or_s}_Gaps', normRange=(0,100), lookBack=lookBack, cols=[f'Sig{l_or_s}_GPivs', f'Sig{l_or_s}_GRtc', f'Sig{l_or_s}_GPP', f'Sig{l_or_s}_GSiz'], scoreType='mean', weight=1), {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type=chartType, row=row)
    ])


def consolidation_ta(f, atrSpan:int=50, consUpperCol:str='CONS_UPPER', consLowerCol:str='CONS_LOWER', maSpan:int=50, lookBack:int=1, colours:list[str]=None, chartType:str='line', scoreRow:int=4):
    colours = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'purple', 'brown', 'pink', 'grey' ] if not colours else colours
    atr_col = f'ATR_{atrSpan}'
    ma_col = f'MA_{maSpan}'
    f.add_ta_batch([
        TAData(ta.ConsolidationZone(hp_column='HP_hi_10', lp_column='LP_lo_10', atr_column='ATR_50', price_tolerance=0.001, max_points_between=1, height_width_ratio=50, limit_zones=2),
            [{'color': "rgba(225, 182, 30, 0.5)", 'fillcolour': "rgba(225, 182, 30, 0.1)", 'width': 2}, # support # green = rgba(0, 255, 0, 0.1)
            {'color': "rgba(225, 182, 30, 0.1)", 'fillcolour': "rgba(225, 182, 30, 0.1)", 'width': 2}], # resistance # red = rgba(255, 0, 0, 0.1)
            chart_type = 'cons'),
        TAData(sig.ConsolidationShape(normRange=(1, 20), consUpperCol=consUpperCol, consLowerCol=consLowerCol, atrCol=atr_col, minBars=10, lookBack=lookBack), [{'dash': 'solid', 'color': color, 'width': 2} for color in colours], chart_type=chartType, row=scoreRow),
        TAData(sig.ConsolidationPosition(normRange=(1, 100), consUpperCol=consUpperCol, consLowerCol=consLowerCol, lookBack=lookBack), [{'dash': 'solid', 'color': color, 'width': 2} for color in colours], chart_type=chartType, row=scoreRow),
        TAData(sig.ConsolidationPreMove(normRange=(1, 15), consUpperCol=consUpperCol, consLowerCol=consLowerCol, maCol=ma_col, atrCol=atr_col, lookBack=lookBack), [{'dash': 'solid', 'color': color, 'width': 2} for color in colours], chart_type=chartType, row=scoreRow),
        TAData(sig.Score(rawName='Cons_Score_1', normRange=(0, 100), lookBack=lookBack, containsAllStrings=['Cons', '1'], scoreType='mean', weight=1), {'dash': 'solid', 'color': 'brown', 'width': 4}, chart_type=chartType, row=scoreRow),
        TAData(sig.Score(rawName='Cons_Score_2', normRange=(0, 100), lookBack=lookBack, containsAllStrings=['Cons', '2'], scoreType='mean', weight=1), {'dash': 'solid', 'color': 'darkgreen', 'width': 4}, chart_type=chartType, row=scoreRow)
    ])


def trending_ta(f, ls:str='LONG', lookBack:int=100, scoreRow:int=4, chartType:str='lines+markers'):
    f.add_ta(sig.ROC(metricCol='MA_cl_21',  rocLookBack=10, lookBack=lookBack), {'dash': 'solid', 'color': 'yellow', 'width': 1}, chart_type='line', row=scoreRow)
    f.add_ta(sig.ROC(metricCol='MA_cl_50',  rocLookBack=10, lookBack=lookBack), {'dash': 'solid', 'color': 'yellow', 'width': 1}, chart_type='line', row=scoreRow)
    f.add_ta(sig.ROC(metricCol='MA_cl_150', rocLookBack=10, lookBack=lookBack), {'dash': 'solid', 'color': 'yellow', 'width': 1}, chart_type='line', row=scoreRow)
    f.add_ta(sig.ROC(metricCol='MA_cl_200', rocLookBack=10, lookBack=lookBack), {'dash': 'solid', 'color': 'yellow', 'width': 1}, chart_type='line', row=scoreRow)
    f.add_ta(sig.Score(name='trend_ROC', cols=['SigL_ROC_MA_cl_21', 'SigL_ROC_MA_cl_50', 'SigL_ROC_MA_cl_150', 'SigL_ROC_MA_cl_200'], scoreType='mean',  weight=10, lookBack=lookBack, normRange=(0,100)), {'dash': 'solid', 'color': 'cyan', 'width': 2}, chart_type='line', row=scoreRow)

    # Sector Bullish Score
    f.add_ta(sig.PctDiff(metricCol1='close', metricCol2='MA_cl_21', lookBack=lookBack), {'dash': 'solid', 'color': 'yellow', 'width': 1}, chart_type='line', row=scoreRow)
    f.add_ta(sig.PctDiff(metricCol1='close', metricCol2='MA_cl_50', lookBack=lookBack), {'dash': 'solid', 'color': 'yellow', 'width': 1}, chart_type='line', row=scoreRow)
    f.add_ta(sig.PctDiff(metricCol1='close', metricCol2='MA_cl_150', lookBack=lookBack), {'dash': 'solid', 'color': 'yellow', 'width': 1}, chart_type='line', row=scoreRow)
    f.add_ta(sig.PctDiff(metricCol1='close', metricCol2='MA_cl_200', lookBack=lookBack), {'dash': 'solid', 'color': 'yellow', 'width': 1}, chart_type='line', row=scoreRow)
    f.add_ta(sig.Score(name='trend_PctDiff', cols=['SigL_PctDiff_MA_cl_21', 'SigL_PctDiff_MA_cl_50' ,'SigL_PctDiff_MA_cl_150', 'SigL_PctDiff_MA_cl_200', ], scoreType='mean',  weight=1, lookBack=lookBack, normRange=(0,100)), {'dash': 'solid', 'color': 'cyan', 'width': 2}, chart_type='line', row=scoreRow)

    #final Score
    f.add_ta(sig.Score(name='trend_FINAL', cols=['Score_trend_ROC', 'Score_trend_PctDiff'], scoreType='mean',  weight=2, lookBack=lookBack, normRange=(0,100)), {'dash': 'solid', 'color': 'magenta', 'width': 3}, chart_type='line', row=scoreRow)


def SIG_compare_to_market_ta(f, marketCol='SPY_close', ls:str='LONG', lookBack:int=100, scoreRow:int=4, chartType:str='line'):
    """As long as Comp_MKT above 0 then it is tradeing positively compared to the market"""
    f.add_ta(ta.MansfieldRSI(stockCol='close', marketCol=marketCol,   span=14), {'dash': 'solid', 'color': 'yellow', 'width': 1}, chart_type=chartType, row=scoreRow)
    f.add_ta(ta.PctChange(metric_column='MA_cl_50'), {'dash': 'solid', 'color': 'yellow', 'width': 1}, chart_type=chartType, row=scoreRow)
    f.add_ta(sig.Score(name='Comp_MKT', cols=['MRSI_14_SPY_close', 'PCT_MA_cl_50_1'], scoreType='mean', validThreshold=0, weight=1, lookBack=lookBack, normRange=(0,1)), {'dash': 'solid', 'color': 'magenta', 'width': 3}, chart_type=chartType, row=scoreRow)
        

#------------------------------------------------------------
# ----------   T A  -----------------------------------------
#------------------------------------------------------------
def TA_TA(f, lookBack:int=100, atrSpan:int=50, pointsSpan:int=10, isDaily:bool=False):
    ma_ta(f, [50, 150, 200])
    f.add_ta(ta.ATR(span=atrSpan), {'dash': 'solid', 'color': 'cyan', 'width': 1}, row=3, chart_type='')
    f.add_ta(ta.HPLP(hi_col='high', lo_col='low', span=3), [{'color': 'green', 'size': 5}, {'color': 'red', 'size': 5}], chart_type = 'points')
    f.add_ta(ta.HPLP(hi_col='high', lo_col='low', span=pointsSpan), [{'color': 'green', 'size': 10}, {'color': 'red', 'size': 10}], chart_type = 'points')
    f.add_ta(ta.SupResAllRows(hi_point_col=f'HP_hi_{pointsSpan}', lo_point_col=f'LP_lo_{pointsSpan}', atr_col=f'ATR_{atrSpan}', tolerance=1, rowsToUpdate=lookBack),
    [{'dash': 'solid', 'main_line_colour': 'green', 'zone_edge_colour': 'rgba(0, 255, 0, 0.3)', 'fillcolour': "rgba(0, 255, 0, 0.1)", 'width': 1}, # support # green = rgba(0, 255, 0, 0.1)
    {'dash': 'solid', 'main_line_colour': 'red', 'fillcolour': "red", 'zone_edge_colour': 'rgba(255, 0, 0, 0.3)', 'fillcolour': "rgba(255, 0, 0, 0.1)", 'width': 1}], # resistance # red = rgba(255, 0, 0, 0.1)
    chart_type = 'support_resistance')

    if not isDaily:
        f.add_ta(ta.VWAP(column='close',  interval='session'), {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=1)
        f.add_ta(ta.VolumeAccumulation(), {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=2)


def TA_Levels(f):
    tas = [
        ta.Levels(level='pre_mkt_high',       ffill=True),
        ta.Levels(level='pre_mkt_low',        ffill=True),
        ta.Levels(level='intraday_high_9.35', ffill=False),
        ta.Levels(level='intraday_low_9.35',  ffill=False),
        ta.Levels(level='intraday_high'),                
        ta.Levels(level='intraday_low'),                 
        ta.Levels(level='prev_day_high'),                
        ta.Levels(level='prev_day_low')
    ]
    batch_add_ta(f, tas,  {'dash': 'dash', 'color': 'yellow', 'width': 1}, chart_type='line', row=1) 

def TA_RTM(f, ls:str='LONG', atrSpan:int=50, lookBack:int=1, TArow:int=3):
    tas = [
        sig.RoomToMove(ls=ls, tgetCol='Res_1_Lower', atrCol=f'ATR_{atrSpan}', unlimitedVal=5, normRange=(0,5), lookBack=lookBack),
    ]
    batch_add_ta(f, tas,  {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=TArow)
    return tas[0].name


#! TODO - Decide what is important to validate
def TA_Daily(f, ls:str='LONG', pointCol:str='HP_hi_10', atrSpan:int=10, lookBack:int=1, TArow:int=2, scoreRow:int=5):
    atr_col = f'ATR_{atrSpan}'
    gap_tas = [
        sig.IsGappedOverPivot(ls=ls, normRange=(0,1), pointCol=pointCol, lookBack=lookBack),
        sig.GappedPivots(ls=ls, normRange=(0, 3), pointCol=pointCol, spanPivots=10, lookBack=lookBack),
        sig.GappedRetracement(ls=ls, normRange=(0,100), pointCol=pointCol, atrCol=atr_col, lookBack=lookBack),
        sig.GappedPastPivot(ls=ls, normRange=(0,100), atrCol=atr_col, pointCol=pointCol, lookBack=lookBack, maxAtrMultiple=10),
        sig.GapSize(ls=ls, normRange=(0,300), atrCol=atr_col, lookBack=lookBack),
        sig.RoomToMove(ls=ls, tgetCol='Res_1_Lower', atrCol=atr_col, unlimitedVal=5, normRange=(0,5), lookBack=lookBack)
    ]
    batch_add_ta(f, gap_tas,  {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=TArow)

    ta_rs = f.add_ta(ta.RS(comparisonPrefix='SPY', ma=14, atr=atrSpan), 
        [{'dash': 'solid', 'color': 'yellow', 'width': 1},
        {'dash': 'solid', 'color': 'cyan', 'width': 1}], 
        chart_type='line', row=TArow)
    

    validations = [
        sig.Validate(f, val1='GapPiv_HP_hi_10',   operator='>', val2=1, lookBack=lookBack),  # has a bar gapped over the pivot
        sig.Validate(f, val1='L_GPivs',           operator='>', val2=1, lookBack=lookBack),  # gapped pivots in the last 10 bars
        sig.Validate(f, val1='L_GRtc',            operator='>', val2=50, lookBack=lookBack),  # gapped retracement of 50% or more
        sig.Validate(f, val1='L_GPP',             operator='>', val2=80, lookBack=lookBack),  # gapped past pivot
        sig.Validate(f, val1='GapSz',             operator='<', val2=400, lookBack=lookBack),  # gapped size less than 400%
        sig.Validate(f, val1='RTM_L_Res_1_Lower', operator='>', val2=50, lookBack=lookBack),  # has room to move
        sig.Validate(f, val1=ta_rs.name,          operator='>', val2=1, lookBack=lookBack),  # has a positive RS compared to the market (Relative Strength)
    ]
    batch_add_ta(f, validations,  {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=TArow)
    scoreCol = add_score(f, validations,  name=f'{ls}_Daily_VALID_Score',  scoreType='mean', lookBack=lookBack, row=scoreRow)
    return scoreCol

def SCORE_Gaps(f, ls:str='LONG', pointCol:str='HP_hi_10', atrSpan:int=10, lookBack:int=1, TArow:int=2, scoreRow:int=5):
    atr_col = f'ATR_{atrSpan}'
    gap_tas = [
        sig.IsGappedOverPivot(ls=ls, normRange=(0,1), pointCol=pointCol, lookBack=lookBack),
        sig.GappedPivots(ls=ls, normRange=(0, 3), pointCol=pointCol, spanPivots=10, lookBack=lookBack),
        sig.GappedRetracement(ls=ls, normRange=(0,100), pointCol=pointCol, atrCol=atr_col, lookBack=lookBack),
        sig.GappedPastPivot(ls=ls, normRange=(0,100), atrCol=atr_col, pointCol=pointCol, lookBack=lookBack, maxAtrMultiple=10),
        sig.GapSize(ls=ls, normRange=(0,300), atrCol=atr_col, lookBack=lookBack),
    ]
    batch_add_ta(f, gap_tas,  {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=TArow)    

    validations = [
        sig.Validate(f, val1='GapPiv_HP_hi_10',   operator='>', val2=1, lookBack=lookBack),  # has a bar gapped over the pivot
        sig.Validate(f, val1='L_GPivs',           operator='>', val2=1, lookBack=lookBack),  # gapped pivots in the last 10 bars
        sig.Validate(f, val1='L_GRtc',            operator='>', val2=50, lookBack=lookBack),  # gapped retracement of 50% or more
        sig.Validate(f, val1='L_GPP',             operator='>', val2=80, lookBack=lookBack),  # gapped past pivot
        sig.Validate(f, val1='GapSz',             operator='<', val2=400, lookBack=lookBack),  # gapped size less than 400%
    ]
    batch_add_ta(f, validations,  {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=TArow)
    scoreCol = add_score(f, validations,  name=f'{ls}_Gaps',  scoreType='mean', lookBack=lookBack, row=scoreRow)
    return scoreCol

#------------------------------------------------------------
# ----------   S C O R E   T A  -----------------------------
#------------------------------------------------------------

def TA_Daily_Volume_Change(f, lookBackDays:int=10, TArow:int=2):
    tas = [
        ta.VolumeAccumulation(),
        ta.VolumeTimeOfDayChangePct(lookbackDays=lookBackDays),
    ]
    batch_add_ta(f, tas,  {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=TArow)
    return tas[1].name

def SCORE_TA_Volume(f, ls:str='LONG', lookBack:int=100, volMA:int=10, TArow:int=2, scoreRow:int=5):
    tas = [
        ta.MA(maCol='volume', period=volMA),
        ta.VolumeAccumulation(),
        ta.VolumeTimeOfDayChangePct(lookbackDays=10)
    ]
    batch_add_ta(f, tas,  {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=2)
    scores = [ 
        sig.VolumeSpike(ls=ls, normRange=(0, 200), volMACol=f'MA_vo_{volMA}', lookBack=lookBack),
        sig.VolumeROC(ls=ls, normRange=(0, 300), lookBack=lookBack)
    ]
    batch_add_ta(f, scores,  {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=TArow)
    score_col = add_score(f, tas,  name=f'{ls}_Vol',  scoreType='max', lookBack=lookBack, row=scoreRow)
    return score_col


def SCORE_TA_touches(f, ls='LONG', lookBack:int=100,  atrSpan:int=10, direction:str='down', toTouchAtrScale=10, pastTouchAtrScale=2,  TArow:int=2, scoreRow:int=3):
    col1 = 'Sup_1_Upper'        if direction == 'down' else 'Res_1_Lower'
    col2 = '1 hour_Sup_1_Upper' if direction == 'down' else '1 hour_Res_1_Lower'
    col3 = '1 day_Sup_1_Upper'  if direction == 'down' else '1 day_Res_1_Lower'
    tas = [
        sig.TouchWithBar(ls=ls, atrCol=f'ATR_{atrSpan}', valCol=col1,            direction=direction, toTouchAtrScale=toTouchAtrScale, pastTouchAtrScale=pastTouchAtrScale, lookBack=lookBack), # has a bar touched a level -- Res 1
        sig.TouchWithBar(ls=ls, atrCol=f'ATR_{atrSpan}', valCol=col2,            direction=direction, toTouchAtrScale=toTouchAtrScale, pastTouchAtrScale=pastTouchAtrScale, lookBack=lookBack), # has a bar touched a level -- Res 3
        sig.TouchWithBar(ls=ls, atrCol=f'ATR_{atrSpan}', valCol=col3,            direction=direction, toTouchAtrScale=toTouchAtrScale, pastTouchAtrScale=pastTouchAtrScale, lookBack=lookBack), # has a bar touched a level -- Res 2
        sig.TouchWithBar(ls=ls, atrCol=f'ATR_{atrSpan}', valCol='prev_day_low',  direction=direction, toTouchAtrScale=toTouchAtrScale, pastTouchAtrScale=pastTouchAtrScale, lookBack=lookBack), # has a bar touched a level -- Res 2
        sig.TouchWithBar(ls=ls, atrCol=f'ATR_{atrSpan}', valCol='prev_day_high', direction=direction, toTouchAtrScale=toTouchAtrScale, pastTouchAtrScale=pastTouchAtrScale, lookBack=lookBack), # has a bar touched a level -- Res 2
    ]
    batch_add_ta(f, tas,  {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=TArow)
    score_col = add_score(f, tas, name=f'{ls}_Touch', scoreType='mean', lookBack=lookBack, row=scoreRow) 
    return score_col


def SCORE_TA_retest(f, ls='LONG', lookBack:int=100,  atrSpan:int=10, direction:str='down', withinAtrRange=(-0.1, 0.1), pastTouchAtrScale=2,  TArow:int=2, scoreRow:int=3):
    tas = [
        sig.Retest(ls=ls, atrCol=f'ATR_{atrSpan}', direction=direction, valCol='HP_hi_3', withinAtrRange=withinAtrRange, rollingLen=5, lookBack=lookBack, normRange=(0,3)), # retest HP withing 10% of ATR
        sig.Retest(ls=ls, atrCol=f'ATR_{atrSpan}', direction=direction, valCol='LP_lo_3', withinAtrRange=withinAtrRange, rollingLen=5, lookBack=lookBack, normRange=(0,3)), # retest LP withing 10% of ATR
        sig.Retest(ls=ls, atrCol=f'ATR_{atrSpan}', direction=direction, valCol='low',     withinAtrRange=withinAtrRange, rollingLen=3,  lookBack=lookBack, normRange=(0,3)),  # retest HP withing 10% of ATR

    ]
    batch_add_ta(f, tas,  {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=TArow)
    scoreCol = add_score(f, tas,  name=f'{ls}_Retest',  scoreType='sum', lookBack=lookBack, row=scoreRow)
    return scoreCol


def SCORE_TA_Pullback(f, ls:str='LONG', lookBack:int=100, atrSpan:int=50, minPbLen:int=3, TArow:int=2, scoreRow:int=3):
    tas = [
        sig.PB_PctHLLH          (ls=ls, normRange=(0,100), minPbLen=minPbLen, atrCol=f'ATR_{atrSpan}', pointCol='HP_hi_3', lookBack=lookBack), # Lower Highs
        sig.PB_ASC              (ls=ls, normRange=(0,100), minPbLen=minPbLen,                          pointCol='HP_hi_3', lookBack=lookBack), # Pct of pull back bars with same colour .eg all red if 'LONG
        sig.PB_CoC_ByCountOpBars(ls=ls, normRange=(0,100), minPbLen=minPbLen,                          pointCol='HP_hi_3', lookBack=lookBack), # count of consecutive same colour prior to CoC bar. eg 3 red bars before CoC bar. Then % as a total of PB bars 
        sig.PB_Overlap          (ls=ls, normRange=(0,100), minPbLen=minPbLen,                          pointCol='HP_hi_3', lookBack=lookBack), # Mean of the % pull back overlap the previous bar
    ]
    batch_add_ta(f, tas,  {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=TArow)
    score_col = add_score(f, tas, name=f'{ls}_PB', scoreType='mean', lookBack=lookBack, row=scoreRow) 
    return score_col



def SCORE_TA_Bar_StrengthWeakness(f, ls:str='LONG', lookBack:int=100, atrSpan:int=50, barSwMA:int=4, TArow:int=2, scoreRow:int=3):
    bsw = sig.BarSW(ls=ls, normRange=(-5,5), atrCol=f'ATR_{atrSpan}', lookBack=lookBack)
    ma = ta.MA(maCol=bsw.name, period=barSwMA)
    bws_diff = sig.PctDiff(metricCol1=ma.name, metricCol2=bsw.name, lookBack=lookBack, normRange=(-100,100))
    
    batch_add_ta(f, [bsw, ma, bws_diff],  {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=TArow)

    score_col = add_score(f, [bws_diff],  name=f'{ls}_BSW', scoreType='min', lookBack=lookBack, row=scoreRow) 
    return score_col
     

def SCORE_TA_RTM_TRADING(f, ls:str='LONG', lookBack:int=100, atrSpan:int=50, TArow:int=2, scoreRow:int=3):
    tas = [
        sig.RoomToMove(ls=ls, atrCol=f'ATR_{atrSpan}', tgetCol='Res_1_Lower',        unlimitedVal=10, normRange=(0,10), lookBack=lookBack),
        sig.RoomToMove(ls=ls, atrCol=f'ATR_{atrSpan}', tgetCol='1 day_Res_1_Lower',  unlimitedVal=10, normRange=(0,10), lookBack=lookBack),
        sig.RoomToMove(ls=ls, atrCol=f'ATR_{atrSpan}', tgetCol='1 hour_Res_1_Lower', unlimitedVal=10, normRange=(0,10), lookBack=lookBack),
        sig.RoomToMove(ls=ls, atrCol=f'ATR_{atrSpan}', tgetCol='prev_day_high',      unlimitedVal=10, normRange=(0,10), lookBack=lookBack),
    ]
    batch_add_ta(f, tas,  {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=TArow)
    score_col = add_score(f, tas,  name=f'{ls}_RTM', scoreType='min', lookBack=lookBack, row=scoreRow) 
    return score_col

def SCORE_TA_RTM_DAILY(f, ls:str='LONG', lookBack:int=100, atrSpan:int=50, TArow:int=3, scoreRow:int=4):
    tgetCol = 'Res_1_Lower' if ls == 'LONG' else 'Sup_1_Upper'
    tas = [
        sig.RoomToMove(ls=ls, atrCol=f'ATR_{atrSpan}', tgetCol=tgetCol, unlimitedVal=5, normRange=(0,5), lookBack=lookBack),
    ]
    batch_add_ta(f, tas,  {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=TArow)
    score_col = add_score(f, tas,  name=f'{ls}_RTM', scoreType='min', lookBack=lookBack, row=scoreRow) 
    return score_col

def TA_RSI(f, lookBack:int=100, rsiPeriod:int=14, TArow:int=3, scoreRow:int=4):
    
    add_score(f, ta_rs,  name='RSI',  scoreType='mean', lookBack=lookBack, row=scoreRow)
    return ta_rs




#------------------------------------------------------------
# ----------  S C O R E    V A L I D A T I O N S  -----------
#------------------------------------------------------------
#! All validation score should be 'mean'. check value so consistant score ranges.  

def SCORE_VALID_premkt_volume(f, ls:str='LONG', lookBack:int=100, sigRow:int=3, validationRow:int=4):
    validations = [
        sig.Validate(f, val1='VOL_TODC_10', operator='>', val2=200, lookBack=lookBack),  # Volume Time of Day Comparison Indicator
    ]
    batch_add_ta(f, validations,  {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=sigRow)
    scoreCol = add_score(f, validations,  name=f'{ls}_PremktVolume',  scoreType='mean', lookBack=lookBack, row=validationRow)
    return scoreCol


def SCORE_VALID_time_of_day(f, ls:str='LONG', lookBack:int=100, sigRow:int=3, validationRow:int=4):
    validations = [
        sig.Validate(f, val1='idx', operator='t>t', val2='09:34', lookBack=lookBack),  # Volume Time of Day Comparison Indicator
        sig.Validate(f, val1='idx', operator='t<t', val2='12:00', lookBack=lookBack),  # Volume Time of Day Comparison Indicator
    ]
    batch_add_ta(f, validations,  {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=sigRow)
    score_col = add_score(f, validations,  name=f'{ls}_Time',  scoreType='mean', lookBack=lookBack, row=validationRow)
    return score_col


def SCORE_VALID_Levels_premkt_and_5minbar(f, ls:str='LONG',  lookBack:int=100, sigRow:int=3, validationRow:int=4):
    if ls == 'LONG':
        validations = [
            sig.Validate(f, val1='close', operator='>', val2='pre_mkt_high',       lookBack=lookBack),  # close > pre_mkt_high
            sig.Validate(f, val1='close', operator='>', val2='intraday_high_9.35', lookBack=lookBack),  # close > intraday_high_9.35
        ]
    else:
        validations = [
            sig.Validate(f, val1='close', operator='<', val2='pre_mkt_low',       lookBack=lookBack),  # close < pre_mkt_low
            sig.Validate(f, val1='close', operator='<', val2='intraday_low_9.35', lookBack=lookBack),  # close < intraday_low_9.35
        ]
    batch_add_ta(f, validations,  {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=sigRow)
    scoreCol = add_score(f, validations,  name=f'{ls}_ValBreaks',  scoreType='mean', lookBack=lookBack, row=validationRow)
    return scoreCol


def SCORE_VALID_reset_if_breaks(f, ls = 'LONG', lookBack:int=100, sigRow:int=3, validationRow:int=4):
    if ls == 'LONG':
        validations = [
            sig.Validate(f, val1='close', operator='v', val2='intraday_low',      lookBack=lookBack),  # close breaks below intraday low
            sig.Validate(f, val1='close', operator='v', val2='pre_mkt_low',       lookBack=lookBack),  # close breaks below pre market low
            sig.Validate(f, val1='close', operator='v', val2='prev_day_low',      lookBack=lookBack),  # close breaks below prev day low
            sig.Validate(f, val1='close', operator='v', val2='intraday_low_9.35', lookBack=lookBack),  # close breaks below intraday low
        ]
    else:  
        validations = [
            sig.Validate(f, val1='close', operator='^', val2='intraday_high',      lookBack=lookBack),  # close breaks above intraday high
            sig.Validate(f, val1='close', operator='^', val2='pre_mkt_high',       lookBack=lookBack),  # close breaks above pre market high
            sig.Validate(f, val1='close', operator='^', val2='prev_day_high',      lookBack=lookBack),  # close breaks above prev day high
            sig.Validate(f, val1='close', operator='^', val2='intraday_high_9.35', lookBack=lookBack),  # close breaks above intraday high
        ]
    batch_add_ta(f, validations,  {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=sigRow)
    scoreCol = add_score(f, validations,  name=f'{ls}_ResetIfBreaks',  scoreType='mean', lookBack=lookBack, row=validationRow)
    return scoreCol


def SCORE_VALID_buy(f, ls='LONG', lookBack:int=100, sigRow:int=3, validationRow:int=4):
    ta_rs = f.add_ta(ta.RS(comparisonPrefix='SPY', ma=10, atr=50), 
            [{'dash': 'solid', 'color': 'yellow', 'width': 1},
            {'dash': 'solid', 'color': 'cyan', 'width': 1}], 
            chart_type='line', row=sigRow)
    
    buy_validations = [
        sig.Validate(f, val1='close',             operator='>',   val2='VWAP_session', lookBack=lookBack), # close > VWAP
        sig.Validate(f, val1='Score_LONG_RTM',    operator='>',   val2=20,             lookBack=lookBack), # Room to move > 2 (measured by ATR units but check normalised score. 2 might = 20%)
        sig.Validate(f, val1=ta_rs.name,       operator='>',   val2=2,              lookBack=lookBack),  # RS > 2 (measured by ATR units)
    ]
    batch_add_ta(f, buy_validations,  {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=sigRow)
    scoreCol = add_score(f, buy_validations,  name=f'{ls}_ValBuy',  scoreType='mean', lookBack=lookBack, row=validationRow) 
    return scoreCol


# other validations that can be added but critical to achieveing high level of confidence 
def SCORE_VALID_bonus(f, ls='LONG',  atrSpan:int=10, lookBack:int=100, sigRow:int=3, validationRow:int=4):
    bonus_validations = [
        sig.Validate(f, val1=('HP_hi_10', -1),                    operator='p<p', val2=('HP_hi_10', -2), lookBack=lookBack),  # Higher High
        sig.Validate(f, val1=('LP_lo_10', -1),                    operator='p<p', val2=('LP_lo_10', -2), lookBack=lookBack),  # Higher Low
        sig.Validate(f, val1='close',                             operator='>',   val2='MA_cl_50',       lookBack=lookBack),  # close > MA50
        sig.Validate(f, val1='close',                             operator='^p',  val2='HP_hi_10',       lookBack=lookBack),  # close breaks HP_hi_10
        sig.Validate(f, val1='Touch_down_1 hour_Sup_1_Upper',     operator='>',   val2=80,               lookBack=lookBack),  # Touch down 1 hour support
        sig.Validate(f, val1='Touch_down_prev_day_high',          operator='>',   val2=80,               lookBack=lookBack),  # Touch down prev day high
    ]
    batch_add_ta(f, bonus_validations,  {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=sigRow)
    score_col = add_score(f, bonus_validations,  name=f'{ls}_ValBonus',  scoreType='mean', lookBack=lookBack, row=validationRow)
    return score_col



#------------------------------------------------------------
# ----------  S T R A T E G I E S  --------------------------
#------------------------------------------------------------

def STRATEGY_daily_goat(f, ls:str='LONG', lookBack:int=1, scoreRow:int=5):
    # todo 
    strat = sig.Strategy('GOAT', lookBack=lookBack)
    strat.add_reset(name='Cl < MA150', valToCheck='close', checkIf='<', colThreshold='MA_cl_150')
    strat.add_reset(name='Cl < MA50',  valToCheck='close', checkIf='<', colThreshold='MA_cl_50')

    strat.add_event(step=1, name='Gap MA50',    valToCheck='close', checkIf='>',  colThreshold='SHIIF_lo')   # close > prev low
    strat.add_event(step=1, name='MA50 Upward', valToCheck='MA50',  checkIf='>',  colThreshold='SHIIF_MA50') # MA50 is going up
    strat.add_event(step=1, name='VolSco > 50',    valToCheck='Score_L_Vol',  checkIf='>', colThreshold=50)


def STRATEGY_daily_consolidation_bo(f, lookBack:int=100, scoreRow:int=5):
    """Strategies are mainly triggers and validations (YES or NO). for additional nuance use seprate scores"""
    strat = sig.Strategy('BO', lookBack=lookBack)
    strat.add_reset(name='Cl < PPiv', valToCheck='close',  checkIf='<', colThreshold='FFILL_LP_lo_10')
    strat.add_reset(name='Cl < MA150', valToCheck='close', checkIf='<', colThreshold='MA_cl_150')

    strat.add_event     (step=1, name='ConsSco_1 > 50', valToCheck='Cons_Score_1', checkIf='>', colThreshold=50)    
    strat.add_event     (step=1, name='brk Cons',       valToCheck='close',        checkIf='>', colThreshold='CONS_UPPER_1')  
    strat.add_event     (step=1, name='VolSco > 50',    valToCheck='Score_L_Vol',  checkIf='>', colThreshold=50)
    strat.add_validation(step=1, name='Cl > MA50',      valToCheck='close',        checkIf='>', colThreshold='MA_cl_50')
    strat.add_validation(step=1, name='Cl > MA150',     valToCheck='close',        checkIf='>', colThreshold='MA_cl_150')
    strat.add_validation(step=1, name='Cl > Cons1',     valToCheck='close',        checkIf='>', colThreshold='CONS_UPPER_1')

    colours = ['cyan', 'yellow', 'green', 'red', 'orange', 'magenta', 'purple'] * 2
    f.add_ta_batch([
        TAData(ta.Ffill(colToFfill='LP_lo_10'), {}, '', 0),
        TAData(sig.Breaks(price_column='close', direction='above', metric_column='MA_cl_50', normRange=(0,1), lookBack=lookBack), {}, '', 0),
        TAData(strat, [{'dash': 'solid', 'color': color, 'width': 1} for color in colours], chart_type = 'lines+markers', row=scoreRow),
    ])


def STRATEGY_pullback_to_cons(f, ls='LONG', lookBack:int=100, scoreRow:int=5, majorPointSpan:int=10, minorPointSpan:int=2, atrSpan:int=50, minScores=50, chartType:str='line'):
    """Must happen for a pullback to be valid"""
    hp_col_1 = f'HP_hi_{majorPointSpan}'
    hp_col_2 = f'HP_hi_{minorPointSpan}'
    lp_col_1 = f'LP_lo_{majorPointSpan}'
    lp_col_2 = f'LP_lo_{minorPointSpan}'
    cons_col = 'CONS_UPPER_1' if ls == 'LONG' else 'CONS_LOWER_1'
    atr_col = f'ATR_{atrSpan}'
    ffill_hp_col_1 = f'FFILL_{hp_col_1}'
    ffill_lp_col_1 = f'FFILL_{lp_col_1}'

    # print(f'ffill_hp_col_1: {ffill_hp_col_1}')
    # print(f'ffill_lp_col_1: {ffill_lp_col_1}')
    # print(f'cons_col: {cons_col}')
    # print(f'atr_col: {atr_col}')
    # print(f'hp_col_1: {hp_col_1}')
    # print(f'hp_col_2: {hp_col_2}')
    # print(f'lp_col_1: {lp_col_1}')
    # print(f'lp_col_2: {lp_col_2}')

    if ls == 'LONG':
        strat = sig.Strategy('PB', lookBack=lookBack)
        strat.add_reset(name='Cl < PPiv', valToCheck='close',  checkIf='<', colThreshold=ffill_lp_col_1)

        strat.add_event (step=1, name='PB_Scores_Av',      valToCheck='PB_LH_ASC_Olap_Av',  checkIf='>', colThreshold=30)
        strat.add_event (step=1, name='RtcPiv > 50',  valToCheck='SigL_RtcPivs',  checkIf='>', colThreshold=minScores)
        strat.add_event (step=1, name='PBnrCons > 50',valToCheck='SigL_PBN_FFI',  checkIf='>', colThreshold=minScores)
        strat.add_validation(step=1, name='Cl > MA50',      valToCheck='close',        checkIf='>', colThreshold='MA_cl_50')
        strat.add_validation(step=1, name='Cl > MA150',     valToCheck='close',        checkIf='>', colThreshold='MA_cl_150')
        strat.add_validation(step=1, name='Cl > Cons1',     valToCheck='close',        checkIf='>', colThreshold='CONS_UPPER_1')

        colours = ['cyan', 'yellow', 'green', 'red', 'orange', 'magenta', 'purple'] * 2


        f.add_ta_batch([
            TAData(ta.ATR(span=atrSpan), {}, '', 0),
            TAData(ta.HPLP(hi_col='high', lo_col='low', span=majorPointSpan), [{'color': 'green', 'size': 10}, {'color': 'red', 'size': 5}], chart_type = 'points'),
            TAData(ta.HPLP(hi_col='high', lo_col='low', span=minorPointSpan), [{'color': 'green', 'size': 5},  {'color': 'red', 'size': 5}], chart_type = 'points'),
            TAData(ta.Ffill(colToFfill=hp_col_1), {}, '', 0),
            TAData(ta.Ffill(colToFfill=hp_col_2), {}, '', 0),
            TAData(ta.Ffill(colToFfill=lp_col_2), {}, '', 0),
            TAData(ta.Ffill(colToFfill=lp_col_1), {}, '', 0),
            TAData(ta.Ffill(colToFfill=cons_col), {}, '', 0),
            TAData(sig.PBPctHLLH            (ls=ls, normRange=(0,100), lookBack=lookBack, pointCol=hp_col_2, toCol=cons_col, atrCol=atr_col, atrMultiple=1), {'dash': 'solid', 'color': 'cyan', 'width': 1},    row=4, chart_type=chartType),
            TAData(sig.PBAllSameColour      (ls=ls, normRange=(0,100), lookBack=lookBack, pointCol=hp_col_2, toCol=cons_col, atrCol=atr_col, atrMultiple=1), {'dash': 'solid', 'color': 'magenta', 'width': 1}, row=4, chart_type=chartType),
            TAData(sig.PBOverlap            (ls=ls, normRange=(0,100), lookBack=lookBack, pointCol=hp_col_2, toCol=cons_col, atrCol=atr_col, atrMultiple=1), {'dash': 'solid', 'color': 'yellow', 'width': 1},  row=4, chart_type=chartType),
            TAData(sig.Score(rawName='PB_LH_ASC_Olap_Av', normRange=(0, 100), lookBack=lookBack, containsAllStrings=['PB'], scoreType='mean', weight=1), {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type=chartType, row=4),
            TAData(sig.Trace(name='RtcPivs', ls=ls, normRange=(0,100), lookBack=lookBack, fromCol=ffill_hp_col_1, toCol=ffill_lp_col_1, optimalRtc=50), {'dash': 'solid', 'color': 'green', 'width': 1}, row=4, chart_type=chartType),
            TAData(sig.PullbackNear         (ls=ls, normRange=(0,100), lookBack=lookBack, fromCol=hp_col_2, toCol='FFILL_CONS_UPPER_1', optimalRetracement=99, atrCol='ATR_50', atrMultiple=1), {'dash': 'solid', 'color': 'orange', 'width': 1}, row=4, chart_type=chartType),
            TAData(strat, [{'dash': 'solid', 'color': color, 'width': 1} for color in colours], chart_type = chartType, row=scoreRow),
            TAData(sig.Score(rawName='PBX_BO_Score', normRange=(0, 100), lookBack=lookBack, containsAllStrings=['Stgy_BO_total'], scoreType='mean', weight=1), {'dash': 'dot', 'color': 'yellow', 'width': 1}, chart_type=chartType, row=6),
            TAData(sig.Score(rawName='PBX_PB_Score', normRange=(0, 100), lookBack=lookBack, containsAllStrings=['Stgy_PB_total'], scoreType='mean', weight=1), {'dash': 'dot', 'color': 'cyan', 'width': 1}, chart_type=chartType, row=6),
            TAData(sig.Score(rawName='PBX_ALL_Scores', normRange=(0, 100), lookBack=lookBack, containsAllStrings=['PBX'], scoreType='mean', weight=1), {'dash': 'solid', 'color': 'lime', 'width': 2}, chart_type=chartType, row=6)
        ])
    else:
        print('STRATEGY_pullback_to_cons :: SHORT STRATEGY NOT IMPLEMENTED YET')



def VALIDATE_tbp(f, ls='LONG', atrSpan:int=10, lookBack:int=100, sigRow:int=3, validationRow:int=4):
    tbp_signals = [
        sig.TBP(ls=ls, atrCol=f'ATR_{atrSpan}',   barsToPlay=3, lookBack=lookBack),
        sig.TBP(ls=ls, atrCol=f'ATR_{atrSpan}',   barsToPlay=4, lookBack=lookBack),
    ]
    batch_add_ta(f, tbp_signals,  {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=sigRow)
    add_score(f, tbp_signals,  name=f'{ls}_TBP',  scoreType='max',  lookBack=lookBack, row=validationRow)


def VALIDATE_turnbars(f, ls='LONG',  atrSpan:int=10, lookBack:int=100, sigRow:int=3, validationRow:int=4):
    turnbar_signals = [
        sig.TurnBar(ls=ls, atrCol=f'ATR_{atrSpan}',  lookBack=lookBack),
    ]
    batch_add_ta(f, turnbar_signals,  {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type='line', row=sigRow)
    add_score(f, turnbar_signals,  name=f'{ls}_TurnBar',  scoreType='max',  lookBack=lookBack, row=validationRow)


def SCORE_VALID_BuySetup(f, ls='LONG', bswCol:str='', retestCol:str='', lookBack:int=100, TArow:int=3, scoreRow:int=4):
    buy_setup_signals = [
        sig.BuySetup(ls=ls, bswCol=bswCol, retestCol=retestCol, minCount=3, minBSW=0.5, minRetest=0.5, lookBack=lookBack)
    ]
    f.add_multi_ta(buy_setup_signals[0], [
        ChartArgs({'dash': 'solid', 'color': 'cyan', 'width':5},     chart_type='lines+markers', row=1, columns=[f'{ls[0].upper()}_BuySetup_Entry']),
        ChartArgs({'dash': 'solid', 'color': 'magenta', 'width': 5}, chart_type='lines+markers', row=1, columns=[f'{ls[0].upper()}_BuySetup_Stop']),
        ChartArgs({'dash': 'solid', 'color': 'cyan', 'width': 2},    chart_type='line', row=3, columns=[f'{ls[0].upper()}_BuySetup_isBuy']),
    ])
    
    add_score(f, buy_setup_signals,  name=f'{ls}_BuySetup',  scoreType='max',  lookBack=lookBack, row=scoreRow)

## --------------------------------------------------------------
## --------------- E X I T   S T R A T E G I E S -----------------
## --------------------------------------------------------------
def req_for_price_x(f, entryName, stopName, targetName, riskName, stopNameCol='StopName'):
    f.add_ta(ta.HPLP(hi_col='high', lo_col='low', span=10), [{'color': 'green', 'size': 10}, {'color': 'red', 'size': 10}], chart_type = 'points')
    f.add_ta(ta.HPLP(hi_col='high', lo_col='low', span=3), [{'color': 'green', 'size': 10}, {'color': 'red', 'size': 10}], chart_type = 'points')
    f.add_ta(ta.ATR(span=14),{'dash': 'dot', 'color': 'red', 'width': 1}, chart_type = 'ine', row=1)
    f.add_ta(ta.ACC('close', fast_ma=8, slow_ma=21, max_accel=50), {'dash': 'dot', 'color': 'red', 'width': 1}, chart_type = 'line', row=3)
    f.add_ta(ta.Ffill(colToFfill='HP_hi_3'), {'dash': 'dot', 'color': 'green', 'width': 1}, chart_type = 'line', row=1)
    f.add_ta(ta.Ffill(colToFfill='LP_lo_3'), {'dash': 'dot', 'color': 'red', 'width': 1}, chart_type = 'line', row=1)
    f.add_ta(ta.LowestHighest(hi_col='high', lo_col='low', span=3), [{'color': 'green', 'size': 1}, {'color': 'red', 'size': 5}], chart_type = 'points')
    f.add_ta(ta.LowestHighest(hi_col='high', lo_col='low', span=1), [{'color': 'green', 'size': 1}, {'color': 'red', 'size': 5}], chart_type = 'points')

    f.add_ta(ta.AddColumn(entryName), [{'color': 'yellow', 'size': 3}, {'color': 'red', 'size': 3}], chart_type='points', row=1)
    f.add_ta(ta.AddColumn(stopName), [{'color': 'magenta', 'size': 5}], chart_type='points', row=1, nameCol=stopNameCol)
    f.add_ta(ta.AddColumn(targetName), {'dash': 'solid', 'color': 'cyan', 'width': 3}, chart_type='lines+markers', row=1)
    f.add_ta(ta.AddColumn(riskName), {'dash': 'solid', 'color': 'red', 'width': 3}, chart_type='lines+markers', row=3)

