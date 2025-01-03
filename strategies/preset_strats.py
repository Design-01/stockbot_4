
import pandas as pd
import strategies.ta as ta
import strategies.signals as sig
from  strategies.ta import TAData
from frame import Frame
from data.random_data import RandomOHLCV
from dataclasses import dataclass



def require_ta_for_all(f, pointsSpan:int=10, atrSpan:int=50):
    f.add_ta_batch([
        TAData(ta.ATR(span=atrSpan), {'dash': 'solid', 'color': 'cyan', 'width': 1}, row=3, chart_type=''),
        TAData(ta.HPLP(hi_col='high', lo_col='low', span=pointsSpan), [{'color': 'green', 'size': 10}, {'color': 'red', 'size': 10}], chart_type = 'points'),
        TAData(ta.SupRes(hi_point_col=f'HP_hi_{pointsSpan}', lo_point_col=f'LP_lo_{pointsSpan}', atr_col=f'ATR_{atrSpan}', tolerance=1),
            [{'dash': 'solid', 'color': 'green', 'fillcolour': "rgba(0, 255, 0, 0.1)", 'width': 2}, # support # green = rgba(0, 255, 0, 0.1)
            {'dash': 'solid', 'color': 'red', 'fillcolour': "rgba(255, 0, 0, 0.1)", 'width': 2}], # resistance # red = rgba(255, 0, 0, 0.1)
            chart_type = 'support_resistance')
    ])

def import_to_daily_df(f, spy:pd.DataFrame=None, etf:pd.DataFrame=None, RSIRow:int=4):
    if spy is not None:
        f.import_data(spy, has_columns=['close'], prefix='SPY_')
        f.add_ta(ta.MansfieldRSI(close_col='close', market_col='SPY_close', span=14), {'dash': 'solid', 'color': 'yellow', 'width': 1}, row=RSIRow)
    if etf is not None:
        f.import_data(etf, has_columns=['close'], prefix='ETF_')
        f.add_ta(ta.MansfieldRSI(close_col='close', market_col='ETF_close', span=14), {'dash': 'solid', 'color': 'magenta', 'width': 1}, row=RSIRow)

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
    # Create the batch list with the correct color and size
    batch_list = [
        TAData(ta.MA(ma_col, period), {'dash': 'solid', 'color': ma_colour_map[period]['colour'], 'width': ma_colour_map[period]['size']}, row=row)
        for period in periods if period in ma_colour_map
    ]
    
    # Add the batch list to the technical analysis
    f.add_ta_batch(batch_list)

def gaps_ta(f, ls:str='LONG', pointCol:str='HP_hi_10', atrCol:str='ATR_50', lookBack:int=1, row:int=4, chartType:str='lines+markers'):
    l_or_s = ls[0]
    f.add_ta_batch([
        TAData(sig.IsGappedOverPivot(ls=ls, normRange=(0,1), pointCol=pointCol, lookBack=lookBack), {'dash': 'solid', 'color': 'blue', 'width': 1}, chart_type=chartType, row=row),
        TAData(sig.GappedPivots(ls=ls, normRange=(0, 3), pointCol=pointCol, span=400, lookBack=lookBack), {'dash': 'solid', 'color': 'orange', 'width': 1}, chart_type=chartType, row=row),
        TAData(sig.GappedRetracement(ls=ls, normRange=(0,100), pointCol=pointCol, atrCol=atrCol, lookBack=lookBack), {'dash': 'solid', 'color': 'magenta', 'width': 1}, chart_type=chartType, row=row),
        TAData(sig.GappedPastPivot(ls=ls, normRange=(0,100), atrCol=atrCol, pointCol=pointCol, lookBack=lookBack, maxAtrMultiple=10), {'dash': 'solid', 'color': 'red', 'width': 1}, chart_type=chartType, row=row),
        TAData(sig.GapSize(ls=ls, normRange=(0,300), pointCol=pointCol, atrCol=atrCol, lookBack=lookBack), {'dash': 'solid', 'color': 'red', 'width': 1}, chart_type=chartType, row=row),
        TAData(sig.Score(name=f'{ls}_Gaps', normRange=(0,100), lookBack=lookBack, cols=[f'Sig{l_or_s}_GPivs', f'Sig{l_or_s}_GRtc', f'Sig{l_or_s}_GPP', f'Sig{l_or_s}_GSiz'], scoreType='mean', weight=1), {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type=chartType, row=row)
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

def volume_ta(f, ls:str='LONG', ma:int=10, lookBack:int=100, scoreRow:int=3, chartType:str='lines+markers'):
    volMACol = f'MA_vo_{ma}'
    l_or_s = ls[0]
    f.add_ta_batch([
        TAData(ta.MA('volume', ma), {'dash': 'solid', 'color': 'yellow', 'width': 2}, row=2),
        TAData(sig.VolumeSpike(ls=ls, normRange=(0, 200), volMACol=volMACol, lookBack=lookBack), {'dash': 'solid', 'color': 'blue', 'width': 1}, chart_type=chartType, row=scoreRow),
        TAData(sig.VolumeROC(ls=ls, normRange=(0, 300), lookBack=lookBack), {'dash': 'solid', 'color': 'red', 'width': 1}, chart_type=chartType, row=scoreRow),
        TAData(sig.Score(name=f'{l_or_s}_Vol', cols=[f'Sig{l_or_s}_VolSpike', f'Sig{l_or_s}_VolROC'], scoreType='max', weight=1, lookBack=lookBack), {'dash': 'solid', 'color': 'yellow', 'width': 2}, chart_type=chartType, row=scoreRow)
    ])

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