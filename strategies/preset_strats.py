from dataclasses import dataclass
from copy import deepcopy

# my moudles
import strategies.signals as sig
import strategies.ta as ta
from chart.chart_args import ChartArgs

@dataclass
class TAPresetsBase:
    ca: ChartArgs = ChartArgs()
    atrSpan: int = 14
    pointSpanMajor: int = 10
    pointSpanMinor: int = 3
    supResTolerance: int = 1
    supResSeparation: int = 2
    lookBack: int = 100

    def __post_init__(self):
        self.ATR       = ta.ATR(span=self.atrSpan).add_plot_args(self.ca.ATR)
        self.HPLPMajor = ta.HPLP(hi_col='high', lo_col='low', span=self.pointSpanMajor).add_plot_args(self.ca.HPLPMajor)
        self.HPLPMinor = ta.HPLP(hi_col='high', lo_col='low', span=self.pointSpanMinor).add_plot_args(self.ca.HPLPMinor)
        self.SupRes    = ta.SupResAllRows(hi_point_col=self.HPLPMajor.name_hp, lo_point_col=self.HPLPMajor.name_lp, tolerance=self.supResTolerance, separation=self.supResSeparation, atr_col=self.ATR.name, rowsToUpdate=self.lookBack).add_plot_args([self.ca.SupZone, self.ca.ResZone, self.ca.SupLine, self.ca.ResLine]) #! in production this can be used as it only looks at the last row
        self.l_base = [self.ATR, self.HPLPMajor, self.HPLPMinor, self.SupRes]
        self.add_to_ta_list(self.l_base)

    def get_ta_list(self):
        return self.ta_list
    
    def add_to_ta_list(self, ta_list:list):
        if not hasattr(self, 'ta_list'):
            self.ta_list = []
        self.ta_list += ta_list # #! Do not use append as it will create a list of lists





@dataclass
class TAPresets1D(TAPresetsBase):
    name: str = '1D'
    ls: str = 'LONG'    
    lookBack: int = 10
    GappedWRBs_normRange:      tuple[int, int] = (0,100)
    GappedPivots_normRange:    tuple[int, int] = (0, 3)
    GappedPastPivot_normRange: tuple[int, int] = (0,100)
    BarSW_normRange:           tuple[int, int] = (-3,3)
    RoomToMove_normRange:      tuple[int, int] = (0,5)
    isSpy: bool = False

    def __post_init__(self):
        super().__post_init__()
        print(f'{self.name} :: {self.ls} :: {self.lookBack}')
        self.MA50   = ta.MA(maCol='close', period=50).add_plot_args(self.ca.MA50)
        self.MA150  = ta.MA(maCol='close', period=150).add_plot_args(self.ca.MA150)
        self.MA200  = ta.MA(maCol='close', period=200).add_plot_args(self.ca.MA200)
        self.l_ma = [self.MA50, self.MA150, self.MA200]
        self.add_to_ta_list(self.l_ma)

        # Signals (Primary)
        self.BarSW           = sig.BarSW(ls=self.ls, normRange=self.BarSW_normRange, atrCol=self.ATR.name, lookBack=self.lookBack).add_plot_args(self.ca.BarSW) 
        self.add_to_ta_list([self.BarSW])


        self.l_sigs = []

        if not self.isSpy:
            self.RS = ta.RS(comparisonPrefix='SPY', ma=10, atr=50).add_plot_args(self.ca.RS)
            normRange = (0, 5) if self.ls == 'LONG' else (-5, 0)
            self.s_RSScore = sig.Score(name='RS', ls=self.ls, sigs=[self.RS], scoreType='mean', operator='>=', threshold=50, normRange=normRange, invertScoreIfShort=True, lookBack=self.lookBack).add_plot_args(self.ca.RSScore)
            self.l_sigs += [self.RS, self.s_RSScore]

        piv_name = self.HPLPMajor.name_hp if self.ls == 'LONG' else self.HPLPMajor.name_lp
 
        self.RoomToMove      = sig.RoomToMove(ls=self.ls, tgetCol='Res_1_Lower' if self.ls == 'LONG' else 'Sup_1_Upper', atrCol=self.ATR.name, unlimitedVal=5, normRange=self.RoomToMove_normRange, lookBack=self.lookBack).add_plot_args(self.ca.RoomToMove)
        self.GappedWRBs      = sig.GappedWRBs(ls=self.ls, bswCol=self.BarSW.name, normRange=self.GappedWRBs_normRange, lookBack=self.lookBack).add_plot_args(self.ca.GappedWRBs)
        self.GappedPivots    = sig.GappedPivots(ls=self.ls, normRange=self.GappedPivots_normRange, pointCol=piv_name, spanPivots=10, lookBack=self.lookBack).add_plot_args(self.ca.GappedPivots)
        self.GappedPastPivot = sig.GappedPastPivot(ls=self.ls, normRange=self.GappedPastPivot_normRange, atrCol=self.ATR.name, pointCol=piv_name, lookBack=self.lookBack, maxAtrMultiple=10).add_plot_args(self.ca.GappedPastPivot)
        self.l_sigs += [self.RoomToMove, self.GappedWRBs, self.GappedPivots, self.GappedPastPivot] # NOTE Order of list is important when one sig refernecs anotehr sig
        self.add_to_ta_list(self.l_sigs)

        # Validate Signals
        self.l_vads = []
        if not self.isSpy:
            self.v_RS = sig.Validate(ls=self.ls, val1=self.RS.name, operator='>', val2=1, lookBack=self.lookBack).add_plot_args(self.ca.RS)
            self.l_vads += [self.v_RS]

        self.v_RoomToMove      = sig.Validate(ls=self.ls, val1=self.RoomToMove.name,      operator='>', val2=1,  lookBack=self.lookBack)  # room to move
        self.v_GappedWRBs      = sig.Validate(ls=self.ls, val1=self.GappedWRBs.name,      operator='>', val2=50, lookBack=self.lookBack)  # gapped retracement of 50% or more
        self.v_GappedPivots    = sig.Validate(ls=self.ls, val1=self.GappedPivots.name,    operator='>', val2=1,  lookBack=self.lookBack)  # gapped pivots in the last 10 bars
        self.v_GappedPastPivot = sig.Validate(ls=self.ls, val1=self.GappedPastPivot.name, operator='>', val2=80, lookBack=self.lookBack)  # gapped past pivot
        self.l_vads += [self.v_GappedWRBs, self.v_GappedPivots, self.v_GappedPastPivot, self.v_RoomToMove]
        self.add_to_ta_list(self.l_vads)

        # # Scores
        # meanWeights Used for max_plus_weighted_mean_capped / List of weights to apply to the non-maximum values
        self.s_1D  = sig.Score(name='1Ds',  ls=self.ls, sigs=self.l_sigs, scoreType='mean', normRange=(0,100), operator='>=', threshold=50, meanWeights=[0.8,0.5,0.3],  lookBack=self.lookBack).add_plot_args(self.ca.Score1D) 
        self.sv_1D = sig.Score(name='1Dv', ls=self.ls, sigs=self.l_vads, scoreType='mean', normRange=(0,100), operator='>=', threshold=100, meanWeights=[0.8,0.5,0.3],  lookBack=self.lookBack).add_plot_args(self.ca.ScoreV1D)
        self.add_to_ta_list([self.s_1D, self.sv_1D])
8



@dataclass
class TAPresets1H(TAPresetsBase):
    name: str = '1H'
    ls: str = 'LONG'
    lookBack: int = 10
    volChgPctThreshold: int = 80
    isSpy: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.MA21 = ta.MA(maCol='close', period=21).add_plot_args(self.ca.MA21)
        self.MA50 = ta.MA(maCol='close', period=50).add_plot_args(self.ca.MA50)

        # Signals (Primary) - Volume
        self.VolAcum   = ta.VolumeAccumulation().add_plot_args(self.ca.volAcum)
        self.VolChgPct = ta.VolumeTimeOfDayChangePct(lookbackDays=self.lookBack).add_plot_args(self.ca.VolChgPct)
        self.l_sigs = [self.VolAcum, self.VolChgPct]

        # Validate Signals
        self.v_VolChgPct = sig.Validate(val1=self.VolChgPct.name, operator='>', val2=self.volChgPctThreshold, lookBack=self.lookBack)
        self.l_vads = [self.v_VolChgPct]

        # Scores
        self.s_1H  = sig.Score(name='1H',  ls=self.ls, sigs=[self.VolChgPct],   scoreType='mean', operator='>',  threshold=50,  normRange=(0,100), lookBack=self.lookBack).add_plot_args(self.ca.Score1H)
        self.sv_1H = sig.Score(name='1Hv', ls=self.ls, sigs=[self.v_VolChgPct], scoreType='mean', operator='>=', threshold=100, normRange=(0,1), lookBack=self.lookBack).add_plot_args(self.ca.ScoreV1H)

        # fails
        self.s_1H_fail  = sig.Score(name='1Hf', ls=self.ls, sigs=[self.VolChgPct], scoreType='mean', operator='<', threshold=30, normRange=(0,100), lookBack=self.lookBack).add_plot_args(self.ca.Score1H)

        self.add_to_ta_list(self.l_sigs + self.l_vads + [self.s_1H, self.sv_1H, self.s_1H_fail])

        
        strat = sig.Strategy(name='strat1H', lookBack=10).add_plot_args(self.ca.Stategy1)
        strat.add_step(scoreObj=self.s_1H, failObj=self.s_1H_fail, ifFailStartFromStep=1)
        self.add_to_ta_list([strat])


@dataclass
class TAPresets5M2M1M(TAPresetsBase):
    name: str = '5M2M1M'
    ls: str = 'LONG'
    lookBack: int = 100
    set_levels_SupRes    : bool = False # True
    set_levels_prevDay   : bool = False # True
    set_levels_preMarket : bool = False # True
    set_levels_intraday  : bool = False # True
    set_levels_935       : bool = False # True
    set_levels_SupResHour: bool = False
    set_levels_SupResDay : bool = False
    set_pullback         : bool = False # True
    set_pullback_bounus  : bool = True # True

    touch_toTouchAtrScale: int = 2
    touch_pastTouchAtrScale: int = 1
    pb_minPbLen: int = 3
    PctHLLH_normRange:           tuple[int, int] = (0,100)
    ASC_normRange:               tuple[int, int] = (0,100)
    CoC_ByCountOpBars_normRange: tuple[int, int] = (0,100)
    Overlap_normRange:           tuple[int, int] = (0,100)
    retest_atrRange:             tuple[int, int] = (-0.1,0.1)
    retest_normRange:            tuple[int, int] = (0,3)
    retest_rollingLen:           int = 5
    isSpy: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.MA9  = ta.MA(maCol='close', period=9).add_plot_args(self.ca.MA9)
        self.MA21 = ta.MA(maCol='close', period=21).add_plot_args(self.ca.MA21)
        self.VolMA = ta.MA(maCol='volume', period=14).add_plot_args(self.ca.MAVol)
        self.l_ma = [self.MA9, self.MA21, self.VolMA]
        self.add_to_ta_list(self.l_ma)


        if self.set_pullback:
            self.PB_PctHLLH           = sig.PB_PctHLLH          (ls=self.ls, hpCol=self.HPLPMinor.name_hp, lpCol=self.HPLPMinor.name_lp,  atrCol=self.ATR.name, normRange=self.PctHLLH_normRange,           minPbLen=self.pb_minPbLen, lookBack=self.lookBack).add_plot_args(self.ca.PB_PctHLLH)
            self.PB_ASC               = sig.PB_ASC              (ls=self.ls, hpCol=self.HPLPMinor.name_hp, lpCol=self.HPLPMinor.name_lp,  atrCol=self.ATR.name, normRange=self.ASC_normRange,               minPbLen=self.pb_minPbLen, lookBack=self.lookBack).add_plot_args(self.ca.PB_ASC)
            self.PB_CoC_ByCountOpBars = sig.PB_CoC_ByCountOpBars(ls=self.ls, hpCol=self.HPLPMinor.name_hp, lpCol=self.HPLPMinor.name_lp,  atrCol=self.ATR.name, normRange=self.CoC_ByCountOpBars_normRange, minPbLen=self.pb_minPbLen, lookBack=self.lookBack).add_plot_args(self.ca.PB_CoC_ByCountOpBars)
            self.PB_Overlap           = sig.PB_Overlap          (ls=self.ls, hpCol=self.HPLPMinor.name_hp, lpCol=self.HPLPMinor.name_lp,  atrCol=self.ATR.name, normRange=self.Overlap_normRange,           minPbLen=self.pb_minPbLen, lookBack=self.lookBack).add_plot_args(self.ca.PB_Overlap)
            self.l_pullback = [self.PB_PctHLLH, self.PB_ASC, self.PB_CoC_ByCountOpBars, self.PB_Overlap]
            self.add_to_ta_list(self.l_pullback)

            self.s_pullback_passed = sig.Score(name='s_pullback_passed', sigs=self.l_pullback, scoreType='mean', operator='>=', threshold=50, lookBack=self.lookBack).add_plot_args(deepcopy(self.ca.DefaultPassed))
            self.s_pullback_failed = sig.Score(name='s_pullback_failed', sigs=self.l_pullback, scoreType='mean', operator='<=', threshold=0,  lookBack=self.lookBack).add_plot_args(deepcopy(self.ca.DefaultFailed))
            self.add_to_ta_list([self.s_pullback_passed, self.s_pullback_failed])




        key_lev_base_config = {
            "lookBack"   : self.lookBack,
            "normRange"  : (0, 1),
            "touchCol"   : '', #! NOTE: this is set in the key level object
            "levelCol"   : '', #! NOTE: this is set in the key level object
            "levelTol"   : 0.5,
            "timeDecay"  : 0.005,
            "volChgCol"  : 'volume',
            "atrCol"     : self.ATR.name,
            "tailSettings"     : {'weight': 0.35, 'norm': 3, 'confBonThresh': 0.5}, # for each touch point: the weight of the tail, the normalisation range, and the threshold for the confluence bonus
            "volSettings"      : {'weight': 0.35, 'norm': 1, 'confBonThresh': 0.5},
            "rejectionSettings": {'weight': 0.2, 'norm': 10, 'confBonThresh': 0.5},
            "spanSettings"     : {'weight': 0.1, 'norm': 20, 'confBonThresh': 0.5},
            "timeDecaySettings": {'weight': 0.1, 'norm': None, 'confBonThresh': 0.5},
            "scoreWeights"     : [0.8, 0.5, 0.3], # each touch point at this level is scored.  this is the weight of each touch point in the score. The max score is always inlcuded,  the weight are for the remainder which gets added to the max score. Order is next highest and so on.
            "confBonThresh"    : {'tail': 0.5, 'vol': 0.50, 'rejection': 0.50, 'span': 0.50, 'decay': 0.50}, # the threshold for the confluence bonus
            "confWeight"       : 0.2, # the weight of the confluence in the score
            "confIncluded"     :True, # decide if to include the confluence in the score
            "debug": False #  Truns on various debug print statements
        }

        # Levels #$NOTE: uses deep copy so as not to overwrite the chart plots as their dataCols get set by the sig or ta object
        self.l_levels = []
        self.l_touches = []
        self.l_levscores = []
        self.l_bonus = []
        direction = 'down' if self.ls == 'LONG' else 'up'
        col1 = 'Sup_1_Upper'        if direction == 'down' else 'Res_1_Lower'
        col2 = '1 hour_Sup_1_Upper' if direction == 'down' else '1 hour_Res_1_Lower'
        col3 = '1 day_Sup_1_Upper'  if direction == 'down' else '1 day_Res_1_Lower'


        if self.set_levels_SupRes:
            # the actual SupRes levels are calculated in the base class
            self.touchSupRes = sig.TouchWithBar(ls=self.ls, atrCol=self.ATR.name, valCol=col1, direction=direction, toTouchAtrScale=self.touch_toTouchAtrScale, pastTouchAtrScale=self.touch_pastTouchAtrScale, lookBack=self.lookBack).add_plot_args(deepcopy(self.ca.TouchWithBar)) 
            self.add_to_ta_list([self.touchSupRes])
            
            config_supres = key_lev_base_config.copy()
            config_supres['levelCol'] = col1
            config_supres['touchCol'] = self.touchSupRes.name
            self.ScoreKeyLev_SupRes = sig.ScoreKeyLevel(ls=self.ls, **config_supres).add_plot_args(deepcopy(self.ca.DefaultScore))
            self.l_levscores += [self.ScoreKeyLev_SupRes]
            self.add_to_ta_list(self.l_levscores)

        if self.set_levels_prevDay:
            # levels
            self.LevelPrevDayHi = ta.Levels(level='prev_day_high', ffill=True).add_plot_args(deepcopy(self.ca.LevPrevDay))
            self.LevelPrevDayLo = ta.Levels(level='prev_day_low', ffill=True).add_plot_args(deepcopy(self.ca.LevPrevDay))
            self.l_levels += [self.LevelPrevDayHi, self.LevelPrevDayLo]
            self.add_to_ta_list([self.LevelPrevDayHi, self.LevelPrevDayLo])
            # touches
            self.touchPrevDayLo   = sig.TouchWithBar(ls=self.ls, atrCol=self.ATR.name, valCol='prev_day_low', direction=direction, toTouchAtrScale=self.touch_toTouchAtrScale, pastTouchAtrScale=self.touch_pastTouchAtrScale, lookBack=self.lookBack).add_plot_args(deepcopy(self.ca.TouchWithBar))
            self.touchPrevDayHi   = sig.TouchWithBar(ls=self.ls, atrCol=self.ATR.name, valCol='prev_day_high', direction=direction, toTouchAtrScale=self.touch_toTouchAtrScale, pastTouchAtrScale=self.touch_pastTouchAtrScale, lookBack=self.lookBack).add_plot_args(deepcopy(self.ca.TouchWithBar))
            self.l_touches += [self.touchPrevDayLo, self.touchPrevDayHi]
            self.add_to_ta_list([self.touchPrevDayLo, self.touchPrevDayHi])
            # key level scores
            prev_day_high_config = key_lev_base_config.copy()
            prev_day_low_config  = key_lev_base_config.copy()
            prev_day_high_config['levelCol']   = self.touchPrevDayHi.name_level
            prev_day_low_config['levelCol']    = self.touchPrevDayLo.name_level
            prev_day_high_config['touchCol'] = self.touchPrevDayHi.name
            prev_day_low_config['touchCol']  = self.touchPrevDayLo.name
            self.ScoreLev_PrevDayHi = sig.ScoreKeyLevel(ls=self.ls, **prev_day_high_config).add_plot_args(deepcopy(self.ca.DefaultScore))
            self.ScoreLev_PrevDayLo = sig.ScoreKeyLevel(ls=self.ls, **prev_day_low_config).add_plot_args(deepcopy(self.ca.DefaultScore))
            self.l_levscores += [self.ScoreLev_PrevDayHi, self.ScoreLev_PrevDayLo]
            self.add_to_ta_list([self.ScoreLev_PrevDayHi, self.ScoreLev_PrevDayLo])

        if self.set_levels_preMarket:
            # levels
            self.LevelPreMktHi = ta.Levels(level='pre_mkt_high', ffill=True).add_plot_args(deepcopy(self.ca.LevPreMkt))
            self.LevelPreMktLo = ta.Levels(level='pre_mkt_low', ffill=True).add_plot_args(deepcopy(self.ca.LevPreMkt))
            self.l_levels += [self.LevelPreMktHi, self.LevelPreMktLo]
            self.add_to_ta_list([self.LevelPreMktHi, self.LevelPreMktLo])
            # touches
            self.touchPreMktLo    = sig.TouchWithBar(ls=self.ls, atrCol=self.ATR.name, valCol='pre_mkt_low', direction=direction, toTouchAtrScale=self.touch_toTouchAtrScale, pastTouchAtrScale=self.touch_pastTouchAtrScale, lookBack=self.lookBack).add_plot_args(deepcopy(self.ca.TouchWithBar))
            self.touchPreMktHi    = sig.TouchWithBar(ls=self.ls, atrCol=self.ATR.name, valCol='pre_mkt_high', direction=direction, toTouchAtrScale=self.touch_toTouchAtrScale, pastTouchAtrScale=self.touch_pastTouchAtrScale, lookBack=self.lookBack).add_plot_args(deepcopy(self.ca.TouchWithBar))
            self.l_touches += [self.touchPreMktLo, self.touchPreMktHi]
            self.add_to_ta_list([self.touchPreMktLo, self.touchPreMktHi])
            # key level scores
            pre_mkt_high_config = key_lev_base_config.copy()
            pre_mkt_low_config  = key_lev_base_config.copy()
            pre_mkt_high_config['levelCol']   = self.touchPreMktHi.name_level
            pre_mkt_low_config['levelCol']    = self.touchPreMktLo.name_level
            pre_mkt_high_config['touchCol'] = self.touchPreMktHi.name
            pre_mkt_low_config['touchCol']  = self.touchPreMktLo.name
            self.ScoreLev_PreMktHi = sig.ScoreKeyLevel(ls=self.ls, **pre_mkt_high_config).add_plot_args(deepcopy(self.ca.DefaultScore))
            self.ScoreLev_PreMktLo = sig.ScoreKeyLevel(ls=self.ls, **pre_mkt_low_config).add_plot_args(deepcopy(self.ca.DefaultScore))
            self.l_levscores += [self.ScoreLev_PreMktHi, self.ScoreLev_PreMktLo]
            self.add_to_ta_list([self.ScoreLev_PreMktHi, self.ScoreLev_PreMktLo])

        if self.set_levels_intraday:
            # levels
            self.LevelIntraHi     = ta.Levels(level='intraday_high', ffill=False).add_plot_args(deepcopy(self.ca.LevIntraDay))
            self.LevelIntraLo     = ta.Levels(level='intraday_low', ffill=False).add_plot_args(deepcopy(self.ca.LevIntraDay))
            self.l_levels += [self.LevelIntraHi, self.LevelIntraLo]
            self.add_to_ta_list([self.LevelIntraHi, self.LevelIntraLo])
            # touches
            self.touchIntyraHi    = sig.TouchWithBar(ls=self.ls, atrCol=self.ATR.name, valCol='intraday_high', direction=direction, toTouchAtrScale=self.touch_toTouchAtrScale, pastTouchAtrScale=self.touch_pastTouchAtrScale, lookBack=self.lookBack).add_plot_args(deepcopy(self.ca.TouchWithBar))
            self.touchIntyraLo    = sig.TouchWithBar(ls=self.ls, atrCol=self.ATR.name, valCol='intraday_low', direction=direction, toTouchAtrScale=self.touch_toTouchAtrScale, pastTouchAtrScale=self.touch_pastTouchAtrScale, lookBack=self.lookBack).add_plot_args(deepcopy(self.ca.TouchWithBar))
            self.l_touches += [self.touchIntyraHi, self.touchIntyraLo]
            self.add_to_ta_list([self.touchIntyraHi, self.touchIntyraLo])
            # key level scores
            intra_day_high_config = key_lev_base_config.copy()
            intra_day_low_config  = key_lev_base_config.copy()
            intra_day_high_config['levelCol']   = self.touchIntyraHi.name_level
            intra_day_low_config['levelCol']    = self.touchIntyraLo.name_level
            intra_day_high_config['touchCol'] = self.touchIntyraHi.name
            intra_day_low_config['touchCol']  = self.touchIntyraLo.name
            self.ScoreKeyLev_intradayHi = sig.ScoreKeyLevel(ls=self.ls, **intra_day_high_config).add_plot_args(deepcopy(self.ca.DefaultScore))
            self.ScoreKeyLev_intradayLo = sig.ScoreKeyLevel(ls=self.ls, **intra_day_low_config).add_plot_args(deepcopy(self.ca.DefaultScore))
            self.l_levscores += [self.ScoreKeyLev_intradayHi, self.ScoreKeyLev_intradayLo]
            self.add_to_ta_list([self.ScoreKeyLev_intradayHi, self.ScoreKeyLev_intradayLo])

        if self.set_levels_935:
            # levels
            self.LevelIntraHi0935 = ta.Levels(level='intraday_high_9.35', ffill=False).add_plot_args(deepcopy(self.ca.Lev935))
            self.LevelIntraLo0935 = ta.Levels(level='intraday_low_9.35', ffill=False).add_plot_args(deepcopy(self.ca.Lev935))
            self.l_levels += [self.LevelIntraHi0935, self.LevelIntraLo0935]
            self.add_to_ta_list([self.LevelIntraHi0935, self.LevelIntraLo0935])
            # touches
            self.touchIntyra935Hi = sig.TouchWithBar(ls=self.ls, atrCol=self.ATR.name, valCol='intraday_high_9.35', direction=direction, toTouchAtrScale=self.touch_toTouchAtrScale, pastTouchAtrScale=self.touch_pastTouchAtrScale, lookBack=self.lookBack).add_plot_args(deepcopy(self.ca.TouchWithBar))
            self.touchIntyra935Lo = sig.TouchWithBar(ls=self.ls, atrCol=self.ATR.name, valCol='intraday_low_9.35', direction=direction, toTouchAtrScale=self.touch_toTouchAtrScale, pastTouchAtrScale=self.touch_pastTouchAtrScale, lookBack=self.lookBack).add_plot_args(deepcopy(self.ca.TouchWithBar))
            self.l_touches += [self.touchIntyra935Hi, self.touchIntyra935Lo]
            self.add_to_ta_list([self.touchIntyra935Hi, self.touchIntyra935Lo])
            # key level scores
            intra_day_935_high_config = key_lev_base_config.copy()
            intra_day_935_low_config  = key_lev_base_config.copy()
            intra_day_935_high_config['levelCol']   = self.touchIntyra935Hi.name_level
            intra_day_935_low_config['levelCol']    = self.touchIntyra935Lo.name_level
            intra_day_935_high_config['touchCol'] = self.touchIntyra935Hi.name
            intra_day_935_low_config['touchCol']  = self.touchIntyra935Lo.name
            self.ScoreLevs_935Hi = sig.ScoreKeyLevel(ls=self.ls, **intra_day_935_high_config).add_plot_args(deepcopy(self.ca.DefaultScore))
            self.ScoreLevs_935Lo = sig.ScoreKeyLevel(ls=self.ls, **intra_day_935_low_config).add_plot_args(deepcopy(self.ca.DefaultScore))
            self.l_levscores += [self.ScoreLevs_935Hi, self.ScoreLevs_935Lo]
            self.add_to_ta_list([self.ScoreLevs_935Hi, self.ScoreLevs_935Lo])

        #! TODO: add the 1H levels to the chart.  self.set_levels_SupResHour and self.set_levels_SupResDay set to False by default
        if self.set_levels_SupResHour:
            # levels get imported from the hour chart

            # touches
            self.touchSupRes1Hour = sig.TouchWithBar(ls=self.ls, atrCol=self.ATR.name, valCol=col2, direction=direction, toTouchAtrScale=self.touch_toTouchAtrScale, pastTouchAtrScale=self.touch_pastTouchAtrScale, lookBack=self.lookBack).add_plot_args(deepcopy(self.ca.TouchWithBar))
            self.l_touches += [self.touchSupRes1Hour]
            self.add_to_ta_list([self.touchSupRes1Hour])
            # key level scores
            sup_res_config = key_lev_base_config.copy()
            sup_res_config['levelCol'] = col2
            sup_res_config['touchCol'] = self.touchSupRes1Hour.name
            self.ScoreKeyLev_SupRes = sig.ScoreKeyLevel(ls=self.ls, **sup_res_config).add_plot_args(deepcopy(self.ca.DefaultScore))
            self.l_levscores += [self.ScoreKeyLev_SupRes]
            self.add_to_ta_list([self.ScoreKeyLev_SupRes])

        #! TODO: add the 1D levels to the chart.  self.set_levels_SupResHour and self.set_levels_SupResDay set to False by default
        if self.set_levels_SupResDay:
            # levels get imported from the day chart

            # touches
            self.touchSupRes1Day  = sig.TouchWithBar(ls=self.ls, atrCol=self.ATR.name, valCol=col3, direction=direction, toTouchAtrScale=self.touch_toTouchAtrScale, pastTouchAtrScale=self.touch_pastTouchAtrScale, lookBack=self.lookBack).add_plot_args(deepcopy(self.ca.TouchWithBar))
            self.l_touches += [self.touchSupRes1Hour, self.touchSupRes1Day]
            self.add_to_ta_list([self.touchSupRes1Hour, self.touchSupRes1Day])
            # key level scores
            sup_res_config = key_lev_base_config.copy()
            sup_res_config['levelCol'] = col2
            sup_res_config['touchCol'] = self.touchSupRes1Hour.name
            self.ScoreKeyLev_SupRes = sig.ScoreKeyLevel(ls=self.ls, **sup_res_config).add_plot_args(deepcopy(self.ca.DefaultScore))
            self.l_levscores += [self.ScoreKeyLev_SupRes]
            self.add_to_ta_list([self.ScoreKeyLev_SupRes])

        

        # Score Touches
        if len(self.l_touches) > 0:
            self.s_touch = sig.Score(name='s_touch', ls=self.ls, sigs=self.l_levscores, scoreType='max_plus_weighted_mean_capped', operator='>=', threshold=50, lookBack=self.lookBack).add_plot_args(deepcopy(self.ca.DefaultScore))
            self.add_to_ta_list([self.s_touch])

        if self.set_pullback_bounus:
            barswLS  = 'SHORT' if self.ls == 'LONG' else 'SHORT'
            testCol  = 'low' if self.ls == 'LONG' else 'high'
            # self.barSW   = sig.BarSW(ls=self.ls, normRange=(0,1), atrCol=self.ATR.name, lookBack=self.lookBack).add_plot_args(self.ca.BarSW)
            self.BarTail = sig.BarTail(ls=self.ls, normRange=(0,1), atrCol=self.ATR.name, lookBack=self.lookBack).add_plot_args(self.ca.BarTail)
            self.NBB     = sig.NarrowBobyBar(ls=self.ls, normRange=(0,1), atrCol=self.ATR.name, lookBack=self.lookBack).add_plot_args(self.ca.NBB)
            self.volSpike = sig.VolumeSpike(ls=self.ls, normRange=(0,1), volMACol=self.VolMA.name, lookBack=self.lookBack).add_plot_args(self.ca.VolSpike)
            self.retestHL = sig.Retest(ls=self.ls, atrCol=self.ATR.name, direction=direction, valCol=testCol, withinAtrRange=self.retest_atrRange, rollingLen=self.retest_rollingLen, lookBack=self.lookBack, normRange=self.retest_normRange).add_plot_args(deepcopy(self.ca.Retest))
            self.l_bonus = [self.BarTail, self.NBB, self.volSpike, self.retestHL]
            self.s_bonus = sig.Score(name='s_bonus', ls=self.ls, sigs=self.l_bonus, scoreType='max_plus_weighted_mean_capped', operator='>=', threshold=50, lookBack=self.lookBack).add_plot_args(deepcopy(self.ca.DefaultScore))
            self.add_to_ta_list(self.l_bonus + [self.s_bonus])

            #! todo -  create is valid pullback as a signal then use that in the strategy in each bonus item.  This will help remove the noise of the barSW and NBB etc and targte only where it makes sense.
            #! todo - apply isValidPullback to BarTail
            #! todo - Apply isValidPullback to NBB
            #! todo - apply isValidPullback to VolSpike
            #! todo - apply isValidPullback to Retest
 
        # # # Retest
        # self.retestHP = sig.Retest(ls=self.ls, atrCol=self.ATR.name, direction=direction, valCol=self.HPLPMinor.hi_col, withinAtrRange=self.retest_atrRange, rollingLen=self.retest_rollingLen, lookBack=self.lookBack, normRange=self.retest_normRange).add_plot_args(deepcopy(self.ca.Retest)) 
        # self.retestLP = sig.Retest(ls=self.ls, atrCol=self.ATR.name, direction=direction, valCol=self.HPLPMinor.lo_col, withinAtrRange=self.retest_atrRange, rollingLen=self.retest_rollingLen, lookBack=self.lookBack, normRange=self.retest_normRange).add_plot_args(deepcopy(self.ca.Retest)) 
        # self.retestLo = sig.Retest(ls=self.ls, atrCol=self.ATR.name, direction=direction, valCol='low',                 withinAtrRange=self.retest_atrRange, rollingLen=self.retest_rollingLen, lookBack=self.lookBack, normRange=self.retest_normRange).add_plot_args(deepcopy(self.ca.Retest)) 
        # self.retestHi = sig.Retest(ls=self.ls, atrCol=self.ATR.name, direction=direction, valCol='high',                withinAtrRange=self.retest_atrRange, rollingLen=self.retest_rollingLen, lookBack=self.lookBack, normRange=self.retest_normRange).add_plot_args(deepcopy(self.ca.Retest)) 
        # self.l_retests = [self.retestHP, self.retestLP, self.retestLo, self.retestHi]
        # self.add_to_ta_list(self.l_retests)

        # # # Time of day
        # self.v_past_935 = sig.Validate(val1='idx', operator='t>t', val2='09:34', lookBack=self.lookBack).add_plot_args(self.ca.NoPlot)
        # self.v_past_EOD = sig.Validate(val1='idx', operator='t>t', val2='15:35', lookBack=self.lookBack).add_plot_args(self.ca.NoPlot)
        # self.s_past_935 = sig.Score(name='s_past_935', sigs=[self.v_past_935], scoreType='all', operator='>=', threshold=100, lookBack=self.lookBack).add_plot_args(deepcopy(self.ca.DefaultScore))
        # self.s_past_EOD = sig.Score(name='s_past_EOD', sigs=[self.v_past_EOD], scoreType='all', operator='>=', threshold=100, lookBack=self.lookBack).add_plot_args(deepcopy(self.ca.DefaultScore))
        # self.add_to_ta_list([self.v_past_935, self.v_past_EOD, self.s_past_935, self.s_past_EOD])

        # # Validate Breaks
        # self.v_above_5min     = sig.Validate(val1='close', operator='>',     val2='pre_mkt_high',       lookBack=self.lookBack)  # close > pre_mkt_high
        # self.v_above_premtkHi = sig.Validate(val1='close', operator='>',     val2='intraday_high_9.35', lookBack=self.lookBack)  # close > intraday_high_9.35
        # self.v_below_5min     = sig.Validate(val1='close', operator='<',     val2='pre_mkt_low',        lookBack=self.lookBack)
        # self.v_below_premtkHi = sig.Validate(val1='close', operator='<',     val2='intraday_low_9.35',  lookBack=self.lookBack)
        # self.s_above_5minPreMkt = sig.Score(name='s_above_5minPreMkt', sigs=[self.v_above_5min, self.v_above_premtkHi], scoreType='mean', operator='>=', threshold=50, lookBack=self.lookBack)
        # self.s_below_5minPreMkt = sig.Score(name='s_below_5minPreMkt', sigs=[self.v_below_5min, self.v_below_premtkHi], scoreType='mean', operator='>=', threshold=50, lookBack=self.lookBack)
        
        
        # # Trends
        # self.isMA21Trending = sig.IsMATrending(ls=self.ls, maCol=self.MA21.name, lookBack=self.lookBack, normRange=(0,1)).add_plot_args(self.ca.MA21)
        # self.isMajorPointTrending = sig.IsPointsTrending(ls=self.ls, hpCol=self.HPLPMajor.name_hp, lpCol=self.HPLPMajor.name_lp, lookBack=self.lookBack, normRange=(0,1)).add_plot_args(self.ca.HPLPMajor)
        # self.isMMinorPointTrending = sig.IsPointsTrending(ls=self.ls, hpCol=self.HPLPMinor.name_hp, lpCol=self.HPLPMinor.name_lp, lookBack=self.lookBack, normRange=(0,1)).add_plot_args(self.ca.HPLPMinor)
        # self.l_trends = [self.isMA21Trending, self.isMajorPointTrending, self.isMMinorPointTrending]
        # self.s_trends_passed = sig.Score(name='s_trends', sigs=self.l_trends, scoreType='mean', operator='>=', threshold=50, lookBack=self.lookBack).add_plot_args(deepcopy(self.ca.DefaultPassed))
        # self.s_trends_failed = sig.Score(name='s_trends', sigs=self.l_trends, scoreType='mean', operator='<=', threshold=0, lookBack=self.lookBack).add_plot_args(deepcopy(self.ca.DefaultFailed))
        # self.add_to_ta_list(self.l_trends)
        # self.add_to_ta_list([self.s_trends_passed, self.s_trends_failed])



        # #! PB signals
        # self.llx2 = sig.Lower(col='high', allLower=True, span=2, lookBack=self.lookBack)
        # self.lhx2 = sig.Lower(col='low',  allLower=True, span=2, lookBack=self.lookBack)
        # self.hhx2 = sig.Higher(col='high', allHigher=True, span=2, lookBack=self.lookBack)
        # self.hlx2 = sig.Higher(col='low',  allHigher=True, span=2, lookBack=self.lookBack)

        # #! Bounce Signals 
        # self.rt_day_lo = sig.Retest(ls=self.ls, atrCol=self.ATR.name, direction=direction, valCol='day_low', withinAtrRange=self.retest_atrRange, rollingLen=self.retest_rollingLen, lookBack=self.lookBack, normRange=self.retest_normRange)
        # self.colur_withLSx2 = sig.ColourWithLS(ls=self.ls, lookBack=self.lookBack)
        # self.l_micro = [self.llx2, self.lhx2, self.hhx2, self.hlx2, self.rt_day_lo, self.colur_withLSx2]  #! add barSW, 
        # self.s_micro_passed = sig.Score(sigs=self.l_micro, scoreType='cumsum', operator='>=', threshold=2, lookBack=self.lookBack)
        # self.s_micro_failed = sig.Score(sigs=self.l_micro, scoreType='cumsum', operator='<=', threshold=0, lookBack=self.lookBack)



        # # resets if new HP is formed
        # strat = sig.Strategy('PB', lookBack=self.lookBack, riskAmount=100.00).add_plot_args(self.ca.Stategy1)

        # """
        # -- premarket higher volume than average
        # -- time > 9:35 (wait for first 5 mins to play out)
        # -- breaks premkt high
        # -- price moves above first 5 min bar high
        # -- price pulls back 
        #     a. to max 50% of the first 5 min bar
        #     b. 2 or more lower highs (LH)
        #     c. sequential pullback with less than 50% overlap on any bar
        #     d. touches a support level (prev day high, this day low, daily Res 1 lower )
        # -- bullish bar completes (BSW ..  bot tail or CoC)
        # -- buy signal is confirmed (RTM, RS, break prev bar high)
        # """

        # # validates various metris. each step must be fully validated before moving to the next step
        # #! note that Although an object can score or fail within itself it is not always a simple matter of choosing whether the step scores or fails so therefore we need a second fail object to determine whether the step has failed
        
        # # prerequisites
        # strat.add_step( scoreObj=self.s_past_935, failObj=self.s_past_EOD, ifFailStartFromStep=1)
        # strat.add_step( scoreObj=self.s_past_935, failObj=self.s_past_EOD, ifFailStartFromStep=1)

        # # validate pullback
        # strat.add_step( scoreObj=self.s_pullback_passed, failObj=self.s_pullback_failed, ifFailStartFromStep=1)