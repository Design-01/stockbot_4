from typing import Literal, List, Union
from dataclasses import dataclass

"""
https://plotly.com/python/marker-style/
"""

@dataclass
class PlotArgs:
    plotType: Literal['lines', 'lines+markers', 'lines+markers+text', 'zone', 'points', 'points+text', 'buysell'] = 'lines'
    plotRow: int = 1
    plotCol: int = 1
    name:str = None # gets used to get the name from the dataframe
    dataCols: str | List[str] = None # the columns from the dataframe for the data
    textCols : str | List[str] = None # the columns from the dataframe for the labels
    colContains: str | List[str] = None # fileter for the columns in the dataframe
    colours: str | List[str] = None # the colour of the line/s
    opacities: float | List[float] = None # the opacity of the line/s
    fillColours: str | List[str] = None # the colour of the fill/s
    fillOpacites: float | List[float] = None # the opacity of the fill/s
    dashes: str | List[str] = None # the dash of the line/s
    lineWidths: str | List[int] = None # the width of the line/s
    markerSizes: str | List[int] = None # the size of the marker/s
    markerPosXs: str | List[str] = None # the position of the marker/s
    markerPosYs: str | List[str] = None # the position of the marker/s
    markerSymbols: str | List[str] = None # the symbol of the marker/s
    textPositions: str | List[str] = None # the position of the text/s
    textSizes: str | List[int] = None # the size of the text/s


@dataclass
class ChartArgs:
    MA200 = PlotArgs(plotType='lines', plotRow=1, plotCol=1, colours='darkslateblue', dashes='solid', lineWidths=5)
    MA150 = PlotArgs(plotType='lines', plotRow=1, plotCol=1, colours='cornflowerblue', dashes='solid', lineWidths=4)
    MA50  = PlotArgs(plotType='lines', plotRow=1, plotCol=1, colours='darkmagenta', dashes='solid', lineWidths=3)
    MA21  = PlotArgs(plotType='lines', plotRow=1, plotCol=1, colours='hotpink', dashes='solid', lineWidths=2)
    MA13  = PlotArgs(plotType='lines', plotRow=1, plotCol=1, colours='deepskyblue', dashes='solid', lineWidths=1)
    MA9   = PlotArgs(plotType='lines', plotRow=1, plotCol=1, colours='khaki', dashes='solid', lineWidths=1)

    ATR         = PlotArgs(plotType='', plotRow=1, plotCol=1)
    VWAP        = PlotArgs(plotType='lines', plotRow=1, plotCol=1, colours='yellow', dashes='solid', lineWidths=1)
    volAcum     = PlotArgs(plotType='lines', plotRow=2, plotCol=1, colours='yellow', dashes='solid', lineWidths=1)
    VolChgPct   = PlotArgs(plotType='lines', plotRow=3, plotCol=1, colours='yellow', dashes='solid', lineWidths=1)
    VolumeSpike = PlotArgs(plotType='lines', plotRow=3, plotCol=1, colours='yellow', dashes='solid', lineWidths=1)
    VolumeROC   = PlotArgs(plotType='lines', plotRow=3, plotCol=1, colours='yellow', dashes='solid', lineWidths=1)

    HPLPMajor   = PlotArgs(plotType='points+text', plotRow=1, plotCol=1, colours=['green', 'red'], markerSizes=10, opacities=0.8, textCols=['high', 'low'], textPositions=['top right', 'bottom left'], textSizes=10)
    HPLPMinor   = PlotArgs(plotType='points',      plotRow=1, plotCol=1, colours=['green', 'red'], markerSizes=5,  opacities=0.8, textCols=['high', 'low'], textPositions=['top right', 'bottom left'], textSizes=10)
    SupZone     = PlotArgs(plotType='zone', plotRow=1, plotCol=1, colContains=['Sup_1', 'Sup_2'], colours=['green', 'darkgreen'], dashes='dot', lineWidths=1, opacities=[0.3, 0.2], markerSizes=10, fillColours='green', fillOpacites=[0.2, 0.2])
    ResZone     = PlotArgs(plotType='zone', plotRow=1, plotCol=1, colContains=['Res_1', 'Res_2'], colours=['red', 'darkred'],     dashes='dot', lineWidths=1, opacities=[0.3, 0.2], markerSizes=10, fillColours='red',   fillOpacites=[0.2, 0.2])
    SupLine     = PlotArgs(plotType='lines', plotRow=1, plotCol=1, dataCols=['Sup_1', 'Sup_2'], colours=['green', 'darkgreen'], dashes='solid', lineWidths=1)
    ResLine     = PlotArgs(plotType='lines', plotRow=1, plotCol=1, dataCols=['Res_1', 'Res_2'], colours=['red', 'darkred'],    dashes='solid', lineWidths=1)

    LevPrevDay  = PlotArgs(plotType='lines', plotRow=1, plotCol=1, colours='yellow', dashes='dash', lineWidths=1)
    LevPreMkt   = PlotArgs(plotType='lines', plotRow=1, plotCol=1, colours='yellow', dashes='dash', lineWidths=1)
    LevIntraDay = PlotArgs(plotType='lines', plotRow=1, plotCol=1, colours='yellow', dashes='dash', lineWidths=1)

    GappedWRBs      = PlotArgs(plotType='lines', plotRow=3, plotCol=1, colours='orange', dashes='solid', lineWidths=3)
    GappedPivots    = PlotArgs(plotType='lines', plotRow=3, plotCol=1, colours='orange', dashes='solid', lineWidths=3)
    GappedPastPivot = PlotArgs(plotType='lines', plotRow=3, plotCol=1, colours='orange', dashes='solid', lineWidths=3)

    BarSW      = PlotArgs(plotType='lines', plotRow=3, plotCol=1, colours='magenta', dashes='solid', lineWidths=3)
    RoomToMove = PlotArgs(plotType='lines', plotRow=3, plotCol=1, colours='pink', dashes='solid', lineWidths=3)
    RS         = PlotArgs(plotType='lines', plotRow=3, plotCol=1, colours='purple', dashes='solid', lineWidths=3)
    RSScore    = PlotArgs(plotType='lines', plotRow=4, plotCol=1, colours='cyan', dashes='solid', lineWidths=3, name='debug 1')

    TouchWithBar = PlotArgs(plotType='lines', plotRow=3, plotCol=1, colours='yellow', dashes='solid', lineWidths=3)
    Retest       = PlotArgs(plotType='lines', plotRow=3, plotCol=1, colours='yellow', dashes='solid', lineWidths=3)

    Strategy1 = PlotArgs(plotType='lines', plotRow=4, plotCol=1, colours='yellow', dashes='solid', lineWidths=3, name='debug 2')
    Strategy2 = PlotArgs(plotType='lines', plotRow=4, plotCol=1, colours='yellow', dashes='solid', lineWidths=3, name='debug 3')
    Strategy3 = PlotArgs(plotType='lines', plotRow=4, plotCol=1, colours='yellow', dashes='solid', lineWidths=3, name='debug 4')

    PB_PctHLLH           = PlotArgs(plotType='lines', plotRow=3, plotCol=1, colours='yellow', dashes='solid', lineWidths=3)
    PB_ASC               = PlotArgs(plotType='lines', plotRow=3, plotCol=1, colours='yellow', dashes='solid', lineWidths=3)
    PB_CoC_ByCountOpBars = PlotArgs(plotType='lines', plotRow=3, plotCol=1, colours='yellow', dashes='solid', lineWidths=3)
    PB_Overlap           = PlotArgs(plotType='lines', plotRow=3, plotCol=1, colours='yellow', dashes='solid', lineWidths=3)

    Score1D  = PlotArgs(plotType='lines', plotRow=4, plotCol=1, colours='magenta', dashes='solid', lineWidths=2, name='debug 5')
    ScoreV1D = PlotArgs(plotType='lines', plotRow=4, plotCol=1, colours='cyan',    dashes='solid', lineWidths=2, name='debug 6')
    Score1H  = PlotArgs(plotType='lines', plotRow=4, plotCol=1, colours='magenta', dashes='solid', lineWidths=2, name='debug 7')
    ScoreV1H = PlotArgs(plotType='lines', plotRow=4, plotCol=1, colours='cyan',    dashes='solid', lineWidths=2, name='debug 8')

    Validation1 = PlotArgs(plotType='lines', plotRow=5, plotCol=1, colours='yellow', dashes='solid', lineWidths=3)
    Validation2 = PlotArgs(plotType='lines', plotRow=5, plotCol=1, colours='yellow', dashes='solid', lineWidths=3)
    Validation3 = PlotArgs(plotType='lines', plotRow=5, plotCol=1, colours='yellow', dashes='solid', lineWidths=3)

    ScoreValidation1 = PlotArgs(plotType='lines', plotRow=5, plotCol=1, colours='magenta', dashes='solid', lineWidths=3)
    ScoreValidation2 = PlotArgs(plotType='lines', plotRow=5, plotCol=1, colours='magenta', dashes='solid', lineWidths=3)
    ScoreValidation3 = PlotArgs(plotType='lines', plotRow=5, plotCol=1, colours='magenta', dashes='solid', lineWidths=3)

    scoreTouchWithBar1 = PlotArgs(plotType='lines', plotRow=5, plotCol=1, colours='yellow', dashes='solid', lineWidths=3)
    scoreTouchWithBar2 = PlotArgs(plotType='lines', plotRow=5, plotCol=1, colours='yellow', dashes='solid', lineWidths=3)
    scoreTouchWithBar3 = PlotArgs(plotType='lines', plotRow=5, plotCol=1, colours='yellow', dashes='solid', lineWidths=3)

    ScoreRetest1 = PlotArgs(plotType='lines', plotRow=5, plotCol=1, colours='magenta', dashes='solid', lineWidths=3)
    ScoreRetest2 = PlotArgs(plotType='lines', plotRow=5, plotCol=1, colours='magenta', dashes='solid', lineWidths=3)
    ScoreRetest3 = PlotArgs(plotType='lines', plotRow=5, plotCol=1, colours='magenta', dashes='solid', lineWidths=3)

    StratPctComplete = PlotArgs(plotType='lines', plotRow=5, plotCol=1, colours='magenta', dashes='solid', lineWidths=3)
    StratMeanScore   = PlotArgs(plotType='lines', plotRow=5, plotCol=1, colours='magenta', dashes='solid', lineWidths=3)
    StratScores      = PlotArgs(plotType='lines', plotRow=5, plotCol=1, colours=['magenta', 'green', 'red', 'orange'], dashes='solid', lineWidths=2)
    StratFails       = PlotArgs(plotType='lines', plotRow=5, plotCol=1, colours=['magenta', 'green', 'red', 'orange'], dashes='dash',  lineWidths=1)
    StratSubItems    = PlotArgs(plotType='lines', plotRow=5, plotCol=1, colours=['magenta', 'green', 'red', 'orange'], dashes='solid', lineWidths=1)

    BuySell = PlotArgs(plotType='buysell', plotRow=1, plotCol=1, colours=['cyan', 'yellow'], markerSizes=10, opacities=1, textCols=['open', 'close'], textPositions=['top right', 'bottom left'], textSizes=10, markerSymbols=['arrow-right', 'arrow-left'])