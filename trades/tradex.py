from dataclasses import dataclass
import pandas as pd
import copy
from trades.sb_order import OrderX
import trades.price_x as price_x 
from trades.price_x import PriceX, EntryX, StopX, TargetX, RiskX, TrailX, AccelX, TraceX, QtyX
import strategies.ta as ta
import strategies.preset_strats as ps
import my_ib_utils 
from ib_insync import IB

@dataclass
class TradeSummery:
    status:str
    outsideRth:bool
    isOutSideRth:bool
    qtyPct:int
    qty:int
    riskAmount:float
    totalValue:float
    potentialLoss:float
    entryPrice:float
    entryLimitPrice:float
    stopPrice:float
    targetPrice:float
    entryOrderType:str
    actualEntryOrderType:str

@dataclass
class TradeXStatus:
    ENTERING = 'ENTERING'
    IN_TRADE = 'IN_TRADE'
    EXITED   = 'EXITED'
    VARIOUS_STATES = 'VARIOUS_STATES'

def all_elements_in_list(list1, list2):
    return all(elem in list2 for elem in list1)


@dataclass
class TraderX:
    ib:IB = None
    symbol:str =''
    ls:str = ''

    def __post_init__(self):
        self.orderX = OrderX('Strat1', self.ib, self.symbol, self.ls) # order manager
        self.pricexGroups = []  # list of priceX objects to manage multiple priceX objects. this is so that we can have multiple brackets ech forming a % of the total risk.  alowws for complex trades
        self.tradeSummeries = [] # list of trade summeries for each priceX object

    def set_ls(self, ls:str):
        """Set the LONG or SHORT position"""
        if ls not in ['LONG', 'SHORT']:
            raise ValueError("ls must be either 'LONG' or 'SHORT'")
        self.ls = ls

    def validate_frame(self, f):
        """Place holder for validation of the frame. 
        1. if MKT order is the frame set to force download meaning has it got the latest data
        """
        pass

    def add_entry(self, entryx:EntryX=None):
            self.entryx = entryx

    def add_stop_and_target(self, qtyPct=25, targetx:TargetX=None, initStop:StopX=None, trailingStopPrices:list[TrailX]=None):
        name  = f"{self.name}_Strat{len(self.pricexGroups)+1}"
        pricex = price_x.PriceX(name=name, ls=self.ls, includeTarget=True)
        pricex.entry = copy.deepcopy(self.entryx)
        pricex.stop = initStop
        if targetx: 
            pricex.target = targetx

        if trailingStopPrices: 
            pricex.trails = trailingStopPrices

        pricex.set_names()
        self.pricexGroups += [(qtyPct, pricex)]

    def add_stop(self, qtyPct=25, initStop:StopX=None, trailingStopPrices:list[TrailX]=None):
        name  = f"{self.name}_Strat{len(self.pricexGroups)+1}"
        pricex = price_x.PriceX(name=name, ls=self.ls, includeTarget=False)
        pricex.entry = copy.deepcopy(self.entryx)
        pricex.stop = initStop
        if trailingStopPrices: 
            pricex.trails = trailingStopPrices
        
        pricex.set_names()
        self.pricexGroups += [(qtyPct, pricex)]

    def set_columns(self, data):
        for qtyPct, pricex in self.pricexGroups:
            pricex.set_ls()
            pricex.set_columns(data)

    def set_orders(self, data=None, riskAmount=None, outsideRth=False):
        totalQty = 0
        potential_loss = 0
        total_value = 0
        is_rth = my_ib_utils.is_within_trading_hours(self.ib, self.symbol, 'SMART')

        for qtyPct, pricex in self.pricexGroups:
            pricex.run_row(data, is_rth)

            riskAmountPerBracket = riskAmount / 100 * qtyPct
            pricex.compute_qty(riskAmountPerBracket)
            totalQty += pricex.qty.get_qty()
            potential_loss += pricex.qty.potential_loss
            total_value += pricex.qty.total_value

            # helps to collect all the various values for the trade summary
            self.tradeSummeries += [TradeSummery(
                outsideRth=outsideRth,
                isOutSideRth= not is_rth,
                entryOrderType=pricex.entry.orderType,
                actualEntryOrderType='',
                qtyPct=qtyPct,
                qty=pricex.qty.get_qty(),
                entryPrice=pricex.entry.get_price(),
                entryLimitPrice=pricex.entry.get_limit_price(),
                stopPrice=pricex.stop.get_price(),
                targetPrice=pricex.target.get_price() if pricex.includeTarget else None,
                totalValue=pricex.qty.total_value,
                potentialLoss=pricex.qty.potential_loss,
                riskAmount=riskAmountPerBracket,
                status=pricex.status
            )]


        # display(pd.DataFrame(self.tradeSummeries))

        # set entry type to market if entry price is the close price of the current bar
        entryObj = self.pricexGroups[0][1].entry
        entryPrice = entryObj.get_price() if entryObj.orderType in ['STP', 'STP LMT'] else None
        limitEntryPrice = entryObj.get_limit_price() if entryObj.orderType in ['LMT', 'STP LMT'] else None

        self.orderX.set_entry(
            isRth            = is_rth,
            entryOrderType   = self.entryx.orderType,  # gets set by EntryX object
            qty              = totalQty,               # gets set above by computing the total qty for all the brackets
            outsideRth       = outsideRth,             # gets set directly by the args passed in
            limitPrice       = limitEntryPrice,        # get set dierctly by the args passed in OR if outsideRth is True then it gets set by the EntryX object
            entryPrice       = entryPrice              # gets set by the EntryX object only if the entry type is STP otherwise it is None whcih means it is a market order
            )
            
        for qtyPct, pricex in self.pricexGroups:
            if pricex.includeTarget:
                entry_order, stop_order, target_order = self.orderX.add_bracket_order(qtyPct=qtyPct, stop_price=pricex.stop.get_price(), target_price=pricex.target.get_price())
                pricex.entry.ibId = entry_order.orderId
                pricex.stop.ibId = stop_order.orderId
                pricex.target.ibId = target_order.orderId
            else:
                entry_order, stop_order = self.orderX.add_stop_order(qtyPct=qtyPct, stop_price=pricex.stop.get_price())
                pricex.entry.ibId = entry_order.orderId
                pricex.stop.ibId = stop_order.orderId

    def update_filled_status(self):
        def get_remaining_by_orderid(df, order_id):
            s = df.loc[df['orderId'] == order_id, 'remaining']
            if len(s) == 0:
                return None
            return s.iloc[0]

        order_status_df = self.orderX.get_orders_status_as_df()

        for pct, pricex in self.pricexGroups:
            entry_remainng = get_remaining_by_orderid(order_status_df, pricex.entry.ibId)
            stop_remainng = get_remaining_by_orderid(order_status_df, pricex.stop.ibId)
            tget_remainng = get_remaining_by_orderid(order_status_df, pricex.target.ibId)

            #  do it this way so that when running pricex.run_row() it will trigger the has_triggered method
            pricex.entry.has_triggered(forceTrigger=entry_remainng == 0)
            pricex.stop.has_triggered(forceTrigger=stop_remainng == 0)
            pricex.target.has_triggered(forceTrigger=tget_remainng == 0)

    def get_stop_price_by_id(self, stop_id):
        for qtyPct, pricex in self.pricexGroups:
            if pricex.stop.ibId == stop_id:
                return pricex.get_stop_price()
        return None

    def update_stops(self):
        for stop_id in self.orderX.stoploss_ids:
            stop_order = self.orderX.get_order_by_id(stop_id)
            stop_price_new = self.get_stop_price_by_id(stop_id)
            stop_price_old = stop_order.auxPrice
            if stop_price_new != stop_price_old:
                self.orderX.modify_order_price(stop_id, stop_price_new)
            print(f"{stop_id=}, {stop_price_old=}, {stop_price_new=}") 

    def update_price_groups(self, data):
        for qtyPct, pricex in self.pricexGroups:
            pricex.run_row(data)

    def add_ta(self, f):
        pointSpan = 10
        atrSpan = 14
        ps.ma_ta(f, [8, 21, 50, 200])
        f.add_ta(ta.HPLP(hi_col='high', lo_col='low', span=pointSpan), [{'color': 'green', 'size': 10}, {'color': 'red', 'size': 10}], chart_type = 'points')
        f.add_ta(ta.HPLP(hi_col='high', lo_col='low', span=3), [{'color': 'green', 'size': 10}, {'color': 'red', 'size': 10}], chart_type = 'points')
        f.add_ta(ta.ATR(span=atrSpan),{'dash': 'dot', 'color': 'red', 'width': 1}, chart_type = 'ine', row=1)
        f.add_ta(ta.ACC('close', fast_ma=8, slow_ma=21, max_accel=50), {'dash': 'dot', 'color': 'red', 'width': 1}, chart_type = 'line', row=3)
        f.add_ta(ta.Ffill(colToFfill='HP_hi_3'), {'dash': 'dot', 'color': 'green', 'width': 1}, chart_type = 'line', row=1)
        f.add_ta(ta.Ffill(colToFfill='LP_lo_3'), {'dash': 'dot', 'color': 'red', 'width': 1}, chart_type = 'line', row=1)
        f.add_ta(ta.LowestHighest(hi_col='high', lo_col='low', span=3), [{'color': 'green', 'size': 1}, {'color': 'red', 'size': 5}], chart_type = 'points')
        f.add_ta(ta.LowestHighest(hi_col='high', lo_col='low', span=1), [{'color': 'green', 'size': 1}, {'color': 'red', 'size': 5}], chart_type = 'points')
        f.add_ta(ta.SupRes(hi_point_col=f'HP_hi_{pointSpan}', lo_point_col=f'LP_lo_{pointSpan}', atr_col=f'ATR_{atrSpan}', tolerance=1),
            [{'dash': 'solid', 'color': 'green', 'fillcolour': "rgba(0, 255, 0, 0.1)", 'width': 2}, # support # green = rgba(0, 255, 0, 0.1)
            {'dash': 'solid', 'color': 'red', 'fillcolour': "rgba(255, 0, 0, 0.1)", 'width': 2}], # resistance # red = rgba(255, 0, 0, 0.1)
            chart_type = 'support_resistance')

        # f.add_ta(ta.AddColumn(self.entryName), [{'color': 'yellow', 'size': 3}, {'color': 'red', 'size': 3}], chart_type='points', row=1)
        # f.add_ta(ta.AddColumn(self.stopName), [{'color': 'magenta', 'size': 5}], chart_type='points', row=1, nameCol=stopNameCol)
        # f.add_ta(ta.AddColumn(self.targetName), {'dash': 'solid', 'color': 'cyan', 'width': 3}, chart_type='lines+markers', row=1)
        # f.add_ta(ta.AddColumn(self.riskName), {'dash': 'solid', 'color': 'red', 'width': 3}, chart_type='lines+markers', row=3)
        
    def veiw_orders(self):
        pd.set_option('display.float_format', '{:.2f}'.format)
        print("===============================================")
        print (f"    Trader: {self.name} -- {self.symbol} -- {self.ls}")
        print("===============================================\n")
        print (" ---- Trade Summeries ---- ")
        display(pd.DataFrame(self.tradeSummeries))
        print (" ---- Orders to Send to IB ---- ")
        display(self.orderX.get_orders_as_df())
    
    def place_orders(self, delay_between_orders=1):
        self.orderX.place_orders( delay_between_orders)

    def get_status(self):
        statuses = [px[1].status for px in self.pricexGroups]
        print(statuses)
        if all_elements_in_list(statuses, price_x.ENTERING_STATES):
            return TradeXStatus.ENTERING
        elif all_elements_in_list(statuses, price_x.TRADE_STATES):
            return TradeXStatus.IN_TRADE
        elif all_elements_in_list(statuses, price_x.EXIT_STATES):
            return TradeXStatus.EXITED

    def START(self, f, riskAmount, outsideRth=False, delay_between_orders=1):
        self.validate_frame(f.data)
        self.set_columns(f.data)
        self.add_ta(f)
        self.set_orders(f.data, riskAmount, outsideRth, delay_between_orders)
        self.place_orders(delay_between_orders)

    def UPDATE(self, f):
        status = self.get_status()

        if status == TradeXStatus.EXITED:
            return status
        
        if status == TradeXStatus.ENTERING:
            self.update_filled_status()
            self.update_price_groups(f.data)
            return self.get_status()
        
        if status == TradeXStatus.IN_TRADE:
            self.update_stops()
            self.update_filled_status()
            self.update_price_groups(f.data)
            return self.get_status()
        
        if status == TradeXStatus.VARIOUS_STATES:
            return status
        

  
