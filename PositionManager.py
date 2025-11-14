import numpy as np
import account as ac

#职责，进行仓位管理

#整体交易逻辑为：在收盘时计算信号，在第二日开盘时以开盘价进行交易
class PositionManager:
    """
    主要任务：
    1、根据信号类型决策交易规模
    2、中等信号需要对冲，故需要计算最优对冲比例
    3、执行风险管理

    todo：资金分层问题后续可以更新一个自动更新机制——例如，根据历史数据动态调整
        另外，做空机制也需要后续完善，比如在有多头的时候即使没有那么多卖出量也可以卖空。
    """

    #预定义资金分配比例
    ALLOCATION_PRESETS = {
        "aggressive": {
            "strong": 0.6,
            "medium": 0.3,
            "weak": 0.1
        },
        "moderate": {
            "strong": 0.4,
            "medium": 0.5,
            "weak": 0.1
        },
        "conservative": {
            "strong": 0.3,
            "medium": 0.7,
            "weak": 0.0
        }
    }

    #预定义在卖出时不同信号的卖出比例——这是针对强信号的
    SELL_SIZE = {
        "strong": 0.5,
        "medium": 0.3,
        "weak": 0.1
    }
    

    def __init__(self, account: ac.Account,
                 allow_short: bool = False, min_trade_unit: int = 100,
                 custom_allocation: dict = None, allocation_type: str = "moderate"):
        """
        考虑到后续交易标的主要为国内市场的指数
        所以他需要满足下面的特点
        1、T+1交易，无法实现当天日内买卖
        2、具有涨跌停限制，一般为10%，ETF标的则为±5%
        3、最小交易单位是100股
        4、融资融券标的可无限制做空

        注意核心参数：
        资金分配类型
        预定义分为三类：
        1,aggressive：强信号60%、中等信号30%、弱信号10%
        2、moderate：强信号40%、中等信号50%、弱信号10%
        3、conservative：强信号30%、中等信号70%、弱信号0%
        调用时可以直接选择预定义的分配类型，传入allocation_type 参数
        可以选择自定义分配方案，传入custom_allocation参数，格式为字典

        其他参数：
        max_position_pct：单个标的最大仓位比例，防止过度集中
        allow_short：是否允许做空

        todo：后续可以增加单标的最大仓位比例控制，仅针对全局，防止过度集中
        """
        self.account = account
        self.allow_short = allow_short
        self.min_trade_unit = min_trade_unit
        self.current_date = None

        #设定信号强度交易比例
        if custom_allocation is not None:
            # 使用自定义分配
            self.position_map = self._validate_allocation(custom_allocation)
            print(f"使用自定义资金分配: {self.position_map}")
        else:
            # 使用预设分配
            if allocation_type not in self.ALLOCATION_PRESETS:
                raise ValueError(f"Invalid allocation_type. Choose from: {list(self.ALLOCATION_PRESETS.keys())}")
            self.position_map = self.ALLOCATION_PRESETS[allocation_type].copy()
            print(f"使用预设分配方案 '{allocation_type}': {self.position_map}")


        #ETF涨跌停限制
        self.price_limit = 0.05

        #根据信号类型设置三个子账户
        # 【修改】传递allow_short参数给子账户
        self.sub_accounts = {
            "strong": ac.SubAccount("strong", self.account, self.position_map["strong"], allow_short=self.allow_short),
            "medium": ac.SubAccount("medium", self.account, self.position_map["medium"], allow_short=self.allow_short),
            "weak": ac.SubAccount("weak", self.account, self.position_map["weak"], allow_short=self.allow_short)
        }
        print("子账户初始化完成，资金分配如下：")
        for key, sub_acc in self.sub_accounts.items():
            print(f"  {key}: {sub_acc.allocated_pct:.2%}")
        if self.allow_short:
            print("  ⚠️ 做空模式已启用，需确保标的可融券且考虑融券成本")

    def set_current_date(self, current_date):
        self.current_date = current_date
        #用于T+1进行判断

    #由于用户可以自定义资金分配方案，这里需要进行验证合理性
    def _validate_allocation(self, allocation):
        total = sum(allocation.values())
        if not np.isclose(total, 1.0):
            raise ValueError("Allocation percentages must sum to 1.0")
        for key in ["strong", "medium", "weak"]:
            if key not in allocation:
                raise ValueError(f"Missing allocation for signal type: {key}")
        return allocation


    #计算可卖出数量
    def calc_sellable_quantity(self, symbol,signal_type=None):
        """
        考虑T+1限制下的可卖出数量，基于子账户架构
        """
        min_trade_unit = self.min_trade_unit
        if signal_type is None:
            raise ValueError("Signal type must be provided to calculate sellable quantity")
        sub_acc = self.sub_accounts[signal_type]
        theoretical_quantity = sub_acc.get_available_to_sell(symbol, self.current_date)
        #对齐最小交易单位
        sellable_quantity = (theoretical_quantity // min_trade_unit) * min_trade_unit
        return sellable_quantity
    
    #计算交易量——这个适用于强信号和弱信号，只做单向交易，且在指定规模内交易
    #single trade pct：单词交易占剩余已分配的可用资金的比例，默认25%,可自定义，避免一次性交易过大
    #默认allow_short为False，禁止卖空，且考虑中国股市T+1限制
    #这里是buy侧和hold侧的计算，sell侧考虑t+1和禁止short sell用另一个函数处理
    #这里使用子账户来进行记账
    def calc_buy_size(self, signal_dict, price, single_trade_pct=0.25, symbol=None):
        min_trade_unit = self.min_trade_unit
        if symbol is None:
            raise ValueError("Symbol must be provided to calculate position size")
        #提取交易信息
        signal_type = signal_dict.get("signal_type","none")
        action = signal_dict.get("action","hold")

        if action == "hold" or signal_type == "none":
            return 0
        
        #如果是中等信号，不在这里处理，交给对冲函数
        if signal_type == "medium":
            return 0
        
        #定义子账户
        sub_acc = self.sub_accounts[signal_type]

        #该层级的可用资金
        allocated_funds = sub_acc.get_allocated_funds()
        #目前该子账户的持仓
        layer_position = sub_acc.holdings.get(symbol, 0)
        
        # 【修改】根据allow_short处理已用资金计算
        if self.allow_short:
            # 做空模式:负持仓(做空)占用资金为负,增加可用资金
            # 正持仓占用资金为正,减少可用资金
            used_funds = layer_position * price
            if layer_position < 0:
                print(f"[做空模式] 当前持有{symbol}空头 {abs(layer_position)}股")
        else:
            # 不允许做空:保持原有逻辑
            used_funds = layer_position * price
        
        remaining_funds = allocated_funds - used_funds

        #建仓
        trade_funds = remaining_funds * single_trade_pct
        trade_funds = min(trade_funds, self.account.balance)
        #计算买入量
        trade_quantity = int(trade_funds // price)
        trade_quantity = (trade_quantity // min_trade_unit) * min_trade_unit
        if trade_quantity <= 0:
            print(f"资金不足，无法买入 {symbol}")
            return 0
        return trade_quantity

    #处理卖出的情况
    def calc_sell_size(self, signal_dict, price, symbol=None):
        """处理卖出的情况(使用子账户)"""
        min_trade_unit = self.min_trade_unit
        
        if symbol is None:
            raise ValueError("Symbol must be provided to calculate position size")
        
        # 提取信号
        signal_type = signal_dict.get("signal_type", "none")
        action = signal_dict.get("action", "hold")

        if action == "hold" or signal_type == "none":
            return 0
        
        # 如果是中等信号，不在这里处理，交给对冲函数
        if signal_type == "medium":
            return 0
        
        #给出子账户
        sub_acc = self.sub_accounts[signal_type]
        #获取该层级的持仓
        layer_position = sub_acc.holdings.get(symbol, 0)

        # 【修改】根据allow_short决定卖出逻辑
        if self.allow_short:
            # 做空模式:允许卖出(包括做空)
            if layer_position <= 0:
                # 无持仓或已是空头,计算做空数量
                # 做空规模 = 分配资金 * 卖出比例 * 单次交易比例(默认25%) / 价格
                sell_pct = self.SELL_SIZE.get(signal_type, 0)
                allocated_funds = sub_acc.get_allocated_funds()
                short_funds = allocated_funds * sell_pct * 0.25  # 默认单次做空25%
                short_quantity = int(short_funds // price)
                short_quantity = (short_quantity // min_trade_unit) * min_trade_unit
                if short_quantity <= 0:
                    print(f"[做空模式] 计算做空数量为0，无法做空 {symbol}")
                    return 0
                print(f"[做空模式] 计划做空 {symbol} {short_quantity}股")
                return short_quantity
            else:
                # 有多头持仓,先卖出持仓。
                sell_pct = self.SELL_SIZE.get(signal_type, 0)
                required_sell_quantity = int(layer_position * sell_pct)
                required_sell_quantity = (required_sell_quantity // min_trade_unit) * min_trade_unit
                if required_sell_quantity <= 0:
                    print(f"卖出数量计算为0，无法卖出 {symbol}")
                    return 0
                # T+1限制
                sellable_quantity = self.calc_sellable_quantity(symbol, signal_type=signal_type)
                if sellable_quantity <= 0:
                    print(f"T+1限制，当前无法卖出 {symbol}")
                    return 0
                trade_quantity = min(required_sell_quantity, sellable_quantity)
                return trade_quantity
        else:
            # 不允许做空:保持原有逻辑
            if layer_position <= 0:
                print(f"子账户 {signal_type} 层无持仓，无法卖出 {symbol}")
                return 0
            
            #根据信号强度计算卖出比例
            sell_pct = self.SELL_SIZE.get(signal_type, 0)
            required_sell_quantity = int(layer_position * sell_pct)
            required_sell_quantity = (required_sell_quantity // min_trade_unit) * min_trade_unit

            if required_sell_quantity <= 0:
                print(f"卖出数量计算为0，无法卖出 {symbol}")
                return 0
            
            #t+1限制和可卖出数量
            sellable_quantity = self.calc_sellable_quantity(symbol, signal_type=signal_type)
            if sellable_quantity <= 0:
                print(f"T+1限制，当前无法卖出 {symbol}")
                return 0
            trade_quantity = min(required_sell_quantity, sellable_quantity)
            return trade_quantity

    #单边交易执行
    def execute_single_side_trade(self, symbol, price, signal_dict):
        """
        执行单边交易（强信号和弱信号）
        
        参数:
        symbol: 交易标的代码
        price: 当前价格
        signal_dict: 信号字典，包含'action'和'signal_type'
        
        返回:
        actual_quantity: 实际成交数量（正=买入，负=卖出，0=未交易）
        """
        if self.current_date is None:
            raise ValueError("Current date not set")
        
        # 提取信号信息
        signal_type = signal_dict.get("signal_type", "none")
        action = signal_dict.get("action", "hold")
        
        # 过滤无效信号
        if signal_type == "none" or action == "hold":
            return 0
        
        # 中等信号由对冲函数处理
        if signal_type == "medium":
            print(f"中等信号应使用对冲函数处理")
            return 0
        
        # 获取对应子账户
        sub_acc = self.sub_accounts[signal_type]
        min_trade_unit = self.min_trade_unit
        
        print(f"\n{'='*60}")
        print(f"执行单边交易 - {signal_type}层")
        print(f"{'='*60}")
        
        # ===== 买入操作 =====
        if action.startswith("buy"):
            # 计算买入数量
            quantity = self.calc_buy_size(signal_dict, price, symbol=symbol)
            
            if quantity <= 0:
                print(f"无法买入 {symbol}：资金不足或计算结果为0")
                return 0
            
            try:
                # 通过子账户执行买入
                sub_acc.buy_stock(symbol, price, quantity, self.current_date, min_trade_unit)
                print(f"[OK] [{signal_type}层] 买入 {symbol} {quantity}股 @{price:.2f}")
                print(f"  耗资: {quantity * price:,.2f}")
                print(f"{'='*60}\n")
                return quantity
            
            except ValueError as e:
                print(f"买入失败: {e}")
                print(f"{'='*60}\n")
                return 0
        
        # ===== 卖出操作 =====
        elif action.startswith("sell"):
            # 计算卖出数量
            quantity = self.calc_sell_size(signal_dict, price, symbol=symbol)
            
            if quantity <= 0:
                print(f"无法卖出 {symbol}：无持仓或T+1限制")
                return 0
            
            try:
                # 通过子账户执行卖出
                sub_acc.sell_stock(symbol, price, quantity, self.current_date, min_trade_unit)
                print(f"[OK] [{signal_type}层] 卖出 {symbol} {quantity}股 @{price:.2f}")
                print(f"  回收: {quantity * price:,.2f}")
                print(f"{'='*60}\n")
                return -quantity  # 返回负数表示卖出
            
            except ValueError as e:
                print(f"卖出失败: {e}")
                print(f"{'='*60}\n")
                return 0
        
        else:
            print(f"未知操作: {action}")
            print(f"{'='*60}\n")
            return 0

    #考虑到有对冲，需要计算对冲比例——这个适用于中等信号，需要双向交易
    
    def calc_optimal_hedge_ratio(self, symbol1_return, symbol2_return, risk_free_rate=0.05):
        """
        计算两个标的的最优对冲比例
        计算方法为：使组合方差最小 or 夏普比率最大化
        注意，需要考虑已有仓位的情况，要进行全局优化
        当出现一次交易信号时就根据全局情况进行优化

        reminder:
        symbol1是B资产收益率序列，是被对冲标的
        symbol2是A资产收益率序列，是对冲工具
        """
        if len(symbol1_return) != len(symbol2_return):
            raise ValueError("Return series must have the same length")
        
        r1 = np.array(symbol1_return)
        r2 = np.array(symbol2_return)

        #计算基本参数
        u1 = np.mean(r1)
        u2 = np.mean(r2)
        sigma1 = np.std(r1)
        sigma2 = np.std(r2)
        rho = np.corrcoef(r1, r2)[0,1]

        #异常检查（0方差情况）
        if sigma1 == 0 and sigma2 == 0:
            print("Warning: Both assets have zero volatility, check the data input")
            return None
        if np.isnan(rho):
            print("Warning: Correlation is NaN, check the data input")
            return None
        
        # 【修改】根据allow_short参数决定网格搜索范围
        if self.allow_short:
            # 允许做空:权重范围扩展到[-1, 2],允许做空和杠杆
            weights = np.linspace(-1, 2, 301)
            print("[做空模式] 网格搜索范围: [-1, 2]")
        else:
            # 不允许做空:保持原有范围[0, 1]
            weights = np.linspace(0, 1, 101)
        
        best_sharpe = -np.inf
        best_weight = 0.5  # initial
        for w2 in weights:
            w1 = 1 - w2
            ret = w1 * u1 + w2 * u2
            vol = np.sqrt((w1**2)*(sigma1**2) + (w2**2)*(sigma2**2) + 2*w1*w2*rho*sigma1*sigma2)
            if vol > 0:
                sharpe = (ret - risk_free_rate) / vol
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_weight = w2
        if best_sharpe < 0:
            print("Warning: No positive Sharpe ratio found, check the data input")
            return None
        return best_weight
    
    #解析模式下的最优权重——优先使用以确保精度。在解析解不成立时使用网格搜索的数值解
    def calc_optimal_hedge_ratio_analytical(self, symbol1_return, symbol2_return, risk_free_rate=0.05):
        """
        注意这里symbol 1 是B 资产收益率序列，是被对冲标的
        symbol 2 是A 资产收益率序列，是对冲工具
        返回为A资产的最优权重
        解析解来源于夏普比率最大化的解析解推导
        """
        #基本参数
        r1 = np.array(symbol1_return)
        r2 = np.array(symbol2_return)
        u1 = np.mean(r1)
        u2 = np.mean(r2)#都是日频， rf也需要是日频
        rf = risk_free_rate/252
        sigma1 = np.std(r1)
        sigma2 = np.std(r2)
        rho = np.corrcoef(r1, r2)[0,1]
        #异常检查（0方差情况）
        if sigma1 == 0 and sigma2 == 0:
            #这里不跳转因为数值解也没有意义
            print("Warning: Both assets have zero volatility, check the data input")
            return None
        if np.isnan(rho):
            print("Warning: Correlation is NaN, check the data input")
            return None
        
        #计算超额
        excess1 = u1 - rf
        excess2 = u2 - rf

        #解析解
        numerator = excess1 * sigma2**2 - excess2 * rho * sigma1 * sigma2
        denominator = (excess1 *sigma2**2 + excess2 * sigma1**2 - 
                       (excess1 + excess2) * rho * sigma1 * sigma2)
        
        #检查分母
        if denominator == 0:
            print("Warning: Denominator in analytical solution is zero, falling back to numerical method")
            return self.calc_optimal_hedge_ratio(symbol1_return, symbol2_return, risk_free_rate)
        optimal_weight_A = numerator / denominator
        
        # 【修改】根据allow_short参数决定权重约束
        if self.allow_short:
            # 允许做空:权重可以为负(做空)或>1(杠杆做多),不做截断
            # 负权重表示做空该资产,>1表示做空另一资产来加杠杆
            print(f"[做空模式] 最优权重A={optimal_weight_A:.4f}, 无截断约束")
            return optimal_weight_A
        else:
            # 不允许做空:保持原有逻辑,权重限制在[0,1]
            optimal_weight_A = np.clip(optimal_weight_A, 0, 1)
            return optimal_weight_A
    
    #对冲执行
    def calc_hedge_quantity(self,symbolB,priceB,symbolA,priceA,
                       returnB,returnA,risk_free_rate=0.05,signal_dict=None,
                       target_exposure = 1.0,min_sharpe_increase=0.05):
        """
        完全基于子账户进行对冲交易的执行——中等信号专用
        核心逻辑：
        1. 计算最优对冲比例
        2. 从中等层子账户获取分配资金
        3. 计算目标持仓（基于最优权重）
        4. 计算交易数量（目标 - 当前）
        5. 检查资金可行性（先卖后买，考虑T+1限制）
        6. 如果T+1限制导致无法完全卖出，此时需要添加检查判断是否调仓
        检查方法为：判断调仓前和调仓后夏普比率是否提升，如不提升则放弃调仓
        注意，由于交易成本的存在，而对冲比例时日频调整的，需要确保夏普比率提升超过某个阈值
        
        参数：
        symbolB: B资产代码（被对冲标的，numeraire）
        priceB: B资产当前价格
        symbolA: A资产代码（对冲工具，denominator）
        priceA: A资产当前价格
        returnB_series: B资产历史收益率序列
        returnA_series: A资产历史收益率序列
        risk_free_rate: 无风险利率（默认5%）
        signal_dict: 信号字典（可选）
        target_exposure: 目标敞口比例，默认1.0表示全额对冲
        这个参数最主要的用途在于，每次中等信号出现触发时，用户可以不必一次性全额对冲
        允许用户分多次逐步建立对冲仓位，从而减少交易冲击
        例如，target_exposure=0.5表示只建立50%的对冲仓
        min_sharpe_increase: 最小夏普比率提升阈值，默认0.05
        只有当调仓后夏普比率提升超过该阈值时，才执行调仓操作，防止过度交易
        """
        if self.current_date is None:
            raise ValueError("Current date is not set. Please set it using set_current_date() method.")
        
        min_trade_unit = self.min_trade_unit

        #calculate optimal hedge ratio
        optimal_weight_A = self.calc_optimal_hedge_ratio_analytical(returnB, returnA, risk_free_rate)
        if optimal_weight_A is None:
            print("无法计算最优对冲比例，跳过对冲交易")
            return
        
        optimal_weight_B = 1 - optimal_weight_A

        #获取中等子账户
        sub_acc = self.sub_accounts["medium"]
        allocated_funds = sub_acc.get_allocated_funds()

        #当前持仓
        current_posB = sub_acc.holdings.get(symbolB, 0)
        current_posA = sub_acc.holdings.get(symbolA, 0)
        current_valueB = current_posB * priceB
        current_valueA = current_posA * priceA
        total_current_value = current_valueB + current_valueA

        #计算当前夏普比率----用于后续进行夏普比率检测和调仓判断
        if total_current_value > 0:
            current_weight_B = current_valueB / total_current_value
            current_weight_A = current_valueA / total_current_value
            current_sharpe = self.calc_portfolio_sharpe(
                returnB,returnA,current_weight_B,current_weight_A,risk_free_rate
            )
            print(f"当前夏普比率: {current_sharpe:.4f}")
        else:
            current_sharpe = 0
            print("当前无持仓，夏普比率设为0")

        #计算目标持仓
        #目标市值
        target_total_value = allocated_funds * target_exposure
        target_valueB = target_total_value * optimal_weight_B
        target_valueA = target_total_value * optimal_weight_A

        # 【修改】根据allow_short参数处理负权重(做空)情况
        if self.allow_short:
            # 允许做空:权重可以为负(表示做空该资产)
            # 负权重表示做空,正权重表示做多
            # 例如:weight_A=-0.2表示做空20%的A资产
            target_posB = int(target_valueB // priceB) if target_valueB >= 0 else -int(abs(target_valueB) // priceB)
            target_posB = (target_posB // min_trade_unit) * min_trade_unit
            target_posA = int(target_valueA // priceA) if target_valueA >= 0 else -int(abs(target_valueA) // priceA)
            target_posA = (target_posA // min_trade_unit) * min_trade_unit
            
            if target_posB < 0 or target_posA < 0:
                print(f"[做空模式] 目标持仓: B={target_posB}, A={target_posA} (负数表示做空)")
        else:
            # 不允许做空:保持原有逻辑,权重限制在[0,1]
            target_posB = int(target_valueB // priceB)
            target_posB = (target_posB // min_trade_unit) * min_trade_unit
            target_posA = int(target_valueA // priceA)
            target_posA = (target_posA // min_trade_unit) * min_trade_unit
        
        #计算交易量
        trade_quantityB = target_posB - current_posB
        trade_quantityA = target_posA - current_posA

        #检查资金可行性
        #先卖后买，确定资金充足
        #计算卖出部分,考虑T+1限制
        actual_tradeB = trade_quantityB
        actual_tradeA = trade_quantityA
        #设立constraint标志，表示是否由于T+1限制导致无法完全卖出or买入
        #如果出现该情况，则需要进行夏普比率提升检查
        constraint_ = False

        released_cash = 0
        if trade_quantityB < 0:
            need_sell_B = abs(trade_quantityB)
            
            # 【修改】根据allow_short参数决定卖出限制
            if self.allow_short:
                # 允许做空:可以卖出超过持仓的数量(真正的做空)
                # 但仍需考虑T+1限制(当日买入的不能卖出)
                sellable_B = self.calc_sellable_quantity(symbolB, signal_type="medium")
                # 做空情况下,可以卖出的数量=可卖出持仓+目标做空数量
                # 这里简化处理:如果目标是做空,允许全部卖出
                if target_posB < 0:
                    # 目标是负持仓(做空),允许全额执行
                    actual_sell_B = need_sell_B
                    print(f"[做空模式] 做空 {symbolB} {actual_sell_B} 股")
                else:
                    # 目标是正持仓,按T+1限制
                    actual_sell_B = min(need_sell_B, sellable_B)
                    if actual_sell_B < need_sell_B:
                        print(f"T+1限制，无法完全卖出 {symbolB}，只能卖出 {actual_sell_B} 股")
                        constraint_ = True
                    else:
                        print(f"卖出 {symbolB} {actual_sell_B} 股")
            else:
                # 不允许做空:保持原有逻辑,只能卖出持仓内的部分
                sellable_B = self.calc_sellable_quantity(symbolB, signal_type="medium")
                actual_sell_B = min(need_sell_B, sellable_B)
                if actual_sell_B < need_sell_B:
                    print(f"T+1限制，无法完全卖出 {symbolB}，只能卖出 {actual_sell_B} 股")
                    constraint_ = True
                else:
                    print(f"卖出 {symbolB} {actual_sell_B} 股")
            
            actual_tradeB = -actual_sell_B
            released_cash += actual_sell_B * priceB
        
        if trade_quantityA < 0:
            need_sell_A = abs(trade_quantityA)
            
            # 【修改】根据allow_short参数决定卖出限制
            if self.allow_short:
                # 允许做空:可以卖出超过持仓的数量(真正的做空)
                sellable_A = self.calc_sellable_quantity(symbolA, signal_type="medium")
                if target_posA < 0:
                    # 目标是负持仓(做空),允许全额执行
                    actual_sell_A = need_sell_A
                    print(f"[做空模式] 做空 {symbolA} {actual_sell_A} 股")
                else:
                    # 目标是正持仓,按T+1限制
                    actual_sell_A = min(need_sell_A, sellable_A)
                    if actual_sell_A < need_sell_A:
                        print(f"T+1限制，无法完全卖出 {symbolA}，只能卖出 {actual_sell_A} 股")
                        constraint_ = True
                    else:
                        print(f"卖出 {symbolA} {actual_sell_A} 股")
            else:
                # 不允许做空:保持原有逻辑
                sellable_A = self.calc_sellable_quantity(symbolA, signal_type="medium")
                actual_sell_A = min(need_sell_A, sellable_A)
                if actual_sell_A < need_sell_A:
                    print(f"T+1限制，无法完全卖出 {symbolA}，只能卖出 {actual_sell_A} 股")
                    constraint_ = True
                else:
                    print(f"卖出 {symbolA} {actual_sell_A} 股")
            
            actual_tradeA = -actual_sell_A
            released_cash += actual_sell_A * priceA
        
        #计算买入部分所需资金
        required_cash = 0
        
        if actual_tradeB > 0:
            required_cash += actual_tradeB * priceB
        if actual_tradeA > 0:
            required_cash += actual_tradeA * priceA
        
        #检查资金是否充足
        available_cash = self.account.balance + released_cash
        if required_cash > available_cash:
            print("资金不足,按比例缩减买入量")
            constraint_ = True
            scale = available_cash / required_cash
            if actual_tradeB > 0:
                actual_tradeB = int(actual_tradeB * scale)
                actual_tradeB = (actual_tradeB // min_trade_unit) * min_trade_unit
                print(f"调整后买入 {symbolB} 数量: {actual_tradeB}")
            if actual_tradeA > 0:
                actual_tradeA = int(actual_tradeA * scale)
                actual_tradeA = (actual_tradeA // min_trade_unit) * min_trade_unit
                print(f"调整后买入 {symbolA} 数量: {actual_tradeA}")
        else:
            print("资金充足，按计划买入")
        
        #如果由于T+1限制或资金不足导致无法完全执行交易，则进行夏普比率提升检查
        if constraint_:
            print("由于T+1限制或资金不足，进行夏普比率提升检查")

            #计算调仓后的持仓
            new_posB = current_posB + actual_tradeB
            new_posA = current_posA + actual_tradeA
            new_valueB = new_posB * priceB
            new_valueA = new_posA * priceA

            #计算调仓后的夏普比率
            total_new_value = new_valueB + new_valueA
            if total_new_value > 0:
                new_weight_B = new_valueB / total_new_value
                new_weight_A = new_valueA / total_new_value
                new_sharpe = self.calc_portfolio_sharpe(
                    returnB, returnA, new_weight_B, new_weight_A, risk_free_rate
                )
                print(f"调仓后夏普比率: {new_sharpe:.4f}")
                #检查夏普比率提升是否超过阈值
                if new_sharpe - current_sharpe < min_sharpe_increase:
                    print(f"夏普比率提升不足 {min_sharpe_increase}, 放弃调仓")
                    return (0,0)
                else:
                    print("夏普比率提升满足要求，执行调仓，B资产交易量:", actual_tradeB, "A资产交易量:", actual_tradeA)
                    return (actual_tradeB, actual_tradeA)
        #无约束，直接执行交易
        else:
            print("无资金或T+1约束，直接执行交易，B资产交易量:", actual_tradeB, "A资产交易量:", actual_tradeA)
            return (actual_tradeB, actual_tradeA)
        
    #计算组合夏普比率的辅助函数
    def calc_portfolio_sharpe(self, returnB, returnA, weight_B,weight_A,risk_free_rate=0.05):
        r1 = np.array(returnB)
        r2 = np.array(returnA)

        u1 = np.mean(r1)
        u2 = np.mean(r2)
        sigma1 = np.std(r1)
        sigma2 = np.std(r2)
        rho = np.corrcoef(r1, r2)[0,1]

        port_return = weight_B * u1 + weight_A * u2
        port_vol = np.sqrt((weight_B**2)*(sigma1**2) + (weight_A**2)*(sigma2**2) + 2*weight_B*weight_A*rho*sigma1*sigma2)
        if port_vol > 0:
            sharpe = (port_return - risk_free_rate) / port_vol
        else:
            sharpe = 0
        return sharpe
    
    #执行对冲交易（指定calc_hedge_quantity的返回值）
    def execute_hedge_trade(self, symbolB, symbolA, priceB, priceA,
                        trade_quantityB, trade_quantityA):
        """
        执行对冲交易，基于calc_hedge_quantity的返回值
        
        修改：使用子账户的 buy_stock 和 sell_stock 方法
        """
        if self.current_date is None:
            raise ValueError("Current date not set")
        
        sub_acc = self.sub_accounts["medium"]
        min_trade_unit = self.min_trade_unit
        
        print(f"\n{'='*60}")
        print(f"执行对冲交易")
        print(f"{'='*60}")
        
        # 执行B资产交易
        if trade_quantityB != 0:
            if trade_quantityB > 0:
                try:
                    sub_acc.buy_stock(symbolB, priceB, trade_quantityB, self.current_date, min_trade_unit)
                    print(f"[OK] 对冲交易：买入 {symbolB} {trade_quantityB}股 @{priceB:.2f}")
                except ValueError as e:
                    print(f"B资产买入失败: {e}")
            else:
                try:
                    sub_acc.sell_stock(symbolB, priceB, abs(trade_quantityB), self.current_date, min_trade_unit)
                    print(f"[OK] 对冲交易：卖出 {symbolB} {abs(trade_quantityB)}股 @{priceB:.2f}")
                except ValueError as e:
                    print(f"B资产卖出失败: {e}")
        
        # 执行A资产交易
        if trade_quantityA != 0:
            if trade_quantityA > 0:
                try:
                    sub_acc.buy_stock(symbolA, priceA, trade_quantityA, self.current_date, min_trade_unit)
                    print(f"[OK] 对冲交易：买入 {symbolA} {trade_quantityA}股 @{priceA:.2f}")
                except ValueError as e:
                    print(f"A资产买入失败: {e}")
            else:
                try:
                    sub_acc.sell_stock(symbolA, priceA, abs(trade_quantityA), self.current_date, min_trade_unit)
                    print(f"[OK] 对冲交易：卖出 {symbolA} {abs(trade_quantityA)}股 @{priceA:.2f}")
                except ValueError as e:
                    print(f" A资产卖出失败: {e}")
        
        print(f"{'='*60}\n")