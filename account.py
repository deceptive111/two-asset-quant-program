from option import Option

class Account:
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.balance = initial_capital  # 主账户只管理现金余额
        self.total_asset = initial_capital

        # ===== 主账户不再直接管理持仓 =====
        # 持仓由各个子账户管理
        # 交易记录（保留用于审计）
        self.trading_history = []

        # 收益率记录
        self.return_history = []
        self.daily_return_history = []
        

    def record_trade(self, trade_type, symbol, price, quantity):
        """记录交易（仅用于审计）"""
        trade = {
            "type": trade_type,
            "symbol": symbol,
            "price": price,
            "quantity": quantity
        }
        self.trading_history.append(trade)
    # ===== 更新总资产（从子账户汇总）=====
    def update_total_asset(self, market_prices: dict, sub_accounts: list = None):
        """
        更新总资产（从子账户汇总持仓）
        
        参数：
        market_prices: {symbol: price} 当前市场价格
        sub_accounts: SubAccount 列表
        """
        # 现金
        total_value = self.balance
        
        # 从各子账户汇总持仓市值
        if sub_accounts:
            for sub_acc in sub_accounts:
                for symbol, quantity in sub_acc.holdings.items():
                    if symbol in market_prices:
                        total_value += quantity * market_prices[symbol]
        
        self.total_asset = total_value
        return self.total_asset
    
    def record_return(self):
        """记录收益率"""
        total_ret = (self.total_asset - self.initial_capital) / self.initial_capital
        self.return_history.append(total_ret)
    
    def get_latest_return(self):
        """获取最新收益率"""
        if not self.return_history:
            return 0
        return self.return_history[-1]

class SubAccount:
    """子账户，独立执行交易和持仓管理"""
    def __init__(self, name, main, allocated_pct, allow_short=False):
        self.name = name
        self.main_account = main
        self.allocated_pct = allocated_pct
        self.allow_short = allow_short  # 【新增】是否允许做空
        
        # 子账户自己的余额（虚拟的，用于跟踪）
        self.balance = 0
        
        # 跟踪该层级的持仓
        self.holdings = {}
        
        # T+1交易限制：记录每天的买入
        self.daily_purchases = {}  # {date: {symbol: quantity}}
        
        # 交易记录
        self.trading_history = []

    def get_allocated_funds(self):
        """计算该层级的分配资金额度"""
        total_asset = self.main_account.total_asset
        return total_asset * self.allocated_pct
    
    def get_used_funds(self, market_prices: dict, option_prices: dict = None):
        """计算已经使用的资金"""
        used_funds = 0
        for symbol, quantity in self.holdings.items():
            if symbol in market_prices:
                used_funds += market_prices[symbol] * quantity
        return used_funds
    
    def get_available_funds(self, market_prices: dict, option_prices: dict = None):
        """计算剩余可用资金"""
        allocated_funds = self.get_allocated_funds()
        used_funds = self.get_used_funds(market_prices, option_prices)
        return allocated_funds - used_funds
    
    def get_available_to_sell(self, symbol, current_date):
        """
        计算在T+1限制下可卖出的数量（子账户独立计算）
        当天买入的不能当天卖出
        """
        total_holdings = self.holdings.get(symbol, 0)
        if total_holdings == 0:
            return 0
        
        # 减去当天买入的数量
        today_purchases = 0
        if current_date in self.daily_purchases and symbol in self.daily_purchases[current_date]:
            today_purchases = self.daily_purchases[current_date][symbol]
        
        return total_holdings - today_purchases
    
    def buy_stock(self, symbol, price, quantity, trade_date, min_trade_unit=100):
        """
        子账户执行买入（独立的交易逻辑）
        
        检查：
        1. 最小交易单位
        2. 资金是否充足（基于主账户余额）
        3. 更新子账户持仓
        4. 记录T+1买入
        5. 扣除主账户余额
        """
        # 检查最小交易单位
        if quantity % min_trade_unit != 0:
            raise ValueError(f"Quantity must be multiple of {min_trade_unit}")
        
        cost = price * quantity
        
        # 检查主账户余额
        if cost > self.main_account.balance:
            raise ValueError(f"SubAccount {self.name}: Insufficient balance in main account")
        
        # 扣除主账户余额
        self.main_account.balance -= cost
        
        # 更新子账户持仓
        if symbol not in self.holdings:
            self.holdings[symbol] = 0
        self.holdings[symbol] += quantity
        
        # 记录T+1买入
        if trade_date not in self.daily_purchases:
            self.daily_purchases[trade_date] = {}
        if symbol not in self.daily_purchases[trade_date]:
            self.daily_purchases[trade_date][symbol] = 0
        self.daily_purchases[trade_date][symbol] += quantity
        
        # 记录交易
        self.trading_history.append({
            "date": trade_date,
            "type": "buy",
            "symbol": symbol,
            "price": price,
            "quantity": quantity
        })
        
        print(f"[{self.name}层] 买入 {quantity} 股 {symbol} @{price:.2f} (成本 {cost:,.2f})")
    
    def sell_stock(self, symbol, price, quantity, trade_date, min_trade_unit=100):
        """
        子账户执行卖出（独立的交易逻辑）
        
        检查：
        1. 最小交易单位
        2. 持仓是否充足(如果allow_short=False)
        3. T+1限制
        4. 更新子账户持仓(可能为负,表示做空)
        5. 增加主账户余额
        """
        # 检查最小交易单位
        if quantity % min_trade_unit != 0:
            raise ValueError(f"Quantity must be multiple of {min_trade_unit}")
        
        # 【修改】根据allow_short决定是否检查持仓
        if self.allow_short:
            # 允许做空:可以卖出超过持仓的数量,持仓可以为负
            current_holdings = self.holdings.get(symbol, 0)
            # 检查T+1限制(只对正持仓部分有效)
            if current_holdings > 0:
                sellable = self.get_available_to_sell(symbol, trade_date)
                # 如果要卖出的数量超过可卖数量,超出部分视为做空
                if quantity > current_holdings:
                    # 部分是卖出持仓,部分是做空
                    if current_holdings > sellable:
                        # T+1限制影响卖出部分
                        raise ValueError(f"SubAccount {self.name}: T+1 restriction - can only sell {sellable} shares (shorting allowed)")
            # 允许全额卖出(包括做空)
            print(f"[{self.name}层][做空模式] 卖出 {quantity} 股 {symbol} @{price:.2f} (当前持仓:{current_holdings})")
        else:
            # 不允许做空:保持原有逻辑
            # 检查持仓
            if symbol not in self.holdings or self.holdings[symbol] < quantity:
                raise ValueError(f"SubAccount {self.name}: Insufficient holdings of {symbol}")
            
            # 检查T+1
            sellable = self.get_available_to_sell(symbol, trade_date)
            if quantity > sellable:
                raise ValueError(f"SubAccount {self.name}: T+1 restriction - can only sell {sellable} shares")
        
        gain = price * quantity
        
        # 增加主账户余额
        self.main_account.balance += gain
        
        # 【修改】更新子账户持仓(允许为负)
        if symbol not in self.holdings:
            self.holdings[symbol] = 0
        self.holdings[symbol] -= quantity
        
        # 如果持仓为0且不在做空状态,删除该symbol
        if not self.allow_short and self.holdings[symbol] == 0:
            del self.holdings[symbol]
        
        # 记录交易
        self.trading_history.append({
            "date": trade_date,
            "type": "sell",
            "symbol": symbol,
            "price": price,
            "quantity": quantity
        })
        
        print(f"[{self.name}层] 卖出 {quantity} 股 {symbol} @{price:.2f} (回款 {gain:,.2f})")
    