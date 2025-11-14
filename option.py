'''
这个文件定义了option和optionchain两个类
他们在策略的作用分别为：
Option类用来储存单个option的基本信息
OptionChain类用来储存同一标的资产下不同option的组合
为什么需要定义OptionChain类?
用于简化策略中对同一标的资产下不同option的管理
方便我们快速找到风险中性密度并对未来payoff进行预测

'''



from dataclasses import dataclass
from datetime import datetime
import yfinance as yf
@dataclass
class Option:
    symbol:str #代码
    underlying_symbol: str #标的资产性质
    strike_price:float#行权价
    spot_price: float #标的资产现价
    expiry_date:str#到期日
    option_type:str  # 'call' or 'put'
    bid_price: float
    ask_price: float
    last_price:float#最新成交价
    volume: int#成交量
    open_interest: int#未平仓合约数
    underlying_quantity:int#标的资产数量   

    #更新最新价格(期权价格)
    def update_market_price(self, price):
        self.last_price = price
    #更新最新价格（标的资产价格)
    def update_spot_price(self, spot_price):
        self.spot_price = spot_price

    #获取价格——若没有买一卖一价，用前日收盘价
    def get_price(self):
        return self.last_price
    #获取价格——若有买一卖一价，用中间价
    def get_price_mid(self):
        price = (self.ask_price+self.bid_price)/2
        return price

    #获取underlying_assets
    def underlying_assets(self):
        return self.underlying_symbol
    
    #剩余期限
    def time_remain(self, CurrentDate):
        ExpireDate = datetime.strptime(self.expiry_date, "%Y-%m-%d")
        DaysToExpire = (ExpireDate - CurrentDate).days
        return DaysToExpire
    
    #实虚值情况
    def moneyness(self, SpotPrice):
        return SpotPrice/self.strike_price
    
    #内在价值情况
    def intrinsic_value(self,SpotPrice):
        if self.option_type == "call":
            return max(SpotPrice - self.strike_price,0)
        else:
            return max(self.strike_price - SpotPrice,0)
        


class OptionChain:
    #这是同一标的资产下不同期限的option的组合
    def __init__(self, underlying, spot_price, risk_free_rate):
        self.underlying_symbol = underlying
        self.spot_price = spot_price
        
        #二维嵌套字典，第一层key是到期日，第二层key是行权价，value是option对象
        #{expiry_date: {strike_price: Option}}
        self.option_chain = {}

        #获取无风险利率

        self.risk_free_rate = risk_free_rate

    def add_option(self, option:Option):
        expiry_date = option.expiry_date
        strike_price = option.strike_price
        
        # 如果该到期日不存在，先创建一个空字典
        if expiry_date not in self.option_chain:
            self.option_chain[expiry_date] = {}
        
        # 在对应的到期日字典中添加期权
        self.option_chain[expiry_date][strike_price] = option

    #获取所有到期日
    def get_expiry_dates(self):
        return self.option_chain.keys()
    
    #获取所有行权价（所有到期日的行权价集合）
    def get_strikes(self):
        all_strikes = set()
        for expiry_options in self.option_chain.values():
            all_strikes.update(expiry_options.keys())
        return sorted(all_strikes)
    
    #获取特定到期日的所有行权价
    def get_strikes_by_expiry(self, expiry_date):
        if expiry_date in self.option_chain:
            return sorted(self.option_chain[expiry_date].keys())
        return []
    
    #获取其中的看涨期权列表
    def get_call_options(self):
        call_options = []
        for expiry_options in self.option_chain.values():
            for op in expiry_options.values():
                if op.option_type == "call":
                    call_options.append(op)
        return call_options
    
    #获取其中的看跌期权列表
    def get_put_options(self):
        put_options = []
        for expiry_options in self.option_chain.values():
            for op in expiry_options.values():
                if op.option_type == "put":
                    put_options.append(op)
        return put_options
    
    #获取其中处于ATM的部分
    def ATM_options(self, sig_value):
        #这里的sigvalue代表了一种tolerance，用来采用类似于假设检验的办法来确定是否处于ATM
        #为什么不用假设检验的原因在于，对每个option而言，假设检验的样本量为1
        #无法计算标准差，所以无法计算检验统计量（进行对数化作t检验是不成立的）
        atm_options = []
        for expiry_options in self.option_chain.values():
            for op in expiry_options.values():
                moneyness = op.moneyness(self.spot_price)
                #只要超出这个tolerance的范围，就认为不在ATM状态
                if abs(moneyness - 1) <= sig_value:
                    atm_options.append(op)
        return atm_options
    
    #更新标的资产价格
    def update_spot_price(self, spot_price):
        self.spot_price = spot_price
        for expiry_options in self.option_chain.values():
            for option in expiry_options.values():
                option.update_spot_price(spot_price)

class OptionChainGenerator:
    def __init__(self, underlying_symbol, risk_free_rate):
        self.underlying_symbol = underlying_symbol
        self.risk_free_rate = risk_free_rate

    def generate_option_chain(self):
        symbol = self.underlying_symbol
        ticker = yf.Ticker(symbol)
        options = ticker.options

        #对单个数据进行option对象的创建并加入optionchain中
        #初始化一个optionchain
        spot_price = ticker.history(period="1d")['Close'][0]
        #注意这个risk_free_rate是一个假设值
        opc = OptionChain(underlying=symbol, spot_price=spot_price, risk_free_rate=self.risk_free_rate)

        #按到期日遍历
        for expiry_data in options:
            opt = ticker.option_chain(expiry_data)
            calls = opt.calls
            puts = opt.puts

            #遍历看涨期权 逐个创建option对象并加入optionchain
            for _, row in calls.iterrows():
                option = Option(
                    symbol=row['contractSymbol'],
                    underlying_symbol=symbol,
                    strike_price=row['strike'],
                    spot_price=spot_price,
                    expiry_date=expiry_data,
                    option_type='call',
                    bid_price=row['bid'],
                    ask_price=row['ask'],
                    last_price=row['lastPrice'],
                    volume=row['volume'],
                    open_interest=row['openInterest'],
                    underlying_quantity=100 #假设每张合约对应100股标的资产
                )
                opc.add_option(option)

            #遍历看跌期权
            for _, row in puts.iterrows():
                option = Option(
                    symbol=row['contractSymbol'],
                    underlying_symbol=symbol,
                    strike_price=row['strike'],
                    spot_price=spot_price,
                    expiry_date=expiry_data,
                    option_type='put',
                    bid_price=row['bid'],
                    ask_price=row['ask'],
                    last_price=row['lastPrice'],
                    volume=row['volume'],
                    open_interest=row['openInterest'],
                    underlying_quantity=100 #假设每张合约对应100股标的资产
                )
                opc.add_option(option)
            
            return opc