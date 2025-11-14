from option import Option, OptionChain
import numpy as np
from scipy.stats import norm


#todo，计算对象变为option chain类，直接使其返回一个隐含波动率序列
class ImpliedVolatility:
    def __init__(self, opc: OptionChain):
        self.opc = opc

    #这里是对每一个单独的option的计算
    def bs_price(self,sigma,CurrentDate,risk_free_rate, option:Option):
        #计算BS价格
        S = option.spot_price
        K = option.strike_price
        T = option.time_remain(CurrentDate)/365
        rf = risk_free_rate
        if T <= 0:
            return option.intrinsic_value(S)
        d1 = (np.log(S / K)+(rf + 0.5*pow(sigma,2)*T))/(sigma*np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option.option_type == "call":
            price = norm.cdf(d1)*S-norm.cdf(d2)*K*np.exp(-rf*T)
        else:
            price = norm.cdf(-d2)*K*np.exp(-rf*T)-norm.cdf(-d1)*S
        
        return price

    def calculate(self,CurrentDate,risk_free_rate, option:Option):
        #使用二分法计算隐含波动率（布伦特法）
        lower_bound = 0.0001
        upper_bound = 5.0
        #猜值范围

        max_iterations = 1000
        #设置最大迭代次数保证效率
        
        T = option.time_remain(CurrentDate)/365
        if T <= 0:
            return None #期权到期，无法计算IV

        market_price = option.get_price()
        
        # 【关键修复1】检查市场价格有效性
        if market_price is None or market_price <= 0:
            return None
        
        # 【关键修复2】检查内在价值，排除异常定价
        S = option.spot_price
        K = option.strike_price
        intrinsic = option.intrinsic_value(S)
        
        # 市场价格不能低于内在价值（套利机会）
        if market_price < intrinsic * 0.95:  # 允许5%误差
            return None
        
        # 【关键修复3】动态调整容忍度，对低价期权更宽松
        tolerance = max(0.001, market_price * 0.01)  # 1%的相对误差或0.001的绝对误差
        
        # 【关键修复4】检查边界条件
        price_lower = self.bs_price(lower_bound, CurrentDate, risk_free_rate, option)
        price_upper = self.bs_price(upper_bound, CurrentDate, risk_free_rate, option)
        
        # 如果市场价格不在边界范围内，返回None
        if market_price < price_lower or market_price > price_upper:
            return None

        for i in range(max_iterations):
            mid = (lower_bound + upper_bound) / 2
            price = self.bs_price(mid,CurrentDate,risk_free_rate,option)
            
            if abs(price - market_price) <= tolerance:
                return mid
            
            #Vega恒正，所以上下界调整如下
            elif price < market_price:
                lower_bound = mid
            else:
                upper_bound = mid
            
            # 检查收敛性：如果区间太小仍未收敛，返回当前值
            if (upper_bound - lower_bound) < 0.00001:
                return mid
        
        # 【关键修复5】达到最大迭代次数后，检查最终误差
        final_price = self.bs_price(mid, CurrentDate, risk_free_rate, option)
        if abs(final_price - market_price) / market_price < 0.05:  # 允许5%误差
            return mid
        
        return None  # 无法收敛，返回None
        
        #直接计算IV序列
    def IV_series(self, CurrentDate, risk_free_rate):
        """
        计算所有期权的隐含波动率序列
        遍历所有到期日和行权价
        """
        iv_list = []
        for expiry_date, strikes_dict in self.opc.option_chain.items():
            for strike, option in strikes_dict.items():
                iv = self.calculate(CurrentDate, risk_free_rate, option)
                iv_list.append(iv)
        return iv_list
        