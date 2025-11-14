'''
这是风险中性计算器
用来计算在风险中性下的定价
采用Breeden方法获取风险中性密度

后由于我已经有了隐含波动率计算方法
将使用蒙特卡洛模拟法进行定价（替代原有的特征函数法）

优点：
1. 运行速度更快，避免特征函数的复杂数值积分
2. 数值稳定性更好，避免欧拉方法的累积误差
3. 易于理解和调试
4. 可以处理更复杂的波动率曲面

问题（to be updated）:
1. IV曲线构建的合理性——应该以日期进行分类形成iv surface
2. IVsurface的插值用线性并不合理，需要改用SSVI等方法
3. 期权流动性过滤问题：避免流动性差的期权的不公允价格影响结果
'''
from option import Option, OptionChain
from ImpliedV import ImpliedVolatility
import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d

#利用Breeden法计算风险中性密度

class RiskNeutralCalculator:
    def __init__(self, option: Option, opc: OptionChain = None):
        self.option = option
        self.opc = opc
    
    #计算风险中性密度
    def risk_neutral_density(self, CurrentDate, risk_free_rate):
        #这里的min_volume_threshold是流动性过滤的阈值，防止流动差的定价不够公允的期权影响结果
        #获取隐含波动率
        if self.opc is not None:
            iv_calculator = ImpliedVolatility(self.opc)
            implied_vol = iv_calculator.calculate(CurrentDate, risk_free_rate, self.option)
        else:
            # 创建临时的 OptionChain 用于单个期权
            temp_opc = OptionChain(self.option.underlying_symbol, self.option.spot_price, risk_free_rate)
            temp_opc.add_option(self.option)
            iv_calculator = ImpliedVolatility(temp_opc)
            implied_vol = iv_calculator.calculate(CurrentDate, risk_free_rate, self.option)
        if implied_vol is None:
            raise ValueError("Cannot compute implied volatility for the option")
        #计算d1和d2
        S = self.option.spot_price
        K = self.option.strike_price
        T = self.option.time_remain(CurrentDate)/365
        rf = risk_free_rate
        sigma = implied_vol

        if T <= 0:
            raise ValueError("Option has expired, cannot compute risk-neutral density")
        
        d1 = (np.log(S / K)+(rf + 0.5*pow(sigma,2)*T))/(sigma*np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if self.option.option_type == "call":
            density = ((np.exp(-rf*T))*norm.pdf(d2))/(K * sigma * np.sqrt(T))
        elif self.option.option_type == "put":
            density = ((np.exp(-rf*T))*norm.pdf(-d2))/(K * sigma * np.sqrt(T))
        return density
    
class RnPricing:
    #一个基于特征函数的定价法
    def __init__(self, opc: OptionChain, risk_free_rate, current_date):
        self.option_chain = opc.option_chain
        self.rf = risk_free_rate
        self.current_date = current_date
        self.spot_price = opc.spot_price
        
        # 添加缓存
        self._characteristic_cache = {}
    
    #构建隐含波动率曲线
    def build_iv_curve(self):
        strikes = []
        ivs = []
        
        # 创建临时OptionChain对象用于IV计算（避免修改原对象）
        from option import OptionChain
        temp_opc = OptionChain(
            underlying=list(self.option_chain.values())[0][list(self.option_chain.values())[0].keys().__iter__().__next__()].underlying_symbol,
            spot_price=self.spot_price,
            risk_free_rate=self.rf
        )
        
        # 添加所有期权到临时OptionChain
        for expiry_date, strikes_dict in self.option_chain.items():
            for option in strikes_dict.values():
                temp_opc.add_option(option)
        
        # 计算所有期权的IV
        iv_calculator = ImpliedVolatility(temp_opc)
        iv_list = iv_calculator.IV_series(self.current_date, self.rf)
        
        # 收集所有有效的 strike 和 IV
        iv_index = 0
        valid_count = 0
        for expiry_date, strikes_dict in self.option_chain.items():
            for strike, option in strikes_dict.items():
                if iv_index < len(iv_list):
                    current_iv = iv_list[iv_index]
                    # 检查IV是否有效（不是None，且在合理范围内）
                    if current_iv is not None and 0.01 < current_iv < 2.0:
                        strikes.append(strike)
                        ivs.append(current_iv)
                        valid_count += 1
                iv_index += 1
        
        print(f"  IV曲线构建: 总期权数={iv_index}, 有效IV数={valid_count}")
        
        if len(strikes) < 2:
            raise ValueError(f"Not enough valid options to build IV curve (需要>=2, 实际={len(strikes)})")
            
        #利用线性插值来构建一条平滑的曲线
        sorted_indices = np.argsort(strikes)
        strikes_sorted = np.array(strikes)[sorted_indices]
        ivs_sorted = np.array(ivs)[sorted_indices]

        #插值
        #这里在边界上不用线性外插的原因为防止负波动率的产生
        iv_curve = interp1d(strikes_sorted, ivs_sorted, kind='linear', fill_value=(ivs_sorted[0], ivs_sorted[-1]), bounds_error=False)
        return iv_curve
    
    #接下来是利用期权链对获取全部的风险中性密度
    def risk_neutral_density_dic(self):
        """
        返回按到期日分组的密度字典
        结构: {expiry_date: {strike: density}}
        """
        densities = {}
        for expiry_date, strikes_dict in self.option_chain.items():
            densities[expiry_date] = {}
            for strike, option in strikes_dict.items():
                try:
                    rnc = RiskNeutralCalculator(option)
                    density = rnc.risk_neutral_density(self.current_date, self.rf)
                    densities[expiry_date][strike] = density
                except Exception as e:
                    print(f"Warning: Could not calculate density for option {option.symbol}: {e}")
                    continue      
        return densities
    
    #计算资产理论价格（蒙特卡洛方法）
    def calculate_asset_theoretical_price(self, num_simulations=10000):
        """
        使用蒙特卡洛模拟计算资产理论价格
        
        参数:
        num_simulations: 模拟路径数量，默认10000
        
        返回:
        理论价格（风险中性测度下的期望现值）
        """
        # 获取第一个到期日的第一个期权
        first_expiry = list(self.option_chain.keys())[0]
        sample_option = list(self.option_chain[first_expiry].values())[0]
        T = sample_option.time_remain(self.current_date)/365

        if T <= 0:
            raise ValueError("All options have expired")

        # 构建IV曲线
        iv_curve = self.build_iv_curve()
        
        # 获取所有行权价（跨所有到期日）
        all_strikes = []
        for strikes_dict in self.option_chain.values():
            all_strikes.extend(strikes_dict.keys())
        
        S0 = self.spot_price
        rf = self.rf
        
        # 获取当前价格对应的隐含波动率作为基准
        try:
            base_vol = iv_curve(S0)
        except:
            # 如果插值失败，使用所有IV的平均值
            base_vol = np.mean([iv_curve(k) for k in all_strikes])
        
        # 蒙特卡洛模拟
        dt = T / 252  # 假设一年252个交易日
        num_steps = max(int(T * 252), 1)  # 至少1步
        
        # 初始化价格路径矩阵
        price_paths = np.zeros((num_simulations, num_steps + 1))
        price_paths[:, 0] = S0
        
        # 生成随机数（一次性生成所有，提高效率）
        random_shocks = np.random.standard_normal((num_simulations, num_steps))
        
        # 模拟价格路径
        for i in range(num_steps):
            S_current = price_paths[:, i]
            
            # 对每条路径，使用当前价格对应的局部波动率
            local_vols = np.array([iv_curve(max(min(s, max(all_strikes)), min(all_strikes))) 
                                   for s in S_current])
            
            # 风险中性漂移项：r - 0.5*sigma^2
            drift = (rf - 0.5 * local_vols**2) * dt
            
            # 扩散项：sigma * sqrt(dt) * Z
            diffusion = local_vols * np.sqrt(dt) * random_shocks[:, i]
            
            # 更新价格（对数价格演化）
            price_paths[:, i + 1] = S_current * np.exp(drift + diffusion)
        
        # 获取终端价格
        terminal_prices = price_paths[:, -1]
        
        # 计算期望价格
        expected_final_price = np.mean(terminal_prices)
        
        # 贴现到当前
        theoretical_price = expected_final_price * np.exp(-rf * T)
        
        # 计算标准误差（可选，用于诊断）
        std_error = np.std(terminal_prices) / np.sqrt(num_simulations) * np.exp(-rf * T)
        
        return theoretical_price