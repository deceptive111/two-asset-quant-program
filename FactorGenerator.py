import BNC as bnc
import RNC as rnc
import option as op
import rho
import pandas as pd
import yfinance as yf
import numpy as np

class FactorGenerator:
    def __init__(self, opca: op.OptionChain, opcb: op.OptionChain, risk_free_rate: float,current_date: pd.Timestamp,rho_calculator: rho.RhoCalculator):
        self.opca = opca
        self.opcb = opcb
        self.bnc_calculator = bnc.BassetPricingCalculator(opca, opcb, risk_free_rate)
        self.current_date = current_date
        self.rnc_calculator = rnc.RnPricing(opca, risk_free_rate, current_date)
        self.risk_free_rate = risk_free_rate
        self.rho_calculator = rho_calculator

        #设定因子队列，便于后续直接信号直接生成后进行回测
        self.factor1_history = []
        self.factor2_history = []
        self.factor3_history = []

    #一次性计算所有因子
    #因子1：BNC与RNC定价差异因子
    #因子2：RNC与市价定价差异因子
    #因子3：BNC与市价定价差异因子
    # 这里的seriesA和seriesB是用来计算相关系数的两个时间序列，其长度相同，且是标的资产价格序列（不是期权序列）
    def compute_factors(self, seriesA: list, seriesB: list):
        relation = self.rho_calculator.compute_rho(seriesA, seriesB)
        print(f"Computed correlation (rho) between seriesA and seriesB: {relation:.4f}")
        bnc_price = self.bnc_calculator.expected_price_A_in_B(self.current_date, relation)
        print(f"Computed BNC expected price of asset A in terms of asset B: {bnc_price:.4f}")
        rnc_price = self.rnc_calculator.calculate_asset_theoretical_price()
        print(f"Computed RNC theoretical price of asset A: {rnc_price:.4f}")
        market_price = self.opca.spot_price
        print(f"Market price of asset A: {market_price:.4f}")

        factor1 = bnc_price - rnc_price
        factor2 = rnc_price - market_price
        factor3 = bnc_price - market_price

        #keep the history
        self.factor1_history.append(factor1)
        self.factor2_history.append(factor2)
        self.factor3_history.append(factor3)
        return factor1, factor2, factor3

    #获取两个series
    def get_series(self, symbolA: str, symbolB: str, start_date: str, end_date: str):
        dataA = yf.download(symbolA, start=start_date, end=end_date)
        dataB = yf.download(symbolB, start=start_date, end=end_date)

        seriesA = dataA['Close'].tolist()
        seriesB = dataB['Close'].tolist()

        return seriesA, seriesB
    
    # 动态阈值计算：基于历史分位数
    def calculate_dynamic_threshold(self, factor_history, base_threshold=0.02, lookback=10):
        """
        基于历史因子分位数动态调整阈值
        
        参数:
        - factor_history: 因子历史序列
        - base_threshold: 基础阈值（相对于现货价格的百分比）
        - lookback: 回溯窗口（周数）
        
        返回:
        - 动态阈值（绝对值）
        """
        if len(factor_history) < 3:
            return self.opca.spot_price * base_threshold
        
        recent = factor_history[-lookback:] if len(factor_history) >= lookback else factor_history
        abs_factors = [abs(f) for f in recent]
        
        # 动态阈值 = max(历史75分位数, 基础阈值)
        # 使用75分位数确保只有真正显著的因子才会触发信号
        percentile_75 = np.percentile(abs_factors, 50)
        dynamic_threshold = max(percentile_75, self.opca.spot_price * base_threshold)
        return dynamic_threshold
    
    # 判断因子显著性（返回带符号的结果）
    def test_significance(self, factor_history, base_threshold=0.02, lookback=10):
        """
        判断因子是否显著，返回带符号的结果
        
        返回:
        - 0: 不显著
        - +1: 显著偏高（理论价 > 现实价）
        - -1: 显著偏低（理论价 < 现实价）
        """
        if len(factor_history) < 1:
            return 0
        
        current_factor = factor_history[-1]
        dynamic_threshold = self.calculate_dynamic_threshold(factor_history, base_threshold, lookback)
        
        # 绝对值比较
        if abs(current_factor) > dynamic_threshold:
            # 返回符号：正数表示理论价高于市价，负数表示理论价低于市价
            return 1 if current_factor > 0 else -1
        else:
            return 0
    
    # 统一检验接口（保持与原代码兼容）
    def test_all(self, base_threshold=0.02, lookback=10):
        """
        检验所有因子的显著性
        
        参数:
        - base_threshold: 基础阈值（相对于现货价格的百分比）
        - lookback: 回溯窗口（周数）
        
        返回字典:
        - 'factor1': 0/1/-1 (不显著/显著偏高/显著偏低)
        - 'factor2': 0/1/-1
        - 'factor3': 0/1/-1
        """
        return {
            'factor1': self.test_significance(self.factor1_history, base_threshold, lookback),
            'factor2': self.test_significance(self.factor2_history, base_threshold, lookback),
            'factor3': self.test_significance(self.factor3_history, base_threshold, lookback)
        }
    
    # 获取因子历史（保持兼容）
    def get_factor_history(self, factor_name: str):
        if factor_name == 'factor1':
            return self.factor1_history
        elif factor_name == 'factor2':
            return self.factor2_history
        elif factor_name == 'factor3':
            return self.factor3_history
        else:
            raise ValueError("Invalid factor name. Choose from 'factor1', 'factor2', 'factor3'.")
    