#这是一个相关性系数计算器，用于在BNC中计算相关性以生成copula

import numpy as np
class RhoCalculator:
    def __init__(self):
        pass
    
    def calculate_rho(self, seriesA, seriesB):
        if len(seriesA) != len(seriesB):
            raise ValueError("Input series must have the same length")
        return np.corrcoef(seriesA, seriesB)[0, 1]
    
    def compute_rho(self, seriesA, seriesB):
        """别名方法，兼容FactorGenerator调用"""
        return self.calculate_rho(seriesA, seriesB)
