import FactorGenerator as fg


class Singnal_Generator:
    """
    根据factor来生成交易信号
    逻辑为：
    分别检验三个因子信号是否满足特定组合的要求：
    1、强信号
    BNC 约等于 RNC，且BNC与RNC给出的价格均不等于市场价格
    若更高则买入资产，更低则卖出

    2、中等信号
    RNC和市价几乎一致，但BNC和市价差距较大
    此时说明B资产定价有一定问题，如果BNC定价高于市价，则买入B资产，反之卖出B资产
    
    理论基础
    BNC定价时，numeraire为B资产，定价式子为：
    S_A(0) = P_B(0) · E^{Q^B}[S_A(T) / P_B(T)]
    RNC定价时，numeraire为无风险资产，这里我们假设为面值为P_B(0)的零息债券，定价式子为：
    S_A(0) = e^{-rT} · E^Q[S_A(T)]
    若RNC定价≈市价 > BNC定价，实则说明B资产收益率大于无风险收益率，B资产被低估，应做多B资产
    如果RNC定价≈市价 < BNC定价，说明B资产收益率小于无风险收益率，B资产被高估，应做空B资产

    但注意，由于测度转换中存在rho的影响，导致差异的可能并非完全来自于B资产的定价问题，
    因此该信号强度为中等。故需要一定的对冲（利用A资产进行对冲）

    对冲比例计算为：使组合总方差最小化。    

    3、弱信号
    BNC和市价几乎一致，但RNC和市价差距较大
    此时问题更可能处在RNC和BNC模型本身
    比如，rho估计错误，RNC模型中IV计算误差，期权流动性问题和breeden风险中性密度的数值误差
    因此该信号强度为弱，建议观望or小规模单向交易，进一步诊断模型问题。
    todo：IV计算误差主要来自于期权流动性问题和隐含波动率曲面质量，尤其是其中采用的线性插值本身是不合适的
    单向交易的方向为相信RNC定价

    4、无信号
    BNC = RNC = 市价
    或者三者互不相等。
    

    反应到因子指标上为：
    强信号:factor1约等于0，且factor2 和 factor3 均不等于0
    中等信号：factor 2约等于0，且factor1和factor3不等于0
    弱信号：factor3约等于0，且factor1和factor2不等于0
    """
    def __init__(self,fgenerator: fg.FactorGenerator,threshold: float=1e-2):
        self.fgenerator = fgenerator
        self.threshold = threshold



    def generate_signal(self, seriesA, seriesB, base_threshold=0.02, lookback=10):
        # 计算因子
        factor1, factor2, factor3 = self.fgenerator.compute_factors(seriesA, seriesB)
        
        # 进行显著性检验（返回 0/1/-1），使用动态阈值
        sig_results = self.fgenerator.test_all(base_threshold, lookback)
        f1_sig = sig_results['factor1']
        f2_sig = sig_results['factor2']
        f3_sig = sig_results['factor3']
        
        # 将带符号的结果转换为布尔值（用于信号分类）
        # 只要不为0就认为显著
        f1_significant = (f1_sig != 0)
        f2_significant = (f2_sig != 0)
        f3_significant = (f3_sig != 0)

        signal = self.classify_signal(
            f1_significant, f2_significant, f3_significant,
            factor1, factor2, factor3
        )
        return signal
    
    def classify_signal(self, f1_sig: bool, f2_sig: bool, f3_sig: bool,f1,f2,f3):
        # 强信号：BNC ≈ RNC (factor1不显著)，但都与市价不同 (factor2和factor3显著)
        if not f1_sig and f2_sig and f3_sig:
            #这是强信号
            action = "buy_A" if f2 < 0 else "sell_A"
            return {
                "signal_type": "strong",
                "action": action,
                "factors": {
                    "factor1": f1,
                    "factor2": f2,
                    "factor3": f3
                },
                "reason": "BNC ≈ RNC, both differ from market price."
            }
        elif f1_sig and not f2_sig and f3_sig:
            #这是中等信号
            action = ["buy_B","sell_A"] if f1 < 0 else ["sell_B","buy_A"]
            return {
                "signal_type": "medium",
                "action": action,
                "factors": {
                    "factor1": f1,
                    "factor2": f2,
                    "factor3": f3
                },
                "reason": "RNC ≈ Market, BNC differs significantly."
            }
        
        elif f1_sig and f2_sig and not f3_sig:
            #弱信号
            #即BNC和Market一致但和RNC不一致
            #可能是RNC模型本身问题，小规模交易
            action = "buy_A" if f2 > 0 else "sell_A"
            return {
                "signal_type": "weak",
                "action": action,
                "factors": {
                    "factor1": f1,
                    "factor2": f2,
                    "factor3": f3
                },
                "reason": "BNC ≈ Market, RNC differs significantly."
            }

        else:
            #无信号，市场杂乱
            return {
                "signal_type": "none",
                "action": "hold",
                "factors": {
                    "factor1": f1,
                    "factor2": f2,
                    "factor3": f3
                },
                "reason": "No clear relationship among BNC, RNC, and Market."
            }