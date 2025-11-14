"""
这是在B测度下的定价计算器

需要解决的问题：
测度转换需要rn_derivative
计算rn_derivative需要A资产和B资产的风险中性密度以及历史价格数据
在这种情况下我们可以利用copula获取在风险中性世界下的联合分布情况

由于在B测度下，PA/PB是一个鞅，那么我们可以在有联合分布的情况下，计算PA/PB的期望，并进行折现，后再利用PB0的价格来计算PA0的价格

注意在折现时使用的B测度下的新的SDF

"""

from option import Option, OptionChain
from ImpliedV import ImpliedVolatility
import numpy as np
from RNC import RiskNeutralCalculator
from scipy import stats
from scipy.interpolate import RegularGridInterpolator

class BassetPricingCalculator:
    def __init__(self, opcA: OptionChain, opcB: OptionChain, risk_free_rate: float):
        self.opcA = opcA
        self.opcB = opcB
        self.risk_free_rate = risk_free_rate
        self.iv_calculatorA = ImpliedVolatility(opcA)
        self.iv_calculatorB = ImpliedVolatility(opcB)
    
    #计算风险中性下的密度函数字典（用optionchain）
    def risk_neutral_density_dic(self, CurrentDate, opc: OptionChain, normalize=True):
        """
        返回按到期日分组的密度字典
        结构: {expiry_date: {strike: density}}
        :param normalize: 是否归一化密度（使其总和为1）
        """
        density_dic = {}
        option_chain = opc.option_chain
        
        for expiry_date, strikes_dict in option_chain.items():
            density_dic[expiry_date] = {}
            strikes = sorted(strikes_dict.keys())
            
            # 计算原始密度（跳过IV计算失败的期权）
            raw_densities = {}
            for K in strikes:
                option = strikes_dict[K]
                rnc = RiskNeutralCalculator(option, opc)
                try:
                    density = rnc.risk_neutral_density(CurrentDate, self.risk_free_rate)
                    if density is not None and density > 0:  # 确保密度有效
                        raw_densities[K] = density
                except (ValueError, Exception) as e:
                    # 跳过IV计算失败的期权
                    continue
            
            # 如果该到期日没有有效的密度，跳过
            if len(raw_densities) == 0:
                continue
            
            # 重新获取有效的strikes
            strikes = sorted(raw_densities.keys())
            
            # 归一化密度（使用梯形积分近似）
            if normalize and len(strikes) > 1:
                # 计算总概率质量（使用梯形法则）
                total_prob = 0.0
                for i in range(len(strikes) - 1):
                    K1, K2 = strikes[i], strikes[i+1]
                    dK = K2 - K1
                    total_prob += 0.5 * (raw_densities[K1] + raw_densities[K2]) * dK
                
                # 归一化
                if total_prob > 0:
                    for K in strikes:
                        density_dic[expiry_date][K] = raw_densities[K] / total_prob
                else:
                    # 如果总概率为0，使用均匀分布
                    for K in strikes:
                        density_dic[expiry_date][K] = 1.0 / len(strikes)
            else:
                density_dic[expiry_date] = raw_densities
        
        return density_dic
    
    #cdf计算式
    def cdf_calc(self, densities):
        return np.cumsum(densities)/np.sum(densities)
    

    
    #利用copula构建联合分布后计算rn_derivative
    #rho需要通过历史数据计算
    #gaussian copula具备一个严重的问题是：其不具备非负的特征
    #或许我们可以利用lnA和lnB具有正态的特征先进行高斯copula
    #后再利用exp进行变换
    def build_copula(self, CurrentDate, rho):
        """
        构建 Copula 联合分布
        返回: {(expiry_A, expiry_B): {(log_strikeA, log_strikeB): copula_density}}
        """
        # 获取 A 和 B 的风险中性密度字典（已归一化）
        densityA_dic = self.risk_neutral_density_dic(CurrentDate, self.opcA, normalize=True)
        densityB_dic = self.risk_neutral_density_dic(CurrentDate, self.opcB, normalize=True)

        # 为每个到期日组合构建 Copula
        copula_density_all = {}
        
        for expiry_A, densities_A_dict in densityA_dic.items():
            for expiry_B, densities_B_dict in densityB_dic.items():
                # 确保行权价是有序的
                strikesA = sorted(densities_A_dict.keys())
                strikesB = sorted(densities_B_dict.keys())
                
                # 转换到对数空间
                log_strikesA = np.log(strikesA)
                log_strikesB = np.log(strikesB)
                
                densitiesA = np.array([densities_A_dict[k] for k in strikesA])
                densitiesB = np.array([densities_B_dict[k] for k in strikesB])

                # 计算 CDF（这是关键：CDF 是边缘分布的累积分布函数，值在 [0,1] 之间）
                cdf_a = self.cdf_calc(densitiesA)
                cdf_b = self.cdf_calc(densitiesB)

                # 在对数空间构建 Copula 联合密度
                log_copula_density = {}
                for i in range(len(log_strikesA)):
                    for j in range(len(log_strikesB)):
                        # 使用 CDF 值（0-1 之间）而不是对数行权价值
                        u = cdf_a[i]
                        v = cdf_b[j]
                        
                        # 避免边界值导致的 ppf 计算问题
                        u = np.clip(u, 0.0001, 0.9999)
                        v = np.clip(v, 0.0001, 0.9999)
                        
                        # 转换为标准正态分布的分位数
                        z_u = stats.norm.ppf(u)
                        z_v = stats.norm.ppf(v)

                        # 计算联合正态分布密度
                        joint_density = stats.multivariate_normal.pdf([z_u, z_v], mean=[0, 0], cov=[[1, rho], [rho, 1]])

                        # 计算边缘正态分布密度
                        marginal_density_u = stats.norm.pdf(z_u)
                        marginal_density_v = stats.norm.pdf(z_v)

                        # Copula 密度 = 联合密度 / (边缘密度的乘积)
                        copula_density = joint_density / (marginal_density_u * marginal_density_v)
                        
                        # 联合密度 = Copula 密度 * 边缘密度A * 边缘密度B
                        joint_prob_density = copula_density * densitiesA[i] * densitiesB[j]
                        
                        # 存储在对数空间：键是 (ln(strikeA), ln(strikeB))
                        log_copula_density[(log_strikesA[i], log_strikesB[j])] = joint_prob_density
                
                # 存储该到期日组合的 Copula 密度
                copula_density_all[(expiry_A, expiry_B)] = {
                    'copula_density': log_copula_density,
                    'strikesA': strikesA,
                    'strikesB': strikesB
                }
        
        return copula_density_all
    
    #进行插值处理，确保连续
    def interpolate_density(self, log_copula_density, strikesA, strikesB):
        """
        在对数空间对 Copula 密度进行插值
        log_copula_density 的键是对数行权价 (ln(strikeA), ln(strikeB))
        这样插值只在内部进行，自然保证非负性
        """
        log_strikesA = np.log(strikesA)
        log_strikesB = np.log(strikesB)

        # 构建密度矩阵（在对数空间）
        density_matrix = np.zeros((len(log_strikesA), len(log_strikesB)))
        for i, log_strikeA in enumerate(log_strikesA):
            for j, log_strikeB in enumerate(log_strikesB):
                # 使用对数行权价作为键
                density_matrix[i, j] = log_copula_density.get((log_strikeA, log_strikeB), 0)

        # 在对数空间进行插值
        interpolator = RegularGridInterpolator(
            (log_strikesA, log_strikesB), 
            density_matrix, 
            bounds_error=False, 
            fill_value=0
        )
        return interpolator

    #进行网格积分得到A资产在B测度下的期望价格（使用B测度鞅性质）
    def expected_price_A_in_B(self, CurrentDate, rho, use_dynamic_bounds=True):
        """
        计算 A 资产在 B 测度下的期望价格
        
        关键：在 B 测度下，S_A/S_B 是一个鞅，因此：
        E^B[S_A(T)/S_B(T)] = S_A(0)/S_B(0)
        
        所以：S_A(0) = S_B(0) * E^B[S_A(T)/S_B(T)]
        
        注意：不需要贴现因子！因为 numeraire 已经是 S_B 本身。
        
        :param CurrentDate: 当前日期
        :param rho: A 和 B 资产的相关系数
        :param use_dynamic_bounds: 是否使用动态积分上下界
        :return: A 资产的期望价格（不含贴现）
        """
        from datetime import datetime
        
        # 构建所有到期日组合的 Copula
        copula_density_all = self.build_copula(CurrentDate, rho)
        
        # 将 CurrentDate 转换为 datetime 对象（如果需要）
        if isinstance(CurrentDate, str):
            current_dt = datetime.strptime(CurrentDate, "%Y-%m-%d")
        else:
            current_dt = CurrentDate
        
        # 累积所有到期日组合的期望比值
        total_expected_ratio = 0.0
        total_weight = 0.0
        
        for (expiry_A, expiry_B), copula_data in copula_density_all.items():
            log_copula_density = copula_data['copula_density']
            strikesA = copula_data['strikesA']
            strikesB = copula_data['strikesB']
            
            # 计算到期时间 T（以年为单位）
            expiry_dt_A = datetime.strptime(expiry_A, "%Y-%m-%d")
            T_A = (expiry_dt_A - current_dt).days / 365.0
            
            if T_A < 0:
                continue  # 跳过已到期的期权
            
            # 创建插值器
            interpolator = self.interpolate_density(log_copula_density, strikesA, strikesB)

            # 动态计算积分上下界
            if use_dynamic_bounds:
                spot_price_A = self.opcA.spot_price
                spot_price_B = self.opcB.spot_price
                
                # 使用 IV_series 获取隐含波动率序列
                iv_list_A = self.iv_calculatorA.IV_series(CurrentDate, self.risk_free_rate)
                iv_list_B = self.iv_calculatorB.IV_series(CurrentDate, self.risk_free_rate)
                
                # 过滤掉 None 值并计算平均波动率
                valid_iv_A = [iv for iv in iv_list_A if iv is not None]
                valid_iv_B = [iv for iv in iv_list_B if iv is not None]
                
                if not valid_iv_A or not valid_iv_B:
                    raise ValueError("Cannot calculate implied volatility for bounds estimation")
                
                sigma_A = np.mean(valid_iv_A)
                sigma_B = np.mean(valid_iv_B)

                lower_bound_A, upper_bound_A = self.get_dynamic_bounds(spot_price_A, sigma_A, k=3)
                lower_bound_B, upper_bound_B = self.get_dynamic_bounds(spot_price_B, sigma_B, k=3)
            else:
                lower_bound_A, upper_bound_A = self.get_extended_bounds(strikesA)
                lower_bound_B, upper_bound_B = self.get_extended_bounds(strikesB)

            # 网格积分：计算 E^B[S_A(T) / S_B(T)]
            expected_ratio = 0.0
            total_density = 0.0

            for i in range(len(strikesA)-1):
                for j in range(len(strikesB)-1):
                    mid_strikeA = (strikesA[i] + strikesA[i+1]) / 2
                    mid_strikeB = (strikesB[j] + strikesB[j+1]) / 2
                    
                    # 计算网格面积
                    dS_A = strikesA[i+1] - strikesA[i]
                    dS_B = strikesB[j+1] - strikesB[j]
                    grid_area = dS_A * dS_B

                    # 检查是否在积分范围内
                    if mid_strikeA < lower_bound_A or mid_strikeA > upper_bound_A:
                        continue
                    if mid_strikeB < lower_bound_B or mid_strikeB > upper_bound_B:
                        continue
                    
                    # 避免除以零
                    if mid_strikeB <= 0:
                        continue

                    # 计算密度
                    density = interpolator((np.log(mid_strikeA), np.log(mid_strikeB)))
                    if density is None or density <= 0:
                        continue

                    # 关键：计算比值 S_A / S_B 的期望（不含贴现！）
                    ratio = mid_strikeA / mid_strikeB
                    expected_ratio += ratio * density * grid_area
                    total_density += density * grid_area

            if total_density > 0:
                # 归一化：除以总概率质量
                normalized_expected_ratio = expected_ratio / total_density
                total_expected_ratio += normalized_expected_ratio
                total_weight += 1

        # 对所有到期日组合取平均
        if total_weight > 0:
            avg_expected_ratio = total_expected_ratio / total_weight
            # 乘以 S_B(0) 得到 S_A(0) 的估计
            # 这就是鞅性质：S_A(0) = S_B(0) * E^B[S_A(T)/S_B(T)]
            return avg_expected_ratio * self.opcB.spot_price
        else:
            return 0.0
    
    #dynamic函数
    def get_dynamic_bounds(self, spot_price, sigma, k=3):
        """
        动态计算积分上下界（基于对数正态分布假设）
        :param spot_price: 标的资产当前价格
        :param sigma: 隐含波动率
        :param k: 扩展系数（默认为 3）
        :return: (lower_bound, upper_bound)
        """
        lower_bound = spot_price * np.exp(-k * sigma)
        upper_bound = spot_price * np.exp(k * sigma)
        return lower_bound, upper_bound
    
    #extended函数
    def get_extended_bounds(self, strikes, factor=1.5):
        """
        基于行权价范围扩展积分上下界
        :param strikes: 行权价列表
        :param factor: 扩展因子（默认为 1.5）
        :return: (lower_bound, upper_bound)
        """
        lower_bound = min(strikes) * (2 - factor)
        upper_bound = max(strikes) * factor
        return lower_bound, upper_bound
    
    def verify_risk_neutral_pricing(self, CurrentDate):
        """
        验证风险中性定价的基本原理：
        在风险中性世界下，E^Q[exp(-rT) * S_T] 应该等于 S_0
        
        :param CurrentDate: 当前日期
        :return: (理论价格, 现货价格, 相对误差)
        """
        from datetime import datetime
        
        print("\n验证风险中性定价原理...")
        
        for asset_name, opc in [("资产 A", self.opcA), ("资产 B", self.opcB)]:
            print(f"\n{asset_name}:")
            spot_price = opc.spot_price
            
            # 获取归一化的风险中性密度
            density_dic = self.risk_neutral_density_dic(CurrentDate, opc, normalize=True)
            
            for expiry_date, densities in density_dic.items():
                strikes = sorted(densities.keys())
                
                # 计算到期时间
                if isinstance(CurrentDate, str):
                    current_dt = datetime.strptime(CurrentDate, "%Y-%m-%d")
                else:
                    current_dt = CurrentDate
                
                expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
                T = (expiry_dt - current_dt).days / 365.0
                
                if T <= 0:
                    continue
                
                # 计算贴现因子
                discount_factor = np.exp(-self.risk_free_rate * T)
                
                # 计算期望价格：E[S_T]
                expected_price = 0.0
                total_prob = 0.0
                
                for i in range(len(strikes) - 1):
                    K1, K2 = strikes[i], strikes[i+1]
                    dK = K2 - K1
                    mid_K = (K1 + K2) / 2
                    mid_density = (densities[K1] + densities[K2]) / 2
                    
                    expected_price += mid_K * mid_density * dK
                    total_prob += mid_density * dK
                
                # 贴现后的期望价格
                discounted_expected = expected_price * discount_factor
                
                # 计算误差
                error = (discounted_expected - spot_price) / spot_price * 100
                
                print(f"  到期日 {expiry_date}:")
                print(f"    现货价 S_0: {spot_price:.4f}")
                print(f"    期望价格 E[S_T]: {expected_price:.4f}")
                print(f"    贴现后 E[exp(-rT)*S_T]: {discounted_expected:.4f}")
                print(f"    总概率质量: {total_prob:.6f}")
                print(f"    相对误差: {error:+.2f}%")
                
                if abs(error) > 5:
                    print(f"    ⚠️ 警告：误差超过 5%，风险中性密度可能有问题")
                else:
                    print(f"    ✅ 风险中性定价验证通过")
        
