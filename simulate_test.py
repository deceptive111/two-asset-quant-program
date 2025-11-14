import numpy as np
from account import Account
from PositionManager import PositionManager


def main():
    np.random.seed(42)

    # 初始化账户与仓位管理器
    initial_capital = 1_000_000
    acct = Account(initial_capital)
    pm = PositionManager(acct, allocation_type="moderate")

    # 设置价格与符号
    symbolB = "BNC"  # 被对冲标的
    symbolA = "RNC"  # 对冲工具
    priceB = 10.0
    priceA = 20.0

    # 1) 在 2025-01-01 进行初始建仓（前一日建仓以避免 T+1 限制）
    pm.set_current_date("2025-01-01")
    medium_acc = pm.sub_accounts["medium"]

    # 初始买入一些头寸（要能被卖出以测试 sell 流程，故买在前一日）
    try:
        medium_acc.buy_stock(symbolB, priceB, 1000, "2025-01-01")
        medium_acc.buy_stock(symbolA, priceA, 1000, "2025-01-01")
    except Exception as e:
        print("初始建仓失败:", e)

    # 更新主账户总资产信息（用于分配计算）
    acct.update_total_asset({symbolB: priceB, symbolA: priceA}, sub_accounts=list(pm.sub_accounts.values()))
    print(f"主账户余额: {acct.balance:,.2f}, 总资产: {acct.total_asset:,.2f}")

    # 2) 次日进行信号计算与对冲
    pm.set_current_date("2025-01-02")

    # 制造相关的价格序列并由此得到收益率序列（包含 A 的 market price）
    n = 60
    mu = np.array([0.0005, 0.0003])
    sigma = np.array([0.01, 0.008])
    rho = 0.3
    cov = np.array([[sigma[0]**2, rho * sigma[0] * sigma[1]],
                    [rho * sigma[0] * sigma[1], sigma[1]**2]])

    # 生成对数收益
    rets = np.random.multivariate_normal(mu, cov, size=n)
    returnsB = rets[:, 0].tolist()  # 这里把 B 的收益率作为被对冲标的的收益序列
    returnsA = rets[:, 1].tolist()  # A 的市场收益率序列（实际为 A 的 market price 的收益）

    # 根据收益率序列生成价格序列（用于当前价格的后验）
    pricesB = [priceB]
    pricesA = [priceA]
    for rb, ra in rets:
        pricesB.append(pricesB[-1] * (1 + rb))
        pricesA.append(pricesA[-1] * (1 + ra))

    # 更新当前价格为序列最后一项，模拟次日开盘价
    priceB = pricesB[-1]
    priceA = pricesA[-1]

    # 计算对冲交易计划
    plan = pm.calc_hedge_quantity(
        symbolB, priceB, symbolA, priceA,
        returnsB, returnsA,
        risk_free_rate=0.0,  # 使用接近0的无风险利率（返回为日收益率），方便发现正Sharpe组合
        signal_dict={"signal_type": "medium", "action": "hedge"},
        target_exposure=1.0,
        min_sharpe_increase=0.0  # 降低阈值以便在示例中更容易执行
    )

    print("calc_hedge_quantity 返回:", plan)

    # 如果计划非空，执行对冲交易
    if plan and isinstance(plan, tuple) and (plan[0] != 0 or plan[1] != 0):
        tradeB, tradeA = plan
        pm.execute_hedge_trade(symbolB, symbolA, priceB, priceA, tradeB, tradeA)
    else:
        print("没有需要执行的对冲交易或计算返回为空")

    # 3) 测试一次强信号买入（single-side）
    strong_signal = {"signal_type": "strong", "action": "buy"}
    buy_qty = pm.execute_single_side_trade(symbolB, priceB, strong_signal)
    print(f"强信号买入返回数量: {buy_qty}")

    # 最后汇总并打印账户状态
    acct.update_total_asset({symbolB: priceB, symbolA: priceA}, sub_accounts=list(pm.sub_accounts.values()))
    acct.record_return()
    print("\n=== 最终账户汇总 ===")
    print(f"主账户余额: {acct.balance:,.2f}")
    print(f"主账户总资产: {acct.total_asset:,.2f}")
    print("各子账户持仓: ")
    for name, sa in pm.sub_accounts.items():
        print(f"  - {name}: {sa.holdings}")

if __name__ == '__main__':
    main()
