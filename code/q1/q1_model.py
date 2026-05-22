import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def model_q1_1(population_matrix, transition_matrix, years=5, death_rate=0.05, growth_rate=0.07):
    """
    问题1.1：未来5年老人数量预测
    使用马尔可夫状态转移+死亡率修正+新增人口补充的改进模型
    
    参数：
    - population_matrix: 10×3的当前人口矩阵（第0年）
    - transition_matrix: 10×2的转移概率矩阵
    - years: 预测年数，默认5年
    - death_rate: 自然死亡率，默认5%
    - growth_rate: 新增老人比例，默认7%
    
    返回：
    - prediction_result: (years+1)×10×3的预测结果数组
    """
    print("\n" + "=" * 60)
    print("问题1.1：未来5年老人数量预测")
    print("=" * 60)
    print(f"\n【核心假设】")
    print("  1. 五年内各小区老人健康状态转移概率保持不变")
    print("  2. 老人状态只能由自理向半失能、由半失能向失能转移")
    print("  3. 新增老人默认进入自理老人状态")
    print("  4. 自然死亡率对三类老人统一适用")
    print(f"\n【预测参数】")
    print(f"  死亡率d = {death_rate*100:.1f}%")
    print(f"  新增率g = {growth_rate*100:.1f}%")
    
    # 获取小区数量
    num_communities = population_matrix.shape[0]
    
    # 初始化预测结果数组 (years+1) × 10 × 3
    # 第0年为初始状态
    result = np.zeros((years + 1, num_communities, 3), dtype=int)
    result[0] = population_matrix.copy()
    
    print("\n【递推计算过程】")
    for t in range(years):
        print(f"\n  --- 第{t+1}年预测 ---")
        
        # 获取当前年份数据
        current_N1 = result[t, :, 0]  # 自理老人
        current_N2 = result[t, :, 1]  # 半失能老人
        current_N3 = result[t, :, 2]  # 失能老人
        
        # step1: 计算当前老人总数
        T = current_N1 + current_N2 + current_N3
        
        # step2: 计算新增老人数量
        A = np.round(growth_rate * T).astype(int)
        
        # step3: 计算下一年自理老人数量
        p12 = transition_matrix[:, 0]
        next_N1 = np.round((1 - death_rate) * (1 - p12) * current_N1 + A).astype(int)
        
        # step4: 计算下一年半失能老人数量
        p23 = transition_matrix[:, 1]
        next_N2 = np.round((1 - death_rate) * (p12 * current_N1 + (1 - p23) * current_N2)).astype(int)
        
        # step5: 计算下一年失能老人数量
        next_N3 = np.round((1 - death_rate) * (p23 * current_N2 + current_N3)).astype(int)
        
        # 保存到结果数组
        result[t + 1, :, 0] = next_N1
        result[t + 1, :, 1] = next_N2
        result[t + 1, :, 2] = next_N3
        
        # 打印当年预测摘要
        total_current = T.sum()
        total_next = (next_N1 + next_N2 + next_N3).sum()
        print(f"    第{t}年末老人总数: {total_current} 人")
        print(f"    新增老人数量: {A.sum()} 人")
        print(f"    第{t+1}年末老人总数: {total_next} 人")
    
    print("\n【预测完成】")
    return result

def model_q1_2(prediction_result, demand_matrix):
    """
    问题1.2：第5年末理论服务需求预测
    根据第5年人口预测结果和需求频次矩阵，计算各小区各类服务的理论月需求
    
    模型选择：老人类型—服务类型需求矩阵模型（矩阵乘法模型）
    
    参数：
    - prediction_result: (years+1)×10×3的人口预测结果
    - demand_matrix: 3×6的需求频次矩阵Q
    
    返回：
    - demand_result: 10×6的服务需求矩阵（各小区对六类服务的月需求）
    - detailed_demand: 10×3×6的详细需求矩阵（各小区各类老人对各类服务的月需求）
    """
    print("\n" + "=" * 60)
    print("问题1.2：第5年末理论服务需求预测")
    print("=" * 60)
    
    print("\n【模型选择与解释】")
    print("  采用老人类型—服务类型需求矩阵模型（矩阵乘法模型）")
    print("  核心思想：第5年某小区某类老人人数 × 该类老人对某项服务的月均需求次数")
    print("           = 该类老人对该项服务的月需求总次数")
    
    print("\n【模型推导】")
    print("  step1: 计算分老人类型、分服务类型需求")
    print("    R_i,k,s^theory = N_i,k^5 × q_k,s")
    print("  step2: 计算某小区某项服务总需求")
    print("    R_i,s^theory = Σ_k R_i,k,s^theory = Σ_k (N_i,k^5 × q_k,s)")
    print("  step3: 计算某小区全部服务总需求")
    print("    R_i^theory = Σ_s Σ_k (N_i,k^5 × q_k,s)")
    print("  step4: 计算全区域某项服务总需求")
    print("    R_s^theory = Σ_i R_i,s^theory")
    
    # 获取第5年末的人口数据
    year_5_population = prediction_result[5]  # 10×3矩阵
    
    # step1: 计算分老人类型、分服务类型需求（10×3×6）
    detailed_demand = np.zeros((10, 3, 6))
    for i in range(10):  # 小区
        for k in range(3):  # 老人类型
            detailed_demand[i, k] = year_5_population[i, k] * demand_matrix[k]
    
    # step2: 计算各小区各类服务总需求（10×6）
    demand_result = detailed_demand.sum(axis=1)  # 10×6
    
    # 取整
    demand_result = np.round(demand_result).astype(int)
    detailed_demand = np.round(detailed_demand).astype(int)
    
    print("\n【预测完成】")
    return demand_result, detailed_demand

def model_q1_3(prediction_result, demand_matrix, service_prices, consumption_limits):
    """
    问题1.3：消费约束下服务需求修正模型
    若某类老人的理论服务费用总额超过消费上限，则等比例削减各类服务次数
    
    模型选择：消费约束下的服务需求修正模型
    
    参数：
    - prediction_result: (years+1)×10×3的人口预测结果
    - demand_matrix: 3×6的需求频次矩阵
    - service_prices: 6维服务价格向量（元/次）
    - consumption_limits: 10×3的消费上限矩阵（各小区各类老人的月消费上限，元）
    
    返回：
    - actual_demand: 10×3×6的实际需求矩阵（各小区各类老人对各类服务的实际月需求）
    - theoretical_demand: 10×3×6的理论需求矩阵
    - theta: 10×3的消费修正系数矩阵
    - realization_rate: 10×3的需求实现率矩阵
    """
    print("\n" + "=" * 60)
    print("问题1.3：消费约束下服务需求修正模型")
    print("=" * 60)
    
    print("\n【模型选择与解释】")
    print("  采用消费约束下的服务需求修正模型")
    print("  核心思想：先计算理论需求对应的总消费金额，再与消费上限比较")
    print("           如果超过上限，则等比例压缩服务需求，使总消费不超过消费上限")
    
    print("\n【模型推导】")
    print("  step1: 计算理论消费金额")
    print("    M_i,k^theory = Σ_s (R_i,k,s^theory × a_s)")
    print("  step2: 计算最大可承受消费金额")
    print("    M_i,k^max = N_i,k^5 × u_k")
    print("  step3: 判断理论消费是否超过消费上限")
    print("  step4: 计算消费修正系数")
    print("    θ_i,k = min(1, M_i,k^max / M_i,k^theory)")
    print("  step5: 修正服务需求")
    print("    R_i,k,s^real = θ_i,k × R_i,k,s^theory")
    
    # 获取第5年末的人口数据
    year_5_population = prediction_result[5]  # 10×3矩阵
    
    # step1: 计算分类型分服务的理论需求（10×3×6）
    theoretical_demand = np.zeros((10, 3, 6))
    for i in range(10):  # 小区
        for k in range(3):  # 老人类型
            theoretical_demand[i, k] = year_5_population[i, k] * demand_matrix[k]
    
    # step2: 计算理论消费金额（10×3）
    theoretical_cost = np.zeros((10, 3))
    for i in range(10):
        for k in range(3):
            theoretical_cost[i, k] = np.sum(theoretical_demand[i, k] * service_prices)
    
    # step3: 计算最大可承受消费金额（10×3）
    max_cost = consumption_limits * year_5_population  # 消费上限是人均的，乘以人数得到整体上限
    
    # step4: 计算消费修正系数（10×3）
    theta = np.ones((10, 3))
    for i in range(10):
        for k in range(3):
            if theoretical_cost[i, k] > 0 and theoretical_cost[i, k] > max_cost[i, k]:
                theta[i, k] = max_cost[i, k] / theoretical_cost[i, k]
    
    # step5: 修正服务需求
    actual_demand = np.round(theoretical_demand * theta[:, :, np.newaxis]).astype(int)
    
    # 确保非负
    actual_demand = np.maximum(actual_demand, 0)
    
    # 计算需求实现率
    realization_rate = np.zeros((10, 3))
    for i in range(10):
        for k in range(3):
            total_theory = theoretical_demand[i, k].sum()
            if total_theory > 0:
                realization_rate[i, k] = actual_demand[i, k].sum() / total_theory
    
    print("\n【预测完成】")
    return actual_demand, theoretical_demand, theta, realization_rate

# ========== 保存结果函数 ==========

def save_q1_1_result(prediction_result):
    """保存问题1.1的预测结果到CSV文件"""
    years = prediction_result.shape[0] - 1
    
    with open('q1_1_population_prediction.csv', 'w', encoding='utf-8') as f:
        f.write('小区编号,年份,自理老人,半失能老人,失能老人,老人总数\n')
        for year in range(years + 1):
            for i in range(10):
                n1 = prediction_result[year, i, 0]
                n2 = prediction_result[year, i, 1]
                n3 = prediction_result[year, i, 2]
                total = n1 + n2 + n3
                year_label = f"第{year}年" if year > 0 else "初始" 
                f.write(f'{chr(65+i)},{year_label},{n1},{n2},{n3},{total}\n')
    
    print("\n问题1.1结果已保存到 q1_1_population_prediction.csv")

def save_q1_2_result(demand_result):
    """保存问题1.2的预测结果到CSV文件"""
    services = ['助餐', '日间照料', '上门护理', '康复理疗', '助浴', '紧急救助']
    
    # 表1：各小区六类服务理论月需求
    with open('q1_2_service_demand_by_community.csv', 'w', encoding='utf-8') as f:
        f.write('小区编号,助餐,日间照料,上门护理,康复理疗,助浴,紧急救助,合计\n')
        for i in range(10):
            total = demand_result[i].sum()
            f.write(f'{chr(65+i)},{demand_result[i,0]},{demand_result[i,1]},{demand_result[i,2]},{demand_result[i,3]},{demand_result[i,4]},{demand_result[i,5]},{total}\n')
    
    print("\n问题1.2表1（各小区六类服务理论月需求）已保存到 q1_2_service_demand_by_community.csv")
    
    # 表2：全区域六类服务理论月需求
    total_demand = demand_result.sum(axis=0)
    with open('q1_2_service_demand_total.csv', 'w', encoding='utf-8') as f:
        f.write('服务类型,理论月需求次数\n')
        for i, service in enumerate(services):
            f.write(f'{service},{total_demand[i]}\n')
    
    print("问题1.2表2（全区域六类服务理论月需求）已保存到 q1_2_service_demand_total.csv")
    
    return total_demand

def save_q1_3_result(actual_demand, theoretical_demand):
    """保存问题1.3的预测结果到CSV文件"""
    
    # 计算各小区实际需求汇总（按服务类型）
    actual_demand_total = actual_demand.sum(axis=1)  # 10×6
    
    # 计算各小区理论需求汇总（按服务类型）
    theoretical_demand_total = theoretical_demand.sum(axis=1)  # 10×6
    
    # 表1：各小区消费约束后的实际服务需求
    with open('q1_3_actual_demand.csv', 'w', encoding='utf-8') as f:
        f.write('小区编号,助餐,日间照料,上门护理,康复理疗,助浴,紧急救助,合计\n')
        for i in range(10):
            total = actual_demand_total[i].sum()
            f.write(f'{chr(65+i)},{actual_demand_total[i,0]},{actual_demand_total[i,1]},{actual_demand_total[i,2]},{actual_demand_total[i,3]},{actual_demand_total[i,4]},{actual_demand_total[i,5]},{total}\n')
    
    print("\n问题1.3表1（各小区实际服务需求）已保存到 q1_3_actual_demand.csv")
    
    # 表2：需求对比表
    with open('q1_3_demand_comparison.csv', 'w', encoding='utf-8') as f:
        f.write('小区编号,理论需求总次数,消费约束后需求总次数,需求实现率\n')
        for i in range(10):
            theory_total = int(theoretical_demand_total[i].sum())
            actual_total = int(actual_demand_total[i].sum())
            rate = actual_total / theory_total if theory_total > 0 else 0
            f.write(f'{chr(65+i)},{theory_total},{actual_total},{rate:.4f}\n')
    
    print("问题1.3表2（需求对比表）已保存到 q1_3_demand_comparison.csv")

# ========== 打印结果函数 ==========

def print_q1_1_result(prediction_result):
    """打印问题1.1的预测结果"""
    years = prediction_result.shape[0] - 1
    
    print("\n" + "=" * 70)
    print("问题1.1：未来5年各小区老人数量预测结果")
    print("=" * 70)
    
    for year in range(years + 1):
        year_label = f"第{year}年" if year > 0 else "初始" 
        print(f"\n【{year_label}】")
        print("小区 | 自理老人 | 半失能老人 | 失能老人 | 老人总数")
        print("-----|---------|-----------|---------|---------")
        
        for i in range(10):
            n1 = prediction_result[year, i, 0]
            n2 = prediction_result[year, i, 1]
            n3 = prediction_result[year, i, 2]
            total = n1 + n2 + n3
            print(f"  {chr(65+i)}  |    {n1:4d}   |    {n2:4d}    |   {n3:3d}   |   {total:4d}")

def print_q1_2_result(demand_result, detailed_demand, total_demand):
    """打印问题1.2的预测结果"""
    services = ['助餐', '日间照料', '上门护理', '康复理疗', '助浴', '紧急救助']
    elderly_types = ['自理老人', '半失能老人', '失能老人']
    
    print("\n" + "=" * 80)
    print("问题1.2：第5年末各小区六类服务理论月需求")
    print("=" * 80)
    
    # 表1：各小区六类服务理论月需求（汇总）
    print("\n表1：各小区六类服务理论月需求")
    print("小区 | 助餐 | 日间照料 | 上门护理 | 康复理疗 | 助浴 | 紧急救助 | 合计")
    print("-----|------|----------|----------|----------|------|----------|------")
    
    for i in range(10):
        total = demand_result[i].sum()
        print(f"  {chr(65+i)}  | {demand_result[i,0]:5d} |  {demand_result[i,1]:6d}   |  {demand_result[i,2]:6d}   |  {demand_result[i,3]:6d}   | {demand_result[i,4]:4d} |  {demand_result[i,5]:6d}   | {total:5d}")
    
    # 表2：各小区分类型需求（简表，按服务类型汇总）
    print("\n" + "=" * 80)
    print("表2：各小区分类型服务需求汇总")
    print("=" * 80)
    
    for k, elderly in enumerate(elderly_types):
        print(f"\n【{elderly}】")
        print("小区 | 助餐 | 日间照料 | 上门护理 | 康复理疗 | 助浴 | 紧急救助 | 合计")
        print("-----|------|----------|----------|----------|------|----------|------")
        for i in range(10):
            total = detailed_demand[i, k].sum()
            print(f"  {chr(65+i)}  | {detailed_demand[i,k,0]:5d} |  {detailed_demand[i,k,1]:6d}   |  {detailed_demand[i,k,2]:6d}   |  {detailed_demand[i,k,3]:6d}   | {detailed_demand[i,k,4]:4d} |  {detailed_demand[i,k,5]:6d}   | {total:5d}")
    
    # 表3：全区域六类服务理论月需求
    print("\n" + "=" * 80)
    print("表3：全区域六类服务理论月需求")
    print("=" * 80)
    print("\n服务类型 | 理论月需求次数")
    print("----------|----------------")
    
    for i, service in enumerate(services):
        print(f"{service:8s} | {total_demand[i]:14d}")
    
    print(f"\n{'合计':8s} | {total_demand.sum():14d}")

def print_q1_3_summary(actual_demand, theoretical_demand, theta, realization_rate):
    """打印问题1.3的预测结果摘要"""
    services = ['助餐', '日间照料', '上门护理', '康复理疗', '助浴', '紧急救助']
    elderly_types = ['自理老人', '半失能老人', '失能老人']
    
    print("\n" + "=" * 80)
    print("问题1.3：消费约束下服务需求修正结果")
    print("=" * 80)
    
    # 计算总理论需求和总实际需求
    total_theoretical = theoretical_demand.sum()
    total_actual = actual_demand.sum()
    
    print(f"\n【总体统计】")
    print(f"  总理论需求次数: {int(total_theoretical):,} 次/月")
    print(f"  总实际需求次数: {int(total_actual):,} 次/月")
    print(f"  需求实现率: {total_actual/total_theoretical*100:.2f}%")
    
    # 表1：各小区消费修正系数
    print("\n" + "=" * 80)
    print("表1：各小区消费修正系数")
    print("=" * 80)
    print("\n小区 | 自理老人 | 半失能老人 | 失能老人")
    print("-----|----------|------------|--------")
    for i in range(10):
        print(f"  {chr(65+i)}  |   {theta[i,0]:.2f}    |    {theta[i,1]:.2f}      |   {theta[i,2]:.2f}")
    
    # 表2：各小区需求实现率
    print("\n" + "=" * 80)
    print("表2：各小区需求实现率")
    print("=" * 80)
    print("\n小区 | 自理老人 | 半失能老人 | 失能老人")
    print("-----|----------|------------|--------")
    for i in range(10):
        print(f"  {chr(65+i)}  |  {realization_rate[i,0]:.1%}  |   {realization_rate[i,1]:.1%}    |  {realization_rate[i,2]:.1%}")
    
    # 表3：各小区实际服务需求汇总
    print("\n" + "=" * 80)
    print("表3：各小区实际服务需求汇总")
    print("=" * 80)
    
    actual_demand_total = actual_demand.sum(axis=1)  # 10×6
    
    print("\n小区 | 助餐 | 日间照料 | 上门护理 | 康复理疗 | 助浴 | 紧急救助 | 合计")
    print("-----|------|----------|----------|----------|------|----------|------")
    for i in range(10):
        total = actual_demand_total[i].sum()
        print(f"  {chr(65+i)}  | {actual_demand_total[i,0]:5d} |  {actual_demand_total[i,1]:6d}   |  {actual_demand_total[i,2]:6d}   |  {actual_demand_total[i,3]:6d}   | {actual_demand_total[i,4]:4d} |  {actual_demand_total[i,5]:6d}   | {total:5d}")

# ========== 可视化模块 ==========
# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 常量定义
COMMUNITIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
ELDERLY_TYPES = ['自理', '半失能', '失能']
SERVICES = ['助餐', '日间照料', '上门护理', '康复理疗', '助浴', '紧急救助']
COLORS_3 = ['#4CAF50', '#FF9800', '#F44336']
COLORS_10 = plt.cm.Set3(np.linspace(0, 1, 10))

def create_visualizations(prediction_result, theoretical_demand, actual_demand, theta, realization_rate, monthly_income):
    """
    创建问题1的可视化图表（13张图）
    
    参数:
        prediction_result: (6, 10, 3) 预测结果
        theoretical_demand: (10, 3, 6) 理论需求
        actual_demand: (10, 3, 6) 实际需求
        theta: (10, 3) 消费修正系数
        realization_rate: (10, 3) 需求实现率
        monthly_income: (10,) 各小区人均月收入
    """
    print("\n" + "=" * 60)
    print("开始生成可视化图表...")
    print("=" * 60)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ===== 问题1.1 人口预测可视化 =====
    print("\n【1.1 人口预测可视化】")
    
    # 图1：全区域老人数量及结构变化趋势图
    plot_population_trend(prediction_result, output_dir)
    
    # 图2：第5年末各小区老人结构对比图
    plot_community_structure_year5(prediction_result, output_dir)
    
    # 图3：各小区老人数量增长热力图
    plot_population_heatmap(prediction_result, output_dir)
    
    # 图4：三老人占比变化趋势图
    plot_elderly_proportion_trend(prediction_result, output_dir)
    
    # 图5：各小区年均人口增长率对比图
    plot_community_growth_rate(prediction_result, output_dir)
    
    # ===== 问题1.2 理论服务需求可视化 =====
    print("\n【1.2 理论服务需求可视化】")
    
    # 图6：全区域各类服务需求占比图
    plot_service_demand_pie(theoretical_demand, output_dir)
    
    # 图7：各小区服务需求总量对比图
    plot_community_demand_comparison(theoretical_demand, output_dir)
    
    # 图8：分老人类型服务需求对比图
    plot_demand_by_elderly_type(theoretical_demand, output_dir)
    
    # 图9：小区-服务类型需求热力图
    plot_demand_heatmap(theoretical_demand, output_dir)
    
    # ===== 问题1.3 消费约束需求修正可视化 =====
    print("\n【1.3 消费约束需求修正可视化】")
    
    # 图10：理论需求与实际需求对比图
    plot_theoretical_vs_actual(theoretical_demand, actual_demand, output_dir)
    
    # 图11：消费修正系数热力图
    plot_theta_heatmap(theta, output_dir)
    
    # 图12：各类服务需求损失率对比图
    plot_service_loss_rate(theoretical_demand, actual_demand, output_dir)
    
    # 图13：小区消费能力与需求实现率散点图（加分项）
    plot_income_realization_scatter(monthly_income, realization_rate, output_dir)
    
    print("\n" + "=" * 60)
    print(f"所有图表已保存到: {output_dir}")
    print("=" * 60)

def plot_population_trend(prediction_result, output_dir):
    """图1：全区域老人数量及结构变化趋势图"""
    years = range(6)
    totals = {
        '自理': [prediction_result[t, :, 0].sum() for t in years],
        '半失能': [prediction_result[t, :, 1].sum() for t in years],
        '失能': [prediction_result[t, :, 2].sum() for t in years]
    }
    total_all = [sum(totals[k][t] for k in totals) for t in years]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for idx, (label, color) in enumerate(zip(['自理', '半失能', '失能'], COLORS_3)):
        ax.plot(years, totals[label], 'o-', color=color, linewidth=2, markersize=8, label=label)
    
    ax.plot(years, total_all, 's-', color='navy', linewidth=3, markersize=10, label='总人数')
    
    for t in years:
        ax.annotate(f'{total_all[t]}', (t, total_all[t]), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9)
    
    ax.set_xlabel('年份', fontsize=12)
    ax.set_ylabel('老人数量（人）', fontsize=12)
    ax.set_title('全区域老人数量及结构变化趋势图', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(years)
    ax.set_xticklabels([f'第{y}年' for y in years])
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'q1_fig1_population_trend.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: q1_fig1_population_trend.png")

def plot_community_structure_year5(prediction_result, output_dir):
    """图2：第5年末各小区老人结构对比图"""
    year5_data = prediction_result[5]  # (10, 3)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(10)
    width = 0.6
    
    bottom = np.zeros(10)
    for idx, (label, color) in enumerate(zip(ELDERLY_TYPES, COLORS_3)):
        values = year5_data[:, idx]
        bars = ax.bar(x, values, width, label=label, bottom=bottom, color=color)
        for i, v in enumerate(values):
            ax.text(i, bottom[i] + v/2, f'{int(v)}', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        bottom += values
    
    for i, total in enumerate(bottom):
        ax.text(i, total + 20, f'{int(total)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('小区', fontsize=12)
    ax.set_ylabel('老人数量（人）', fontsize=12)
    ax.set_title('第5年末各小区老人结构对比图', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(COMMUNITIES)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'q1_fig2_community_structure.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: q1_fig2_community_structure.png")

def plot_population_heatmap(prediction_result, output_dir):
    """图3：各小区老人数量增长热力图"""
    total_pop = np.zeros((10, 6))
    for t in range(6):
        total_pop[:, t] = prediction_result[t, :, :].sum(axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(total_pop, cmap='YlOrRd', aspect='auto')
    
    for i in range(10):
        for j in range(6):
            text = ax.text(j, i, f'{int(total_pop[i, j])}', ha='center', va='center', color='black', fontsize=9)
    
    ax.set_xticks(range(6))
    ax.set_xticklabels([f'第{t}年' for t in range(6)])
    ax.set_yticks(range(10))
    ax.set_yticklabels(COMMUNITIES)
    ax.set_xlabel('年份', fontsize=12)
    ax.set_ylabel('小区', fontsize=12)
    ax.set_title('各小区老人数量增长热力图', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('老人总数', fontsize=10)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'q1_fig3_population_heatmap.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: q1_fig3_population_heatmap.png")

def plot_elderly_proportion_trend(prediction_result, output_dir):
    """图4：三老人占比变化趋势图"""
    years = range(6)
    proportions = {k: [] for k in ELDERLY_TYPES}
    
    for t in years:
        total = sum(prediction_result[t, :, idx].sum() for idx in range(3))
        for idx, label in enumerate(ELDERLY_TYPES):
            proportions[label].append(prediction_result[t, :, idx].sum() / total)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for label, color in zip(ELDERLY_TYPES, COLORS_3):
        ax.fill_between(years, 0, proportions[label], alpha=0.4, color=color, label=label)
        ax.plot(years, proportions[label], 'o-', color=color, linewidth=2)
    
    ax.set_xlabel('年份', fontsize=12)
    ax.set_ylabel('占比', fontsize=12)
    ax.set_title('三类老人占比变化趋势图', fontsize=14, fontweight='bold')
    ax.legend(loc='center right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(years)
    ax.set_xticklabels([f'第{y}年' for y in years])
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'q1_fig4_proportion_trend.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: q1_fig4_proportion_trend.png")

def plot_community_growth_rate(prediction_result, output_dir):
    """图5：各小区年均人口增长率对比图（加分项）"""
    growth_rates = []
    for i in range(10):
        pop0 = prediction_result[0, i, :].sum()
        pop5 = prediction_result[5, i, :].sum()
        annual_rate = (pop5 / pop0) ** (1/5) - 1
        growth_rates.append(annual_rate * 100)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(COMMUNITIES, growth_rates, color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, 10)))
    
    for bar, rate in zip(bars, growth_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
               f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.axhline(y=np.mean(growth_rates), color='red', linestyle='--', linewidth=1.5, label=f'平均值{np.mean(growth_rates):.1f}%')
    
    ax.set_xlabel('小区', fontsize=12)
    ax.set_ylabel('年均人口增长率（%）', fontsize=12)
    ax.set_title('各小区年均人口增长率对比图', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'q1_fig5_growth_rate.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: q1_fig5_growth_rate.png")

def plot_service_demand_pie(theoretical_demand, output_dir):
    """图6：全区域各类服务需求占比图（必放）"""
    service_totals = theoretical_demand.sum(axis=(0, 1))  # (6,)
    total = service_totals.sum()
    percentages = service_totals / total * 100
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set2(np.linspace(0, 1, 6))
    wedges, texts, autotexts = ax.pie(service_totals, labels=SERVICES, autopct='%1.1f%%',
                                       colors=colors, startangle=90, pctdistance=0.75)
    
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    for text in texts:
        text.set_fontsize(11)
    
    ax.set_title('全区域各类服务需求占比图', fontsize=14, fontweight='bold')
    
    legend_labels = [f'{s}: {int(v):,}次/月' for s, v in zip(SERVICES, service_totals)]
    ax.legend(wedges, legend_labels, title="服务类型", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'q1_fig6_service_demand_pie.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: q1_fig6_service_demand_pie.png")

def plot_community_demand_comparison(theoretical_demand, output_dir):
    """图7：各小区服务需求总量对比图（必放）"""
    community_totals = theoretical_demand.sum(axis=(1, 2))  # (10,)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(COMMUNITIES, community_totals, color=plt.cm.Blues(np.linspace(0.4, 0.9, 10)))
    
    for bar, total in zip(bars, community_totals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
               f'{int(total):,}', ha='center', va='bottom', fontsize=9)
    
    sorted_idx = np.argsort(community_totals)
    ax.axhline(y=community_totals.mean(), color='red', linestyle='--', linewidth=1.5, label=f'平均值: {community_totals.mean():,.0f}')
    
    ax.set_xlabel('小区', fontsize=12)
    ax.set_ylabel('月总需求次数', fontsize=12)
    ax.set_title('各小区服务需求总量对比图', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'q1_fig7_community_demand.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: q1_fig7_community_demand.png")

def plot_demand_by_elderly_type(theoretical_demand, output_dir):
    """图8：分老人类型服务需求对比图（必放）"""
    type_totals = theoretical_demand.sum(axis=0)  # (3, 6)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(6)
    width = 0.25
    
    for idx, (label, color) in enumerate(zip(ELDERLY_TYPES, COLORS_3)):
        values = type_totals[idx, :]
        bars = ax.bar(x + idx * width, values, width, label=label, color=color)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30, 
                   f'{int(v)}', ha='center', va='bottom', fontsize=7)
    
    ax.set_xlabel('服务类型', fontsize=12)
    ax.set_ylabel('月需求次数', fontsize=12)
    ax.set_title('分老人类型服务需求对比图', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(SERVICES, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'q1_fig8_demand_by_elderly_type.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: q1_fig8_demand_by_elderly_type.png")

def plot_demand_heatmap(theoretical_demand, output_dir):
    """图9：小区-服务类型需求热力图（加分项）"""
    demand_matrix = theoretical_demand.sum(axis=1)  # (10, 6)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(demand_matrix, cmap='YlGnBu', aspect='auto')
    
    for i in range(10):
        for j in range(6):
            ax.text(j, i, f'{int(demand_matrix[i, j])}', ha='center', va='center', 
                   color='white' if demand_matrix[i, j] > demand_matrix.max() * 0.5 else 'black', fontsize=9)
    
    ax.set_xticks(range(6))
    ax.set_xticklabels(SERVICES, fontsize=10)
    ax.set_yticks(range(10))
    ax.set_yticklabels(COMMUNITIES)
    ax.set_xlabel('服务类型', fontsize=12)
    ax.set_ylabel('小区', fontsize=12)
    ax.set_title('小区-服务类型需求热力图', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('月需求次数', fontsize=10)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'q1_fig9_demand_heatmap.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: q1_fig9_demand_heatmap.png")

def plot_theoretical_vs_actual(theoretical_demand, actual_demand, output_dir):
    """图10：理论需求与实际需求对比图（必放）"""
    theoretical_totals = theoretical_demand.sum(axis=(1, 2))  # (10,)
    actual_totals = actual_demand.sum(axis=(1, 2))  # (10,)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(10)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, theoretical_totals, width, label='理论需求', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x + width/2, actual_totals, width, label='实际需求', color='#4CAF50', alpha=0.8)
    
    for bar, th, ac in zip(bars1, theoretical_totals, actual_totals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
               f'{int(th):,}', ha='center', va='bottom', fontsize=7, color='#2196F3')
    for bar, ac in zip(bars2, actual_totals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
               f'{int(ac):,}', ha='center', va='bottom', fontsize=7, color='#4CAF50')
    
    realization_rates = actual_totals / theoretical_totals * 100
    for i, rate in enumerate(realization_rates):
        ax.text(i, max(theoretical_totals[i], actual_totals[i]) + 200, 
               f'{rate:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('小区', fontsize=12)
    ax.set_ylabel('月需求次数', fontsize=12)
    ax.set_title('各小区理论需求与实际需求对比图', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(COMMUNITIES)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'q1_fig10_theoretical_vs_actual.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: q1_fig10_theoretical_vs_actual.png")

def plot_theta_heatmap(theta, output_dir):
    """图11：消费修正系数热力图（必放）"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(theta, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    for i in range(10):
        for j in range(3):
            ax.text(j, i, f'{theta[i, j]:.2f}', ha='center', va='center', 
                   color='white' if theta[i, j] < 0.5 else 'black', fontsize=11, fontweight='bold')
    
    ax.set_xticks(range(3))
    ax.set_xticklabels(ELDERLY_TYPES, fontsize=11)
    ax.set_yticks(range(10))
    ax.set_yticklabels(COMMUNITIES, fontsize=11)
    ax.set_xlabel('老人类型', fontsize=12)
    ax.set_ylabel('小区', fontsize=12)
    ax.set_title('消费修正系数热力图', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('修正系数 θ', fontsize=10)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'q1_fig11_theta_heatmap.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: q1_fig11_theta_heatmap.png")

def plot_service_loss_rate(theoretical_demand, actual_demand, output_dir):
    """图12：各类服务需求损失率对比图（必放）"""
    theoretical_service = theoretical_demand.sum(axis=(0, 1))  # (6,)
    actual_service = actual_demand.sum(axis=(0, 1))  # (6,)
    loss_rates = (1 - actual_service / theoretical_service) * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#FF6B6B' if r > 20 else '#FFD93D' if r > 10 else '#6BCB77' for r in loss_rates]
    bars = ax.bar(SERVICES, loss_rates, color=colors)
    
    for bar, rate in zip(bars, loss_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
               f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.axhline(y=loss_rates.mean(), color='red', linestyle='--', linewidth=1.5, label=f'平均损失率: {loss_rates.mean():.1f}%')
    
    ax.set_xlabel('服务类型', fontsize=12)
    ax.set_ylabel('需求损失率（%）', fontsize=12)
    ax.set_title('各类服务需求损失率对比图', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'q1_fig12_service_loss_rate.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: q1_fig12_service_loss_rate.png")

def plot_income_realization_scatter(monthly_income, realization_rate, output_dir):
    """图13：小区消费能力与需求实现率散点图（加分项）"""
    avg_realization = realization_rate.mean(axis=1)  # (10,)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(monthly_income, avg_realization * 100, c=range(10), cmap='Set1', s=150, edgecolors='black', linewidth=1.5)
    
    for i, community in enumerate(COMMUNITIES):
        ax.annotate(community, (monthly_income[i], avg_realization[i] * 100), 
                   textcoords="offset points", xytext=(5, 5), fontsize=10)
    
    z = np.polyfit(monthly_income, avg_realization * 100, 1)
    p = np.poly1d(z)
    ax.plot(sorted(monthly_income), p(sorted(monthly_income)), "r--", alpha=0.8, label='趋势线')
    
    ax.set_xlabel('小区人均月收入（元）', fontsize=12)
    ax.set_ylabel('需求实现率（%）', fontsize=12)
    ax.set_title('小区消费能力与需求实现率关系图', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    correlation = np.corrcoef(monthly_income, avg_realization)[0, 1]
    ax.text(0.05, 0.95, f'相关系数: {correlation:.3f}', transform=ax.transAxes, fontsize=11, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'q1_fig13_income_realization.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: q1_fig13_income_realization.png")

# ========== 主函数 ==========

if __name__ == "__main__":
    # 导入数据预处理模块
    import sys
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessing_dir = os.path.join(current_dir, '..', '..', 'data_preprocessing', 'q1')
    sys.path.append(os.path.abspath(preprocessing_dir))
    
    from q1_data_preprocessing import (
        preprocess_population_data, 
        preprocess_transition_probabilities, 
        preprocess_service_demand_matrix,
        preprocess_service_prices,
        preprocess_consumption_limits
    )
    
    # 数据预处理
    print("=" * 60)
    print("数据预处理")
    print("=" * 60)
    
    population_matrix = preprocess_population_data()
    transition_matrix = preprocess_transition_probabilities()
    demand_matrix = preprocess_service_demand_matrix()
    service_prices = preprocess_service_prices()
    U_ratio, monthly_income, consumption_limits = preprocess_consumption_limits()
    
    # 问题1.1：未来5年老人数量预测
    prediction_result = model_q1_1(population_matrix, transition_matrix, years=5)
    print_q1_1_result(prediction_result)
    save_q1_1_result(prediction_result)
    
    # 问题1.2：第5年末理论服务需求预测
    demand_result, detailed_demand = model_q1_2(prediction_result, demand_matrix)
    total_demand = save_q1_2_result(demand_result)
    print_q1_2_result(demand_result, detailed_demand, total_demand)
    
    # 问题1.3：消费约束下服务需求修正
    actual_demand, theoretical_demand, theta, realization_rate = model_q1_3(prediction_result, demand_matrix, service_prices, consumption_limits)
    save_q1_3_result(actual_demand, theoretical_demand)
    print_q1_3_summary(actual_demand, theoretical_demand, theta, realization_rate)
    
    # 生成可视化图表
    create_visualizations(prediction_result, theoretical_demand, actual_demand, theta, realization_rate, monthly_income)
    
    print("\n" + "=" * 60)
    print("问题1所有子问题计算完成！")
    print("=" * 60)