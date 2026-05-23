import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')
import os
import random
import time

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'



BRIGHT_COLORS = ['#4A90D9', '#F5A623', '#7ED321', '#D0021B', '#9013FE',
                 '#BD3E3E', '#FF69B4', '#50E3C2', '#F8E71C', '#2EC5C0']


BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, '数据')
OUT_DIR = os.path.join(BASE, '求解', '问题二')
PAPER_DIR = os.path.join(BASE, '论文', 'figures', '问题二')
PREP_DIR = os.path.join(OUT_DIR, '数据预处理')
os.makedirs(os.path.join(OUT_DIR, 'figures'), exist_ok=True)
os.makedirs(PREP_DIR, exist_ok=True)
os.makedirs(PAPER_DIR, exist_ok=True)


def save_fig(fig, name_cn):
    fig.savefig(os.path.join(OUT_DIR, 'figures', name_cn))
    fig.savefig(os.path.join(PAPER_DIR, name_cn))
    plt.close(fig)

def save_csv(df, name_cn):
    df.to_csv(os.path.join(OUT_DIR, name_cn), index=False, encoding='utf-8-sig')

random.seed(42)
np.random.seed(42)

print("=" * 60)
print("问题二：服务站选址与规模优化")
print("=" * 60)


# 1. 数据加载
print("\n>>> 加载数据...")

# 问题一预测结果：第5年末约束后需求
df_demand_q1 = pd.read_csv(f'{BASE}/求解/问题一/数据预处理/预处理数据.csv')
print(f"  需求数据: {df_demand_q1.shape}")

# 转为日需求 q_i（人次/日），取月需求/30
# q_i = Q_i / 30，其中Q_i为小区i的月需求量
df_demand_q1['日需求次数'] = df_demand_q1['约束后月需求次数'] / 30.0
q_i = df_demand_q1.groupby('小区')['日需求次数'].sum().to_dict()
print("  日均需求量 q_i（人次/日）:")
for comm in 'ABCDEFGHIJ':
    if comm in q_i:
        print(f"    小区{comm}: {q_i[comm]:.1f}")

# 加载老人数 N_i（从问题一预测结果）
# N_i表示第5年末小区i的老人总数
df_pop_pred = pd.read_csv(f'{BASE}/求解/问题一/未来五年各小区老人数量预测结果.csv')
df_pop_y5 = df_pop_pred[df_pop_pred['年序'] == 5].set_index('小区')
N_i = df_pop_y5['老人总数'].to_dict()
print(f"\n  第5年末各小区老人数 N_i:")
for comm in 'ABCDEFGHIJ':
    print(f"    小区{comm}: {N_i[comm]}人")

# 附件3：建设运营成本
station_configs = {
    1: {'name': '小型', 'build_cost': 18, 'daily_cost': 2000, 'max_daily': 1000, 'daily_subsidy_limit': 1000},
    2: {'name': '中型', 'build_cost': 32, 'daily_cost': 3200, 'max_daily': 2000, 'daily_subsidy_limit': 1800},
    3: {'name': '大型', 'build_cost': 45, 'daily_cost': 4400, 'max_daily': 3000, 'daily_subsidy_limit': 2600},
}

# 附件4：距离矩阵 d_ij
# d_ij表示小区i到小区j的距离（米）
df_dist_raw = pd.read_excel(
    os.path.join(DATA_DIR , '附件4：小区间距离矩阵.xlsx'), sheet_name='小区间距离矩阵')
dist_data = df_dist_raw.iloc[1:, 1:].values.astype(float)
communities = list('ABCDEFGHIJ')
d_ij = {}  # d_ij: 距离矩阵
for i, ci in enumerate(communities):
    d_ij[ci] = {}
    for j, cj in enumerate(communities):
        d_ij[ci][cj] = dist_data[i][j]

print(f"\n  距离矩阵 d_ij 加载完成 ({len(communities)}×{len(communities)})")

# 检查哪些小区对在1000m内
within_range = {}
for ci in communities:
    within_range[ci] = [cj for cj in communities if d_ij[ci][cj] <= 1000]
print("  各小区1000m内可达小区:")
for ci in communities:
    print(f"    {ci}: {within_range[ci]}")


# 2.2 数据预处理
print("\n>>> 2.2 数据预处理...")

# Step1: 数据清洗与标准化
print("  [Step1] 数据清洗与标准化...")
# 小区集合 I = {1,2,...,10}
I = list(range(1, 11))
# 候选服务站集合 J = {1,2,...,10}
J = list(range(1, 11))
# 服务站规模集合 K = {S, M, L}
K = ['S', 'M', 'L']
# 服务类型集合 R = {1,2,3,4,5,6}
R = list(range(1, 7))

df_sets = pd.DataFrame({
    '集合': ['I', 'J', 'K', 'R'],
    '名称': ['小区集合', '候选服务站集合', '服务站规模集合', '服务类型集合'],
    '元素': [str(I), str(J), str(K), str(R)],
    '含义': ['需求点/候选建站点', '候选建站点', '小型/中型/大型', '助餐/日间照料/上门护理/康复理疗/助浴/紧急救助']
})
df_sets.to_csv(os.path.join(PREP_DIR, 'Step1_集合定义.csv'), index=False, encoding='utf-8-sig')
print(f"    集合定义已保存")

# Step2: 距离矩阵预处理
print("  [Step2] 距离矩阵预处理...")
# d_ij: 小区i到小区j的距离(米)
dist_array = np.zeros((10, 10))
for i, ci in enumerate(communities):
    for j, cj in enumerate(communities):
        dist_array[i, j] = d_ij[ci][cj]

df_dist = pd.DataFrame(dist_array, index=communities, columns=communities)
df_dist.index.name = '小区i'
df_dist.columns.name = '小区j'
df_dist.to_csv(os.path.join(PREP_DIR, 'Step2_距离矩阵_d_ij.csv'), encoding='utf-8-sig')

# 可达矩阵 a_ij
reach_array = np.zeros((10, 10), dtype=int)
for i, ci in enumerate(communities):
    for j, cj in enumerate(communities):
        reach_array[i, j] = 1 if d_ij[ci][cj] <= 1000 else 0

df_reach = pd.DataFrame(reach_array, index=communities, columns=communities)
df_reach.index.name = '小区i'
df_reach.columns.name = '小区j'
df_reach.to_csv(os.path.join(PREP_DIR, 'Step2_可达矩阵_a_ij.csv'), encoding='utf-8-sig')
print(f"    距离矩阵({dist_array.shape})和可达矩阵({reach_array.shape})已保存")

# Step3: 需求量预处理
print("  [Step3] 需求量预处理...")
# Q_ir: 小区i对服务r的月理论需求次数
monthly_demand = df_demand_q1.groupby(['小区', '服务项目'])['约束后月需求次数'].sum().unstack()
df_Q_ir = pd.DataFrame(monthly_demand)
df_Q_ir.index.name = '小区(i)'
df_Q_ir.columns.name = '服务类型(r)'
df_Q_ir.to_csv(os.path.join(PREP_DIR, 'Step3_月需求量矩阵_Q_ir.csv'), encoding='utf-8-sig')

# Q_i = sum_r Q_ir: 小区i的总月需求量
Q_i = monthly_demand.sum(axis=1)

# q_i = Q_i / 30: 日均需求量(人次/日)
q_i = Q_i / 30.0

df_daily_demand = pd.DataFrame({
    '月总需求量(Q_i)': Q_i,
    '日均需求量(q_i)': q_i
})
df_daily_demand.index.name = '小区(i)'
df_daily_demand.to_csv(os.path.join(PREP_DIR, 'Step3_日均需求量_q_i.csv'), encoding='utf-8-sig')
print(f"    月需求量矩阵和日均需求量已保存")

# Step4: 老年人口数据预处理
print("  [Step4] 老年人口数据预处理...")
# N_i_series: 第5年末小区i的老人总数
N_i_series = df_pop_y5['老人总数']

# N = sum_i N_i: 全区域老人总数
N_total = N_i_series.sum()

df_pop = pd.DataFrame({
    '老人总数(N_i)': N_i_series,
    '占比': N_i_series / N_total
})
df_pop.index.name = '小区(i)'
df_pop.to_csv(os.path.join(PREP_DIR, 'Step4_老年人口数据_N_i.csv'), encoding='utf-8-sig')

with open(os.path.join(PREP_DIR, 'Step4_人口汇总.txt'), 'w', encoding='utf-8') as f:
    f.write(f"全区域老人总数 N = {N_total} 人\n")
    f.write(f"小区数量 |I| = {len(I)}\n")
    f.write(f"候选服务站数量 |J| = {len(J)}\n")
    f.write(f"服务站规模类型 |K| = {len(K)}\n")
    f.write(f"服务类型数量 |R| = {len(R)}\n")

print(f"    老年人口数据已保存，全区域老人总数 N = {N_total}人")

# Step5: 服务站配置参数
print("  [Step5] 服务站配置参数...")
df_station = pd.DataFrame({
    '规模等级': ['S', 'M', 'L'],
    '规模名称': ['小型', '中型', '大型'],
    '建设成本(万元)': [18, 32, 45],
    '日运营成本(元)': [2000, 3200, 4400],
    '最大日服务量(人次)': [1000, 2000, 3000],
    '日补贴上限(元)': [1000, 1800, 2600]
})
df_station.set_index('规模等级', inplace=True)
df_station.to_csv(os.path.join(PREP_DIR, 'Step5_服务站配置参数.csv'), encoding='utf-8-sig')
print(f"    服务站配置参数已保存")

print(f"\n  预处理数据已保存至: {PREP_DIR}")


# 2.3 满意度函数
def calc_S1(distance):
    """距离满意度"""
    if distance <= 300:
        return 1.00
    elif distance <= 500:
        return 0.90
    elif distance <= 650:
        return 0.75
    elif distance <= 1000:
        return 0.60
    else:
        return 0.0

def calc_S2(utilization):
    """响应满意度（利用率）"""
    if utilization <= 0.60:
        return 1.00
    elif utilization <= 0.75:
        return 0.93
    elif utilization <= 0.85:
        return 0.85
    elif utilization <= 0.95:
        return 0.72
    elif utilization <= 1.00:
        return 0.60
    else:
        return 0.45  # 超负荷

def calc_S3(price_ratio):
    """价格满意度（price_ratio = 定价/基准价）"""
    if price_ratio <= 1.0:
        return 1.00
    elif price_ratio <= 1.10:
        return 0.90
    elif price_ratio <= 1.20:
        return 0.75
    else:
        return 0.60


# 3. 适应度评估（容量约束的贪心分配）
def evaluate_config(station_config, return_details=False):
    """
    带容量约束的评估：按S1排序，贪心分配，容量满则选次优
    """
    # 小区集合 I，服务站集合 J
    if isinstance(station_config, dict):
        stations = station_config  # y_jk: 小区j是否建设规模k的服务站
    else:
        stations = {}  # y_jk: 小区j是否建设规模k的服务站
        for item in station_config:
            if isinstance(item, tuple):
                stations[item[0]] = item[1]

    if len(stations) == 0:
        return (0, 0, 0) if not return_details else (0, 0, 0, {})

    # 建设预算约束：总建设成本 <= 120万元
    total_build = sum(station_configs[s]['build_cost'] for s in stations.values())
    if total_build > 120:
        return (-1000, 0, 0) if not return_details else (-1000, 0, 0, {})

    J_list = list(stations.keys())  # 已建设服务站的小区集合
    # 每个站点的剩余容量 Cap_j
    capacity_remain = {sj: station_configs[stations[sj]]['max_daily'] for sj in J_list}

    # Step 1: 预计算所有(小区→站点)的距离满意度 S^d_ij，按S^d降序排列
    candidates = []
    for ci in communities:
        for sj in J_list:
            d = d_ij[ci][sj]
            if d <= 1000:
                S_d = calc_S1(d)  # S^d_ij: 距离满意度
                candidates.append((S_d, ci, sj, d))
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Step 2: 贪心分配（优先S^d最高的配对）
    assignment = {}  # z_ij: 小区i是否被服务站j服务
    assigned = set()

    for S_d, ci, sj, d in candidates:
        if ci in assigned:
            continue
        if capacity_remain[sj] >= q_i.get(ci, 0):
            assignment[ci] = sj
            capacity_remain[sj] -= q_i.get(ci, 0)
            assigned.add(ci)

    # 未被分配的小区尝试次优选择（必须满足容量约束）
    for ci in communities:
        if ci in assigned:
            continue
        best_sj = None
        best_S_d = 0
        for sj in J_list:
            d = d_ij[ci][sj]
            if d <= 1000 and capacity_remain[sj] >= q_i.get(ci, 0):
                S_d = calc_S1(d)
                if S_d > best_S_d:
                    best_S_d = S_d
                    best_sj = sj
        if best_sj:
            assignment[ci] = best_sj
            capacity_remain[best_sj] -= q_i.get(ci, 0)

    if len(assignment) == 0:
        return (0, 0, 0) if not return_details else (0, 0, 0, {})

    # Step 3: 计算负荷率 ρ_j → 响应满意度 S^r_ij
    D_j = {}  # D_j: 服务站j每日承担的服务需求量
    for sj in J_list:
        load = 0.0
        for ci, sj2 in assignment.items():
            if sj2 == sj:
                load += q_i.get(ci, 0)
        D_j[sj] = load

    rho_j = {}  # ρ_j: 服务站j的负荷率
    station_S2 = {}
    for sj in J_list:
        max_cap = station_configs[stations[sj]]['max_daily']
        rho_j[sj] = D_j.get(sj, 0) / max_cap
        station_S2[sj] = calc_S2(rho_j[sj])

    # Step 4: 综合满意度 S_ij = α_d*S^d + α_r*S^r + α_p*S^p
    # 其中 α_d=0.2, α_r=0.3, α_p=0.5, S^p=1.0（价格满意度暂设为1）
    final_satisfaction = {}
    for ci, sj in assignment.items():
        d = d_ij[ci][sj]
        S_d = calc_S1(d)
        S_r_j = station_S2[sj]
        S = 0.2 * S_d + 0.3 * S_r_j + 0.5 * 1.0
        final_satisfaction[ci] = S

    # Step 5: 计算覆盖率 Cov = Σ N_i * u_i / Σ N_i
    covered_N = sum(N_i.get(ci, 0) for ci in assignment)
    total_N = sum(N_i.values())
    Cov = covered_N / total_N if total_N > 0 else 0

    # Step 6: 计算平均满意度 S̄
    avg_S = np.mean(list(final_satisfaction.values())) if final_satisfaction else 0

    # 惩罚未使用的站点
    unused_penalty = 0
    for sj in J_list:
        if D_j.get(sj, 0) < 1.0:  # 日服务<1人次视为闲置
            unused_penalty += 0.05

    # 综合目标函数 F = λ*Cov + (1-λ)*S̄，其中λ=0.6
    F = 0.6 * Cov + 0.4 * avg_S - unused_penalty

    if return_details:
        details = {
            'stations': stations,          # y_jk: 服务站建设方案
            'assignment': assignment,      # z_ij: 小区-站点分配关系
            'satisfaction': final_satisfaction,  # S_ij: 各小区满意度
            'station_load': D_j,           # D_j: 各站点负荷
            'station_util': rho_j,         # ρ_j: 各站点负荷率
            'coverage': Cov,               # Cov: 服务覆盖率
            'avg_satisfaction': avg_S,     # S̄: 平均满意度
            'total_build': total_build,    # 总建设成本
            'fitness': F                   # F: 综合目标函数值
        }
        return F, Cov, avg_S, details

    return F, Cov, avg_S


# 4. 遗传算法求解
print("\n>>> 遗传算法优化...")

POP_SIZE = 150
N_GENERATIONS = 200
MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.8
ELITE_SIZE = 15

def decode_chromosome(chrom):
    """解码染色体为站点配置字典"""
    stations = {}
    for i, gene in enumerate(chrom):
        if gene > 0:
            stations[communities[i]] = int(gene)
    return stations

def fitness(chrom):
    """适应度函数"""
    stations = decode_chromosome(chrom)
    # 检查预算
    total_build = sum(station_configs[s]['build_cost'] for s in stations.values())
    if total_build > 120:
        return -1000 + (120 - total_build) * 0.1  # 罚函数
    fit, _, _ = evaluate_config(stations)
    return fit

def create_individual():
    """创建随机个体"""
    chrom = np.zeros(10, dtype=int)
    budget_used = 0
    # 随机决定建站数量和位置
    order = list(range(10))
    random.shuffle(order)
    for i in order:
        if budget_used >= 120:
            break
        # 随机选规模
        available = []
        for scale in [1, 2, 3]:
            if budget_used + station_configs[scale]['build_cost'] <= 120:
                available.append(scale)
        if available:
            scale = random.choice(available)
            chrom[i] = scale
            budget_used += station_configs[scale]['build_cost']
    return chrom

def crossover(p1, p2):
    """均匀交叉"""
    if random.random() > CROSSOVER_RATE:
        return p1.copy(), p2.copy()
    c1, c2 = p1.copy(), p2.copy()
    for i in range(10):
        if random.random() < 0.5:
            c1[i], c2[i] = c2[i], c1[i]
    return c1, c2

def mutate(chrom):
    """变异"""
    chrom = chrom.copy()
    for i in range(10):
        if random.random() < MUTATION_RATE:
            chrom[i] = random.randint(0, 3)
    return chrom

# 初始化种群
population = [create_individual() for _ in range(POP_SIZE)]

# 进化
best_fitness_history = []
best_solution = None
best_fitness_val = -float('inf')

start_time = time.time()

for gen in range(N_GENERATIONS):
    # 评估适应度
    fitnesses = [fitness(ind) for ind in population]

    # 记录最佳
    best_idx = np.argmax(fitnesses)
    if fitnesses[best_idx] > best_fitness_val:
        best_fitness_val = fitnesses[best_idx]
        best_solution = population[best_idx].copy()

    best_fitness_history.append(best_fitness_val)

    if gen % 50 == 0:
        print(f"  第{gen}代: 最佳适应度={best_fitness_val:.4f}")

    # 精英保留
    elite_indices = np.argsort(fitnesses)[-ELITE_SIZE:]
    elites = [population[i].copy() for i in elite_indices]

    # 选择
    new_population = elites.copy()
    while len(new_population) < POP_SIZE:
        t1 = random.randint(0, POP_SIZE - 1)
        t2 = random.randint(0, POP_SIZE - 1)
        parent1 = population[t1] if fitnesses[t1] > fitnesses[t2] else population[t2]

        t1 = random.randint(0, POP_SIZE - 1)
        t2 = random.randint(0, POP_SIZE - 1)
        parent2 = population[t1] if fitnesses[t1] > fitnesses[t2] else population[t2]

        # 交叉
        child1, child2 = crossover(parent1, parent2)
        # 变异
        child1 = mutate(child1)
        child2 = mutate(child2)

        new_population.append(child1)
        if len(new_population) < POP_SIZE:
            new_population.append(child2)

    population = new_population

elapsed = time.time() - start_time
print(f"\n  GA完成: {N_GENERATIONS}代, 耗时{elapsed:.1f}秒")
print(f"  最佳适应度: {best_fitness_val:.4f}")


# 5. 局部搜索优化
print("\n>>> 局部搜索优化...")

def local_search(chrom, iterations=500):
    """对GA结果进行局部搜索"""
    best_chrom = chrom.copy()
    best_fit = fitness(best_chrom)

    for _ in range(iterations):
        # 随机修改一个位置
        i = random.randint(0, 9)
        old_gene = best_chrom[i]
        new_gene = random.randint(0, 3)
        if new_gene == old_gene:
            continue

        best_chrom[i] = new_gene
        new_fit = fitness(best_chrom)
        if new_fit > best_fit:
            best_fit = new_fit
        else:
            best_chrom[i] = old_gene  # 回退

    return best_chrom, best_fit

best_solution, best_fitness_val = local_search(best_solution)
print(f"  局部搜索后适应度: {best_fitness_val:.4f}")


# 6. 结果分析
print("\n>>> 最优方案分析...")

best_stations = decode_chromosome(best_solution)
fit, cov, sat, details = evaluate_config(best_stations, return_details=True)

print(f"\n  === 最优方案 ===")
print(f"  站点数量: {len(best_stations)}")
print(f"  总建设成本: {details['total_build']}万元")
print(f"  服务覆盖率: {details['coverage']*100:.1f}%")
print(f"  平均满意度: {details['avg_satisfaction']:.4f}")
print(f"  适应度得分: {details['fitness']:.4f}")

print(f"\n  各服务站详情:")
station_info_rows = []
for sj, scale in best_stations.items():
    cfg = station_configs[scale]
    load = details['station_load'].get(sj, 0)
    util = details['station_util'].get(sj, 0)
    S2 = calc_S2(util)
    covered = [ci for ci, st in details['assignment'].items() if st == sj]
    print(f"    小区{sj} ({cfg['name']}): 建设费{cfg['build_cost']}万, "
          f"日管理费{cfg['daily_cost']}元, 日服务{load:.0f}人次, "
          f"利用率{util*100:.1f}%, 覆盖{len(covered)}个小区: {covered}")

    station_info_rows.append({
        '服务站位置': sj,
        '规模': cfg['name'],
        '建设成本(万元)': cfg['build_cost'],
        '日管理成本(元)': cfg['daily_cost'],
        '日最大服务人次': cfg['max_daily'],
        '实际日服务人次': round(load, 1),
        '利用率': f"{util*100:.1f}%",
        '覆盖小区数': len(covered),
        '覆盖小区': ','.join(covered)
    })

df_station_info = pd.DataFrame(station_info_rows)
save_csv(df_station_info, '最优服务站选址与规模方案.csv')

# 覆盖明细
print(f"\n  各小区满意度:")
coverage_rows = []
for ci in communities:
    if ci in details['assignment']:
        sj = details['assignment'][ci]
        d = d_ij[ci][sj]
        S1 = calc_S1(d)
        scale = best_stations[sj]
        S2 = calc_S2(details['station_util'][sj])
        S = details['satisfaction'][ci]
        print(f"    {ci} → {sj}: 距离{d}m, S1={S1:.2f}, S2={S2:.2f}, S={S:.4f}")
        coverage_rows.append({
            '小区': ci,
            '分配服务站': sj,
            '距离(m)': d,
            'S1距离满意度': S1,
            'S2响应满意度': S2,
            'S3价格满意度': 1.0,
            '综合满意度S': round(S, 4)
        })
    else:
        print(f"    {ci}: 未被覆盖!")
        coverage_rows.append({
            '小区': ci,
            '分配服务站': '无',
            '距离(m)': '-',
            'S1距离满意度': '-',
            'S2响应满意度': '-',
            'S3价格满意度': '-',
            '综合满意度S': '-'
        })

df_coverage = pd.DataFrame(coverage_rows)
save_csv(df_coverage, '各服务站覆盖小区明细.csv')

# 年度利润估算（问题2按基准价）
print(f"\n  === 年度利润估算 ===")
profit_rows = []
for sj, scale in best_stations.items():
    cfg = station_configs[scale]
    load = details['station_load'].get(sj, 0)

    # 年服务营收和直接支出（按基准价）
    annual_revenue = 0
    annual_direct_cost = 0
    covered_comms = [ci for ci, st in details['assignment'].items() if st == sj]

    # 重新加载营收和支出数据来计算
    df_revenue = pd.read_excel(
        os.path.join(DATA_DIR , '附件2：服务需求数据.xlsx'), sheet_name='服务营收及支出')
    df_revenue = df_revenue.iloc[1:].copy()
    df_revenue.columns = ['服务项目', '单次服务营收', '单次服务直接支出']
    svc_price = {}
    svc_cost = {}
    for _, row in df_revenue.iterrows():
        svc = row['服务项目']
        p = row['单次服务营收']
        c = row['单次服务直接支出']
        if pd.isna(p) or (isinstance(p, str) and '免费' in str(p)):
            p = 0.0
        if pd.isna(c) or (isinstance(c, str) and '免费' in str(c)):
            c = 0.0
        svc_price[svc] = float(p)
        svc_cost[svc] = float(c)

    # 从需求数据计算各服务站营收
    demand_data = df_demand_q1[df_demand_q1['小区'].isin(covered_comms)]
    for _, row in demand_data.iterrows():
        svc = row['服务项目']
        daily_d = row['日需求次数']
        annual_revenue += svc_price[svc] * daily_d * 365
        annual_direct_cost += svc_cost[svc] * daily_d * 365

    annual_op_cost = cfg['build_cost'] * 10000 / 20 + cfg['daily_cost'] * 365
    # 服务利润 = 营收 - 直接支出
    annual_service_profit = annual_revenue - annual_direct_cost
    # 问题二无补贴，利润 = 服务利润 - 运营成本
    annual_net_profit = annual_service_profit - annual_op_cost
    profit_rate = annual_net_profit / annual_op_cost * 100 if annual_op_cost > 0 else 0

    print(f"    站点{sj}({cfg['name']}): 年营收{annual_revenue/10000:.1f}万, "
          f"年直接成本{annual_direct_cost/10000:.1f}万, 年服务利润{annual_service_profit/10000:.1f}万, "
          f"年运营成本{annual_op_cost/10000:.1f}万, 年净利润{annual_net_profit/10000:.1f}万, "
          f"利润率{profit_rate:.1f}%")

    profit_rows.append({
        '服务站': f"小区{sj}({cfg['name']})",
        '年营收(万元)': round(annual_revenue/10000, 2),
        '年直接成本(万元)': round(annual_direct_cost/10000, 2),
        '年服务利润(万元)': round(annual_service_profit/10000, 2),
        '年运营成本(万元)': round(annual_op_cost/10000, 2),
        '年净利润(万元)': round(annual_net_profit/10000, 2),
        '利润率': f"{profit_rate:.1f}%"
    })

df_profit = pd.DataFrame(profit_rows)


# 7. 可视化
print("\n>>> 生成可视化...")

# 图1：收敛曲线
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(best_fitness_history, color='#4A90D9', linewidth=2)
ax.set_xlabel('迭代代数', fontsize=12)
ax.set_ylabel('最佳适应度', fontsize=12)
ax.set_title('遗传算法优化收敛曲线', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, '优化目标收敛曲线.png')
print("  [1/12] 收敛曲线 已保存")

# 图2：服务站选址与覆盖关系图
fig, ax = plt.subplots(figsize=(14, 12))

# 绘制小区位置（居中布局，稍向右移）
positions = {
    'A': (-2.5, 1.5), 'B': (-0.5, 1.5), 'C': (1.5, 1.5), 'D': (3.5, 1.5),
    'E': (-2.5, -0.5), 'F': (-0.5, -0.5), 'G': (1.5, -0.5), 'H': (3.5, -0.5),
    'I': (-1.5, -2.5), 'J': (2.5, -2.5)
}

# 绘制背景网格
major_ticks = np.arange(-4, 5, 2)
minor_ticks = np.arange(-4, 5, 1)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.grid(which='major', color='#E0E0E0', linestyle='-', linewidth=1.5)
ax.grid(which='minor', color='#F0F0F0', linestyle=':', linewidth=0.5)

# 绘制覆盖区域（站点周围的渐变圆）
for ci in best_stations:
    x, y = positions[ci]
    # 外圈 - 1000m服务半径
    circle_outer = plt.Circle((x, y), 1.8, fill=True, color='#4A90D9', alpha=0.08)
    ax.add_patch(circle_outer)
    # 内圈 - 650m优质服务半径
    circle_mid = plt.Circle((x, y), 1.2, fill=True, color='#7ED321', alpha=0.1)
    ax.add_patch(circle_mid)
    # 核心圈 - 300m最佳服务半径
    circle_inner = plt.Circle((x, y), 0.6, fill=True, color='#F5A623', alpha=0.15)
    ax.add_patch(circle_inner)

for ci, sj in details['assignment'].items():
    x1, y1 = positions[ci]
    x2, y2 = positions[sj]
    # 连接线颜色根据距离满意度变化
    d = d_ij[ci][sj]
    if d <= 300:
        line_color = '#7ED321'
    elif d <= 500:
        line_color = '#F5A623'
    elif d <= 650:
        line_color = '#FFB347'
    else:
        line_color = '#D0021B'
    ax.plot([x1, x2], [y1, y2], color=line_color, linewidth=2.5, 
            alpha=0.7, zorder=3, linestyle='-')
    ax.scatter(x1, y1, s=30, c=line_color, zorder=4)

# 绘制站点
scale_info = {1: {'name': '小型', 'size': 350, 'color': '#4A90D9', 'marker': 'h'},
              2: {'name': '中型', 'size': 500, 'color': '#F5A623', 'marker': 'h'},
              3: {'name': '大型', 'size': 700, 'color': '#D0021B', 'marker': 'h'}}

for ci in communities:
    x, y = positions[ci]
    if ci in best_stations:
        scale = best_stations[ci]
        info = scale_info[scale]
        ax.scatter(x, y, s=info['size'], c=info['color'], edgecolors='black',
                  linewidth=2, zorder=6, marker=info['marker'],
                  label=f'{info["name"]}服务站' if ci == list(best_stations.keys())[0] else '')
        # 站点编号标注
        ax.text(x, y + 0.25, ci, fontsize=12, ha='center', va='bottom',
               fontweight='bold', zorder=7, color=info['color'])
    else:
        ax.scatter(x, y, s=180, c='white', edgecolors='#90A4AE',
                  linewidth=2, zorder=5, marker='o',
                  label='普通小区' if ci == communities[0] else '')
        ax.text(x, y + 0.2, ci, fontsize=11, ha='center', va='bottom',
               fontweight='bold', zorder=7, color='#546E7A')

ax.plot([-3, -1], [-3.5, -3.5], color='#90A4AE', linewidth=2)
ax.text(-2, -3.8, '2000m', fontsize=10, ha='center', color='#546E7A')

ax.set_xlim(-4.2, 4.8)
ax.set_ylim(-4.0, 3.5)
ax.set_aspect('equal')
ax.set_title('服务站选址与覆盖关系图', fontsize=16, fontweight='bold', pad=20)

# 图例 - 简化并放在左下角
from matplotlib.lines import Line2D

# 服务站类型图例
station_handles = []
station_labels = []
for scale in [1, 2, 3]:
    info = scale_info[scale]
    station_handles.append(Line2D([0], [0], marker=info['marker'], color=info['color'], 
                                  markersize=9, linestyle='None', markeredgecolor='black'))
    station_labels.append(f"{info['name']}站")
station_handles.append(Line2D([0], [0], marker='o', color='white', 
                              markersize=7, linestyle='None', markeredgecolor='#90A4AE'))
station_labels.append("小区")

# 距离满意度图例
dist_handles = [
    Line2D([0], [0], color='#7ED321', lw=2.5, label='≤300m'),
    Line2D([0], [0], color='#F5A623', lw=2.5, label='300-500m'),
    Line2D([0], [0], color='#FFB347', lw=2.5, label='500-650m'),
    Line2D([0], [0], color='#D0021B', lw=2.5, label='650-1000m')
]
dist_labels = ['≤300m', '300-500m', '500-650m', '650-1000m']

# 创建组合图例
handles = station_handles + dist_handles
labels = station_labels + dist_labels

ax.legend(handles, labels, loc='lower left', fontsize=8.5, 
          framealpha=0.95, borderaxespad=1, ncol=2,
          title='图例', title_fontsize=10)

fig.tight_layout()
save_fig(fig, '服务站选址与覆盖关系图.png')
print("  [2/12] 选址与覆盖关系图 已保存")

# 图3：各小区满意度分解图（堆叠条形图）
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(communities))
width = 0.6

s1_vals = []
s2_vals = []
s3_vals = []
for ci in communities:
    if ci in details['assignment']:
        sj = details['assignment'][ci]
        d = d_ij[ci][sj]
        s1_vals.append(0.2 * calc_S1(d))
        s2_vals.append(0.3 * calc_S2(details['station_util'][sj]))
        s3_vals.append(0.5 * 1.0)
    else:
        s1_vals.append(0)
        s2_vals.append(0)
        s3_vals.append(0)

ax.bar(x, s1_vals, width, label='0.2×S1(距离)', color='#4A90D9', alpha=0.85)
ax.bar(x, s2_vals, width, bottom=s1_vals, label='0.3×S2(响应)', color='#F5A623', alpha=0.85)
ax.bar(x, s3_vals, width, bottom=np.array(s1_vals)+np.array(s2_vals),
       label='0.5×S3(价格)', color='#7ED321', alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(communities, fontsize=11)
ax.set_ylabel('满意度得分', fontsize=12)
ax.set_title('各小区满意度分解', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0, 1.1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, '各小区满意度分解图.png')
print("  [3/12] 满意度分解图 已保存")

# 图4：各小区日均需求柱状图
fig, ax = plt.subplots(figsize=(12, 6))
demand_vals = [q_i.get(ci, 0) for ci in communities]
bars = ax.bar(communities, demand_vals, color=BRIGHT_COLORS[:10], alpha=0.85)
ax.axhline(y=1000, color='#D0021B', linestyle='--', label='小型站容量(1000)')
ax.axhline(y=2000, color='#F5A623', linestyle='--', label='中型站容量(2000)')
ax.axhline(y=3000, color='#7ED321', linestyle='--', label='大型站容量(3000)')
ax.set_xlabel('小区', fontsize=12)
ax.set_ylabel('日均需求量(人次/日)', fontsize=12)
ax.set_title('各小区日均需求柱状图', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, '各小区日均需求柱状图.png')
print("  [4/12] 各小区日均需求柱状图 已保存")

# 图5：需求热力图（小区×服务类型）
fig, ax = plt.subplots(figsize=(12, 7))
monthly_demand_matrix = df_demand_q1.pivot_table(values='约束后月需求次数', 
                                                  index='小区', columns='服务项目', 
                                                  aggfunc='sum', fill_value=0)
monthly_demand_matrix = monthly_demand_matrix.reindex(communities)
import seaborn as sns
im = sns.heatmap(monthly_demand_matrix, cmap='BuPu', annot=True, fmt='.0f', 
                 ax=ax, cbar=True, linewidths=0.5)
ax.set_xlabel('服务类型', fontsize=12)
ax.set_ylabel('小区', fontsize=12)
ax.set_title('需求热力图（小区×服务类型）', fontsize=14, fontweight='bold')
fig.tight_layout()
save_fig(fig, '需求热力图_小区服务类型.png')
print("  [5/12] 需求热力图 已保存")


# 图6：站点负荷条形图（实际/容量）
fig, ax = plt.subplots(figsize=(10, 6))
station_names = list(details['station_load'].keys())
load_vals = [details['station_load'][sj] for sj in station_names]
max_cap = [station_configs[best_stations[sj]]['max_daily'] for sj in station_names]
x = np.arange(len(station_names))
width = 0.35
ax.bar(x - width/2, load_vals, width, label='实际负荷', color='#4A90D9', alpha=0.85)
ax.bar(x + width/2, max_cap, width, label='最大容量', color='#F5A623', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f'站点{s}' for s in station_names], fontsize=11)
ax.set_ylabel('日服务人次', fontsize=12)
ax.set_title('站点负荷条形图', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.7)
fig.tight_layout()
fig.tight_layout()
save_fig(fig, '站点负荷条形图.png')
print("  [6/12] 站点负荷条形图 已保存")


# 9. 成本与效益类图表
# 图7：总建设成本构成饼图
build_costs = {sj: station_configs[best_stations[sj]]['build_cost'] for sj in best_stations}
fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(build_costs.values(), labels=build_costs.keys(), 
                                   autopct='%1.1f%%', colors=BRIGHT_COLORS[:len(build_costs)],
                                   startangle=90, pctdistance=0.85)
ax.set_title('总建设成本构成', fontsize=14, fontweight='bold')
fig.tight_layout()
save_fig(fig, '建设成本构成饼图.png')
print("  [7/12] 建设成本构成饼图 已保存")

# 图8：年运营成本构成
fig, ax = plt.subplots(figsize=(12, 6))
fixed_costs = [station_configs[best_stations[sj]]['daily_cost'] * 365 / 10000 
               for sj in station_names]
variable_costs = [details['station_load'].get(sj, 0) * 30 * 12 * 20 / 10000  # 假设20元/次
                  for sj in station_names]
ax.bar(station_names, fixed_costs, label='固定成本(万元/年)', color='#4A90D9', alpha=0.85)
ax.bar(station_names, variable_costs, bottom=fixed_costs, 
       label='可变成本(万元/年)', color='#7ED321', alpha=0.85)
ax.set_xlabel('站点', fontsize=12)
ax.set_ylabel('年成本(万元)', fontsize=12)
ax.set_title('年运营成本构成', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.7)
fig.tight_layout()
save_fig(fig, '年运营成本构成图.png')
print("  [8/12] 年运营成本构成图 已保存")


# 10. 满意度与距离效应类图表
# 图9：S1距离满意度衰减曲线
fig, ax = plt.subplots(figsize=(10, 6))
distances = np.linspace(0, 1200, 1201)
S1_vals = np.array([calc_S1(d) for d in distances])
ax.plot(distances, S1_vals, linewidth=3, color='#4A90D9', alpha=0.9)
ax.scatter([300, 500, 650, 1000], [1.0, 0.9, 0.75, 0.6], 
           s=100, color='#D0021B', zorder=5)
ax.set_xlabel('距离(米)', fontsize=12)
ax.set_ylabel('S1距离满意度', fontsize=12)
ax.set_title('S1距离满意度衰减曲线', fontsize=14, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_xlim(0, 1200)
ax.set_ylim(0, 1.1)
fig.tight_layout()
save_fig(fig, 'S1距离满意度衰减曲线.png')
print("  [9/12] S1距离满意度衰减曲线 已保存")

# 图10：S2响应满意度函数曲线
fig, ax = plt.subplots(figsize=(10, 6))
utilization = np.linspace(0, 1.2, 121)
S2_vals = np.array([calc_S2(u) for u in utilization])
ax.plot(utilization, S2_vals, linewidth=3, color='#F5A623', alpha=0.9)
ax.set_xlabel('利用率', fontsize=12)
ax.set_ylabel('S2响应满意度', fontsize=12)
ax.set_title('S2响应满意度函数曲线', fontsize=14, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_xlim(0, 1.2)
ax.set_ylim(0, 1.1)
fig.tight_layout()
save_fig(fig, 'S2响应满意度函数曲线.png')
print("  [10/12] S2响应满意度函数曲线 已保存")

# 图11：各小区实际距离 vs S1满意度散点图
fig, ax = plt.subplots(figsize=(10, 6))
for ci in details['assignment']:
    sj = details['assignment'][ci]
    d = d_ij[ci][sj]
    s1 = calc_S1(d)
    pop_size = N_i.get(ci, 0) / 10
    ax.scatter(d, s1, s=pop_size, alpha=0.7, edgecolors='black',
               color=BRIGHT_COLORS[station_names.index(sj) % len(BRIGHT_COLORS)],
               label=f'站点{sj}' if sj not in [details['assignment'].get(c, '') for c in ax.collections] else '')
ax.plot([0, 1000], [1.0, 0.6], 'k--', alpha=0.5)
ax.set_xlabel('距离(米)', fontsize=12)
ax.set_ylabel('S1满意度', fontsize=12)
ax.set_title('各小区实际距离 vs S1满意度', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_xlim(0, 1100)
fig.tight_layout()
save_fig(fig, '距离vs满意度散点图.png')
print("  [11/12] 距离vs满意度散点图 已保存")


# 11. 综合Dashboard
# 图12：选址方案综合仪表板
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4)

# 子图1：选址示意图
ax1 = fig.add_subplot(gs[:, :2])
for ci in communities:
    x, y = positions[ci]
    if ci in best_stations:
        ax1.scatter(x, y, s=300, c='#D0021B', edgecolors='black', linewidth=2, marker='s', zorder=5)
    else:
        ax1.scatter(x, y, s=200, c='#ABD9E9', edgecolors='black', linewidth=1, marker='o', zorder=4)
    ax1.text(x, y + 0.2, ci, fontsize=12, ha='center', va='bottom', fontweight='bold')
for ci in best_stations:
    x, y = positions[ci]
    circle = plt.Circle((x, y), 1.2, fill=False, color='#D0021B', alpha=0.2, linestyle='-')
    ax1.add_patch(circle)
ax1.set_xlim(-4.5, 4.5)
ax1.set_ylim(-4.5, 4.5)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('服务站选址示意图', fontsize=13, fontweight='bold')

# 子图2：负荷率
ax2 = fig.add_subplot(gs[0, 2])
util_rates = [details['station_util'].get(sj, 0) * 100 for sj in station_names]
ax2.barh(station_names, util_rates, color=BRIGHT_COLORS[:len(station_names)], alpha=0.85)
ax2.axvline(x=100, color='#D0021B', linestyle='--')
ax2.set_xlabel('负荷率(%)')
ax2.set_title('站点负荷率', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 120)

# 子图3：满意度分布
ax3 = fig.add_subplot(gs[0, 3])
sat_vals = list(details['satisfaction'].values())
ax3.hist(sat_vals, bins=10, color='#4A90D9', alpha=0.85)
ax3.set_xlabel('满意度')
ax3.set_ylabel('小区数')
ax3.set_title('满意度分布', fontsize=12, fontweight='bold')

# 子图4：成本利润表
ax4 = fig.add_subplot(gs[1:, 2:])
ax4.axis('off')
# 计算覆盖老人数
covered_pop_count = sum(N_i.get(ci, 0) for ci in details['assignment'])
summary_data = [
    ['指标', '数值'],
    ['服务站数量', len(best_stations)],
    ['建设成本(万元)', sum(station_configs[s]['build_cost'] for s in best_stations.values())],
    ['日运营成本(元)', sum(station_configs[s]['daily_cost'] for s in best_stations.values())],
    ['覆盖老人数', covered_pop_count],
    ['覆盖率(%)', f'{details["coverage"] * 100:.1f}'],
    ['平均满意度', f'{details["avg_satisfaction"]:.3f}']
]
table = ax4.table(cellText=summary_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.5)
ax4.set_title('方案汇总表', fontsize=12, fontweight='bold')

fig.tight_layout()
save_fig(fig, '选址方案综合仪表板.png')
print("  [12/12] 选址方案综合仪表板 已保存")


print("\n" + "=" * 60)
print("问题二求解完成！")
print("=" * 60)

print(f"""
  >>> 算法时间复杂度分析 <<<
  遗传算法: O(G × P × N²)
    G = {N_GENERATIONS} (代数)
    P = {POP_SIZE} (种群规模)
    N = 10 (小区数)
  每次适应度评估需要迭代分配(最多5轮)，每轮遍历N个小区和M个站点(M≤N)
  总体复杂度: O(G × P × N²) ≈ {N_GENERATIONS} × {POP_SIZE} × 100 ≈ {N_GENERATIONS * POP_SIZE * 100:,} 次基本操作

  >>> 模型局限性 <<<
  1. 静态需求假设：当前模型仅基于第5年末的静态需求进行选址，未考虑6-10年内人口结构的持续变化带来的需求演化
  2. 确定性建模：所有参数（需求、成本、距离）均为确定值，未纳入实际运营中的随机波动（如需求季节性变化、突发公共卫生事件）
  3. 单一分配假设：每位老人只能选择一个服务站，实际中老人可能根据服务类型选择不同站点
  4. 满意度独立性假设：S1、S2、S3之间可能存在交互效应，但模型采用线性加权综合

  >>> 改进方向 <<<
  引入多期动态选址模型（Multi-period Facility Location），在规划初期就考虑未来10-15年
  内各小区老龄化演进的差异化路径，通过随机规划或鲁棒优化方法在选址决策中预留弹性扩展空间。
""")
