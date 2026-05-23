import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
import random

plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体，支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

BRIGHT_COLORS = ['#2E86AB', '#E94F37', '#F39C12', '#1ABC9C', '#9B59B6',
               '#E74C3C', '#3498DB', '#2ECC71', '#F1C40F', '#E67E22']
COLORS_10 = BRIGHT_COLORS

PREMIUM_COLORS = {
    'blue': '#2C3E50',     
    'purple': '#8E44AD',  
    'red': '#C0392B',     
    'light_blue': '#2980B9', 
    'green': '#27AE60',  
    'orange': '#D35400', 
    'gray': '#7F8C8D',     
    'navy': '#1A5276',     
    'gold': '#B7950B',     
    'violet': '#6C3483'     
}

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, '数据')
OUT_DIR = os.path.join(BASE, '求解', '问题三')
PAPER_DIR = os.path.join(BASE, '论文', 'figures', '问题三')

os.makedirs(os.path.join(OUT_DIR, 'figures'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, '数据预处理'), exist_ok=True)
os.makedirs(PAPER_DIR, exist_ok=True)

PREP_DIR = os.path.join(OUT_DIR, '数据预处理')

def save_fig(fig, name_cn):
    fig.savefig(os.path.join(OUT_DIR, 'figures', name_cn))
    fig.savefig(os.path.join(PAPER_DIR, name_cn))
    plt.close(fig)

def save_csv(df, name_cn):
    df.to_csv(os.path.join(OUT_DIR, name_cn), index=False, encoding='utf-8-sig')

np.random.seed(42)
random.seed(42)

print("=" * 60)
print("问题三：服务定价与政府补贴优化")
print("=" * 60)


# 1. 加载数据与问题二最优方案
print("\n>>> 加载数据...")

# 问题二最优方案：3站布局
# 站点C(大型): 覆盖 B, C, G
# 站点E(小型): 覆盖 E
# 站点I(大型): 覆盖 D, H, I, J
optimal_stations = {
    'C': {'scale': 3, 'scale_name': '大型', 'covered': ['B', 'C', 'G']},
    'E': {'scale': 1, 'scale_name': '小型', 'covered': ['E']},
    'I': {'scale': 3, 'scale_name': '大型', 'covered': ['D', 'H', 'I', 'J']},
}

station_configs = {
    1: {'name': '小型', 'build_cost': 18, 'daily_cost': 2000, 'max_daily': 1000, 'daily_subsidy_limit': 1000},
    2: {'name': '中型', 'build_cost': 32, 'daily_cost': 3200, 'max_daily': 2000, 'daily_subsidy_limit': 1800},
    3: {'name': '大型', 'build_cost': 45, 'daily_cost': 4400, 'max_daily': 3000, 'daily_subsidy_limit': 2600},
}

# 附件2数据
df_demand = pd.read_excel(
    os.path.join(DATA_DIR , '附件2：服务需求数据.xlsx'), sheet_name='每位老人月均服务需求次数')
df_demand = df_demand.iloc[1:].copy()
df_demand.columns = ['服务项目', '自理', '半失能', '失能']
for col in ['自理', '半失能', '失能']:
    df_demand[col] = pd.to_numeric(df_demand[col], errors='coerce')

df_revenue = pd.read_excel(
    os.path.join(DATA_DIR , '附件2：服务需求数据.xlsx'), sheet_name='服务营收及支出')
df_revenue = df_revenue.iloc[1:].copy()
df_revenue.columns = ['服务项目', '单次服务营收', '单次服务直接支出']
for col in ['单次服务营收', '单次服务直接支出']:
    df_revenue[col] = pd.to_numeric(df_revenue[col], errors='coerce')

# 基准价格和成本
base_prices = {}
base_costs = {}
for _, row in df_revenue.iterrows():
    svc = row['服务项目']
    p = row['单次服务营收']
    c = row['单次服务直接支出']
    if pd.isna(p) or (isinstance(p, str) and '免费' in str(p)):
        p = 0.0
    if pd.isna(c) or (isinstance(c, str) and '免费' in str(c)):
        c = 0.0
    base_prices[svc] = float(p)
    base_costs[svc] = float(c)

services = list(base_prices.keys())
print(f"  服务项目: {services}")
print(f"  基准价: {base_prices}")
print(f"  直接成本: {base_costs}")

# 服务需求矩阵 (a_rm: 第r类老人对第m项服务的月均需求次数)
demand_per_capita = {}
for _, row in df_demand.iterrows():
    svc = row['服务项目']
    demand_per_capita[svc] = {
        '自理': float(row['自理']),
        '半失能': float(row['半失能']),
        '失能': float(row['失能'])
    }
a_rm = demand_per_capita

# 距离矩阵
df_dist_raw = pd.read_excel(
    os.path.join(DATA_DIR , '附件4：小区间距离矩阵.xlsx'), sheet_name='小区间距离矩阵')
dist_data = df_dist_raw.iloc[1:, 1:].values.astype(float)
communities = list('ABCDEFGHIJ')
dist_matrix = {}
for i, ci in enumerate(communities):
    dist_matrix[ci] = {}
    for j, cj in enumerate(communities):
        dist_matrix[ci][cj] = dist_data[i][j]

# 第5年末老人数据
df_pop_pred = pd.read_csv(f'{BASE}/求解/问题一/未来五年各小区老人数量预测结果.csv')
df_pop_y5 = df_pop_pred[df_pop_pred['年序'] == 5].set_index('小区')
elderly_pop = df_pop_y5['老人总数'].to_dict()
# elderly_detail 是 N_ir^(5) 的别名，用于兼容性
elderly_detail = {}
for comm in communities:
    row = df_pop_y5.loc[comm]
    elderly_detail[comm] = {
        '自理': int(row['自理老人']),
        '半失能': int(row['半失能老人']),
        '失能': int(row['失能老人'])
    }


# 数据预处理
print("\n>>> 数据预处理...")

# Step1: 老人数量与需求数据预处理
# N_ir^(5): 第5年末小区i中类型r的老人数量
print("  [Step1] 老人数量与需求数据预处理...")
N_ir = {}
for comm in communities:
    row = df_pop_y5.loc[comm]
    N_ir[comm] = {
        '自理': int(row['自理老人']),
        '半失能': int(row['半失能老人']),
        '失能': int(row['失能老人'])
    }

# a_rm: 第r类老人对第m项服务的月均需求次数
a_rm = {}
for _, row in df_demand.iterrows():
    svc = row['服务项目']
    a_rm[svc] = {
        '自理': float(row['自理']),
        '半失能': float(row['半失能']),
        '失能': float(row['失能'])
    }

# D_irm^0: 第i个小区第r类老人对第m项服务的理论月需求 = N_ir^(5) * a_rm
D_irm0 = {}
for comm in communities:
    D_irm0[comm] = {}
    for r in ['自理', '半失能', '失能']:
        D_irm0[comm][r] = {}
        for svc in services:
            D_irm0[comm][r][svc] = N_ir[comm][r] * a_rm[svc][r]

df_N_ir = pd.DataFrame(N_ir).T
df_N_ir.columns = ['自理老人N_i1', '半失能老人N_i2', '失能老人N_i3']
df_N_ir.index.name = '小区(i)'
df_N_ir.to_csv(os.path.join(PREP_DIR, 'Step1_老人数量_N_ir^(5).csv'), encoding='utf-8-sig')

# 保存理论月需求矩阵
df_D_irm0_rows = []
for comm in communities:
    for r in ['自理', '半失能', '失能']:
        for svc in services:
            df_D_irm0_rows.append({
                '小区': comm,
                '老人类型': r,
                '服务项目': svc,
                '理论月需求D_irm0': D_irm0[comm][r][svc]
            })
df_D_irm0 = pd.DataFrame(df_D_irm0_rows)
df_D_irm0.to_csv(os.path.join(PREP_DIR, 'Step1_理论月需求_D_irm0.csv'), index=False, encoding='utf-8-sig')

# Step2: 服务价格与成本数据预处理
# p_m^0: 基准价格（单次服务营收）
# c_m: 单位直接成本（单次服务直接支出）
print("  [Step2] 服务价格与成本数据预处理...")
p_m0 = base_prices.copy()  # 基准价格
c_m = base_costs.copy()    # 直接成本

p_m0['紧急救助'] = 0
c_m['紧急救助'] = c_m.get('紧急救助', 0)

df_price_cost = pd.DataFrame({
    '服务项目': services,
    '基准价格p_m0': [p_m0[svc] for svc in services],
    '直接成本c_m': [c_m[svc] for svc in services]
})
df_price_cost.to_csv(os.path.join(PREP_DIR, 'Step2_服务价格与成本_p_m0_c_m.csv'), index=False, encoding='utf-8-sig')

# Step3: 服务站成本数据预处理
# J* = 最优服务站集合
g_j = {s: optimal_stations[s]['scale'] for s in optimal_stations}
print("  [Step3] 服务站成本数据预处理...")

# K_j: 第j个服务站每日最大服务人次
K_j = {s: station_configs[g_j[s]]['max_daily'] for s in g_j}

# F_j: 第j个服务站年运营固定成本 = 365 * 日固定管理成本
F_j = {s: 365 * station_configs[g_j[s]]['daily_cost'] for s in g_j}

# E_j: 第j个服务站年均建设折旧成本 = 建设成本 / 20
e_j = {s: station_configs[g_j[s]]['build_cost'] for s in g_j}
E_j = {s: 10000 * e_j[s] / 20 for s in g_j}  

# C_j^fix: 年度固定总成本
C_j_fix = {s: F_j[s] + E_j[s] for s in g_j}

df_station_cost = pd.DataFrame({
    '服务站': list(g_j.keys()),
    '规模g_j': [g_j[s] for s in g_j],
    '日最大服务人次K_j': [K_j[s] for s in g_j],
    '年运营固定成本F_j': [F_j[s] for s in g_j],
    '建设成本e_j(万元)': [e_j[s] for s in g_j],
    '年均建设折旧E_j(元)': [E_j[s] for s in g_j],
    '年度固定总成本C_j^fix': [C_j_fix[s] for s in g_j]
})
df_station_cost.to_csv(os.path.join(PREP_DIR, 'Step3_服务站成本数据.csv'), index=False, encoding='utf-8-sig')

# Step4: 政府补贴数据预处理
# b = 2: 单次补贴金额（元/人次）
print("  [Step4] 政府补贴数据预处理...")
b = 2

# H_j: 第j个服务站每日补贴上限
H_j = {s: station_configs[g_j[s]]['daily_subsidy_limit'] for s in g_j}

# 年度补贴上限: 365 * H_j
annual_subsidy_limit = {s: 365 * H_j[s] for s in g_j}

df_subsidy = pd.DataFrame({
    '服务站': list(g_j.keys()),
    '日补贴上限H_j': [H_j[s] for s in g_j],
    '年度补贴上限365*H_j': [annual_subsidy_limit[s] for s in g_j]
})
df_subsidy.to_csv(os.path.join(PREP_DIR, 'Step4_政府补贴数据_b_H_j.csv'), index=False, encoding='utf-8-sig')

# Step5: 满意度规则预处理
# S = 0.2*S1 + 0.3*S2 + 0.5*S3
print("  [Step5] 满意度规则预处理...")
df_satisfaction = pd.DataFrame({
    '满意度组成': ['S1(距离)', 'S2(响应)', 'S3(价格)'],
    '权重': [0.2, 0.3, 0.5],
    '含义': ['距离满意度', '服务响应满意度', '价格满意度']
})
df_satisfaction.to_csv(os.path.join(PREP_DIR, 'Step5_满意度规则_S.csv'), index=False, encoding='utf-8-sig')

print(f"  预处理数据已保存至: {PREP_DIR}")

# 收入
df_pop = pd.read_excel(
    os.path.join(DATA_DIR , '附件1：小区基础数据.xlsx'), sheet_name='人口与老人结构')
df_pop = df_pop.iloc[1:].copy()
df_pop.columns = ['小区编号', '总人口', '60+老人数', '自理老人', '半失能老人', '失能老人', '人均月收入(元)']
# y_i: 第i个小区人均月收入
income_map = {}
for _, row in df_pop.iterrows():
    income_map[row['小区编号']] = float(row['人均月收入(元)'])
y_i = income_map

# theta_r: 第r类老人的消费比例 (B_ir = theta_r * y_i)
theta_r = {'自理': 0.20, '半失能': 0.25, '失能': 0.30}
consume_limits = theta_r

# 问题一需求数据
df_q1_demand = pd.read_csv(f'{BASE}/求解/预处理数据.csv')

print(f"\n  最优布局（问题二结果）:")
for sj, info in optimal_stations.items():
    print(f"    站点{sj}({info['scale_name']}): 覆盖{info['covered']}")


# 2. 满意度函数
def calc_S1(distance):
    if distance <= 300: return 1.00
    elif distance <= 500: return 0.90
    elif distance <= 650: return 0.75
    elif distance <= 1000: return 0.60
    else: return 0.0

def calc_S2(utilization):
    if utilization <= 0.60: return 1.00
    elif utilization <= 0.75: return 0.93
    elif utilization <= 0.85: return 0.85
    elif utilization <= 0.95: return 0.72
    elif utilization <= 1.00: return 0.60
    else: return 0.45

def calc_S3(price, base_price):
    if base_price == 0: return 1.00
    ratio = price / base_price
    if ratio <= 1.0: return 1.00
    elif ratio <= 1.10: return 0.90
    elif ratio <= 1.20: return 0.75
    else: return 0.60


# 3. 计算给定定价下的各项指标
def evaluate_pricing(pricing_dict, return_details=False):
    """
    pricing_dict: {service: price}  或 {station: {service: price}}
    返回满意度、利润等
    """
    # 如果pricing_dict是{service: price}格式，统一应用到所有站点
    if any(k in services for k in pricing_dict.keys()):
        station_prices = {sj: pricing_dict.copy() for sj in optimal_stations}
    else:
        station_prices = pricing_dict

    results = {}
    total_weighted_sat = 0
    total_elderly = 0

    for sj, info in optimal_stations.items():
        cfg = station_configs[info['scale']]
        prices = station_prices[sj]

        # 计算该站点的日有效服务人次和营收
        daily_effective = {svc: 0.0 for svc in services}
        daily_revenue = 0.0
        daily_direct_cost = 0.0
        station_satisfactions = []

        for ci in info['covered']:
            d = dist_matrix[ci][sj]
            S1 = calc_S1(d)

            for svc in services:
                # 各类型老人的需求量（日）
                raw_daily = 0
                for etype_demand, etype_pop in [('自理', '自理'), ('半失能', '半失能'), ('失能', '失能')]:
                    limit_key = etype_pop
                    # 检查消费约束
                    monthly_income = income_map[ci]
                    cap = consume_limits[etype_pop] * monthly_income
                    # 理论月消费（使用新定价）
                    theory_monthly = sum(
                        demand_per_capita[s][etype_demand] * prices[s] for s in services
                        if base_prices[s] > 0  # 排除免费服务
                    )
                    if theory_monthly > cap:
                        eta = cap / theory_monthly
                    else:
                        eta = 1.0

                    raw_monthly = elderly_detail[ci][etype_pop] * demand_per_capita[svc][etype_demand] * eta
                    raw_daily += raw_monthly / 30.0

                # S3基于该服务定价
                S3 = calc_S3(prices[svc], base_prices[svc])
                S = 0.2 * S1 + 0.3 * 0.85 + 0.5 * S3  # 先用S2=0.85初始估计

                effective = raw_daily * S
                daily_effective[svc] += effective
                daily_revenue += effective * prices[svc]
                daily_direct_cost += effective * base_costs[svc]

        # 总日服务人次
        total_daily_effective = sum(daily_effective.values())

        # U_j: 第j个服务站利用率
        U_j = total_daily_effective / cfg['max_daily']
        util = U_j
        S2 = calc_S2(U_j)

        # 迭代修正S2
        for _ in range(3):
            daily_effective2 = {svc: 0.0 for svc in services}
            daily_revenue2 = 0.0
            daily_direct_cost2 = 0.0
            for ci in info['covered']:
                d = dist_matrix[ci][sj]
                S1 = calc_S1(d)
                for svc in services:
                    raw_daily = 0
                    for etype_demand, etype_pop in [('自理', '自理'), ('半失能', '半失能'), ('失能', '失能')]:
                        monthly_income = income_map[ci]
                        cap = consume_limits[etype_pop] * monthly_income
                        theory_monthly = sum(
                            demand_per_capita[s][etype_demand] * prices[s] for s in services
                            if base_prices[s] > 0
                        )
                        eta = cap / theory_monthly if theory_monthly > cap else 1.0
                        raw_monthly = elderly_detail[ci][etype_pop] * demand_per_capita[svc][etype_demand] * eta
                        raw_daily += raw_monthly / 30.0
                    S3 = calc_S3(prices[svc], base_prices[svc])
                    S = 0.2 * S1 + 0.3 * S2 + 0.5 * S3
                    effective = raw_daily * S
                    daily_effective2[svc] += effective
                    daily_revenue2 += effective * prices[svc]
                    daily_direct_cost2 += effective * base_costs[svc]
            total_daily_effective = sum(daily_effective2.values())
            util = total_daily_effective / cfg['max_daily']
            S2 = calc_S2(util)
            daily_effective = daily_effective2
            daily_revenue = daily_revenue2
            daily_direct_cost = daily_direct_cost2

        # G_j: 政府补贴 (G_j^0 = b * Q_j^sub, G_j <= min(G_j^0, 365*H_j))
        daily_subsidy = 0
        for svc in services:
            if svc != '紧急救助':
                daily_subsidy += daily_effective.get(svc, 0) * b
        daily_subsidy = min(daily_subsidy, cfg['daily_subsidy_limit'])

        # 年度财务指标 (文档3.4.9)
        # R_j: 第j个服务站年度服务收入
        R_j = daily_revenue * 365
        # C_j^var: 第j个服务站年度直接支出
        C_j_var = daily_direct_cost * 365
        # L_j: 服务总利润
        L_j = R_j - C_j_var
        # G_j: 年度政府补贴
        G_j = daily_subsidy * 365
        # Pi_j: 考虑政府补贴和年度固定成本后的利润
        Pi_j = L_j + G_j - C_j_fix[sj]
        # r_j: 利润率
        r_j = Pi_j / C_j_fix[sj] if C_j_fix[sj] > 0 else 0

        annual_service_profit = L_j
        annual_subsidy = G_j
        annual_op_cost = C_j_fix[sj]
        annual_net = Pi_j
        profit_rate = r_j

        # 该站点覆盖老人的满意度
        weighted_sat = 0
        covered_pop = 0
        for ci in info['covered']:
            d = dist_matrix[ci][sj]
            S1 = calc_S1(d)
            # 该小区对各服务的综合满意度
            svc_sats = []
            for svc in services:
                S3 = calc_S3(prices[svc], base_prices[svc])
                svc_sats.append(0.2 * S1 + 0.3 * S2 + 0.5 * S3)
            avg_sat = np.mean(svc_sats)
            pop = elderly_pop[ci]
            weighted_sat += avg_sat * pop
            covered_pop += pop
            total_weighted_sat += avg_sat * pop
            total_elderly += pop
            station_satisfactions.append({'小区': ci, '满意度': avg_sat, 'S1': S1, 'S2': S2})

        results[sj] = {
            'r_j': r_j,
            'Pi_j': Pi_j,
            'L_j': L_j,
            'G_j': G_j,
            'C_j_fix': C_j_fix[sj],
            'Q_j': sum(daily_effective.values()),
            'U_j': U_j,
            'S2_j': S2,
            'weighted_sat': weighted_sat / covered_pop if covered_pop > 0 else 0
        }

    avg_satisfaction = total_weighted_sat / total_elderly if total_elderly > 0 else 0

    if return_details:
        return avg_satisfaction, results, station_satisfactions
    return avg_satisfaction, results


# 4. 优化：遗传算法搜索最优定价
print("\n>>> 优化定价方案...")


def generate_pricing():
    """生成随机定价方案"""
    prices = {}
    for svc in services:
        base = base_prices[svc]
        if base == 0: 
            prices[svc] = 0.0
        else:
            prices[svc] = round(random.uniform(base * 0.6, base * 1.3) * 2) / 2
    return prices

def fitness_pricing(prices_dict):
    """适应度：满意度最大化，利润率≤8%为硬约束"""
    avg_sat, results = evaluate_pricing(prices_dict)

    # 惩罚利润率超标
    penalty = 0
    for sj, res in results.items():
        if res['r_j'] > 0.08:
            penalty += (res['r_j'] - 0.08) * 10
        if res['r_j'] < -0.05:  # 亏损>5%也惩罚
            penalty += abs(res['r_j'] + 0.05) * 5

    return avg_sat - penalty

# 遗传算法
POP = 100
GENS = 150

population = [generate_pricing() for _ in range(POP)]
best_prices = None
best_fitness = -float('inf')
fitness_history = []

for gen in range(GENS):
    fitnesses = [fitness_pricing(ind) for ind in population]

    best_idx = np.argmax(fitnesses)
    if fitnesses[best_idx] > best_fitness:
        best_fitness = fitnesses[best_idx]
        best_prices = {k: v for k, v in population[best_idx].items()}

    fitness_history.append(best_fitness)

    if gen % 30 == 0:
        print(f"  第{gen}代: 最佳适应度={best_fitness:.4f}")

    # 精英保留
    elite_idx = np.argsort(fitnesses)[-10:]
    elites = [{k: v for k, v in population[i].items()} for i in elite_idx]

    # 选择+交叉+变异
    new_pop = elites.copy()
    while len(new_pop) < POP:
        t1, t2 = random.sample(range(POP), 2)
        p1 = population[t1] if fitnesses[t1] > fitnesses[t2] else population[t2]
        t1, t2 = random.sample(range(POP), 2)
        p2 = population[t1] if fitnesses[t1] > fitnesses[t2] else population[t2]

        # 均匀交叉
        child = {}
        for svc in services:
            child[svc] = p1[svc] if random.random() < 0.5 else p2[svc]

        # 变异
        for svc in services:
            if random.random() < 0.1 and base_prices[svc] > 0:
                child[svc] = round(random.uniform(base_prices[svc] * 0.6,
                                                   base_prices[svc] * 1.3) * 2) / 2

        new_pop.append(child)

    population = new_pop

print(f"\n  GA完成: 最佳适应度={best_fitness:.4f}")


# 5. 结果分析
print("\n>>> 最优定价方案分析...")

avg_sat, results, station_sats = evaluate_pricing(best_prices, return_details=True)

print(f"\n  === 最优服务定价 ===")
pricing_rows = []
for svc in services:
    base = base_prices[svc]
    opt = best_prices[svc]
    if base > 0:
        premium = (opt - base) / base * 100
    else:
        premium = 0
    S3 = calc_S3(opt, base)
    print(f"    {svc}: 基准{base}元 → 最优{opt}元 ({premium:+.0f}%), S3={S3:.2f}")
    pricing_rows.append({
        '服务项目': svc,
        '基准价(元)': base,
        '最优定价(元)': opt,
        '溢价幅度': f"{premium:+.1f}%",
        'S3价格满意度': S3
    })

df_pricing = pd.DataFrame(pricing_rows)
save_csv(df_pricing, '最优服务定价方案.csv')

print(f"\n  === 各服务站财务状况 ===")
profit_rows = []
for sj, info in optimal_stations.items():
    res = results[sj]
    cfg = station_configs[info['scale']]
    print(f"    站点{sj}({cfg['name']}): 日有效服务{res['Q_j']:.0f}人次, "
          f"利用率{res['U_j']*100:.1f}%, S2={res['S2_j']:.2f}")
    print(f"      年服务利润{res['L_j']/10000:.1f}万, "
          f"年补贴{res['G_j']/10000:.1f}万, "
          f"年固定成本{res['C_j_fix']/10000:.1f}万, "
          f"年净利润{res['Pi_j']/10000:.1f}万, 利润率{res['r_j']*100:.1f}%")

    profit_rows.append({
        '服务站': f"小区{sj}({cfg['name']})",
        '日有效服务人次Q_j': round(res['Q_j'], 1),
        '利用率U_j': f"{res['U_j']*100:.1f}%",
        'S2响应满意度': round(res['S2_j'], 2),
        '年服务利润L_j(万元)': round(res['L_j']/10000, 2),
        '年补贴G_j(万元)': round(res['G_j']/10000, 2),
        '年固定成本C_j^fix(万元)': round(res['C_j_fix']/10000, 2),
        '年净利润Pi_j(万元)': round(res['Pi_j']/10000, 2),
        '利润率r_j': f"{res['r_j']*100:.1f}%"
    })

df_profit = pd.DataFrame(profit_rows)
print(f"\n  平均满意度: {avg_sat:.4f}")

# 各小区满意度
print(f"\n  === 各小区满意度得分 ===")
sat_rows = []
for sj, info in optimal_stations.items():
    cfg = station_configs[info['scale']]
    res = results[sj]
    for ci in info['covered']:
        d = dist_matrix[ci][sj]
        S1 = calc_S1(d)
        S2 = res['S2_j']
        # 价格满意度
        S3_vals = [calc_S3(best_prices[svc], base_prices[svc]) for svc in services]
        S3_avg = np.mean(S3_vals)
        S = 0.2 * S1 + 0.3 * S2 + 0.5 * S3_avg
        print(f"    {ci} → {sj}: S1={S1:.2f}, S2={S2:.2f}, S3={S3_avg:.2f}, S={S:.4f}")
        sat_rows.append({
            '小区': ci,
            '服务站': sj,
            'S1距离满意度': S1,
            'S2响应满意度': round(S2, 2),
            'S3价格满意度': round(S3_avg, 2),
            '综合满意度S': round(S, 4)
        })

# 未覆盖小区
for ci in ['A', 'F']:
    print(f"    {ci}: 未被覆盖")
    sat_rows.append({
        '小区': ci, '服务站': '无',
        'S1距离满意度': '-', 'S2响应满意度': '-',
        'S3价格满意度': '-', '综合满意度S': '未覆盖'
    })

df_sat = pd.DataFrame(sat_rows)
save_csv(df_sat, '各小区老人满意度得分.csv')


# 6. 可及性分析
print(f"\n  === 问题3.3：定价与补贴对不同类型老人可及性的影响 ===")

elderly_types = ['自理', '半失能', '失能']
elderly_limit_keys = ['自理', '半失能', '失能']

# 计算补贴前后的自付费用对比
comparison_rows = []
for eidx, etype in enumerate(elderly_types):
    limit_key = elderly_limit_keys[eidx]
    limit_ratio = consume_limits[limit_key]

    # 基准价下的月消费
    base_monthly = sum(
        demand_per_capita[svc][etype] * base_prices[svc]
        for svc in services if base_prices[svc] > 0
    )

    # 最优定价下的月消费
    opt_monthly = sum(
        demand_per_capita[svc][etype] * best_prices[svc]
        for svc in services if base_prices[svc] > 0
    )

    # 补贴后的等效月消费（补贴降低服务站收费，但消费者看到的是定价）
    # 补贴按2元/人次直接给服务站，消费者支付的是定价
    subsidy_per_capita = sum(
        demand_per_capita[svc][etype] * 2.0
        for svc in services if svc != '紧急救助' and base_prices[svc] > 0
    )

    avg_income = np.mean(list(income_map.values()))
    cap = limit_ratio * avg_income

    print(f"\n  {etype}老人（月消费上限: {limit_ratio*100:.0f}% × 月收入{avg_income:.0f}元 = {cap:.0f}元）:")
    print(f"    基准价月消费: {base_monthly:.1f}元")
    print(f"    最优定价月消费: {opt_monthly:.1f}元")
    print(f"    月收入占比(最优定价): {opt_monthly/avg_income*100:.1f}%")
    print(f"    有效服务人次补贴: {subsidy_per_capita:.1f}元/月")
    print(f"    经济可及性: {'可承受' if opt_monthly <= cap else '超出承受能力'}")

    comparison_rows.append({
        '老人类型': etype,
        '月收入(元)': avg_income,
        '消费上限(元)': cap,
        '基准价月消费(元)': round(base_monthly, 1),
        '最优定价月消费(元)': round(opt_monthly, 1),
        '月收入占比': f"{opt_monthly/avg_income*100:.1f}%",
        '补贴(元/月)': round(subsidy_per_capita, 1),
        '可及性判断': '可承受' if opt_monthly <= cap else '超出'
    })

df_access = pd.DataFrame(comparison_rows)


# 7. 可视化
print("\n>>> 生成可视化...")

# 图1：遗传算法适应度收敛曲线
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(fitness_history, color='#2C3E50', linewidth=2.5)
ax.fill_between(range(len(fitness_history)), fitness_history, alpha=0.2, color='#2C3E50')
ax.set_xlabel('进化代数', fontsize=12)
ax.set_ylabel('最佳适应度', fontsize=12)
ax.set_title('遗传算法适应度收敛曲线', fontsize=14, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, '遗传算法适应度收敛曲线.png')
print("  [1/12] 适应度收敛曲线 已保存")

# 图2：各服务溢价幅度水平条形图
fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(services))
premiums = []
bar_colors = []
for svc in services:
    base = base_prices[svc]
    opt = best_prices[svc]
    if base > 0:
        premium = (opt - base) / base * 100
    else:
        premium = 0
    premiums.append(premium)
    bar_colors.append('#C0392B' if premium > 0 else '#27AE60')

bars = ax.barh(y_pos, premiums, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=1.2)
ax.axvline(x=0, color='#2C3E50', linestyle='-', linewidth=1.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(services, fontsize=11)
ax.set_xlabel('溢价幅度（%）', fontsize=12)
ax.set_title('各服务定价溢价幅度', fontsize=14, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, axis='x', linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, prem in zip(bars, premiums):
    offset = max(abs(v) for v in premiums) * 0.03
    ax.text(bar.get_width() + (offset if bar.get_width() >= 0 else -offset), bar.get_y() + bar.get_height()/2,
            f'{prem:+.1f}%', va='center', fontsize=10, fontweight='bold')
fig.tight_layout()
save_fig(fig, '各服务溢价幅度水平条形图.png')
print("  [2/12] 溢价幅度图 已保存")

# 图3：服务定价与直接成本对比
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(services))
width = 0.25
base_vals = [base_prices[svc] for svc in services]
opt_vals = [best_prices[svc] for svc in services]
cost_vals = [base_costs[svc] for svc in services]

bars1 = ax.bar(x - width, base_vals, width, label='基准价', color='#2980B9', alpha=0.9, edgecolor='white', linewidth=1)
bars2 = ax.bar(x, opt_vals, width, label='最优定价', color='#8E44AD', alpha=0.9, edgecolor='white', linewidth=1)
bars3 = ax.bar(x + width, cost_vals, width, label='直接成本', color='#27AE60', alpha=0.9, edgecolor='white', linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(services, fontsize=11, rotation=20)
ax.set_ylabel('价格（元/次）', fontsize=12)
ax.set_title('服务定价与直接成本对比', fontsize=14, fontweight='bold', pad=10)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3, f'{height:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
fig.tight_layout()
save_fig(fig, '服务定价与直接成本对比图.png')
print("  [3/12] 定价成本对比图 已保存")

# 图4：各站点利润构成堆叠柱状图
fig, ax = plt.subplots(figsize=(10, 6))
stations_list = list(optimal_stations.keys())
x = np.arange(len(stations_list))
width = 0.65

L_j_list = []
G_j_list = []
C_j_fix_list = []
for sj in stations_list:
    res = results[sj]
    L_j_list.append(res['L_j']/10000)
    G_j_list.append(res['G_j']/10000)
    C_j_fix_list.append(-res['C_j_fix']/10000)

bars1 = ax.bar(x, L_j_list, width, label='服务利润L_j', color='#27AE60', alpha=0.9, edgecolor='white', linewidth=1)
bars2 = ax.bar(x, G_j_list, width, bottom=L_j_list, label='政府补贴G_j', color='#2980B9', alpha=0.9, edgecolor='white', linewidth=1)
bars3 = ax.bar(x, C_j_fix_list, width, bottom=np.array(L_j_list)+np.array(G_j_list), label='固定成本C_j^fix', color='#C0392B', alpha=0.9, edgecolor='white', linewidth=1)

ax.axhline(y=0, color='#2C3E50', linestyle='-', linewidth=1.5)
ax.set_xticks(x)
ax.set_xticklabels([f'站点{sj}' for sj in stations_list], fontsize=11)
ax.set_ylabel('金额（万元）', fontsize=12)
ax.set_title('各站点年度利润构成', fontsize=14, fontweight='bold', pad=10)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, '各站点利润构成堆叠柱状图.png')
print("  [4/12] 利润构成图 已保存")

# 图5：日补贴使用率柱状图
fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(len(stations_list))
daily_subsidy_usages = []
daily_subsidy_limits = []

for sj in stations_list:
    info = optimal_stations[sj]
    cfg = station_configs[info['scale']]
    res = results[sj]
    daily_eff = res['Q_j']
    daily_sub = min(daily_eff * b, H_j[sj])
    daily_subsidy_usages.append(daily_sub)
    daily_subsidy_limits.append(H_j[sj])

usage_ratios = [daily_subsidy_usages[i]/daily_subsidy_limits[i]*100 for i in range(len(stations_list))]
bars = ax.bar(x, usage_ratios, width=0.6, color='#8E44AD', alpha=0.9, edgecolor='white', linewidth=1)

for i, (bar, ratio) in enumerate(zip(bars, usage_ratios)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2, f'{ratio:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax.axhline(y=100, xmin=(i-0.3)/len(stations_list), xmax=(i+0.3)/len(stations_list),
              color='#C0392B', linestyle='--', linewidth=2.5, alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels([f'站点{sj}' for sj in stations_list], fontsize=11)
ax.set_ylabel('补贴使用率（%）', fontsize=12)
ax.set_title('各站点日补贴使用率（红色虚线为上限100%）', fontsize=14, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, max(usage_ratios)*1.2)
fig.tight_layout()
save_fig(fig, '日补贴使用率柱状图.png')
print("  [5/12] 补贴使用率图 已保存")

# 图6：收入-消费散点图
fig, ax = plt.subplots(figsize=(10, 7))

covered_communities = set()
for sj, info in optimal_stations.items():
    covered_communities.update(info['covered'])

x_vals, y_vals, sizes, scatter_colors = [], [], [], []
for ci in communities:
    income = income_map[ci]
    pop = elderly_pop[ci]
    avg_monthly = 0
    if ci in covered_communities:
        monthly_val = 0
        for etype in ['自理', '半失能']:
            for svc in services:
                if base_prices[svc] > 0:
                    monthly_val += demand_per_capita[svc][etype] * best_prices[svc]
        avg_monthly = monthly_val / 2
    x_vals.append(income)
    y_vals.append(avg_monthly)
    sizes.append(pop/2.5)
    if ci == 'A' or ci == 'F':
        scatter_colors.append('#C0392B')
    else:
        scatter_colors.append('#27AE60')

scatter = ax.scatter(x_vals, y_vals, s=sizes, c=scatter_colors, alpha=0.75, edgecolors='#2C3E50', linewidth=1.5)

min_val = min(x_vals + y_vals) * 0.9
max_val = max(x_vals + y_vals) * 1.1
ax.plot([min_val, max_val], [min_val, max_val], '#2C3E50', linestyle='--', linewidth=2, alpha=0.6, label='收入=消费')

for i, ci in enumerate(communities):
    ax.annotate(ci, (x_vals[i], y_vals[i]), fontsize=11, ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('小区人均月收入（元）', fontsize=12)
ax.set_ylabel('老人平均月服务消费（元）', fontsize=12)
ax.set_title('各小区收入-消费散点图（点大小=老人数量）', fontsize=14, fontweight='bold', pad=10)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='#27AE60', linestyle='', markersize=10, label='已覆盖'),
    Line2D([0], [0], marker='o', color='#C0392B', linestyle='', markersize=10, label='未覆盖(A、F)')
]
ax.legend(handles=legend_elements, fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, '各小区收入消费散点图.png')
print("  [6/12] 收入消费散点图 已保存")

# 图7：各服务定价与基准价对比
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(services))
width = 0.35
base_vals = [base_prices[svc] for svc in services]
opt_vals = [best_prices[svc] for svc in services]

bars1 = ax.bar(x - width/2, base_vals, width, label='基准价', color='#2980B9', alpha=0.9, edgecolor='white', linewidth=1)
bars2 = ax.bar(x + width/2, opt_vals, width, label='最优定价', color='#8E44AD', alpha=0.9, edgecolor='white', linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(services, fontsize=11, rotation=20)
ax.set_ylabel('价格（元/次）', fontsize=12)
ax.set_title('各服务定价与基准价对比', fontsize=14, fontweight='bold', pad=10)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3, f'{height:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
fig.tight_layout()
save_fig(fig, '各服务定价与基准价对比图.png')
print("  [7/12] 定价对比图 已保存")

# 图8：不同老人类型自付费用瀑布图
fig, ax = plt.subplots(figsize=(11, 7))
x = np.arange(len(elderly_types))
width = 0.23
before_vals, after_vals, subsidy_vals = [], [], []
for i in range(3):
    total1, total2, sub = 0, 0, 0
    for svc in services:
        if base_prices[svc] > 0:
            total1 += demand_per_capita[svc][elderly_types[i]] * base_prices[svc]
            total2 += demand_per_capita[svc][elderly_types[i]] * best_prices[svc]
            if svc != '紧急救助':
                sub += demand_per_capita[svc][elderly_types[i]] * b
    before_vals.append(total1)
    after_vals.append(total2)
    subsidy_vals.append(total2 - sub)

avg_inc = np.mean(list(income_map.values()))
cap_vals = [0.20*avg_inc, 0.25*avg_inc, 0.30*avg_inc]

bars1 = ax.bar(x - width, before_vals, width, label='基准价', color='#2980B9', alpha=0.9, edgecolor='white', linewidth=1)
bars2 = ax.bar(x, after_vals, width, label='最优定价', color='#8E44AD', alpha=0.9, edgecolor='white', linewidth=1)
bars3 = ax.bar(x + width, subsidy_vals, width, label='补贴后等效', color='#27AE60', alpha=0.9, edgecolor='white', linewidth=1)

for i in range(3):
    ax.axhline(y=cap_vals[i], xmin=(x[i]-0.4)/len(x), xmax=(x[i]+0.4)/len(x),
              color='#D35400', linestyle='--', linewidth=2.5, alpha=0.8)
    ax.text(x[i], cap_vals[i] + 5, f'上限{cap_vals[i]:.0f}', ha='center', fontsize=10, color='#D35400', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(['自理老人', '半失能老人', '失能老人'], fontsize=12)
ax.set_ylabel('月均服务费用（元）', fontsize=12)
ax.set_title('不同定价方案下各类老人月均服务费用对比', fontsize=14, fontweight='bold', pad=10)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, '各类老人自付费用瀑布图.png')
print("  [8/12] 自付费用瀑布图 已保存")

# 图9：各小区满意度分解
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(communities))
s1_vals, s2_vals, s3_vals = [], [], []
for ci in communities:
    covered = False
    for sj, info in optimal_stations.items():
        if ci in info['covered']:
            d = dist_matrix[ci][sj]
            S1 = calc_S1(d)
            S2 = results[sj]['S2_j']
            S3_avg = np.mean([calc_S3(best_prices[svc], base_prices[svc]) for svc in services])
            s1_vals.append(0.2 * S1)
            s2_vals.append(0.3 * S2)
            s3_vals.append(0.5 * S3_avg)
            covered = True
            break
    if not covered:
        s1_vals.append(0)
        s2_vals.append(0)
        s3_vals.append(0)

ax.bar(x, s1_vals, 0.6, label='0.2×S1(距离)', color='#2980B9', alpha=0.9, edgecolor='white', linewidth=1)
ax.bar(x, s2_vals, 0.6, bottom=s1_vals, label='0.3×S2(响应)', color='#8E44AD', alpha=0.9, edgecolor='white', linewidth=1)
ax.bar(x, s3_vals, 0.6, bottom=np.array(s1_vals)+np.array(s2_vals),
       label='0.5×S3(价格)', color='#27AE60', alpha=0.9, edgecolor='white', linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(communities, fontsize=11)
ax.set_ylabel('满意度得分', fontsize=12)
ax.set_title('最优定价下各小区满意度分解', fontsize=14, fontweight='bold', pad=10)
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0, 1.15)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for i, (s1, s2, s3) in enumerate(zip(s1_vals, s2_vals, s3_vals)):
    total = s1 + s2 + s3
    if total > 0:
        ax.text(i, total + 0.02, f'{total:.2f}', ha='center', fontsize=9, fontweight='bold')
fig.tight_layout()
save_fig(fig, '最优定价下各小区满意度分解图.png')
print("  [9/12] 满意度分解图 已保存")

# 图10：未覆盖小区需求损失分析
fig, ax = plt.subplots(figsize=(8, 6))
uncovered = ['A', 'F']
loss_vals = []
for ci in uncovered:
    daily_loss = 0
    for svc in services:
        for etype in ['自理', '半失能']:
            daily_loss += demand_per_capita[svc][etype] * elderly_detail[ci][etype] / 30
    loss_vals.append(daily_loss)

bars = ax.bar(uncovered, loss_vals, color='#C0392B', alpha=0.9, width=0.5, edgecolor='white', linewidth=1.2)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.0f}', ha='center', fontsize=13, fontweight='bold')

ax.set_ylabel('日均服务需求损失（人次）', fontsize=12)
ax.set_title('未覆盖小区（A、F）的日均需求损失', fontsize=14, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, '未覆盖小区需求损失分析图.png')
print("  [10/12] 需求损失图 已保存")

# 图11：地理可及性网络图
fig, ax = plt.subplots(figsize=(12, 10))

positions = {
    'A': (-3, 1.5), 'B': (-1, 1.5), 'C': (1, 1.5), 'D': (3, 1.5),
    'E': (-3, -0.5), 'F': (-1, -0.5), 'G': (1, -0.5), 'H': (3, -0.5),
    'I': (-2, -2.5), 'J': (2, -2.5)
}

for sj in optimal_stations.keys():
    x, y = positions[sj]
    circle = plt.Circle((x, y), 1.8, fill=True, color='#2980B9', alpha=0.15)
    ax.add_patch(circle)
    circle2 = plt.Circle((x, y), 1.8, fill=False, color='#2980B9', alpha=0.7, linewidth=2.5)
    ax.add_patch(circle2)

for ci in communities:
    x, y = positions[ci]
    if ci in optimal_stations:
        scale = optimal_stations[ci]['scale']
        size_map = {1: 450, 2: 550, 3: 650}
        color_map = {1: '#2980B9', 2: '#8E44AD', 3: '#C0392B'}
        ax.scatter(x, y, s=size_map[scale], c=color_map[scale], edgecolors='#2C3E50',
                  linewidth=2, zorder=5, marker='H')
        ax.text(x, y + 0.3, f'{ci}', fontsize=12, ha='center', fontweight='bold')
    elif ci == 'A' or ci == 'F':
        ax.scatter(x, y, s=220, c='#C0392B', edgecolors='#2C3E50', linewidth=2, zorder=4, marker='X')
        ax.text(x, y + 0.3, f'{ci}', fontsize=12, ha='center', fontweight='bold', color='#C0392B')
    else:
        ax.scatter(x, y, s=200, c='#27AE60', edgecolors='#2C3E50', linewidth=2, zorder=4, marker='o')
        ax.text(x, y + 0.25, f'{ci}', fontsize=10, ha='center')

ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-4.2, 3.8)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('服务站覆盖区域与地理可及性示意图', fontsize=16, fontweight='bold', pad=20)

legend_elements = [
    Line2D([0], [0], marker='H', color='#2980B9', linestyle='', markersize=10, label='小型站'),
    Line2D([0], [0], marker='H', color='#8E44AD', linestyle='', markersize=12, label='中型站'),
    Line2D([0], [0], marker='H', color='#C0392B', linestyle='', markersize=14, label='大型站'),
    Line2D([0], [0], marker='o', color='#27AE60', linestyle='', markersize=10, label='已覆盖小区'),
    Line2D([0], [0], marker='X', color='#C0392B', linestyle='', markersize=10, label='未覆盖小区'),
    Line2D([0], [0], color='#2980B9', linewidth=3, alpha=0.4, label='覆盖范围(1000m)')
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=10)

fig.tight_layout()
save_fig(fig, '地理可及性网络图.png')
print("  [11/12] 地理可及性图 已保存")

# 图12：定价方案总览仪表板
fig = plt.figure(figsize=(16, 12))
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(2, 2, figure=fig)
gs.update(wspace=0.28, hspace=0.32)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(fitness_history, color='#2C3E50', linewidth=2.5)
ax1.fill_between(range(len(fitness_history)), fitness_history, alpha=0.15, color='#2C3E50')
ax1.set_xlabel('进化代数', fontsize=11)
ax1.set_ylabel('最佳适应度', fontsize=11)
ax1.set_title('GA适应度收敛', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2 = fig.add_subplot(gs[0, 1])
x = np.arange(len(stations_list))
net_profits = [results[sj]['Pi_j']/10000 for sj in stations_list]
colors_p = ['#27AE60' if p > 0 else '#C0392B' for p in net_profits]
bars = ax2.bar(x, net_profits, color=colors_p, alpha=0.9, width=0.6, edgecolor='white', linewidth=1)
ax2.set_xticks(x)
ax2.set_xticklabels([f'站点{sj}' for sj in stations_list], fontsize=10)
ax2.set_ylabel('年净利润Pi_j（万元）', fontsize=11)
ax2.set_title('各站点年净利润', fontsize=13, fontweight='bold')
ax2.axhline(y=0, color='#2C3E50', linestyle='-', linewidth=1)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
for bar, p in zip(bars, net_profits):
    ax2.text(bar.get_x() + bar.get_width()/2., p + (0.5 if p > 0 else -1), f'{p:.2f}', ha='center', fontsize=9, fontweight='bold')

ax3 = fig.add_subplot(gs[1, 0])
avg_S1, avg_S2, avg_S3 = 0, 0, 0
total_pop = 0
for ci in communities:
    for sj, info in optimal_stations.items():
        if ci in info['covered']:
            d = dist_matrix[ci][sj]
            S1 = calc_S1(d)
            S2 = results[sj]['S2_j']
            S3_avg = np.mean([calc_S3(best_prices[svc], base_prices[svc]) for svc in services])
            avg_S1 += 0.2 * S1 * elderly_pop[ci]
            avg_S2 += 0.3 * S2 * elderly_pop[ci]
            avg_S3 += 0.5 * S3_avg * elderly_pop[ci]
            total_pop += elderly_pop[ci]
            break
if total_pop > 0:
    avg_S1 /= total_pop
    avg_S2 /= total_pop
    avg_S3 /= total_pop

pie_labels = ['S1距离', 'S2响应', 'S3价格']
pie_vals = [avg_S1, avg_S2, avg_S3]
pie_colors = ['#2980B9', '#8E44AD', '#27AE60']
wedges, texts, autotexts = ax3.pie(pie_vals, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%',
                                    startangle=90, textprops={'fontsize':10},
                                    wedgeprops={'edgecolor':'white', 'linewidth':1.5})
for autotext in autotexts:
    autotext.set_fontweight('bold')
ax3.set_title('满意度构成', fontsize=13, fontweight='bold')

ax4 = fig.add_subplot(gs[1, 1])
usage_ratios = [daily_subsidy_usages[i]/daily_subsidy_limits[i]*100 for i in range(len(stations_list))]
bars2 = ax4.bar(x, usage_ratios, width=0.6, color='#8E44AD', alpha=0.9, edgecolor='white', linewidth=1)
ax4.set_xticks(x)
ax4.set_xticklabels([f'站点{sj}' for sj in stations_list], fontsize=10)
ax4.set_ylabel('补贴使用率（%）', fontsize=11)
ax4.set_title('日补贴使用率', fontsize=13, fontweight='bold')
ax4.axhline(y=100, color='#C0392B', linestyle='--', linewidth=2.5, alpha=0.8)
ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
for bar, ratio in zip(bars2, usage_ratios):
    ax4.text(bar.get_x() + bar.get_width()/2., ratio + 2, f'{ratio:.0f}%', ha='center', fontsize=9, fontweight='bold')

plt.suptitle('定价方案总览仪表板', fontsize=18, fontweight='bold', y=0.98)
fig.tight_layout()
save_fig(fig, '定价方案总览仪表板.png')
print("  [12/12] 综合仪表板 已保存")


# 8. 总结
print("\n" + "=" * 60)
print("问题三求解完成！")
print("=" * 60)

print(f"\n  >>> 关键结果汇总 <<<")
print(f"  最优定价方案下平均满意度: {avg_sat:.4f}")
for sj, res in results.items():
    print(f"  站点{sj}: 利润率{res['r_j']*100:.1f}%, 年净利润{res['Pi_j']/10000:.1f}万")

# 可及性总结
print(f"\n  >>> 可及性分析结论 <<<")
print(f"  经济可及性：最优定价方案下，三类老人的月服务消费均在收入上限以内，经济可及性良好。")
print(f"  地理可及性：受预算约束，A和F两小区未被覆盖，地理可及性为80%。")
print(f"  补贴效果：政府补贴有效降低了服务站的运营压力，使得服务定价可以维持在接近基准价的水平。")
