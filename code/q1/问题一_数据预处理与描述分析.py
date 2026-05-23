import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
import seaborn as sns


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

COLORS_10 = ['#4A90D9', '#F5A623', '#7ED321', '#D0021B', '#9013FE',
             '#BD3E3E', '#FF69B4', '#50E3C2', '#F8E71C', '#2EC5C0']


BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, '数据')
OUT_DIR = os.path.join(BASE, '求解', '问题一')
PAPER_DIR = os.path.join(BASE, '论文', 'figures', '问题一')
os.makedirs(os.path.join(OUT_DIR, 'figures'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, '数据预处理'), exist_ok=True)
os.makedirs(PAPER_DIR, exist_ok=True)


def save_fig(fig, name_cn):
    fig.savefig(os.path.join(OUT_DIR, 'figures', name_cn))
    fig.savefig(os.path.join(PAPER_DIR, name_cn))
    plt.close(fig)

def save_csv(df, name_cn):
    df.to_csv(os.path.join(OUT_DIR, name_cn), index=False, encoding='utf-8-sig')
    # 也保存到求解根目录供后续问题使用
    if '预处理数据' in name_cn:
        df.to_csv(f'{BASE}/求解/{name_cn}', index=False, encoding='utf-8-sig')

print("=" * 60)
print("问题一：未来五年老人数量与服务需求量预测")
print("=" * 60)


# 1. 数据加载
print("\n>>> 加载数据...")

# 附件1：人口与老人结构
df_pop_raw = pd.read_excel(
    os.path.join(DATA_DIR , '附件1：小区基础数据.xlsx'), sheet_name='人口与老人结构')
df_pop = df_pop_raw.iloc[1:].copy()  # 跳过标题行
df_pop.columns = ['小区编号', '总人口', '60+老人数', '自理老人', '半失能老人', '失能老人', '人均月收入(元)']
df_pop = df_pop.reset_index(drop=True)
for col in ['总人口', '60+老人数', '自理老人', '半失能老人', '失能老人', '人均月收入(元)']:
    df_pop[col] = pd.to_numeric(df_pop[col], errors='coerce')
df_pop = df_pop.dropna(subset=['小区编号'])
print(f"  10个小区人口数据加载完成，共{len(df_pop)}条记录")

# 转移概率
df_trans_raw = pd.read_excel(
    os.path.join(DATA_DIR , '附件1：小区基础数据.xlsx'), sheet_name='转移概率')
p_ZB = float(df_trans_raw.iloc[1, 1])  # 自理→半失能
p_BS = float(df_trans_raw.iloc[2, 1])  # 半失能→失能
print(f"  转移概率: 自理→半失能={p_ZB}, 半失能→失能={p_BS}")

# 附件2：服务需求数据
df_demand = pd.read_excel(
    os.path.join(DATA_DIR , '附件2：服务需求数据.xlsx'), sheet_name='每位老人月均服务需求次数')
df_demand = df_demand.iloc[1:].copy()
df_demand.columns = ['服务项目', '自理', '半自理', '失能']
df_demand = df_demand.reset_index(drop=True)
for col in ['自理', '半自理', '失能']:
    df_demand[col] = pd.to_numeric(df_demand[col], errors='coerce')

df_revenue = pd.read_excel(
    os.path.join(DATA_DIR , '附件2：服务需求数据.xlsx'), sheet_name='服务营收及支出')
df_revenue = df_revenue.iloc[1:].copy()
df_revenue.columns = ['服务项目', '单次服务营收', '单次服务直接支出']
df_revenue = df_revenue.reset_index(drop=True)
for col in ['单次服务营收', '单次服务直接支出']:
    df_revenue[col] = pd.to_numeric(df_revenue[col], errors='coerce')

df_consume = pd.read_excel(
    os.path.join(DATA_DIR , '附件2：服务需求数据.xlsx'), sheet_name='月服务消费上限')
# 消费上限比例
consume_limit = {
    '自理': 0.20,
    '半失能': 0.25,
    '失能': 0.30
}

print(f"  服务需求数据加载完成，共{len(df_demand)}类服务")


# 2. 问题1.1：未来五年老人数量递推预测
print("\n>>> 问题1.1：未来五年老人数量递推预测...")

# 模型参数
d = 0.05      # d: 年均自然死亡率
g = 0.07      # g: 年均新增老人比例

# 转移概率
# p_i12: 自理老人转半失能概率
# p_i23: 半失能老人转失能概率
p_i12 = p_ZB
p_i23 = p_BS

# 初始化：第0年末（当前）
# N_i1: 自理老人数量
# N_i2: 半失能老人数量  
# N_i3: 失能老人数量
communities = df_pop['小区编号'].values
init_data = {}
for _, row in df_pop.iterrows():
    comm = row['小区编号']
    init_data[comm] = {
        'N_i1': int(row['自理老人']),   # N_i1: 自理老人
        'N_i2': int(row['半失能老人']),  # N_i2: 半失能老人
        'N_i3': int(row['失能老人'])     # N_i3: 失能老人
    }

# 递推预测
years = list(range(0, 6))  # 0到5年
pred_results = {comm: {'N_i1': [init_data[comm]['N_i1']],
                        'N_i2': [init_data[comm]['N_i2']],
                        'N_i3': [init_data[comm]['N_i3']]} for comm in communities}

for t in range(1, 6):
    for comm in communities:
        # 获取第t年末各类老人数量
        N_i1_t = pred_results[comm]['N_i1'][-1]
        N_i2_t = pred_results[comm]['N_i2'][-1]
        N_i3_t = pred_results[comm]['N_i3'][-1]

        # Step1: 计算每年各小区老人总数
        T_i_t = N_i1_t + N_i2_t + N_i3_t  # T_i^t = N_i1^t + N_i2^t + N_i3^t

        # Step2: 计算新增老人数量
        A_i_t = g * T_i_t  # A_i^t = g * T_i^t

        # Step3: 自理老人数量递推
        # N_i1^{t+1} = (1-d)(1-p_i12)N_i1^t + A_i^t
        N_i1_t1 = (1 - d) * (1 - p_i12) * N_i1_t + A_i_t

        # Step4: 半失能老人数量递推
        # N_i2^{t+1} = (1-d)[p_i12*N_i1^t + (1-p_i23)*N_i2^t]
        N_i2_t1 = (1 - d) * (p_i12 * N_i1_t + (1 - p_i23) * N_i2_t)

        # Step5: 失能老人数量递推
        # N_i3^{t+1} = (1-d)[p_i23*N_i2^t + N_i3^t]
        N_i3_t1 = (1 - d) * (p_i23 * N_i2_t + N_i3_t)

        pred_results[comm]['N_i1'].append(int(round(N_i1_t1)))
        pred_results[comm]['N_i2'].append(int(round(N_i2_t1)))
        pred_results[comm]['N_i3'].append(int(round(N_i3_t1)))

# 整理为DataFrame
pred_rows = []
for comm in communities:
    for t in years:
        pred_rows.append({
            '小区': comm,
            '年份': f'第{t}年末' if t > 0 else '当前',
            '年序': t,
            '自理老人': pred_results[comm]['N_i1'][t],
            '半失能老人': pred_results[comm]['N_i2'][t],
            '失能老人': pred_results[comm]['N_i3'][t],
            '老人总数': pred_results[comm]['N_i1'][t] + pred_results[comm]['N_i2'][t] + pred_results[comm]['N_i3'][t]
        })

df_pred = pd.DataFrame(pred_rows)
save_csv(df_pred, '未来五年各小区老人数量预测结果.csv')

print("\n  === 第5年末各小区老人数量 ===")
df_y5 = df_pred[df_pred['年序'] == 5][['小区', '自理老人', '半失能老人', '失能老人', '老人总数']]
print(df_y5.to_string(index=False))


# 3. 问题1.2：理论月服务需求预测（第5年末）
print("\n>>> 问题1.2：第5年末理论月服务需求预测...")

services = df_demand['服务项目'].values
demand_matrix = {}  # demand_matrix[service][category]
for _, row in df_demand.iterrows():
    svc = row['服务项目']
    demand_matrix[svc] = {
        '自理': row['自理'],
        '半自理': row['半自理'],
        '失能': row['失能']
    }

theory_rows = []
for comm in communities:
    N_i1_5 = pred_results[comm]['N_i1'][5]
    N_i2_5 = pred_results[comm]['N_i2'][5]
    N_i3_5 = pred_results[comm]['N_i3'][5]

    for svc in services:
        demand_dict = demand_matrix[svc]
        total_demand = N_i1_5 * demand_dict['自理'] + N_i2_5 * demand_dict['半自理'] + N_i3_5 * demand_dict['失能']
        theory_rows.append({
            '小区': comm,
            '服务项目': svc,
            '理论月需求次数': round(total_demand, 1),
            '自理需求': N_i1_5 * demand_dict['自理'],
            '半失能需求': N_i2_5 * demand_dict['半自理'],
            '失能需求': N_i3_5 * demand_dict['失能']
        })

df_theory = pd.DataFrame(theory_rows)
save_csv(df_theory, '第5年末理论月服务需求次数.csv')

# 汇总表
df_theory_summary = df_theory.pivot_table(
    values='理论月需求次数', index='小区', columns='服务项目', aggfunc='sum')
df_theory_summary = df_theory_summary[services]  # 按原始顺序排列
print("\n  === 第5年末各小区理论月服务需求次数 ===")
print(df_theory_summary.round(1).to_string())


# 4. 问题1.3：消费约束下的实际月服务需求
print("\n>>> 问题1.3：消费约束下实际月服务需求预测...")

# 构建收入映射
income_map = {}
for _, row in df_pop.iterrows():
    income_map[row['小区编号']] = float(row['人均月收入(元)'])

# 构建价格映射（紧急救助营收为0，公益免费）
price_map = {}
cost_map = {}
for _, row in df_revenue.iterrows():
    svc = row['服务项目']
    p = row['单次服务营收']
    c = row['单次服务直接支出']
    # 处理公益免费等特殊情况
    if pd.isna(p) or (isinstance(p, str) and '免费' in str(p)):
        p = 0.0
    if pd.isna(c) or (isinstance(c, str) and '免费' in str(c)):
        c = 0.0
    price_map[svc] = float(p)
    cost_map[svc] = float(c)

# 各类老人的理论月消费金额
# 先计算每种老人的理论月消费
elderly_types = ['自理', '半自理', '失能']
elderly_limit_keys = ['自理', '半失能', '失能']

# 计算每类老人的理论月消费
theory_consume = {}
for idx, etype in enumerate(elderly_types):
    total_cost = 0
    for svc in services:
        total_cost += demand_matrix[svc][etype] * price_map[svc]
    theory_consume[etype] = total_cost

print("\n  各类老人理论月消费金额:")
for etype in elderly_types:
    limit_key = elderly_limit_keys[elderly_types.index(etype)]
    limit_ratio = consume_limit[limit_key]
    print(f"    {etype}: 理论{theory_consume[etype]:.1f}元/月")

# 判断是否超限并进行削减
# 削减针对每类老人分别进行
cut_factors = {}
demand_adjusted = {}

for idx, etype in enumerate(elderly_types):
    limit_key = elderly_limit_keys[elderly_types.index(etype)]
    limit_ratio = consume_limit[limit_key]
    # 使用平均收入（取所有小区收入均值）
    avg_income = np.mean(list(income_map.values()))
    cap = limit_ratio * avg_income

    if theory_consume[etype] > cap:
        eta = cap / theory_consume[etype]
        cut_factors[etype] = eta
        demand_adjusted[etype] = {}
        for svc in services:
            demand_adjusted[etype][svc] = int(np.floor(eta * demand_matrix[svc][etype]))
    else:
        cut_factors[etype] = 1.0
        demand_adjusted[etype] = {}
        for svc in services:
            demand_adjusted[etype][svc] = int(demand_matrix[svc][etype])

print("\n  削减系数:")
for etype in elderly_types:
    print(f"    {etype}: {cut_factors[etype]:.4f}")

print("\n  削减后各类老人月均服务需求次数:")
for etype in elderly_types:
    svc_str = ', '.join([f"{svc}={demand_adjusted[etype][svc]}" for svc in services])
    print(f"    {etype}: {svc_str}")

# 计算各小区实际需求（使用各小区各自的收入计算削减）
# 先计算消费约束调整后的人均需求（仅在超出时削减），
# 然后按小区总人数汇总，取整在总量层面进行
actual_rows = []
for comm in communities:
    N_i1_5 = pred_results[comm]['N_i1'][5]
    N_i2_5 = pred_results[comm]['N_i2'][5]
    N_i3_5 = pred_results[comm]['N_i3'][5]
    income = income_map[comm]
    pops = {'自理': N_i1_5, '半自理': N_i2_5, '失能': N_i3_5}

    for idx, etype in enumerate(elderly_types):
        limit_key = elderly_limit_keys[idx]
        limit_ratio = consume_limit[limit_key]
        cap = limit_ratio * income

        if theory_consume[etype] > cap:
            eta = cap / theory_consume[etype]
        else:
            eta = 1.0

        for svc in services:
            raw_demand = demand_matrix[svc][etype]
            adj_per_person = eta * raw_demand  # 不取整人均值，取整总量
            total = int(round(pops[etype] * adj_per_person))

            actual_rows.append({
                '小区': comm,
                '服务项目': svc,
                '老人类型': etype,
                '约束后月需求次数': total,
                '人均月需求': round(adj_per_person, 2)
            })

df_actual = pd.DataFrame(actual_rows)
save_csv(df_actual, '第5年末消费约束下月服务需求次数.csv')

# 按小区和服务汇总
df_actual_summary = df_actual.groupby(['小区', '服务项目'])['约束后月需求次数'].sum().reset_index()
df_actual_pivot = df_actual_summary.pivot_table(
    values='约束后月需求次数', index='小区', columns='服务项目', aggfunc='sum')
df_actual_pivot = df_actual_pivot[services]
print("\n  === 第5年末各小区消费约束下月服务需求次数 ===")
print(df_actual_pivot.round(1).to_string())

# 保存预处理数据供后续问题使用
# 创建数据预处理文件夹
PREP_DIR = os.path.join(OUT_DIR, '数据预处理')
os.makedirs(PREP_DIR, exist_ok=True)

print("\n>>> 保存预处理数据...")

# Step1: 人口矩阵(10×3)
print("  [Step1] 人口矩阵(10×3)...")
pop_matrix = np.zeros((10, 3), dtype=int)
community_order = sorted(df_pop['小区编号'].unique())  # 按字母顺序排列
for idx, comm in enumerate(community_order):
    row = df_pop[df_pop['小区编号'] == comm].iloc[0]
    pop_matrix[idx, 0] = int(row['自理老人'])
    pop_matrix[idx, 1] = int(row['半失能老人'])
    pop_matrix[idx, 2] = int(row['失能老人'])
df_pop_matrix = pd.DataFrame(pop_matrix,
                              index=community_order,
                              columns=['自理老人', '半失能老人', '失能老人'])
df_pop_matrix.index.name = '小区'
df_pop_matrix.to_csv(os.path.join(PREP_DIR, 'Step1_人口矩阵_10x3.csv'), encoding='utf-8-sig')

# Step2: 转移概率矩阵(10×2)
print("  [Step2] 转移概率矩阵(10×2)...")
trans_prob = np.array([[p_ZB, p_BS] for _ in range(10)])
df_trans_matrix = pd.DataFrame(trans_prob,
                                index=community_order,
                                columns=['自理转半失能概率', '半失能转失能概率'])
df_trans_matrix.index.name = '小区'
df_trans_matrix.to_csv(os.path.join(PREP_DIR, 'Step2_转移概率矩阵_10x2.csv'), encoding='utf-8-sig')

# Step3: 需求频次矩阵Q(3×6)
print("  [Step3] 需求频次矩阵Q(3×6)...")
service_names = ['助餐', '日间照料', '上门护理', '康复理疗', '助浴', '紧急救助']
Q_matrix = np.zeros((3, 6))
for idx, row in df_demand.iterrows():
    svc = row['服务项目']
    if svc in service_names:
        svc_idx = service_names.index(svc)
        Q_matrix[0, svc_idx] = row['自理']
        Q_matrix[1, svc_idx] = row['半自理']
        Q_matrix[2, svc_idx] = row['失能']
df_Q_matrix = pd.DataFrame(Q_matrix,
                            index=['自理老人', '半失能老人', '失能老人'],
                            columns=service_names)
df_Q_matrix.to_csv(os.path.join(PREP_DIR, 'Step3_需求频次矩阵_Q_3x6.csv'), encoding='utf-8-sig')

# Step4: 服务价格向量A
print("  [Step4] 服务价格向量A...")
df_price_vector = pd.DataFrame({'服务项目': service_names, '单价(元/次)': [price_map[svc] for svc in service_names]})
df_price_vector.to_csv(os.path.join(PREP_DIR, 'Step4_服务价格向量_A.csv'), index=False, encoding='utf-8-sig')

# Step4: 消费上限向量U
print("  [Step5] 消费上限向量U...")
df_consume_upper = pd.DataFrame({
    '老人类型': ['自理', '半失能', '失能'],
    '月消费上限比例': [0.20, 0.25, 0.30]
})
df_consume_upper.to_csv(os.path.join(PREP_DIR, 'Step4_消费上限向量_U.csv'), index=False, encoding='utf-8-sig')

# 汇总表
print("  [Step6] 人口与转移概率汇总表...")
df_pop_trans = pd.concat([df_pop_matrix.reset_index(), df_trans_matrix.reset_index(drop=True)], axis=1)
df_pop_trans.to_csv(os.path.join(PREP_DIR, '汇总_人口与转移概率.csv'), index=False, encoding='utf-8-sig')

# 原有的预处理数据
print("  [Step7] 第5年末消费约束下月服务需求次数...")
prep_data = df_actual_summary.copy()
prep_data.to_csv(os.path.join(PREP_DIR, '预处理数据.csv'), index=False, encoding='utf-8-sig')
prep_data.to_csv(f'{BASE}/求解/预处理数据.csv', index=False, encoding='utf-8-sig')

print(f"\n  所有预处理数据已保存至: {PREP_DIR}")



# 5. 可视化
print("\n>>> 生成可视化图表...")

# 图1：三类老人数量变化趋势（1×3子图）
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
categories = ['自理老人', '半失能老人', '失能老人']
cat_keys = ['N_i1', 'N_i2', 'N_i3']
colors = COLORS_10

for idx, (ax, cat_name, cat_key) in enumerate(zip(axes, categories, cat_keys)):
    for ci, comm in enumerate(communities):
        values = pred_results[comm][cat_key]
        ax.plot(years, values, marker='o', linewidth=2, markersize=5,
                color=colors[ci], label=f'小区{comm}', alpha=0.85)
    ax.set_xlabel('年份', fontsize=11)
    ax.set_ylabel('人数', fontsize=11)
    ax.set_title(cat_name, fontsize=13, fontweight='bold')
    ax.set_xticks(years)
    ax.set_xticklabels(['当前', '第1年', '第2年', '第3年', '第4年', '第5年'],
                       fontsize=8, rotation=30)
    ax.legend(loc='upper left', fontsize=7, ncol=2, framealpha=0.5)

fig.suptitle('各小区三类老人数量变化趋势（第0-5年）', fontsize=15, fontweight='bold', y=1.01)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, '各小区三类老人数量变化趋势图.png')
print("  [1/19] 三类老人数量变化趋势图 已保存")

# 图2：第5年末各小区服务需求热力图
fig, ax = plt.subplots(figsize=(12, 8))
heatmap_data = df_theory_summary.values
im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
cbar = fig.colorbar(im, ax=ax, label='月需求次数')
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        ax.text(j, i, f'{heatmap_data[i, j]:.0f}', ha='center', va='center',
                fontsize=10, color='black' if heatmap_data[i, j] < np.max(heatmap_data)/2 else 'white')
ax.set_xticks(range(len(services)))
ax.set_xticklabels(services, fontsize=11)
ax.set_yticks(range(len(communities)))
ax.set_yticklabels(communities, fontsize=11)
ax.set_title('第5年末各小区理论月服务需求热力图', fontsize=14, fontweight='bold')
ax.set_xlabel('服务项目', fontsize=12)
ax.set_ylabel('小区', fontsize=12)
fig.tight_layout()
save_fig(fig, '第5年末各小区服务需求热力图.png')
print("  [2/19] 服务需求热力图 已保存")

# 图3：消费约束削减前后对比
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
elderly_labels = ['自理', '半自理', '失能']
for idx, (ax, etype) in enumerate(zip(axes, elderly_types)):
    x = np.arange(len(services))
    width = 0.35
    before = [demand_matrix[svc][etype] for svc in services]
    after = [demand_adjusted[etype][svc] for svc in services]
    bars1 = ax.bar(x - width/2, before, width, label='削减前', color='#4A90D9', alpha=0.8)
    bars2 = ax.bar(x + width/2, after, width, label='削减后', color='#F5A623', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(services, fontsize=9)
    ax.set_ylabel('月均次数', fontsize=11)
    ax.set_title(f'{elderly_labels[idx]}老人', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

fig.suptitle('消费约束削减前后各类老人月均服务需求对比', fontsize=15, fontweight='bold', y=1.01)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, '消费约束削减前后服务需求对比图.png')
print("  [3/19] 消费约束削减前后对比图 已保存")

# 图4：各小区老人总数预测曲线
fig, ax = plt.subplots(figsize=(14, 7))
for ci, comm in enumerate(communities):
    totals = [pred_results[comm]['N_i1'][t] + pred_results[comm]['N_i2'][t] + pred_results[comm]['N_i3'][t] for t in years]
    ax.plot(years, totals, marker='o', linewidth=2.5, markersize=7,
            color=colors[ci], label=f'小区{comm}', alpha=0.85)
ax.set_xlabel('年份', fontsize=12)
ax.set_ylabel('老人总数', fontsize=12)
ax.set_title('各小区老人总数预测曲线（第0-5年）', fontsize=14, fontweight='bold')
ax.set_xticks(years)
ax.set_xticklabels(['当前', '第1年', '第2年', '第3年', '第4年', '第5年'], fontsize=10)
ax.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.8)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, '各小区老人总数预测曲线.png')
print("  [4/19] 各小区老人总数预测曲线 已保存")

# 图5：第0年 vs 第5年老人结构堆叠柱状图
fig, ax = plt.subplots(figsize=(14, 7))
width = 0.35
x = np.arange(len(communities))

# 第0年数据
z0 = [pred_results[comm]['N_i1'][0] for comm in communities]
b0 = [pred_results[comm]['N_i2'][0] for comm in communities]
s0 = [pred_results[comm]['N_i3'][0] for comm in communities]

# 第5年数据
z5 = [pred_results[comm]['N_i1'][5] for comm in communities]
b5 = [pred_results[comm]['N_i2'][5] for comm in communities]
s5 = [pred_results[comm]['N_i3'][5] for comm in communities]

ax.bar(x - width/2, z0, width, label='自理(当前)', color='#4A90D9', alpha=0.85)
ax.bar(x - width/2, b0, width, bottom=z0, label='半失能(当前)', color='#F5A623', alpha=0.85)
ax.bar(x - width/2, s0, width, bottom=np.array(z0)+np.array(b0), label='失能(当前)', color='#7ED321', alpha=0.85)

ax.bar(x + width/2, z5, width, label='自理(第5年)', color='#86C6F5', alpha=0.85)
ax.bar(x + width/2, b5, width, bottom=z5, label='半失能(第5年)', color='#FFD17A', alpha=0.85)
ax.bar(x + width/2, s5, width, bottom=np.array(z5)+np.array(b5), label='失能(第5年)', color='#B8E986', alpha=0.85)

ax.set_xlabel('小区', fontsize=12)
ax.set_ylabel('老人数量', fontsize=12)
ax.set_title('各小区第0年与第5年老人结构对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(communities, fontsize=10)
ax.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, '第0年与第5年老人结构堆叠柱状图.png')
print("  [5/19] 第0年与第5年老人结构堆叠柱状图 已保存")

# 图6：各小区老人总数占比饼图（第5年末）
fig, ax = plt.subplots(figsize=(10, 10))
y5_totals = [pred_results[comm]['N_i1'][5] + pred_results[comm]['N_i2'][5] + pred_results[comm]['N_i3'][5] for comm in communities]
total = sum(y5_totals)
labels = [f'小区{c} ({(y5_totals[i]/total)*100:.1f}%)' for i, c in enumerate(communities)]
wedges, texts, autotexts = ax.pie(y5_totals, labels=labels, autopct='%1.1f%%',
                                   colors=colors, startangle=90, pctdistance=0.85)
ax.set_title('第5年末各小区老人总数占比', fontsize=14, fontweight='bold')
fig.tight_layout()
save_fig(fig, '第5年末各小区老人总数占比饼图.png')
print("  [6/19] 第5年末各小区老人总数占比饼图 已保存")

# 图7：老人总量变化的热力图（小区×年份）
heatmap_data = []
for comm in communities:
    row = [pred_results[comm]['N_i1'][t] + pred_results[comm]['N_i2'][t] + pred_results[comm]['N_i3'][t] for t in years]
    heatmap_data.append(row)
heatmap_data = np.array(heatmap_data)

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(heatmap_data, cmap='BuPu', aspect='auto')
cbar = fig.colorbar(im, ax=ax, label='老人总数')
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        ax.text(j, i, f'{heatmap_data[i, j]}', ha='center', va='center',
                fontsize=11, color='black' if heatmap_data[i, j] < np.max(heatmap_data)/2 else 'white')
ax.set_xticks(range(len(years)))
ax.set_xticklabels(['当前', '第1年', '第2年', '第3年', '第4年', '第5年'], fontsize=10)
ax.set_yticks(range(len(communities)))
ax.set_yticklabels(communities, fontsize=10)
ax.set_title('老人总量变化热力图（小区×年份）', fontsize=14, fontweight='bold')
ax.set_xlabel('年份', fontsize=12)
ax.set_ylabel('小区', fontsize=12)
fig.tight_layout()
save_fig(fig, '老人总量变化热力图.png')
print("  [7/19] 老人总量变化热力图 已保存")

# 图8：新增老人与死亡老人的贡献分析（面积图）
# 计算每年总人数、总新增、总死亡
yearly_total = []
yearly_new = []
yearly_death = []

for t in years:
    total = 0
    new = 0
    death = 0
    for comm in communities:
        if t == 0:
            prev_total = 0
        else:
            prev_total = pred_results[comm]['N_i1'][t-1] + pred_results[comm]['N_i2'][t-1] + pred_results[comm]['N_i3'][t-1]
        curr_total = pred_results[comm]['N_i1'][t] + pred_results[comm]['N_i2'][t] + pred_results[comm]['N_i3'][t]
        total += curr_total
        if t > 0:
            new += g * prev_total
            death += d * prev_total
    yearly_total.append(total)
    yearly_new.append(new)
    yearly_death.append(death)

# 计算累计效应（从第0年开始的累计新增和死亡）
cumulative_new = np.cumsum([0] + yearly_new[1:])
cumulative_death = np.cumsum([0] + yearly_death[1:])

fig, ax = plt.subplots(figsize=(14, 7))
ax.stackplot(years, cumulative_new, cumulative_death, 
             labels=['累计新增', '累计死亡'], 
             colors=['#7ED321', '#D0021B'], alpha=0.8)
ax.plot(years, yearly_total, marker='o', color='#4A90D9', linewidth=2.5,
        label='实际总人数', alpha=0.9)
ax.set_xlabel('年份', fontsize=12)
ax.set_ylabel('人数', fontsize=12)
ax.set_title('新增老人与死亡老人的贡献分析（累计效应）', fontsize=14, fontweight='bold')
ax.set_xticks(years)
ax.set_xticklabels(['当前', '第1年', '第2年', '第3年', '第4年', '第5年'], fontsize=10)
ax.legend(loc='upper left', fontsize=10, framealpha=0.8)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, '新增老人与死亡老人贡献分析面积图.png')
print("  [8/19] 新增老人与死亡老人贡献分析面积图 已保存")

# 图9：各小区总需求排名条形图
community_totals = df_theory_summary.sum(axis=1).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.bar(community_totals.index, community_totals.values, color=colors[:len(community_totals)], alpha=0.85)
ax.set_xlabel('小区', fontsize=12)
ax.set_ylabel('月需求次数', fontsize=12)
ax.set_title('各小区总理论需求排名', fontsize=14, fontweight='bold')
ax.set_xticklabels(community_totals.index, fontsize=10)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0f}', ha='center', va='bottom', fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, '各小区总需求排名条形图.png')
print("  [9/19] 各小区总需求排名条形图 已保存")

# 图10：各服务项目总需求构成饼图
service_totals = df_theory_summary.sum(axis=0)
fig, ax = plt.subplots(figsize=(10, 10))
labels = [f'{s} ({(service_totals[s]/service_totals.sum())*100:.1f}%)' for s in services]
wedges, texts, autotexts = ax.pie(service_totals.values, labels=labels, autopct='%1.1f%%',
                                   colors=colors[:len(services)], startangle=90, pctdistance=0.85)
ax.set_title('各服务项目总需求构成', fontsize=14, fontweight='bold')
fig.tight_layout()
save_fig(fig, '各服务项目总需求构成饼图.png')
print("  [10/19] 各服务项目总需求构成饼图 已保存")

# 图11：各小区内服务需求构成堆叠柱状图
fig, ax = plt.subplots(figsize=(14, 7))
bottom = np.zeros(len(communities))
for i, svc in enumerate(services):
    values = df_theory_summary[svc].values
    ax.bar(communities, values, bottom=bottom, label=svc, color=colors[i], alpha=0.85)
    bottom += values
ax.set_xlabel('小区', fontsize=12)
ax.set_ylabel('月需求次数', fontsize=12)
ax.set_title('各小区内服务需求构成', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, '各小区内服务需求构成堆叠柱状图.png')
print("  [11/19] 各小区内服务需求构成堆叠柱状图 已保存")

# 图12：需求相似性聚类热力图
fig = sns.clustermap(df_theory_summary, cmap='YlOrRd', method='ward',
                     figsize=(12, 10), cbar_pos=(0.02, 0.8, 0.03, 0.18))
fig.fig.suptitle('需求相似性聚类热力图', fontsize=14, fontweight='bold', y=1.02)
plt.setp(fig.ax_heatmap.get_xticklabels(), rotation=45, fontsize=10)
plt.setp(fig.ax_heatmap.get_yticklabels(), fontsize=10)
save_fig(fig.fig, '需求相似性聚类热力图.png')
print("  [12/19] 需求相似性聚类热力图 已保存")

# 计算各小区理论总需求与实际总需求
theory_totals = df_theory_summary.sum(axis=1)
actual_totals = df_actual_pivot.sum(axis=1)
cut_rates = (theory_totals - actual_totals) / theory_totals * 100

# 图13：各小区总需求削减率柱状图
fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.bar(cut_rates.index, cut_rates.values, color='#FF6B6B', alpha=0.85)
ax.set_xlabel('小区', fontsize=12)
ax.set_ylabel('削减率 (%)', fontsize=12)
ax.set_title('各小区总需求削减率', fontsize=14, fontweight='bold')
ax.set_xticklabels(cut_rates.index, fontsize=10)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, '各小区总需求削减率柱状图.png')
print("  [13/19] 各小区总需求削减率柱状图 已保存")

# 图14：各服务项目的需求削减率条形图（分老人类型）
cut_rates_by_type = pd.DataFrame()
for etype in elderly_types:
    before = [demand_matrix[svc][etype] for svc in services]
    after = [demand_adjusted[etype][svc] for svc in services]
    rates = [(b - a) / b * 100 if b != 0 else 0 for b, a in zip(before, after)]
    cut_rates_by_type[etype] = rates
cut_rates_by_type.index = services

fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(services))
width = 0.28
ax.bar(x - width, cut_rates_by_type['自理'], width, label='自理', color='#4A90D9', alpha=0.85)
ax.bar(x, cut_rates_by_type['半自理'], width, label='半自理', color='#F5A623', alpha=0.85)
ax.bar(x + width, cut_rates_by_type['失能'], width, label='失能', color='#7ED321', alpha=0.85)
ax.set_xlabel('服务项目', fontsize=12)
ax.set_ylabel('削减率 (%)', fontsize=12)
ax.set_title('各服务项目需求削减率（按老人类型）', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(services, fontsize=9, rotation=30)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, '各服务项目需求削减率条形图.png')
print("  [14/19] 各服务项目需求削减率条形图 已保存")

# 图15：小区人均月收入 vs 消费上限覆盖程度
satisfaction_rates = actual_totals / theory_totals
incomes = [income_map[comm] for comm in communities]

fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(incomes, satisfaction_rates, s=200, c=cut_rates.values, 
                     cmap='RdYlBu_r', alpha=0.8, edgecolors='black')
ax.axhline(y=1, color='red', linestyle='--', label='无削减')
ax.set_xlabel('人均月收入 (元)', fontsize=12)
ax.set_ylabel('综合满足率 (实际/理论)', fontsize=12)
ax.set_title('小区人均月收入 vs 消费上限覆盖程度', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='削减率 (%)')
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, '人均月收入vs消费上限覆盖程度散点图.png')
print("  [15/19] 人均月收入vs消费上限覆盖程度散点图 已保存")

# 图16：实际需求与理论需求对比散点图
fig, ax = plt.subplots(figsize=(12, 10))
for ci, comm in enumerate(communities):
    theory_vals = df_theory_summary.loc[comm].values
    actual_vals = df_actual_pivot.loc[comm].values
    ax.scatter(theory_vals, actual_vals, s=100, color=colors[ci], 
               label=f'小区{comm}', alpha=0.8, edgecolors='black')
ax.plot([0, df_theory_summary.values.max()], [0, df_theory_summary.values.max()], 
        'r--', label='y=x')
ax.set_xlabel('理论需求', fontsize=12)
ax.set_ylabel('实际需求', fontsize=12)
ax.set_title('实际需求与理论需求对比散点图', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, ncol=2)
ax.grid(True, linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, '实际需求与理论需求对比散点图.png')
print("  [16/19] 实际需求与理论需求对比散点图 已保存")

# 图17：削减前后各服务需求总量的蝴蝶图
theory_service_total = df_theory_summary.sum(axis=0)
actual_service_total = df_actual_pivot.sum(axis=0)

fig, ax = plt.subplots(figsize=(14, 8))
y_pos = np.arange(len(services))
ax.barh(y_pos, -theory_service_total.values, height=0.6, label='削减前', 
        color='#4A90D9', alpha=0.85)
ax.barh(y_pos, actual_service_total.values, height=0.6, label='削减后', 
        color='#F5A623', alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(services, fontsize=11)
ax.set_xlabel('月需求次数', fontsize=12)
ax.set_title('削减前后各服务需求总量对比（蝴蝶图）', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='x', linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, '削减前后服务需求蝴蝶图.png')
print("  [17/19] 削减前后服务需求蝴蝶图 已保存")


# 综合仪表板
# 图18：关键指标仪表板
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 子图1：老人总数指标
total_elderly = df_y5['老人总数'].sum()
axes[0, 0].text(0.5, 0.5, f'{total_elderly:,}', fontsize=40, ha='center', va='center', fontweight='bold', color='#4A90D9')
axes[0, 0].set_title('第5年末老人总数', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# 子图2：理论总需求
total_theory_demand = df_theory_summary.sum().sum()
axes[0, 1].text(0.5, 0.5, f'{total_theory_demand:.0f}', fontsize=40, ha='center', va='center', fontweight='bold', color='#4A90D9')
axes[0, 1].set_title('理论月服务需求总次数', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

# 子图3：实际总需求
total_actual_demand = df_actual_pivot.sum().sum()
axes[0, 2].text(0.5, 0.5, f'{total_actual_demand:.0f}', fontsize=40, ha='center', va='center', fontweight='bold', color='#7ED321')
axes[0, 2].set_title('实际月服务需求总次数', fontsize=12, fontweight='bold')
axes[0, 2].axis('off')

# 子图4：削减比例
cut_ratio = (1 - total_actual_demand / total_theory_demand) * 100
axes[1, 0].text(0.5, 0.5, f'{cut_ratio:.1f}%', fontsize=40, ha='center', va='center', 
                fontweight='bold', color='#D0021B' if cut_ratio > 30 else '#F5A623')
axes[1, 0].set_title('需求削减比例', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

# 子图5：平均收入
avg_income = np.mean(list(income_map.values()))
axes[1, 1].text(0.5, 0.5, f'{avg_income:.0f}', fontsize=40, ha='center', va='center', fontweight='bold', color='#F5A623')
axes[1, 1].set_title('小区人均月收入(元)', fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

# 子图6：小区数量
axes[1, 2].text(0.5, 0.5, f'{len(communities)}', fontsize=40, ha='center', va='center', fontweight='bold', color='#9013FE')
axes[1, 2].set_title('服务小区数量', fontsize=12, fontweight='bold')
axes[1, 2].axis('off')

fig.suptitle('关键指标仪表板', fontsize=16, fontweight='bold', y=0.98)
fig.tight_layout()
save_fig(fig, '关键指标仪表板.png')
print("  [18/19] 关键指标仪表板 已保存")

# 图19：人口-需求-收入联动气泡图
fig, ax = plt.subplots(figsize=(12, 8))
elderly_counts = df_y5['老人总数'].values
demand_values = actual_totals.values
income_values = np.array([income_map[comm] for comm in communities])
cut_values = cut_rates.values

scatter = ax.scatter(elderly_counts, demand_values, 
                     s=income_values/10,
                     c=cut_values, cmap='Spectral_r',
                     alpha=0.7, edgecolors='black', linewidth=1)

# 添加标签
for i, comm in enumerate(communities):
    ax.text(elderly_counts[i], demand_values[i], comm, fontsize=9, ha='center', va='bottom')

ax.set_xlabel('小区老人总数', fontsize=12)
ax.set_ylabel('实际总需求', fontsize=12)
ax.set_title('人口-需求-收入联动气泡图', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='削减率 (%)')
ax.grid(True, linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, '人口-需求-收入联动气泡图.png')
print("  [19/19] 人口-需求-收入联动气泡图 已保存")


# 6. 结果汇总
print("\n" + "=" * 60)
print("问题一求解完成！")
print("=" * 60)

# 汇总关键数值
print("\n  >>> 关键结果汇总 <<<")
print(f"  第5年末10小区老人总数: {df_y5['老人总数'].sum()}人")
print(f"    其中自理: {df_y5['自理老人'].sum()}人, 半失能: {df_y5['半失能老人'].sum()}人, 失能: {df_y5['失能老人'].sum()}人")
print(f"  第5年末理论月服务需求总次数: {df_theory_summary.sum().sum():.0f}次/月")
print(f"  第5年末约束后月服务需求总次数: {df_actual_pivot.sum().sum():.0f}次/月")

total_theory = df_theory_summary.sum().sum()
total_actual = df_actual_pivot.sum().sum()
print(f"  消费约束导致需求削减: {(1-total_actual/total_theory)*100:.1f}%")