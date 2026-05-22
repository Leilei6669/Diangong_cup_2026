"""
问题3数据预处理模块
功能：生成Python数据文件，包含服务定价与政府补贴优化所需的所有参数
"""

import numpy as np
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_q3_data():
    """生成问题3所需的所有数据"""
    print("=" * 60)
    print("问题3数据预处理")
    print("=" * 60)
    
    # ----------------------
    # 1. 老人数量与需求数据（假设第5年末数据）
    # ----------------------
    print("\n1. 老人数量与需求数据")
    
    # 小区编号
    communities = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    
    # 老人类型：自理、半失能、失能
    care_types = ['自理', '半失能', '失能']
    
    # 第5年末各小区各类型老人数量 N_ir^(5)
    # 数据来源：问题1的预测结果（此处使用模拟数据）
    N_ir = np.array([
        [600, 300, 300],   # A
        [700, 350, 300],   # B
        [500, 250, 230],   # C
        [600, 300, 250],   # D
        [550, 270, 260],   # E
        [650, 320, 280],   # F
        [520, 260, 240],   # G
        [620, 310, 250],   # H
        [580, 290, 230],   # I
        [600, 300, 300]    # J
    ])
    
    print(f"各小区老人数量矩阵({len(communities)}×{len(care_types)}):")
    for i, name in enumerate(communities):
        print(f"  {name}: {N_ir[i]}")
    
    # 各类型老人对各项服务的月均需求次数 a_rm
    # 服务类型：助餐、日间照料、上门护理、康复理疗、助浴、紧急救助
    services = ['助餐', '日间照料', '上门护理', '康复理疗', '助浴', '紧急救助']
    
    # a_rm[r, m]: 第r类老人对第m项服务的月均需求次数
    a_rm = np.array([
        [12, 4, 2, 3, 2, 0.5],    # 自理老人
        [15, 12, 8, 6, 4, 1.0],    # 半失能老人
        [20, 20, 15, 10, 8, 2.0]   # 失能老人
    ])
    
    print("\n各类老人月均服务需求次数:")
    for r, care_type in enumerate(care_types):
        print(f"  {care_type}: {a_rm[r]}")
    
    # 计算理论月需求 D_irm^0 = N_ir * a_rm
    # D_irm[i, r, m]: 第i小区第r类老人对第m项服务的理论月需求
    D_irm_0 = np.zeros((10, 3, 6))
    for i in range(10):
        for r in range(3):
            for m in range(6):
                D_irm_0[i, r, m] = N_ir[i, r] * a_rm[r, m]
    
    print("\n理论月需求总量:")
    for m, service in enumerate(services):
        total = D_irm_0[:, :, m].sum()
        print(f"  {service}: {total:.0f}次/月")
    
    # ----------------------
    # 2. 服务价格与成本数据
    # ----------------------
    print("\n2. 服务价格与成本数据")
    
    # 基准价格 p_m^0（元/次）
    # 紧急救助为公益免费，价格为0
    p_m_0 = np.array([10, 20, 30, 28, 25, 0])
    
    # 单位直接成本 c_m（元/次）
    c_m = np.array([8, 16, 24, 23, 20, 8])
    
    print("服务价格与成本表:")
    print(f"{'服务':<10} {'基准价格':<10} {'直接成本':<10}")
    print("-" * 30)
    for m, service in enumerate(services):
        print(f"{service:<10} {p_m_0[m]:<10} {c_m[m]:<10}")
    
    # ----------------------
    # 3. 服务站成本数据
    # ----------------------
    print("\n3. 服务站成本数据")
    
    # 假设问题2确定的服务站集合（模拟数据）
    station_info = [
        {'community': 'B', 'size': '中型'},
        {'community': 'E', 'size': '小型'},
        {'community': 'H', 'size': '大型'}
    ]
    
    # 服务站规模参数
    size_params = {
        '小型': {'capacity': 1000, 'build_cost': 18, 'daily_operating': 2000, 'daily_subsidy_limit': 1000},
        '中型': {'capacity': 2000, 'build_cost': 32, 'daily_operating': 3200, 'daily_subsidy_limit': 1800},
        '大型': {'capacity': 3000, 'build_cost': 45, 'daily_operating': 4400, 'daily_subsidy_limit': 2600}
    }
    
    # 计算各服务站年度固定成本
    stations_data = []
    for info in station_info:
        size = info['size']
        params = size_params[size]
        
        # 年运营固定成本 F_j = 365 * 日均管理成本(元) / 10000 = 万元/年
        F_j = params['daily_operating'] * 365 / 10000
        
        # 年均建设折旧成本 E_j = 建设成本(万元) * 10000 / 20 / 10000 = 万元/年
        E_j = params['build_cost'] / 20
        
        # 年度固定总成本
        C_j_fix = F_j + E_j
        
        stations_data.append({
            'community': info['community'],
            'size': size,
            'capacity': params['capacity'],
            'build_cost': params['build_cost'],
            'annual_operating': F_j,
            'annual_depreciation': E_j,
            'annual_fixed_cost': C_j_fix,
            'daily_subsidy_limit': params['daily_subsidy_limit'],
            'annual_subsidy_limit': params['daily_subsidy_limit'] * 365 / 10000  # 万元/年
        })
    
    print("服务站成本表:")
    print(f"{'小区':<6} {'规模':<6} {'容量':<8} {'年固定成本(万元)':<15} {'年补贴上限(万元)':<15}")
    print("-" * 50)
    for s in stations_data:
        print(f"{s['community']:<6} {s['size']:<6} {s['capacity']:<8} {s['annual_fixed_cost']:<15.2f} {s['annual_subsidy_limit']:<15.2f}")
    
    # ----------------------
    # 4. 政府补贴数据
    # ----------------------
    print("\n4. 政府补贴数据")
    
    # 单次补贴金额（元/人次）
    b = 2
    
    # 紧急救助不补贴标记
    subsidized_services = [True, True, True, True, True, False]
    
    print(f"单次补贴金额: {b}元/人次")
    print(f"补贴服务: {[services[i] for i in range(6) if subsidized_services[i]]}")
    print(f"无补贴服务: {[services[i] for i in range(6) if not subsidized_services[i]]}")
    
    # ----------------------
    # 5. 满意度规则数据
    # ----------------------
    print("\n5. 满意度规则数据")
    
    # 满意度权重
    alpha_d = 0.2  # 距离满意度权重
    alpha_r = 0.3  # 响应满意度权重
    alpha_p = 0.5  # 价格满意度权重
    
    print(f"满意度权重: S = {alpha_d}*S1 + {alpha_r}*S2 + {alpha_p}*S3")
    print("其中: S1-距离满意度, S2-响应满意度, S3-价格满意度")
    
    # ----------------------
    # 生成Python数据文件
    # ----------------------
    print("\n6. 生成Python数据文件")
    
    python_data = f'''"""
问题3 数据文件 - 自动生成
所有单位说明：
- 价格、成本：元/次
- 年度成本：万元/年
- 需求量：次/月
- 补贴：元/人次
"""

import numpy as np

# ========== 1. 老人数量与需求数据 ==========

# 小区编号
COMMUNITIES = {communities}

# 老人类型
CARE_TYPES = {care_types}

# 服务类型
SERVICES = {services}

# 第5年末各小区各类型老人数量 N_ir^(5) (人)
N_ir = np.array({N_ir.tolist()})

# 各类型老人对各项服务的月均需求次数 a_rm (次/月/人)
a_rm = np.array({a_rm.tolist()})

# 理论月需求 D_irm^0 = N_ir * a_rm (次/月)
D_irm_0 = np.array({D_irm_0.tolist()})

# ========== 2. 服务价格与成本数据 ==========

# 基准价格 p_m^0 (元/次)，紧急救助为0（公益免费）
P_M_0 = np.array({p_m_0.tolist()})

# 单位直接成本 c_m (元/次)
C_M = np.array({c_m.tolist()})

# ========== 3. 服务站数据 ==========

# 服务站信息
STATIONS = {[{'community': s['community'], 'size': s['size']} for s in stations_data]}

# 服务站容量 (人次/日)
STATION_CAPACITIES = np.array([{', '.join(str(s['capacity']) for s in stations_data)}])

# 服务站年度固定成本 (万元/年)
STATION_FIXED_COSTS = np.array([{', '.join(f"{s['annual_fixed_cost']:.2f}" for s in stations_data)}])

# 服务站年度补贴上限 (万元/年)
STATION_SUBSIDY_LIMITS = np.array([{', '.join(f"{s['annual_subsidy_limit']:.2f}" for s in stations_data)}])

# ========== 4. 政府补贴参数 ==========

# 单次补贴金额 (元/人次)
SUBSIDY_PER_SERVICE = {b}

# 是否享受补贴（紧急救助除外）
SUBSIDIZED_SERVICES = {subsidized_services}

# ========== 5. 满意度权重 ==========

# 距离满意度权重
ALPHA_D = {alpha_d}

# 响应满意度权重
ALPHA_R = {alpha_r}

# 价格满意度权重
ALPHA_P = {alpha_p}

# ========== 6. 其他参数 ==========

# 利润率上限
MAX_PROFIT_RATE = 0.08

# 折现率（用于计算净现值）
DISCOUNT_RATE = 0.05
'''
    
    output_file = os.path.join(OUTPUT_DIR, 'q3_data.py')
    with open(output_file, 'w', encoding='utf-8-sig') as f:
        f.write(python_data)
    
    print(f"数据文件已保存到: {output_file}")
    print("\n" + "=" * 60)
    print("问题3数据预处理完成！")
    print("=" * 60)

if __name__ == "__main__":
    generate_q3_data()