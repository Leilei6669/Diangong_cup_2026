"""
问题4数据预处理模块
功能：生成扰动情景参数表和结果比较所需数据
"""

import numpy as np
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_q4_data():
    """生成问题4所需的所有数据"""
    print("=" * 60)
    print("问题4数据预处理")
    print("=" * 60)
    
    # ----------------------
    # 1. 扰动情景设定
    # ----------------------
    print("\n1. 扰动情景设定")
    
    # 情景定义
    scenarios = {
        'S0': {
            'name': '基准情景',
            'description': '使用问题1、2、3原始参数',
            'params': {}
        },
        'S1': {
            'name': '老人增长与失能转移增强情景',
            'description': 'g=8%, p12=5.5%, p23=9.5%',
            'params': {
                'growth_rate': 0.08,        # 人口增长率8%
                'p12': 0.055,              # 自理转半失能率5.5%
                'p23': 0.095               # 半失能转失能率9.5%
            }
        },
        'S2': {
            'name': '成本上升情景',
            'description': '日固定管理成本增加20%',
            'params': {
                'cost_increase_rate': 0.20  # 成本增加20%
            }
        },
        'S3': {
            'name': '预算增加情景',
            'description': '总建设预算调整为140万元',
            'params': {
                'budget': 140  # 万元
            }
        },
        'S4': {
            'name': '综合压力情景',
            'description': '同时采用老人增长增强、成本上升和预算增加',
            'params': {
                'growth_rate': 0.08,
                'p12': 0.055,
                'p23': 0.095,
                'cost_increase_rate': 0.20,
                'budget': 140
            }
        }
    }
    
    print("\n扰动情景表:")
    print("-" * 80)
    print(f"{'情景':<6} {'名称':<20} {'参数变化'}")
    print("-" * 80)
    for key, scenario in scenarios.items():
        print(f"{key:<6} {scenario['name']:<20} {scenario['description']}")
    
    # ----------------------
    # 2. 服务站参数（基准情景）
    # ----------------------
    print("\n2. 服务站参数（基准情景）")
    
    # 服务站基础参数
    station_base_params = {
        '小型': {'capacity': 1000, 'build_cost': 18, 'daily_operating': 2000, 'subsidy_limit': 1000},
        '中型': {'capacity': 2000, 'build_cost': 32, 'daily_operating': 3200, 'subsidy_limit': 1800},
        '大型': {'capacity': 3000, 'build_cost': 45, 'daily_operating': 4400, 'subsidy_limit': 2600}
    }
    
    # 计算基准情景服务站数据
    base_stations = [
        {'community': 'B', 'size': '中型'},
        {'community': 'E', 'size': '小型'},
        {'community': 'H', 'size': '大型'}
    ]
    
    stations_data = []
    for st in base_stations:
        params = station_base_params[st['size']]
        
        # 年运营固定成本 F_j = 365 * 日均管理成本 / 10000 (万元/年)
        F_j = params['daily_operating'] * 365 / 10000
        
        # 年均建设折旧成本 E_j = 建设成本 / 20 (万元/年)
        E_j = params['build_cost'] / 20
        
        # 年度固定总成本
        C_j_fix = F_j + E_j
        
        stations_data.append({
            'community': st['community'],
            'size': st['size'],
            'capacity': params['capacity'],
            'build_cost': params['build_cost'],
            'annual_operating': F_j,
            'annual_depreciation': E_j,
            'annual_fixed_cost': C_j_fix,
            'daily_subsidy_limit': params['subsidy_limit'],
            'annual_subsidy_limit': params['subsidy_limit'] * 365 / 10000  # 万元/年
        })
    
    print("\n服务站基础参数:")
    print(f"{'小区':<6} {'规模':<6} {'容量':<8} {'建设成本':<10} {'年固定成本':<12} {'年补贴上限':<12}")
    print("-" * 60)
    for s in stations_data:
        print(f"{s['community']:<6} {s['size']:<6} {s['capacity']:<8} "
              f"{s['build_cost']:<10} {s['annual_fixed_cost']:<12.2f} {s['annual_subsidy_limit']:<12.2f}")
    
    # ----------------------
    # 3. 各情景服务站参数调整
    # ----------------------
    print("\n3. 各情景服务站参数调整")
    
    # S2情景：成本上升20%
    s2_stations = []
    for s in stations_data:
        s2_s = s.copy()
        s2_s['annual_fixed_cost'] = s['annual_fixed_cost'] * 1.20
        s2_stations.append(s2_s)
    
    # S3情景：预算增加到140万（需要重新优化选址，可能增加服务站）
    # 此处先保持原有服务站，预算变化主要影响问题2的选址结果
    s3_stations = stations_data.copy()
    
    # ----------------------
    # 4. 情景参数汇总表
    # ----------------------
    print("\n4. 情景参数汇总表")
    
    # 定义用于问题2、3的关键参数
    q2_q3_params = {
        'BUDGET': 120,                    # 基准预算（万元）
        'BUDGET_S3': 140,                 # S3情景预算（万元）
        'growth_rate_base': 0.05,         # 基准人口增长率
        'growth_rate_S1': 0.08,           # S1情景人口增长率
        'p12_base': 0.04,                 # 基准自理转半失能率
        'p12_S1': 0.055,                 # S1情景自理转半失能率
        'p23_base': 0.08,                 # 基准半失能转失能率
        'p23_S1': 0.095,                 # S1情景半失能转失能率
        'cost_increase_rate': 0.20,       # 成本上升比例
    }
    
    # ----------------------
    # 5. 结果比较表结构
    # ----------------------
    print("\n5. 结果比较表结构")
    
    comparison_metrics = {
        '问题2_选址结果': ['服务站位置', '服务站规模', '服务站数量', '总建设成本', '覆盖率'],
        '问题3_定价结果': ['各服务站价格方案', '各类服务价格', '总满意度', '平均价格满意度'],
        '问题4_情景分析': {
            'S0': ['基准情景满意度', '基准情景利润', '基准情景补贴'],
            'S1': ['老人增长情景满意度', '老人增长情景利润', '老人增长情景补贴'],
            'S2': ['成本上升情景满意度', '成本上升情景利润', '成本上升情景补贴'],
            'S3': ['预算增加情景满意度', '预算增加情景利润', '预算增加情景补贴'],
            'S4': ['综合压力情景满意度', '综合压力情景利润', '综合压力情景补贴']
        }
    }
    
    print("\n比较指标:")
    print("-" * 60)
    for category, metrics in comparison_metrics.items():
        print(f"\n{category}:")
        if isinstance(metrics, dict):
            for scenario, items in metrics.items():
                print(f"  {scenario}: {items}")
        else:
            print(f"  {metrics}")
    
    # ----------------------
    # 6. 生成Python数据文件
    # ----------------------
    print("\n6. 生成Python数据文件")
    
    python_data = f'''"""
问题4 数据文件 - 自动生成
功能：扰动情景参数和结果比较数据
"""

import numpy as np

# ========== 1. 扰动情景定义 ==========

SCENARIOS = {scenarios}

# ========== 2. 服务站基础参数 ==========

STATION_BASE_PARAMS = {station_base_params}

# 基准情景服务站配置
BASE_STATIONS = {base_stations}

# 各情景服务站数据
STATIONS_DATA = {{
    'S0': {stations_data},
    'S2': {s2_stations},
    'S3': {s3_stations},
    'S1': {stations_data},  # S1只影响人口结构，不影响服务站
    'S4': {s2_stations}      # S4与S2服务站成本相同
}}

# ========== 3. 问题2、3关键参数 ==========

Q2_Q3_PARAMS = {q2_q3_params}

# 基准预算（万元）
BUDGET_BASE = {q2_q3_params['BUDGET']}

# S3情景预算（万元）
BUDGET_S3 = {q2_q3_params['BUDGET_S3']}

# 人口增长参数
GROWTH_RATE_BASE = {q2_q3_params['growth_rate_base']}
GROWTH_RATE_S1 = {q2_q3_params['growth_rate_S1']}

# 失能转移概率
P12_BASE = {q2_q3_params['p12_base']}
P12_S1 = {q2_q3_params['p12_S1']}
P23_BASE = {q2_q3_params['p23_base']}
P23_S1 = {q2_q3_params['p23_S1']}

# 成本变化参数
COST_INCREASE_RATE = {q2_q3_params['cost_increase_rate']}

# ========== 4. 结果比较指标 ==========

COMPARISON_METRICS = {comparison_metrics}

# ========== 5. 数据一致性检查标准 ==========

DATA_CONSISTENCY = {{
    '人口数量单位': '人',
    '服务需求单位': '次/月',
    '服务能力单位': '人次/日',
    '建设成本单位': '万元',
    '运营成本单位': '元/日 或 万元/年',
    '补贴单位': '元/年 或 万元/年',
    '满意度范围': [0.6, 1.0],
    '距离矩阵': '保持不变'
}}

# ========== 6. 情景参数获取函数 ==========

def get_scenario_params(scenario_name):
    \"\"\"
    获取指定情景的参数
    
    参数:
        scenario_name: 情景名称 ('S0', 'S1', 'S2', 'S3', 'S4')
    
    返回:
        dict: 情景参数字典
    \"\"\"
    return SCENARIOS.get(scenario_name, SCENARIOS['S0'])

def get_stations_data(scenario_name):
    \"\"\"
    获取指定情景的服务站数据
    
    参数:
        scenario_name: 情景名称
    
    返回:
        list: 服务站数据列表
    \"\"\"
    return STATIONS_DATA.get(scenario_name, STATIONS_DATA['S0'])

def apply_scenario_adjustments(scenario_name, data):
    \"\"\"
    对基准数据进行情景调整
    
    参数:
        scenario_name: 情景名称
        data: 基准数据
    
    返回:
        调整后的数据
    \"\"\"
    scenario = SCENARIOS.get(scenario_name, SCENARIOS['S0'])
    params = scenario['params']
    
    if scenario_name in ['S2', 'S4']:
        # 成本上升情景：服务站固定成本增加
        if 'annual_fixed_cost' in data:
            data = data.copy()
            data['annual_fixed_cost'] = data['annual_fixed_cost'] * (1 + params.get('cost_increase_rate', 0))
    
    if scenario_name in ['S3', 'S4']:
        # 预算增加情景：建设预算增加
        if 'budget' in data:
            data = data.copy()
            data['budget'] = params.get('budget', 140)
    
    return data
'''
    
    output_file = os.path.join(OUTPUT_DIR, 'q4_data.py')
    with open(output_file, 'w', encoding='utf-8-sig') as f:
        f.write(python_data)
    
    print(f"数据文件已保存到: {output_file}")
    print("\n" + "=" * 60)
    print("问题4数据预处理完成！")
    print("=" * 60)

if __name__ == "__main__":
    generate_q4_data()