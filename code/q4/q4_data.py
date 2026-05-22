"""
问题4 数据文件 - 自动生成
功能：扰动情景参数和结果比较数据
"""

import numpy as np

# ========== 1. 扰动情景定义 ==========

SCENARIOS = {'S0': {'name': '基准情景', 'description': '使用问题1、2、3原始参数', 'params': {}}, 'S1': {'name': '老人增长与失能转移增强情景', 'description': 'g=8%, p12=5.5%, p23=9.5%', 'params': {'growth_rate': 0.08, 'p12': 0.055, 'p23': 0.095}}, 'S2': {'name': '成本上升情景', 'description': '日固定管理成本增加20%', 'params': {'cost_increase_rate': 0.2}}, 'S3': {'name': '预算增加情景', 'description': '总建设预算调整为140万元', 'params': {'budget': 140}}, 'S4': {'name': '综合压力情景', 'description': '同时采用老人增长增强、成本上升和预算增加', 'params': {'growth_rate': 0.08, 'p12': 0.055, 'p23': 0.095, 'cost_increase_rate': 0.2, 'budget': 140}}}

# ========== 2. 服务站基础参数 ==========

STATION_BASE_PARAMS = {'小型': {'capacity': 1000, 'build_cost': 18, 'daily_operating': 2000, 'subsidy_limit': 1000}, '中型': {'capacity': 2000, 'build_cost': 32, 'daily_operating': 3200, 'subsidy_limit': 1800}, '大型': {'capacity': 3000, 'build_cost': 45, 'daily_operating': 4400, 'subsidy_limit': 2600}}

# 基准情景服务站配置
BASE_STATIONS = [{'community': 'B', 'size': '中型'}, {'community': 'E', 'size': '小型'}, {'community': 'H', 'size': '大型'}]

# 各情景服务站数据
STATIONS_DATA = {
    'S0': [{'community': 'B', 'size': '中型', 'capacity': 2000, 'build_cost': 32, 'annual_operating': 116.8, 'annual_depreciation': 1.6, 'annual_fixed_cost': 118.39999999999999, 'daily_subsidy_limit': 1800, 'annual_subsidy_limit': 65.7}, {'community': 'E', 'size': '小型', 'capacity': 1000, 'build_cost': 18, 'annual_operating': 73.0, 'annual_depreciation': 0.9, 'annual_fixed_cost': 73.9, 'daily_subsidy_limit': 1000, 'annual_subsidy_limit': 36.5}, {'community': 'H', 'size': '大型', 'capacity': 3000, 'build_cost': 45, 'annual_operating': 160.6, 'annual_depreciation': 2.25, 'annual_fixed_cost': 162.85, 'daily_subsidy_limit': 2600, 'annual_subsidy_limit': 94.9}],
    'S2': [{'community': 'B', 'size': '中型', 'capacity': 2000, 'build_cost': 32, 'annual_operating': 116.8, 'annual_depreciation': 1.6, 'annual_fixed_cost': 142.07999999999998, 'daily_subsidy_limit': 1800, 'annual_subsidy_limit': 65.7}, {'community': 'E', 'size': '小型', 'capacity': 1000, 'build_cost': 18, 'annual_operating': 73.0, 'annual_depreciation': 0.9, 'annual_fixed_cost': 88.68, 'daily_subsidy_limit': 1000, 'annual_subsidy_limit': 36.5}, {'community': 'H', 'size': '大型', 'capacity': 3000, 'build_cost': 45, 'annual_operating': 160.6, 'annual_depreciation': 2.25, 'annual_fixed_cost': 195.42, 'daily_subsidy_limit': 2600, 'annual_subsidy_limit': 94.9}],
    'S3': [{'community': 'B', 'size': '中型', 'capacity': 2000, 'build_cost': 32, 'annual_operating': 116.8, 'annual_depreciation': 1.6, 'annual_fixed_cost': 118.39999999999999, 'daily_subsidy_limit': 1800, 'annual_subsidy_limit': 65.7}, {'community': 'E', 'size': '小型', 'capacity': 1000, 'build_cost': 18, 'annual_operating': 73.0, 'annual_depreciation': 0.9, 'annual_fixed_cost': 73.9, 'daily_subsidy_limit': 1000, 'annual_subsidy_limit': 36.5}, {'community': 'H', 'size': '大型', 'capacity': 3000, 'build_cost': 45, 'annual_operating': 160.6, 'annual_depreciation': 2.25, 'annual_fixed_cost': 162.85, 'daily_subsidy_limit': 2600, 'annual_subsidy_limit': 94.9}],
    'S1': [{'community': 'B', 'size': '中型', 'capacity': 2000, 'build_cost': 32, 'annual_operating': 116.8, 'annual_depreciation': 1.6, 'annual_fixed_cost': 118.39999999999999, 'daily_subsidy_limit': 1800, 'annual_subsidy_limit': 65.7}, {'community': 'E', 'size': '小型', 'capacity': 1000, 'build_cost': 18, 'annual_operating': 73.0, 'annual_depreciation': 0.9, 'annual_fixed_cost': 73.9, 'daily_subsidy_limit': 1000, 'annual_subsidy_limit': 36.5}, {'community': 'H', 'size': '大型', 'capacity': 3000, 'build_cost': 45, 'annual_operating': 160.6, 'annual_depreciation': 2.25, 'annual_fixed_cost': 162.85, 'daily_subsidy_limit': 2600, 'annual_subsidy_limit': 94.9}],  # S1只影响人口结构，不影响服务站
    'S4': [{'community': 'B', 'size': '中型', 'capacity': 2000, 'build_cost': 32, 'annual_operating': 116.8, 'annual_depreciation': 1.6, 'annual_fixed_cost': 142.07999999999998, 'daily_subsidy_limit': 1800, 'annual_subsidy_limit': 65.7}, {'community': 'E', 'size': '小型', 'capacity': 1000, 'build_cost': 18, 'annual_operating': 73.0, 'annual_depreciation': 0.9, 'annual_fixed_cost': 88.68, 'daily_subsidy_limit': 1000, 'annual_subsidy_limit': 36.5}, {'community': 'H', 'size': '大型', 'capacity': 3000, 'build_cost': 45, 'annual_operating': 160.6, 'annual_depreciation': 2.25, 'annual_fixed_cost': 195.42, 'daily_subsidy_limit': 2600, 'annual_subsidy_limit': 94.9}]      # S4与S2服务站成本相同
}

# ========== 3. 问题2、3关键参数 ==========

Q2_Q3_PARAMS = {'BUDGET': 120, 'BUDGET_S3': 140, 'growth_rate_base': 0.05, 'growth_rate_S1': 0.08, 'p12_base': 0.04, 'p12_S1': 0.055, 'p23_base': 0.08, 'p23_S1': 0.095, 'cost_increase_rate': 0.2}

# 基准预算（万元）
BUDGET_BASE = 120

# S3情景预算（万元）
BUDGET_S3 = 140

# 人口增长参数
GROWTH_RATE_BASE = 0.05
GROWTH_RATE_S1 = 0.08

# 失能转移概率
P12_BASE = 0.04
P12_S1 = 0.055
P23_BASE = 0.08
P23_S1 = 0.095

# 成本变化参数
COST_INCREASE_RATE = 0.2

# ========== 4. 结果比较指标 ==========

COMPARISON_METRICS = {'问题2_选址结果': ['服务站位置', '服务站规模', '服务站数量', '总建设成本', '覆盖率'], '问题3_定价结果': ['各服务站价格方案', '各类服务价格', '总满意度', '平均价格满意度'], '问题4_情景分析': {'S0': ['基准情景满意度', '基准情景利润', '基准情景补贴'], 'S1': ['老人增长情景满意度', '老人增长情景利润', '老人增长情景补贴'], 'S2': ['成本上升情景满意度', '成本上升情景利润', '成本上升情景补贴'], 'S3': ['预算增加情景满意度', '预算增加情景利润', '预算增加情景补贴'], 'S4': ['综合压力情景满意度', '综合压力情景利润', '综合压力情景补贴']}}

# ========== 5. 数据一致性检查标准 ==========

DATA_CONSISTENCY = {
    '人口数量单位': '人',
    '服务需求单位': '次/月',
    '服务能力单位': '人次/日',
    '建设成本单位': '万元',
    '运营成本单位': '元/日 或 万元/年',
    '补贴单位': '元/年 或 万元/年',
    '满意度范围': [0.6, 1.0],
    '距离矩阵': '保持不变'
}

# ========== 6. 情景参数获取函数 ==========

def get_scenario_params(scenario_name):
    """
    获取指定情景的参数
    
    参数:
        scenario_name: 情景名称 ('S0', 'S1', 'S2', 'S3', 'S4')
    
    返回:
        dict: 情景参数字典
    """
    return SCENARIOS.get(scenario_name, SCENARIOS['S0'])

def get_stations_data(scenario_name):
    """
    获取指定情景的服务站数据
    
    参数:
        scenario_name: 情景名称
    
    返回:
        list: 服务站数据列表
    """
    return STATIONS_DATA.get(scenario_name, STATIONS_DATA['S0'])

def apply_scenario_adjustments(scenario_name, data):
    """
    对基准数据进行情景调整
    
    参数:
        scenario_name: 情景名称
        data: 基准数据
    
    返回:
        调整后的数据
    """
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
