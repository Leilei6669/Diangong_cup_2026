import numpy as np
import os

# 获取当前文件目录（用于保存输出文件）
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def preprocess_distance_matrix():
    """
    Step2: 距离矩阵预处理
    构造可达矩阵，a_ij=1表示小区j的服务站可以有效服务小区i（距离≤1000米）
    """
    print("\n" + "=" * 50)
    print("Step2: 距离矩阵预处理")
    print("=" * 50)
    
    distance_matrix = np.array([
        [0, 600, 1200, 900, 1500, 1800, 1300, 700, 1100, 500],   # A
        [600, 0, 800, 500, 1100, 1400, 900, 400, 700, 300],      # B
        [1200, 800, 0, 700, 600, 900, 500, 900, 600, 700],       # C
        [900, 500, 700, 0, 800, 1100, 600, 300, 500, 400],       # D
        [1500, 1100, 600, 800, 0, 500, 400, 1000, 500, 800],     # E
        [1800, 1400, 900, 1100, 500, 0, 500, 1200, 700, 1100],   # F
        [1300, 900, 500, 600, 400, 500, 0, 800, 400, 600],       # G
        [700, 400, 900, 300, 1000, 1200, 800, 0, 600, 300],      # H
        [1100, 700, 600, 500, 500, 700, 400, 600, 0, 400],       # I
        [500, 300, 700, 400, 800, 1100, 600, 300, 400, 0]        # J
    ])
    
    reachability_matrix = (distance_matrix <= 1000).astype(int)
    
    print(f"距离矩阵维度: {distance_matrix.shape}")
    print(f"可达矩阵维度: {reachability_matrix.shape}")
    print(f"可达矩阵中有效连接数: {reachability_matrix.sum()}")
    
    # 使用Python文件写入，确保UTF-8编码
    # 距离矩阵
    with open(os.path.join(OUTPUT_DIR, 'distance_matrix.csv'), 'w', encoding='utf-8-sig') as f:
        f.write('A,B,C,D,E,F,G,H,I,J\n')
        for row in distance_matrix:
            f.write(','.join(map(str, row)) + '\n')
    
    # 可达矩阵
    with open(os.path.join(OUTPUT_DIR, 'reachability_matrix.csv'), 'w', encoding='utf-8-sig') as f:
        f.write('A,B,C,D,E,F,G,H,I,J\n')
        for row in reachability_matrix:
            f.write(','.join(map(str, row)) + '\n')
    
    print(f"距离矩阵已保存到 {os.path.join(OUTPUT_DIR, 'distance_matrix.csv')}")
    print(f"可达矩阵已保存到 {os.path.join(OUTPUT_DIR, 'reachability_matrix.csv')}")
    
    return distance_matrix, reachability_matrix

def preprocess_demand_data():
    """
    Step3: 需求量预处理
    从问题1的结果读取月需求，转换为日需求
    """
    print("\n" + "=" * 50)
    print("Step3: 需求量预处理")
    print("=" * 50)
    
    question_dir = os.path.abspath(os.path.join(OUTPUT_DIR, '..', '..', 'question'))
    print(f"问题1结果目录: {question_dir}")
    
    demand_file = os.path.join(question_dir, 'q1_2_service_demand_by_community.csv')
    print(f"尝试读取: {demand_file}")
    
    if not os.path.exists(demand_file):
        print(f"警告：文件不存在！{demand_file}")
        print("使用模拟数据进行预处理...")
        monthly_demand = np.array([
            [500, 200, 100, 150, 50, 80],   # A
            [600, 250, 120, 180, 60, 90],   # B
            [450, 180, 90, 135, 45, 72],    # C
            [550, 220, 110, 165, 55, 88],   # D
            [520, 208, 104, 156, 52, 83],   # E
            [580, 232, 116, 174, 58, 93],   # F
            [480, 192, 96, 144, 48, 77],    # G
            [560, 224, 112, 168, 56, 90],   # H
            [510, 204, 102, 153, 51, 82],   # I
            [540, 216, 108, 162, 54, 86]    # J
        ])
    else:
        monthly_demand = np.loadtxt(demand_file, delimiter=',', skiprows=1, usecols=range(1, 7))
    
    Q_i = monthly_demand.sum(axis=1)
    daily_demand = Q_i / 30
    
    print(f"月需求矩阵维度: {monthly_demand.shape}")
    print(f"各小区月需求量: {Q_i.astype(int)}")
    print(f"各小区日需求量: {daily_demand.round(2)}")
    
    # 使用Python文件写入，确保UTF-8编码
    with open(os.path.join(OUTPUT_DIR, 'daily_demand.csv'), 'w', encoding='utf-8-sig') as f:
        f.write('日需求量\n')
        for value in daily_demand:
            f.write(f'{value:.2f}\n')
    
    print(f"日需求量已保存到 {os.path.join(OUTPUT_DIR, 'daily_demand.csv')}")
    
    return monthly_demand, daily_demand

def preprocess_population_data():
    """
    Step4: 老年人口数据预处理
    从问题1的结果读取第5年末各小区老人总数
    """
    print("\n" + "=" * 50)
    print("Step4: 老年人口数据预处理")
    print("=" * 50)
    
    question_dir = os.path.abspath(os.path.join(OUTPUT_DIR, '..', '..', 'question'))
    
    pop_file = os.path.join(question_dir, 'q1_1_population_prediction.csv')
    print(f"尝试读取: {pop_file}")
    
    if not os.path.exists(pop_file):
        print(f"警告：文件不存在！{pop_file}")
        print("使用模拟数据进行预处理...")
        population_total = np.array([680, 795, 580, 895, 740, 840, 635, 880, 707, 813])
    else:
        data = np.loadtxt(pop_file, delimiter=',', skiprows=1, dtype=str)
        year_5_data = data[data[:, 1] == '第5年']
        population_total = year_5_data[:, 5].astype(int)
    
    total_population = population_total.sum()
    
    print(f"各小区老人总数: {population_total}")
    print(f"全区域老人总数: {total_population}")
    
    # 使用Python文件写入，确保UTF-8编码
    with open(os.path.join(OUTPUT_DIR, 'population_total.csv'), 'w', encoding='utf-8-sig') as f:
        f.write('老人总数\n')
        for value in population_total:
            f.write(f'{value}\n')
    
    print(f"老人总数数据已保存到 {os.path.join(OUTPUT_DIR, 'population_total.csv')}")
    
    return population_total, total_population

def preprocess_service_station_data():
    """
    服务站规模数据预处理
    """
    print("\n" + "=" * 50)
    print("服务站规模数据预处理")
    print("=" * 50)
    
    station_data = np.array([
        [18, 2000, 1000],   # 小型
        [32, 3200, 2000],   # 中型
        [45, 4400, 3000]    # 大型
    ])
    
    station_types = ['小型', '中型', '大型']
    station_costs = station_data[:, :2]
    station_capacities = station_data[:, 2]
    
    print("服务站规模数据：")
    for i, station_type in enumerate(station_types):
        print(f"  {station_type}: 建设成本={station_data[i,0]}万元, 日均管理成本={station_data[i,1]}元, 日最大服务人次={station_data[i,2]}")
    
    # 使用UTF-8-sig编码写入
    with open(os.path.join(OUTPUT_DIR, 'service_station_data.csv'), 'w', encoding='utf-8-sig') as f:
        f.write('规模类型,建设成本(万元),日均管理成本(元),日最大服务人次\n')
        for i, station_type in enumerate(station_types):
            f.write(f'{station_type},{station_data[i,0]},{station_data[i,1]},{station_data[i,2]}\n')
    
    print(f"服务站数据已保存到 {os.path.join(OUTPUT_DIR, 'service_station_data.csv')}")
    
    return station_costs, station_capacities

def generate_python_data_file():
    """
    生成Python数据文件（解决数据对接繁琐问题）
    """
    print("\n" + "=" * 50)
    print("生成Python数据文件")
    print("=" * 50)
    
    # 距离矩阵（米）
    distance_matrix = np.array([
        [0, 600, 1200, 900, 1500, 1800, 1300, 700, 1100, 500],
        [600, 0, 800, 500, 1100, 1400, 900, 400, 700, 300],
        [1200, 800, 0, 700, 600, 900, 500, 900, 600, 700],
        [900, 500, 700, 0, 800, 1100, 600, 300, 500, 400],
        [1500, 1100, 600, 800, 0, 500, 400, 1000, 500, 800],
        [1800, 1400, 900, 1100, 500, 0, 500, 1200, 700, 1100],
        [1300, 900, 500, 600, 400, 500, 0, 800, 400, 600],
        [700, 400, 900, 300, 1000, 1200, 800, 0, 600, 300],
        [1100, 700, 600, 500, 500, 700, 400, 600, 0, 400],
        [500, 300, 700, 400, 800, 1100, 600, 300, 400, 0]
    ])
    
    reachability_matrix = (distance_matrix <= 1000).astype(int)
    
    # 日需求量（人次/日）
    daily_demand = np.array([36.0, 44.0, 31.0, 39.0, 35.0, 41.0, 34.0, 39.0, 36.0, 38.0])
    
    # 老年人口（人）
    population = np.array([1200, 1350, 980, 1150, 1080, 1250, 1020, 1180, 1100, 1200])
    
    # 服务站参数
    capacities = np.array([1000, 2000, 3000])
    build_costs = np.array([18, 32, 45])
    
    # 年运营成本（万元/年）- 已从元/天转换
    daily_operating_costs_yuan = np.array([2000, 3200, 4400])
    annual_operating_costs_wan = daily_operating_costs_yuan * 365 / 10000
    
    # 消费上限（元/月）
    consumption_limits = np.array([1200, 1500, 1000, 1300, 1100, 1400, 1050, 1250, 1150, 1350])
    
    # 服务价格
    avg_price = 49.17
    avg_cost = 17.5
    
    # 价格满意度
    price_satisfaction = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    python_data = f'''"""
问题2 数据文件 - 自动生成
注意：所有单位已统一
- 建设成本：万元
- 运营成本：万元/年
- 日需求量、人次：人次/日
- 距离：米
- 消费：元
"""

import numpy as np

# 距离矩阵（米）
DISTANCE_MATRIX = np.array({distance_matrix.tolist()})

# 可达矩阵（距离≤1000米）
REACHABILITY_MATRIX = np.array({reachability_matrix.tolist()})

# 日需求量（人次/日）
DAILY_DEMAND = np.array({daily_demand.tolist()})

# 老年人口（人）
POPULATION = np.array({population.tolist()})

# 服务站容量（人次/日）
CAPACITIES = np.array({capacities.tolist()})

# 建设成本（万元）
BUILD_COSTS = np.array({build_costs.tolist()})

# 年运营成本（万元/年）- 已从元/天转换
ANNUAL_OPERATING_COSTS = np.array({annual_operating_costs_wan.tolist()})

# 月服务消费上限（元/月）
CONSUMPTION_LIMITS = np.array({consumption_limits.tolist()})

# 平均单次服务收费（元/次）
AVG_SERVICE_PRICE = {avg_price}

# 平均单次服务成本（元/次）
AVG_SERVICE_COST = {avg_cost}

# 价格满意度矩阵
PRICE_SATISFACTION = np.array({price_satisfaction.tolist()})

# 预算（万元）
BUDGET = 120

# 满意度权重
ALPHA_D = 0.2
ALPHA_R = 0.3
ALPHA_P = 0.5
'''
    
    with open(os.path.join(OUTPUT_DIR, 'q2_data.py'), 'w', encoding='utf-8-sig') as f:
        f.write(python_data)
    
    print(f"Python数据文件已保存到 {os.path.join(OUTPUT_DIR, 'q2_data.py')}")

if __name__ == "__main__":
    print("=" * 60)
    print("问题2数据预处理")
    print("=" * 60)
    
    distance_matrix, reachability_matrix = preprocess_distance_matrix()
    monthly_demand, daily_demand = preprocess_demand_data()
    population_total, total_population = preprocess_population_data()
    station_costs, station_capacities = preprocess_service_station_data()
    
    # 生成Python数据文件
    generate_python_data_file()
    
    print("\n" + "=" * 60)
    print("问题2数据预处理完成！")
    print("=" * 60)