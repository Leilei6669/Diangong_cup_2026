import numpy as np

def preprocess_population_data():
    """
    预处理老人数量数据，整理为10×3的人口矩阵
    第1列：自理老人
    第2列：半失能老人
    第3列：失能老人
    行表示不同小区（A-J）
    """
    # 从附件1提取的原始数据
    # 小区编号: [自理老人, 半失能老人, 失能老人]
    raw_data = {
        'A': [496, 152, 64],
        'B': [408, 136, 64],
        'C': [632, 208, 80],
        'D': [368, 120, 56],
        'E': [536, 176, 72],
        'F': [328, 104, 40],
        'G': [592, 192, 80],
        'H': [392, 128, 48],
        'I': [504, 168, 64],
        'J': [456, 144, 56]
    }
    
    # 转换为10×3的numpy矩阵
    population_matrix = np.array([
        raw_data['A'],
        raw_data['B'],
        raw_data['C'],
        raw_data['D'],
        raw_data['E'],
        raw_data['F'],
        raw_data['G'],
        raw_data['H'],
        raw_data['I'],
        raw_data['J']
    ])
    
    # 数据检查
    print("=== 数据检查 ===")
    valid = True
    
    # 检查是否为空
    if np.any(population_matrix == 0):
        print("警告：存在老人数量为0的情况")
    
    # 检查是否为负数
    if np.any(population_matrix < 0):
        print("错误：存在老人数量为负数的情况")
        valid = False
    
    # 检查是否为整数
    if not np.all(population_matrix == population_matrix.astype(int)):
        print("错误：存在非整数的老人数量")
        valid = False
    
    # 检查三类老人总数是否与小区老人总数一致
    # 原始数据中60+老人数分别为：712, 608, 920, 544, 784, 472, 864, 568, 736, 656
    total_elderly = np.array([712, 608, 920, 544, 784, 472, 864, 568, 736, 656])
    calculated_total = population_matrix.sum(axis=1)
    
    for i, (calc, expected) in enumerate(zip(calculated_total, total_elderly)):
        if calc != expected:
            print(f"错误：小区{chr(65+i)}的三类老人总数({calc})与60+老人数({expected})不一致")
            valid = False
    
    if valid:
        print("所有数据检查通过！")
    
    return population_matrix

def preprocess_transition_probabilities():
    """
    预处理状态转移概率
    返回10×2的转移概率矩阵
    第1列：自理转半失能概率 p_i,12
    第2列：半失能转失能概率 p_i,23
    """
    print("\n=== 转移概率数据 ===")
    
    # 从附件1提取的转移概率数据
    # 自理 → 半失能：4.50% = 0.045
    # 半失能 → 失能：10% = 0.10
    # 所有小区使用相同的转移概率
    p_12 = 4.50 / 100  # 自理转半失能概率
    p_23 = 10 / 100    # 半失能转失能概率
    
    # 创建10×2的转移概率矩阵（所有小区使用相同概率）
    transition_probs = np.array([
        [p_12, p_23],  # 小区A
        [p_12, p_23],  # 小区B
        [p_12, p_23],  # 小区C
        [p_12, p_23],  # 小区D
        [p_12, p_23],  # 小区E
        [p_12, p_23],  # 小区F
        [p_12, p_23],  # 小区G
        [p_12, p_23],  # 小区H
        [p_12, p_23],  # 小区I
        [p_12, p_23],  # 小区J
    ])
    
    # 数据检查：概率范围[0,1]
    if np.any(transition_probs < 0) or np.any(transition_probs > 1):
        print("警告：转移概率超出[0,1]范围，请检查数据")
        return None
    else:
        print("转移概率数据检查通过！")
    
    return transition_probs

def preprocess_service_demand_matrix():
    """
    预处理服务需求频次矩阵Q（3×6）
    行：老人类型（自理、半失能、失能）
    列：服务类型（助餐、日间照料、上门护理、康复理疗、助浴、紧急救助）
    """
    print("\n=== 服务需求频次矩阵 ===")
    
    # 从附件2提取的需求频次数据（次/月）
    # 行：自理(k=1)、半失能(k=2)、失能(k=3)
    # 列：助餐(s=1)、日间照料(s=2)、上门护理(s=3)、康复理疗(s=4)、助浴(s=5)、紧急救助(s=6)
    Q = np.array([
        [14, 8, 0, 2, 0, 0.15],   # 自理老人
        [20, 14, 6, 4, 2, 1],      # 半失能老人
        [22, 18, 12, 6, 4, 3]      # 失能老人
    ])
    
    # 数据检查：所有元素是否>=0
    if np.any(Q < 0):
        print("警告：服务需求频次存在负数，请检查数据")
        return None
    else:
        print("服务需求频次数据检查通过！")
    
    # 打印需求频次矩阵
    services = ['助餐', '日间照料', '上门护理', '康复理疗', '助浴', '紧急救助']
    elderly_types = ['自理老人', '半失能老人', '失能老人']
    
    print("\n需求频次矩阵Q（次/月）：")
    print("\t" + "\t".join(services))
    for i, elderly in enumerate(elderly_types):
        row_str = elderly + "\t" + "\t".join([f'{q:.2f}' for q in Q[i]])
        print(row_str)
    
    return Q

def preprocess_service_prices():
    """
    预处理服务价格向量A（1×6）
    服务类型：助餐、日间照料、上门护理、康复理疗、助浴、紧急救助
    单位：元/次
    """
    print("\n=== 服务价格数据 ===")
    
    # 从附件2提取的服务价格数据（单次服务营收，元/次）
    # 顺序：助餐、日间照料、上门护理、康复理疗、助浴、紧急救助
    A = np.array([10, 20, 30, 28, 25, 0])
    
    # 数据检查：所有价格是否>=0
    if np.any(A < 0):
        print("警告：服务价格存在负数，请检查数据")
        return None
    else:
        print("服务价格数据检查通过！")
    
    # 打印服务价格向量
    services = ['助餐', '日间照料', '上门护理', '康复理疗', '助浴', '紧急救助']
    print("\n服务价格向量A（元/次）：")
    for i, service in enumerate(services):
        print(f"  {service}: {A[i]}元/次")
    
    # 保存服务价格
    with open('service_prices.csv', 'w', encoding='utf-8') as f:
        f.write('服务类型,单次价格(元/次)\n')
        for i, service in enumerate(services):
            f.write(f'{service},{A[i]}\n')
    print("\n服务价格已保存到 service_prices.csv")
    
    return A

def preprocess_consumption_limits():
    """
    预处理消费上限数据
    返回：
    - U: 三类老人消费上限比例向量（自理20%, 半失能25%, 失能30%）
    - monthly_income: 各小区人均月收入向量（10×1）
    - actual_limits: 各小区各类老人实际消费上限（10×3，单位：元/月）
    """
    print("\n=== 消费上限数据 ===")
    
    # 从附件2提取的消费上限比例
    # 顺序：自理老人、半失能老人、失能老人
    U_ratio = np.array([0.20, 0.25, 0.30])
    
    # 从附件1提取的各小区人均月收入（元）
    # 顺序：小区A-J
    monthly_income = np.array([3400, 3100, 3800, 2900, 3500, 2700, 3600, 3000, 3300, 3200])
    
    # 计算各小区各类老人的实际消费上限（元/月）
    # actual_limits[i,k] = monthly_income[i] * U_ratio[k]
    actual_limits = monthly_income.reshape(-1, 1) @ U_ratio.reshape(1, -1)
    
    # 数据检查：所有消费上限是否>=0
    if np.any(actual_limits < 0):
        print("警告：消费上限存在负数，请检查数据")
        return None, None, None
    else:
        print("消费上限数据检查通过！")
    
    # 打印消费上限信息
    elderly_types = ['自理老人', '半失能老人', '失能老人']
    print("\n消费上限比例U：")
    for i, elderly in enumerate(elderly_types):
        print(f"  {elderly}: {U_ratio[i]*100:.0f}%")
    
    print("\n各小区人均月收入（元）：")
    for i in range(10):
        print(f"  小区{chr(65+i)}: {monthly_income[i]}元")
    
    # 保存消费上限数据
    with open('consumption_limits.csv', 'w', encoding='utf-8') as f:
        f.write('小区编号,人均月收入(元),自理老人消费上限(元),半失能老人消费上限(元),失能老人消费上限(元)\n')
        for i in range(10):
            f.write(f'{chr(65+i)},{monthly_income[i]},{actual_limits[i,0]:.0f},{actual_limits[i,1]:.0f},{actual_limits[i,2]:.0f}\n')
    print("\n消费上限数据已保存到 consumption_limits.csv")
    
    return U_ratio, monthly_income, actual_limits

if __name__ == "__main__":
    # 预处理老人数量数据
    print("=" * 50)
    print("数据预处理")
    print("=" * 50)
    
    print("\n--- Step1: 老人数量数据预处理 ---")
    population_matrix = preprocess_population_data()
    
    print("\n人口矩阵（10×3）：")
    print("小区 | 自理老人 | 半失能老人 | 失能老人")
    print("-----|---------|-----------|---------")
    for i in range(10):
        print(f"  {chr(65+i)}  |    {population_matrix[i,0]:4d}   |    {population_matrix[i,1]:4d}    |   {population_matrix[i,2]:3d}")
    
    # 保存人口矩阵
    with open('population_matrix.csv', 'w', encoding='utf-8') as f:
        f.write('小区编号,自理老人,半失能老人,失能老人\n')
        for i in range(10):
            f.write(f'{chr(65+i)},{population_matrix[i,0]},{population_matrix[i,1]},{population_matrix[i,2]}\n')
    print("\n人口矩阵已保存到 population_matrix.csv")
    
    # Step2: 状态转移概率预处理
    print("\n--- Step2: 状态转移概率预处理 ---")
    transition_matrix = preprocess_transition_probabilities()
    
    if transition_matrix.size > 0:
        with open('transition_matrix.csv', 'w', encoding='utf-8') as f:
            f.write('小区编号,自理转半失能概率p_i12,半失能转失能概率p_i23\n')
            for i in range(10):
                f.write(f'{chr(65+i)},{transition_matrix[i,0]},{transition_matrix[i,1]}\n')
        print("转移概率矩阵已保存到 transition_matrix.csv")
    
    # Step3: 服务需求频次数据预处理
    print("\n--- Step3: 服务需求频次数据预处理 ---")
    demand_matrix = preprocess_service_demand_matrix()
    
    # 保存需求频次矩阵
    services = ['助餐', '日间照料', '上门护理', '康复理疗', '助浴', '紧急救助']
    elderly_types = ['自理老人', '半失能老人', '失能老人']
    with open('demand_matrix.csv', 'w', encoding='utf-8') as f:
        f.write('老人类型,助餐,日间照料,上门护理,康复理疗,助浴,紧急救助\n')
        for i, elderly in enumerate(elderly_types):
            f.write(f'{elderly},{demand_matrix[i,0]},{demand_matrix[i,1]},{demand_matrix[i,2]},{demand_matrix[i,3]},{demand_matrix[i,4]},{demand_matrix[i,5]}\n')
    print("\n需求频次矩阵已保存到 demand_matrix.csv")
    
    # Step4: 服务价格与消费上限预处理
    print("\n--- Step4: 服务价格与消费上限预处理 ---")
    service_prices = preprocess_service_prices()
    U_ratio, monthly_income, actual_limits = preprocess_consumption_limits()
    
    print("\n" + "=" * 50)
    print("所有数据预处理完成！")
    print("=" * 50)