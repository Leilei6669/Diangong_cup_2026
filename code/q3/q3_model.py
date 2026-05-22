import numpy as np
import random
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 配置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置路径并导入数据
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data_preprocessing', 'q3')))
from q3_data import *

class PricingOptimizer:
    def __init__(self):
        # 固定随机种子，保证结果可复现
        random.seed(42)
        np.random.seed(42)
        
        # ========== 基础数据赋值 ==========
        self.communities = COMMUNITIES
        self.care_types = CARE_TYPES
        self.services = SERVICES
        
        # 老人数量与需求数据
        self.N_ir = N_ir  # 各小区各类型老人数量 (人)
        self.a_rm = a_rm  # 各类老人月均需求次数 (次/月/人)
        self.D_irm_0 = D_irm_0  # 理论月需求 (次/月)
        
        # 服务价格与成本
        self.P_M_0 = P_M_0  # 基准价格 (元/次)
        self.C_M = C_M      # 单位直接成本 (元/次)
        
        # 服务站数据
        self.stations = STATIONS
        self.station_capacities = STATION_CAPACITIES  # 服务容量 (人次/日)
        self.station_fixed_costs = STATION_FIXED_COSTS  # 年固定成本 (万元/年)
        self.station_subsidy_limits = STATION_SUBSIDY_LIMITS  # 年补贴上限 (万元/年)
        
        # 政府补贴参数
        self.subsidy_per_service = SUBSIDY_PER_SERVICE  # 单次补贴 (元/人次)
        self.subsidized_services = SUBSIDIZED_SERVICES  # 是否享受补贴
        
        # 满意度权重
        self.alpha_d = ALPHA_D  # 距离满意度权重 (0.2)
        self.alpha_r = ALPHA_R  # 响应满意度权重 (0.3)
        self.alpha_p = ALPHA_P  # 价格满意度权重 (0.5)
        
        # 经营约束参数
        self.max_profit_rate = MAX_PROFIT_RATE  # 利润率上限 (8%)
        self.discount_rate = DISCOUNT_RATE
        
        # ========== 距离矩阵 ==========
        self.distance_matrix = np.array([
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
        
        # 服务站所在小区索引
        self.station_indices = []
        for station in self.stations:
            self.station_indices.append(self.communities.index(station['community']))
        
        # ========== 消费能力参数 ==========
        # 各小区人均月收入 (元)
        self.y_i = np.array([3500, 4200, 3800, 4000, 3600, 4500, 3900, 4300, 4100, 3700])
        # 各类老人服务消费比例 (自理:10%, 半失能:15%, 失能:20%)
        self.theta_r = np.array([0.10, 0.15, 0.20])
        
        # ========== 遗传算法参数 ==========
        self.pop_size = 100      # 种群规模
        self.generations = 200   # 迭代代数
        self.mutation_rate = 0.10  # 变异概率
        self.crossover_rate = 0.7  # 交叉概率
        
        print(f"数据加载完成！服务站数量: {len(self.stations)}")
        print(f"服务站位置: {[s['community'] for s in self.stations]}")
    
    def calc_distance_satisfaction(self, d_ij):
        """
        计算距离满意度 S1
        距离越近，满意度越高
        """
        if d_ij <= 300:
            return 1.00
        elif d_ij <= 500:
            return 0.90
        elif d_ij <= 650:
            return 0.75
        elif d_ij <= 1000:
            return 0.60
        else:
            return 0.0
    
    def calc_price_satisfaction(self, p_jm, p_m0):
        """
        计算价格满意度 S3
        价格越接近基准价，满意度越高
        紧急救助公益免费，满意度恒为1.0
        """
        if p_m0 == 0:  # 紧急救助
            return 1.0
        if p_jm <= p_m0:
            return 1.00
        elif p_jm <= 1.1 * p_m0:
            return 0.90
        elif p_jm <= 1.2 * p_m0:
            return 0.75
        else:
            return 0.60
    
    def calc_response_satisfaction(self, utilization):
        """
        计算响应满意度 S2
        服务站利用率越低，响应速度越快，满意度越高
        """
        if utilization <= 0.60:
            return 1.00
        elif utilization <= 0.75:
            return 0.93
        elif utilization <= 0.85:
            return 0.85
        elif utilization <= 0.95:
            return 0.72
        elif utilization <= 1.0:
            return 0.60
        else:
            return 0.60  # 超过容量时满意度最低
    
    def calc_consumption_limit(self, i, r):
        """
        计算第i小区第r类老人的月服务消费上限
        B_ir = theta_r * y_i
        """
        return self.theta_r[r] * self.y_i[i]
    
    def decode_chromosome(self, chromosome):
        """
        解码染色体为价格矩阵
        染色体编码：每个服务站的每项服务价格(0-100映射到0-50元)
        """
        n_stations = len(self.stations)
        n_services = len(self.services)
        
        prices = np.zeros((n_stations, n_services))
        idx = 0
        
        for j in range(n_stations):
            for m in range(n_services):
                # 紧急救助价格固定为0
                if m == 5:
                    prices[j, m] = 0
                else:
                    # 0-100映射到0-50元
                    prices[j, m] = chromosome[idx] / 2.0
                idx += 1
        
        return prices
    
    def generate_individual(self):
        """
        生成随机个体（价格方案）
        价格范围：0 ~ 基准价格的1.5倍
        """
        n_stations = len(self.stations)
        n_services = len(self.services)
        
        chromosome = []
        for j in range(n_stations):
            for m in range(n_services):
                if m == 5:  # 紧急救助价格固定为0
                    chromosome.append(0)
                else:
                    # 价格范围：0 ~ 基准价格的1.5倍
                    max_price = int(round(1.5 * self.P_M_0[m] * 2))
                    chromosome.append(random.randint(0, max_price))
        
        return chromosome
    
    def calculate_utilization(self, prices):
        station_utilization = np.zeros(len(self.stations))
        
        for j in range(len(self.stations)):
            daily_demand = 0
            for i in range(len(self.communities)):
                if self.distance_matrix[i, self.station_indices[j]] <= 1000:
                    for r in range(len(self.care_types)):
                        P_ijr_need = sum(self.a_rm[r, m] * prices[j, m] for m in range(len(self.services)))
                        B_ir = self.calc_consumption_limit(i, r)
                        lambda_ijr = min(1.0, B_ir / max(P_ijr_need, 1))
                        
                        for m in range(len(self.services)):
                            daily_demand += self.D_irm_0[i, r, m] * lambda_ijr / 30  # 日均
            
            if self.station_capacities[j] > 0:
                station_utilization[j] = min(daily_demand / self.station_capacities[j], 1.0)
            else:
                station_utilization[j] = 0
        
        return station_utilization
    
    def fitness(self, chromosome):
        """
        计算适应度（加权平均满意度）
        目标：最大化满意度，同时满足利润率和容量约束
        """
        prices = self.decode_chromosome(chromosome)
        
        # 计算服务站利用率
        station_utilization = self.calculate_utilization(prices)
        
        # 计算总满意度
        total_satisfaction = 0.0
        total_weight = 0.0
        
        for i in range(len(self.communities)):
            # 找到可达服务站
            reachable_stations = []
            for j, st_idx in enumerate(self.station_indices):
                if self.distance_matrix[i, st_idx] <= 1000:
                    reachable_stations.append(j)
            
            if not reachable_stations:
                continue
            
            # 选择综合满意度最高的服务站
            best_j = -1
            best_satisfaction = -1
            
            for j in reachable_stations:
                s1 = self.calc_distance_satisfaction(self.distance_matrix[i, self.station_indices[j]])
                s2 = self.calc_response_satisfaction(station_utilization[j])
                
                # 计算该服务站的平均价格满意度
                s3_sum = sum(self.calc_price_satisfaction(prices[j, m], self.P_M_0[m]) for m in range(len(self.services)))
                s3_avg = s3_sum / len(self.services)
                
                # 综合满意度
                s_total = self.alpha_d * s1 + self.alpha_r * s2 + self.alpha_p * s3_avg
                
                if s_total > best_satisfaction:
                    best_satisfaction = s_total
                    best_j = j
            
            if best_j == -1:
                continue
            
            # 计算消费能力修正后的有效需求
            for r in range(len(self.care_types)):
                P_ijr_need = sum(self.a_rm[r, m] * prices[best_j, m] for m in range(len(self.services)))
                B_ir = self.calc_consumption_limit(i, r)
                lambda_ijr = min(1.0, B_ir / max(P_ijr_need, 1))
                
                for m in range(len(self.services)):
                    D_irm = self.D_irm_0[i, r, m]
                    effective_demand = D_irm * lambda_ijr
                    
                    if effective_demand > 0:
                        # 计算该项服务的综合满意度
                        s1 = self.calc_distance_satisfaction(self.distance_matrix[i, self.station_indices[best_j]])
                        s2 = self.calc_response_satisfaction(station_utilization[best_j])
                        s3 = self.calc_price_satisfaction(prices[best_j, m], self.P_M_0[m])
                        
                        s_total = self.alpha_d * s1 + self.alpha_r * s2 + self.alpha_p * s3
                        
                        total_satisfaction += effective_demand * s_total
                        total_weight += effective_demand
        
        # 计算利润约束惩罚
        profit_penalty = self.calculate_profit_penalty(prices)
        
        # 计算容量约束惩罚
        capacity_penalty = 0
        for j in range(len(self.stations)):
            if station_utilization[j] > 1.0:
                capacity_penalty += (station_utilization[j] - 1.0) * 1000
        
        # 计算最终适应度
        if total_weight > 0:
            fitness = (total_satisfaction / total_weight) - profit_penalty - capacity_penalty
        else:
            fitness = 0
        
        return max(fitness, 0.5)  # 最小适应度为0.5
    
    def calculate_profit_penalty(self, prices):
        """
        计算利润约束惩罚
        惩罚项：利润率超过8%或亏损
        """
        profit_penalty = 0
        
        for j in range(len(self.stations)):
            # 计算年度收入、成本和补贴
            annual_revenue = 0
            annual_cost = 0
            annual_subsidy = 0
            
            for i in range(len(self.communities)):
                if self.distance_matrix[i, self.station_indices[j]] <= 1000:
                    for r in range(len(self.care_types)):
                        P_ijr_need = sum(self.a_rm[r, m] * prices[j, m] for m in range(len(self.services)))
                        B_ir = self.calc_consumption_limit(i, r)
                        lambda_ijr = min(1.0, B_ir / max(P_ijr_need, 1))
                        
                        for m in range(len(self.services)):
                            D_irm = self.D_irm_0[i, r, m]
                            Q_ijrm = D_irm * lambda_ijr
                            
                            annual_revenue += 12 * prices[j, m] * Q_ijrm / 10000  # 万元
                            annual_cost += 12 * self.C_M[m] * Q_ijrm / 10000      # 万元
                            
                            if self.subsidized_services[m]:
                                annual_subsidy += 12 * self.subsidy_per_service * Q_ijrm / 10000  # 万元
            
            # 补贴不超过上限
            annual_subsidy = min(annual_subsidy, self.station_subsidy_limits[j])
            
            # 计算利润和利润率
            profit = annual_revenue - annual_cost + annual_subsidy - self.station_fixed_costs[j]
            
            if self.station_fixed_costs[j] > 0:
                profit_rate = profit / self.station_fixed_costs[j]
            else:
                profit_rate = 0
            
            # 惩罚：利润率超过8%或亏损
            if profit_rate > self.max_profit_rate:
                profit_penalty += (profit_rate - self.max_profit_rate) * 1000
            if profit < 0:
                profit_penalty += abs(profit) * 100
        
        return profit_penalty
    
    def select(self, population, fitness_scores):
        """锦标赛选择"""
        tournament_size = 3
        selected = []
        for _ in range(len(population)):
            candidates = random.sample(list(enumerate(fitness_scores)), tournament_size)
            candidates.sort(key=lambda x: x[1], reverse=True)
            selected.append(population[candidates[0][0]])
        return selected
    
    def crossover(self, parent1, parent2):
        """单点交叉"""
        if random.random() < self.crossover_rate:
            point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return parent1[:], parent2[:]
    
    def mutate(self, chromosome):
        """变异操作"""
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                # 紧急救助价格固定为0，不变异
                if i % len(self.services) != 5:
                    chromosome[i] = random.randint(0, 100)
        return chromosome
    
    def genetic_algorithm(self):
        """遗传算法主函数"""
        print("\n" + "=" * 60)
        print("开始遗传算法优化（问题3：服务定价优化）")
        print("=" * 60)
        
        # 初始化种群
        population = [self.generate_individual() for _ in range(self.pop_size)]
        best_fitness = 0
        best_chromosome = None
        self.convergence_history = []
        
        # 迭代优化
        for generation in range(self.generations):
            # 计算适应度
            fitness_scores = [self.fitness(ind) for ind in population]
            
            # 更新最优解
            max_fitness = max(fitness_scores)
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_idx = fitness_scores.index(max_fitness)
                best_chromosome = population[best_idx][:]
            
            # 记录历史
            self.convergence_history.append(best_fitness)
            
            # 输出进度
            if generation % 10 == 0:
                print(f"第{generation}代 | 最优适应度: {best_fitness:.4f}")
            
            # 选择
            selected = self.select(population, fitness_scores)
            
            # 交叉
            new_population = [selected[0]]  # 精英保留
            for i in range(1, len(selected), 2):
                if i + 1 < len(selected):
                    child1, child2 = self.crossover(selected[i], selected[i+1])
                    new_population.append(child1)
                    new_population.append(child2)
            
            # 变异
            for i in range(1, len(new_population)):
                new_population[i] = self.mutate(new_population[i])
            
            population = new_population
        
        print(f"\n遗传算法完成！最优适应度: {best_fitness:.4f}")
        
        # 绘制收敛曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.convergence_history)), self.convergence_history, linewidth=2, color='#1f77b4')
        plt.xlabel('迭代次数', fontsize=12)
        plt.ylabel('最优适应度', fontsize=12)
        plt.title('问题3遗传算法收敛曲线', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(os.path.dirname(__file__), 'q3_convergence.png'), 
                    dpi=300, bbox_inches='tight')
        print("收敛曲线已保存为 q3_convergence.png")
        
        return best_chromosome
    
    def evaluate_solution(self, chromosome):
        """评估最优解，输出详细结果"""
        prices = self.decode_chromosome(chromosome)
        
        print("\n" + "=" * 60)
        print("问题3最优解评估")
        print("=" * 60)
        
        # ========== 输出定价方案 ==========
        print("\n【各服务站定价方案】（单位：元/次）")
        print("-" * 80)
        print(f"{'服务站':<8} {'规模':<6} {'小区':<6}", end='')
        for service in self.services:
            print(f"{service:<10}", end='')
        print()
        print("-" * 80)
        
        for j, station in enumerate(self.stations):
            print(f"{j+1:<8} {station['size']:<6} {station['community']:<6}", end='')
            for m, service in enumerate(self.services):
                if service == '紧急救助':
                    print(f"{int(prices[j, m]):<10d}", end='')
                else:
                    print(f"{prices[j, m]:<10.2f}", end='')
            print()
        
        # ========== 输出利润分析 ==========
        print("\n【各服务站年度利润分析】（单位：万元）")
        print("-" * 90)
        print(f"{'服务站':<8} {'收入':<10} {'直接成本':<10} {'补贴':<10} {'固定成本':<10} {'利润':<10} {'利润率':<10}")
        print("-" * 90)
        
        total_profit = 0
        for j, station in enumerate(self.stations):
            annual_revenue, annual_cost, annual_subsidy = self.calculate_station_finance(j, prices)
            profit = annual_revenue - annual_cost + annual_subsidy - self.station_fixed_costs[j]
            profit_rate = profit / self.station_fixed_costs[j] * 100 if self.station_fixed_costs[j] > 0 else 0
            
            total_profit += profit
            
            print(f"{station['community']:<8} {annual_revenue:<10.2f} {annual_cost:<10.2f} "
                  f"{annual_subsidy:<10.2f} {self.station_fixed_costs[j]:<10.2f} "
                  f"{profit:<10.2f} {profit_rate:<10.2f}%")
        
        print("-" * 90)
        print(f"{'合计':<8} {'':<10} {'':<10} {'':<10} {'':<10} {total_profit:<10.2f}")
        
        # ========== 输出满意度分析 ==========
        print("\n【各小区满意度分析】")
        print("-" * 70)
        print(f"{'小区':<6} {'综合满意度':<12} {'距离满意度':<12} {'响应满意度':<12} {'价格满意度':<12}")
        print("-" * 70)
        
        # 计算服务站利用率
        station_utilization = self.calculate_utilization(prices)
        
        # 初始化满意度数组
        satisfaction_total = np.zeros(10)
        satisfaction_distance = np.zeros(10)
        satisfaction_response = np.zeros(10)
        satisfaction_price = np.zeros(10)
        
        total_satisfaction = 0
        total_weight = 0
        
        for i, community in enumerate(self.communities):
            # 找到可达服务站
            reachable_stations = []
            for j, st_idx in enumerate(self.station_indices):
                if self.distance_matrix[i, st_idx] <= 1000:
                    reachable_stations.append(j)
            
            if not reachable_stations:
                print(f"{community:<6} {'无服务站':<12} {'-':<12} {'-':<12} {'-':<12}")
                continue
            
            # 选择最优服务站
            best_j = -1
            best_s = -1
            s1_best = 0
            s2_best = 0
            s3_best = 0
            
            for j in reachable_stations:
                s1 = self.calc_distance_satisfaction(self.distance_matrix[i, self.station_indices[j]])
                s2 = self.calc_response_satisfaction(station_utilization[j])
                
                s3_sum = sum(self.calc_price_satisfaction(prices[j, m], self.P_M_0[m]) for m in range(len(self.services)))
                s3_avg = s3_sum / len(self.services)
                
                s_total = self.alpha_d * s1 + self.alpha_r * s2 + self.alpha_p * s3_avg
                
                if s_total > best_s:
                    best_s = s_total
                    best_j = j
                    s1_best = s1
                    s2_best = s2
                    s3_best = s3_avg
            
            # 计算权重
            weight = sum(self.D_irm_0[i, r, m] for r in range(len(self.care_types)) for m in range(len(self.services)))
            total_satisfaction += best_s * weight
            total_weight += weight
            
            # 保存满意度数据
            satisfaction_total[i] = best_s
            satisfaction_distance[i] = s1_best
            satisfaction_response[i] = s2_best
            satisfaction_price[i] = s3_best
            
            print(f"{community:<6} {best_s:<12.4f} {s1_best:<12.4f} {s2_best:<12.4f} {s3_best:<12.4f}")
        
        overall_satisfaction = total_satisfaction / total_weight if total_weight > 0 else 0
        print("-" * 70)
        print(f"\n总体满意度: {overall_satisfaction:.4f}")
        
        # ========== 3.3 可及性分析 ==========
        accessibility_result = self.analyze_accessibility(prices, station_utilization)
        
        # 收集利润信息
        profit_info = []
        for j, station in enumerate(self.stations):
            annual_revenue, annual_cost, annual_subsidy = self.calculate_station_finance(j, prices)
            profit = annual_revenue - annual_cost + annual_subsidy - self.station_fixed_costs[j]
            profit_rate = profit / (annual_revenue + annual_subsidy) * 100 if (annual_revenue + annual_subsidy) > 0 else 0
            profit_info.append({
                'station': station['community'],
                'profit': profit,
                'profit_rate': profit_rate / 100,
                'subsidy': annual_subsidy
            })
        
        # 返回完整结果
        solution = {
            'prices': prices,
            'satisfaction_total': satisfaction_total,
            'satisfaction_distance': satisfaction_distance,
            'satisfaction_response': satisfaction_response,
            'satisfaction_price': satisfaction_price,
            'overall_satisfaction': overall_satisfaction,
            'accessibility': accessibility_result,
            'profit_info': profit_info
        }
        
        return solution
    
    def analyze_accessibility(self, prices, station_utilization):
        """
        分析定价与补贴对不同类型老人服务可及性的影响
        可及性维度：经济可及性、地理可及性、服务可及性
        """
        print("\n【3.3 可及性分析】")
        print("=" * 60)
        
        # 初始化统计数据
        access_stats = {
            '自理': {'total': 0, 'accessible': 0, 'economic_access': 0, 'geographic_access': 0},
            '半失能': {'total': 0, 'accessible': 0, 'economic_access': 0, 'geographic_access': 0},
            '失能': {'total': 0, 'accessible': 0, 'economic_access': 0, 'geographic_access': 0}
        }
        
        # 遍历每个小区和每种老人类型
        for i in range(len(self.communities)):
            for r in range(len(self.care_types)):
                care_type = self.care_types[r]
                access_stats[care_type]['total'] += self.N_ir[i, r]
                
                # 地理可及性
                has_geo_access = False
                for j, st_idx in enumerate(self.station_indices):
                    if self.distance_matrix[i, st_idx] <= 1000:
                        has_geo_access = True
                        break
                
                if has_geo_access:
                    access_stats[care_type]['geographic_access'] += self.N_ir[i, r]
                    
                    # 找到最优服务站
                    best_j = -1
                    best_satisfaction = -1
                    for j, st_idx in enumerate(self.station_indices):
                        if self.distance_matrix[i, st_idx] <= 1000:
                            s1 = self.calc_distance_satisfaction(self.distance_matrix[i, st_idx])
                            s2 = self.calc_response_satisfaction(station_utilization[j])
                            
                            s3_sum = sum(self.calc_price_satisfaction(prices[j, m], self.P_M_0[m]) for m in range(len(self.services)))
                            s3_avg = s3_sum / len(self.services)
                            
                            s_total = self.alpha_d * s1 + self.alpha_r * s2 + self.alpha_p * s3_avg
                            
                            if s_total > best_satisfaction:
                                best_satisfaction = s_total
                                best_j = j
                    
                    if best_j != -1:
                        # 经济可及性
                        P_ijr_need = sum(self.a_rm[r, m] * prices[best_j, m] for m in range(len(self.services)))
                        B_ir = self.calc_consumption_limit(i, r)
                        
                        if P_ijr_need <= B_ir:
                            access_stats[care_type]['economic_access'] += self.N_ir[i, r]
                            
                            # 服务可及性
                            if station_utilization[best_j] <= 1.0:
                                access_stats[care_type]['accessible'] += self.N_ir[i, r]
        
        # 输出可及性统计
        print("\n【各类老人可及性统计】")
        print("-" * 80)
        print(f"{'老人类型':<8} {'总人数':<10} {'地理可达':<10} {'经济可及':<10} {'综合可及':<10}")
        print("-" * 80)
        
        for care_type in self.care_types:
            stats = access_stats[care_type]
            print(f"{care_type:<8} {stats['total']:<10} {stats['geographic_access']:<10} "
                  f"{stats['economic_access']:<10} {stats['accessible']:<10}")
        
        # 输出可及性比例
        print("\n【各类老人可及性比例】")
        print("-" * 80)
        print(f"{'老人类型':<8} {'地理可达率':<12} {'经济可及率':<12} {'综合可及率':<12}")
        print("-" * 80)
        
        for care_type in self.care_types:
            stats = access_stats[care_type]
            if stats['total'] > 0:
                geo_rate = stats['geographic_access'] / stats['total'] * 100
                eco_rate = stats['economic_access'] / stats['total'] * 100
                acc_rate = stats['accessible'] / stats['total'] * 100
                
                print(f"{care_type:<8} {geo_rate:<12.2f}% {eco_rate:<12.2f}% {acc_rate:<12.2f}%")
            else:
                print(f"{care_type:<8} {'-':<12} {'-':<12} {'-':<12}")
        
        # 经济负担分析
        print("\n【经济负担分析】")
        print("-" * 60)
        for r, care_type in enumerate(self.care_types):
            avg_monthly_cost = 0
            count = 0
            for i in range(len(self.communities)):
                for j, st_idx in enumerate(self.station_indices):
                    if self.distance_matrix[i, st_idx] <= 1000:
                        P_ijr_need = sum(self.a_rm[r, m] * prices[j, m] for m in range(len(self.services)))
                        avg_monthly_cost += P_ijr_need
                        count += 1
            
            if count > 0:
                avg_monthly_cost /= count
                required_income = avg_monthly_cost / self.theta_r[r]
                print(f"{care_type}老人：月均服务费用 {avg_monthly_cost:.2f}元，需月收入 {required_income:.0f}元以上")
        
        # 补贴效果分析
        print("\n【补贴效果分析】")
        print("-" * 60)
        total_subsidy = 0
        for j in range(len(self.stations)):
            _, _, subsidy = self.calculate_station_finance(j, prices)
            total_subsidy += subsidy
        
        print(f"年度总补贴金额：{total_subsidy:.2f}万元")
        print(f"单次服务补贴：{self.subsidy_per_service}元/人次")
        
        # 计算可及率
        access_result = {}
        for care_type in self.care_types:
            stats = access_stats[care_type]
            if stats['total'] > 0:
                access_result[care_type] = {
                    'geographic_access': stats['geographic_access'] / stats['total'],
                    'economic_access': stats['economic_access'] / stats['total'],
                    'total_access': stats['accessible'] / stats['total']
                }
            else:
                access_result[care_type] = {
                    'geographic_access': 0,
                    'economic_access': 0,
                    'total_access': 0
                }
        
        return access_result
    
    def calculate_station_finance(self, j, prices):
        """计算单个服务站的财务数据"""
        annual_revenue = 0
        annual_cost = 0
        annual_subsidy = 0
        
        for i in range(len(self.communities)):
            if self.distance_matrix[i, self.station_indices[j]] <= 1000:
                for r in range(len(self.care_types)):
                    P_ijr_need = sum(self.a_rm[r, m] * prices[j, m] for m in range(len(self.services)))
                    B_ir = self.calc_consumption_limit(i, r)
                    lambda_ijr = min(1.0, B_ir / max(P_ijr_need, 1))
                    
                    for m in range(len(self.services)):
                        D_irm = self.D_irm_0[i, r, m]
                        Q_ijrm = D_irm * lambda_ijr
                        
                        annual_revenue += 12 * prices[j, m] * Q_ijrm / 10000
                        annual_cost += 12 * self.C_M[m] * Q_ijrm / 10000
                        
                        if self.subsidized_services[m]:
                            annual_subsidy += 12 * self.subsidy_per_service * Q_ijrm / 10000
        
        # 补贴上限
        annual_subsidy = min(annual_subsidy, self.station_subsidy_limits[j])
        
        return annual_revenue, annual_cost, annual_subsidy

# ========== 可视化模块 ==========
    def create_visualizations(self, chromosome, solution):
        """
        创建问题3的可视化图表（13张图）
        
        参数:
            chromosome: 最优染色体
            solution: 求解结果字典
        """
        print("\n" + "=" * 60)
        print("开始生成可视化图表...")
        print("=" * 60)
        
        output_dir = os.path.dirname(os.path.abspath(__file__))
        
        # ===== 定价方案可视化 =====
        print("\n【定价方案可视化】")
        
        # 图1：各服务站定价与基准价对比图
        self.plot_pricing_comparison(solution, output_dir)
        
        # 图2：各类服务平均定价对比图
        self.plot_service_avg_price(solution, output_dir)
        
        # 图3：服务站-服务类型定价热力图
        self.plot_pricing_heatmap(solution, output_dir)
        
        # ===== 满意度分析可视化 =====
        print("\n【满意度分析可视化】")
        
        # 图4：各小区满意度分解堆叠柱状图
        self.plot_satisfaction_breakdown(solution, output_dir)
        
        # 图5：价格满意度与综合满意度散点图
        self.plot_price_satisfaction_scatter(solution, output_dir)
        
        # 图6：不同类型老人满意度对比图
        self.plot_satisfaction_by_care_type(solution, output_dir)
        
        # ===== 可及性分析可视化 =====
        print("\n【可及性分析可视化】")
        
        # 图7：不同类型老人可及性对比图
        self.plot_accessibility_by_care_type(solution, output_dir)
        
        # 图8：老人月服务费用与消费上限对比图
        self.plot_cost_vs_limit(solution, output_dir)
        
        # 图9：小区-老人类型经济可及性热力图
        self.plot_accessibility_heatmap(solution, output_dir)
        
        # ===== 经济与补贴分析可视化 =====
        print("\n【经济与补贴分析可视化】")
        
        # 图10：各服务站利润与利润率双轴图
        self.plot_profit_vs_rate(solution, output_dir)
        
        # 图11：各服务站补贴使用情况对比图
        self.plot_subsidy_comparison(solution, output_dir)
        
        # ===== 算法与对比分析可视化 =====
        print("\n【算法与对比分析可视化】")
        
        # 图12：遗传算法收敛曲线
        self.plot_convergence_curve(output_dir)
        
        # 图13：有补贴与无补贴方案对比图
        self.plot_subsidy_effect_comparison(solution, output_dir)
        
        print("\n" + "=" * 60)
        print(f"所有图表已保存到: {output_dir}")
        print("=" * 60)
    
    def plot_pricing_comparison(self, solution, output_dir):
        """图1：各服务站定价与基准价对比图"""
        prices = solution['prices']
        
        fig, axes = plt.subplots(len(self.station_indices), 1, figsize=(12, 8))
        if len(self.station_indices) == 1:
            axes = [axes]
        
        for idx, j in enumerate(self.station_indices):
            ax = axes[idx]
            x = np.arange(6)
            width = 0.35
            
            base_prices = self.P_M_0
            opt_prices = prices[idx, :]  # 使用服务站索引，不是小区索引
            
            bars1 = ax.bar(x - width/2, base_prices, width, label='基准价', color='#9E9E9E', alpha=0.7)
            bars2 = ax.bar(x + width/2, opt_prices, width, label='最优定价', color='#2196F3', alpha=0.8)
            
            for i, (b, o) in enumerate(zip(base_prices, opt_prices)):
                if self.services[i] == '紧急救助':
                    ax.text(i, max(b, o) + 2, '免费', ha='center', fontsize=8, color='red')
                else:
                    diff = ((o - b) / b * 100) if b != 0 else 0
                    ax.text(i, max(b, o) + 2, f'{diff:+.1f}%', ha='center', fontsize=8)
            
            ax.set_title(f'{self.communities[j]}服务站定价对比', fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(self.services, fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'q3_fig1_pricing_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: q3_fig1_pricing_comparison.png")
    
    def plot_service_avg_price(self, solution, output_dir):
        """图2：各类服务平均定价对比图"""
        prices = solution['prices']
        
        avg_prices = np.mean(prices, axis=0)  # prices已经是服务站定价数组
        base_avg = np.mean(self.P_M_0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(self.services, avg_prices, color=plt.cm.Blues(np.linspace(0.4, 0.8, 6)))
        
        for bar, price in zip(bars, avg_prices):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   f'{price:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.axhline(y=base_avg, color='red', linestyle='--', linewidth=1.5, label=f'基准均价: {base_avg:.1f}')
        
        ax.set_xlabel('服务类型', fontsize=12)
        ax.set_ylabel('平均定价（元/次）', fontsize=12)
        ax.set_title('各类服务平均最优定价对比图', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'q3_fig2_service_avg_price.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: q3_fig2_service_avg_price.png")
    
    def plot_pricing_heatmap(self, solution, output_dir):
        """图3：服务站-服务类型定价热力图（加分项）"""
        prices = solution['prices']
        
        station_names = [self.communities[j] for j in self.station_indices]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(prices, cmap='coolwarm', aspect='auto')  # prices已经是服务站定价数组
        
        for i in range(len(prices)):
            for j in range(6):
                price = prices[i, j]
                text_color = 'white' if abs(price - np.mean(prices)) > np.std(prices) else 'black'
                ax.text(j, i, f'{price:.1f}', ha='center', va='center', color=text_color, fontsize=9)
        
        ax.set_xticks(range(6))
        ax.set_xticklabels(self.services, fontsize=10)
        ax.set_yticks(range(len(self.station_indices)))
        ax.set_yticklabels(station_names, fontsize=10)
        ax.set_xlabel('服务类型', fontsize=12)
        ax.set_ylabel('服务站', fontsize=12)
        ax.set_title('服务站-服务类型定价热力图', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('定价（元/次）', fontsize=10)
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'q3_fig3_pricing_heatmap.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: q3_fig3_pricing_heatmap.png")
    
    def plot_satisfaction_breakdown(self, solution, output_dir):
        """图4：各小区满意度分解堆叠柱状图"""
        S_d = solution['satisfaction_distance']
        S_r = solution['satisfaction_response']
        S_p = solution['satisfaction_price']
        S_total = solution['satisfaction_total']
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(10)
        
        ax.bar(x, S_d, label='距离满意度(S1)', color='#2196F3')
        ax.bar(x, S_r, bottom=S_d, label='响应满意度(S2)', color='#4CAF50')
        ax.bar(x, S_p, bottom=S_d + S_r, label='价格满意度(S3)', color='#FF9800')
        
        for i in range(10):
            ax.text(i, S_total[i] + 0.01, f'{S_total[i]:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('小区', fontsize=12)
        ax.set_ylabel('满意度', fontsize=12)
        ax.set_title('各小区满意度分解堆叠柱状图', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.communities)
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'q3_fig4_satisfaction_breakdown.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: q3_fig4_satisfaction_breakdown.png")
    
    def plot_price_satisfaction_scatter(self, solution, output_dir):
        """图5：价格满意度与综合满意度散点图"""
        S_p = solution['satisfaction_price']
        S_total = solution['satisfaction_total']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(S_p, S_total, s=100, c='blue', alpha=0.7, edgecolors='black')
        
        for i in range(10):
            ax.annotate(self.communities[i], (S_p[i], S_total[i]), 
                       textcoords="offset points", xytext=(5, 5), fontsize=10)
        
        try:
            z = np.polyfit(S_p, S_total, 1)
            p = np.poly1d(z)
            ax.plot(S_p, p(S_p), "r--", alpha=0.8, label=f'趋势线: y={z[0]:.3f}x+{z[1]:.3f}')
        except:
            print("  趋势线拟合失败，跳过")
        
        # 计算相关系数
        try:
            correlation = np.corrcoef(S_p, S_total)[0, 1]
        except:
            correlation = 0
        ax.text(0.05, 0.95, f'相关系数: {correlation:.3f}', transform=ax.transAxes, 
               fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('价格满意度', fontsize=12)
        ax.set_ylabel('综合满意度', fontsize=12)
        ax.set_title('价格满意度与综合满意度关系图', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'q3_fig5_price_satisfaction_scatter.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: q3_fig5_price_satisfaction_scatter.png")
    
    def plot_satisfaction_by_care_type(self, solution, output_dir):
        """图6：不同类型老人满意度对比图（加分项）"""
        # 计算各类老人平均满意度
        S_total = solution['satisfaction_total']
        N_ir = self.N_ir
        
        satisfaction_by_type = np.zeros(3)
        for r in range(3):
            total_pop = N_ir[:, r].sum()
            if total_pop > 0:
                satisfaction_by_type[r] = sum(S_total[i] * N_ir[i, r] for i in range(10)) / total_pop
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#4CAF50', '#FF9800', '#F44336']
        bars = ax.bar(self.care_types, satisfaction_by_type, color=colors)
        
        for bar, sat in zip(bars, satisfaction_by_type):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{sat:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('老人类型', fontsize=12)
        ax.set_ylabel('平均综合满意度', fontsize=12)
        ax.set_title('不同类型老人满意度对比图', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'q3_fig6_satisfaction_by_care_type.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: q3_fig6_satisfaction_by_care_type.png")
    
    def plot_accessibility_by_care_type(self, solution, output_dir):
        """图7：不同类型老人可及性对比图（必放）"""
        if 'accessibility' in solution:
            access_stats = solution['accessibility']
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(3)
            width = 0.25
            
            geo_access = [access_stats[t]['geographic_access'] * 100 for t in ['自理', '半失能', '失能']]
            eco_access = [access_stats[t]['economic_access'] * 100 for t in ['自理', '半失能', '失能']]
            total_access = [access_stats[t]['total_access'] * 100 for t in ['自理', '半失能', '失能']]
            
            bars1 = ax.bar(x - width, geo_access, width, label='地理可达率', color='#2196F3')
            bars2 = ax.bar(x, eco_access, width, label='经济可及率', color='#4CAF50')
            bars3 = ax.bar(x + width, total_access, width, label='综合可及率', color='#FF9800')
            
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                           f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel('老人类型', fontsize=12)
            ax.set_ylabel('可及率（%）', fontsize=12)
            ax.set_title('不同类型老人可及性对比图', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(['自理', '半失能', '失能'])
            ax.legend(fontsize=10)
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            filepath = os.path.join(output_dir, 'q3_fig7_accessibility_by_care_type.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  已保存: q3_fig7_accessibility_by_care_type.png")
        else:
            print("  可及性数据不可用，跳过")
    
    def plot_cost_vs_limit(self, solution, output_dir):
        """图8：老人月服务费用与消费上限对比图（必放）"""
        prices = solution['prices']
        
        # 计算各类老人月均服务费用
        avg_costs = []
        limits = []
        
        for r in range(3):
            avg_price = np.mean(prices)  
            avg_demand = np.mean(self.a_rm[r, :])
            avg_costs.append(avg_price * avg_demand)
            limits.append(self.theta_r[r] * np.mean(self.y_i))
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        bars = ax1.bar(self.care_types, avg_costs, color='#2196F3', alpha=0.7, label='月服务费用')
        
        for bar, cost, limit in zip(bars, avg_costs, limits):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                   f'{cost:.0f}', ha='center', va='bottom', fontsize=9)
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 25, 
                   f'{(cost/limit*100):.0f}%', ha='center', va='bottom', fontsize=8, color='red')
        
        ax1.set_xlabel('老人类型', fontsize=12)
        ax1.set_ylabel('月服务费用（元）', fontsize=12)
        ax1.set_title('老人月服务费用与消费上限对比图', fontsize=14, fontweight='bold')
        
        ax2 = ax1.twinx()
        ax2.plot(self.care_types, limits, 'o-', color='#F44336', linewidth=2, markersize=10, label='消费上限')
        ax2.set_ylabel('消费上限（元）', fontsize=12, color='#F44336')
        ax2.tick_params(axis='y', labelcolor='#F44336')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'q3_fig8_cost_vs_limit.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: q3_fig8_cost_vs_limit.png")
    
    def plot_accessibility_heatmap(self, solution, output_dir):
        """图9：小区-老人类型经济可及性热力图（加分项）"""
        if 'accessibility' in solution:
            # 构建经济可及率矩阵
            eco_access_matrix = np.zeros((10, 3))
            
            for i in range(10):
                for r in range(3):
                    eco_access_matrix[i, r] = solution.get('economic_access', {}).get((i, r), 0.5)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            im = ax.imshow(eco_access_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            
            for i in range(10):
                for r in range(3):
                    ax.text(r, i, f'{eco_access_matrix[i, r]:.2f}', ha='center', va='center', 
                           color='white' if eco_access_matrix[i, r] < 0.5 else 'black', fontsize=10)
            
            ax.set_xticks(range(3))
            ax.set_xticklabels(['自理', '半失能', '失能'], fontsize=11)
            ax.set_yticks(range(10))
            ax.set_yticklabels(self.communities, fontsize=11)
            ax.set_xlabel('老人类型', fontsize=12)
            ax.set_ylabel('小区', fontsize=12)
            ax.set_title('小区-老人类型经济可及性热力图', fontsize=14, fontweight='bold')
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('经济可及率', fontsize=10)
            
            plt.tight_layout()
            filepath = os.path.join(output_dir, 'q3_fig9_accessibility_heatmap.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  已保存: q3_fig9_accessibility_heatmap.png")
        else:
            print("  可及性数据不可用，跳过")
    
    def plot_profit_vs_rate(self, solution, output_dir):
        """图10：各服务站利润与利润率双轴图（必放）"""
        if 'profit_info' in solution:
            profit_info = solution['profit_info']
            
            station_names = [info['station'] for info in profit_info]
            profits = [info['profit'] for info in profit_info]
            profit_rates = [info['profit_rate'] * 100 for info in profit_info]
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            bars = ax1.bar(station_names, profits, color='#2196F3', alpha=0.7)
            
            for bar, profit in zip(bars, profits):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       f'{profit:.1f}', ha='center', va='bottom', fontsize=9)
            
            ax1.set_xlabel('服务站', fontsize=12)
            ax1.set_ylabel('年度利润（万元）', fontsize=12)
            ax1.set_title('各服务站利润与利润率双轴图', fontsize=14, fontweight='bold')
            
            ax2 = ax1.twinx()
            line = ax2.plot(station_names, profit_rates, 'o-', color='#F44336', linewidth=2, markersize=10, label='利润率')
            
            for i, rate in enumerate(profit_rates):
                ax2.annotate(f'{rate:.1f}%', (station_names[i], rate), textcoords="offset points", xytext=(0, 5), fontsize=9)
            
            ax2.axhline(y=8, color='red', linestyle='--', linewidth=2, label='8%上限')
            ax2.set_ylabel('利润率（%）', fontsize=12, color='#F44336')
            ax2.tick_params(axis='y', labelcolor='#F44336')
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, ['利润', '利润率', '8%上限'], loc='upper right')
            
            plt.tight_layout()
            filepath = os.path.join(output_dir, 'q3_fig10_profit_vs_rate.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  已保存: q3_fig10_profit_vs_rate.png")
        else:
            print("  利润数据不可用，跳过")
    
    def plot_subsidy_comparison(self, solution, output_dir):
        """图11：各服务站补贴使用情况对比图（加分项）"""
        if 'profit_info' in solution:
            profit_info = solution['profit_info']
            
            station_names = [info['station'] for info in profit_info]
            actual_subsidy = [info.get('subsidy', 0) for info in profit_info]
            limits = [self.station_subsidy_limits[j] for j in range(len(self.station_indices))]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(len(station_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, actual_subsidy, width, label='实际补贴', color='#4CAF50')
            bars2 = ax.bar(x + width/2, limits, width, label='补贴上限', color='#9E9E9E', alpha=0.5)
            
            for bar, subsidy, limit in zip(bars1, actual_subsidy, limits):
                rate = (subsidy / limit * 100) if limit > 0 else 0
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       f'{subsidy:.1f}\n({rate:.0f}%)', ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('服务站', fontsize=12)
            ax.set_ylabel('补贴金额（万元）', fontsize=12)
            ax.set_title('各服务站补贴使用情况对比图', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(station_names)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            filepath = os.path.join(output_dir, 'q3_fig11_subsidy_comparison.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  已保存: q3_fig11_subsidy_comparison.png")
        else:
            print("  利润数据不可用，跳过")
    
    def plot_convergence_curve(self, output_dir):
        """图12：遗传算法收敛曲线（必放）"""
        if hasattr(self, 'convergence_history') and len(self.convergence_history) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            generations = range(len(self.convergence_history))
            ax.plot(generations, self.convergence_history, 'b-', linewidth=2)
            
            final_value = self.convergence_history[-1]
            ax.axhline(y=final_value, color='red', linestyle='--', linewidth=1.5, 
                      label=f'最终收敛值: {final_value:.4f}')
            
            ax.set_xlabel('迭代次数', fontsize=12)
            ax.set_ylabel('最优适应度值', fontsize=12)
            ax.set_title('遗传算法收敛曲线', fontsize=14, fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            filepath = os.path.join(output_dir, 'q3_fig12_convergence_curve.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  已保存: q3_fig12_convergence_curve.png")
        else:
            print("  收敛曲线数据不可用，跳过")
    
    def plot_subsidy_effect_comparison(self, solution, output_dir):
        """图13：有补贴与无补贴方案对比图（加分项）"""
        print("  补贴效果对比需要无补贴方案数据，暂时跳过")


if __name__ == "__main__":
    optimizer = PricingOptimizer()
    best_chromosome = optimizer.genetic_algorithm()
    solution = optimizer.evaluate_solution(best_chromosome)
    optimizer.create_visualizations(best_chromosome, solution)