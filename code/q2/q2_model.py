import numpy as np
import random
import os
from q2_data import *

class ServiceStationOptimizer:
    def __init__(self):
        # 固定随机种子，保证每次运行结果完全一致
        random.seed(42)
        np.random.seed(42)
        
        # 赋值给类属性
        self.distance_matrix = DISTANCE_MATRIX
        self.reachability_matrix = REACHABILITY_MATRIX
        self.daily_demand = DAILY_DEMAND
        self.population = POPULATION
        self.capacities = CAPACITIES
        self.build_costs = BUILD_COSTS
        self.annual_operating_costs = ANNUAL_OPERATING_COSTS
        self.avg_price = AVG_SERVICE_PRICE
        self.avg_cost = AVG_SERVICE_COST
        self.price_satisfaction = PRICE_SATISFACTION
        self.BUDGET = BUDGET
        self.alpha_d = ALPHA_D
        self.alpha_r = ALPHA_R
        self.alpha_p = ALPHA_P
        
        self.station_types = ['小型', '中型', '大型']
        self.community_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        
        # 遗传算法参数
        self.pop_size = 100
        self.generations = 200
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        
        print(f"数据加载完成！老人总数: {self.population.sum()}, 日总需求: {self.daily_demand.sum():.1f}人次")
    
    def calc_distance_satisfaction(self, d_ij):
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
    
    def calc_response_satisfaction(self, utilization):
        utilization = max(0, min(1, utilization))
        if utilization <= 0.60:
            return 1.00
        elif utilization <= 0.75:
            return 0.93
        elif utilization <= 0.85:
            return 0.85
        elif utilization <= 0.95:
            return 0.72
        else:
            return 0.60
    
    def calc_comprehensive_satisfaction(self, d_ij, utilization, price_sat):
        S1 = self.calc_distance_satisfaction(d_ij)
        S2 = self.calc_response_satisfaction(utilization)
        S3 = price_sat
        return self.alpha_d * S1 + self.alpha_r * S2 + self.alpha_p * S3
    
    def decode(self, chromosome):
        stations = []
        sizes = []
        cost = 0
        for j in range(10):
            if chromosome[j] == 1:
                stations.append(j)
                size = chromosome[10 + j]
                sizes.append(size)
                cost += self.build_costs[size]
        return {'stations': stations, 'sizes': sizes, 'cost': cost}
    
    def evaluate(self, solution):
        stations = solution['stations']
        sizes = solution['sizes']
        
        if not stations:
            return {'coverage_rate': 0, 'avg_satisfaction': 0, 'total_effective': 0,
                    'allocation': np.zeros((10, 10)), 'satisfaction': np.zeros(10),
                    'utilization': np.array([])}
        
        n = 10
        z = np.zeros((n, n), dtype=int)
        capacities = np.array([self.capacities[s] for s in sizes])
        remaining = capacities.copy()
        
        sat_matrix = np.zeros((n, len(stations)))
        for i in range(n):
            for j_idx, j in enumerate(stations):
                if self.reachability_matrix[i, j] == 1:
                    sat_matrix[i, j_idx] = self.calc_comprehensive_satisfaction(
                        self.distance_matrix[i, j], 0, self.price_satisfaction[i])
        
        for i in range(n):
            reachable = []
            for j_idx, j in enumerate(stations):
                if self.reachability_matrix[i, j] == 1 and remaining[j_idx] >= self.daily_demand[i]:
                    reachable.append((j_idx, sat_matrix[i, j_idx]))
            
            if reachable:
                reachable.sort(key=lambda x: -x[1])
                best_j_idx, _ = reachable[0]
                z[i, stations[best_j_idx]] = 1
                remaining[best_j_idx] -= self.daily_demand[i]
        
        utilization = np.zeros(len(stations))
        for j_idx, j in enumerate(stations):
            served = sum(self.daily_demand[i] * z[i, j] for i in range(n))
            if capacities[j_idx] > 0:
                utilization[j_idx] = served / capacities[j_idx]
        
        final_satisfaction = np.zeros(n)
        for i in range(n):
            for j_idx, j in enumerate(stations):
                if z[i, j] == 1:
                    final_satisfaction[i] = self.calc_comprehensive_satisfaction(
                        self.distance_matrix[i, j], utilization[j_idx], self.price_satisfaction[i])
                    break
        
        covered_pop = sum(self.population[i] for i in range(n) if z[i].sum() > 0)
        coverage_rate = covered_pop / self.population.sum()
        
        covered_idx = np.where(z.sum(axis=1) > 0)[0]
        if len(covered_idx) > 0:
            avg_satisfaction = sum(self.population[i] * final_satisfaction[i] for i in covered_idx) / covered_pop
        else:
            avg_satisfaction = 0
        
        total_effective = sum(self.daily_demand[i] * final_satisfaction[i] for i in covered_idx)
        
        return {
            'coverage_rate': coverage_rate,
            'avg_satisfaction': avg_satisfaction,
            'total_effective': total_effective,
            'allocation': z,
            'satisfaction': final_satisfaction,
            'utilization': utilization
        }
    
    def fitness(self, chromosome):
        solution = self.decode(chromosome)
        if solution['cost'] > self.BUDGET or len(solution['stations']) < 1:
            return 0
        eval_result = self.evaluate(solution)
        return 0.6 * eval_result['coverage_rate'] + 0.4 * eval_result['avg_satisfaction']
    
    def selection(self, population, fitness_scores):
        tournament = random.sample(list(zip(population, fitness_scores)), 3)
        tournament.sort(key=lambda x: -x[1])
        return tournament[0][0]
    
    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            point = random.randint(1, 19)
            return parent1[:point] + parent2[point:]
        return parent1.copy()
    
    def mutate(self, chromosome):
        for i in range(20):
            if random.random() < self.mutation_rate:
                if i < 10:
                    chromosome[i] = 1 - chromosome[i]
                else:
                    chromosome[i] = random.randint(0, 2)
        if sum(chromosome[:10]) == 0:
            chromosome[random.randint(0, 9)] = 1
        return chromosome
    
    def generate_individual(self):
        chromosome = [0] * 20
        num_stations = random.randint(2, 4)
        selected = random.sample(range(10), num_stations)
        
        cost = 0
        for j in selected:
            size = random.randint(0, 2)
            chromosome[j] = 1
            chromosome[10 + j] = size
            cost += self.build_costs[size]
        
        if cost > self.BUDGET:
            return self.generate_individual()
        return chromosome
    
    def genetic_algorithm(self):
        print("\n" + "=" * 60)
        print("开始遗传算法优化")
        print("=" * 60)
        
        population = [self.generate_individual() for _ in range(self.pop_size)]
        best_fitness = 0
        best_chromosome = None
        
        for generation in range(self.generations):
            fitness_scores = [self.fitness(ind) for ind in population]
            
            max_fitness = max(fitness_scores)
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_idx = fitness_scores.index(max_fitness)
                best_chromosome = population[best_idx].copy()
            
            if generation % 10 == 0:
                print(f"第{generation}代 | 最优适应度: {best_fitness:.4f}")
            
            new_population = [population[fitness_scores.index(max(fitness_scores))].copy()]
            
            while len(new_population) < self.pop_size:
                p1 = self.selection(population, fitness_scores)
                p2 = self.selection(population, fitness_scores)
                child = self.mutate(self.crossover(p1, p2))
                new_population.append(child)
            
            population = new_population
        
        print(f"\n遗传算法完成！最优适应度: {best_fitness:.4f}")
        return best_chromosome
        
    
    def calculate_profit(self, solution):
        candidate = solution['candidate']
        eval_result = solution['evaluation']
        stations = candidate['stations']
        sizes = candidate['sizes']
        
        profits = []
        total_profit = 0
        
        for j, (station_idx, size_idx) in enumerate(zip(stations, sizes)):
            served = [i for i in range(10) if eval_result['allocation'][i, station_idx] == 1]
            daily_effective = sum(self.daily_demand[i] * eval_result['satisfaction'][i] for i in served)
            
            annual_revenue = daily_effective * self.avg_price * 365 / 10000
            annual_direct_cost = daily_effective * self.avg_cost * 365 / 10000
            annual_operating = self.annual_operating_costs[size_idx]
            annual_depreciation = self.build_costs[size_idx] / 20
            
            profit = annual_revenue - annual_direct_cost - annual_operating - annual_depreciation
            
            profits.append({
                'station': self.community_names[station_idx],
                'size': self.station_types[size_idx],
                'daily_effective': daily_effective,
                'annual_revenue': annual_revenue,
                'annual_cost': annual_direct_cost + annual_operating + annual_depreciation,
                'annual_profit': profit
            })
            total_profit += profit
        
        return profits, total_profit
    
    def solve(self):
        best_chromosome = self.genetic_algorithm()
        
        if best_chromosome is None:
            print("未找到可行方案！")
            return None
        
        best_candidate = self.decode(best_chromosome)
        best_eval = self.evaluate(best_candidate)
        best_score = self.fitness(best_chromosome)
        
        solution = {
            'candidate': best_candidate,
            'evaluation': best_eval,
            'score': best_score
        }
        
        profits, total_profit = self.calculate_profit(solution)
        self.print_results(solution, profits, total_profit)
        self.save_results(solution, profits)
        
        return solution
    
    def print_results(self, solution, profits, total_profit):
        candidate = solution['candidate']
        eval_result = solution['evaluation']
        
        print("\n" + "=" * 60)
        print("问题2.3：最优方案结果")
        print("=" * 60)
        
        print("\n【站点配置】")
        print(f"建设站点数: {len(candidate['stations'])}")
        print(f"建设成本: {candidate['cost']}万元 (预算: {self.BUDGET}万元)")
        print(f"预算剩余: {self.BUDGET - candidate['cost']}万元")
        
        print("\n【站点详情】")
        print("小区 | 规模 | 容量(人次/日) | 负荷率")
        print("-----|------|-------------|--------")
        for j, (station_idx, size_idx, util) in enumerate(zip(
            candidate['stations'], candidate['sizes'], eval_result['utilization'])):
            print(f"  {self.community_names[station_idx]}  | {self.station_types[size_idx]} |    {self.capacities[size_idx]:4d}    | {util:.2f}")
        
        print("\n【覆盖关系】")
        print("站点 | 覆盖小区")
        print("-----|---------")
        for station_idx in candidate['stations']:
            covered = [self.community_names[i] for i in range(10) if eval_result['allocation'][i, station_idx] == 1]
            print(f"  {self.community_names[station_idx]}  | {', '.join(covered)}")
        
        print("\n【核心指标】")
        print(f"服务覆盖率: {eval_result['coverage_rate']*100:.2f}%")
        print(f"平均满意度: {eval_result['avg_satisfaction']:.4f}")
        print(f"日有效服务人次: {eval_result['total_effective']:.2f}")
        
        print("\n【年度利润】(单位：万元)")
        print("小区 | 规模 | 日有效 | 年收入 | 年成本 | 年利润")
        print("-----|------|--------|--------|--------|--------")
        for p in profits:
            print(f"  {p['station']}  | {p['size']} | {p['daily_effective']:5.2f} | {p['annual_revenue']:6.2f} | {p['annual_cost']:6.2f} | {p['annual_profit']:6.2f}")
        print(f"\n总年度利润: {total_profit:.2f}万元")
    
    def save_results(self, solution, profits):
        candidate = solution['candidate']
        eval_result = solution['evaluation']
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        with open(os.path.join(current_dir, 'q2_station_info.csv'), 'w', encoding='utf-8-sig') as f:
            f.write('小区编号,规模,建设成本(万元),容量(人次/日),负荷率\n')
            for j, (station_idx, size_idx, util) in enumerate(zip(
                candidate['stations'], candidate['sizes'], eval_result['utilization'])):
                f.write(f'{self.community_names[station_idx]},{self.station_types[size_idx]},'
                        f'{self.build_costs[size_idx]},{self.capacities[size_idx]},{util:.4f}\n')
        
        with open(os.path.join(current_dir, 'q2_satisfaction.csv'), 'w', encoding='utf-8-sig') as f:
            f.write('小区编号,满意度,是否覆盖\n')
            for i in range(10):
                covered = 1 if eval_result['allocation'][i].sum() > 0 else 0
                f.write(f'{self.community_names[i]},{eval_result["satisfaction"][i]:.4f},{covered}\n')
        
        with open(os.path.join(current_dir, 'q2_metrics.csv'), 'w', encoding='utf-8-sig') as f:
            f.write('指标,数值\n')
            f.write(f'站点数,{len(candidate["stations"])}\n')
            f.write(f'建设成本(万元),{candidate["cost"]}\n')
            f.write(f'覆盖率,{eval_result["coverage_rate"]:.4f}\n')
            f.write(f'满意度,{eval_result["avg_satisfaction"]:.4f}\n')
        
        with open(os.path.join(current_dir, 'q2_profit.csv'), 'w', encoding='utf-8-sig') as f:
            f.write('小区编号,规模,日有效人次,年收入(万元),年成本(万元),年利润(万元)\n')
            for p in profits:
                f.write(f'{p["station"]},{p["size"]},{p["daily_effective"]:.2f},'
                        f'{p["annual_revenue"]:.2f},{p["annual_cost"]:.2f},{p["annual_profit"]:.2f}\n')
        
        print(f"\n结果已保存到: {current_dir}")

if __name__ == "__main__":
    optimizer = ServiceStationOptimizer()
    optimizer.solve()