import numpy as np
import random
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
        
        # 初始化收敛曲线记录
        self.convergence_history = []
        
        population = [self.generate_individual() for _ in range(self.pop_size)]
        best_fitness = 0
        best_chromosome = None
        
        for generation in range(self.generations):
            fitness_scores = [self.fitness(ind) for ind in population]
            
            max_fitness = max(fitness_scores)
            self.convergence_history.append(max_fitness)
            
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_idx = fitness_scores.index(max_fitness)
                best_chromosome = population[best_idx].copy()
            
            if generation % 20 == 0:
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
    
    # ========== 可视化模块 ==========
    def create_visualizations(self, solution, profits):
        """
        创建问题2的可视化图表（11张图）
        
        参数:
            solution: 求解结果字典
            profits: 各服务站利润列表
        """
        print("\n" + "=" * 60)
        print("开始生成可视化图表...")
        print("=" * 60)
        
        output_dir = os.path.dirname(os.path.abspath(__file__))
        
        candidate = solution['candidate']
        eval_result = solution['evaluation']
        
        # ===== 选址与覆盖可视化 =====
        print("\n【选址与覆盖可视化】")
        
        # 图1：服务站选址与覆盖范围示意图（必放）
        self.plot_station_coverage_map(candidate, eval_result, output_dir)
        
        # 图2：小区-服务站覆盖关系热力图（必放）
        self.plot_coverage_heatmap(candidate, eval_result, output_dir)
        
        # 图3：各小区服务覆盖情况对比图（必放）
        self.plot_community_coverage(candidate, eval_result, output_dir)
        
        # 图4：服务站覆盖重叠度热力图（加分项）
        self.plot_overlap_heatmap(candidate, output_dir)
        
        # ===== 服务效果可视化 =====
        print("\n【服务效果可视化】")
        
        # 图5：各小区综合满意度对比图（必放）
        self.plot_satisfaction_comparison(eval_result, output_dir)
        
        # 图6：服务站负荷率对比图（必放）
        self.plot_utilization_comparison(candidate, eval_result, output_dir)
        
        # 图7：满意度分解柱状图（加分项）
        self.plot_satisfaction_breakdown(candidate, eval_result, output_dir)
        
        # ===== 经济分析可视化 =====
        print("\n【经济分析可视化】")
        
        # 图8：各服务站年度利润对比图（必放）
        self.plot_profit_comparison(profits, output_dir)
        
        # 图9：服务站成本结构饼图（加分项）
        self.plot_cost_structure(profits, output_dir)
        
        # ===== 算法过程可视化 =====
        print("\n【算法过程可视化】")
        
        # 图10：遗传算法收敛曲线（必放）
        self.plot_convergence_curve(output_dir)
        
        # ===== 对比分析可视化 =====
        print("\n【对比分析可视化】")
        
        # 图11：不同预算下的最优方案对比图（加分项）
        self.plot_budget_comparison(output_dir)
        
        print("\n" + "=" * 60)
        print(f"所有图表已保存到: {output_dir}")
        print("=" * 60)
    
    def plot_station_coverage_map(self, candidate, eval_result, output_dir):
        """图1：服务站选址与覆盖范围示意图"""
        stations = candidate['stations']
        sizes = candidate['sizes']
        z = eval_result['allocation']
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 假设小区坐标（根据距离矩阵估算的近似位置）
        coord_x = [0, 6, 12, 3, 9, 15, 6, 2, 8, 4]
        coord_y = [0, 2, 0, 4, 4, 6, 8, 6, 10, 10]
        
        # 绘制所有小区
        for i in range(10):
            covered = z[i].sum() > 0
            color = '#F44336' if i in stations else '#9E9E9E'
            size = 300 if i in stations else 150
            ax.scatter(coord_x[i], coord_y[i], c=color, s=size, edgecolors='black', zorder=5)
            ax.annotate(f'{self.community_names[i]}\n({self.population[i]})', 
                       (coord_x[i], coord_y[i]), textcoords="offset points", 
                       xytext=(0, 15), ha='center', fontsize=10)
        
        # 绘制服务站覆盖半径
        size_names = ['小型', '中型', '大型']
        radius_map = {'小型': 0.8, '中型': 1.0, '大型': 1.2}
        for j_idx, j in enumerate(stations):
            size_idx = sizes[j_idx]
            size_name = size_names[size_idx]
            radius = radius_map[size_name] * 1.5
            circle = plt.Circle((coord_x[j], coord_y[j]), radius, 
                               fill=False, color='#2196F3', linewidth=2, linestyle='--')
            ax.add_patch(circle)
            ax.annotate(f'{size}', (coord_x[j], coord_y[j]), fontsize=9, ha='center', va='bottom')
        
        ax.set_xlim(-2, 17)
        ax.set_ylim(-2, 12)
        ax.set_xlabel('X坐标', fontsize=12)
        ax.set_ylabel('Y坐标', fontsize=12)
        ax.set_title('服务站选址与覆盖范围示意图', fontsize=14, fontweight='bold')
        ax.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336', markersize=15, label='已建站'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#9E9E9E', markersize=12, label='未建站')
        ], loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'q2_fig1_station_coverage_map.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: q2_fig1_station_coverage_map.png")
    
    def plot_coverage_heatmap(self, candidate, eval_result, output_dir):
        """图2：小区-服务站覆盖关系热力图"""
        stations = candidate['stations']
        sizes = candidate['sizes']
        
        if not stations:
            return
        
        # 构建距离矩阵（已建站到各小区）
        dist_matrix = np.zeros((10, len(stations)))
        for j_idx, j in enumerate(stations):
            for i in range(10):
                dist_matrix[i, j_idx] = self.distance_matrix[i, j]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(dist_matrix, cmap='YlGnBu', aspect='auto')
        
        for i in range(10):
            for j_idx in range(len(stations)):
                j = stations[j_idx]
                text_color = 'white' if dist_matrix[i, j_idx] > 500 else 'black'
                covered = '✓' if self.reachability_matrix[i, j] == 1 else ''
                ax.text(j_idx, i, f'{int(dist_matrix[i, j_idx])}m\n{covered}', 
                       ha='center', va='center', color=text_color, fontsize=8)
        
        size_names = ['小型', '中型', '大型']
        station_labels = [f'{self.community_names[j]}({size_names[sizes[j_idx]]})' for j_idx, j in enumerate(stations)]
        ax.set_xticks(range(len(stations)))
        ax.set_xticklabels(station_labels, fontsize=10)
        ax.set_yticks(range(10))
        ax.set_yticklabels(self.community_names, fontsize=10)
        ax.set_xlabel('服务站', fontsize=12)
        ax.set_ylabel('小区', fontsize=12)
        ax.set_title('小区-服务站覆盖关系热力图', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('距离（米）', fontsize=10)
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'q2_fig2_coverage_heatmap.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: q2_fig2_coverage_heatmap.png")
    
    def plot_community_coverage(self, candidate, eval_result, output_dir):
        """图3：各小区服务覆盖情况对比图"""
        z = eval_result['allocation']
        satisfaction = eval_result['satisfaction']
        
        covered = [1 if z[i].sum() > 0 else 0 for i in range(10)]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['#4CAF50' if c == 1 else '#F44336' for c in covered]
        bars = ax.bar(self.community_names, satisfaction, color=colors)
        
        for bar, sat, cov in zip(bars, satisfaction, covered):
            label = f'{sat:.2f}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   label, ha='center', va='bottom', fontsize=9)
        
        ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='优秀线(0.9)')
        ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='良好线(0.8)')
        
        ax.set_xlabel('小区', fontsize=12)
        ax.set_ylabel('距离满意度', fontsize=12)
        ax.set_title('各小区服务覆盖情况与满意度对比图', fontsize=14, fontweight='bold')
        ax.legend(['优秀线(0.9)', '良好线(0.8)'], loc='upper right')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加图例说明
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#4CAF50', label='已覆盖'),
                         Patch(facecolor='#F44336', label='未覆盖')]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'q2_fig3_community_coverage.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: q2_fig3_community_coverage.png")
    
    def plot_overlap_heatmap(self, candidate, output_dir):
        """图4：服务站覆盖重叠度热力图（加分项）"""
        stations = candidate['stations']
        
        if not stations:
            return
        
        # 计算每个小区被多少个服务站覆盖
        overlap = np.zeros(10)
        for i in range(10):
            for j in stations:
                if self.reachability_matrix[i, j] == 1:
                    overlap[i] += 1
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(self.community_names, overlap, color=plt.cm.Reds(overlap / max(overlap.max(), 1)))
        
        for bar, val in zip(bars, overlap):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                   f'{int(val)}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('小区', fontsize=12)
        ax.set_ylabel('覆盖服务站数量', fontsize=12)
        ax.set_title('服务站覆盖重叠度分析', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'q2_fig4_overlap_heatmap.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: q2_fig4_overlap_heatmap.png")
    
    def plot_satisfaction_comparison(self, eval_result, output_dir):
        """图5：各小区综合满意度对比图（必放）"""
        satisfaction = eval_result['satisfaction']
        covered = satisfaction > 0
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sorted_idx = np.argsort(satisfaction)[::-1]
        sorted_sat = satisfaction[sorted_idx]
        sorted_comm = [self.community_names[i] for i in sorted_idx]
        
        colors = ['#4CAF50' if s >= 0.9 else '#FF9800' if s >= 0.8 else '#F44336' for s in sorted_sat]
        bars = ax.bar(sorted_comm, sorted_sat, color=colors)
        
        for bar, sat in zip(bars, sorted_sat):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{sat:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='优秀(≥0.9)')
        ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='良好(≥0.8)')
        
        ax.set_xlabel('小区（按满意度排序）', fontsize=12)
        ax.set_ylabel('综合满意度', fontsize=12)
        ax.set_title('各小区综合满意度对比图', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'q2_fig5_satisfaction_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: q2_fig5_satisfaction_comparison.png")
    
    def plot_utilization_comparison(self, candidate, eval_result, output_dir):
        """图6：服务站负荷率对比图（必放）"""
        stations = candidate['stations']
        sizes = candidate['sizes']
        utilization = eval_result['utilization']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        size_names = ['小型', '中型', '大型']
        station_labels = [f'{self.community_names[j]}({size_names[sizes[j_idx]]})' for j_idx, j in enumerate(stations)]
        
        colors = ['#F44336' if u > 0.85 else '#4CAF50' if u >= 0.6 else '#FF9800' for u in utilization]
        bars = ax.bar(station_labels, utilization * 100, color=colors)
        
        for bar, u in zip(bars, utilization):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{u*100:.1f}%', ha='center', va='bottom', fontsize=10)
        
        ax.axhline(y=85, color='red', linestyle='--', linewidth=2, label='警戒线(85%)')
        ax.axhline(y=60, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='最优下限(60%)')
        
        ax.set_xlabel('服务站', fontsize=12)
        ax.set_ylabel('负荷率（%）', fontsize=12)
        ax.set_title('服务站负荷率对比图', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'q2_fig6_utilization_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: q2_fig6_utilization_comparison.png")
    
    def plot_satisfaction_breakdown(self, candidate, eval_result, output_dir):
        """图7：满意度分解柱状图（加分项）"""
        stations = candidate['stations']
        z = eval_result['allocation']
        
        if not stations:
            return
        
        # 计算每个小区的三类满意度
        S1_list = []
        S2_list = []
        S3_list = []
        
        for i in range(10):
            if z[i].sum() > 0:
                for j_idx, j in enumerate(stations):
                    if z[i, j] == 1:
                        d = self.distance_matrix[i, j]
                        S1_list.append(self.calc_distance_satisfaction(d))
                        S2_list.append(self.calc_response_satisfaction(eval_result['utilization'][j_idx]))
                        S3_list.append(self.price_satisfaction[i])
                        break
            else:
                S1_list.append(0)
                S2_list.append(0)
                S3_list.append(0)
        
        x = np.arange(10)
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        bars1 = ax.bar(x - width, S1_list, width, label='距离满意度(S1)', color='#2196F3')
        bars2 = ax.bar(x, S2_list, width, label='响应满意度(S2)', color='#4CAF50')
        bars3 = ax.bar(x + width, S3_list, width, label='价格满意度(S3)', color='#FF9800')
        
        ax.set_xlabel('小区', fontsize=12)
        ax.set_ylabel('满意度得分', fontsize=12)
        ax.set_title('各小区满意度分解对比图', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.community_names)
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.2)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'q2_fig7_satisfaction_breakdown.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: q2_fig7_satisfaction_breakdown.png")
    
    def plot_profit_comparison(self, profits, output_dir):
        """图8：各服务站年度利润对比图（必放）"""
        if not profits:
            return
        
        station_names = [f'{p["station"]}({p["size"]})' for p in profits]
        revenues = [p['annual_revenue'] for p in profits]
        costs = [p['annual_cost'] for p in profits]
        profits_val = [p['annual_profit'] for p in profits]
        
        x = np.arange(len(profits))
        width = 0.3
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        bars1 = ax1.bar(x - width/2, revenues, width, label='年收入', color='#2196F3', alpha=0.8)
        bars2 = ax1.bar(x + width/2, costs, width, label='年成本', color='#F44336', alpha=0.8)
        
        ax1.set_xlabel('服务站', fontsize=12)
        ax1.set_ylabel('金额（万元）', fontsize=12, color='black')
        ax1.set_xticks(x)
        ax1.set_xticklabels(station_names)
        ax1.tick_params(axis='y')
        
        ax2 = ax1.twinx()
        line = ax2.plot(x, profits_val, 'o-', color='#4CAF50', linewidth=2, markersize=10, label='年利润')
        
        for i, p in enumerate(profits_val):
            ax2.annotate(f'{p:.1f}', (i, p), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
        
        ax2.set_ylabel('年利润（万元）', fontsize=12, color='#4CAF50')
        ax2.tick_params(axis='y', labelcolor='#4CAF50')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax1.set_title('各服务站年度经济指标对比图', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'q2_fig8_profit_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: q2_fig8_profit_comparison.png")
    
    def plot_cost_structure(self, profits, output_dir):
        """图9：服务站成本结构饼图（加分项）"""
        if not profits:
            return
        
        total_revenue = sum(p['annual_revenue'] for p in profits)
        total_cost = sum(p['annual_cost'] for p in profits)
        total_profit = sum(p['annual_profit'] for p in profits)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：收入成本利润构成
        labels = ['年收入', '年成本', '年利润']
        values = [total_revenue, total_cost, max(0, total_profit)]
        colors = ['#2196F3', '#F44336', '#4CAF50']
        axes[0].pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0].set_title('年度经济构成', fontsize=12, fontweight='bold')
        
        # 右图：各站利润占比
        profit_values = [max(0, p['annual_profit']) for p in profits]
        station_names = [f'{p["station"]}' for p in profits]
        colors2 = plt.cm.Set3(np.linspace(0, 1, len(profits)))
        axes[1].pie(profit_values, labels=station_names, autopct='%1.1f%%', colors=colors2, startangle=90)
        axes[1].set_title('各服务站利润占比', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'q2_fig9_cost_structure.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: q2_fig9_cost_structure.png")
    
    def plot_convergence_curve(self, output_dir):
        """图10：遗传算法收敛曲线（必放）"""
        if not hasattr(self, 'convergence_history') or len(self.convergence_history) == 0:
            print("  收敛曲线数据不可用，跳过")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        generations = range(len(self.convergence_history))
        ax.plot(generations, self.convergence_history, 'b-', linewidth=2)
        
        final_value = self.convergence_history[-1]
        ax.axhline(y=final_value, color='red', linestyle='--', linewidth=1.5, label=f'最终收敛值: {final_value:.4f}')
        
        # 找到收敛点
        for i in range(len(self.convergence_history) - 1):
            if abs(self.convergence_history[i] - final_value) < 0.001:
                ax.axvline(x=i, color='green', linestyle=':', alpha=0.7, label=f'收敛代数: {i}')
                break
        
        ax.set_xlabel('迭代次数', fontsize=12)
        ax.set_ylabel('最优适应度值', fontsize=12)
        ax.set_title('遗传算法收敛曲线', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'q2_fig10_convergence_curve.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: q2_fig10_convergence_curve.png")
    
    def plot_budget_comparison(self, output_dir):
        """图11：不同预算下的最优方案对比图（加分项）"""
        print("  预算敏感度分析需要多次求解，暂时跳过（可手动运行多次）")
    
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
        
        # 生成可视化图表
        self.create_visualizations(solution, profits)
        
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