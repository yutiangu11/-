import matplotlib
matplotlib.use('TkAgg')  # 必须在 plt 之前设置


import os
import random
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 全局配置：解决 Matplotlib 中文显示乱码问题
# ==========================================
import matplotlib

matplotlib.use('Agg')  # 防止GUI崩溃，后台静态画图
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 1. 基础逻辑引擎类
# ==========================================
class Person:
    def __init__(self, pid, target_floor, arrival_time=0.0):
        self.pid = pid
        self.target_floor = target_floor
        # 新增：记录该乘客具体到达的出发层 (-1, 1 或 3)
        self.start_floor = -1
        self.arrival_time = arrival_time

        self.wait_time = 0.0
        self.floor_arrival_time = 0.0


class Elevator:
    def __init__(self, name, allowed_floors, capacity=10):
        self.name = name
        self.allowed_floors = set(allowed_floors)
        self.capacity = capacity

        self.current_floor = -1  # 初始化均在车库 -1 层待命

        self.time_per_floor = 3.0  # 层间运行 3s
        self.stop_time = 10.0  # 停靠 10s

        self.trips = 0
        self.total_time = 0.0

    def get_level(self, floor):
        """物理楼层映射：中国大楼没有0层，-1层上一层即是1层"""
        if floor < 0:
            return floor + 1
        return floor

    def move_time(self, from_floor, to_floor):
        """计算两层之间的匀速运行时间"""
        distance = abs(self.get_level(to_floor) - self.get_level(from_floor))
        return distance * self.time_per_floor


class SimulationEnv:
    def __init__(self):
        self.people = []
        self.elevators = []

    def generate_people(self, floors=range(4, 19), people_per_floor=60, arrival_fn=None):
        pid = 1
        for f in floors:
            for _ in range(people_per_floor):
                p = Person(pid, f)

                # 条件2：根据概率分配出发层 (40%去-1, 30%去1, 30%去3)
                rand_val = random.random()
                if rand_val < 0.4:
                    p.start_floor = -1
                elif rand_val < 0.7:
                    p.start_floor = 1
                else:
                    p.start_floor = 3

                self.people.append(p)
                pid += 1

        random.shuffle(self.people)

        # 条件1：时间戳生成，按到达先后排序
        if arrival_fn is not None:
            for p in self.people:
                p.arrival_time = float(arrival_fn())
            self.people.sort(key=lambda p: p.arrival_time)

    def add_elevator(self, elevator):
        self.elevators.append(elevator)

    def run(self):
        waiting = self.people[:]
        for e in self.elevators:
            e.current_floor = -1
            e.total_time = 0.0

        while waiting:
            best = None
            # 筛选出还有能力接管当前剩余等待乘客的电梯
            active_elevators = [e for e in self.elevators if any(p.target_floor in e.allowed_floors for p in waiting)]

            if not active_elevators:
                raise ValueError("出现严重逻辑错误：存在无人接管的楼层目标")

            # --- 派梯大脑：寻找最早能接到人的电梯 ---
            for idx, e in enumerate(active_elevators):
                valid_waiting = [p for p in waiting if p.target_floor in e.allowed_floors]
                # 谁此时此刻已经在等这台电梯了？
                arrived_now = [p for p in valid_waiting if p.arrival_time <= e.total_time]

                if arrived_now:
                    # 💡 条件落实：电梯下降时，到“有人员到达的最高层”去接人
                    pickup_floor = max(p.start_floor for p in arrived_now)
                    reach_time = e.total_time + e.move_time(e.current_floor, pickup_floor)
                else:
                    # 如果当前还没人来，等最先到来的那波人
                    first_arr = min(p.arrival_time for p in valid_waiting)
                    earliest_people = [p for p in valid_waiting if p.arrival_time == first_arr]
                    pickup_floor = max(p.start_floor for p in earliest_people)
                    reach_time = max(e.total_time + e.move_time(e.current_floor, pickup_floor), first_arr)

                candidate = (reach_time, pickup_floor, idx, e)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate

            reach_time, pickup_floor, _, e = best

            # ==============================
            # 1. 核心接驳阶段 (Initial Pickup)
            # ==============================
            clock = reach_time
            e.current_floor = pickup_floor
            passengers = []

            # 停靠 10 秒
            clock += e.stop_time

            # 扫描此时在这个楼层等待的所有可载乘客
            waiting_here = [p for p in waiting if
                            p.start_floor == pickup_floor and p.target_floor in e.allowed_floors and p.arrival_time <= clock]

            for p in waiting_here:
                if len(passengers) < e.capacity:
                    passengers.append(p)
                    p.wait_time = reach_time - p.arrival_time  # 等待时间以开门瞬间为准
                    waiting.remove(p)

            # 💡 条件3落实：如果没满，强制等待 2 秒
            if len(passengers) < e.capacity:
                clock += 2.0
                late_comers = [p for p in waiting if
                               p.start_floor == pickup_floor and p.target_floor in e.allowed_floors and p.arrival_time <= clock]
                for p in late_comers:
                    if len(passengers) < e.capacity:
                        passengers.append(p)
                        p.wait_time = clock - p.arrival_time  # 压哨进的，等了0秒
                        waiting.remove(p)

            # ==============================
            # 2. 向上扫楼阶段 (Upward Sweep)
            # ==============================
            # 只检查比当前高的出发层 (例如落在-1层，才会顺路检查 1,3 层)
            base_floors_to_check = [f for f in [-1, 1, 3] if f > pickup_floor]

            for next_base in base_floors_to_check:
                arrival_at_base = clock + e.move_time(e.current_floor, next_base)

                # 是否有人在这个途径层需要这台电梯？
                waiting_there = [p for p in waiting if
                                 p.start_floor == next_base and p.target_floor in e.allowed_floors and p.arrival_time <= arrival_at_base]

                # 💡 条件4落实：如果途中层有人，必定停靠 (哪怕电梯已经满了，也要按规矩停靠开门)
                if waiting_there:
                    clock = arrival_at_base
                    e.current_floor = next_base
                    clock += e.stop_time

                    for p in waiting_there:
                        if len(passengers) < e.capacity:
                            passengers.append(p)
                            p.wait_time = arrival_at_base - p.arrival_time
                            waiting.remove(p)

                    # 同样，没满则加等 2 秒
                    if len(passengers) < e.capacity:
                        clock += 2.0
                        late_comers = [p for p in waiting if
                                       p.start_floor == next_base and p.target_floor in e.allowed_floors and p.arrival_time <= clock]
                        for p in late_comers:
                            if len(passengers) < e.capacity:
                                passengers.append(p)
                                p.wait_time = clock - p.arrival_time
                                waiting.remove(p)

            # 如果极端情况下接了半天一根毛没接到 (被别的车抢了)，留在原地更新时钟
            if not passengers:
                e.total_time = clock
                continue

            # ==============================
            # 3. 高层卸客阶段 (Delivery)
            # ==============================
            e.trips += 1
            stops = sorted(list(set(p.target_floor for p in passengers)))

            for target in stops:
                clock += e.move_time(e.current_floor, target)
                clock += e.stop_time
                for p in passengers:
                    if p.target_floor == target:
                        p.floor_arrival_time = clock
                e.current_floor = target

            # 卸客完毕，电梯停在顶层，绝对时钟结算
            e.total_time = clock

        return max([e.total_time for e in self.elevators])

    def wait_stats(self):
        waits = sorted(p.wait_time for p in self.people)
        n = len(waits)
        return {
            "mean": sum(waits) / n,
            "max": waits[-1],
            "p90": waits[int(0.9 * (n - 1))],
        }


# ==========================================
# 辅助分布工厂 (Beta分布高度拟合早高峰)
# ==========================================
def beta_arrivals(T=1800):
    # 左偏钟形曲线：大部分人集中在前中期到达，尾部拖长
    return lambda: T * random.betavariate(3, 2)


# ==========================================
# 2. 蒙特卡洛仿真批处理
# ==========================================
def run_monte_carlo(allowed_list, name: str, num_simulations=1000):
    print(f"\n[🚀] 正在评估策略: 【{name}】 | 正在执行 {num_simulations} 次动态仿真...")
    makespans = []
    mean_waits = []
    p90_waits = []

    for seed in range(num_simulations):
        random.seed(seed)
        env = SimulationEnv()

        # 激活时间流：使用 Beta 分布模拟早高峰 1800秒 (半小时) 内到达大厅的动态客流
        env.generate_people(floors=range(4, 19), people_per_floor=60, arrival_fn=beta_arrivals(1800))

        # 挂载分区电梯
        for index, allowed in enumerate(allowed_list):
            e = Elevator(f"Elevator_{chr(65 + index)}", allowed_floors=allowed)
            env.add_elevator(e)

        makespan = env.run()
        makespans.append(makespan)

        ws = env.wait_stats()
        mean_waits.append(ws["mean"])
        p90_waits.append(ws["p90"])

        if (seed + 1) % 100 == 0:
            print(f"  [WAIT] 仿真计算进度: {(seed + 1) / num_simulations * 100:.1f}%")

    makespans = np.array(makespans)
    mean_val, std_val = np.mean(makespans), np.std(makespans)
    min_val, max_val = np.min(makespans), np.max(makespans)

    wait_mean = np.mean(mean_waits)
    p90_mean = np.mean(p90_waits)

    print("=" * 55)
    print(f"[STAT] 【{name}】动态乘梯仿真结果 (高级拦截机制)")
    print(f"系统平均清空时间: {mean_val:.1f} 秒 ({mean_val / 60:.2f} 分钟)")
    print(f"极端最长清空时间: {max_val:.1f} 秒 ({max_val / 60:.2f} 分钟)")
    print("-" * 55)
    print(f"[TIME] 乘客平均候梯: {wait_mean:.1f} 秒 ({wait_mean / 60:.2f} 分钟)")
    print(f"[TIME] 90分位最长候梯: {p90_mean:.1f} 秒 ({p90_mean / 60:.2f} 分钟)")
    print("=" * 55)

    # 画图并保存
    plt.figure(figsize=(10, 6), facecolor='#F8F9FA')
    plt.hist(makespans, bins=40, color='#9B59B6', edgecolor='white', alpha=0.85)
    plt.axvline(mean_val, color='#E74C3C', linestyle='-', linewidth=2.5, label=f'期望均值 = {mean_val:.1f}s')
    plt.axvline(max_val, color='#E67E22', linestyle='--', linewidth=1.5, label=f'恶劣极值 = {max_val:.1f}s')
    plt.axvspan(mean_val - 1.96 * std_val, mean_val + 1.96 * std_val, color='#E74C3C', alpha=0.1, label='95% 置信区间')

    plt.title(f'【{name}】高层截停动态仿真 (N={num_simulations}次)', fontsize=15, fontweight='bold', pad=20)
    plt.xlabel('系统总清空时间 Makespan (秒)', fontsize=13, labelpad=10)
    plt.ylabel('出现频次 (次)', fontsize=13, labelpad=10)
    plt.legend(fontsize=11, loc='upper right', framealpha=0.9)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    file_name = f'Sim_{name}_1000.png'
    plt.show()
    # plt.savefig(file_name, dpi=400, bbox_inches='tight')
    plt.close()


# ==========================================
# 主程序执行入口
# ==========================================
if __name__ == "__main__":
    # 对比：问题 1 结论方案 (4台电梯 高低区均匀分区)
    allowed_4_zones = [[4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18]]
    run_monte_carlo(allowed_list=allowed_4_zones, name="高低区均分(4台)", num_simulations=1000)

    # 对比：问题 2 结论方案 (6台电梯)
    allowed_6_zones = [[4, 5], [6, 7], [8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]]
    run_monte_carlo(allowed_list=allowed_6_zones, name="精细分区(6台)", num_simulations=1000)