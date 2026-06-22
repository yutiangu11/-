import matplotlib
# 强制指定使用 Agg 后端，避开所有 GUI 窗口渲染冲突
matplotlib.use('Agg')


import random
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 全局配置：解决 Matplotlib 中文显示乱码问题
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 1. 基础逻辑引擎类 (Person, Elevator, SimulationEnv)
# ==========================================
class Person:
    def __init__(self, pid, target_floor, arrival_time=0.0):
        self.pid = pid
        self.target_floor = target_floor
        # 到达时间(记录): 该乘客抵达 -1 层大厅、开始排队候梯的时刻 (秒)。
        # =0 表示早高峰 t=0 时已在场 (题目基础理想化); >0 表示陆续涌入大厅。
        self.arrival_time = arrival_time
        # 等待时长(指标): 上梯时刻 - 到达时刻, 即真实排队候梯时间 (秒)。
        # 由 process_trip 赋值。当 arrival_time=0 时退化为“上梯时刻”。
        self.wait_time = 0.0
        # 到达目标楼层时间(记录): 所乘电梯抵达其目标层、开门的绝对时刻 (秒)。
        # 由 process_trip 赋值。旅程终点; 总旅程时间 = 该值 - arrival_time。
        self.floor_arrival_time = 0.0

class Elevator:
    def __init__(self, name, allowed_floors, start_floor=-1, capacity=10):
        # 电梯id
        self.name = name
        # 允许停靠集合
        self.allowed_floors = set(allowed_floors)
        # 限制人数
        self.capacity = capacity
        # 起始楼层
        self.start_floor = start_floor
        # 当前楼层
        self.current_floor = start_floor

        # --- 题目基础条件设定 ---
        self.time_per_floor = 3.0  # 每层楼之间电梯的运行时间 (秒)
        self.stop_time = 10.0  # 每层电梯的停留时间 (秒)

        self.trips = 0  # 该电梯累计运行趟数

        self.total_time = 0.0

    def move_time(self, from_floor, to_floor):
        """计算两层之间的匀速运行时间"""
        distance = abs(to_floor - from_floor)
        return distance * self.time_per_floor

    def process_trip(self, passengers, boarding_time):
        """处理一趟完整运送任务 (事件驱动)。

        boarding_time: 由调度器给出的上梯时刻 = max(本梯空闲时刻, 乘客到达大厅时刻)。
                       事件驱动保证批内每人 arrival_time <= boarding_time, 故候梯非负。
        """
        if not passengers: return

        self.trips += 1

        # 候梯等待 = 上梯时刻 - 到达大厅时刻 (因果由调度器保证, 必非负)
        for p in passengers:
            p.wait_time = boarding_time - p.arrival_time

        # 用绝对时钟 clock 从上梯时刻起推进本趟, 记录每层开门时刻。
        clock = boarding_time

        # 1. 在起始层停留装客
        clock += self.stop_time

        # 2. 梳理目标楼层，按由低到高排序
        stops = set(p.target_floor for p in passengers)
        sorted_stops = sorted(list(stops))

        # 3. 沿途匀速前往并停靠卸客, 记录各目标层的开门时刻
        floor_arrival = {}
        curr_floor = self.start_floor
        for target in sorted_stops:
            clock += self.move_time(curr_floor, target)  # 运行到目标层
            clock += self.stop_time                      # 卸客停留
            floor_arrival[target] = clock                # 到达楼层时间
            curr_floor = target

        # 把到达楼层时间盖到每位乘客身上
        for p in passengers:
            p.floor_arrival_time = floor_arrival[p.target_floor]

        # 4. 空载匀速返回起始层待命; clock 末值即本梯下次空闲时刻
        clock += self.move_time(curr_floor, self.start_floor)
        self.current_floor = self.start_floor
        self.total_time = clock


class SimulationEnv:
    def __init__(self, start_floor=-1):
        # 起始楼层
        self.start_floor = start_floor
        # 乘客
        self.people = []
        # 电梯
        self.elevators = []

    def generate_people(self, floors=range(4, 19), people_per_floor=60, arrival_fn=None):
        """生成大楼人群并随机打乱排队顺序。

        arrival_fn: 可选, 一个无参函数, 每调用一次返回一个到达时刻 (秒), 用于模拟
                    乘客陆续涌入大厅 (如均匀/泊松/峰形, 见上方 arrival_fn 工厂)。
                    默认 None -> 全体 t=0 在场, 与题目基础理想化完全一致 (逐位不变)。
        """
        pid = 1
        for f in floors:
            for _ in range(people_per_floor):
                self.people.append(Person(pid, f))
                pid += 1
        # 先打乱(顺序受外部蒙特卡洛 random.seed() 严格控制)
        random.shuffle(self.people)
        if arrival_fn is not None:
            # 按打乱后的顺序逐个赋到达时刻: 即便 arrival_fn 有状态(如泊松返回值递增),
            # 递增的到达时刻也落到随机楼层的乘客上, 保证“到达时刻 ⊥ 目标楼层”。
            for p in self.people:
                p.arrival_time = float(arrival_fn())
            # 再按到达时刻排序, 使列表顺序 = 到达先后。这样无论 arrival_fn 有序(泊松)
            # 还是无序(均匀/峰形), run() 按列表顺序装客都统一为 FIFO(先到先上),
            # 避免“排队规则”混入对比(平均候梯不变, 但尾部 p90/最长会被显著拉低), 也更
            # 贴近真实大厅排队。默认 arrival_fn=None 时不进入此分支, 行为逐位不变。
            self.people.sort(key=lambda p: p.arrival_time)

    def add_elevator(self, elevator):
        self.elevators.append(elevator)

    def run(self):
        """事件驱动派梯仿真。

        维护全局时钟 (每台电梯的 total_time = 其下次空闲时刻)。每步选出能“最早
        完成上梯”的电梯发车, 且电梯只能接已抵达大厅 (arrival_time <= 上梯时刻) 的
        乘客; 若某空闲电梯当前无已到达的可服务乘客, 则其上梯时刻顺延至下一位可服务
        乘客的到达时刻。保证因果 (候梯必非负) 且工作保持 (空闲车主动兜底)。
        当全体 arrival_time=0 时, 与原贪心逐位等价。
        """
        waiting = self.people[:]  # 列表顺序(已打乱)近似大厅排队先后
        indexed = list(enumerate(self.elevators))  # 携带插入序做稳定 tie-break

        while waiting:
            # —— 选梯阶段 —— 为每台电梯算出它“能开始上梯的最早时刻”bt, 取全局最小者发车。
            best = None  # (boarding_time, idx, elevator): 最早能发车的候选
            for idx, e in indexed:
                ready = False          # 该梯是否已有“到达且可服务”的乘客在等
                earliest_arr = None    # 若暂无, 记录其可服务乘客中最早的到达时刻
                for p in waiting:
                    if p.target_floor in e.allowed_floors:
                        # 人已到大厅、且电梯也已空闲 -> 立即可载, 无需再扫
                        if p.arrival_time <= e.total_time:
                            ready = True
                            break
                        # 人尚未到 -> 记录最早到达时刻, 作为电梯需空转等待的下限
                        if earliest_arr is None or p.arrival_time < earliest_arr:
                            earliest_arr = p.arrival_time
                if ready:
                    bt = e.total_time                  # 已有乘客在等, 上梯时刻=空闲时刻
                elif earliest_arr is not None:
                    bt = earliest_arr                  # 无人在等, 顺延到最早乘客到达时
                else:
                    continue                           # 该梯已无任何可服务乘客, 跳过
                # 选 bt 最小者; bt 相等时按插入序 idx 打破平局, 保证结果可复现
                if best is None or (bt, idx) < (best[0], best[1]):
                    best = (bt, idx, e)

            # 无任何电梯能服务剩余乘客 -> 分区有楼层漏配, 直接报错
            if best is None:
                unreachable_floors = sorted(set(p.target_floor for p in waiting))
                raise ValueError(f"存在无法被任何电梯服务的楼层: {unreachable_floors}")

            # —— 装载阶段 —— 选定电梯按列表(排队)顺序, 载入至多 capacity 名
            # “已到达且本梯可服务”的乘客; 未到达者本趟不带, 留待后续。
            boarding_time, _, e = best
            batch = []
            for p in waiting:
                if p.target_floor in e.allowed_floors and p.arrival_time <= boarding_time:
                    batch.append(p)
                    if len(batch) >= e.capacity:
                        break

            # 从等待集移除本趟乘客, 交由电梯执行运送(内部推进其 total_time)
            picked = set(p.pid for p in batch)
            waiting = [p for p in waiting if p.pid not in picked]
            e.process_trip(batch, boarding_time)

        # 全部运完, 系统清空时间取决于最慢的电梯(木桶效应)
        return max([e.total_time for e in self.elevators])

    def wait_stats(self):
        """汇总全体乘客的候梯等待时长(指标)。需在 run() 之后调用。"""
        waits = sorted(p.wait_time for p in self.people)
        n = len(waits)
        mean = sum(waits) / n
        p90 = waits[int(0.9 * (n - 1))]
        return {
            "mean": mean,
            "max": waits[-1],
            "min": waits[0],
            "p90": p90,
        }

    def journey_stats(self):
        """汇总端到端总旅程时长 = 到达楼层时刻 - 到达大厅时刻 (指标)。
        涵盖候梯 + 乘梯全过程, 是乘客真正体感的总耗时。需在 run() 之后调用。"""
        journeys = sorted(p.floor_arrival_time - p.arrival_time for p in self.people)
        n = len(journeys)
        mean = sum(journeys) / n
        p90 = journeys[int(0.9 * (n - 1))]
        return {
            "mean": mean,
            "max": journeys[-1],
            "min": journeys[0],
            "p90": p90,
        }

    def print_results(self):
        print("\n" + "=" * 50)
        print("🚀 湘江楼早高峰电梯精细物理仿真结果")
        print("=" * 50)
        max_time = 0
        bottleneck = ""

        for e in self.elevators:
            floors_str = str(sorted(list(e.allowed_floors)))
            print(f"电梯 [{e.name}]")
            print(f"  - 停靠楼层: {floors_str}")
            print(f"  - 运行趟数: {e.trips} 趟")
            print(f"  - 累计耗时: {e.total_time:.1f} 秒 ({e.total_time / 60:.2f} 分钟)")
            print("-" * 30)

            # 木桶效应：寻找耗时最长的电梯
            if e.total_time > max_time:
                max_time = e.total_time
                bottleneck = e.name

        print(f"🌟 系统总体清空时间 (取决于最慢的电梯): {max_time:.1f} 秒 ({max_time / 60:.2f} 分钟)")
        print(f"⚠️ 系统的瓶颈是: 电梯 [{bottleneck}]")
        ws = self.wait_stats()
        js = self.journey_stats()
        print("-" * 30)
        print(f"⏱️ 乘客候梯等待时长 (指标):")
        print(f"  - 平均等待: {ws['mean']:.1f} 秒 ({ws['mean'] / 60:.2f} 分钟)")
        print(f"  - 90分位等待: {ws['p90']:.1f} 秒    最长等待: {ws['max']:.1f} 秒")
        print(f"🚶 乘客总旅程时长 (到达楼层 - 到达大厅):")
        print(f"  - 平均旅程: {js['mean']:.1f} 秒 ({js['mean'] / 60:.2f} 分钟)")
        print(f"  - 90分位旅程: {js['p90']:.1f} 秒    最长旅程: {js['max']:.1f} 秒")
        print("=" * 50)

# ==========================================
# 1.5 到达过程模型 (arrival_fn 工厂)
# ----------------------------------------
# 每个函数返回一个"无参 arrival_fn", 每调用一次给出一名乘客的到达大厅时刻(秒),
# 直接传给 SimulationEnv.generate_people(arrival_fn=...)。参数 T 为早高峰涌入时间窗。
# 提醒: 窗口 T 相对系统清空能力(~4450 秒 / 900 人)的大小, 决定系统处于
#       "过载 / 临界饱和 / 欠载", 比分布形状更主导结果, 是真正该扫描的变量。
# ==========================================
def uniform_arrivals(T):
    """① 均匀到达: 在 [0, T] 内等概率到达, 到达率恒定。
    数学上等价于齐次泊松过程各到达时刻的边际分布, 是最简、最易解释的基线。"""
    return lambda: random.uniform(0, T)


def poisson_arrivals(T, n=900):
    """② 泊松过程到达: 相邻乘客到达间隔服从指数分布 (排队论标准模型)。
    取速率 rate = n/T (人/秒), 使 n 名乘客平均在 T 秒内到齐。
    需用闭包累计绝对时刻, 故每次调用返回"下一个"到达时刻 (随调用递增)。"""
    rate = n / T
    clock = [0.0]                                 # 列表装可变状态, 供闭包跨次累加
    def fn():
        clock[0] += random.expovariate(rate)      # 指数间隔 -> 泊松到达流
        return clock[0]
    return fn


def peak_arrivals(T):
    """③ 峰形到达 (截断正态): 多数人集中在窗口中段 T/2, 两头稀疏 ——
    最贴近真实早高峰"先涨后落"的人流, 对应到达率 λ(t) 呈钟形的非齐次泊松过程。
    用 min/max 把高斯尾部截断回 [0, T], 避免越界。"""
    mu, sigma = T / 2.0, T / 6.0                  # 3σ≈半窗宽, 截断损失极小
    return lambda: min(max(random.gauss(mu, sigma), 0.0), T)


# ==========================================
# 2. 1000次 蒙特卡洛仿真与分布绘图
# ==========================================
def run_monte_carlo(
        name:str,
        num_simulations=1000,
        allowed_a=[-1, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
        allowed_b=[-1, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
        allowed_c=[-1, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
        allowed_d=[-1, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
    ):
    print(f"🚀 正在基于【基础条件】执行 {num_simulations} 次蒙特卡洛仿真，请稍候...")
    makespans = []
    mean_waits = []   # 每轮全体乘客的平均候梯等待 (指标)
    p90_waits = []    # 每轮 90 分位等待 (指标)

    for seed in range(num_simulations):
        random.seed(seed)

        env = SimulationEnv(start_floor=3)
        # --- 到达过程设定 ---
        # 默认: 全体 t=0 同时在场 (题目基础理想化)。
        env.generate_people(floors=range(4, 19), people_per_floor=60)

        # 若要模拟乘客在时间窗 T 内陆续涌入大厅, 传入对应分布的 arrival_fn (任选其一):
        # T = 1800  # 早高峰涌入时间窗(秒); T 相对系统清空能力(~4450s/900人)的大小决定过载程度
        # env.generate_people(floors=range(4, 19), people_per_floor=60, arrival_fn=uniform_arrivals(T))  # ① 均匀/恒定率
        # env.generate_people(floors=range(4, 19), people_per_floor=60, arrival_fn=poisson_arrivals(T))  # ② 泊松/指数间隔
        # env.generate_people(floors=range(4, 19), people_per_floor=60, arrival_fn=peak_arrivals(T))     # ③ 峰形/截断正态(最贴近真实早高峰)

        # --- 在这里设置你的分区策略 ---
        # 示例：高低区均衡划分
        eA = Elevator("A", allowed_floors=allowed_a, start_floor=-1)
        eB = Elevator("B", allowed_floors=allowed_b, start_floor=-1)
        eC = Elevator("C", allowed_floors=allowed_c, start_floor=-1)
        eD = Elevator("D", allowed_floors=allowed_d, start_floor=-1)

        # 奇偶停放
        # eA = Elevator("A", allowed_floors=[4,6,8,10,12,14,16,18], start_floor=3)
        # eB = Elevator("B", allowed_floors=[4,6,8,10,12,14,16,18], start_floor=3)
        # eC = Elevator("C", allowed_floors=[5,7,9,11,13,15,17], start_floor=3)
        # eD = Elevator("D", allowed_floors=[5,7,9,11,13,15,17], start_floor=3)

        # # 特定区域
        # # eA = Elevator("A", allowed_floors=[4, 5, 6, 7], start_floor=3)
        # # eB = Elevator("B", allowed_floors=[8, 9, 10, 11], start_floor=3)
        # # eC = Elevator("C", allowed_floors=[12, 13, 14, 15], start_floor=3)
        # # eD = Elevator("D", allowed_floors=[16, 17, 18], start_floor=3)

        # # 交叉停放
        # eA = Elevator("A", allowed_floors=[4,8,12,16,], start_floor=3)
        # eB = Elevator("B", allowed_floors=[5,9,13,17], start_floor=3)
        # eC = Elevator("C", allowed_floors=[6,10,14,18], start_floor=3)
        # eD = Elevator("D", allowed_floors=[7,8,9,10,11,12,13,14,15,16,17,18], start_floor=3)

        env.add_elevator(eA)
        env.add_elevator(eB)
        env.add_elevator(eC)
        env.add_elevator(eD)

        # 运行并记录结果
        makespan = env.run()
        makespans.append(makespan)

        # 候梯等待指标
        ws = env.wait_stats()
        mean_waits.append(ws["mean"])
        p90_waits.append(ws["p90"])

        # env.print_results()
        # print(makespans)

        # exit()


        # 进度提示
        if (seed + 1) % 100 == 0:
            progress = (seed + 1) / num_simulations * 100
            print(f"  ⏳ 仿真进度: {progress:.1f}% ({seed + 1} / {num_simulations})")

    # --- 统计学核心计算 ---
    makespans = np.array(makespans)
    mean_val = np.mean(makespans)
    std_val = np.std(makespans)
    max_val = np.max(makespans)
    min_val = np.min(makespans)

    # 候梯等待指标的跨轮统计
    mean_waits = np.array(mean_waits)
    p90_waits = np.array(p90_waits)
    wait_mean = np.mean(mean_waits)
    wait_std = np.std(mean_waits)
    p90_mean = np.mean(p90_waits)

    print("\n" + "=" * 40)
    print(f"📊 【基础条件】{num_simulations}次 蒙特卡洛仿真统计结果")
    print(f"最理想清空时间: {min_val:.1f} 秒")
    print(f"最恶劣清空时间: {max_val:.1f} 秒")
    print(f"系统平均清空时间: {mean_val:.1f} 秒")
    print(f"标准差 (波动幅度): {std_val:.1f} 秒")
    print(f"95% 置信区间: [{mean_val - 1.96 * std_val:.1f}, {mean_val + 1.96 * std_val:.1f}]")
    print("-" * 40)
    print(f"⏱️ 平均候梯等待: {wait_mean:.1f} 秒 ({wait_mean / 60:.2f} 分钟)  [标准差 {wait_std:.1f}]")
    print(f"⏱️ 90分位候梯等待: {p90_mean:.1f} 秒 ({p90_mean / 60:.2f} 分钟)")
    print("=" * 40)

    # --- 绘制分布直方图 ---
    plt.figure(figsize=(10, 6), facecolor='#F8F9FA')

    counts, bins, patches = plt.hist(makespans, bins=40, color='#9B59B6', edgecolor='white', alpha=0.85)

    plt.axvline(mean_val, color='#E74C3C', linestyle='-', linewidth=2.5, label=f'期望均值 $\\mu$ = {mean_val:.1f}s')
    plt.axvline(max_val, color='#E67E22', linestyle='--', linewidth=1.5, label=f'恶劣极值 = {max_val:.1f}s')
    plt.axvline(min_val, color='#2ECC71', linestyle='--', linewidth=1.5, label=f'理想极值 = {min_val:.1f}s')

    plt.axvspan(mean_val - 1.96 * std_val, mean_val + 1.96 * std_val, color='#E74C3C', alpha=0.1, label='95% 置信区间')

    plt.title(f'{name}电梯调度鲁棒性分析 (基础条件 | N={num_simulations}次)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('系统总清空时间 Makespan (秒)', fontsize=14, labelpad=10)
    plt.ylabel('出现频次 (次)', fontsize=14, labelpad=10)

    plt.legend(fontsize=11, loc='upper right', framealpha=0.9)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    file_name = f'basic_monte_carlo_{num_simulations}_runs.png'
    plt.savefig(file_name, dpi=400, bbox_inches='tight')
    print(f"✅ 分布图已保存至当前目录: {file_name}")

    # plt.show()

    # 返回两个指标的均值，便于跨策略横向对比
    return {"makespan": mean_val, "mean_wait": wait_mean, "p90_wait": p90_mean}


# ==========================================
# 主程序执行入口
# ==========================================
if __name__ == "__main__":
    # 基准随机
    allowed_random = [-1, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    run_monte_carlo(name="随机",num_simulations=1000,
                    allowed_a = allowed_random,allowed_b = allowed_random,allowed_c = allowed_random,allowed_d = allowed_random,
                    )

    # 1.特定区域（均分）
    allowed_a=[4, 5, 6, 7]
    allowed_b=[8, 9, 10, 11]
    allowed_c=[12, 13, 14, 15]
    allowed_d=[16, 17, 18]
    run_monte_carlo(name="特定区域（均分）",num_simulations=1000,
                    allowed_a = allowed_a,allowed_b = allowed_b,allowed_c = allowed_c,allowed_d = allowed_d,
                    )
    # 2.奇偶停放
    allowed_a=[4,6,8,10,12,14,16,18]
    allowed_b=[4,6,8,10,12,14,16,18]
    allowed_c=[5,7,9,11,13,15,17]
    allowed_d=[5,7,9,11,13,15,17]
    run_monte_carlo(name="奇偶停放",num_simulations=1000,
                    allowed_a = allowed_a,allowed_b = allowed_b,allowed_c = allowed_c,allowed_d = allowed_d,
                    )

    # 3.区域只分上下层区域
    allowed_a=[4, 5, 6, 7, 8, 9, 10, 11]
    allowed_b=[4, 5, 6, 7, 8, 9, 10, 11]
    allowed_c=[12, 13, 14, 15, 16, 17, 18]
    allowed_d=[12, 13, 14, 15, 16, 17, 18]
    run_monte_carlo(name="区域只分上下层区域",num_simulations=1000,
                    allowed_a = allowed_a,allowed_b = allowed_b,allowed_c = allowed_c,allowed_d = allowed_d,
                    )

    # 4.交叉停靠（加速度）
    allowed_a=[4,8,12,16,]
    allowed_b=[5,9,13,17]
    allowed_c=[6,10,14,18]
    allowed_d=[7,11,15]
    run_monte_carlo(name="交叉停靠",num_simulations=1000,
                    allowed_a = allowed_a,allowed_b = allowed_b,allowed_c = allowed_c,allowed_d = allowed_d,
                    )

    # 5.低中高不均匀分配
    allowed_a=[4, 5, 6, 7, 8, 9]
    allowed_b=[10, 11, 12, 13, 14]
    allowed_c=[15, 16, 17, 18]
    allowed_d=[15, 16, 17, 18]
    run_monte_carlo(name="低中高不均匀分配",num_simulations=1000,
                    allowed_a = allowed_a,allowed_b = allowed_b,allowed_c = allowed_c,allowed_d = allowed_d,
                    )
    # run_monte_carlo(num_simulations=1000)
    # run_monte_carlo(num_simulations=1000)
    # run_monte_carlo(num_simulations=1000)