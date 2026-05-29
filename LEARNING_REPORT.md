# RL-WDS 项目学习报表

> 面向初学者的完整项目分析 | 生成日期: 2026-05-26
> 论文: Hajgató, G.; Gyires-Tóth, B.; Paál, G. (2020) - *Journal of Water Resources Planning and Management*

---

## 目录

1. [项目是什么](#1-项目是什么)
2. [核心概念图解](#2-核心概念图解)
3. [目录结构与文件说明](#3-目录结构与文件说明)
4. [核心技术栈](#4-核心技术栈)
5. [逐文件深度分析](#5-逐文件深度分析)
6. [数据流与训练流程](#6-数据流与训练流程)
7. [关键设计模式与技巧](#7-关键设计模式与技巧)
8. [初学者学习路线](#8-初学者学习路线)

---

## 1. 项目是什么

### 一句话概括
用**深度强化学习(DQN)**训练 AI 智能体，让它能实时控制城市供水系统中的水泵转速，取代传统慢速的数学优化方法。

### 要解决的实际问题
城市供水系统中有多台水泵。如何设定每台水泵的转速，使得：
- **省电**（水泵运行在高效区间）
- **水压安全**（不会爆管，也不会供水不足）
- **满足用户用水需求**

传统做法是用数学优化算法（如 Nelder-Mead），但每次调整都需要运行水力仿真（计算量大、慢）。这个项目的创新点是：**训练一个神经网络，它只看传感器传来的水压数据就能决定水泵该怎么调，不需要再跑仿真。**

### 训练方法（核心 trick）
训练时让经典优化器 Nelder-Mead 当"老师"：
- Nelder-Mead 花大量计算找到最优泵速 → 作为正确答案
- DQN 智能体学习模仿这个答案，但**只看水压数据**（不需要仿真）
- 训练完成后，智能体可以独立工作，速度快 100 倍以上

---

## 2. 核心概念图解

```
┌────────────────────────────────────────────────────────────────┐
│                        训练阶段                                │
│                                                                │
│  用水需求  ──→  Nelder-Mead优化器  ──→  最优泵速（参考答案）    │
│                  （需要大量仿真）                               │
│                                                                │
│  用水需求  ──→  EPANET水力仿真  ──→  水压数据（观测值）        │
│                                        ↓                       │
│                                  DQN智能体                     │
│                                   ↓                            │
│                              调整泵速                          │
│                                   ↓                            │
│                    比较当前泵速 vs 最优泵速 → 计算奖励          │
│                                   ↓                            │
│                         更新神经网络权重                       │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                       部署阶段                                 │
│                                                                │
│  传感器水压数据  ──→  DQN智能体（训练好的）  ──→  泵速指令     │
│                                                                │
│  不需要水力仿真！不需要优化器！只需要传感器！                  │
│  耗时：毫秒级 vs 传统方法的秒~分钟级                           │
└────────────────────────────────────────────────────────────────┘
```

### 关键术语表

| 术语 | 解释 |
|------|------|
| **WDS** | Water Distribution System，供水管网系统 |
| **EPANET** | 美国环保署开发的开源供水管网水力仿真软件 |
| **epynet** | EPANET 的 Python 封装库 |
| **Junction** | 节点/交汇点，水管连接处，也是用户用水的地方 |
| **Head** | 水头/压力水头，单位米，衡量水压的指标 |
| **Pump** | 水泵，提高水压的设备 |
| **Tank** | 储水池/水塔 |
| **Reservoir** | 水库/水源 |
| **DQN** | Deep Q-Network，深度 Q 网络，一种强化学习算法 |
| **Dueling DQN** | DQN 的改进版，本项目使用的架构 |
| **Nelder-Mead** | 一种不需要梯度的数值优化方法 |
| **State Value** | 状态价值，本项目定义的复合奖励函数 |

---

## 3. 目录结构与文件说明

```
rl-wds/
├── wdsEnv.py                  ← 🔥 核心：强化学习环境（必读）
├── train.py                   ← 训练脚本（必读）
├── demo.py                    ← 交互式可视化 Demo
├── generate_vld_and_tst_dta.py ← 生成验证/测试数据
├── ho_anytown.py              ← Anytown 网络超参数优化
├── ho_dtown.py                ← D-Town 网络超参数优化
├── requirements.txt           ← Python 依赖清单
├── Dockerfile                 ← Docker 容器配置
├── README.md                  ← 项目说明
├── banner.png                 ← 项目 Logo
├── apikey.txt                 ← API密钥（项目代码未使用）
│
├── opti_algorithms/           ← 优化算法库
│   ├── nm.py                  ← Nelder-Mead 方法封装
│   ├── rs.py                  ← 随机搜索
│   ├── pso.py                 ← 粒子群优化
│   └── de.py                  ← 差分进化
│
├── water_networks/            ← 供水管网模型文件
│   ├── anytown_master.inp     ← 小型示例管网（22节点）
│   ├── d-town_master.inp      ← 大型复杂管网（388+节点）
│   └── from_Exeter_db/        ← 原始基准文件
│
├── experiments/
│   ├── hyperparameters/       ← YAML 训练配置（13个文件）
│   ├── models/                ← 预训练模型（2个.zip文件）
│   ├── history/               ← 训练历史记录
│   └── tensorboard_logs/      ← TensorBoard 日志
│
└── evaluation/                ← Jupyter 评估笔记本
    ├── evaluation_of_anytown.ipynb
    ├── evaluation_of_dtown.ipynb
    ├── hyperparameter_optimization.ipynb
    ├── camera_ready_plots.ipynb
    └── proba_funs_for_demand_randomization.ipynb
```

---

## 4. 核心技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| **深度学习框架** | TensorFlow 1.15.0 | 神经网络训练和推理 |
| **强化学习库** | stable-baselines 2.10.0 | DQN 算法实现 |
| **环境接口** | OpenAI Gym 0.17.2 | 标准 RL 环境接口 |
| **水力仿真** | EPANET + epynet 1.1 | 供水管网水力计算 |
| **数值计算** | NumPy, SciPy, Pandas | 数据处理和优化 |
| **可视化** | Bokeh 2.1.0 + Panel 0.9.5 | Web 交互式 Demo |
| **超参数优化** | Optuna | 自动搜索最佳超参数 |
| **进化算法** | DEAP | PSO 和 DE 的实现 |
| **数据存储** | PyTables (HDF5) | 高效存储大量场景数据 |
| **配置管理** | PyYAML | YAML 配置文件读写 |

---

## 5. 逐文件深度分析

### 5.1 `wdsEnv.py` — 核心强化学习环境（443行）

**这是整个项目最重要的文件。**

#### 整体结构
```python
class wds():
    """Gym-like environment for water distribution systems."""
    
    def __init__(self, wds_name, speed_increment, episode_len, pump_groups, ...)
    def step(self, action, training=True)      # RL 核心：执行动作，返回(状态, 奖励, 完成, {})
    def reset(self, training=True)              # 重置环境，开始新回合
    def get_state_value(self)                   # 计算复合奖励值
    def get_observation(self)                   # 获取观测状态
    def optimize_state(self)                    # 用 Nelder-Mead 找最优泵速
    def randomize_demands(self)                 # 随机化用水需求
    # ... 辅助方法
```

#### 动作空间设计
```python
# 假设有 1 个泵组（如 Anytown）
# 动作空间 = 2 * 1 + 1 = 3 个离散动作：
#   动作0: 泵组0 增加转速
#   动作1: 泵组0 减少转速
#   动作2: "午睡"（什么都不做）

self.action_space = gym.spaces.Discrete(2 * self.dimensions + 1)
```

**设计巧思**：每个泵组有"加速"和"减速"两个动作（共 2×N 个），外加一个"午睡"（siesta）动作表示"满意当前状态"。如果连续午睡 3 次，回合自动终止。

#### 观测空间设计
```python
# 观测 = 所有节点的水压 + 当前泵速
# 水压归一化到 [-1, +1]，泵速归一化到 [0, 1]
self.observation_space = gym.spaces.Box(
    low=-1, high=+1,
    shape=(len(junctions) + len(pump_groups),),
    dtype=np.float32
)
```

#### 奖励函数（训练模式）
```python
# 核心思想：引导智能体靠近 Nelder-Mead 找到的最优泵速
distance = np.linalg.norm(optimized_speeds - current_speeds)
if distance < previous_distance:
    reward = distance * baseReward / distanceRange / maxReward  # 正奖励
else:
    reward = wrongMovePenalty  # -1 惩罚
```

#### 状态价值函数（评估模式）
```python
def get_state_value(self):
    # 三个子目标的加权和：
    # 1. 能效 (权重5)：水泵效率 / 理论峰值效率
    # 2. 水压质量 (权重8)：安全水压范围内的节点比例
    # 3. 供水满足度 (权重3)：实际供水量 / 总流量
    
    reward = (5*eff_ratio + 8*valid_heads_ratio + 3*demand_to_total) / 16
```

**关键知识**：水压必须在 15m~120m 之间。低于 15m 供水不足，高于 120m 可能爆管。

#### 学习要点
1. **Gym 环境接口**：`reset()` → `step()` → `observation, reward, done, info` 的标准模式
2. **多项式拟合**：`np.polyfit` + `np.poly1d` 拟合水泵性能曲线
3. **截断正态分布**：`scipy.stats.truncnorm` 用于在限定范围内随机化需求
4. **教师信号模式**：用传统优化器的解作为 RL 的监督信号

---

### 5.2 `train.py` — 训练脚本（235行）

#### 整体流程
```
读取 YAML 配置 → 创建环境 → 定义网络策略 → 初始化 DQN → 训练循环 → 保存模型
```

#### 自定义策略网络
```python
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
            layers=[48, 32, 12],     # YAML 中配置的层大小
            dueling=True,             # 使用 Dueling DQN 架构
            layer_norm=False,
            act_fun=tf.nn.relu)
```

**Dueling DQN 解释**：标准的 DQN 直接估计每个动作的 Q 值。DuelingDQN 将 Q 值拆分为"状态价值 V(s)"和"动作优势 A(s,a)"两部分：Q(s,a) = V(s) + A(s,a)。这样做的好处是智能体能更好地判断哪些状态下动作选择不重要（比如水压已经很好时，选哪个动作差别不大）。

#### 训练回调机制
```python
def callback(_locals, _globals):
    step_id += 1
    if step_id % vldFreq == 0:      # 每 totalSteps/25 步验证一次
        avg_reward = play_scenes(vld_scenes, ...)
        if avg_reward > best_metric:
            model.save(pathToBestModel)   # 保存最佳模型
    return True
```

#### DQN 参数
| 参数 | Anytown | D-Town | 含义 |
|------|---------|--------|------|
| totalSteps | 50,000 | 1,000,000 | 总训练步数 |
| bufferSize | 25,000 | 350,000 | 经验回放缓存大小 |
| batchSize | 16/32 | 64/128 | 批大小 |
| learningStarts | 1,000 | 10,000 | 开始学习前先收集多少经验 |
| gamma | 0.999 | 0.999 | 折扣因子（重视远期奖励） |
| initLrnRate | 1e-4 | 1e-4 | 初始学习率 |

#### 验证/评估设计
```python
def play_scenes(scenes, history, ...):
    for scene in scenes:
        env.wds.junctions.basedemand = scenes.loc[scene_id]  # 设置场景需求
        obs = env.reset(training=False)  # 重置（不重新优化！）
        while not env.done:
            act, _ = model.predict(obs, deterministic=True)  # 确定性策略
            obs, reward, _, _ = env.step(act, training=False)
```

**关键细节**：验证时 `training=False`，意味着：
1. 不会重新运行 Nelder-Mead 找最优值
2. 奖励是真实的 State Value 而非距离-based 奖励
3. 智能体完全依靠学到的策略，没有"参考答案"

---

### 5.3 `demo.py` — 交互式可视化（659行）

#### 技术架构
```
Bokeh (图表渲染) + Panel (UI组件) + Param (响应式数据绑定)
```

#### 核心类

**`environment_wrapper`** — 环境加载器
- 提供 UI 控件：选择水网、需求类型、泵速类型
- 加载预训练 DQN 模型（`.zip` 文件）
- 提取管网拓扑坐标用于可视化

**`optimize_speeds`** — 优化执行器
- `call_dqn()`: 运行 DQN 智能体，记录每一步的水压变化
- `call_nm()`: 运行 Nelder-Mead 优化作为对比
- 两种方法从**完全相同的初始状态**开始（通过 `store_bc()` / `restore_bc()` 保证）

#### 可视化技巧
```python
# 使用 Bokeh 的颜色映射展示管网水压
mapper = linear_cmap(field_name='junc_prop', palette='Viridis10',
                     low=min_prop, high=max_prop)
nodes = fig.circle(x='x', y='y', color=mapper, source=data, size=12)
```

#### 学习要点
1. **Param 框架**：`@param.depends('act_load')` 装饰器实现数据变更自动更新 UI
2. **Bokeh 周期性回调**：`curdoc().add_periodic_callback(animate, 500)` 实现动画
3. **Panel WidgetBox**：快速构建仪表盘式布局

---

### 5.4 `generate_vld_and_tst_dta.py` — 场景数据生成（413行）

#### 目的
生成一组"验证场景"（不同的用水需求组合），并跑一遍所有优化算法，建立性能基线。

#### 优化算法对比

| 算法 | 评估次数 | 特点 |
|------|----------|------|
| Nelder-Mead | ~100次 | 经典无梯度优化，速度快 |
| Differential Evolution | 600次 (30×20) | 全局优化，探索能力强 |
| Particle Swarm | 600次 (30×20) | 群体智能，收敛快 |
| Random Search | 可变 | 随机爬山，简单基准 |
| One-shot RS | 1次 | 随机猜一次，最弱基准 |

#### 输出
生成一个 HDF5 文件（`*_db.h5`），包含：
- `scenes`: 100个随机需求场景
- `results`: 每个场景下每种算法的优化结果

---

### 5.5 优化算法库 `opti_algorithms/`

#### `nm.py` — Nelder-Mead（15行）
```python
def minimize(target, dims, maxfev=1000, init_guess=None):
    # 封装 scipy.optimize.minimize
    # 单纯形法：n+1 个顶点构成单纯形，迭代移动替换最差点
    # 不需要梯度，适合无法求导的目标函数
```

#### `rs.py` — 随机搜索/爬山法（48行）
```python
class rs:
    def sampling_hypersphere(origin):
        # 在超球面上均匀采样，加扰动
        # 本质是随机爬山：接受所有改进，忽略所有变差
```

#### `pso.py` — 粒子群优化（99行）
```python
# 标准PSO公式：
# v_new = v_old + phi1*rand*(pbest - x) + phi2*rand*(gbest - x)
# 
# 30个粒子，20代 = 600次评估
```

#### `de.py` — 差分进化（83行）
```python
# 标准DE：
# 变异： y = a + F * (b - c)    # F=1, 三个随机父代
# 交叉： 每个基因以 CR=0.25 概率来自变异体
# 选择： 子代优于父代才替换
```

---

### 5.6 供水管网 `.inp` 文件

这两种是 EPANET 输入格式，是供水行业的行业标准：

#### Anytown 管网
```
22个节点 | 1个水库 | 2个水塔 | 32根管道 | 1个泵站(2台泵)
```
- 简单的小型示例网络
- 泵站内 2 台泵并行工作，同步控制

#### D-Town 管网
```
388+个节点 | 1个水库 | 7个水塔 | 数百根管道 | 5个泵站(11台泵)
```
- 大型复杂网络
- 11 台泵分成 5 个控制组
- 每组内的泵同步调速

#### EPANET 文件结构（了解一下即可）
```
[TITLE]       标题
[JUNCTIONS]   节点：ID, 高程, 用水量, ...
[RESERVOIRS]  水库：ID, 水头
[TANKS]       水塔：ID, 高程, 直径, ...
[PIPES]       管道：起止节点, 长度, 直径, 粗糙度
[PUMPS]       水泵：起止节点, 性能曲线ID
[CURVES]      性能曲线：流量-水头/效率关系
[VALVES]      阀门
```

---

### 5.7 YAML 配置文件

存放在 `experiments/hyperparameters/`，命名规则为 `网络名+三字母代码.yaml`。

三字母代码含义（HO/OOO/RRO/RRR/OS等）：
- **第1位 (H/O/R)**：Demands — Hand-optimized / Original / Randomized
- **第2位 (O/R/S)**：Pump speeds — Original / Randomized / (special)
- **第3位 (O/R)**：版本变体

```yaml
env:
    waterNet: anytown           # 管网名称
    speedIncrement: 0.05        # 每次调速步长（5%）
    episodeLen: 40              # 每回合最多40步
    pumpGroups: [['78', '79']]  # 泵组定义
    totalDemandLo: 0.3          # 总需求下限（原需求的30%）
    totalDemandHi: 1.1          # 总需求上限（原需求的110%）
    resetOrigDemands: False     # 是否用原始需求（False=随机化）
    resetOrigPumpSpeeds: False  # 是否用原始泵速（False=随机初始化）

model:
    layers: [48, 32, 12]        # 神经网络隐藏层

training:
    initLrnRate: 0.0001         # 学习率
    totalSteps: 50000           # 总训练步数
    gamma: 0.999                # 折扣因子
    batchSize: 16               # 批大小
    learningStarts: 1000        # 学习开始步数
    bufferSize: 25000           # 回放缓存大小
```

---

## 6. 数据流与训练流程

```
第0步: 生成场景数据
  generate_vld_and_tst_dta.py
  └→ 随机生成100个用水需求场景 → 存入 HDF5 (*_db.h5)

第1步: 初始化环境 (每个回合)
  wdsEnv.reset(training=True)
  ├→ 随机化用水需求
  ├→ Nelder-Mead 优化器找最优泵速 → self.optimized_speeds
  ├→ 随机初始化泵速
  └→ 返回观测值 (水压 + 泵速)

第2步: DQN 训练循环
  for step in range(totalSteps):
      action = DQN.predict(observation)    ← ε-greedy 探索
      observation, reward, done = env.step(action)
      DQN.replay_buffer.add(experience)
      if step > learningStarts:
          DQN.train_on_batch()             ← 从回放缓存采样学习

第3步: 定期验证
  每 totalSteps/25 步:
      在验证场景上运行确定性策略
      计算平均奖励
      如果改善了 → 保存模型

第4步: 最终测试
  在新的测试场景上评估最佳模型
  记录详细性能指标
```

---

## 7. 关键设计模式与技巧

### 7.1 教师-学生训练模式
这是本项目最核心的设计思想：
```python
# 训练时：Nelder-Mead 当老师
def reset(self, training=True):
    if training:
        self.optimize_state()  # 运行 Nelder-Mead，找到最优泵速

# step 中用距离-based 奖励引导智能体靠近最优值
def step(self, action, training=True):
    if training:
        distance = ||current_speeds - optimized_speeds||
        reward = f(distance)  # 越近奖励越大

# 部署时：不需要 Nelder-Mead
def step(self, action, training=False):
    # 直接用 State Value 作为奖励
    reward = self.get_state_value()
```

### 7.2 复合奖励设计
```python
reward = (5*能效 + 8*水压安全性 + 3*供水满足度) / 16
```
- 水压安全性权重最高（8/16=50%），因为水压问题直接影响公共安全
- 通过分离的三个维度，平衡经济性、安全性、服务质量

### 7.3 "午睡"机制（Siesta）
```python
if action == siesta_action:  # 什么都不做
    self.n_siesta += 1
    if self.n_siesta == 3:   # 连续3次午睡
        self.done = True      # 智能体表示"满意当前状态"，回合结束
```
鼓励智能体在找到好方案后主动停止，而不是无意义地继续调整。

### 7.4 归一化观测
```python
# 水压归一化到 [-1, +1]
heads = (2 * junction_heads / maxHead) - 1
# 泵速归一化到 [0, 1]  
speeds = pump_speeds / speedLimitHi
```
归一化是深度学习的基本要求，有助于训练稳定。

### 7.5 离散动作空间处理连续控制问题
水泵转速是连续值（0.7 ~ 1.2），但 DQN 只能处理离散动作。解决方案：
- 将转速范围以 0.05 为步长离散化 → 11 个可选值
- 每个动作只是 +0.05 或 -0.05 的增量调整
- 通过多步调整逼近任意连续值

---

## 8. 初学者学习路线

### 第一周：理解问题域
1. **先读 [README.md](./README.md)**，了解项目背景和论文
2. **浏览 EPANET 基础知识**：
   - EPANET 是什么（开源供水管网仿真软件）
   - 什么是 Junctions/Pipes/Pumps/Tanks/Reservoirs
   - 什么是 Head（水头/水压）
3. **运行 Demo**：
   ```bash
   conda activate rlwds
   cd d:/WORK/test_2/rl-wds
   set NO_PROXY=*
   set BOKEH_RESOURCES=cdn
   bokeh serve --port 8080 --allow-websocket-origin="*" ./demo.py
   ```
   浏览器打开 http://localhost:8080/demo，反复操作理解 DQN vs Nelder-Mead 的区别

### 第二周：理解核心代码
4. **精读 [wdsEnv.py](./wdsEnv.py)**（最重要！）：
   - `__init__`：理解环境和动作/观测空间的定义
   - `reset`：理解每个回合开始时做了什么
   - `step`：理解智能体如何与环境交互
   - `get_state_value`：理解奖励函数的设计逻辑
   - `randomize_demands`：理解数据增强方法
5. **然后读 [train.py](./train.py)**：
   - 理解 stable-baselines 的 DQN API
   - 理解回调函数的作用
   - 理解验证/测试的区别

### 第三周：深入理解
6. **读优化算法** (`opti_algorithms/`)：
   - `nm.py` 最简单，先看这个
   - `rs.py` 理解随机爬山法
   - `pso.py` 和 `de.py` 涉及进化算法，可选
7. **了解 EPANET `.inp` 文件格式**（`water_networks/`）
8. **探索评估 Notebook**（`evaluation/`，用 Jupyter 打开）

### 需要的前置知识
| 知识领域 | 具体要求 | 推荐资源 |
|----------|----------|----------|
| Python | 类、函数、NumPy、Pandas | Python官方教程 |
| 强化学习 | DQN基本原理、ε-greedy、经验回放 | Sutton & Barto 教材第16章 |
| 深度学习 | 全连接网络、ReLU、归一化 | 吴恩达 Coursera |
| 优化理论 | Nelder-Mead 基本思想 | SciPy 文档 |
| 供水工程 | Junction/Tank/Pump/Head/Flow | EPANET 用户手册 |

### 关键代码片段速查

```python
# 1. 创建环境
env = wds(
    wds_name='anytown_master',     # 管网名称
    speed_increment=0.05,          # 泵速调整步长
    episode_len=10,                # 每回合步数
    pump_groups=[['78', '79']],    # 泵分组
)

# 2. 训练循环的核心模式
obs = env.reset()                    # 开始新回合
while not done:
    action = model.predict(obs)      # 选择动作
    obs, reward, done, _ = env.step(action)  # 执行动作
    # DQN 内部会自动处理学习

# 3. 加载预训练模型
from stable_baselines import DQN
model = DQN.load('experiments/models/anytownHO1-best.zip')

# 4. 运行优化算法（作为对比基准）
from opti_algorithms import nm
speeds, value, iterations = nm.minimize(
    target=env.reward_to_scipy,
    dims=env.dimensions
)
```

---

## 补充：环境配置参考

本项目已在 Windows 10 上配置完成：

```
Python:    3.7.16 (Conda 环境: rlwds)
TensorFlow: 1.15.0 (CPU)
位置:      C:\Users\STCYANG\.conda\envs\rlwds
```

常用命令：
```bash
conda activate rlwds             # 激活环境
set NO_PROXY=*                  # 解决代理问题（国内网络）
python train.py --params anytownMaster --seed 7   # 训练
bokeh serve --port 8080 ./demo.py    # 启动Demo
```
