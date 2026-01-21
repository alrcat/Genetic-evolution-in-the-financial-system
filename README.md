# 🧬 类生物进化金融交易系统

> **不是寻找最优交易策略，而是构造一个允许自然选择发生的金融生态系统**

## 核心理念

| 金融概念 | 生物类比 |
|---------|---------|
| 金融市场 | 生态系统 |
| 交易策略 | 生物个体（Agent） |
| 策略结构 | DNA（由多个 Gene 组成） |
| 收益 | 不是目标，只是生存的副产品 |
| 存活/繁殖能力 | 真正的选择标准 |

## 不可违背的原则

1. ❌ 系统中**不存在**全局最优解或统一 loss function
2. ⏳ 时间**不可逆**，所有状态只能向前推进
3. 📉 Agent 是**资源受限**实体（资本、风险、算力、注意力）
4. 🧬 DNA **不等于**完整策略，而是可复制、可重组的决策结构
5. 🌍 市场环境**不为**任何 Agent 服务
6. 👤 人类**只能**制定规则，**不能**指定赢家

## 快速开始

**📖 新用户请先阅读 [START.md](START.md) - 详细的快速上手指南**

### 环境准备
```bash
# 1. 克隆或下载项目
cd /path/to/financial-genetic-system

# 2. 安装依赖
pip install -r requirements.txt

# 3. 验证安装
python -c "import genetic_financial_ecosystem; print('安装成功')"
```

### 运行演示
```bash
# 运行默认模拟（1000个时间周期）
python run.py

# 运行指定时长的模拟
python run.py --ticks 5000

# 查看所有可用选项
python run.py --help

# 使用模块方式运行
python -m simulation.demo --ticks 2000 --interval 50
```

### 输出说明
运行时会显示：
- **📋 世界配置** - 当前模拟的参数设置
- **🌍 创建世界** - 初始化种群和环境
- **📊 统计信息** - 每轮显示种群状态、交易统计
- **🎯 进化指标** - 基因多样性、适应性变化趋势

### 自定义配置
修改 `config/world_rules.yaml` 可以调整：
- 初始种群大小
- 最大种群容量
- 突变率和交叉率
- 生存阈值
- 繁殖模式

### 统计数据详解

运行时显示的统计信息及其计算方法：

#### 📊 实时统计指标
| 指标 | 计算方法 | 默认设置位置 |
|------|---------|--------------|
| **存活Agent** | `len(alive_agents)` | `config/world_rules.yaml` |
| **出生数量** | 每个Tick成功繁殖的Agent数 | `simulation/world.py:374` |
| **死亡数量** | 每个Tick因资本不足等原因死亡的Agent数 | `simulation/world.py:302` |
| **交易数量** | 每个Tick成功执行的订单数 | `simulation/world.py:258` |
| **市场价格** | 基于随机游走的价格演化 | `core/environment.py:472` |
| **市场制度** | 基于马尔可夫链的制度转换 | `core/environment.py:276` |

#### 💰 Agent资产分布统计
| 指标 | 计算方法 | 代码位置 |
|------|---------|----------|
| **平均资本** | `sum(capitals) / len(capitals)` - 所有Agent资本的平均值 | `simulation/population.py:239` |
| **中位数资本** | `median(capitals)` - 所有Agent资本的中间值（50%分位数） | `simulation/population.py:244` |
| **众数资本** | `Counter(capitals).most_common(1)[0][0]` - 最频繁出现的资本值 | `simulation/population.py:247-251` |
| **资本方差** | `variance(capitals)` - 资本分布的离散程度（需≥2个数据点） | `simulation/population.py:254-257` |
| **最高资本** | `max(capitals)` - 所有Agent中的最大资本 | `simulation/population.py:240` |
| **最低资本** | `min(capitals)` - 所有Agent中的最小资本 | `simulation/population.py:241` |
| **总资本** | `sum(capitals)` - 所有Agent资本的总和 | `simulation/population.py:238` |

#### 🔢 核心默认参数
```yaml
# config/world_rules.yaml 中的关键默认值
population:
  initial_size: 100        # 初始种群大小
  max_size: 500           # 最大种群容量

reproduction:
  mode: "mixed"           # 繁殖模式：asexual/sexual/mixed
  reproduction_threshold: 100  # 繁殖资本阈值
  mutation_rate: 0.1      # 突变率
  crossover_rate: 0.7     # 交叉率

death:
  survival_threshold: 100.0    # 生存资本阈值
  dying_threshold: 10          # 濒死tick阈值

market:
  initial_price: 100.0         # 初始价格
  base_volatility: 0.02        # 基础波动率
```

#### 🏛️ 市场制度转换概率
| 当前制度 | 保持概率 | 主要转换方向 |
|---------|---------|--------------|
| **BULL** | 90% | SIDEWAYS(5%), BEAR(3%), CRISIS(2%) |
| **BEAR** | 85% | SIDEWAYS(5%), CRISIS(5%), RECOVERY(3%) |
| **SIDEWAYS** | 80% | BULL(10%), BEAR(8%), HIGH_VOLATILITY(2%) |
| **CRISIS** | 70% | RECOVERY(20%), BEAR(10%) |
| **RECOVERY** | 75% | BULL(15%), SIDEWAYS(10%) |

## 项目结构

```
金融遗传系统/
├── run.py               # 快速启动脚本（命令行入口）
├── requirements.txt      # Python依赖包列表
├── pyproject.toml        # 项目配置和构建信息
├── core/                 # 核心模块（基因、DNA、Agent、环境）
├── genetics/             # 遗传机制（突变、交叉、基因库）
├── lifecycle/            # 生命周期（出生、死亡、繁殖）
├── storage/              # 存储层（事件溯源）
├── simulation/           # 模拟运行控制
└── config/               # 配置文件
```

**注意：** 以下目录由 Python 自动生成，不在版本控制中：
- `__pycache__/` - Python 字节码缓存
- `*.egg-info/` - 包安装元数据（运行 `pip install -e .` 时生成）

### 核心文件说明

#### 根目录文件
- **`run.py`** - 主入口脚本，提供命令行界面启动模拟
- **`requirements.txt`** - 项目依赖的Python包列表
- **`pyproject.toml`** - 现代Python项目配置文件，包含项目元数据和构建配置

#### core/ - 核心概念实现
- **`gene.py`** - 基因系统实现，定义6种基因类型（SIGNAL信号、FILTER过滤、POSITION仓位、EXIT退出、RISK风险、TIMING时机、DORMANT休眠）
- **`dna.py`** - DNA结构，基因的有序集合，定义Agent的潜在行为空间
- **`agent.py`** - Agent类，资源受限的生命体（有限资本、注意力、风险承受）
- **`environment.py`** - 市场环境模拟，订单执行和状态管理
- **`resources.py`** - 资源管理（资本、注意力等有限资源）
- **`world_clock.py`** - 世界时钟，管理模拟时间的前向推进

#### genetics/ - 遗传机制
- **`gene_pool.py`** - 基因库管理，预定义各种基因模板和参数范围，支持随机基因生成
- **`mutation.py`** - 突变引擎，实现基因参数的随机变异

#### lifecycle/ - 生命周期管理
- **`birth.py`** - 出生工厂，从基因库生成新的Agent DNA
- **`death.py`** - 死亡机制，基于生存阈值和资源耗尽淘汰Agent
- **`reproduction.py`** - 繁殖引擎，支持无性生殖、有性生殖和混合模式

#### storage/ - 事件溯源存储
- **`event_store.py`** - 事件存储，所有历史事件不可变记录
- **`graveyard.py`** - 墓地管理，已死亡Agent的存档
- **`snapshot.py`** - 快照存储，定期保存世界状态用于分析和恢复

#### simulation/ - 模拟运行控制
- **`world.py`** - 世界主循环，执行规则、管理种群演化
- **`population.py`** - 种群管理，Agent集合的生命周期维护
- **`demo.py`** - 演示程序，提供可视化界面和运行控制

#### config/ - 配置管理
- **`world_rules.yaml`** - 世界规则配置（种群大小、突变率、生存阈值等）

## 核心概念详解

### 基因（Gene）- 最小决策单元
基因是最小可遗传单元，**不是完整的交易策略，而是微观的决策片段**。

**基因类型详解：**
- **SIGNAL（信号基因）** - 产生买卖信号，如价格动量、趋势跟踪、均值回归等
- **FILTER（过滤基因）** - 验证和过滤信号，增加决策的可靠性
- **POSITION（仓位基因）** - 决定持仓比例，管理资金分配
- **EXIT（退出基因）** - 定义退出条件，决定何时平仓
- **RISK（风险基因）** - 感知市场风险，调整风险偏好
- **TIMING（时机基因）** - 决定具体执行时机，优化入场/出场点
- **DORMANT（休眠基因）** - 在特定市场条件下才激活（如高波动、危机时期）

**基因工作原理：**
```python
# 每个基因只负责一个微观决策
signal_gene = PriceMomentumGene(threshold=0.02)  # 只看价格突破
position_gene = PositionSizeGene(base_size=0.1)  # 只决定仓位
risk_gene = StopLossGene(level=0.05)            # 只设置止损
```

### DNA - 基因的有序组合
DNA 是基因的有序集合，定义 Agent 的**潜在行为空间**。

**DNA 构成：**
- **基因序列** - 多个基因按特定顺序排列
- **表达条件** - 每个基因在何时、何种环境下表达
- **交互规则** - 基因间的协作和约束关系

**表型形成：**
```
DNA + 环境状态 + 资源状况 → 实际交易行为
```

### Agent - 资源受限的生命体
Agent 是生态系统中的个体，具有严格的资源约束：

**资源限制：**
- **资本** - 初始资本有限，可因亏损耗尽导致死亡
- **注意力** - 同时只能处理有限的交易信号
- **风险承受** - 最大回撤限制，超过则被淘汰
- **算力** - 决策计算时间有限

**生命周期：**
```
出生 → 生存竞争 → 繁殖 → 死亡 → 基因遗传
```

### 选择机制 - 无中心的进化涌现
**核心原则：没有预设的 fitness function！**

**涌现的选择机制：**
1. **死亡淘汰** - 资源耗尽或风险过高导致自然死亡
2. **繁殖差异** - 存活时间越长、资源越多，繁殖机会越大
3. **资源竞争** - 有限的市场机会导致间接竞争
4. **环境适应** - 基因组合在不同市场环境下的表现差异

**进化涌现：**
```python
# 没有显式的"最优策略"目标
# 适应性通过生存和繁殖率自然涌现
# 市场环境变化驱动基因库的持续演化
```

### 事件溯源 - 不可变的历史记录
所有系统状态变化都记录为不可变事件：

**事件类型：**
- **生命周期事件** - Agent出生、死亡
- **交易事件** - 订单下达、成交、盈亏
- **遗传事件** - 突变、交叉、基因表达
- **环境事件** - 市场状态变化

**设计优势：**
- **可重现性** - 从任意时间点重放历史
- **分析友好** - 支持复杂的行为模式分析
- **调试便利** - 事件序列提供完整的行为轨迹

## 许可证

MIT License
