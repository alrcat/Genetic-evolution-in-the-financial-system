# 🚀 开始文档 - 快速上手指南

欢迎使用**类生物进化金融交易系统**！本文档将帮助你快速开始。

---

## 📋 目录

1. [系统要求](#系统要求)
2. [安装步骤](#安装步骤)
3. [第一次运行](#第一次运行)
4. [理解输出](#理解输出)
5. [自定义配置](#自定义配置)
6. [下一步](#下一步)
7. [常见问题](#常见问题)

---

## 系统要求

### Python 版本
- **Python 3.10 或更高版本**

检查你的Python版本：
```bash
python --version
# 应该显示 Python 3.10.x 或更高
```

### 操作系统
- ✅ Windows 10/11
- ✅ macOS
- ✅ Linux

### 所需依赖
项目会自动安装以下核心依赖：
- `numpy` - 数值计算
- `pandas` - 数据处理
- `pydantic` - 数据验证
- `rich` - 终端美化输出
- `PyYAML` - 配置文件解析

---

## 安装步骤

### 步骤 1: 克隆或下载项目

如果你还没有项目代码，确保所有文件都在你的工作目录中。

### 步骤 2: 创建虚拟环境（推荐）

使用虚拟环境可以避免依赖冲突：

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 步骤 3: 安装依赖

```bash
# 安装所有依赖
pip install -r requirements.txt

# 或者使用 pip 直接安装（如果 requirements.txt 有问题）
pip install numpy pandas pydantic PyYAML rich tqdm
```

### 步骤 4: 验证安装

运行一个简单的检查：

```bash
python -c "from simulation.world import World; print('✅ 安装成功！')"
```

如果看到 `✅ 安装成功！`，说明安装正确。

---

## 第一次运行

### 方式 1: 使用快速启动脚本（推荐）

```bash
# 运行默认演示（1000 ticks）
python run.py

# 运行更长时间（5000 ticks）
python run.py --ticks 5000

# 查看所有选项
python run.py --help
```

### 方式 2: 使用模块方式运行

```bash
# 运行演示
python -m simulation.demo

# 带参数运行
python -m simulation.demo --ticks 2000 --interval 50
```

### 方式 3: 在 Python 中直接使用

```python
from simulation.world import World, WorldConfig

# 创建配置
config = WorldConfig(
    initial_population_size=50,
    max_population_size=200,
)

# 创建世界并运行
world = World(config)
world.initialize()

# 运行 1000 ticks
results = world.run(1000)

# 查看结果
print(f"最终种群: {world.population.alive_count} 个Agent")
```

---

## 理解输出

### 运行时的输出示例

```
╔═══════════════════════════════════════════════════════════════╗
║          🧬 类生物进化金融交易系统 🧬                          ║
╚═══════════════════════════════════════════════════════════════╝

📋 世界配置:
  • 初始种群: 50
  • 最大种群: 200
  • 突变率: 0.1
  • 繁殖模式: mixed

🌍 创建世界...
✅ 初始种群已创建: 50 个Agent

🚀 开始模拟 (1000 ticks)...

📊 Tick 100:
  存活: 48 | 出生: 3 | 死亡: 5 | 交易: 12
  价格: 102.34 | 制度: SIDEWAYS

📊 Tick 200:
  存活: 52 | 出生: 8 | 死亡: 6 | 交易: 28
  价格: 98.76 | 制度: BEAR
  ...
```

### 输出字段说明

| 字段 | 含义 |
|------|------|
| **存活** | 当前存活的Agent数量 |
| **出生** | 本tick新出生的Agent数量 |
| **死亡** | 本tick死亡的Agent数量 |
| **交易** | 本tick执行的交易数量 |
| **价格** | 当前市场价格 |
| **制度** | 当前市场制度（BULL/BEAR/SIDEWAYS/CRISIS等） |

### 最终统计输出

模拟结束后，你会看到：

1. **世界状态表格** - 当前种群和市场状态
2. **模拟统计** - 总出生数、死亡数、交易数等
3. **死亡原因分析** - 各种死因的分布
4. **代际分布** - 各代Agent的数量
5. **最优势DNA** - 携带者最多的DNA类型

---

## 自定义配置

### 修改配置文件

编辑 `config/world_rules.yaml`：

```yaml
# 种群配置
population:
  initial_size: 100        # 初始种群大小
  max_size: 500           # 最大种群大小

# 繁殖规则
reproduction:
  mode: "mixed"            # asexual / sexual / mixed
  mutation_rate: 0.1      # 突变率 [0, 1]

# 死亡规则
death:
  survival_threshold: 100.0    # 生存资本阈值
  enable_natural_death: false  # 是否启用自然死亡
```

### 在代码中自定义配置

```python
from simulation.world import World, WorldConfig

# 创建自定义配置
config = WorldConfig(
    # 种群配置
    initial_population_size=200,
    max_population_size=1000,
    enable_carrying_capacity=True,
    
    # 繁殖配置
    reproduction_mode="mixed",  # "asexual", "sexual", "mixed"
    mutation_rate=0.15,         # 更高的突变率
    crossover_rate=0.7,
    
    # 死亡配置
    survival_threshold=50.0,    # 更低的生存阈值（更容易死亡）
    enable_natural_death=False,
    
    # 环境配置
    initial_price=100.0,
    base_volatility=0.03,       # 更高的基础波动率
    
    # 存储配置
    snapshot_interval=50,       # 更频繁的快照
    persist_path="data/run1",   # 保存数据到文件
)

# 使用配置创建世界
world = World(config)
world.initialize()

# 运行模拟
results = world.run(5000)
```

---

## 下一步

### 1. 观察系统行为

运行几次模拟，观察：
- 哪些DNA类型更常见？
- 不同市场制度下种群如何变化？
- 死亡原因分布如何？
- 代际分布有什么规律？

### 2. 调整参数实验

尝试不同的配置：
- **更高的突变率** → 更多多样性，但可能不稳定
- **更低的生存阈值** → 更强的选择压力
- **更大的种群** → 更丰富的基因库
- **不同的市场制度序列** → 测试适应性

### 3. 分析数据

```python
# 获取统计信息
stats = world.population.get_statistics()
print(f"DNA多样性: {stats.unique_dna}")
print(f"代际分布: {stats.generations}")

# 查看死亡档案
graveyard_stats = world.graveyard.get_statistics()
print(f"死亡原因: {graveyard_stats['causes']}")

# 查看事件历史
events = world.event_store.get_by_type(EventType.AGENT_DIED)
print(f"总死亡事件: {len(events)}")
```

### 4. 深入研究代码

阅读以下核心模块：
- `core/agent.py` - Agent的生命周期
- `core/dna.py` - DNA表达机制
- `core/environment.py` - 市场环境模拟
- `lifecycle/reproduction.py` - 繁殖机制
- `genetics/gene_pool.py` - 基因库

### 5. 创建自定义基因

```python
from core.gene import Gene, GeneType, GeneExpression
from core.environment import EnvironmentState

class MyCustomGene(Gene):
    """你的自定义基因"""
    
    gene_type = GeneType.SIGNAL
    
    def can_express(self, env_state: EnvironmentState) -> bool:
        # 定义表达条件
        return True
    
    def express(self, env_state: EnvironmentState) -> GeneExpression:
        # 定义表达逻辑
        return GeneExpression(
            gene_id=self.gene_id,
            gene_type=self.gene_type,
            expressed=True,
            output=your_signal,
        )
    
    def mutate(self, mutation_rate: float = 0.1) -> Gene:
        # 定义突变逻辑
        new_gene = self.clone()
        # ... 突变代码
        return new_gene
```

---

## 常见问题

### Q1: 安装依赖时出错

**问题**: `pip install` 失败

**解决**:
```bash
# 升级 pip
python -m pip install --upgrade pip

# 使用国内镜像源（如果网络慢）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q2: 运行时找不到模块

**问题**: `ModuleNotFoundError`

**解决**:
- 确保你在项目根目录运行
- 确保虚拟环境已激活
- 检查 `PYTHONPATH` 是否包含项目目录

### Q3: 种群很快灭绝

**问题**: Agent很快全部死亡

**可能原因**:
- 生存阈值设置太高
- 市场波动太大
- 初始资源不足

**解决**:
```python
config = WorldConfig(
    survival_threshold=50.0,      # 降低阈值
    initial_price=100.0,
    base_volatility=0.01,         # 降低波动率
)
```

### Q4: 如何保存运行数据？

**解决**:
```python
config = WorldConfig(
    persist_path="data/my_experiment",  # 设置保存路径
    snapshot_interval=50,               # 定期快照
)

world = World(config)
world.run(5000)

# 数据保存在 data/my_experiment/ 目录
```

### Q5: 如何查看某个Agent的详细信息？

**解决**:
```python
# 获取所有Agent
agents = world.population.get_alive()

# 查看第一个Agent
agent = agents[0]
print(agent.get_summary())

# 查看DNA
print(agent.dna.get_signature())
print(f"复杂度: {agent.dna.get_complexity()} 个基因")
```

### Q6: 如何触发大灭绝事件？

**解决**:
```python
# 在模拟过程中触发
world.trigger_extinction_event(survival_rate=0.1)  # 90% 死亡
```

### Q7: 如何观察特定DNA的演化？

**解决**:
```python
# 在tick回调中跟踪
def track_dna(result):
    stats = world.population.get_statistics()
    dominant = world.population.get_dominant_dna(5)
    print(f"Top DNA: {dominant}")

world.register_tick_callback(track_dna)
world.run(1000)
```

---

## 重要提醒

⚠️ **记住系统的核心理念**：

1. **这不是优化系统** - 不要期望找到"最优"策略
2. **观察而非控制** - 你的角色是研究者，不是决策者
3. **多样性很重要** - 环境变化时，多样性是生存的关键
4. **死亡是必要的** - 没有死亡就没有选择压力
5. **时间不可逆** - 所有状态变化都是永久的

---

## 获取帮助

如果遇到问题：

1. 查看代码注释 - 每个模块都有详细注释
2. 阅读 README.md - 了解系统架构
3. 检查配置文件 - 确保参数合理
4. 运行小规模测试 - 先运行少量ticks验证

---

## 示例：完整的最小示例

```python
#!/usr/bin/env python3
"""
最小示例 - 运行一个简单的进化模拟
"""

from simulation.world import World, WorldConfig

# 1. 创建配置
config = WorldConfig(
    initial_population_size=50,
    max_population_size=200,
    mutation_rate=0.1,
    initial_price=100.0,
)

# 2. 创建世界
world = World(config)
world.initialize()

print(f"初始种群: {world.population.alive_count} 个Agent")

# 3. 运行模拟
results = world.run(1000)

# 4. 查看结果
stats = world.population.get_statistics()
print(f"\n最终统计:")
print(f"  存活: {stats.alive}")
print(f"  DNA多样性: {stats.unique_dna}")
print(f"  平均资本: {stats.average_capital:.2f}")

# 5. 查看死亡原因
death_stats = world.graveyard.get_statistics()
print(f"\n死亡原因分布:")
for cause, count in death_stats.get("causes", {}).items():
    print(f"  {cause}: {count}")

print("\n✨ 模拟完成！")
```

---

**祝你探索愉快！记住：这不是寻找答案，而是观察自然选择如何发生。** 🌱
