"""
报告生成器 - 生成模拟结束后的详细研究报告

功能：
- 收集所有模拟数据
- 生成含时间戳的报告文件
- 导出JSON格式的原始数据
- 创建人类可读的Markdown报告
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import json
import os

if TYPE_CHECKING:
    from simulation.world import World, WorldConfig, TickResult


@dataclass
class SimulationReport:
    """模拟报告数据结构"""
    
    # 基本信息
    report_id: str = ""
    generated_at: str = ""
    simulation_start_time: str = ""
    simulation_end_time: str = ""
    
    # 配置信息
    config: Dict[str, Any] = field(default_factory=dict)
    
    # 运行概要
    total_ticks: int = 0
    total_births: int = 0
    total_deaths: int = 0
    total_trades: int = 0
    
    # 最终种群状态
    final_population: Dict[str, Any] = field(default_factory=dict)
    
    # 市场数据
    market_summary: Dict[str, Any] = field(default_factory=dict)
    price_history: List[float] = field(default_factory=list)
    volatility_history: List[float] = field(default_factory=list)
    regime_changes: List[Dict[str, Any]] = field(default_factory=list)
    
    # 死亡分析
    death_statistics: Dict[str, Any] = field(default_factory=dict)
    death_by_cause: Dict[str, int] = field(default_factory=dict)
    death_by_generation: Dict[int, int] = field(default_factory=dict)
    
    # 代际分析
    generation_distribution: Dict[int, int] = field(default_factory=dict)
    max_generation: int = 0
    
    # DNA分析
    unique_dna_count: int = 0
    dominant_dna: List[Dict[str, Any]] = field(default_factory=list)
    
    # 资本分析
    capital_statistics: Dict[str, float] = field(default_factory=dict)
    capital_distribution: List[float] = field(default_factory=list)
    
    # 繁殖分析
    reproduction_statistics: Dict[str, Any] = field(default_factory=dict)
    
    # 存活分析
    survival_statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Tick历史数据（采样）
    tick_samples: List[Dict[str, Any]] = field(default_factory=list)
    
    # 事件统计
    event_statistics: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


class ReportGenerator:
    """
    报告生成器
    
    在模拟结束后收集所有数据并生成详细报告。
    """
    
    DEFAULT_OUTPUT_DIR = "simulation_reports"
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir or self.DEFAULT_OUTPUT_DIR
        self._ensure_output_dir()
    
    def _ensure_output_dir(self) -> None:
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def _generate_report_id(self) -> str:
        """生成报告ID（基于时间戳）"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def collect_data(
        self,
        world: World,
        tick_results: List[TickResult],
        start_time: datetime,
        end_time: datetime,
        regime_changes: Optional[List[tuple]] = None,
    ) -> SimulationReport:
        """
        收集模拟数据
        
        Args:
            world: 世界实例
            tick_results: 所有tick的结果
            start_time: 模拟开始时间
            end_time: 模拟结束时间
            regime_changes: 制度变化记录
            
        Returns:
            SimulationReport实例
        """
        report = SimulationReport()
        
        # 基本信息
        report.report_id = self._generate_report_id()
        report.generated_at = datetime.now().isoformat()
        report.simulation_start_time = start_time.isoformat()
        report.simulation_end_time = end_time.isoformat()
        
        # 配置信息
        report.config = self._collect_config(world.config)
        
        # 运行概要
        report.total_ticks = world.clock.current_tick
        report.total_births = sum(r.births for r in tick_results)
        report.total_deaths = sum(r.deaths for r in tick_results)
        report.total_trades = sum(r.trades for r in tick_results)
        
        # 最终种群状态
        report.final_population = self._collect_population_data(world)
        
        # 市场数据
        report.market_summary = world.environment.get_market_summary()
        report.price_history = list(world.environment.price_history)
        report.volatility_history = list(world.environment.volatility_history)
        
        if regime_changes:
            report.regime_changes = [
                {"tick": rc[0], "from": rc[1], "to": rc[2]}
                for rc in regime_changes
            ]
        
        # 死亡分析
        report.death_statistics = self._collect_death_statistics(world)
        
        # 代际分析
        pop_stats = world.population.get_statistics()
        report.generation_distribution = dict(pop_stats.generations)
        report.max_generation = max(pop_stats.generations.keys()) if pop_stats.generations else 0
        
        # DNA分析
        report.unique_dna_count = pop_stats.unique_dna
        dominant_dna = world.population.get_dominant_dna(10)
        report.dominant_dna = [
            {"dna_id": dna_id, "carrier_count": count}
            for dna_id, count in dominant_dna
        ]
        
        # 资本分析
        report.capital_statistics = {
            "total": pop_stats.total_capital,
            "average": pop_stats.average_capital,
            "median": pop_stats.median_capital,
            "mode": pop_stats.mode_capital,
            "variance": pop_stats.variance_capital,
            "max": pop_stats.max_capital,
            "min": pop_stats.min_capital if pop_stats.min_capital != float('inf') else 0,
        }
        
        # 收集所有存活Agent的资本
        report.capital_distribution = [
            agent.resources.capital 
            for agent in world.population.get_alive()
        ]
        
        # 繁殖分析
        report.reproduction_statistics = {
            "agents_can_reproduce": pop_stats.agents_can_reproduce,
            "total_offspring": pop_stats.total_offspring,
        }
        
        # 存活分析
        report.survival_statistics = self._collect_survival_statistics(world, tick_results)
        
        # Tick采样（每10个tick保存一次摘要）
        sample_interval = max(1, len(tick_results) // 100)
        report.tick_samples = [
            {
                "tick": r.tick,
                "alive_count": r.alive_count,
                "births": r.births,
                "deaths": r.deaths,
                "trades": r.trades,
                "total_capital": r.total_capital,
                "price": r.market_state.get("price", 0),
                "regime": r.market_state.get("regime", ""),
            }
            for i, r in enumerate(tick_results)
            if i % sample_interval == 0 or i == len(tick_results) - 1
        ]
        
        # 事件统计
        report.event_statistics = self._collect_event_statistics(world)
        
        return report
    
    def _collect_config(self, config: WorldConfig) -> Dict[str, Any]:
        """收集配置信息"""
        return {
            "initial_population_size": config.initial_population_size,
            "max_population_size": config.max_population_size,
            "enable_carrying_capacity": config.enable_carrying_capacity,
            "reproduction_mode": config.reproduction_mode,
            "mutation_rate": config.mutation_rate,
            "crossover_rate": config.crossover_rate,
            "survival_threshold": config.survival_threshold,
            "enable_natural_death": config.enable_natural_death,
            "max_lifespan": config.max_lifespan,
            "initial_price": config.initial_price,
            "base_volatility": config.base_volatility,
            "snapshot_interval": config.snapshot_interval,
        }
    
    def _collect_population_data(self, world: World) -> Dict[str, Any]:
        """收集种群数据"""
        stats = world.population.get_statistics()
        
        # 收集每个存活Agent的详细信息
        agents_data = []
        for agent in world.population.get_alive():
            agents_data.append({
                "agent_id": agent.agent_id,
                "dna_id": agent.dna.dna_id,
                "generation": agent.lineage.generation,
                "birth_tick": agent.birth_tick,
                "age": world.clock.current_tick - agent.birth_tick,
                "capital": agent.resources.capital,
                "total_pnl": agent.stats.total_pnl,
                "total_trades": agent.stats.total_trades,
                "win_rate": agent.stats.win_rate,
                "max_drawdown": agent.stats.max_drawdown,
                "offspring_count": agent.offspring_count,
                "reproductive_energy": agent.reproductive_energy,
                "gene_count": sum(len(c) for c in agent.dna.chromosomes),
            })
        
        return {
            "alive_count": stats.alive,
            "dead_count": stats.dead,
            "total_count": stats.total,
            "agents": agents_data,
        }
    
    def _collect_death_statistics(self, world: World) -> Dict[str, Any]:
        """收集死亡统计"""
        graveyard_stats = world.graveyard.get_statistics()
        
        # 按死因统计
        death_by_cause = graveyard_stats.get("causes", {})
        
        # 按代际统计
        death_by_generation = graveyard_stats.get("generations", {})
        
        # 计算平均存活时间
        avg_lifespan = graveyard_stats.get("average_lifespan", 0)
        
        # 死亡Agent的详细信息（最近100个）
        recent_deaths = []
        for entry in world.graveyard.get_recent(100):
            recent_deaths.append({
                "agent_id": entry.certificate.agent_id,
                "death_tick": entry.certificate.death_tick,
                "cause": entry.certificate.cause.name,
                "lifespan": entry.certificate.lifespan,
                "generation": entry.certificate.generation,
                "final_capital": entry.certificate.final_capital,
                "total_pnl": entry.certificate.total_pnl,
                "offspring_count": entry.certificate.offspring_count,
            })
        
        return {
            "total_deaths": len(world.graveyard),
            "by_cause": death_by_cause,
            "by_generation": death_by_generation,
            "average_lifespan": avg_lifespan,
            "recent_deaths": recent_deaths,
        }
    
    def _collect_survival_statistics(
        self, 
        world: World,
        tick_results: List[TickResult]
    ) -> Dict[str, Any]:
        """收集存活统计"""
        if not tick_results:
            return {}
        
        # 计算种群变化趋势
        population_trend = [r.alive_count for r in tick_results]
        
        # 计算存活率
        initial_pop = tick_results[0].alive_count if tick_results else 0
        final_pop = tick_results[-1].alive_count if tick_results else 0
        
        survival_rate = final_pop / initial_pop if initial_pop > 0 else 0
        
        # 峰值种群
        peak_population = max(r.alive_count for r in tick_results)
        peak_tick = next(r.tick for r in tick_results if r.alive_count == peak_population)
        
        # 最低种群
        min_population = min(r.alive_count for r in tick_results)
        min_tick = next(r.tick for r in tick_results if r.alive_count == min_population)
        
        return {
            "initial_population": initial_pop,
            "final_population": final_pop,
            "survival_rate": survival_rate,
            "peak_population": peak_population,
            "peak_tick": peak_tick,
            "min_population": min_population,
            "min_tick": min_tick,
            "population_trend_sample": population_trend[::max(1, len(population_trend)//100)],
        }
    
    def _collect_event_statistics(self, world: World) -> Dict[str, int]:
        """收集事件统计"""
        return world.event_store.get_event_counts()
    
    def generate_report(
        self,
        world: World,
        tick_results: List[TickResult],
        start_time: datetime,
        end_time: datetime,
        regime_changes: Optional[List[tuple]] = None,
    ) -> str:
        """
        生成完整报告
        
        Args:
            world: 世界实例
            tick_results: 所有tick的结果
            start_time: 模拟开始时间
            end_time: 模拟结束时间
            regime_changes: 制度变化记录
            
        Returns:
            报告目录路径
        """
        # 收集数据
        report = self.collect_data(
            world, tick_results, start_time, end_time, regime_changes
        )
        
        # 创建报告目录
        report_dir = os.path.join(
            self.output_dir,
            f"report_{report.report_id}"
        )
        os.makedirs(report_dir, exist_ok=True)
        
        # 生成JSON数据文件
        self._write_json_report(report, report_dir)
        
        # 生成Markdown报告
        self._write_markdown_report(report, report_dir)
        
        # 导出价格历史CSV
        self._write_price_history_csv(report, report_dir)
        
        # 导出种群历史CSV
        self._write_population_history_csv(report, report_dir)
        
        return report_dir
    
    def _write_json_report(self, report: SimulationReport, report_dir: str) -> None:
        """写入JSON报告"""
        json_path = os.path.join(report_dir, "report_data.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    
    def _write_markdown_report(self, report: SimulationReport, report_dir: str) -> None:
        """写入Markdown报告"""
        md_path = os.path.join(report_dir, "report.md")
        
        duration = datetime.fromisoformat(report.simulation_end_time) - \
                   datetime.fromisoformat(report.simulation_start_time)
        
        content = f"""# 进化金融系统模拟报告

## 报告信息

- **报告ID**: {report.report_id}
- **生成时间**: {report.generated_at}
- **模拟开始**: {report.simulation_start_time}
- **模拟结束**: {report.simulation_end_time}
- **运行时长**: {duration}

---

## 模拟配置

| 参数 | 值 |
|------|-----|
| 初始种群大小 | {report.config.get('initial_population_size', 'N/A')} |
| 最大种群大小 | {report.config.get('max_population_size', 'N/A')} |
| 启用承载力 | {report.config.get('enable_carrying_capacity', 'N/A')} |
| 繁殖模式 | {report.config.get('reproduction_mode', 'N/A')} |
| 突变率 | {report.config.get('mutation_rate', 'N/A')} |
| 交叉率 | {report.config.get('crossover_rate', 'N/A')} |
| 生存阈值 | {report.config.get('survival_threshold', 'N/A')} |
| 初始价格 | {report.config.get('initial_price', 'N/A')} |
| 基础波动率 | {report.config.get('base_volatility', 'N/A')} |

---

## 运行概要

| 统计项 | 值 |
|--------|-----|
| 总运行Tick | {report.total_ticks} |
| 总出生数 | {report.total_births} |
| 总死亡数 | {report.total_deaths} |
| 总交易数 | {report.total_trades} |
| 净增长 | {report.total_births - report.total_deaths} |

---

## 种群状态

### 最终状态

| 统计项 | 值 |
|--------|-----|
| 存活Agent | {report.final_population.get('alive_count', 0)} |
| 死亡Agent | {report.final_population.get('dead_count', 0)} |
| 总计 | {report.final_population.get('total_count', 0)} |

### 存活分析

| 统计项 | 值 |
|--------|-----|
| 初始种群 | {report.survival_statistics.get('initial_population', 0)} |
| 最终种群 | {report.survival_statistics.get('final_population', 0)} |
| 存活率 | {report.survival_statistics.get('survival_rate', 0):.2%} |
| 峰值种群 | {report.survival_statistics.get('peak_population', 0)} (Tick {report.survival_statistics.get('peak_tick', 0)}) |
| 最低种群 | {report.survival_statistics.get('min_population', 0)} (Tick {report.survival_statistics.get('min_tick', 0)}) |

---

## 资本分析

| 统计项 | 值 |
|--------|-----|
| 总资本 | {report.capital_statistics.get('total', 0):.2f} |
| 平均资本 | {report.capital_statistics.get('average', 0):.2f} |
| 中位数资本 | {report.capital_statistics.get('median', 0):.2f} |
| 众数资本 | {report.capital_statistics.get('mode', 0):.2f} |
| 资本方差 | {report.capital_statistics.get('variance', 0):.2f} |
| 最高资本 | {report.capital_statistics.get('max', 0):.2f} |
| 最低资本 | {report.capital_statistics.get('min', 0):.2f} |

---

## 市场数据

| 统计项 | 值 |
|--------|-----|
| 最终价格 | {report.market_summary.get('price', 0):.2f} |
| 当前波动率 | {report.market_summary.get('volatility', 0):.4f} |
| 市场制度 | {report.market_summary.get('regime', 'N/A')} |
| 流动性 | {report.market_summary.get('liquidity', 0):.2f} |

### 制度变化记录

共 {len(report.regime_changes)} 次制度变化。

"""
        
        if report.regime_changes:
            content += "| Tick | 从 | 到 |\n|------|-----|----|\n"
            for rc in report.regime_changes[:20]:  # 最多显示20条
                content += f"| {rc['tick']} | {rc['from']} | {rc['to']} |\n"
            if len(report.regime_changes) > 20:
                content += f"\n*（仅显示前20条，共{len(report.regime_changes)}条）*\n"
        
        content += f"""
---

## 死亡分析

### 总体统计

- 总死亡数: {report.death_statistics.get('total_deaths', 0)}
- 平均存活时间: {report.death_statistics.get('average_lifespan', 0):.2f} tick

### 按死因分类

| 死因 | 数量 |
|------|------|
"""
        
        for cause, count in report.death_statistics.get('by_cause', {}).items():
            content += f"| {cause} | {count} |\n"
        
        content += f"""
---

## 代际分析

- 最高代数: {report.max_generation}
- 唯一DNA数量: {report.unique_dna_count}
- 可繁殖Agent: {report.reproduction_statistics.get('agents_can_reproduce', 0)}
- 总后代数: {report.reproduction_statistics.get('total_offspring', 0)}

### 代际分布（存活）

| 代数 | 数量 |
|------|------|
"""
        
        for gen in sorted(report.generation_distribution.keys()):
            content += f"| {gen} | {report.generation_distribution[gen]} |\n"
        
        content += f"""
---

## 优势DNA

| DNA ID | 携带者数量 |
|--------|-----------|
"""
        
        for dna in report.dominant_dna[:10]:
            content += f"| {dna['dna_id'][:16]}... | {dna['carrier_count']} |\n"
        
        content += f"""
---

## 事件统计

| 事件类型 | 数量 |
|----------|------|
"""
        
        for event_type, count in sorted(report.event_statistics.items(), key=lambda x: -x[1]):
            content += f"| {event_type} | {count} |\n"
        
        content += """
---

## 数据文件

本报告包含以下数据文件：

- `report_data.json` - 完整的JSON格式报告数据
- `price_history.csv` - 价格和波动率历史数据
- `population_history.csv` - 种群变化历史数据

---

*报告由进化金融系统自动生成*
"""
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _write_price_history_csv(self, report: SimulationReport, report_dir: str) -> None:
        """写入价格历史CSV"""
        csv_path = os.path.join(report_dir, "price_history.csv")
        
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("tick,price,volatility\n")
            for i, (price, vol) in enumerate(zip(
                report.price_history, 
                report.volatility_history
            )):
                f.write(f"{i},{price:.6f},{vol:.6f}\n")
    
    def _write_population_history_csv(self, report: SimulationReport, report_dir: str) -> None:
        """写入种群历史CSV"""
        csv_path = os.path.join(report_dir, "population_history.csv")
        
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("tick,alive_count,births,deaths,trades,total_capital,price,regime\n")
            for sample in report.tick_samples:
                f.write(
                    f"{sample['tick']},"
                    f"{sample['alive_count']},"
                    f"{sample['births']},"
                    f"{sample['deaths']},"
                    f"{sample['trades']},"
                    f"{sample['total_capital']:.2f},"
                    f"{sample['price']:.2f},"
                    f"{sample['regime']}\n"
                )
