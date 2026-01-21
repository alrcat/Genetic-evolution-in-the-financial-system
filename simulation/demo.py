"""
演示脚本 - 运行进化金融系统的示例

使用方法:
    python -m simulation.demo
"""

import sys
import os
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.world import World, WorldConfig
from reports import ReportGenerator
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout


console = Console()


def print_banner():
    """打印横幅"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║          🧬 类生物进化金融交易系统 🧬                         ║
    ║                                                               ║
    ║     不是寻找最优策略，而是让自然选择发生                      ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def create_status_table(world: World, tick: int) -> Table:
    """创建状态表格"""
    stats = world.population.get_statistics()
    market = world.environment.get_market_summary()
    
    table = Table(title=f"🌍 世界状态 - Tick {tick}", show_header=True)
    table.add_column("指标", style="cyan")
    table.add_column("值", style="green")
    
    # 种群信息
    table.add_row("存活Agent", str(stats.alive))
    table.add_row("死亡Agent", str(stats.dead))
    table.add_row("DNA多样性", str(stats.unique_dna))
    table.add_row("可繁殖Agent", str(stats.agents_can_reproduce))
    table.add_row("总后代数", str(stats.total_offspring))
    
    # 代际信息
    if stats.generations:
        gen_range = f"{min(stats.generations.keys())} - {max(stats.generations.keys())}"
        table.add_row("代际范围", gen_range)
    
    # 资本信息
    table.add_row("平均资本", f"{stats.average_capital:.2f}")
    table.add_row("中位数资本", f"{stats.median_capital:.2f}")
    table.add_row("众数资本", f"{stats.mode_capital:.2f}")
    table.add_row("资本方差", f"{stats.variance_capital:.2f}")
    table.add_row("最高资本", f"{stats.max_capital:.2f}")
    table.add_row("最低资本", f"{stats.min_capital:.2f}")
    table.add_row("总资本", f"{stats.total_capital:.2f}")
    
    # 市场信息
    table.add_row("───────────", "───────────")
    table.add_row("市场价格", f"{market.get('price', 0):.2f}")
    table.add_row("波动率", f"{market.get('volatility', 0):.4f}")
    table.add_row("市场制度", market.get('regime', 'N/A'))
    
    return table


def run_demo(ticks: int = 1000, print_interval: int = 100, generate_report: bool = True):
    """
    运行演示
    
    Args:
        ticks: 运行的tick数
        print_interval: 打印间隔
        generate_report: 是否在结束时生成报告
    """
    print_banner()
    
    # 记录开始时间
    start_time = datetime.now()
    
    # 创建世界配置
    config = WorldConfig(
        initial_population_size=50,
        max_population_size=200,
        enable_carrying_capacity=True,
        reproduction_mode="mixed",
        mutation_rate=0.1,
        survival_threshold=100.0,
        initial_price=100.0,
        base_volatility=0.02,
        snapshot_interval=100,
    )
    
    console.print("\n📋 世界配置:", style="bold yellow")
    console.print(f"  • 初始种群: {config.initial_population_size}")
    console.print(f"  • 最大种群: {config.max_population_size}")
    console.print(f"  • 突变率: {config.mutation_rate}")
    console.print(f"  • 繁殖模式: {config.reproduction_mode}")
    console.print(f"  • 生存阈值: {config.survival_threshold}")
    
    # 创建世界
    console.print("\n🌍 创建世界...", style="bold")
    world = World(config)
    world.initialize()
    
    console.print(f"✅ 初始种群已创建: {world.population.alive_count} 个Agent\n")
    
    # 统计变量
    total_births = 0
    total_deaths = 0
    total_trades = 0
    regime_changes = []
    last_regime = None
    tick_results = []  # 保存所有tick结果用于报告
    
    # 运行模拟
    console.print(f"🚀 开始模拟 ({ticks} ticks)...\n", style="bold green")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("进化中...", total=ticks)

        for i in range(ticks):
            result = world.tick()
            tick_results.append(result)  # 保存结果用于报告

            total_births += result.births
            total_deaths += result.deaths
            total_trades += result.trades

            # 检测制度变化
            current_regime = result.market_state.get('regime')
            if current_regime != last_regime and last_regime is not None:
                regime_changes.append((result.tick, last_regime, current_regime))
            last_regime = current_regime

            # 定期打印状态
            if (i + 1) % print_interval == 0 or i == ticks - 1:
                progress.update(task, advance=print_interval if i > 0 else 1)

                # 获取种群统计
                stats = world.population.get_statistics()

                console.print(f"\n📊 Tick {result.tick}:")
                console.print(f"  存活: {result.alive_count} | "
                            f"出生: {result.births} | "
                            f"死亡: {result.deaths} | "
                            f"交易: {result.trades}")
                console.print(f"  价格: {result.market_state['price']:.2f} | "
                            f"制度: {result.market_state['regime']}")
                console.print(f"  资产统计: "
                            f"平均={stats.average_capital:.2f} | "
                            f"中位数={stats.median_capital:.2f} | "
                            f"众数={stats.mode_capital:.2f} | "
                            f"方差={stats.variance_capital:.2f}")

            # 检查种群灭绝
            if result.alive_count == 0:
                console.print("\n💀 种群灭绝!", style="bold red")
                break

            progress.update(task, completed=i + 1)
    
    # 记录结束时间
    end_time = datetime.now()

    # 打印最终结果
    console.print("\n" + "=" * 60)
    console.print("📈 模拟结束 - 最终统计", style="bold cyan")
    console.print("=" * 60)

    # 最终状态表格
    final_table = create_status_table(world, world.clock.current_tick)
    console.print(final_table)

    # 统计摘要
    summary_table = Table(title="📊 模拟统计", show_header=True)
    summary_table.add_column("统计项", style="cyan")
    summary_table.add_column("值", style="green")

    summary_table.add_row("总运行Tick", str(world.clock.current_tick))
    summary_table.add_row("总出生数", str(total_births))
    summary_table.add_row("总死亡数", str(total_deaths))
    summary_table.add_row("总交易数", str(total_trades))
    summary_table.add_row("制度变化次数", str(len(regime_changes)))

    console.print(summary_table)

    # 死亡原因分析
    graveyard_stats = world.graveyard.get_statistics()
    if graveyard_stats.get("causes"):
        death_table = Table(title="💀 死亡原因分析", show_header=True)
        death_table.add_column("死因", style="red")
        death_table.add_column("数量", style="yellow")

        for cause, count in graveyard_stats["causes"].items():
            death_table.add_row(cause, str(count))

        console.print(death_table)

    # 代际分布
    stats = world.population.get_statistics()
    if stats.generations:
        gen_table = Table(title="🧬 代际分布（存活）", show_header=True)
        gen_table.add_column("代数", style="cyan")
        gen_table.add_column("数量", style="green")

        for gen in sorted(stats.generations.keys()):
            gen_table.add_row(str(gen), str(stats.generations[gen]))

        console.print(gen_table)

    # 最优势DNA
    dominant_dna = world.population.get_dominant_dna(5)
    if dominant_dna:
        dna_table = Table(title="🏆 最优势DNA（按携带Agent数）", show_header=True)
        dna_table.add_column("DNA ID", style="cyan")
        dna_table.add_column("携带者数量", style="green")

        for dna_id, count in dominant_dna:
            dna_table.add_row(dna_id[:16] + "...", str(count))

        console.print(dna_table)

    # 生成详细报告
    if generate_report and tick_results:
        console.print("\n📝 生成详细报告...", style="bold")
        try:
            report_generator = ReportGenerator()
            report_dir = report_generator.generate_report(
                world=world,
                tick_results=tick_results,
                start_time=start_time,
                end_time=end_time,
                regime_changes=regime_changes,
            )

            console.print(f"✅ 报告已生成: {report_dir}", style="bold green")
            console.print(f"   • 包含完整数据分析和历史记录", style="dim")

        except Exception as e:
            console.print(f"❌ 报告生成失败: {e}", style="bold red")

    console.print("\n✨ 演示完成!", style="bold green")
    console.print("记住: 这不是寻找最优策略，而是观察自然选择的发生。\n")

    return world


def main():
    """主函数"""
    import argparse
    from config import load_config

    # 从配置文件加载默认值
    config = load_config("world_rules.yaml")
    simulation_config = config.get("simulation", {})
    default_ticks = simulation_config.get("default_ticks", 1000)
    default_interval = simulation_config.get("print_interval", 100)

    parser = argparse.ArgumentParser(description="运行进化金融系统演示")
    parser.add_argument("--ticks", type=int, default=default_ticks, help="运行的tick数")
    parser.add_argument("--interval", type=int, default=default_interval, help="打印间隔")
    parser.add_argument("--no-report", action="store_true", help="不生成详细报告")

    args = parser.parse_args()

    try:
        run_demo(
            ticks=args.ticks,
            print_interval=args.interval,
            generate_report=not args.no_report
        )
    except KeyboardInterrupt:
        console.print("\n\n⚠️ 用户中断模拟", style="bold yellow")
    except Exception as e:
        console.print(f"\n\n❌ 错误: {e}", style="bold red")
        raise


if __name__ == "__main__":
    main()
