"""
维度2：成交量与流动性维度数据获取测试（触觉+声音）

包含维度：
- 成交量：单位时间内成交股数
- 成交额：金额成交总量
- 买卖盘深度：各档挂单数量
- 点差：买一卖一价差
- 换手率：成交量/流通股本
- 大单/主力成交：大额买卖

Usage:
    python environment/test_dimension_2_volume_liquidity.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import sys
import os


def calculate_volume_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算成交量相关指标

    Args:
        df: 原始数据

    Returns:
        增强后的DataFrame
    """
    df = df.copy()

    # 成交量移动平均
    df['成交量_MA5'] = df['成交量'].rolling(window=5).mean()
    df['成交量_MA10'] = df['成交量'].rolling(window=10).mean()
    df['成交量_MA20'] = df['成交量'].rolling(window=20).mean()

    # 成交额移动平均
    df['成交额_MA5'] = df['成交额'].rolling(window=5).mean()
    df['成交额_MA10'] = df['成交额'].rolling(window=10).mean()
    df['成交额_MA20'] = df['成交额'].rolling(window=20).mean()

    # 量比（当日成交量/过去5日平均成交量）
    df['量比'] = df['成交量'] / df['成交量_MA5']
    df['量比'] = df['量比'].replace([np.inf, -np.inf], np.nan)

    # 成交额比
    df['额比'] = df['成交额'] / df['成交额_MA5']
    df['额比'] = df['额比'].replace([np.inf, -np.inf], np.nan)

    # 成交量变化率
    df['成交量变化率'] = df['成交量'].pct_change() * 100

    # 成交额变化率
    df['成交额变化率'] = df['成交额'].pct_change() * 100

    # 成交量标准差（波动性）
    df['成交量波动_5日'] = df['成交量'].rolling(window=5).std()
    df['成交量波动_10日'] = df['成交量'].rolling(window=10).std()

    # 成交量相对强度
    df['成交量相对强度'] = (df['成交量'] - df['成交量'].rolling(20).min()) / \
                        (df['成交量'].rolling(20).max() - df['成交量'].rolling(20).min())
    df['成交量相对强度'] = df['成交量相对强度'].replace([np.inf, -np.inf], np.nan)

    # 价量相关性（5日滚动）
    df['价量相关性_5日'] = df['收盘'].rolling(5).corr(df['成交量'])

    # 累计成交量（用于识别放量/缩量）
    df['累计成交量_5日'] = df['成交量'].rolling(5).sum()
    df['累计成交量_10日'] = df['成交量'].rolling(10).sum()

    # 成交量能量指标（价格变动 * 成交量）
    df['成交量能量'] = abs(df['收盘'].pct_change()) * df['成交量']

    # 换手率（如果有流通股本数据）
    if '换手率' in df.columns:
        df['换手率_MA5'] = df['换手率'].rolling(5).mean()
        df['换手率_MA10'] = df['换手率'].rolling(10).mean()
        df['换手率相对值'] = df['换手率'] / df['换手率_MA10']
        df['换手率相对值'] = df['换手率相对值'].replace([np.inf, -np.inf], np.nan)

    return df


def get_volume_liquidity_data(
    symbol: str = "600519",
    start_date: str = "20240101",
    end_date: str = "20241231",
    adjust: str = "qfq"
) -> Optional[pd.DataFrame]:
    """
    获取成交量与流动性数据

    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        adjust: 复权方式

    Returns:
        包含成交量指标的DataFrame
    """
    try:
        import akshare as ak

        print(f"正在获取股票 {symbol} 的成交量数据...")

        # 获取日线数据
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust
        )

        if df is None or df.empty:
            print("未获取到数据")
            return None

        # 计算成交量指标
        df = calculate_volume_metrics(df)

        return df

    except ImportError:
        print("错误：需要安装 akshare 库")
        print("请运行：pip install akshare")
        return None
    except Exception as e:
        print(f"获取数据失败：{e}")
        import traceback
        traceback.print_exc()
        return None


def get_realtime_bid_ask(symbol: str = "600519") -> Optional[pd.DataFrame]:
    """
    获取实时买卖盘口数据（五档行情）

    Args:
        symbol: 股票代码

    Returns:
        实时盘口数据
    """
    try:
        import akshare as ak

        print(f"正在获取股票 {symbol} 的实时盘口数据...")

        # 获取实时行情
        df = ak.stock_zh_a_spot_em()

        # 筛选指定股票
        stock_data = df[df['代码'] == symbol]

        if stock_data.empty:
            print(f"未找到股票 {symbol} 的实时数据")
            return None

        return stock_data

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取实时数据失败：{e}")
        return None


def get_money_flow_data(symbol: str = "600519") -> Optional[pd.DataFrame]:
    """
    获取资金流向数据（主力/大单/中单/小单）

    Args:
        symbol: 股票代码

    Returns:
        资金流向数据
    """
    try:
        import akshare as ak

        print(f"正在获取股票 {symbol} 的资金流向数据...")

        # 获取个股资金流
        df = ak.stock_individual_fund_flow_rank(symbol=symbol)

        return df

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取资金流向数据失败：{e}")
        return None


def get_tick_data(symbol: str = "600519", trade_date: str = None) -> Optional[pd.DataFrame]:
    """
    获取分笔成交数据（Tick数据）

    Args:
        symbol: 股票代码
        trade_date: 交易日期，格式：YYYYMMDD

    Returns:
        分笔数据
    """
    try:
        import akshare as ak

        if trade_date is None:
            trade_date = datetime.now().strftime('%Y%m%d')

        print(f"正在获取股票 {symbol} 在 {trade_date} 的分笔数据...")

        # 获取分笔数据
        df = ak.stock_zh_a_hist_min_em(
            symbol=symbol,
            period='1',
            start_date=trade_date + ' 09:30:00',
            end_date=trade_date + ' 15:00:00',
            adjust='qfq'
        )

        if df is not None and not df.empty:
            # 计算分笔成交量指标
            df['成交量累计'] = df['成交量'].cumsum()
            df['成交额累计'] = df['成交额'].cumsum()
            df['均价'] = df['成交额累计'] / df['成交量累计']
            df['均价'] = df['均价'].replace([np.inf, -np.inf], np.nan)

        return df

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取分笔数据失败：{e}")
        return None


def get_market_turnover_data(
    start_date: str = "20240101",
    end_date: str = "20241231"
) -> Optional[pd.DataFrame]:
    """
    获取市场整体成交数据

    Args:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        市场成交数据
    """
    try:
        import akshare as ak

        print("正在获取市场整体成交数据...")

        # 获取沪深市场总貌
        df = ak.stock_sse_summary()

        return df

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取市场数据失败：{e}")
        return None


def print_data_summary(df: pd.DataFrame, title: str = "数据摘要"):
    """打印数据摘要"""
    if df is None or df.empty:
        print(f"{title}: 无数据")
        return

    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(f"数据行数: {len(df)}")

    date_col = '日期' if '日期' in df.columns else ('时间' if '时间' in df.columns else None)
    if date_col and date_col in df.columns:
        print(f"日期范围: {df[date_col].min()} 至 {df[date_col].max()}")

    print(f"\n列名: {list(df.columns)}")
    print(f"\n前5行数据:")
    print(df.head())
    print(f"\n数据统计:")
    print(df.describe())
    print(f"{'='*80}\n")


def main():
    """主测试函数"""
    print("="*80)
    print("维度2：成交量与流动性维度数据测试（触觉+声音）")
    print("="*80)

    # 输出文件名
    excel_file = "dimension_2_volume_liquidity_data.xlsx"

    # 存储所有数据
    all_data = {}

    # 测试1: 获取茅台成交量数据
    print("\n[测试1] 获取贵州茅台(600519)成交量与流动性数据")
    stock_data = get_volume_liquidity_data(
        symbol="600519",
        start_date="20240101",
        end_date="20241231",
        adjust="qfq"
    )

    if stock_data is not None:
        print_data_summary(stock_data, "茅台成交量数据（含流动性指标）")
        all_data['茅台_成交量'] = stock_data

    # 测试2: 获取平安银行成交量数据
    print("\n[测试2] 获取平安银行(000001)成交量数据")
    pa_data = get_volume_liquidity_data(
        symbol="000001",
        start_date="20240101",
        end_date="20241231",
        adjust="qfq"
    )

    if pa_data is not None:
        print_data_summary(pa_data, "平安银行成交量数据")
        all_data['平安银行_成交量'] = pa_data

    # 测试3: 获取实时盘口数据
    print("\n[测试3] 获取实时买卖盘口数据")
    realtime_data = get_realtime_bid_ask(symbol="600519")

    if realtime_data is not None:
        print_data_summary(realtime_data, "实时盘口数据")
        all_data['实时盘口'] = realtime_data

    # 测试4: 获取资金流向数据
    print("\n[测试4] 获取资金流向数据（主力/大单）")
    money_flow = get_money_flow_data(symbol="600519")

    if money_flow is not None:
        print_data_summary(money_flow, "资金流向数据")
        all_data['资金流向'] = money_flow

    # 测试5: 获取市场整体成交数据
    print("\n[测试5] 获取市场整体成交数据")
    market_data = get_market_turnover_data()

    if market_data is not None:
        print_data_summary(market_data, "市场整体成交数据")
        all_data['市场成交'] = market_data

    # 保存所有数据到Excel
    if all_data:
        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                for sheet_name, df in all_data.items():
                    # Excel工作表名称限制
                    safe_name = sheet_name[:31]
                    df.to_excel(writer, sheet_name=safe_name, index=False)
                    print(f"已保存工作表: {safe_name}")

            print(f"\n所有数据已保存至: {excel_file}")
            print(f"文件位置: {os.path.abspath(excel_file)}")

        except ImportError:
            print("警告：需要安装 openpyxl 库来保存Excel文件")
            print("请运行：pip install openpyxl")
        except Exception as e:
            print(f"保存Excel文件失败：{e}")
            import traceback
            traceback.print_exc()
    else:
        print("没有数据可保存")

    print("\n测试完成！")
    print("\n成交量与流动性维度说明：")
    print("- 成交量/成交额：市场交易活跃度")
    print("- 量比/额比：相对历史平均的交易强度")
    print("- 成交量波动：交易稳定性")
    print("- 价量相关性：价格与成交量的关系")
    print("- 换手率：流通性指标")
    print("- 买卖盘深度：市场流动性和阻力")
    print("- 资金流向：主力/散户资金动向")


if __name__ == "__main__":
    main()
