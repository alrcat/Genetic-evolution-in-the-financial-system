"""
维度1：价格相关维度数据获取测试（基础光感）

包含维度：
- 当前价格/最新价：最近成交价
- 开盘价：当日第一笔成交
- 收盘价：当日最后成交
- 最高价/最低价：当日价格极值
- 前收盘价：昨日收盘价
- 涨跌额/涨跌幅：与前收盘比较
- 均价：成交加权平均价
- OHLCK：日/周/月开高低收及成交量

Usage:
    python environment/test_dimension_1_price.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import sys
import os


def calculate_price_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算价格相关指标

    Args:
        df: 原始价格数据

    Returns:
        增强后的DataFrame
    """
    df = df.copy()

    # 基础价格维度已包含在原始数据中：
    # - 开盘、收盘、最高、最低

    # 前收盘价
    df['前收盘'] = df['收盘'].shift(1)

    # 涨跌额
    df['涨跌额'] = df['收盘'] - df['前收盘']

    # 涨跌幅 (%)
    df['涨跌幅'] = (df['涨跌额'] / df['前收盘']) * 100

    # 当日振幅 (%)
    df['振幅'] = ((df['最高'] - df['最低']) / df['前收盘']) * 100

    # 均价（成交额/成交量）
    df['均价'] = df['成交额'] / df['成交量']
    df['均价'] = df['均价'].replace([np.inf, -np.inf], np.nan)

    # 价格位置（当前价格在最高最低之间的位置，0-1）
    df['价格位置'] = (df['收盘'] - df['最低']) / (df['最高'] - df['最低'])
    df['价格位置'] = df['价格位置'].replace([np.inf, -np.inf], np.nan)

    # 上影线比例
    df['上影线'] = df['最高'] - df[['开盘', '收盘']].max(axis=1)
    df['上影线比例'] = df['上影线'] / (df['最高'] - df['最低'])
    df['上影线比例'] = df['上影线比例'].replace([np.inf, -np.inf], np.nan)

    # 下影线比例
    df['下影线'] = df[['开盘', '收盘']].min(axis=1) - df['最低']
    df['下影线比例'] = df['下影线'] / (df['最高'] - df['最低'])
    df['下影线比例'] = df['下影线比例'].replace([np.inf, -np.inf], np.nan)

    # 实体比例（开盘收盘差/振幅）
    df['实体'] = abs(df['收盘'] - df['开盘'])
    df['实体比例'] = df['实体'] / (df['最高'] - df['最低'])
    df['实体比例'] = df['实体比例'].replace([np.inf, -np.inf], np.nan)

    # 涨跌状态（1=阳线，-1=阴线，0=平）
    df['涨跌状态'] = np.sign(df['收盘'] - df['开盘'])

    # 价格距离高点
    df['距20日高点'] = (df['最高'].rolling(20).max() - df['收盘']) / df['收盘'] * 100

    # 价格距离低点
    df['距20日低点'] = (df['收盘'] - df['最低'].rolling(20).min()) / df['收盘'] * 100

    return df


def calculate_weekly_monthly_ohlc(df: pd.DataFrame) -> dict:
    """
    计算周线和月线OHLCK数据

    Args:
        df: 日线数据

    Returns:
        包含周线和月线数据的字典
    """
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df.set_index('日期', inplace=True)

    # 周线数据
    weekly = pd.DataFrame()
    weekly['开盘'] = df['开盘'].resample('W').first()
    weekly['最高'] = df['最高'].resample('W').max()
    weekly['最低'] = df['最低'].resample('W').min()
    weekly['收盘'] = df['收盘'].resample('W').last()
    weekly['成交量'] = df['成交量'].resample('W').sum()
    weekly['成交额'] = df['成交额'].resample('W').sum()
    weekly['前收盘'] = weekly['收盘'].shift(1)
    weekly['涨跌额'] = weekly['收盘'] - weekly['前收盘']
    weekly['涨跌幅'] = (weekly['涨跌额'] / weekly['前收盘']) * 100

    # 月线数据
    monthly = pd.DataFrame()
    monthly['开盘'] = df['开盘'].resample('M').first()
    monthly['最高'] = df['最高'].resample('M').max()
    monthly['最低'] = df['最低'].resample('M').min()
    monthly['收盘'] = df['收盘'].resample('M').last()
    monthly['成交量'] = df['成交量'].resample('M').sum()
    monthly['成交额'] = df['成交额'].resample('M').sum()
    monthly['前收盘'] = monthly['收盘'].shift(1)
    monthly['涨跌额'] = monthly['收盘'] - monthly['前收盘']
    monthly['涨跌幅'] = (monthly['涨跌额'] / monthly['前收盘']) * 100

    weekly.reset_index(inplace=True)
    monthly.reset_index(inplace=True)

    return {
        'weekly': weekly,
        'monthly': monthly
    }


def get_price_dimension_data(
    symbol: str = "600519",
    start_date: str = "20240101",
    end_date: str = "20241231",
    adjust: str = "qfq"
) -> Optional[dict]:
    """
    获取价格维度相关数据

    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        adjust: 复权方式

    Returns:
        包含日线、周线、月线数据的字典
    """
    try:
        import akshare as ak

        print(f"正在获取股票 {symbol} 的价格数据...")

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

        # 计算价格指标
        df_daily = calculate_price_metrics(df)

        # 计算周线和月线
        period_data = calculate_weekly_monthly_ohlc(df)

        return {
            'daily': df_daily,
            'weekly': period_data['weekly'],
            'monthly': period_data['monthly']
        }

    except ImportError:
        print("错误：需要安装 akshare 库")
        print("请运行：pip install akshare")
        return None
    except Exception as e:
        print(f"获取数据失败：{e}")
        import traceback
        traceback.print_exc()
        return None


def get_index_price_data(
    symbol: str = "000001",
    start_date: str = "20240101",
    end_date: str = "20241231"
) -> Optional[dict]:
    """
    获取指数价格数据

    Args:
        symbol: 指数代码
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        包含日线、周线、月线数据的字典
    """
    try:
        import akshare as ak

        print(f"正在获取指数 {symbol} 的价格数据...")

        # 确定市场前缀
        if symbol.startswith('000'):
            em_symbol = f"sh{symbol}"
        else:
            em_symbol = f"sz{symbol}"

        # 获取指数日线数据
        df = ak.stock_zh_index_daily_em(
            symbol=em_symbol,
            start_date=start_date,
            end_date=end_date
        )

        if df is None or df.empty:
            print("未获取到数据")
            return None

        # 重命名列以匹配股票数据格式
        df = df.rename(columns={
            'date': '日期',
            'open': '开盘',
            'close': '收盘',
            'high': '最高',
            'low': '最低',
            'volume': '成交量',
            'amount': '成交额'
        })

        # 计算价格指标
        df_daily = calculate_price_metrics(df)

        # 计算周线和月线
        period_data = calculate_weekly_monthly_ohlc(df)

        return {
            'daily': df_daily,
            'weekly': period_data['weekly'],
            'monthly': period_data['monthly']
        }

    except ImportError:
        print("错误：需要安装 akshare 库")
        print("请运行：pip install akshare")
        return None
    except Exception as e:
        print(f"获取数据失败：{e}")
        import traceback
        traceback.print_exc()
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

    date_col = '日期' if '日期' in df.columns else df.index.name
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
    print("维度1：价格相关维度数据测试（基础光感）")
    print("="*80)

    # 输出文件名
    excel_file = "dimension_1_price_data.xlsx"

    # 存储所有数据
    all_data = {}

    # 测试1: 获取茅台股票价格数据（日线+周线+月线）
    print("\n[测试1] 获取贵州茅台(600519)价格维度数据")
    stock_data = get_price_dimension_data(
        symbol="600519",
        start_date="20240101",
        end_date="20241231",
        adjust="qfq"
    )

    if stock_data:
        print_data_summary(stock_data['daily'], "茅台日线数据（含价格指标）")
        print_data_summary(stock_data['weekly'], "茅台周线数据")
        print_data_summary(stock_data['monthly'], "茅台月线数据")

        all_data['茅台_日线'] = stock_data['daily']
        all_data['茅台_周线'] = stock_data['weekly']
        all_data['茅台_月线'] = stock_data['monthly']

    # 测试2: 获取上证指数价格数据
    print("\n[测试2] 获取上证指数(000001)价格维度数据")
    index_data = get_index_price_data(
        symbol="000001",
        start_date="20240101",
        end_date="20241231"
    )

    if index_data:
        print_data_summary(index_data['daily'], "上证指数日线数据")
        print_data_summary(index_data['weekly'], "上证指数周线数据")
        print_data_summary(index_data['monthly'], "上证指数月线数据")

        all_data['上证_日线'] = index_data['daily']
        all_data['上证_周线'] = index_data['weekly']
        all_data['上证_月线'] = index_data['monthly']

    # 测试3: 获取平安银行数据
    print("\n[测试3] 获取平安银行(000001)价格维度数据")
    pa_data = get_price_dimension_data(
        symbol="000001",
        start_date="20240101",
        end_date="20241231",
        adjust="qfq"
    )

    if pa_data:
        print_data_summary(pa_data['daily'], "平安银行日线数据")
        all_data['平安银行_日线'] = pa_data['daily']
        all_data['平安银行_周线'] = pa_data['weekly']
        all_data['平安银行_月线'] = pa_data['monthly']

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
    print("\n价格维度说明：")
    print("- 前收盘：昨日收盘价，用于计算涨跌")
    print("- 涨跌额/涨跌幅：当日价格变动")
    print("- 振幅：当日波动范围")
    print("- 均价：成交加权平均价")
    print("- 价格位置：收盘价在当日高低点之间的位置")
    print("- 上/下影线比例：K线形态分析")
    print("- 实体比例：K线实体占比")
    print("- 涨跌状态：阳线/阴线标识")
    print("- 距高低点：价格相对位置")


if __name__ == "__main__":
    main()
