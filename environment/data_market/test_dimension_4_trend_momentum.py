"""
维度4：趋势与动量维度数据获取测试（光+节奏感）

包含维度：
- 均线（MA、EMA等）：平滑价格
- MACD：动量趋势指标
- RSI：超买超卖
- KDJ：随机指标
- ADX：趋势强度

Usage:
    python environment/test_dimension_4_trend_momentum.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import sys
import os


def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算各种移动平均线

    Args:
        df: 原始数据

    Returns:
        增强后的DataFrame
    """
    df = df.copy()

    # 简单移动平均（SMA）
    for period in [5, 10, 20, 30, 60, 120, 250]:
        df[f'MA{period}'] = df['收盘'].rolling(window=period).mean()

    # 指数移动平均（EMA）
    for period in [5, 10, 20, 30, 60]:
        df[f'EMA{period}'] = df['收盘'].ewm(span=period, adjust=False).mean()

    # 加权移动平均（WMA）
    def wma(series, period):
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    df['WMA10'] = wma(df['收盘'], 10)
    df['WMA20'] = wma(df['收盘'], 20)

    # 均线多头/空头排列判断
    df['均线多头'] = (df['MA5'] > df['MA10']) & (df['MA10'] > df['MA20']) & (df['MA20'] > df['MA60'])
    df['均线空头'] = (df['MA5'] < df['MA10']) & (df['MA10'] < df['MA20']) & (df['MA20'] < df['MA60'])

    # 价格与均线的偏离度
    df['偏离MA5'] = ((df['收盘'] - df['MA5']) / df['MA5']) * 100
    df['偏离MA20'] = ((df['收盘'] - df['MA20']) / df['MA20']) * 100
    df['偏离MA60'] = ((df['收盘'] - df['MA60']) / df['MA60']) * 100

    return df


def calculate_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """
    计算MACD指标

    Args:
        df: 原始数据
        fast: 快线周期
        slow: 慢线周期
        signal: 信号线周期

    Returns:
        增强后的DataFrame
    """
    df = df.copy()

    # 计算EMA
    ema_fast = df['收盘'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['收盘'].ewm(span=slow, adjust=False).mean()

    # MACD线
    df['MACD'] = ema_fast - ema_slow

    # 信号线
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()

    # MACD柱
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # MACD金叉死叉
    df['MACD金叉'] = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
    df['MACD死叉'] = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))

    return df


def calculate_rsi(df: pd.DataFrame, period=14) -> pd.DataFrame:
    """
    计算RSI指标

    Args:
        df: 原始数据
        period: 周期

    Returns:
        增强后的DataFrame
    """
    df = df.copy()

    # 价格变化
    delta = df['收盘'].diff()

    # 涨跌分离
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # 计算平均涨跌
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # RS和RSI
    rs = avg_gain / avg_loss
    df[f'RSI{period}'] = 100 - (100 / (1 + rs))

    # RSI超买超卖信号
    df['RSI超买'] = df[f'RSI{period}'] > 70
    df['RSI超卖'] = df[f'RSI{period}'] < 30

    # 多周期RSI
    for p in [6, 14, 24]:
        delta = df['收盘'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=p).mean()
        avg_loss = loss.rolling(window=p).mean()
        rs = avg_gain / avg_loss
        df[f'RSI{p}'] = 100 - (100 / (1 + rs))

    return df


def calculate_kdj(df: pd.DataFrame, n=9, m1=3, m2=3) -> pd.DataFrame:
    """
    计算KDJ指标

    Args:
        df: 原始数据
        n: RSV周期
        m1: K值平滑周期
        m2: D值平滑周期

    Returns:
        增强后的DataFrame
    """
    df = df.copy()

    # 计算RSV
    low_min = df['最低'].rolling(window=n).min()
    high_max = df['最高'].rolling(window=n).max()

    rsv = ((df['收盘'] - low_min) / (high_max - low_min)) * 100
    rsv = rsv.fillna(50)  # 初始值设为50

    # 计算K值
    df['K'] = rsv.ewm(alpha=1/m1, adjust=False).mean()

    # 计算D值
    df['D'] = df['K'].ewm(alpha=1/m2, adjust=False).mean()

    # 计算J值
    df['J'] = 3 * df['K'] - 2 * df['D']

    # KDJ金叉死叉
    df['KDJ金叉'] = (df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))
    df['KDJ死叉'] = (df['K'] < df['D']) & (df['K'].shift(1) >= df['D'].shift(1))

    # KDJ超买超卖
    df['KDJ超买'] = (df['K'] > 80) & (df['D'] > 80)
    df['KDJ超卖'] = (df['K'] < 20) & (df['D'] < 20)

    return df


def calculate_adx(df: pd.DataFrame, period=14) -> pd.DataFrame:
    """
    计算ADX（Average Directional Index）趋势强度指标

    Args:
        df: 原始数据
        period: 周期

    Returns:
        增强后的DataFrame
    """
    df = df.copy()

    # 计算+DM和-DM
    high_diff = df['最高'].diff()
    low_diff = -df['最低'].diff()

    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

    # 计算TR（True Range）
    high_low = df['最高'] - df['最低']
    high_close = np.abs(df['最高'] - df['收盘'].shift(1))
    low_close = np.abs(df['最低'] - df['收盘'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # 平滑+DM, -DM, TR
    atr = tr.rolling(window=period).mean()
    plus_dm_smooth = plus_dm.rolling(window=period).mean()
    minus_dm_smooth = minus_dm.rolling(window=period).mean()

    # 计算+DI和-DI
    df['plusDI'] = (plus_dm_smooth / atr) * 100
    df['minusDI'] = (minus_dm_smooth / atr) * 100

    # 计算DX
    dx = (np.abs(df['plusDI'] - df['minusDI']) / (df['plusDI'] + df['minusDI'])) * 100

    # 计算ADX
    df['ADX'] = dx.rolling(window=period).mean()

    # ADX趋势强度判断
    df['强趋势'] = df['ADX'] > 25
    df['弱趋势'] = df['ADX'] < 20

    return df


def calculate_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算动量指标

    Args:
        df: 原始数据

    Returns:
        增强后的DataFrame
    """
    df = df.copy()

    # 价格动量（Price Momentum）
    for period in [5, 10, 20]:
        df[f'动量{period}日'] = df['收盘'] - df['收盘'].shift(period)
        df[f'动量{period}日_pct'] = (df['收盘'] / df['收盘'].shift(period) - 1) * 100

    # 变化率（ROC - Rate of Change）
    for period in [5, 10, 20]:
        df[f'ROC{period}'] = ((df['收盘'] - df['收盘'].shift(period)) / df['收盘'].shift(period)) * 100

    # 威廉指标（Williams %R）
    for period in [14, 20]:
        high_max = df['最高'].rolling(window=period).max()
        low_min = df['最低'].rolling(window=period).min()
        df[f'WilliamsR{period}'] = ((high_max - df['收盘']) / (high_max - low_min)) * -100

    # CCI（Commodity Channel Index）
    period = 20
    tp = (df['最高'] + df['最低'] + df['收盘']) / 3  # 典型价格
    ma_tp = tp.rolling(window=period).mean()
    md = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    df['CCI'] = (tp - ma_tp) / (0.015 * md)

    return df


def calculate_trend_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算综合趋势与动量指标

    Args:
        df: 原始数据

    Returns:
        增强后的DataFrame
    """
    df = df.copy()

    # 计算各类指标
    df = calculate_moving_averages(df)
    df = calculate_macd(df)
    df = calculate_rsi(df)
    df = calculate_kdj(df)
    df = calculate_adx(df)
    df = calculate_momentum_indicators(df)

    return df


def get_trend_momentum_data(
    symbol: str = "600519",
    start_date: str = "20240101",
    end_date: str = "20241231",
    adjust: str = "qfq"
) -> Optional[pd.DataFrame]:
    """
    获取趋势与动量数据

    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        adjust: 复权方式

    Returns:
        包含趋势动量指标的DataFrame
    """
    try:
        import akshare as ak

        print(f"正在获取股票 {symbol} 的数据...")

        # 获取股票数据
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

        # 计算趋势和动量指标
        df = calculate_trend_metrics(df)

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


def print_data_summary(df: pd.DataFrame, title: str = "数据摘要"):
    """打印数据摘要"""
    if df is None or df.empty:
        print(f"{title}: 无数据")
        return

    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(f"数据行数: {len(df)}")

    if '日期' in df.columns:
        print(f"日期范围: {df['日期'].min()} 至 {df['日期'].max()}")

    print(f"\n列名（共{len(df.columns)}列）:")
    print(list(df.columns))

    print(f"\n最近5天数据:")
    key_cols = ['日期', '收盘', 'MA5', 'MA20', 'MACD', 'RSI14', 'K', 'D', 'ADX']
    available_cols = [col for col in key_cols if col in df.columns]
    print(df[available_cols].tail())

    print(f"{'='*80}\n")


def main():
    """主测试函数"""
    print("="*80)
    print("维度4：趋势与动量维度数据测试（光+节奏感）")
    print("="*80)

    # 输出文件名
    excel_file = "dimension_4_trend_momentum_data.xlsx"

    # 存储所有数据
    all_data = {}

    # 测试1: 获取茅台的趋势动量数据
    print("\n[测试1] 获取贵州茅台(600519)趋势与动量数据")
    moutai_data = get_trend_momentum_data(
        symbol="600519",
        start_date="20240101",
        end_date="20241231",
        adjust="qfq"
    )

    if moutai_data is not None:
        print_data_summary(moutai_data, "茅台趋势动量指标")
        all_data['茅台_趋势动量'] = moutai_data

    # 测试2: 获取平安银行数据
    print("\n[测试2] 获取平安银行(000001)趋势与动量数据")
    pa_data = get_trend_momentum_data(
        symbol="000001",
        start_date="20240101",
        end_date="20241231",
        adjust="qfq"
    )

    if pa_data is not None:
        print_data_summary(pa_data, "平安银行趋势动量指标")
        all_data['平安银行_趋势动量'] = pa_data

    # 测试3: 获取宁德时代数据
    print("\n[测试3] 获取宁德时代(300750)趋势与动量数据")
    catl_data = get_trend_momentum_data(
        symbol="300750",
        start_date="20240101",
        end_date="20241231",
        adjust="qfq"
    )

    if catl_data is not None:
        print_data_summary(catl_data, "宁德时代趋势动量指标")
        all_data['宁德时代_趋势动量'] = catl_data

    # 保存所有数据到Excel
    if all_data:
        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                for sheet_name, df in all_data.items():
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
    print("\n趋势与动量维度说明：")
    print("- 移动平均线(MA/EMA)：识别价格趋势方向")
    print("- MACD：捕捉动量变化和趋势转折")
    print("- RSI：判断超买超卖状态")
    print("- KDJ：短期买卖信号")
    print("- ADX：衡量趋势强度")
    print("- ROC/Williams R/CCI：多角度动量分析")


if __name__ == "__main__":
    main()
