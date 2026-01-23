"""
维度3：波动性与风险维度数据获取测试（温度感知）

包含维度：
- 历史波动率：一段时间内价格标准差
- 隐含波动率：期权反映未来波动
- Beta：与市场整体相关性
- 最大回撤：历史最大下跌幅度
- ATR（平均真实波幅）：考虑跳空的波动

Usage:
    python environment/test_dimension_3_volatility_risk.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
import sys
import os


def calculate_historical_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    计算历史波动率（年化）

    Args:
        prices: 价格序列
        window: 窗口期

    Returns:
        波动率序列
    """
    returns = prices.pct_change()
    volatility = returns.rolling(window).std() * np.sqrt(252)  # 年化
    return volatility


def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series, window: int = 60) -> pd.Series:
    """
    计算Beta系数（滚动窗口）

    Args:
        stock_returns: 股票收益率
        market_returns: 市场收益率
        window: 窗口期

    Returns:
        Beta序列
    """
    beta = stock_returns.rolling(window).cov(market_returns) / market_returns.rolling(window).var()
    return beta


def calculate_max_drawdown(prices: pd.Series, window: int = None) -> pd.Series:
    """
    计算最大回撤

    Args:
        prices: 价格序列
        window: 窗口期，None表示从开始计算

    Returns:
        最大回撤序列
    """
    if window is None:
        # 计算累计最大回撤
        cummax = prices.expanding().max()
    else:
        # 计算滚动窗口最大回撤
        cummax = prices.rolling(window).max()

    drawdown = (prices - cummax) / cummax
    return drawdown


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    计算平均真实波幅（ATR）

    Args:
        high: 最高价
        low: 最低价
        close: 收盘价
        period: 周期

    Returns:
        ATR序列
    """
    high_low = high - low
    high_close = np.abs(high - close.shift(1))
    low_close = np.abs(low - close.shift(1))

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()

    return atr


def calculate_risk_metrics(df: pd.DataFrame, market_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    计算风险相关指标

    Args:
        df: 股票数据
        market_df: 市场指数数据（用于计算Beta）

    Returns:
        增强后的DataFrame
    """
    df = df.copy()

    # 1. 历史波动率（多个周期）
    df['波动率_5日'] = calculate_historical_volatility(df['收盘'], 5)
    df['波动率_10日'] = calculate_historical_volatility(df['收盘'], 10)
    df['波动率_20日'] = calculate_historical_volatility(df['收盘'], 20)
    df['波动率_60日'] = calculate_historical_volatility(df['收盘'], 60)

    # 2. ATR（平均真实波幅）
    df['ATR_14'] = calculate_atr(df['最高'], df['最低'], df['收盘'], 14)
    df['ATR_20'] = calculate_atr(df['最高'], df['最低'], df['收盘'], 20)

    # ATR百分比（相对于价格）
    df['ATR_14_pct'] = (df['ATR_14'] / df['收盘']) * 100
    df['ATR_20_pct'] = (df['ATR_20'] / df['收盘']) * 100

    # 3. 最大回撤
    df['最大回撤_20日'] = calculate_max_drawdown(df['收盘'], 20) * 100
    df['最大回撤_60日'] = calculate_max_drawdown(df['收盘'], 60) * 100
    df['最大回撤_累计'] = calculate_max_drawdown(df['收盘'], None) * 100

    # 4. 收益率
    df['日收益率'] = df['收盘'].pct_change() * 100
    df['对数收益率'] = np.log(df['收盘'] / df['收盘'].shift(1)) * 100

    # 5. 下行波动率（只考虑负收益）
    returns = df['收盘'].pct_change()
    downside_returns = returns.copy()
    downside_returns[downside_returns > 0] = 0
    df['下行波动率_20日'] = downside_returns.rolling(20).std() * np.sqrt(252) * 100

    # 6. Sharpe比率估计（假设无风险利率3%）
    risk_free_rate = 0.03 / 252  # 日化
    excess_returns = returns - risk_free_rate
    df['Sharpe_20日'] = (excess_returns.rolling(20).mean() / returns.rolling(20).std()) * np.sqrt(252)
    df['Sharpe_60日'] = (excess_returns.rolling(60).mean() / returns.rolling(60).std()) * np.sqrt(252)

    # 7. 最大单日跌幅（滚动窗口）
    df['最大单日跌幅_20日'] = df['日收益率'].rolling(20).min()
    df['最大单日涨幅_20日'] = df['日收益率'].rolling(20).max()

    # 8. 价格波动范围
    df['高低波动幅度'] = ((df['最高'] - df['最低']) / df['收盘']) * 100

    # 9. 收益率标准差
    df['收益率std_20日'] = df['日收益率'].rolling(20).std()
    df['收益率std_60日'] = df['日收益率'].rolling(60).std()

    # 10. VaR (Value at Risk) - 95%置信度
    df['VaR_95_20日'] = df['日收益率'].rolling(20).quantile(0.05)
    df['VaR_95_60日'] = df['日收益率'].rolling(60).quantile(0.05)

    # 11. Beta系数（如果提供了市场数据）
    if market_df is not None:
        try:
            # 对齐日期
            market_df = market_df.copy()
            market_df['日期'] = pd.to_datetime(market_df['日期'])
            df['日期'] = pd.to_datetime(df['日期'])

            merged = pd.merge(df[['日期', '收盘']],
                            market_df[['日期', '收盘']],
                            on='日期',
                            suffixes=('_stock', '_market'))

            stock_returns = merged['收盘_stock'].pct_change()
            market_returns = merged['收盘_market'].pct_change()

            beta_20 = calculate_beta(stock_returns, market_returns, 20)
            beta_60 = calculate_beta(stock_returns, market_returns, 60)

            df['Beta_20日'] = beta_20
            df['Beta_60日'] = beta_60

            # 相关系数
            df['市场相关性_20日'] = stock_returns.rolling(20).corr(market_returns)
            df['市场相关性_60日'] = stock_returns.rolling(60).corr(market_returns)
        except Exception as e:
            print(f"计算Beta时出错：{e}")

    return df


def get_volatility_risk_data(
    symbol: str = "600519",
    start_date: str = "20240101",
    end_date: str = "20241231",
    market_symbol: str = "000001",
    adjust: str = "qfq"
) -> Optional[dict]:
    """
    获取波动性与风险数据

    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        market_symbol: 市场指数代码（用于计算Beta）
        adjust: 复权方式

    Returns:
        包含风险指标的数据字典
    """
    try:
        import akshare as ak

        print(f"正在获取股票 {symbol} 的数据...")

        # 获取股票数据
        df_stock = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust
        )

        if df_stock is None or df_stock.empty:
            print("未获取到股票数据")
            return None

        # 获取市场指数数据（用于计算Beta）
        print(f"正在获取市场指数 {market_symbol} 的数据...")

        em_symbol = f"sh{market_symbol}" if market_symbol.startswith('000') else f"sz{market_symbol}"

        df_market = ak.stock_zh_index_daily_em(
            symbol=em_symbol,
            start_date=start_date,
            end_date=end_date
        )

        if df_market is not None and not df_market.empty:
            df_market = df_market.rename(columns={
                'date': '日期',
                'close': '收盘'
            })
        else:
            print("警告：未获取到市场数据，将不计算Beta")
            df_market = None

        # 计算风险指标
        df_result = calculate_risk_metrics(df_stock, df_market)

        return {
            'stock_data': df_result,
            'market_data': df_market
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

    if '日期' in df.columns:
        print(f"日期范围: {df['日期'].min()} 至 {df['日期'].max()}")

    print(f"\n列名: {list(df.columns)}")
    print(f"\n前5行数据:")
    print(df.head())

    # 显示关键风险指标的统计
    risk_cols = [col for col in df.columns if any(x in col for x in ['波动率', 'ATR', '回撤', 'Beta', 'Sharpe', 'VaR'])]
    if risk_cols:
        print(f"\n风险指标统计:")
        print(df[risk_cols].describe())

    print(f"{'='*80}\n")


def main():
    """主测试函数"""
    print("="*80)
    print("维度3：波动性与风险维度数据测试（温度感知）")
    print("="*80)

    # 输出文件名
    excel_file = "dimension_3_volatility_risk_data.xlsx"

    # 存储所有数据
    all_data = {}

    # 测试1: 获取茅台的波动性和风险数据
    print("\n[测试1] 获取贵州茅台(600519)波动性与风险数据")
    moutai_data = get_volatility_risk_data(
        symbol="600519",
        start_date="20240101",
        end_date="20241231",
        market_symbol="000001",
        adjust="qfq"
    )

    if moutai_data:
        print_data_summary(moutai_data['stock_data'], "茅台风险指标数据")
        all_data['茅台_风险指标'] = moutai_data['stock_data']
        if moutai_data['market_data'] is not None:
            all_data['上证指数'] = moutai_data['market_data']

    # 测试2: 获取平安银行的风险数据
    print("\n[测试2] 获取平安银行(000001)波动性与风险数据")
    pa_data = get_volatility_risk_data(
        symbol="000001",
        start_date="20240101",
        end_date="20241231",
        market_symbol="399001",  # 深证成指
        adjust="qfq"
    )

    if pa_data:
        print_data_summary(pa_data['stock_data'], "平安银行风险指标数据")
        all_data['平安银行_风险指标'] = pa_data['stock_data']

    # 测试3: 获取比亚迪的风险数据
    print("\n[测试3] 获取比亚迪(002594)波动性与风险数据")
    byd_data = get_volatility_risk_data(
        symbol="002594",
        start_date="20240101",
        end_date="20241231",
        market_symbol="399001",
        adjust="qfq"
    )

    if byd_data:
        print_data_summary(byd_data['stock_data'], "比亚迪风险指标数据")
        all_data['比亚迪_风险指标'] = byd_data['stock_data']

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
    print("\n波动性与风险维度说明：")
    print("- 历史波动率：价格波动的历史统计（年化）")
    print("- ATR：平均真实波幅，考虑跳空的波动指标")
    print("- 最大回撤：从峰值到谷值的最大跌幅")
    print("- Beta：相对市场的系统性风险")
    print("- Sharpe比率：风险调整后的收益")
    print("- VaR：风险价值，潜在最大损失")
    print("- 下行波动率：只考虑负收益的波动")


if __name__ == "__main__":
    main()
