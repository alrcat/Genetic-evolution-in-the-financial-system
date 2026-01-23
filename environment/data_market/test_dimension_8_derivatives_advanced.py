"""
维度8：衍生品与高级维度数据获取测试（非线性感知）

包含维度：
- 期权 Greeks：Delta, Gamma, Vega
- 波动率曲面：各行权价与期限
- 隐含相关性：多资产相关
- 系统性风险

Usage:
    python environment/test_dimension_8_derivatives_advanced.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import sys
import os


def get_option_quotes() -> Optional[pd.DataFrame]:
    """
    获取期权行情数据

    Returns:
        期权行情DataFrame
    """
    try:
        import akshare as ak

        print("正在获取期权行情数据...")

        # 获取50ETF期权行情
        df = ak.option_finance_board()

        return df

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取期权行情失败：{e}")
        return None


def calculate_option_greeks(
    spot_price: float,
    strike_price: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call"
) -> Dict[str, float]:
    """
    计算期权Greeks（简化版Black-Scholes模型）

    Args:
        spot_price: 标的价格
        strike_price: 行权价
        time_to_maturity: 到期时间（年）
        risk_free_rate: 无风险利率
        volatility: 波动率
        option_type: 期权类型 ("call" or "put")

    Returns:
        Greeks字典
    """
    from scipy.stats import norm
    import math

    # d1和d2
    d1 = (np.log(spot_price / strike_price) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_maturity) / \
         (volatility * np.sqrt(time_to_maturity))
    d2 = d1 - volatility * np.sqrt(time_to_maturity)

    # Delta
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = -norm.cdf(-d1)

    # Gamma
    gamma = norm.pdf(d1) / (spot_price * volatility * np.sqrt(time_to_maturity))

    # Vega
    vega = spot_price * norm.pdf(d1) * np.sqrt(time_to_maturity) / 100

    # Theta
    if option_type == "call":
        theta = (-spot_price * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_maturity)) -
                risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2)) / 365
    else:
        theta = (-spot_price * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_maturity)) +
                risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2)) / 365

    # Rho
    if option_type == "call":
        rho = strike_price * time_to_maturity * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2) / 100
    else:
        rho = -strike_price * time_to_maturity * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2) / 100

    return {
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega,
        'Theta': theta,
        'Rho': rho
    }


def generate_volatility_surface() -> pd.DataFrame:
    """
    生成波动率曲面示例数据

    Returns:
        波动率曲面DataFrame
    """
    print("正在生成波动率曲面示例...")

    # 行权价范围（相对于当前价格的百分比）
    strikes = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]

    # 到期时间（月）
    maturities = [1, 2, 3, 6, 9, 12]

    # 生成波动率数据（示例：微笑曲线）
    data = []
    for maturity in maturities:
        for strike in strikes:
            # 简化的波动率微笑模型
            moneyness = strike
            base_vol = 0.20
            smile_effect = 0.05 * (moneyness - 1.0) ** 2
            term_structure = 0.02 * np.sqrt(maturity / 12)
            implied_vol = base_vol + smile_effect + term_structure

            data.append({
                '到期月数': maturity,
                '行权价比例': strike,
                '隐含波动率': implied_vol,
                '波动率类型': '微笑曲线'
            })

    return pd.DataFrame(data)


def calculate_correlation_matrix(symbols: List[str],
                                 start_date: str = "20240101",
                                 end_date: str = "20241231") -> Optional[pd.DataFrame]:
    """
    计算股票相关性矩阵

    Args:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        相关性矩阵DataFrame
    """
    try:
        import akshare as ak

        print(f"正在计算{len(symbols)}只股票的相关性矩阵...")

        # 获取各股票价格数据
        price_data = {}
        for symbol in symbols:
            try:
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )
                if df is not None and not df.empty:
                    price_data[symbol] = df.set_index('日期')['收盘']
                    print(f"  ✓ {symbol}: {len(df)}天")
            except Exception as e:
                print(f"  ✗ {symbol}: {e}")

        if not price_data:
            return None

        # 合并数据
        prices_df = pd.DataFrame(price_data)

        # 计算收益率
        returns = prices_df.pct_change().dropna()

        # 计算相关性矩阵
        corr_matrix = returns.corr()

        return corr_matrix

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"计算相关性矩阵失败：{e}")
        return None


def calculate_rolling_correlation(symbol1: str, symbol2: str,
                                  window: int = 60,
                                  start_date: str = "20240101",
                                  end_date: str = "20241231") -> Optional[pd.DataFrame]:
    """
    计算滚动相关性

    Args:
        symbol1: 股票1代码
        symbol2: 股票2代码
        window: 滚动窗口
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        滚动相关性DataFrame
    """
    try:
        import akshare as ak

        print(f"正在计算{symbol1}和{symbol2}的滚动相关性...")

        # 获取两只股票数据
        df1 = ak.stock_zh_a_hist(symbol=symbol1, period="daily",
                                 start_date=start_date, end_date=end_date, adjust="qfq")
        df2 = ak.stock_zh_a_hist(symbol=symbol2, period="daily",
                                 start_date=start_date, end_date=end_date, adjust="qfq")

        if df1 is None or df2 is None or df1.empty or df2.empty:
            return None

        # 合并数据
        df1 = df1.set_index('日期')[['收盘']].rename(columns={'收盘': symbol1})
        df2 = df2.set_index('日期')[['收盘']].rename(columns={'收盘': symbol2})

        merged = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')

        # 计算收益率
        returns = merged.pct_change().dropna()

        # 计算滚动相关性
        rolling_corr = returns[symbol1].rolling(window).corr(returns[symbol2])

        result = pd.DataFrame({
            '日期': rolling_corr.index,
            f'{symbol1}_{symbol2}_相关性': rolling_corr.values,
            '窗口期': window
        })

        return result

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"计算滚动相关性失败：{e}")
        return None


def calculate_beta_matrix(symbols: List[str],
                         market_symbol: str = "000001",
                         start_date: str = "20240101",
                         end_date: str = "20241231") -> Optional[pd.DataFrame]:
    """
    计算多只股票相对于市场的Beta值

    Args:
        symbols: 股票代码列表
        market_symbol: 市场指数代码
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        Beta矩阵DataFrame
    """
    try:
        import akshare as ak

        print(f"正在计算{len(symbols)}只股票相对于市场的Beta...")

        # 获取市场指数数据
        em_symbol = f"sh{market_symbol}" if market_symbol.startswith('000') else f"sz{market_symbol}"
        market_df = ak.stock_zh_index_daily_em(symbol=em_symbol,
                                               start_date=start_date,
                                               end_date=end_date)

        if market_df is None or market_df.empty:
            return None

        market_df = market_df.rename(columns={'date': '日期', 'close': '收盘'})
        market_returns = market_df.set_index('日期')['收盘'].pct_change().dropna()

        # 计算各股票Beta
        beta_data = []
        for symbol in symbols:
            try:
                df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                       start_date=start_date, end_date=end_date, adjust="qfq")

                if df is not None and not df.empty:
                    stock_returns = df.set_index('日期')['收盘'].pct_change().dropna()

                    # 对齐日期
                    aligned = pd.DataFrame({
                        'stock': stock_returns,
                        'market': market_returns
                    }).dropna()

                    # 计算Beta
                    covariance = aligned['stock'].cov(aligned['market'])
                    market_variance = aligned['market'].var()
                    beta = covariance / market_variance

                    beta_data.append({
                        '股票代码': symbol,
                        'Beta': beta,
                        '相关系数': aligned['stock'].corr(aligned['market']),
                        '股票波动率': aligned['stock'].std() * np.sqrt(252),
                        '市场波动率': aligned['market'].std() * np.sqrt(252)
                    })
                    print(f"  ✓ {symbol}: Beta={beta:.3f}")

            except Exception as e:
                print(f"  ✗ {symbol}: {e}")

        return pd.DataFrame(beta_data) if beta_data else None

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"计算Beta矩阵失败：{e}")
        return None


def calculate_var_covar_matrix(symbols: List[str],
                               start_date: str = "20240101",
                               end_date: str = "20241231") -> Optional[pd.DataFrame]:
    """
    计算方差-协方差矩阵（用于组合风险计算）

    Args:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        方差-协方差矩阵DataFrame
    """
    try:
        import akshare as ak

        print(f"正在计算{len(symbols)}只股票的方差-协方差矩阵...")

        # 获取各股票价格数据
        price_data = {}
        for symbol in symbols:
            try:
                df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                       start_date=start_date, end_date=end_date, adjust="qfq")
                if df is not None and not df.empty:
                    price_data[symbol] = df.set_index('日期')['收盘']
            except Exception as e:
                print(f"  ✗ {symbol}: {e}")

        if not price_data:
            return None

        # 合并数据
        prices_df = pd.DataFrame(price_data)

        # 计算收益率
        returns = prices_df.pct_change().dropna()

        # 计算协方差矩阵（年化）
        cov_matrix = returns.cov() * 252

        return cov_matrix

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"计算方差-协方差矩阵失败：{e}")
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
    print(f"列数: {len(df.columns)}")

    print(f"\n数据预览:")
    print(df.head(15))

    print(f"{'='*80}\n")


def main():
    """主测试函数"""
    print("="*80)
    print("维度8：衍生品与高级维度数据测试（非线性感知）")
    print("="*80)

    # 输出文件名
    excel_file = "dimension_8_derivatives_advanced_data.xlsx"

    # 存储所有数据
    all_data = {}

    # 测试1: 期权Greeks计算示例
    print("\n[测试1] 期权Greeks计算示例")
    greeks_data = []
    for strike in [90, 95, 100, 105, 110]:
        greeks = calculate_option_greeks(
            spot_price=100,
            strike_price=strike,
            time_to_maturity=0.25,  # 3个月
            risk_free_rate=0.03,
            volatility=0.25,
            option_type="call"
        )
        greeks['行权价'] = strike
        greeks_data.append(greeks)

    greeks_df = pd.DataFrame(greeks_data)
    all_data['期权Greeks'] = greeks_df
    print_data_summary(greeks_df, "期权Greeks示例")

    # 测试2: 波动率曲面
    print("\n[测试2] 生成波动率曲面")
    vol_surface = generate_volatility_surface()
    all_data['波动率曲面'] = vol_surface
    print_data_summary(vol_surface, "波动率曲面")

    # 测试3: 股票相关性矩阵
    print("\n[测试3] 计算股票相关性矩阵")
    test_symbols = ["600519", "000858", "000001", "600036", "601318"]
    corr_matrix = calculate_correlation_matrix(test_symbols,
                                               start_date="20240101",
                                               end_date="20241231")
    if corr_matrix is not None and not corr_matrix.empty:
        all_data['相关性矩阵'] = corr_matrix
        print_data_summary(corr_matrix, "股票相关性矩阵")

    # 测试4: 滚动相关性
    print("\n[测试4] 计算滚动相关性（茅台 vs 五粮液）")
    rolling_corr = calculate_rolling_correlation("600519", "000858",
                                                 window=60,
                                                 start_date="20240101",
                                                 end_date="20241231")
    if rolling_corr is not None and not rolling_corr.empty:
        all_data['滚动相关性_茅台五粮液'] = rolling_corr
        print_data_summary(rolling_corr, "滚动相关性")

    # 测试5: Beta矩阵
    print("\n[测试5] 计算Beta矩阵（相对于上证指数）")
    beta_matrix = calculate_beta_matrix(test_symbols,
                                       market_symbol="000001",
                                       start_date="20240101",
                                       end_date="20241231")
    if beta_matrix is not None and not beta_matrix.empty:
        all_data['Beta矩阵'] = beta_matrix
        print_data_summary(beta_matrix, "Beta矩阵")

    # 测试6: 方差-协方差矩阵
    print("\n[测试6] 计算方差-协方差矩阵")
    cov_matrix = calculate_var_covar_matrix(test_symbols,
                                            start_date="20240101",
                                            end_date="20241231")
    if cov_matrix is not None and not cov_matrix.empty:
        all_data['协方差矩阵'] = cov_matrix
        print_data_summary(cov_matrix, "方差-协方差矩阵")

    # 测试7: 期权行情（如果可用）
    print("\n[测试7] 获取期权行情数据")
    option_quotes = get_option_quotes()
    if option_quotes is not None and not option_quotes.empty:
        all_data['期权行情'] = option_quotes
        print_data_summary(option_quotes, "期权行情")

    # 保存所有数据到Excel
    if all_data:
        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                for sheet_name, df in all_data.items():
                    safe_name = sheet_name[:31]
                    df.to_excel(writer, sheet_name=safe_name, index=True)
                    print(f"已保存工作表: {safe_name}")

            print(f"\n所有数据已保存至: {excel_file}")
            print(f"文件位置: {os.path.abspath(excel_file)}")

        except ImportError:
            print("警告：需要安装 openpyxl 和 scipy 库来保存Excel文件")
            print("请运行：pip install openpyxl scipy")
        except Exception as e:
            print(f"保存Excel文件失败：{e}")
            import traceback
            traceback.print_exc()
    else:
        print("没有数据可保存")

    print("\n测试完成！")
    print("\n衍生品与高级维度说明：")
    print("- 期权Greeks：Delta(价格敏感度)、Gamma(Delta变化率)、Vega(波动率敏感度)")
    print("- 波动率曲面：不同行权价和期限的隐含波动率")
    print("- 相关性矩阵：多资产间的相关关系")
    print("- 滚动相关性：时变的相关性结构")
    print("- Beta矩阵：系统性风险暴露")
    print("- 协方差矩阵：组合风险计算基础")
    print("- 这些高级指标用于：期权定价、风险管理、组合优化")


if __name__ == "__main__":
    main()
