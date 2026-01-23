"""
维度6：基本面维度数据获取测试（味道感知）

包含维度：
- 财务指标：收入、利润、毛利率、净利率
- 估值指标：PE、PB、PS、EV/EBITDA
- 成长指标：EPS增长率、收入增长率
- 偿债能力：资产负债率、流动比率
- 分红与股息：股息率、派息比率

Usage:
    python environment/test_dimension_6_fundamentals.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import sys
import os


def get_financial_statements(symbol: str = "600519") -> Optional[dict]:
    """
    获取财务报表数据

    Args:
        symbol: 股票代码

    Returns:
        包含资产负债表、利润表、现金流量表的字典
    """
    try:
        import akshare as ak

        print(f"正在获取股票 {symbol} 的财务报表...")

        financial_data = {}

        # 1. 资产负债表
        try:
            # 方法1：使用 stock_balance_sheet_by_report_em
            em_symbol = f"sz{symbol}" if symbol.startswith("0") or symbol.startswith("3") else f"sh{symbol}"
            balance_sheet = ak.stock_balance_sheet_by_report_em(symbol=em_symbol)
            if balance_sheet is not None and not balance_sheet.empty:
                financial_data['资产负债表'] = balance_sheet
                print("  ✓ 资产负债表获取成功（方法1：资产负债表报表）")
            else:
                raise Exception("返回数据为空")
        except Exception as e:
            print(f"  ⚠ 方法1失败: {e}")
            try:
                # 方法2：使用 stock_zh_a_balance_sheet
                balance_sheet = ak.stock_zh_a_balance_sheet(symbol=symbol)
                if balance_sheet is not None and not balance_sheet.empty:
                    financial_data['资产负债表'] = balance_sheet
                    print("  ✓ 资产负债表获取成功（方法2：A股市值资产负债表）")
                else:
                    raise Exception("返回数据为空")
            except Exception as e2:
                print(f"  ⚠ 方法2失败: {e2}")
                try:
                    # 方法3：使用 stock_balance_sheet_by_yearly_report_em
                    em_symbol = f"sz{symbol}" if symbol.startswith("0") or symbol.startswith("3") else f"sh{symbol}"
                    balance_sheet = ak.stock_balance_sheet_by_yearly_report_em(symbol=em_symbol)
                    if balance_sheet is not None and not balance_sheet.empty:
                        financial_data['资产负债表'] = balance_sheet
                        print("  ✓ 资产负债表获取成功（方法3：年度资产负债表）")
                    else:
                        raise Exception("返回数据为空")
                except Exception as e3:
                    print(f"  ✗ 资产负债表获取失败: {e3}")

        # 2. 利润表
        em_symbol = f"sz{symbol}" if symbol.startswith("0") or symbol.startswith("3") else f"sh{symbol}"
        try:
            # 方法1：使用 stock_profit_sheet_by_report_em (与资产负债表和现金流量表函数命名一致)
            income_statement = ak.stock_profit_sheet_by_report_em(symbol=em_symbol)
            if income_statement is not None and not income_statement.empty:
                financial_data['利润表'] = income_statement
                print("  ✓ 利润表获取成功（方法1：利润表报表）")
            else:
                raise Exception("返回数据为空")
        except Exception as e:
            print(f"  ⚠ 方法1失败: {e}")
            try:
                # 方法2：使用 stock_profit_em
                income_statement = ak.stock_profit_em(symbol=em_symbol)
                if income_statement is not None and not income_statement.empty:
                    financial_data['利润表'] = income_statement
                    print("  ✓ 利润表获取成功（方法2：利润表数据）")
                else:
                    raise Exception("返回数据为空")
            except Exception as e2:
                print(f"  ⚠ 方法2失败: {e2}")
                try:
                    # 方法3：使用不带前缀的股票代码
                    income_statement = ak.stock_profit_sheet_by_report_em(symbol=symbol)
                    if income_statement is not None and not income_statement.empty:
                        financial_data['利润表'] = income_statement
                        print("  ✓ 利润表获取成功（方法3：不带前缀代码）")
                    else:
                        raise Exception("返回数据为空")
                except Exception as e3:
                    # 利润表获取失败，显示警告
                    print(f"  ⚠ 利润表获取失败，已跳过: {e3}")

        # 3. 现金流量表
        try:
            # 方法1：使用 stock_cash_flow_sheet_by_report_em
            em_symbol = f"sz{symbol}" if symbol.startswith("0") or symbol.startswith("3") else f"sh{symbol}"
            cash_flow = ak.stock_cash_flow_sheet_by_report_em(symbol=em_symbol)
            if cash_flow is not None and not cash_flow.empty:
                financial_data['现金流量表'] = cash_flow
                print("  ✓ 现金流量表获取成功（方法1：现金流量表报表）")
            else:
                raise Exception("返回数据为空")
        except Exception as e:
            print(f"  ⚠ 方法1失败: {e}")
            try:
                # 方法2：使用 stock_zh_a_cash_flow_sheet
                cash_flow = ak.stock_zh_a_cash_flow_sheet(symbol=symbol)
                if cash_flow is not None and not cash_flow.empty:
                    financial_data['现金流量表'] = cash_flow
                    print("  ✓ 现金流量表获取成功（方法2：A股市值现金流量表）")
                else:
                    raise Exception("返回数据为空")
            except Exception as e2:
                print(f"  ⚠ 方法2失败: {e2}")
                try:
                    # 方法3：使用 stock_cash_flow_sheet_by_yearly_report_em
                    em_symbol = f"sz{symbol}" if symbol.startswith("0") or symbol.startswith("3") else f"sh{symbol}"
                    cash_flow = ak.stock_cash_flow_sheet_by_yearly_report_em(symbol=em_symbol)
                    if cash_flow is not None and not cash_flow.empty:
                        financial_data['现金流量表'] = cash_flow
                        print("  ✓ 现金流量表获取成功（方法3：年度现金流量表）")
                    else:
                        raise Exception("返回数据为空")
                except Exception as e3:
                    print(f"  ✗ 现金流量表获取失败: {e3}")

        return financial_data if financial_data else None

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取财务报表失败：{e}")
        return None


def get_financial_indicators(symbol: str = "600519") -> Optional[pd.DataFrame]:
    """
    获取主要财务指标

    Args:
        symbol: 股票代码

    Returns:
        财务指标DataFrame
    """
    try:
        import akshare as ak

        print(f"正在获取股票 {symbol} 的主要财务指标...")

        # 方法1：使用 stock_financial_analysis_indicator
        try:
            df = ak.stock_financial_analysis_indicator(symbol=symbol)
            if df is not None and not df.empty:
                return df
        except Exception as e1:
            pass

        # 方法2：使用 stock_a_indicator_lg
        try:
            df = ak.stock_a_indicator_lg(symbol=symbol)
            if df is not None and not df.empty:
                return df
        except Exception as e2:
            pass

        # 方法3：使用 stock_financial_abstract
        try:
            em_symbol = f"sz{symbol}" if symbol.startswith("0") or symbol.startswith("3") else f"sh{symbol}"
            df = ak.stock_financial_abstract(symbol=em_symbol)
            if df is not None and not df.empty:
                return df
        except Exception as e3:
            pass

        return None

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取财务指标失败：{e}")
        return None


def get_valuation_metrics(symbol: str = "600519") -> Optional[pd.DataFrame]:
    """
    获取估值指标

    Args:
        symbol: 股票代码

    Returns:
        估值指标DataFrame
    """
    try:
        import akshare as ak

        print(f"正在获取股票 {symbol} 的估值指标...")

        # 获取个股信息（包含PE、PB等）
        df = ak.stock_individual_info_em(symbol=symbol)

        return df

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取估值指标失败：{e}")
        return None


def get_profit_forecast(symbol: str = "600519") -> Optional[pd.DataFrame]:
    """
    获取盈利预测数据

    Args:
        symbol: 股票代码

    Returns:
        盈利预测DataFrame
    """
    try:
        import akshare as ak

        print(f"正在获取股票 {symbol} 的盈利预测...")

        # 获取业绩预告
        df = ak.stock_yjyg_em(date=datetime.now().strftime("%Y-%m-%d"))

        # 筛选指定股票
        if df is not None and not df.empty:
            df_filtered = df[df['股票代码'] == symbol]
            return df_filtered if not df_filtered.empty else df

        return df

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取盈利预测失败：{e}")
        return None


def get_dividend_data(symbol: str = "600519") -> Optional[pd.DataFrame]:
    """
    获取分红送股数据

    Args:
        symbol: 股票代码

    Returns:
        分红数据DataFrame
    """
    try:
        import akshare as ak

        print(f"正在获取股票 {symbol} 的分红送股数据...")

        # 方法1：使用 stock_dividend_cninfo
        try:
            df = ak.stock_dividend_cninfo(symbol=symbol)
            if df is not None and not df.empty:
                print("  ✓ 分红数据获取成功（方法1：巨潮资讯分红数据）")
                return df
        except Exception as e1:
            print(f"  ⚠ 方法1失败: {e1}")

        # 方法2：使用 stock_fhps_em（不带参数，获取全部数据后筛选）
        try:
            df = ak.stock_fhps_em()
            if df is not None and not df.empty:
                # 筛选指定股票
                if '代码' in df.columns:
                    df_filtered = df[df['代码'] == symbol]
                elif '股票代码' in df.columns:
                    df_filtered = df[df['股票代码'] == symbol]
                else:
                    df_filtered = df
                if not df_filtered.empty:
                    print("  ✓ 分红数据获取成功（方法2：全部分红数据筛选）")
                    return df_filtered
        except Exception as e2:
            print(f"  ⚠ 方法2失败: {e2}")

        # 方法3：使用 stock_history_dividend
        try:
            df = ak.stock_history_dividend()
            if df is not None and not df.empty:
                if '代码' in df.columns:
                    df_filtered = df[df['代码'] == symbol]
                elif '股票代码' in df.columns:
                    df_filtered = df[df['股票代码'] == symbol]
                else:
                    df_filtered = df
                if not df_filtered.empty:
                    print("  ✓ 分红数据获取成功（方法3：历史分红数据）")
                    return df_filtered
        except Exception as e3:
            print(f"  ⚠ 方法3失败: {e3}")

        print("  ✗ 分红数据获取失败")
        return None

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取分红数据失败：{e}")
        return None


def get_roe_roa_data(symbol: str = "600519") -> Optional[pd.DataFrame]:
    """
    获取ROE、ROA等盈利能力指标

    Args:
        symbol: 股票代码

    Returns:
        盈利能力指标DataFrame
    """
    try:
        import akshare as ak

        print(f"正在获取股票 {symbol} 的ROE/ROA数据...")

        # 方法1：使用 stock_a_indicator_lg（乐股指标）
        try:
            df = ak.stock_a_indicator_lg(symbol=symbol)
            if df is not None and not df.empty:
                print("  ✓ ROE/ROA数据获取成功（方法1：乐股指标）")
                return df
        except Exception as e1:
            print(f"  ⚠ 方法1失败: {e1}")

        # 方法2：使用 stock_financial_abstract_em（财务摘要）
        try:
            em_symbol = f"sz{symbol}" if symbol.startswith("0") or symbol.startswith("3") else f"sh{symbol}"
            df = ak.stock_financial_abstract_em(symbol=em_symbol)
            if df is not None and not df.empty:
                print("  ✓ ROE/ROA数据获取成功（方法2：财务摘要）")
                return df
        except Exception as e2:
            print(f"  ⚠ 方法2失败: {e2}")

        # 方法3：使用个股信息获取基本估值指标
        try:
            df = ak.stock_individual_info_em(symbol=symbol)
            if df is not None and not df.empty:
                print("  ✓ ROE/ROA数据获取成功（方法3：个股信息）")
                return df
        except Exception as e3:
            print(f"  ⚠ 方法3失败: {e3}")

        print("  ✗ ROE/ROA数据获取失败")
        return None

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取ROE/ROA数据失败：{e}")
        return None


def get_debt_metrics(symbol: str = "600519") -> Optional[pd.DataFrame]:
    """
    获取偿债能力指标

    Args:
        symbol: 股票代码

    Returns:
        偿债能力指标DataFrame
    """
    try:
        import akshare as ak

        print(f"正在获取股票 {symbol} 的偿债能力指标...")

        # 方法1：使用 stock_a_indicator_lg（乐股指标）
        try:
            df = ak.stock_a_indicator_lg(symbol=symbol)
            if df is not None and not df.empty:
                print("  ✓ 偿债能力指标获取成功（方法1：乐股指标）")
                return df
        except Exception as e1:
            print(f"  ⚠ 方法1失败: {e1}")

        # 方法2：使用 stock_financial_abstract_em（财务摘要）
        try:
            em_symbol = f"sz{symbol}" if symbol.startswith("0") or symbol.startswith("3") else f"sh{symbol}"
            df = ak.stock_financial_abstract_em(symbol=em_symbol)
            if df is not None and not df.empty:
                print("  ✓ 偿债能力指标获取成功（方法2：财务摘要）")
                return df
        except Exception as e2:
            print(f"  ⚠ 方法2失败: {e2}")

        # 方法3：使用个股信息
        try:
            df = ak.stock_individual_info_em(symbol=symbol)
            if df is not None and not df.empty:
                print("  ✓ 偿债能力指标获取成功（方法3：个股信息）")
                return df
        except Exception as e3:
            print(f"  ⚠ 方法3失败: {e3}")

        print("  ✗ 偿债能力指标获取失败")
        return None

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取偿债能力指标失败：{e}")
        return None


def get_growth_metrics(symbol: str = "600519") -> Optional[pd.DataFrame]:
    """
    获取成长性指标

    Args:
        symbol: 股票代码

    Returns:
        成长性指标DataFrame
    """
    try:
        import akshare as ak

        print(f"正在获取股票 {symbol} 的成长性指标...")

        # 方法1：使用 stock_a_indicator_lg（乐股指标）
        try:
            df = ak.stock_a_indicator_lg(symbol=symbol)
            if df is not None and not df.empty:
                print("  ✓ 成长性指标获取成功（方法1：乐股指标）")
                return df
        except Exception as e1:
            print(f"  ⚠ 方法1失败: {e1}")

        # 方法2：使用 stock_financial_abstract_em（财务摘要）
        try:
            em_symbol = f"sz{symbol}" if symbol.startswith("0") or symbol.startswith("3") else f"sh{symbol}"
            df = ak.stock_financial_abstract_em(symbol=em_symbol)
            if df is not None and not df.empty:
                print("  ✓ 成长性指标获取成功（方法2：财务摘要）")
                return df
        except Exception as e2:
            print(f"  ⚠ 方法2失败: {e2}")

        # 方法3：使用个股信息
        try:
            df = ak.stock_individual_info_em(symbol=symbol)
            if df is not None and not df.empty:
                print("  ✓ 成长性指标获取成功（方法3：个股信息）")
                return df
        except Exception as e3:
            print(f"  ⚠ 方法3失败: {e3}")

        print("  ✗ 成长性指标获取失败")
        return None

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取成长性指标失败：{e}")
        return None


def get_operating_metrics(symbol: str = "600519") -> Optional[pd.DataFrame]:
    """
    获取运营能力指标

    Args:
        symbol: 股票代码

    Returns:
        运营能力指标DataFrame
    """
    try:
        import akshare as ak

        print(f"正在获取股票 {symbol} 的运营能力指标...")

        # 方法1：使用 stock_a_indicator_lg（乐股指标）
        try:
            df = ak.stock_a_indicator_lg(symbol=symbol)
            if df is not None and not df.empty:
                print("  ✓ 运营能力指标获取成功（方法1：乐股指标）")
                return df
        except Exception as e1:
            print(f"  ⚠ 方法1失败: {e1}")

        # 方法2：使用 stock_financial_abstract_em（财务摘要）
        try:
            em_symbol = f"sz{symbol}" if symbol.startswith("0") or symbol.startswith("3") else f"sh{symbol}"
            df = ak.stock_financial_abstract_em(symbol=em_symbol)
            if df is not None and not df.empty:
                print("  ✓ 运营能力指标获取成功（方法2：财务摘要）")
                return df
        except Exception as e2:
            print(f"  ⚠ 方法2失败: {e2}")

        # 方法3：使用个股信息
        try:
            df = ak.stock_individual_info_em(symbol=symbol)
            if df is not None and not df.empty:
                print("  ✓ 运营能力指标获取成功（方法3：个股信息）")
                return df
        except Exception as e3:
            print(f"  ⚠ 方法3失败: {e3}")

        print("  ✗ 运营能力指标获取失败")
        return None

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取运营能力指标失败：{e}")
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

    print(f"\n列名: {list(df.columns)}")
    print(f"\n数据预览:")
    print(df.head())

    print(f"{'='*80}\n")


def main():
    """主测试函数"""
    print("="*80)
    print("维度6：基本面维度数据测试（味道感知）")
    print("="*80)

    # 输出文件名
    excel_file = "dimension_6_fundamentals_data.xlsx"

    # 存储所有数据
    all_data = {}

    # 测试股票列表
    test_stocks = [
        ("600519", "贵州茅台"),
        ("000001", "平安银行"),
        ("600036", "招商银行")
    ]

    for symbol, name in test_stocks:
        print(f"\n{'='*80}")
        print(f"正在获取 {name}({symbol}) 的基本面数据")
        print(f"{'='*80}")

        # 1. 财务报表
        print(f"\n[1] 获取{name}财务报表")
        financial_statements = get_financial_statements(symbol)
        if financial_statements:
            for statement_name, df in financial_statements.items():
                if df is not None and not df.empty:
                    sheet_name = f"{name}_{statement_name}"
                    all_data[sheet_name] = df
                    print(f"  ✓ {statement_name}: {len(df)}行")

        # 2. 主要财务指标
        print(f"\n[2] 获取{name}主要财务指标")
        financial_indicators = get_financial_indicators(symbol)
        if financial_indicators is not None and not financial_indicators.empty:
            all_data[f"{name}_财务指标"] = financial_indicators
            print(f"  ✓ 财务指标: {len(financial_indicators)}行")

        # 3. 估值指标
        print(f"\n[3] 获取{name}估值指标")
        valuation = get_valuation_metrics(symbol)
        if valuation is not None and not valuation.empty:
            all_data[f"{name}_估值"] = valuation
            print(f"  ✓ 估值指标获取成功")

        # 4. 分红数据
        print(f"\n[4] 获取{name}分红数据")
        dividend = get_dividend_data(symbol)
        if dividend is not None and not dividend.empty:
            all_data[f"{name}_分红"] = dividend
            print(f"  ✓ 分红数据: {len(dividend)}行")

        # 5. ROE/ROA
        print(f"\n[5] 获取{name}盈利能力指标")
        roe_roa = get_roe_roa_data(symbol)
        if roe_roa is not None and not roe_roa.empty:
            all_data[f"{name}_盈利能力"] = roe_roa
            print(f"  ✓ 盈利能力: {len(roe_roa)}行")

        # 6. 偿债能力
        print(f"\n[6] 获取{name}偿债能力指标")
        debt = get_debt_metrics(symbol)
        if debt is not None and not debt.empty:
            all_data[f"{name}_偿债能力"] = debt
            print(f"  ✓ 偿债能力: {len(debt)}行")

        # 7. 成长性
        print(f"\n[7] 获取{name}成长性指标")
        growth = get_growth_metrics(symbol)
        if growth is not None and not growth.empty:
            all_data[f"{name}_成长性"] = growth
            print(f"  ✓ 成长性: {len(growth)}行")

        # 8. 运营能力
        print(f"\n[8] 获取{name}运营能力指标")
        operating = get_operating_metrics(symbol)
        if operating is not None and not operating.empty:
            all_data[f"{name}_运营能力"] = operating
            print(f"  ✓ 运营能力: {len(operating)}行")

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
    print("\n基本面维度说明：")
    print("- 财务报表：资产负债表、利润表、现金流量表")
    print("- 财务指标：营收、利润、毛利率、净利率等")
    print("- 估值指标：PE、PB、PS等相对估值")
    print("- 盈利能力：ROE、ROA等")
    print("- 偿债能力：资产负债率、流动比率等")
    print("- 成长性：收入增长率、EPS增长率等")
    print("- 运营能力：周转率等")
    print("- 分红：股息率、派息比率等")


if __name__ == "__main__":
    main()
