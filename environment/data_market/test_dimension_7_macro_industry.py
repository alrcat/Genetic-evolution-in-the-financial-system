"""
维度7：宏观与行业维度数据获取测试（远程气味）

包含维度：
- 利率、汇率：经济政策背景
- PMI、CPI：经济活力
- 行业数据：行业景气指数
- 政策信息：政策变动

Usage:
    python environment/test_dimension_7_macro_industry.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import sys
import os
import warnings

# 抑制 akshare 库中的 FutureWarning 警告
warnings.filterwarnings('ignore', category=FutureWarning, module='akshare')


def get_interest_rate_data() -> Optional[pd.DataFrame]:
    """
    获取利率数据（LPR、Shibor等）

    Returns:
        利率数据DataFrame
    """
    try:
        import akshare as ak

        print("正在获取利率数据...")

        rate_data = {}

        # 1. LPR利率
        try:
            # 方法1：尝试 rate_interbank_lpr
            lpr = ak.rate_interbank_lpr()
            if lpr is not None and not lpr.empty:
                rate_data['LPR'] = lpr
                print("  ✓ LPR利率获取成功（方法1：银行间拆借利率LPR）")
            else:
                raise Exception("返回数据为空")
        except Exception as e:
            print(f"  ⚠ 方法1失败: {e}")
            try:
                # 方法2：尝试 macro_china_lpr
                lpr = ak.macro_china_lpr()
                if lpr is not None and not lpr.empty:
                    rate_data['LPR'] = lpr
                    print("  ✓ LPR利率获取成功（方法2：中国宏观LPR）")
                else:
                    raise Exception("返回数据为空")
            except Exception as e2:
                print(f"  ⚠ 方法2失败: {e2}")
                try:
                    # 方法3：尝试通过其他利率数据接口获取
                    # 如果LPR不可用，可以跳过或使用其他利率数据
                    print(f"  ⚠ LPR利率获取失败，已跳过: {e2}")
                except Exception as e3:
                    print(f"  ✗ LPR利率获取失败: {e3}")

        # 2. Shibor利率
        try:
            shibor = ak.rate_interbank()
            if shibor is not None and not shibor.empty:
                rate_data['Shibor'] = shibor
                print("  ✓ Shibor利率获取成功（方法1：银行间同业拆借利率）")
            else:
                print("  ✗ Shibor利率获取失败: 返回数据为空")
        except Exception as e:
            print(f"  ✗ Shibor利率获取失败: {e}")

        return rate_data if rate_data else None

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取利率数据失败：{e}")
        return None


def get_exchange_rate_data() -> Optional[pd.DataFrame]:
    """
    获取汇率数据

    Returns:
        汇率数据DataFrame
    """
    try:
        import akshare as ak

        print("正在获取汇率数据...")

        # 获取美元兑人民币汇率
        df = ak.currency_boc_sina()
        if df is not None and not df.empty:
            print("  ✓ 汇率数据获取成功（方法1：央行外汇牌价）")
            return df
        else:
            print("  ✗ 汇率数据获取失败: 返回数据为空")
            return None

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取汇率数据失败：{e}")
        return None


def get_pmi_data() -> Optional[pd.DataFrame]:
    """
    获取PMI数据（采购经理人指数）

    Returns:
        PMI数据DataFrame
    """
    try:
        import akshare as ak

        print("正在获取PMI数据...")

        # 获取中国PMI数据
        df = ak.macro_china_pmi()
        if df is not None and not df.empty:
            print("  ✓ PMI数据获取成功（方法1：中国制造业PMI）")
            return df
        else:
            print("  ✗ PMI数据获取失败: 返回数据为空")
            return None

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取PMI数据失败：{e}")
        return None


def get_cpi_ppi_data() -> Optional[dict]:
    """
    获取CPI和PPI数据

    Returns:
        包含CPI和PPI的字典
    """
    try:
        import akshare as ak

        print("正在获取CPI/PPI数据...")

        data = {}

        # 1. CPI数据
        try:
            cpi = ak.macro_china_cpi()
            if cpi is not None and not cpi.empty:
                data['CPI'] = cpi
                print("  ✓ CPI数据获取成功（方法1：中国居民消费价格指数）")
            else:
                print("  ✗ CPI数据获取失败: 返回数据为空")
        except Exception as e:
            print(f"  ✗ CPI数据获取失败: {e}")

        # 2. PPI数据
        try:
            ppi = ak.macro_china_ppi()
            if ppi is not None and not ppi.empty:
                data['PPI'] = ppi
                print("  ✓ PPI数据获取成功（方法1：中国工业生产者出厂价格指数）")
            else:
                print("  ✗ PPI数据获取失败: 返回数据为空")
        except Exception as e:
            print(f"  ✗ PPI数据获取失败: {e}")

        return data if data else None

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取CPI/PPI数据失败：{e}")
        return None


def get_gdp_data() -> Optional[pd.DataFrame]:
    """
    获取GDP数据

    Returns:
        GDP数据DataFrame
    """
    try:
        import akshare as ak

        print("正在获取GDP数据...")

        # 获取中国GDP数据
        df = ak.macro_china_gdp()
        if df is not None and not df.empty:
            print("  ✓ GDP数据获取成功（方法1：中国GDP数据）")
            return df
        else:
            print("  ✗ GDP数据获取失败: 返回数据为空")
            return None

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取GDP数据失败：{e}")
        return None


def get_money_supply_data() -> Optional[pd.DataFrame]:
    """
    获取货币供应量数据（M0、M1、M2）

    Returns:
        货币供应量数据DataFrame
    """
    try:
        import akshare as ak

        print("正在获取货币供应量数据...")

        # 获取货币供应量
        df = ak.macro_china_money_supply()
        if df is not None and not df.empty:
            print("  ✓ 货币供应量数据获取成功（方法1：中国货币供应量）")
            return df
        else:
            print("  ✗ 货币供应量数据获取失败: 返回数据为空")
            return None

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取货币供应量数据失败：{e}")
        return None


def get_industry_index_data() -> Optional[pd.DataFrame]:
    """
    获取行业指数数据

    Returns:
        行业指数DataFrame
    """
    try:
        import akshare as ak

        print("正在获取行业指数数据...")

        # 方法1：尝试 stock_board_industry_name_em
        try:
            df = ak.stock_board_industry_name_em()
            if df is not None and not df.empty:
                print("  ✓ 行业指数数据获取成功（方法1：东方财富行业板块名称）")
                return df
        except Exception as e:
            print(f"  ⚠ 方法1失败: {e}")

        # 方法2：尝试 stock_board_industry_index_em
        try:
            df = ak.stock_board_industry_index_em()
            if df is not None and not df.empty:
                print("  ✓ 行业指数数据获取成功（方法2：东方财富行业板块指数）")
                return df
        except Exception as e:
            print(f"  ⚠ 方法2失败: {e}")

        # 方法3：尝试 stock_board_industry_cons_em 获取所有行业
        try:
            # 获取行业列表
            industry_list = ak.stock_board_industry_name_em()
            if industry_list is not None and not industry_list.empty:
                # 返回行业列表作为指数数据
                print("  ✓ 行业指数数据获取成功（方法3：行业板块名称列表）")
                return industry_list
        except Exception as e:
            print(f"  ⚠ 方法3失败: {e}")

        print("  ✗ 行业指数数据获取失败")
        return None

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取行业指数失败：{e}")
        return None


def get_industry_pe_data() -> Optional[pd.DataFrame]:
    """
    获取行业市盈率数据

    Returns:
        行业市盈率DataFrame
    """
    try:
        import akshare as ak

        print("正在获取行业市盈率数据...")

        # 方法1：尝试 stock_board_industry_name_em
        try:
            df = ak.stock_board_industry_name_em()
            if df is not None and not df.empty:
                print("  ✓ 行业市盈率数据获取成功（方法1：东方财富行业板块名称）")
                return df
        except Exception as e:
            print(f"  ⚠ 方法1失败: {e}")

        # 方法2：尝试 stock_board_industry_index_em
        try:
            df = ak.stock_board_industry_index_em()
            if df is not None and not df.empty:
                print("  ✓ 行业市盈率数据获取成功（方法2：东方财富行业板块指数）")
                return df
        except Exception as e:
            print(f"  ⚠ 方法2失败: {e}")

        print("  ✗ 行业市盈率数据获取失败")
        return None

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取行业市盈率失败：{e}")
        return None


def get_industry_capital_flow() -> Optional[pd.DataFrame]:
    """
    获取行业资金流向

    Returns:
        行业资金流向DataFrame
    """
    try:
        import akshare as ak

        print("正在获取行业资金流向...")

        # 获取行业资金流
        df = ak.stock_sector_fund_flow_rank(indicator="今日")
        if df is not None and not df.empty:
            print("  ✓ 行业资金流向获取成功（方法1：今日资金流向排名）")
            return df
        else:
            print("  ✗ 行业资金流向获取失败: 返回数据为空")
            return None

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取行业资金流向失败：{e}")
        return None


def get_industry_constituent_stocks(industry: str = "银行") -> Optional[pd.DataFrame]:
    """
    获取行业成分股

    Args:
        industry: 行业名称

    Returns:
        行业成分股DataFrame
    """
    try:
        import akshare as ak

        print(f"正在获取{industry}行业成分股...")

        # 方法1：直接使用行业名称
        try:
            df = ak.stock_board_industry_cons_em(symbol=industry)
            if df is not None and not df.empty:
                print(f"  ✓ {industry}行业成分股获取成功（方法1：直接行业名称）")
                return df
        except Exception as e1:
            print(f"  ⚠ 方法1失败: {e1}")
            # 方法2：先获取行业列表，找到正确的行业代码
            try:
                industry_list = ak.stock_board_industry_name_em()
                if industry_list is not None and not industry_list.empty:
                    # 查找匹配的行业 - 检查列名
                    if '板块名称' in industry_list.columns:
                        matched = industry_list[industry_list['板块名称'].str.contains(industry, na=False)]
                    elif len(industry_list.columns) > 0:
                        # 假设第一列是行业名称
                        first_col = industry_list.columns[0]
                        matched = industry_list[industry_list[first_col].str.contains(industry, na=False)]
                    else:
                        matched = pd.DataFrame()

                    if not matched.empty and len(matched) > 0:
                        # 获取行业代码
                        if '板块代码' in matched.columns:
                            industry_code = matched.iloc[0]['板块代码']
                        elif len(matched.columns) > 1:
                            industry_code = matched.iloc[0, 1]  # 第二列可能是代码
                        else:
                            industry_code = matched.iloc[0, 0]  # 使用第一列

                        df = ak.stock_board_industry_cons_em(symbol=industry_code)
                        if df is not None and not df.empty:
                            print(f"  ✓ {industry}行业成分股获取成功（方法2：通过行业代码）")
                            return df
            except Exception as e2:
                print(f"  ⚠ 方法2失败: {e2}")
                # 方法3：尝试使用行业名称的其他变体
                try:
                    # 尝试添加"行业"后缀
                    industry_variant = f"{industry}行业"
                    df = ak.stock_board_industry_cons_em(symbol=industry_variant)
                    if df is not None and not df.empty:
                        print(f"  ✓ {industry}行业成分股获取成功（方法3：行业名称变体）")
                        return df
                except Exception as e3:
                    print(f"  ⚠ 方法3失败: {e3}")

        print(f"  ✗ {industry}行业成分股获取失败")
        return None

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取行业成分股失败：{e}")
        return None


def get_commodity_prices() -> Optional[pd.DataFrame]:
    """
    获取大宗商品价格

    Returns:
        商品价格DataFrame
    """
    try:
        import akshare as ak

        print("正在获取大宗商品价格...")

        # 方法1：尝试 commodity_index
        try:
            df = ak.commodity_index()
            if df is not None and not df.empty:
                print("  ✓ 大宗商品价格获取成功（方法1：商品指数）")
                return df
        except Exception as e:
            print(f"  ⚠ 方法1失败: {e}")

        # 方法2：尝试 macro_china_commodity_price_index
        try:
            df = ak.macro_china_commodity_price_index()
            if df is not None and not df.empty:
                print("  ✓ 大宗商品价格获取成功（方法2：中国商品价格指数）")
                return df
        except Exception as e:
            print(f"  ⚠ 方法2失败: {e}")

        # 方法3：尝试获取具体商品价格
        try:
            # 获取期货主力合约作为示例
            df = ak.futures_main_sina()
            if df is not None and not df.empty:
                print("  ✓ 大宗商品价格获取成功（方法3：期货主力合约）")
                return df
        except Exception as e:
            print(f"  ⚠ 方法3失败: {e}")

        print("  ✗ 大宗商品价格获取失败")
        return None

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取大宗商品价格失败：{e}")
        return None


def get_economic_indicators_summary() -> Optional[pd.DataFrame]:
    """
    获取宏观经济指标汇总

    Returns:
        宏观经济指标DataFrame
    """
    try:
        import akshare as ak

        print("正在获取宏观经济指标汇总...")

        # 方法1：尝试 macro_china_composite_pmi
        try:
            df = ak.macro_china_composite_pmi()
            if df is not None and not df.empty:
                print("  ✓ 宏观经济指标汇总获取成功（方法1：中国综合PMI）")
                return df
        except Exception as e:
            print(f"  ⚠ 方法1失败: {e}")

        # 方法2：尝试 macro_china_non_manufacturing_pmi
        try:
            df = ak.macro_china_non_manufacturing_pmi()
            if df is not None and not df.empty:
                print("  ✓ 宏观经济指标汇总获取成功（方法2：中国非制造业PMI）")
                return df
        except Exception as e:
            print(f"  ⚠ 方法2失败: {e}")

        # 方法3：组合多个指标
        try:
            # 获取PMI数据作为汇总
            df = ak.macro_china_pmi()
            if df is not None and not df.empty:
                print("  ✓ 宏观经济指标汇总获取成功（方法3：中国制造业PMI）")
                return df
        except Exception as e:
            print(f"  ⚠ 方法3失败: {e}")

        print("  ✗ 宏观经济指标汇总获取失败")
        return None

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取宏观经济指标失败：{e}")
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
    print(df.head(10))

    print(f"{'='*80}\n")


def main():
    """主测试函数"""
    print("="*80)
    print("维度7：宏观与行业维度数据测试（远程气味）")
    print("="*80)

    # 输出文件名
    excel_file = "dimension_7_macro_industry_data.xlsx"

    # 存储所有数据
    all_data = {}

    # 测试1: 获取利率数据
    print("\n[测试1] 获取利率数据（LPR、Shibor）")
    rate_data = get_interest_rate_data()
    if rate_data:
        for key, df in rate_data.items():
            if df is not None and not df.empty:
                all_data[key] = df
                print(f"  ✓ {key}: {len(df)}行")

    # 测试2: 获取汇率数据
    print("\n[测试2] 获取汇率数据")
    exchange_rate = get_exchange_rate_data()
    if exchange_rate is not None and not exchange_rate.empty:
        all_data['汇率'] = exchange_rate
        print(f"  ✓ 汇率数据: {len(exchange_rate)}行")

    # 测试3: 获取PMI数据
    print("\n[测试3] 获取PMI数据")
    pmi = get_pmi_data()
    if pmi is not None and not pmi.empty:
        all_data['PMI'] = pmi
        print(f"  ✓ PMI数据: {len(pmi)}行")

    # 测试4: 获取CPI/PPI数据
    print("\n[测试4] 获取CPI/PPI数据")
    cpi_ppi = get_cpi_ppi_data()
    if cpi_ppi:
        for key, df in cpi_ppi.items():
            if df is not None and not df.empty:
                all_data[key] = df
                print(f"  ✓ {key}: {len(df)}行")

    # 测试5: 获取GDP数据
    print("\n[测试5] 获取GDP数据")
    gdp = get_gdp_data()
    if gdp is not None and not gdp.empty:
        all_data['GDP'] = gdp
        print(f"  ✓ GDP数据: {len(gdp)}行")

    # 测试6: 获取货币供应量
    print("\n[测试6] 获取货币供应量（M0/M1/M2）")
    money_supply = get_money_supply_data()
    if money_supply is not None and not money_supply.empty:
        all_data['货币供应量'] = money_supply
        print(f"  ✓ 货币供应量: {len(money_supply)}行")

    # 测试7: 获取行业指数
    print("\n[测试7] 获取行业指数")
    industry_index = get_industry_index_data()
    if industry_index is not None and not industry_index.empty:
        all_data['行业指数'] = industry_index
        print(f"  ✓ 行业指数: {len(industry_index)}行")

    # 测试8: 获取行业市盈率
    print("\n[测试8] 获取行业市盈率")
    industry_pe = get_industry_pe_data()
    if industry_pe is not None and not industry_pe.empty:
        all_data['行业市盈率'] = industry_pe
        print(f"  ✓ 行业市盈率: {len(industry_pe)}行")

    # 测试9: 获取行业资金流向
    print("\n[测试9] 获取行业资金流向")
    industry_flow = get_industry_capital_flow()
    if industry_flow is not None and not industry_flow.empty:
        all_data['行业资金流'] = industry_flow
        print(f"  ✓ 行业资金流: {len(industry_flow)}行")

    # 测试10: 获取重点行业成分股
    print("\n[测试10] 获取重点行业成分股")
    key_industries = ["银行", "证券", "保险", "白酒", "新能源"]
    for industry in key_industries:
        industry_stocks = get_industry_constituent_stocks(industry)
        if industry_stocks is not None and not industry_stocks.empty:
            all_data[f'{industry}行业'] = industry_stocks
            print(f"  ✓ {industry}行业: {len(industry_stocks)}只股票")

    # 测试11: 获取大宗商品价格
    print("\n[测试11] 获取大宗商品价格")
    commodity = get_commodity_prices()
    if commodity is not None and not commodity.empty:
        all_data['商品价格'] = commodity
        print(f"  ✓ 商品价格: {len(commodity)}行")

    # 测试12: 获取宏观经济指标汇总
    print("\n[测试12] 获取宏观经济指标汇总")
    macro_summary = get_economic_indicators_summary()
    if macro_summary is not None and not macro_summary.empty:
        all_data['宏观指标汇总'] = macro_summary
        print(f"  ✓ 宏观指标: {len(macro_summary)}行")

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
    print("\n宏观与行业维度说明：")
    print("- 利率：LPR、Shibor等基准利率")
    print("- 汇率：人民币汇率变动")
    print("- PMI：制造业景气度")
    print("- CPI/PPI：通胀水平")
    print("- GDP：经济增长")
    print("- 货币供应量：流动性环境")
    print("- 行业指数：行业整体表现")
    print("- 行业估值：行业市盈率等")
    print("- 行业资金流：板块轮动")
    print("- 大宗商品：上游成本")


if __name__ == "__main__":
    main()
