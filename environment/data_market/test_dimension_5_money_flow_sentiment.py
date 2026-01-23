"""
维度5：资金流向与情绪维度数据获取测试（气味+味道感知）

包含维度：
- 主力净流入：大资金买入-卖出
- 北向资金流：外资买卖动态
- 情绪指数：新闻/社交舆情情绪
- 板块涨跌幅：行业资金流动

Usage:
    python environment/test_dimension_5_money_flow_sentiment.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import sys
import os


def get_individual_money_flow(symbol: str = "600519") -> Optional[pd.DataFrame]:
    """
    获取个股资金流向数据

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
        print(f"获取个股资金流向失败：{e}")
        return None


def get_market_money_flow(symbol: str = "沪深京A股") -> Optional[pd.DataFrame]:
    """
    获取市场整体资金流向

    Args:
        symbol: 市场类型

    Returns:
        市场资金流向数据
    """
    try:
        import akshare as ak

        print(f"正在获取{symbol}资金流向数据...")

        # 获取市场资金流
        df = ak.stock_market_fund_flow()

        return df

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取市场资金流向失败：{e}")
        return None


def get_sector_money_flow() -> Optional[pd.DataFrame]:
    """
    获取板块资金流向数据

    Returns:
        板块资金流向数据
    """
    try:
        import akshare as ak

        print("正在获取板块资金流向数据...")

        # 获取行业板块资金流
        df = ak.stock_sector_fund_flow_rank(indicator="今日")

        return df

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取板块资金流向失败：{e}")
        return None


def get_concept_money_flow() -> Optional[pd.DataFrame]:
    """
    获取概念板块资金流向

    Returns:
        概念板块资金流向数据
    """
    try:
        import akshare as ak

        print("正在获取概念板块资金流向数据...")

        # 获取概念板块资金流
        df = ak.stock_sector_fund_flow_rank(indicator="今日")

        return df

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取概念资金流向失败：{e}")
        return None


def get_north_money_flow(
    start_date: str = "20240101",
    end_date: str = "20241231"
) -> Optional[pd.DataFrame]:
    """
    获取北向资金流向（沪深港通）

    Args:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        北向资金数据
    """
    try:
        import akshare as ak

        print("正在获取北向资金流向数据...")

        # 获取沪深港通资金流向
        df = ak.stock_hsgt_hist_em(start_date=start_date, end_date=end_date)

        if df is not None and not df.empty:
            # 计算累计流入
            df['北向资金累计'] = df['北向资金'].cumsum()

            # 计算移动平均
            df['北向资金_MA5'] = df['北向资金'].rolling(5).mean()
            df['北向资金_MA10'] = df['北向资金'].rolling(10).mean()
            df['北向资金_MA20'] = df['北向资金'].rolling(20).mean()

        return df

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取北向资金失败：{e}")
        return None


def get_north_top_stocks() -> Optional[pd.DataFrame]:
    """
    获取北向资金十大成交股

    Returns:
        北向资金十大成交股数据
    """
    try:
        import akshare as ak

        print("正在获取北向资金十大成交股...")

        # 获取北向资金十大成交股
        df = ak.stock_hsgt_board_rank_em(symbol="北向资金增持行业板块排行", indicator="今日")

        return df

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取北向资金十大成交股失败：{e}")
        return None


def get_margin_trading_data() -> Optional[pd.DataFrame]:
    """
    获取融资融券数据

    Returns:
        融资融券数据
    """
    try:
        import akshare as ak

        print("正在获取融资融券数据...")

        # 获取融资融券汇总数据
        df = ak.stock_margin_sse()

        return df

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取融资融券数据失败：{e}")
        return None


def get_market_sentiment_index() -> Optional[pd.DataFrame]:
    """
    获取市场情绪指数

    Returns:
        市场情绪指数数据
    """
    try:
        import akshare as ak

        print("正在获取市场情绪指数...")

        # 获取市场情绪指数（通过涨跌家数等指标）
        df = ak.stock_zh_a_spot_em()

        if df is not None and not df.empty:
            # 计算市场情绪指标
            total_stocks = len(df)
            rising_stocks = len(df[df['涨跌幅'] > 0])
            falling_stocks = len(df[df['涨跌幅'] < 0])
            flat_stocks = len(df[df['涨跌幅'] == 0])

            # 涨停跌停数量
            limit_up = len(df[df['涨跌幅'] >= 9.9])
            limit_down = len(df[df['涨跌幅'] <= -9.9])

            sentiment_data = {
                '总股票数': [total_stocks],
                '上涨家数': [rising_stocks],
                '下跌家数': [falling_stocks],
                '平盘家数': [flat_stocks],
                '涨停家数': [limit_up],
                '跌停家数': [limit_down],
                '上涨比例': [rising_stocks / total_stocks * 100],
                '下跌比例': [falling_stocks / total_stocks * 100],
                '涨跌比': [rising_stocks / falling_stocks if falling_stocks > 0 else np.nan],
                '市场情绪': ['强势' if rising_stocks > falling_stocks * 1.5 else ('弱势' if falling_stocks > rising_stocks * 1.5 else '中性')],
                '统计时间': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            }

            return pd.DataFrame(sentiment_data)

        return None

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取市场情绪指数失败：{e}")
        return None


def get_dragon_tiger_list() -> Optional[pd.DataFrame]:
    """
    获取龙虎榜数据（机构和游资动向）

    Returns:
        龙虎榜数据
    """
    try:
        import akshare as ak

        print("正在获取龙虎榜数据...")

        # 获取最近的龙虎榜数据
        today = datetime.now().strftime('%Y%m%d')
        df = ak.stock_lhb_detail_em(start_date="20240101", end_date=today)

        return df

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取龙虎榜数据失败：{e}")
        return None


def get_big_deal_data() -> Optional[pd.DataFrame]:
    """
    获取大宗交易数据

    Returns:
        大宗交易数据
    """
    try:
        import akshare as ak

        print("正在获取大宗交易数据...")

        # 获取大宗交易数据
        today = datetime.now().strftime('%Y%m%d')
        df = ak.stock_dzjy_sctj(start_date="20240101", end_date=today)

        return df

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取大宗交易数据失败：{e}")
        return None


def get_hot_stocks_ranking() -> Optional[pd.DataFrame]:
    """
    获取热门股票排行（关注度）

    Returns:
        热门股票数据
    """
    try:
        import akshare as ak

        print("正在获取热门股票排行...")

        # 获取东方财富人气榜
        df = ak.stock_hot_rank_em()

        return df

    except ImportError:
        print("错误：需要安装 akshare 库")
        return None
    except Exception as e:
        print(f"获取热门股票排行失败：{e}")
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

    print(f"\n列名: {list(df.columns)}")
    print(f"\n前10行数据:")
    print(df.head(10))

    print(f"{'='*80}\n")


def main():
    """主测试函数"""
    print("="*80)
    print("维度5：资金流向与情绪维度数据测试（气味+味道感知）")
    print("="*80)

    # 输出文件名
    excel_file = "dimension_5_money_flow_sentiment_data.xlsx"

    # 存储所有数据
    all_data = {}

    # 测试1: 获取个股资金流向
    print("\n[测试1] 获取个股资金流向（贵州茅台）")
    individual_flow = get_individual_money_flow(symbol="600519")
    if individual_flow is not None:
        print_data_summary(individual_flow, "茅台资金流向")
        all_data['个股资金流_茅台'] = individual_flow

    # 测试2: 获取市场整体资金流向
    print("\n[测试2] 获取市场整体资金流向")
    market_flow = get_market_money_flow()
    if market_flow is not None:
        print_data_summary(market_flow, "市场资金流向")
        all_data['市场资金流'] = market_flow

    # 测试3: 获取板块资金流向
    print("\n[测试3] 获取板块资金流向")
    sector_flow = get_sector_money_flow()
    if sector_flow is not None:
        print_data_summary(sector_flow, "板块资金流向")
        all_data['板块资金流'] = sector_flow

    # 测试4: 获取北向资金流向
    print("\n[测试4] 获取北向资金流向（沪深港通）")
    north_flow = get_north_money_flow(start_date="20240101", end_date="20241231")
    if north_flow is not None:
        print_data_summary(north_flow, "北向资金流向")
        all_data['北向资金'] = north_flow

    # 测试5: 获取北向资金十大成交股
    print("\n[测试5] 获取北向资金十大成交股")
    north_top = get_north_top_stocks()
    if north_top is not None:
        print_data_summary(north_top, "北向资金十大成交股")
        all_data['北向十大成交股'] = north_top

    # 测试6: 获取融资融券数据
    print("\n[测试6] 获取融资融券数据")
    margin_data = get_margin_trading_data()
    if margin_data is not None:
        print_data_summary(margin_data, "融资融券数据")
        all_data['融资融券'] = margin_data

    # 测试7: 获取市场情绪指数
    print("\n[测试7] 获取市场情绪指数")
    sentiment = get_market_sentiment_index()
    if sentiment is not None:
        print_data_summary(sentiment, "市场情绪指数")
        all_data['市场情绪'] = sentiment

    # 测试8: 获取龙虎榜数据
    print("\n[测试8] 获取龙虎榜数据")
    lhb_data = get_dragon_tiger_list()
    if lhb_data is not None:
        print_data_summary(lhb_data, "龙虎榜数据")
        all_data['龙虎榜'] = lhb_data

    # 测试9: 获取大宗交易数据
    print("\n[测试9] 获取大宗交易数据")
    big_deal = get_big_deal_data()
    if big_deal is not None:
        print_data_summary(big_deal, "大宗交易数据")
        all_data['大宗交易'] = big_deal

    # 测试10: 获取热门股票排行
    print("\n[测试10] 获取热门股票排行")
    hot_stocks = get_hot_stocks_ranking()
    if hot_stocks is not None:
        print_data_summary(hot_stocks, "热门股票排行")
        all_data['热门股票'] = hot_stocks

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
    print("\n资金流向与情绪维度说明：")
    print("- 主力资金流向：大资金进出情况")
    print("- 北向资金：外资对A股的态度")
    print("- 板块资金流：行业轮动和热点")
    print("- 融资融券：杠杆资金动向")
    print("- 市场情绪：涨跌家数、涨跌比")
    print("- 龙虎榜：机构和游资动向")
    print("- 大宗交易：大股东和机构交易")
    print("- 热门股票：市场关注度")


if __name__ == "__main__":
    main()
