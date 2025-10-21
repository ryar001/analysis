import pandas as pd
from datetime import datetime, timedelta, timezone
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dataclasses import dataclass, asdict, field
import json
from typing import Tuple, Optional, List, Dict, Any
from enum import Enum
import os
from common.constants import SymbolType
from uuid import NAMESPACE_DNS, uuid5


# Imports for database integration
from analysis.components.pull_db_data import PullDBData
from common.db_utils_pandas import DbUtils
from common.constants import ExchangeName
from order_management_service.Utils.orders_db_const import ORDERS_DB_COL_TYPE_MAPPING
from analysis.components.models.order_stats_cols import OrderStatsCols, RESULTS_TO_LARK_MAPPINGS, Results, NAMESPACE_ID

# From matplotlib, Figure is the top-level container for all the plot elements.
from matplotlib.figure import Figure


class TradeData(Enum):
    TIME = 'time'
    DATETIME = 'datetime'
    DATETIME_GMT8 = 'datetime_gmt8'
    STR_TIME = 'str_time'
    ORDER_SIDE = 'orderSide'
    QUANTITY = 'quantity'
    QUOTE_QTY = 'quoteQty'
    SIGNED_QUANTITY = 'signed_quantity'
    NET_POSITION = 'net_position'
    SYMBOL = 'symbol'
    PRICE = 'price'
    FEE = 'fee'
    FEE_CURRENCY = 'feeCurrency'
    TAKER_MAKER = 'takerMaker'

@dataclass
class RollingStats:
    """A dataclass to hold rolling statistics."""
    data: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_row(self, timestamp: str, **kwargs):
        """Adds a row of data."""
        self.data[timestamp] = kwargs

    def to_df(self) -> pd.DataFrame:
        """Converts the results to a pandas DataFrame."""
        if not self.data:
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(self.data, orient='index')
        df.index.name = 'timestamp'
        return df

    def to_dict(self) -> dict:
        """Converts the results to a dictionary."""
        return self.data

    def to_json(self, indent: int = 4) -> str:
        """Converts the results to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

class TradesAnalysis:
    """
    A class to analyze trade data from the database, calculate key metrics, and generate charts.
    """
    def __init__(self):
        """
        Initializes the TradesAnalysis class.
        """
        self.filtered_trades: Optional[pd.DataFrame] = None
        self.result: Optional[Results] = None
        self.positions_chart: Optional[Figure] = None
        self.rolling_stats: Optional[RollingStats] = None

    def parse_df(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Parses and prepares raw trade data for analysis.
        """
        try:
            if df.empty:
                print("No trades found for the given criteria.")
                return None

            # --- Data Preparation and Column Mapping ---
            df.rename(columns={
                OrderStatsCols.TIMESTAMP: TradeData.TIME.value,
                OrderStatsCols.GLOBAL_SYMBOL: TradeData.SYMBOL.value,
            }, inplace=True)

            gmt8 = timezone(timedelta(hours=8)) # GMT+8 timezone
            df[TradeData.DATETIME.value] = pd.to_datetime(df[TradeData.TIME.value], unit='us', utc=True)
            df[TradeData.DATETIME_GMT8.value] = df[TradeData.DATETIME.value].dt.tz_convert(gmt8)
            df[TradeData.STR_TIME.value] = df[TradeData.DATETIME_GMT8.value].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3] + ' GMT+8'

            df.sort_values(by=TradeData.TIME.value, inplace=True)
            
            return df
        except Exception as e:
            print(f"An unexpected error occurred while parsing data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_weight_price(self, trades_df: pd.DataFrame) -> float:
        """Calculates the weighted average price from a DataFrame of trades."""
        if trades_df.empty:
            return 0.0
        total_quote_qty = trades_df[TradeData.QUOTE_QTY.value].sum()
        total_quantity = trades_df[TradeData.QUANTITY.value].sum()
        if total_quantity == 0:
            return 0.0
        return total_quote_qty / total_quantity

    def turnover(self, buy_trades_df: pd.DataFrame, sell_trades_df: pd.DataFrame) -> float:
        """
        Calculates the turnover PnL.
        """
        if buy_trades_df.empty or sell_trades_df.empty:
            return 0.0
        
        buy_volume = buy_trades_df[TradeData.QUANTITY.value].sum()
        sell_volume = sell_trades_df[TradeData.QUANTITY.value].sum()

        weighted_avg_buy_price = self._get_weight_price(buy_trades_df)
        weighted_avg_sell_price = self._get_weight_price(sell_trades_df)

        return min(buy_volume, sell_volume) * (weighted_avg_sell_price - weighted_avg_buy_price)        

    def calc_longest_no_trades(self, df: pd.DataFrame) -> int:
        """get the longest no trades time in microsec"""
        if df.empty:
            return 0
        max_no_trade_time = 0

        last_trade_time = None

        for i,row in df.iterrows():
            if last_trade_time is None:
                last_trade_time = row.market_last_trade_timestamp
                max_no_trade_time = row.market_longest_no_trades_this_period
                continue
            # account for cross over period
            cross_over_period = row.market_first_trade_timestamp - last_trade_time

            max_no_trade_time = max(max_no_trade_time, cross_over_period, row.market_longest_no_trades_this_period)

            last_trade_time = row.market_last_trade_timestamp

        return max_no_trade_time

    def run_analysis(self, df: pd.DataFrame, symbol: Optional[str] = None, timestamp: Optional[int] = None, exchange: Optional[ExchangeName] = None, account_name: Optional[str] = None, strategy_name: Optional[str] = None, rolling_period: int = 3600000000,
                    gen_chart:bool=False,prev_day_total_pnl:float=0.0) -> Tuple[Optional[Results], Optional[pd.DataFrame]]:
        """
        Runs the full trade analysis and chart generation process on a given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame with trade data, prepared by load_df.
            symbol (str, optional): The symbol to filter by. Defaults to None.
            timestamp (int, optional): The start timestamp to filter by (in microseconds). Defaults to None.
            exchange (ExchangeName, optional): The exchange to filter by. Defaults to None.
            account_name (str, optional): The account name to filter by. Defaults to None.
            strategy_name (str, optional): The strategy name. Defaults to None.
            rolling_period (int): The rolling period in microseconds. Default is 1 hour.

        Returns:
            A tuple containing the Results object and the filtered_trades DataFrame,
            or (None, None) if an error occurs.
        """
        try:
            if df.empty:
                print("Input DataFrame is empty. Cannot run analysis.")
                return None, None
            self.filtered_trades = df.copy()

            # gmt8 = timezone(timedelta(hours=8))

            start_datetime_utc = pd.to_datetime(timestamp, unit='us', utc=True) if timestamp else self.filtered_trades[TradeData.DATETIME_GMT8.value].iloc[0]

            # --- Calculations ---
            # general
            avg_intervals_between_orders = df[TradeData.TIME.value].diff().mean()
            avg_intervals_between_orders_minutes = avg_intervals_between_orders / 1_000_000 / 60

            # Executions Strategy
            exec_rate = 0.0
            total_orders_sent = df.num_of_orders_sent.sum()
            total_notional_value_of_executed_trades = df.total_notional_value_of_executed_trades.sum()
            total_exec = df.num_of_executed_trades.sum()
            if total_orders_sent > 0:
                exec_rate = total_exec / total_orders_sent

            # pnl
            total_executed_sells_size = df.total_executed_sells_size.sum()
            total_executed_buys_size = df.total_executed_buys_size.sum()

            total_executed_buys_notional = df.total_executed_buys_notional.sum()
            total_executed_sells_notional = df.total_executed_sells_notional.sum()

            avg_sell_executed_price = 0.0
            if total_executed_sells_size > 0:
                avg_sell_executed_price = total_executed_sells_notional / total_executed_sells_size
            avg_buy_executed_price = 0.0
            if total_executed_buys_size > 0:
                avg_buy_executed_price = total_executed_buys_notional / total_executed_buys_size

            turnover_pnl = min(total_executed_sells_size, total_executed_buys_size) * (avg_sell_executed_price - avg_buy_executed_price)
            prev_day_total_pnl = prev_day_total_pnl if prev_day_total_pnl else 0.0
            total_pnl = turnover_pnl + prev_day_total_pnl

            # Executions Rate Per Min
            avg_market_trades = 0.0
            exec_freq = 0.0
            if len(df) > 0 and avg_intervals_between_orders_minutes > 0:
                avg_market_trades = df.total_num_of_market_trades.sum()/ len(df)/ avg_intervals_between_orders_minutes
                exec_freq = total_orders_sent / len(df) / avg_intervals_between_orders_minutes
            total_market_trades_notional = df.total_notional_value_of_market_trades.sum()

            # longest no trades
            longest_no_market_trades = self.calc_longest_no_trades(df)

            # BBO Strategy
            avg_bbo_spread_account = df.bbo_spread_account.mean()
            avg_bbo_spread_market = df.bbo_spread_market.mean()
            
            # liquidity strategy
            avg_liquidity_near = df.total_liquidity_near.sum()/len(df)
            avg_liquidity_far = df.total_liquidity_far.sum()/len(df)
            avg_liquidity_near_notional = df.total_liquidity_near_notional.sum()/len(df)
            avg_liquidity_far_notional = df.total_liquidity_far_notional.sum()/len(df)
            avg_num_of_orders_near = df.total_num_orders_near.sum()/len(df)
            avg_num_of_orders_far = df.total_num_orders_far.sum()/len(df)

            # liquidity market
            avg_liquidity_near_market = df.total_liquidity_near_market.sum()/len(df)
            avg_liquidity_far_market = df.total_liquidity_far_market.sum()/len(df)
            avg_liquidity_near_market_notional = df.total_liquidity_near_notional_market.sum()/len(df)
            avg_liquidity_far_market_notional = df.total_liquidity_far_notional_market.sum()/len(df)
            avg_num_of_orders_near_market = df.total_num_orders_near_market.sum()/len(df)
            avg_num_of_orders_far_market = df.total_num_orders_far_market.sum()/len(df)
            
             # --- Determine symbol and exchange for results ---
            account_name_val = account_name if account_name else (df['account_name'].iloc[0] if not df.empty and df['account_name'].nunique() == 1 else "Multiple")
            strategy_name_val = strategy_name if strategy_name else "Unknown"
            date_val = start_datetime_utc.strftime('%Y-%m-%d')
            exchange_val = exchange if exchange else (df['exchange'].iloc[0] if not df.empty and df['exchange'].nunique() == 1 else "Multiple")
            symbol_val = symbol if symbol else (df[TradeData.SYMBOL.value].iloc[0] if not df.empty and df[TradeData.SYMBOL.value].nunique() == 1 else "Multiple")
            symbol_type = SymbolType( symbol_val.split('-')[0]).value

            # --- Store Results ---
            self.result = Results(
                id=uuid5(NAMESPACE_ID, f"{account_name_val}{strategy_name_val}{date_val}{symbol_val}{exchange_val}"), # a uuid of combination account_name,strategy_name,date,global_symbol,exchange
                account_name=account_name_val,
                strategy_name=strategy_name_val,
                start_datetime_utc=start_datetime_utc.strftime('%Y-%m-%d %H:%M:%S'),
                date=date_val,
                global_symbol=symbol_val,
                exchange_symbol=symbol_val,
                exchange=exchange_val,
                product_type=symbol_type,
                total_orders_sent=total_orders_sent,
                total_executed_trades=total_exec,
                total_notional_value_of_executed_trades=total_notional_value_of_executed_trades,
                avg_buy_executed_price=avg_buy_executed_price,
                total_executed_buys_size=total_executed_buys_size,
                total_executed_buys_notional=total_executed_buys_notional,
                avg_sell_executed_price=avg_sell_executed_price,
                total_executed_sells_size=total_executed_sells_size,
                total_executed_sells_notional=total_executed_sells_notional,
                exec_rate=exec_rate,
                exec_freq=exec_freq,
                today_pnl=turnover_pnl,
                total_pnl=total_pnl,
                avg_bbo_spread_account=avg_bbo_spread_account,
                avg_bbo_spread_market=avg_bbo_spread_market,
                avg_num_of_orders_near=avg_num_of_orders_near,
                avg_num_of_orders_far=avg_num_of_orders_far,
                avg_num_of_orders_near_market=avg_num_of_orders_near_market,
                avg_num_of_orders_far_market=avg_num_of_orders_far_market,
                avg_liquidity_near=avg_liquidity_near,
                avg_liquidity_near_market=avg_liquidity_near_market,
                avg_liquidity_far=avg_liquidity_far,
                avg_liquidity_far_market=avg_liquidity_far_market,
                avg_liquidity_near_notional=avg_liquidity_near_notional,
                avg_liquidity_near_market_notional=avg_liquidity_near_market_notional,
                avg_liquidity_far_notional=avg_liquidity_far_notional,
                avg_liquidity_far_market_notional=avg_liquidity_far_market_notional,
                avg_intervals_between_orders=avg_intervals_between_orders,
                avg_market_trades=avg_market_trades,
                total_market_trades_notional=total_market_trades_notional,
                longest_no_market_trades=longest_no_market_trades,
            )

            # # --- Rolling Analysis ---
            # if rolling_period:
            #     self.rolling_stats = RollingStats()
            #     start_time = self.filtered_trades[TradeData.TIME.value].iloc[0]
            #     end_time = self.filtered_trades[TradeData.TIME.value].iloc[-1]

            #     if df is not None and not df.empty:
            #         df.sort_values(by=TradeData.TIME.value, inplace=True)

            #     current_time = start_time
            #     cumulative_pnl = 0
            #     while current_time < end_time:
            #         period_end_time = current_time + rolling_period
            #         period_df = self.filtered_trades[(self.filtered_trades[TradeData.TIME.value] >= current_time) & (self.filtered_trades[TradeData.TIME.value] < period_end_time)]

            #         num_orders_sent_period = 0
            #         period_orders_df = pd.DataFrame()
            #         if df is not None and not df.empty:
            #             period_orders_df = df[(df[TradeData.TIME.value] >= current_time) & (df[TradeData.TIME.value] < period_end_time)]
            #             num_orders_sent_period = len(period_orders_df)

            #         exec_rate_period = 0.0
            #         if not period_orders_df.empty:
            #             executed_sizes_period = pd.to_numeric(period_orders_df[TradeData.QUANTITY.value].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            #             num_executed_orders_period = (executed_sizes_period > 0).sum()
            #             if num_orders_sent_period > 0:
            #                 exec_rate_period = num_executed_orders_period / num_orders_sent_period

            #         if not period_df.empty:
            #             buy_trades_period = period_df[period_df[TradeData.ORDER_SIDE.value] == 'BUY']
            #             sell_trades_period = period_df[period_df[TradeData.ORDER_SIDE.value] == 'SELL']

            #             turnover_value = self.turnover(buy_trades_period, sell_trades_period)

            #             largest_position_period = period_df[TradeData.NET_POSITION.value].max()
            #             smallest_position_period = period_df[TradeData.NET_POSITION.value].min()
            #             cum_size = period_df[TradeData.NET_POSITION.value].iloc[-1]

            #             buy_quote_qty_period = buy_trades_period[TradeData.QUOTE_QTY.value].sum()
            #             sell_quote_qty_period = sell_trades_period[TradeData.QUOTE_QTY.value].sum()
            #             pnl_period = sell_quote_qty_period - buy_quote_qty_period
            #             cumulative_pnl += pnl_period

            #             avg_weighted_buy_price_period = self._get_weight_price(buy_trades_period)
            #             avg_weighted_sell_price_period = self._get_weight_price(sell_trades_period)
            #             num_buy_trades_period = len(buy_trades_period)
            #             num_sell_trades_period = len(sell_trades_period)

            #             lowest_size = period_df[TradeData.QUANTITY.value].min()
            #             largest_size = period_df[TradeData.QUANTITY.value].max()
            #             avg_size = period_df[TradeData.QUANTITY.value].mean()

            #             period_timestamp_str = pd.to_datetime(current_time, unit='us').strftime('%Y-%m-%d %H:%M:%S')

            #             self.rolling_stats.add_row(
            #                 timestamp=period_timestamp_str,
            #                 turnover=turnover_value,
            #                 largest_position=largest_position_period,
            #                 smallest_position=smallest_position_period,
            #                 cum_size=cum_size,
            #                 PnL=pnl_period,
            #                 cumPnl=cumulative_pnl,
            #                 avg_weighted_buy_price=avg_weighted_buy_price_period,
            #                 avg_weighted_sell_price=avg_weighted_sell_price_period,
            #                 num_buy_trades=num_buy_trades_period,
            #                 num_sell_trades=num_sell_trades_period,
            #                 lowest_size=lowest_size,
            #                 largest_size=largest_size,
            #                 avg_size=avg_size,
            #                 exec_rate=exec_rate_period
            #             )

            #         current_time = period_end_time
            
           
            # --- Chart Generation ---
            # if gen_chart:
            #     self._generate_chart()
            
            return self.result, self.filtered_trades

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            return None, None
            
    def _generate_chart(self) -> None:
        """Generates the position chart and stores it in self.positions_chart."""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(15, 7))

        ax.plot(self.filtered_trades[TradeData.DATETIME_GMT8.value], self.filtered_trades[TradeData.NET_POSITION.value], 
                marker='o', linestyle='-', markersize=4, label='Net Position')
        
        ax.fill_between(self.filtered_trades[TradeData.DATETIME_GMT8.value], self.filtered_trades[TradeData.NET_POSITION.value], 0,
                        where=self.filtered_trades[TradeData.NET_POSITION.value] >= 0, facecolor='green', interpolate=True, alpha=0.3)
        ax.fill_between(self.filtered_trades[TradeData.DATETIME_GMT8.value], self.filtered_trades[TradeData.NET_POSITION.value], 0,
                        where=self.filtered_trades[TradeData.NET_POSITION.value] < 0, facecolor='red', interpolate=True, alpha=0.3)

        ax.set_title('Net Position Over Time', fontsize=16, weight='bold')
        ax.set_xlabel('Time (GMT+8)', fontsize=12)
        ax.set_ylabel('Net Position', fontsize=12)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=timezone(timedelta(hours=8))))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()

        ax.legend()
        plt.tight_layout()

        self.positions_chart = fig
        plt.close(fig) # Prevent auto-display in interactive environments

    def save_chart(self, filepath: str = 'position_timeseries_chart.png') -> None:
        """Saves the generated chart to a file."""
        if self.positions_chart:
            self.positions_chart.savefig(filepath)
            print(f"Chart saved to '{filepath}'")
        else:
            print("No chart generated. Please run analysis first.")

    def save_filtered_trades(self, filepath: str = 'filtered_trades_analysis.csv') -> None:
        """Saves the filtered trade data to a CSV file."""
        if self.filtered_trades is not None and not self.filtered_trades.empty:
            output_columns = [
                TradeData.STR_TIME.value, TradeData.SYMBOL.value, TradeData.ORDER_SIDE.value, 
                TradeData.PRICE.value, TradeData.QUANTITY.value, TradeData.NET_POSITION.value, 
                TradeData.QUOTE_QTY.value, TradeData.FEE.value, TradeData.FEE_CURRENCY.value, 
                TradeData.TAKER_MAKER.value, TradeData.TIME.value
            ]
            # Ensure all columns exist before trying to save
            for col in output_columns:
                if col not in self.filtered_trades.columns:
                    self.filtered_trades[col] = None # Add missing columns with None
            
            self.filtered_trades[output_columns].to_csv(filepath, index=False)
            print(f"Filtered trade data saved to '{filepath}'")
        else:
            print("No filtered trades to save. Please run analysis first.")


# --- Main Execution ---
if __name__ == '__main__':
    db_utils = None
    analyzer = None
    
    try:
        # Assumes db.json is in the common directory, and we are running from asset_management root
        db_json_path = os.path.join(os.path.dirname(__file__), '..', 'common', 'db.json')
        with open(db_json_path) as f:
            db_info = json.load(f)['DEFAULT']['Trading']

        db_col_type_mapping = ORDERS_DB_COL_TYPE_MAPPING

        db_utils = DbUtils(
            db_user=db_info['user'],
            db_password=db_info['password'],
            db_host=db_info['host'],
            db_port=db_info['port'],
            db_name=db_info.get('dbname', 'Trading'),
            table_name="test_pull_db_data_orders",  # Default table as per request
            primary_key="order_id",
            unique_pri_key=True,
            db_col_type_mapping=db_col_type_mapping
        )
        db_utils.start()
        puller = PullDBData(db_utils)
        analyzer = TradesAnalysis()

    except (FileNotFoundError, KeyError) as e:
        print(f"Skipping test: db.json not found or configured incorrectly: {e}")
        analyzer = None
    
    if analyzer:
        # --- User Inputs ---
        print("--- Trade Analysis from Database ---")
        symbol = input("Enter symbol (e.g., FX-BTC/USDT) [optional, press Enter to skip]: ").upper() or None
        
        default_ts = int((datetime.now(timezone.utc) - timedelta(hours=24)).timestamp() * 1000)
        ts_input_str = input(f"Enter start timestamp (ms) [default: {default_ts} (24h ago)]: ")
        timestamp = int(ts_input_str) if ts_input_str else default_ts

        to_ts_input_str = input("Enter end timestamp (ms) [optional, press Enter to skip]: ")
        to_timestamp = int(to_ts_input_str) if to_ts_input_str else None

        exchange_input = input("Enter exchange (e.g., XT_FX, BINANCE) [default: XT_FX]: ") or "XT_FX"
        try:
            exchange = ExchangeName(exchange_input.upper())
        except ValueError:
            print(f"Invalid exchange '{exchange_input}'. Setting to None.")
            exchange = None
        
        strategy_name = input("Enter strategy name [optional, press Enter to skip]: ") or None
        strategy_process_name = input("Enter strategy process name [optional, press Enter to skip]: ") or None

        # --- Analysis ---
        result, filtered_trades = analyzer.run_analysis(
            symbol=symbol,
            timestamp=timestamp,
            to_timestamp=to_timestamp,
            exchange=exchange,
            strategy_name=strategy_name,
            strategy_process_name=strategy_process_name
        )

        # --- Output ---
        if result:
            print("\n" + "="*50)
            print("           Trade Analysis Results")
            print("="*50)
            print(result)
            
            if analyzer.rolling_stats:
                print("\n" + "="*50)
                print("           Rolling Stats")
                print("="*50)
                rolling_stats_df = analyzer.rolling_stats.to_df()
                print(rolling_stats_df)

                # --- Save Rolling Stats ---
                current_ts_ms = int(datetime.now().timestamp() * 1000)
                rolling_stats_filename = f"rolling_stats_{current_ts_ms}.csv"
                rolling_stats_df.to_csv(rolling_stats_filename)
                print(f"Rolling stats saved to '{rolling_stats_filename}'")

            # --- Save Files ---
            output_csv_filepath = 'filtered_trades_analysis.csv'
            output_chart_filepath = 'position_timeseries_chart.png'
            
            analyzer.save_filtered_trades(filepath=output_csv_filepath)
            analyzer.save_chart(filepath=output_chart_filepath)
            
            print("\n--- Example of other output formats ---")
            print("dict: ", result.to_dict())
            print("repr: ", repr(result))

    if db_utils:
        db_utils.close()

