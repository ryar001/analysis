import os
import json
from datetime import datetime, timedelta, timezone
from typing import Optional
import pandas as pd

from analysis.trades_analysis import TradesAnalysis
from analysis.components.pull_db_data import PullDBData
from common.db_utils_pandas import DbUtils
from common.constants import ExchangeName
from strategy_trading.StrategyTrading.symbolHelper import SymbolHelper
from strategy_trading.StrategyTrading.tradingHelper import StrategyTradingHelper
import asyncio
from order_management_service.Utils.orders_db_const import ORDERS_DB_COL_TYPE_MAPPING
from pathlib import Path


def load_strategy_helper(conf_fp:str = None)->SymbolHelper:
    strategy_helper = StrategyTradingHelper()

    conf_fp = conf_fp or str(Path(Path(__file__).parent.parent,"common/db.json"))
    print(f"loading conf : {conf_fp}")
    strategy_helper.options = strategy_helper.load_config(conf_fp)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(strategy_helper.load_symbol_helper())
    return strategy_helper.symbol_helper

def load_df(pull_db_data: PullDBData, symbol: Optional[str] = None, timestamp: Optional[int] = None, to_timestamp: Optional[int] = None, exchange: Optional[ExchangeName] = None, strategy_name: Optional[str] = None, strategy_process_name: Optional[str] = None, limit: Optional[int] = 1000) -> Optional[pd.DataFrame]:
    """
    Fetches order data from the database.
    """
    try:
        df = pull_db_data.get_orders(
            global_symbol=symbol,
            timestamp=timestamp,
            to_timestamp=to_timestamp,
            exchange=exchange,
            strategy_name=strategy_name,
            strategy_process_name=strategy_process_name,
            limit=limit
        )
        return df
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_all():
    TABLE_NAME = "orders_history_prod"
    db_utils = None
    analyzer = None
    try:
        strategy_helper = load_strategy_helper()

        # Assumes db.json is in the common directory, and we are running from asset_management root
        db_json_path = os.path.join(os.path.dirname(__file__), '..', 'common', 'db.json')
        with open(db_json_path) as f:
            db_info = json.load(f)['DEFAULT']['Trading']

        db_utils = DbUtils(
            db_user=db_info['user'],
            db_password=db_info['password'],
            db_host=db_info['host'],
            db_port=db_info['port'],
            db_name=db_info.get('dbname', 'Trading'),
            table_name=TABLE_NAME,  # Default table as per request
            primary_key="strategy_order_id",
            unique_pri_key=True,
            db_col_type_mapping=ORDERS_DB_COL_TYPE_MAPPING
        )
        db_utils.start()
        puller = PullDBData(db_utils)
        analyzer = TradesAnalysis()

    except (FileNotFoundError, KeyError) as e:
        print(f"Skipping test: db.json not found or configured incorrectly: {e}")
        analyzer = None
    while 1:
        if not analyzer:
            break
        try:
            # --- User Inputs ---
            print("--- Trade Analysis from Database ---")
            global_symbol = input("Enter global_symbol (e.g., FX-BTC/USDT) [optional, press Enter to skip]: ").upper() or None

            default_ts = int((datetime.now(timezone.utc) - timedelta(hours=24)).timestamp() * 10**6)
            ts_input_str = input(f"Enter start timestamp (us) [default: {default_ts} (24h ago)]: ")
            timestamp = int(ts_input_str) if ts_input_str else default_ts

            to_ts_input_str = input("Enter end timestamp (us) [optional, press Enter to skip]: ")
            to_timestamp = int(to_ts_input_str) if to_ts_input_str else None

            exchange_input = input(f"Enter exchange (e.g., {ExchangeName.XT_FX.value}, {ExchangeName.BINANCE.value}) [default: {ExchangeName.XT_FX.value}]: ") or ExchangeName.XT_FX.value
            
            exch_symbol = None
            if global_symbol:
                symbol_info = strategy_helper.get_info(exchange_input.upper(),global_symbol)
                if symbol_info:
                    exch_symbol = symbol_info.get("exchange_symbol")
            try:
                exchange = ExchangeName(exchange_input.upper())
            except ValueError:
                print(f"Invalid exchange '{exchange_input}'. Setting to None.")
                exchange = None
            
            strategy_name = input("Enter strategy name [optional, press Enter to skip]: ") or None
            strategy_process_name = input("Enter strategy process name [optional, press Enter to skip]: ") or None

            limit_str = input("Enter row limit (default: 1000): ")
            limit = int(limit_str) if limit_str else 1000

            print(f"--- User Inputs ---")
            print(f"global_symbol: {global_symbol} == exchange_symbol: {exch_symbol}")
            print(f"timestamp: {timestamp} ms")
            print(f"to_timestamp: {to_timestamp} ms")
            print(f"exchange: {exchange}")
            print(f"strategy_name: {strategy_name}")
            print(f"strategy_process_name: {strategy_process_name}")
            print(f"limit: {limit}")
            print(f"Loading Data from db, if a of data may take longer ...")

            # --- Analysis ---
            raw_df = load_df(
                pull_db_data=puller,
                symbol=exch_symbol,
                timestamp=timestamp,
                to_timestamp=to_timestamp,
                exchange=exchange,
                strategy_name=strategy_name,
                strategy_process_name=strategy_process_name,
                limit=limit
            )

            if raw_df is None or raw_df.empty:
                print(f"No Data loaded, exiting...")
                return

            parsed_df = analyzer.parse_df(raw_df)
            if parsed_df.empty:
                print(f"No Data parsed, exiting...")
                return


            if parsed_df is None:
                print(f"No Data parsed, exiting...")
                return

            result, filtered_trades = analyzer.run_analysis(
                df=parsed_df,
                symbol=global_symbol,
                timestamp=timestamp,
                exchange=exchange
            )

                    # --- Output ---
            if not result:
                print(f"No Results")
                return
            print("\n" + "="*50)
            print("           Trade Analysis Results")
            print("="*50)
            print(result)
            
            # --- Prepare output paths ---
            time_str = datetime.now().strftime('%y%m%d_%H_%M_%S')
            
            rolling_stats_dir = 'output_files/rolling_stats'
            trades_dir = 'output_files/filters_trades'
            charts_dir = 'output_files/position_timeseries_chart'

            os.makedirs(rolling_stats_dir, exist_ok=True)
            os.makedirs(trades_dir, exist_ok=True)
            os.makedirs(charts_dir, exist_ok=True)

            if analyzer.rolling_stats:
                print("\n" + "="*50)
                print("           Rolling Stats")
                print("="*50)
                rolling_stats_df = analyzer.rolling_stats.to_df()
                print(rolling_stats_df)

                # --- Save Rolling Stats ---
                rolling_stats_filename = f"{rolling_stats_dir}/rolling_stats__{time_str}.csv"
                rolling_stats_df.to_csv(rolling_stats_filename)
                print(f"Rolling stats saved to '{rolling_stats_filename}'")

            # --- Save Files ---
            output_csv_filepath = f"{trades_dir}/filtered_trades_analysis__{time_str}.csv"
            output_chart_filepath = f"{charts_dir}/position_timeseries_chart__{time_str}.png"
            filter_trades_filepath = f"{trades_dir}/filtered_trades__{time_str}.csv"
            
            filtered_trades.to_csv(filter_trades_filepath, index=False)
            print(f"Filtered trade data saved to '{filter_trades_filepath}'")
            
            analyzer.save_filtered_trades(filepath=output_csv_filepath)
            analyzer.save_chart(filepath=output_chart_filepath)
            
            print("\n--- Example of other output formats ---")
            print("dict: ", result.to_dict())
            print("repr: ", repr(result))
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()


    if db_utils:
        db_utils.close()

if __name__ == '__main__':
    run_all()
