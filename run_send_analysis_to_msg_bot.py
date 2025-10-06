
import os
import yaml
import asyncio
import json
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Tuple
from pathlib import Path
import hashlib
from strategy_trading.StrategyTrading.symbolHelper import SymbolHelper
from strategy_trading.StrategyTrading.tradingHelper import StrategyTradingHelper

# Assuming these modules exist and can be imported.
# If not, the user needs to provide them.
try:
    from common.lark_wrapper.msg_bot import MsgBot
    from common.logging_utils import LoggingUtils
    def setup_logger(name, log_setting: Dict[str, Any]):
        return LoggingUtils(log_name=name,log_dir=log_setting.get("log_dir", None),log_level=log_setting.get("log_level", "INFO"),print_output=log_setting.get("print_output", True),json_formatter=log_setting.get("json_formatter", True)).get_logger()
except ImportError:
    print("Could not import common modules. Please ensure they are in the python path.")
    # Define dummy functions for now
    def send_message(webhook_url: str, secret: str, message: str, title: str, msg_type: str):
        print(f"--- MOCK SEND MESSAGE ---\nWebhook: {webhook_url}\nSecret: {secret}\nTitle: {title}\nMsg_Type: {msg_type}\nMessage:\n{message}\n--- END MOCK ---")

    def setup_logger(name, log_setting: Dict[str, Any]):
        import logging
        logger = logging.getLogger(name)
        logger.setLevel(log_setting.get("log_level", "INFO"))
        handler = logging.StreamHandler()
        formatter = logging.Formatter(log_setting.get("log_format", '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger


from analysis.trades_analysis import TradesAnalysis, Results
from analysis.components.pull_db_data import PullDBData
from common.db_utils_pandas import DbUtils
from common.constants import ExchangeName
from order_management_service.Utils.orders_db_const import ORDERS_DB_COL_TYPE_MAPPING
from dataclasses import asdict
import numpy as np


def convert_numpy_types(data):
    """
    Recursively converts numpy types to native Python types
    in a dictionary or a list.
    """
    if isinstance(data, (np.int64, np.int32)):
        return int(data)
    if isinstance(data, (np.float64, np.float32)):
        return float(data)
    return data


STRATEGY_STATS_DB_COL_TYPE_MAPPING = {
    'unique_key': 'VARCHAR(255)',
    'timestamp': 'BIGINT',
    'date': 'TEXT',
    'account_name': 'VARCHAR(255)',
    'strategy_name': 'VARCHAR(255)',
    'process_name': 'VARCHAR(255)',
    'start_datetime_str': 'TEXT',
    'earliest_trade_time': 'TEXT',
    'last_trade_time': 'TEXT',
    'global_symbol': 'TEXT',
    'exchange_symbol': 'TEXT',
    'exchange': 'TEXT',
    'product_type': 'TEXT',
    'first_trade_price': 'DOUBLE PRECISION',
    'last_trade_price': 'DOUBLE PRECISION',
    'pnl': 'DOUBLE PRECISION',
    'turnover': 'DOUBLE PRECISION',
    'executed_buys': 'INTEGER',
    'executed_sells': 'INTEGER',
    'num_buy_trades': 'INTEGER',
    'num_sell_trades': 'INTEGER',
    'total_buy_size': 'DOUBLE PRECISION',
    'total_sell_size': 'DOUBLE PRECISION',
    'avg_weighted_buy_price': 'DOUBLE PRECISION',
    'avg_weighted_sell_price': 'DOUBLE PRECISION',
    'max_long_position': 'DOUBLE PRECISION',
    'max_short_position': 'DOUBLE PRECISION',
    'largest_position': 'DOUBLE PRECISION',
    'position_type': 'TEXT',
    'exec_rate': 'DOUBLE PRECISION',
}

def get_strategy_stats_db_utils(db_info: Dict[str, Any], table_name: str) -> DbUtils:
    """Initializes and returns a DbUtils instance for strategy_stats."""
    return DbUtils(
        db_user=db_info['user'],
        db_password=db_info['password'],
        db_host=db_info['host'],
        db_port=db_info['port'],
        db_name=db_info.get('dbname', 'Trading'),
        table_name=table_name,
        primary_key="unique_key",
        unique_pri_key=True,
        last_update_time_key="timestamp",
        db_col_type_mapping=STRATEGY_STATS_DB_COL_TYPE_MAPPING
    )

async def store_strategy_stats(db_utils: DbUtils, stats_data: dict):
    """Stores strategy stats in the database, updating on conflict."""
    stats_data = {k: convert_numpy_types(v) for k, v in stats_data.items()}
    
    conflict_cols = ["date", "account_name", "strategy_name", "process_name", "global_symbol"]
    
    key_parts = []
    for col in conflict_cols:
        value = stats_data.get(col)
        key_parts.append(str(value) if value is not None else "") 
    
    unique_key_string = "|".join(key_parts)
    unique_key = hashlib.md5(unique_key_string.encode()).hexdigest()
    stats_data["unique_key"] = unique_key

    await asyncio.to_thread(db_utils.upsert_order, order_dict=stats_data)

def load_settings(settings_path: str) -> Dict[str, Any]:
    """Loads settings from a YAML file."""
    with open(settings_path, 'r') as f:
        return yaml.safe_load(f)

def get_db_utils(db_info: Dict[str, Any]) -> DbUtils:
    """Initializes and returns a DbUtils instance."""
    return DbUtils(
        db_user=db_info['user'],
        db_password=db_info['password'],
        db_host=db_info['host'],
        db_port=db_info['port'],
        db_name=db_info.get('dbname', 'Trading'),
        table_name="orders_history_prod",
        primary_key="strategy_order_id",
        unique_pri_key=True,
        db_col_type_mapping=ORDERS_DB_COL_TYPE_MAPPING
    )

async def run_analysis_for_strategy(
    puller: PullDBData,
    analyzer: TradesAnalysis,
    strategy_name: str,
    process_name: str,
    from_timestamp: int,
    to_timestamp: int,
    exchange: ExchangeName,
    symbol: str = None,
) -> Tuple[Results,pd.DataFrame]:
    """Runs analysis for a single strategy."""
    raw_df = puller.get_orders(
        timestamp=from_timestamp,
        to_timestamp=to_timestamp,
        exchange=exchange,
        strategy_name=strategy_name,
        strategy_process_name=process_name,
        symbol=symbol,
        limit=100000  # High limit to get all trades in the last 24h
    )

    if raw_df is None or raw_df.empty:
        return None,raw_df

    parsed_df = analyzer.parse_df(raw_df)
    if parsed_df is None or parsed_df.empty:
        return None,parsed_df

    result, filtered_trades = analyzer.run_analysis(
        df=parsed_df,
        timestamp=from_timestamp,
        exchange=exchange
    )
    return result,filtered_trades

async def load_strategy_helper_async(conf_fp: str = None, logger=None) -> SymbolHelper:
    """Loads the strategy helper asynchronously."""
    strategy_helper = StrategyTradingHelper()
    conf_fp = conf_fp or str(Path(Path(__file__).parent.parent, "common", "db.json"))
    if logger:
        logger.info(f"Loading conf: {conf_fp}")
    else:
        print(f"Loading conf: {conf_fp}")
    strategy_helper.options = strategy_helper.load_config(conf_fp)
    await strategy_helper.load_symbol_helper()
    return strategy_helper.symbol_helper

async def main():
    """Main function to run the analysis and send reports."""
    settings_path = "./send_to_lark_settings.yaml"
    
    try:
        settings = load_settings(settings_path).get("private",{})
    except FileNotFoundError:
        print(f"Settings file not found at {settings_path}. Please create it.")
        return
   
    log_setting = settings.get("log_setting", {})

    logger = setup_logger("run_send_analysis_to_msg_bot", log_setting)
    
    msg_bot_settings = settings.get("msg_bot_settings", {})
    accounts = settings.get("accounts", {})
    logger.info(f"msg_bot_settings: {msg_bot_settings}")
    if not all([msg_bot_settings, accounts]):
        logger.error("Incomplete settings file. Please check the format.")
        return

    msg_bot = MsgBot(**msg_bot_settings)
    send_message = msg_bot.send_msg

    db_json_path = str(Path(__file__).parent.parent/"common/db.json")
    try:
        with open(db_json_path) as f:
            db_info = json.load(f)['DEFAULT']['Trading']
    except (FileNotFoundError, KeyError) as e:
        logger.error(f"db.json not found or configured incorrectly: {e}")
        return

    db_utils = get_db_utils(db_info)
    db_utils.start()
    strategy_stats_table_name = settings.get("table_name")
    strategy_stats_db_utils = None
    if strategy_stats_table_name:
        strategy_stats_db_utils = get_strategy_stats_db_utils(db_info, strategy_stats_table_name)
        
        # --- TEMPORARY CODE TO RECREATE TABLE WITH NEW SCHEMA ---
        # This will delete the existing 'strategy_stats' table to allow it 
        # to be recreated with the new 'unique_key' primary key.
        # This is a one-time operation and this code should be removed after a successful run.
        strategy_stats_db_utils.connect()

        # --- END OF TEMPORARY CODE ---

        strategy_stats_db_utils.start()
        logger.info(f"strategy_stats_db_utils started.")
        
    puller = PullDBData(db_utils)
    analyzer = TradesAnalysis()
    strategy_helper = await load_strategy_helper_async(logger=logger)

    to_timestamp = int(datetime.now(timezone.utc).timestamp() * 10**6)
    from_timestamp = to_timestamp - int(24 * 60 * 60 * 10**6)

    for account_name, account_details in accounts.items():
        strategies = account_details.get("strategy_and_process_name", [])
        
        if not strategies:
            logger.warning(f"No strategies found for account: {account_name}")
            continue

        account_message = f"{account_name} Strategies\n"
        
        for strategy_info in strategies:
            strategy_name = strategy_info.get("strategy_name")
            process_name = strategy_info.get("process_name")
            global_symbol = strategy_info.get("global_symbol")
            exchange_str = strategy_info.get("exchange")
            exchange = ExchangeName(exchange_str.upper()) if exchange_str else None

            if not strategy_name and not process_name:
                logger.warning(f"Skipping incomplete strategy info: {strategy_info}")
                continue

            logger.info(f"Running analysis for account '{account_name}', strategy '{strategy_name}'...")
            
            exch_symbol = None

            if strategy_helper:
                if global_symbol and exchange and strategy_helper:
                                symbol_info = strategy_helper.get_info(exchange.value, global_symbol)
                                if symbol_info:
                                    exch_symbol = symbol_info.get("exchange_symbol")
            analysis_result,_filtered_trades = await run_analysis_for_strategy(
                puller,
                analyzer,
                strategy_name,
                process_name,
                from_timestamp,
                to_timestamp,
                exchange=exchange,
                symbol=exch_symbol
            )

            if not analysis_result:
                send_message(f"No analysis results for {account_name} - {strategy_name} - {process_name}",account_name=account_name)
                continue

            if strategy_stats_db_utils and analysis_result:
                stats_dict = analysis_result.to_dict()
                stats_dict['timestamp'] = to_timestamp
                stats_dict['date'] = datetime.fromtimestamp(to_timestamp / 10**6, tz=timezone.utc).strftime('%Y-%m-%d')
                stats_dict['account_name'] = account_name
                stats_dict['strategy_name'] = strategy_name
                stats_dict['process_name'] = process_name
                stats_dict['global_symbol'] = global_symbol
                stats_dict['exchange'] = exchange.value if exchange else 'N/A'
                await store_strategy_stats(strategy_stats_db_utils, stats_dict)

            account_message += f"strategy_name: {strategy_name}\n"
            account_message += f"process_name: {process_name}\n"
            account_message += f"global_symbol: {global_symbol}\n"
            account_message += f"exchange: {exchange.value if exchange else 'N/A'}\n\n"
            
            if analysis_result:
                account_message += "Strategy Account Analysis Result:\n"
                account_message += f"{analysis_result.to_msg_bot_msg()}\n\n"
            else:
                account_message += "No analysis results for this strategy in the last 24 hours.\n\n"

        logger.info(f"Sending message for account '{account_name}'")
                
        send_message(account_message,account_name=account_name)

    if db_utils:
        db_utils.close()

    if strategy_stats_db_utils:
        strategy_stats_db_utils.close()

if __name__ == "__main__":
    asyncio.run(main())
