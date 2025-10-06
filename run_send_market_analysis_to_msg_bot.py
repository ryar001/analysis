import os
import yaml
import asyncio
import json
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Assuming these modules exist and can be imported.
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

from common.db_utils_pandas import DbUtils

def load_settings(settings_path: str) -> Dict[str, Any]:
    """Loads settings from a YAML file."""
    with open(settings_path, 'r') as f:
        return yaml.safe_load(f)

def get_db_utils(db_info: Dict[str, Any], table_name: str) -> DbUtils:
    """Initializes and returns a DbUtils instance for the table."""
    return DbUtils(
        db_user=db_info['user'],
        db_password=db_info['password'],
        db_host=db_info['host'],
        db_port=db_info['port'],
        db_name=db_info.get('dbname', 'Trading'),
        table_name=table_name,
        primary_key="id", # Assuming 'id' is the primary key
    )

async def get_latest_market_stats(
    db_utils: DbUtils,
    from_timestamp: int,
    exchange: str,
    symbol: str,
    logger=None
) -> pd.DataFrame:
    """Pulls the latest market stats from the table."""
    # Timestamps in the DB are likely in nanoseconds or microseconds.
    # The reference script uses microseconds (ts * 10**6). Let's assume the same for mds_stat.
    where_clause = 'symbol = :symbol AND exchange = :exchange AND "timestamp" >= :from_timestamp ORDER BY "timestamp" DESC'
    params = {
        "symbol": symbol,
        "exchange": exchange,
        "from_timestamp": from_timestamp
    }
    test_where = 'symbol = :symbol AND exchange = :exchange ORDER BY "timestamp" DESC'

    if logger:
        logger.debug(f"Executing get_table_as_df with where_clause: {where_clause} and params: {params}")

    try:
        stats_df = db_utils.get_table_as_df(where_clause=where_clause, params=params, limit=1)
        if stats_df is not None and not stats_df.empty:
            return stats_df
    except Exception as e:
        if logger:
            logger.error(f"Failed to fetch market stats for {symbol}@{exchange}: {e}")

    return pd.DataFrame()


def format_stats_for_message(stats_df: pd.DataFrame) -> str:
    """Formats the stats DataFrame into a string for the message bot."""
    if stats_df.empty:
        return "No recent market stats available."

    # Convert the first (and only) row to a dictionary
    stats_dict = stats_df.iloc[0].to_dict()

    # Format the dictionary into a readable string
    message = ""
    for key, value in stats_dict.items():
        # Convert timestamp to a readable format if it exists
        if 'timestamp' in key and isinstance(value, (int, float)):
            # Assuming timestamp is in microseconds
            dt_object = datetime.fromtimestamp(value / 10**6, tz=timezone.utc)
            value = dt_object.strftime('%Y-%m-%d %H:%M:%S %Z')
        message += f"    {key}: {value}\n"

    return message.strip()


async def main():
    """Main function to run the analysis and send reports."""
    settings_path = "./send_to_lark_settings.yaml"

    try:
        settings = load_settings(settings_path).get("market_stats", {})
    except FileNotFoundError:
        print(f"Settings file not found at {settings_path}. Please create it.")
        return

    log_setting = settings.get("log_setting", {})
    logger = setup_logger("run_send_market_analysis_to_msg_bot", log_setting)

    msg_bot_settings = settings.get("msg_bot_settings", {})
    markets = settings.get("markets", {})

    if not all([msg_bot_settings, markets]):
        logger.error("Incomplete settings in 'market_stats'. Please check the format.")
        return

    msg_bot = MsgBot(**msg_bot_settings)
    db_json_path = str(Path(__file__).parent.parent/"common/db.json")
    # db_json_path = os.path.join(os.path.dirname(__file__), '..', 'common', 'db.json')
    breakpoint()
    try:
        with open(db_json_path) as f:
            db_info = json.load(f)['DEFAULT']['Trading']
    except (FileNotFoundError, KeyError) as e:
        logger.error(f"db.json not found or configured incorrectly: {e}")
        return

    db_utils = get_db_utils(db_info, settings.get("table_name", "mds_stat"))
    db_utils.start()

    # Get data from the last 24 hours
    to_timestamp = int(datetime.now(timezone.utc).timestamp() * 10**6)
    from_timestamp = to_timestamp - int(24 * 60 * 60 * 10**6)

    for market_name, market_details in markets.items():
        symbols_to_check = market_details.get("strategy_and_process_name", []) # Reusing the key from the other config

        if not symbols_to_check:
            logger.warning(f"No symbols found for market group: {market_name}")
            continue

        for symbol_info in symbols_to_check:
            symbol = symbol_info.get("symbol")
            exchange = symbol_info.get("exchange")

            if not symbol or not exchange:
                logger.warning(f"Skipping incomplete symbol info: {symbol_info}")
                continue

            logger.info(f"Fetching latest market stats for {symbol}@{exchange}...")

            stats_df = await get_latest_market_stats(
                db_utils,
                from_timestamp,
                exchange,
                symbol,
                logger
            )

            message_body = format_stats_for_message(stats_df)
            title = f"Market Stats for {symbol}@{exchange}"

            logger.info(f"Sending message for {symbol}@{exchange}")
            # The MsgBot seems to handle title formatting via its own settings,
            # but we pass symbol and exchange for context.
            msg_bot.send_msg(message_body, symbol=symbol, exchange=exchange)

    if db_utils:
        db_utils.close()

if __name__ == "__main__":
    asyncio.run(main())
