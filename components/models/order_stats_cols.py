from enum import Enum
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from datetime import datetime
import json
from uuid import NAMESPACE_DNS, uuid5
import uuid

import pandas as pd

from pydantic.dataclasses import dataclass as pydantic_dataclass
from dataclasses import fields as dc_fields, is_dataclass
from typing import Any, Type


NAMESPACE_ID = uuid.UUID("e48242bb-f528-434a-ac8c-ba12b9e8d531")

class OrderStatsCols(str, Enum):
    UPDATE_ID = "update_id"
    ACCOUNT_NAME = "account_name"
    EXCHANGE = "exchange"
    GLOBAL_SYMBOL = "global_symbol"
    TIMESTAMP = "timestamp"
    NUM_OF_ORDERS_SENT = "num_of_orders_sent"
    NUM_OF_EXECUTED_TRADES = "num_of_executed_trades"
    EXECUTION_RATE = "execution_rate"
    EXECUTION_FREQUENCY = "execution_frequency"
    TOTAL_QTY_OF_EXECUTED_TRADES = "total_qty_of_executed_trades"
    TOTAL_NOTIONAL_VALUE_OF_EXECUTED_TRADES = "total_notional_value_of_executed_trades"
    TOTAL_QTY_OF_MARKET_TRADES = "total_qty_of_market_trades"
    TOTAL_NOTIONAL_VALUE_OF_MARKET_TRADES = "total_notional_value_of_market_trades"
    BBO_SPREAD_ACCOUNT = "bbo_spread_account"
    BBO_SPREAD_MARKET = "bbo_spread_market"
    TOTAL_LIQUIDITY_NEAR = "total_liquidity_near"
    TOTAL_LIQUIDITY_NEAR_MARKET = "total_liquidity_near_market"
    TOTAL_LIQUIDITY_FAR = "total_liquidity_far"
    TOTAL_LIQUIDITY_FAR_MARKET = "total_liquidity_far_market"
    TOTAL_LIQUIDITY_NEAR_NOTIONAL = "total_liquidity_near_notional"
    TOTAL_LIQUIDITY_NEAR_NOTIONAL_MARKET = "total_liquidity_near_notional_market"
    TOTAL_LIQUIDITY_FAR_NOTIONAL = "total_liquidity_far_notional"
    TOTAL_LIQUIDITY_FAR_NOTIONAL_MARKET = "total_liquidity_far_notional_market"
    MARKET_LONGEST_NO_TRADES_THIS_PERIOD = "market_longest_no_trades_this_period"
    MARKET_FIRST_TRADE_TIMESTAMP = "market_first_trade_timestamp"
    MARKET_LAST_TRADE_TIMESTAMP = "market_last_trade_timestamp"


class SqlType(str, Enum):
    UNIQUE_KEY = 'unique_key'
    PRIMARY_KEY = 'primary_key'
    FOREIGN_KEY = 'foreign_key'

class ResultsField(str, Enum):
    ID = "id"
    ACCOUNT_NAME = "account_name"
    STRATEGY_NAME = "strategy_name"
    START_DATETIME_UTC = "start_datetime_utc"
    DATE = "date"
    GLOBAL_SYMBOL = "global_symbol"
    EXCHANGE_SYMBOL = "exchange_symbol"
    EXCHANGE = "exchange"
    PRODUCT_TYPE = "product_type"
    TOTAL_ORDERS_SENT = "total_orders_sent"
    TOTAL_EXECUTED_TRADES = "total_executed_trades"
    TOTAL_NOTIONAL_VALUE_OF_EXECUTED_TRADES = "total_notional_value_of_executed_trades"
    AVG_BUY_EXECUTED_PRICE = "avg_buy_executed_price"
    TOTAL_EXECUTED_BUYS_SIZE = "total_executed_buys_size"
    TOTAL_EXECUTED_BUYS_NOTIONAL = "total_executed_buys_notional"
    AVG_SELL_EXECUTED_PRICE = "avg_sell_executed_price"
    TOTAL_EXECUTED_SELLS_SIZE = "total_executed_sells_size"
    TOTAL_EXECUTED_SELLS_NOTIONAL = "total_executed_sells_notional"
    EXEC_RATE = "exec_rate"
    EXEC_FREQ = "exec_freq"
    TODAY_PNL = "today_pnl"
    TOTAL_PNL = "total_pnl"
    AVG_BBO_SPREAD_ACCOUNT = "avg_bbo_spread_account"
    AVG_BBO_SPREAD_MARKET = "avg_bbo_spread_market"
    AVG_LIQUIDITY_NEAR = "avg_liquidity_near"
    AVG_LIQUIDITY_NEAR_MARKET = "avg_liquidity_near_market"
    AVG_LIQUIDITY_FAR = "avg_liquidity_far"
    AVG_LIQUIDITY_FAR_MARKET = "avg_liquidity_far_market"
    AVG_LIQUIDITY_NEAR_NOTIONAL = "avg_liquidity_near_notional"
    AVG_LIQUIDITY_NEAR_MARKET_NOTIONAL = "avg_liquidity_near_market_notional"
    AVG_LIQUIDITY_FAR_NOTIONAL = "avg_liquidity_far_notional"
    AVG_LIQUIDITY_FAR_MARKET_NOTIONAL = "avg_liquidity_far_market_notional"
    AVG_INTERVALS_BETWEEN_ORDERS = "avg_intervals_between_orders"
    AVG_MARKET_TRADES = "avg_market_trades"
    TOTAL_MARKET_TRADES_NOTIONAL = "total_market_trades_notional"
    LONGEST_NO_MARKET_TRADES = "longest_no_market_trades"
    TIMESTAMP = "timestamp"
    AVG_NUM_OF_ORDERS_NEAR = "avg_num_of_orders_near"
    AVG_NUM_OF_ORDERS_FAR = "avg_num_of_orders_far"
    AVG_NUM_OF_ORDERS_NEAR_MARKET = "avg_num_of_orders_near_market"
    AVG_NUM_OF_ORDERS_FAR_MARKET = "avg_num_of_orders_far_market"

RESULTS_TO_LARK_MAPPINGS = {
    ResultsField.TOTAL_PNL.value: "累计盈亏",
    ResultsField.TODAY_PNL.value: "当日盈亏",
    ResultsField.TOTAL_NOTIONAL_VALUE_OF_EXECUTED_TRADES.value: "当日交易量",
    ResultsField.TOTAL_MARKET_TRADES_NOTIONAL.value: "当日市场交易量",
    ResultsField.EXEC_FREQ.value: "当日成交频率",
    ResultsField.AVG_MARKET_TRADES.value: "市场成交频率",
    ResultsField.LONGEST_NO_MARKET_TRADES.value: "市场最长无成交时间(秒)",
    ResultsField.AVG_BBO_SPREAD_ACCOUNT.value: "当日策略BBO Spread",
    ResultsField.AVG_BBO_SPREAD_MARKET.value: "市场BBO Spread",
    ResultsField.AVG_NUM_OF_ORDERS_NEAR.value: "当日策略平均流动性_订单数_千3",
    ResultsField.AVG_LIQUIDITY_NEAR_NOTIONAL.value: "当日策略平均流动性_订单Notional_千3",
    ResultsField.AVG_LIQUIDITY_NEAR_MARKET_NOTIONAL.value: "当日市场平均流动性_订单Notional_千3",
    ResultsField.AVG_NUM_OF_ORDERS_FAR.value: "当日策略平均流动性_订单数_百3",
    ResultsField.AVG_LIQUIDITY_FAR_NOTIONAL.value: "当日策略平均流动性_订单Notional_百3",
    ResultsField.AVG_LIQUIDITY_FAR_MARKET_NOTIONAL.value: "当日市场平均流动性_订单Notional_百3",
}

@dataclass
class Results:
    """A dataclass to hold the results of the trade analysis."""
    id: uuid = field(metadata={'sql_type': 'UUID', 'unique_key': True, 'primary_key': True})
    account_name: str = field(metadata={'sql_type': 'TEXT', 'foreign_key': True})
    strategy_name: str = field(metadata={'sql_type': 'TEXT', 'foreign_key': True})
    start_datetime_utc: str = field(metadata={'sql_type': 'TEXT'})
    date: str = field(metadata={'sql_type': 'TEXT'})
    global_symbol: str = field(metadata={'sql_type': 'TEXT', 'foreign_key': True})
    exchange_symbol: str = field(metadata={'sql_type': 'TEXT', 'foreign_key': True})
    exchange: str = field(metadata={'sql_type': 'TEXT', 'foreign_key': True})
    product_type: str = field(metadata={'sql_type': 'TEXT'})
    total_orders_sent: int = field(metadata={'sql_type': 'INTEGER'})
    total_executed_trades: int = field(metadata={'sql_type': 'INTEGER'})
    total_notional_value_of_executed_trades: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    avg_buy_executed_price: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    total_executed_buys_size: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    total_executed_buys_notional: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    avg_sell_executed_price: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    total_executed_sells_size: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    total_executed_sells_notional: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    exec_rate: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    exec_freq: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    today_pnl: str = field(metadata={'sql_type': 'TEXT'})
    total_pnl: str = field(metadata={'sql_type': 'TEXT'})
    avg_bbo_spread_account: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    avg_bbo_spread_market: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    avg_num_of_orders_near: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    avg_num_of_orders_far: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    avg_num_of_orders_near_market: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    avg_num_of_orders_far_market: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    avg_liquidity_near: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    avg_liquidity_near_market: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    avg_liquidity_far: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    avg_liquidity_far_market: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    avg_liquidity_near_notional: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    avg_liquidity_near_market_notional: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    avg_liquidity_far_notional: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    avg_liquidity_far_market_notional: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    avg_intervals_between_orders: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    avg_market_trades: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    total_market_trades_notional: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    longest_no_market_trades: float = field(metadata={'sql_type': 'DOUBLE PRECISION'})
    timestamp: int = field(default_factory=lambda: int(datetime.now().timestamp() * 1_000_000), metadata={'sql_type': 'BIGINT'})

    def __post_init__(self):
        pass

    def get_sql_keys_attr(self, sql_type: SqlType = SqlType.UNIQUE_KEY) -> list:
        """
        Identifies and returns the names of fields marked with a specific SQL key type.
        """
        keys = []
        for f in fields(self):
            if f.metadata.get(sql_type.value) is True:
                keys.append(f.name)
        return keys

    def to_dict(self) -> dict:
        """Converts the results to a dictionary."""
        return asdict(self)
    
    def to_db_dict(self) -> dict:
        """Converts the results to a dictionary for database insertion."""
        db_dict = self.to_dict()
        for key, value in db_dict.items():
            if isinstance(value, uuid.UUID):
                db_dict[key] = str(value)
        return db_dict

    def to_json(self, indent: int = 4) -> str:
        """Converts the results to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_df(self) -> pd.DataFrame:
        """Converts the results to a pandas DataFrame."""
        return pd.DataFrame([self.to_dict()])

    def to_lark_format(self, use_df: bool = False):
        """
        Formats the results for Lark, mapping attribute names to Chinese names.
        """
        data = self.to_dict()
        mapped_data = {RESULTS_TO_LARK_MAPPINGS.get(k, k): v for k, v in data.items()}
        
        if use_df:
            return pd.DataFrame([mapped_data])
        else:
            return mapped_data

    def __str__(self) -> str:
        """Provides a user-friendly, multi-line string representation of the results."""
        avg_interval_minutes = self.avg_intervals_between_orders / 1_000_000 / 60
        report = (
            f"## Trade Analysis from {self.start_datetime_utc} UTC onwards\n\n"
            f"**Execution Summary:**\n"
            f"* Execution Rate: {self.exec_rate:.2%}\n"
            f"* Execution Frequency: {self.exec_freq:,.2f} orders per minutes\n\n"
            f"* Total Orders Sent: {self.total_orders_sent}\n"
            f"* Total Executed Trades: {self.total_executed_trades}\n"
            f"* Total Notional Value of Executed Trades: {self.total_notional_value_of_executed_trades:,.4f} USDT\n"
            f"* Total Notional Value of Market Trades: {self.total_market_trades_notional:,.4f} USDT\n"
            f"* Avg Buy Price: {self.avg_buy_executed_price:,.4f}\n"
            f"* Total Buy Size: {self.total_executed_buys_size:,.4f}\n"
            f"* Total Buy Notional: {self.total_executed_buys_notional:,.4f}\n"
            f"* Avg Sell Price: {self.avg_sell_executed_price:,.4f}\n"
            f"* Total Sell Size: {self.total_executed_sells_size:,.4f}\n"
            f"* Total Sell Notional: {self.total_executed_sells_notional:,.4f}\n"
            f"* Avg Interval Between Orders: {avg_interval_minutes:,.2f} minutes\n"
            f"* Avg Market Trades per minutes: {self.avg_market_trades:,.2f}\n\n"
            f"**PnL:**\n"
            f"* Today's PnL: {self.today_pnl}\n"
            f"* Total PnL: {self.total_pnl}\n\n"
            f"**BBO Analysis:**\n"
            f"* Avg BBO Spread (Account): {self.avg_bbo_spread_account:,.4f}\n"
            f"* Avg BBO Spread (Market): {self.avg_bbo_spread_market:,.4f}\n\n"
            f"**Liquidity Analysis:**\n"
            f"* Avg Liquidity at 0.3% BBO (Strategy): {self.avg_liquidity_near:,.4f}\n"
            f"* Avg Liquidity at 0.3% BBO (Market):   {self.avg_liquidity_near_market:,.4f}\n"
            f"* Avg Liquidity at 3.0% BBO (Strategy): {self.avg_liquidity_far:,.4f}\n"
            f"* Avg Liquidity at 3.0% BBO (Market):   {self.avg_liquidity_far_market:,.4f}\n"
            f"* Avg Notional Liquidity at 0.3% BBO (Strategy): {self.avg_liquidity_near_notional:,.4f} USDT\n"
            f"* Avg Notional Liquidity at 0.3% BBO (Market):   {self.avg_liquidity_near_market_notional:,.4f} USDT\n"
            f"* Avg Notional Liquidity at 3.0% BBO (Strategy): {self.avg_liquidity_far_notional:,.4f} USDT\n"
            f"* Avg Notional Liquidity at 3.0% BBO (Market):   {self.avg_liquidity_far_market_notional:,.4f} USDT\n\n"
            f"* Longest No Market Trades: {self.longest_no_market_trades / 1_000_000 :,.2f} seconds\n\n"
            f"--- Methods Available ---\n"
            f" .to_dict(), .to_json(), .to_df()"
        )
        return report

    def to_msg_bot_msg(self) -> str:
        """Creates a nicely formatted string for sending analysis results to a message bot."""
        avg_interval_minutes = self.avg_intervals_between_orders / 1_000_000 / 60
        report = (
            f"## Trade Analysis for {self.strategy_name}@{self.account_name}=>{self.global_symbol} ({self.product_type}) on {self.exchange}\n"
            f"Analysis from **{self.start_datetime_utc} UTC** onwards.\n\n"
            f"**PnL:**\n"
            f"- **{RESULTS_TO_LARK_MAPPINGS[ResultsField.TODAY_PNL.value]}:** {self.today_pnl}\n"
            f"- **{RESULTS_TO_LARK_MAPPINGS[ResultsField.TOTAL_PNL.value]}:** {self.total_pnl}\n\n"
            f"**Execution Summary:**\n"
            f"- **{RESULTS_TO_LARK_MAPPINGS[ResultsField.TOTAL_NOTIONAL_VALUE_OF_EXECUTED_TRADES.value]}:** {self.total_notional_value_of_executed_trades:,.4f} USDT\n"
            f"- **{RESULTS_TO_LARK_MAPPINGS[ResultsField.TOTAL_MARKET_TRADES_NOTIONAL.value]}:** {self.total_market_trades_notional:,.4f} USDT\n"
            f"- **{RESULTS_TO_LARK_MAPPINGS[ResultsField.EXEC_FREQ.value]}:** {self.exec_freq:,.2f} orders per minutes\n"
            f"- **{RESULTS_TO_LARK_MAPPINGS[ResultsField.AVG_MARKET_TRADES.value]}:** {self.avg_market_trades:,.2f} per minutes\n\n"
            f"- **{RESULTS_TO_LARK_MAPPINGS[ResultsField.LONGEST_NO_MARKET_TRADES.value]}:** {self.longest_no_market_trades / 1_000_000 :,.2f} seconds\n\n"
            f"**BBO Analysis:**\n"
            f"- **{RESULTS_TO_LARK_MAPPINGS[ResultsField.AVG_BBO_SPREAD_ACCOUNT.value]}:** {self.avg_bbo_spread_account:,.4f}\n"
            f"- **{RESULTS_TO_LARK_MAPPINGS[ResultsField.AVG_BBO_SPREAD_MARKET.value]}:** {self.avg_bbo_spread_market:,.4f}\n\n"
            f"**Liquidity Analysis:**\n"
            f"- **{RESULTS_TO_LARK_MAPPINGS[ResultsField.AVG_NUM_OF_ORDERS_NEAR.value]}:** {self.avg_num_of_orders_near:,.4f}\n"
            f"- **{RESULTS_TO_LARK_MAPPINGS[ResultsField.AVG_LIQUIDITY_NEAR_NOTIONAL.value]}:** {self.avg_liquidity_near_notional:,.4f} USDT\n"
            f"- **{RESULTS_TO_LARK_MAPPINGS[ResultsField.AVG_LIQUIDITY_NEAR_MARKET_NOTIONAL.value]}:**   {self.avg_liquidity_near_market_notional:,.4f} USDT\n"
            f"- **{RESULTS_TO_LARK_MAPPINGS[ResultsField.AVG_NUM_OF_ORDERS_FAR.value]}:** {self.avg_num_of_orders_far:,.4f}\n"
            f"- **{RESULTS_TO_LARK_MAPPINGS[ResultsField.AVG_LIQUIDITY_FAR_NOTIONAL.value]}:** {self.avg_liquidity_far_notional:,.4f} USDT\n"
            f"- **{RESULTS_TO_LARK_MAPPINGS[ResultsField.AVG_LIQUIDITY_FAR_MARKET_NOTIONAL.value]}:**   {self.avg_liquidity_far_market_notional:,.4f} USDT\n"
        )
        return report

    def __repr__(self) -> str:
        """Provides a developer-friendly, compact string representation."""
        return (
            f"Results(exec_rate={self.exec_rate:.2%}, total_orders={self.total_orders_sent}, "
            f"total_executed={self.total_executed_trades})"
        )

def get_results_db_mapping():
    mapping = {f.name: f.metadata['sql_type'] for f in Results.__dataclass_fields__.values() if 'sql_type' in f.metadata}
    # Add fields that are not in Results dataclass but are in the DB table
    mapping['unique_key'] = 'VARCHAR(255)'
    mapping['timestamp'] = 'BIGINT'
    mapping['date'] = 'TEXT'
    mapping['account_name'] = 'VARCHAR(255)'
    mapping['strategy_name'] = 'VARCHAR(255)'
    mapping['process_name'] = 'VARCHAR(255)'
    return mapping

RESULTS_DB_COL_TYPE_MAPPING = get_results_db_mapping()