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

# Imports for database integration
from analysis.components.pull_db_data import PullDBData
from common.db_utils_pandas import DbUtils
from common.constants import ExchangeName
from order_management_service.Utils.orders_db_const import ORDERS_DB_COL_TYPE_MAPPING,OrderInfo

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

@dataclass
class Results:
    """A dataclass to hold the results of the trade analysis."""
    start_datetime_str: str
    earliest_trade_time: str
    last_trade_time: str
    global_symbol: str
    exchange_symbol: str
    exchange: str
    product_type: str
    first_trade_price: float
    last_trade_price: float
    pnl: float
    turnover: float
    executed_buys: int
    executed_sells: int
    num_buy_trades: int
    num_sell_trades: int
    total_buy_size: float
    total_sell_size: float
    avg_weighted_buy_price: float
    avg_weighted_sell_price: float
    max_long_position: float
    max_short_position: float
    largest_position: float
    position_type: str
    exec_rate: float

    def to_dict(self) -> dict:
        """Converts the results to a dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 4) -> str:
        """Converts the results to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_df(self) -> pd.DataFrame:
        """Converts the results to a pandas DataFrame."""
        return pd.DataFrame([self.to_dict()])

    def __str__(self) -> str:
        """Provides a user-friendly, multi-line string representation of the results."""
        report = (
            f"## Trade Analysis from {self.start_datetime_str} GMT+8 onwards\n\n"
            f"## Earliest Trade Time Found: {self.earliest_trade_time}\n\n"
            f"## Last Trade Time Found: {self.last_trade_time}\n\n"
            f"**Execution Rate:**\n"
            f"* Execution Rate: {self.exec_rate:.2%}\n\n"
            f"**Profit and Loss (PnL):**\n"
            f"* Simplified PnL: {self.pnl:,.4f} USDT\n"
            f"* Turnover: {self.turnover:,.4f} USDT\n\n"
            f"**Trade Counts:**\n"
            f"* Number of Buy Trades:  {self.num_buy_trades}\n"
            f"* Number of Sell Trades: {self.num_sell_trades}\n\n"
            f"**Volume Analysis:**\n"
            f"* Total Buy Size:  {self.total_buy_size:,.6f}\n"
            f"* Total Sell Size: {self.total_sell_size:,.6f}\n\n"
            f"**Price Analysis:**\n"
            f"* First Trade Price: {self.first_trade_price:,.4f} USDT\n"
            f"* Last Trade Price: {self.last_trade_price:,.4f} USDT\n"
            f"* Avg. Weighted Buy Price:  {self.avg_weighted_buy_price:,.4f} USDT\n"
            f"* Avg. Weighted Sell Price: {self.avg_weighted_sell_price:,.4f} USDT\n\n"
            f"**Position Analysis:**\n"
            f"* Largest Long Position Held:  {self.max_long_position:,.6f}\n"
            f"* Largest Short Position Held: {self.max_short_position:,.6f}\n"
            f"* Overall Largest Position:    {self.largest_position:,.6f} ({self.position_type})\n\n"
            f"--- Methods Available ---\n"
            f" .to_dict(), .to_json(), .to_df()"
        )
        return report

    def to_msg_bot_msg(self) -> str:
        """Creates a nicely formatted string for sending analysis results to a message bot."""
        report = (
            f"## Trade Analysis for {self.exchange_symbol} on {self.exchange}\n"
            f"Analysis from **{self.start_datetime_str} GMT+8** onwards.\n\n"
            f"**Period Analyzed:**\n"
            f"- Earliest Trade: {self.earliest_trade_time}\n"
            f"- Last Trade: {self.last_trade_time}\n\n"
            f"**Execution & PnL:**\n"
            f"- **Execution Rate:** {self.exec_rate:.2%}\n"
            f"- **Simplified PnL:** {self.pnl:,.4f} USDT\n"
            f"- **Turnover:** {self.turnover:,.4f} USDT\n\n"
            f"**Trade Summary:**\n"
            f"- **Buy Trades:** {self.num_buy_trades} (Total Size: {self.total_buy_size:,.6f})\n"
            f"- **Sell Trades:** {self.num_sell_trades} (Total Size: {self.total_sell_size:,.6f})\n\n"
            f"**Price Points:**\n"
            f"- **First Trade Price:** {self.first_trade_price:,.4f} USDT\n"
            f"- **Last Trade Price:** {self.last_trade_price:,.4f} USDT\n"
            f"- **Avg. Buy Price:** {self.avg_weighted_buy_price:,.4f} USDT\n"
            f"- **Avg. Sell Price:** {self.avg_weighted_sell_price:,.4f} USDT\n\n"
            f"**Position Highlights:**\n"
            f"- **Max Long Position:** {self.max_long_position:,.6f}\n"
            f"- **Max Short Position:** {self.max_short_position:,.6f}\n"
            f"- **Largest Position:** {self.largest_position:,.6f} ({self.position_type})\n"
        )
        return report

    def __repr__(self) -> str:
        """Provides a developer-friendly, compact string representation."""
        return (
            f"Results(pnl={self.pnl:.2f}, buys={self.num_buy_trades}, "
            f"sells={self.num_sell_trades}, largest_pos={self.largest_position:.4f})"
        )


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
                OrderInfo.CREATED_AT: TradeData.TIME.value,
                OrderInfo.SIDE: TradeData.ORDER_SIDE.value,
                OrderInfo.EXECUTED_SIZE: TradeData.QUANTITY.value,
                OrderInfo.SYMBOL: TradeData.SYMBOL.value,
                OrderInfo.PRICE: TradeData.PRICE.value
            }, inplace=True)
            
            for to_float_col in [TradeData.QUANTITY.value, TradeData.PRICE.value]:
                df[to_float_col] = df[to_float_col].astype(str)
                df[to_float_col] = df[to_float_col].str.replace(',', '').astype(float)
            
            df[TradeData.TIME.value] = df[TradeData.TIME.value].astype(int)
            df[TradeData.ORDER_SIDE.value] = df[TradeData.ORDER_SIDE.value].str.upper()
            df[TradeData.QUOTE_QTY.value] = df[TradeData.QUANTITY.value] * df[TradeData.PRICE.value]
            
            df[TradeData.FEE.value] = 0.0
            df[TradeData.FEE_CURRENCY.value] = 'USDT'
            df[TradeData.TAKER_MAKER.value] = 'taker'

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

    def run_analysis(self, df: pd.DataFrame, symbol: Optional[str] = None, timestamp: Optional[int] = None, exchange: Optional[ExchangeName] = None, rolling_period: int = 3600000000,
                    gen_chart:bool=False) -> Tuple[Optional[Results], Optional[pd.DataFrame]]:
        """
        Runs the full trade analysis and chart generation process on a given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame with trade data, prepared by load_df.
            symbol (str, optional): The symbol to filter by. Defaults to None.
            timestamp (int, optional): The start timestamp to filter by (in microseconds). Defaults to None.
            exchange (ExchangeName, optional): The exchange to filter by. Defaults to None.
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

            gmt8 = timezone(timedelta(hours=8))

            start_datetime_gmt8 = pd.to_datetime(timestamp, unit='us', utc=True).tz_convert(gmt8) if timestamp else self.filtered_trades[TradeData.DATETIME_GMT8.value].iloc[0]

            # --- Calculations ---
            exec_rate = 0.0
            if df is not None and not df.empty:
                num_total_orders = len(df)
                executed_sizes = pd.to_numeric(df[TradeData.QUANTITY.value].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
                num_executed_orders = (executed_sizes > 0).sum()
                if num_total_orders > 0:
                    exec_rate = num_executed_orders / num_total_orders

            buy_trades = self.filtered_trades[self.filtered_trades[TradeData.ORDER_SIDE.value] == 'BUY']
            sell_trades = self.filtered_trades[self.filtered_trades[TradeData.ORDER_SIDE.value] == 'SELL']
            
            executed_buys = (buy_trades[TradeData.QUANTITY.value] > 0).sum()
            executed_sells = (sell_trades[TradeData.QUANTITY.value] > 0).sum()
            
            total_buy_size = buy_trades[TradeData.QUANTITY.value].sum()
            total_sell_size = sell_trades[TradeData.QUANTITY.value].sum()
            total_buy_quote_qty = buy_trades[TradeData.QUOTE_QTY.value].sum()
            total_sell_quote_qty = sell_trades[TradeData.QUOTE_QTY.value].sum()
            avg_weighted_buy_price = total_buy_quote_qty / total_buy_size if total_buy_size > 0 else 0
            avg_weighted_sell_price = total_sell_quote_qty / total_sell_size if total_sell_size > 0 else 0
            pnl = (total_sell_quote_qty - (total_sell_size * avg_weighted_buy_price)) + \
                  ((total_buy_size * avg_weighted_sell_price) - total_buy_quote_qty)

            # --- Position Analysis ---
            self.filtered_trades[TradeData.SIGNED_QUANTITY.value] = np.where(self.filtered_trades[TradeData.ORDER_SIDE.value] == 'BUY', self.filtered_trades[TradeData.QUANTITY.value], -self.filtered_trades[TradeData.QUANTITY.value])
            self.filtered_trades[TradeData.NET_POSITION.value] = self.filtered_trades[TradeData.SIGNED_QUANTITY.value].cumsum()
            max_long_position = self.filtered_trades[TradeData.NET_POSITION.value].max()
            max_short_position = self.filtered_trades[TradeData.NET_POSITION.value].min()
            largest_position = max(max_long_position, abs(max_short_position))
            position_type = "Long" if largest_position == max_long_position else "Short"

            # --- Rolling Analysis ---
            if rolling_period:
                self.rolling_stats = RollingStats()
                start_time = self.filtered_trades[TradeData.TIME.value].iloc[0]
                end_time = self.filtered_trades[TradeData.TIME.value].iloc[-1]

                if df is not None and not df.empty:
                    df.sort_values(by=TradeData.TIME.value, inplace=True)

                current_time = start_time
                cumulative_pnl = 0
                while current_time < end_time:
                    period_end_time = current_time + rolling_period
                    period_df = self.filtered_trades[(self.filtered_trades[TradeData.TIME.value] >= current_time) & (self.filtered_trades[TradeData.TIME.value] < period_end_time)]

                    num_orders_sent_period = 0
                    period_orders_df = pd.DataFrame()
                    if df is not None and not df.empty:
                        period_orders_df = df[(df[TradeData.TIME.value] >= current_time) & (df[TradeData.TIME.value] < period_end_time)]
                        num_orders_sent_period = len(period_orders_df)

                    exec_rate_period = 0.0
                    if not period_orders_df.empty:
                        executed_sizes_period = pd.to_numeric(period_orders_df[TradeData.QUANTITY.value].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
                        num_executed_orders_period = (executed_sizes_period > 0).sum()
                        if num_orders_sent_period > 0:
                            exec_rate_period = num_executed_orders_period / num_orders_sent_period

                    if not period_df.empty:
                        buy_trades_period = period_df[period_df[TradeData.ORDER_SIDE.value] == 'BUY']
                        sell_trades_period = period_df[period_df[TradeData.ORDER_SIDE.value] == 'SELL']

                        turnover_value = self.turnover(buy_trades_period, sell_trades_period)

                        largest_position_period = period_df[TradeData.NET_POSITION.value].max()
                        smallest_position_period = period_df[TradeData.NET_POSITION.value].min()
                        cum_size = period_df[TradeData.NET_POSITION.value].iloc[-1]

                        buy_quote_qty_period = buy_trades_period[TradeData.QUOTE_QTY.value].sum()
                        sell_quote_qty_period = sell_trades_period[TradeData.QUOTE_QTY.value].sum()
                        pnl_period = sell_quote_qty_period - buy_quote_qty_period
                        cumulative_pnl += pnl_period

                        avg_weighted_buy_price_period = self._get_weight_price(buy_trades_period)
                        avg_weighted_sell_price_period = self._get_weight_price(sell_trades_period)
                        num_buy_trades_period = len(buy_trades_period)
                        num_sell_trades_period = len(sell_trades_period)

                        lowest_size = period_df[TradeData.QUANTITY.value].min()
                        largest_size = period_df[TradeData.QUANTITY.value].max()
                        avg_size = period_df[TradeData.QUANTITY.value].mean()

                        period_timestamp_str = pd.to_datetime(current_time, unit='us').strftime('%Y-%m-%d %H:%M:%S')

                        self.rolling_stats.add_row(
                            timestamp=period_timestamp_str,
                            turnover=turnover_value,
                            largest_position=largest_position_period,
                            smallest_position=smallest_position_period,
                            cum_size=cum_size,
                            PnL=pnl_period,
                            cumPnl=cumulative_pnl,
                            avg_weighted_buy_price=avg_weighted_buy_price_period,
                            avg_weighted_sell_price=avg_weighted_sell_price_period,
                            num_buy_trades=num_buy_trades_period,
                            num_sell_trades=num_sell_trades_period,
                            lowest_size=lowest_size,
                            largest_size=largest_size,
                            avg_size=avg_size,
                            exec_rate=exec_rate_period
                        )

                    current_time = period_end_time
            
            # --- Determine symbol and exchange for results ---
            exchange_val = exchange.value if exchange else (df['exchange'].iloc[0] if not df.empty and df['exchange'].nunique() == 1 else "Multiple")
            symbol_val = symbol if symbol else (df[TradeData.SYMBOL.value].iloc[0] if not df.empty and df[TradeData.SYMBOL.value].nunique() == 1 else "Multiple")

            # --- Store Results ---
            self.result = Results(
                start_datetime_str=start_datetime_gmt8.strftime('%Y-%m-%d %H:%M:%S'),
                earliest_trade_time=self.filtered_trades[TradeData.STR_TIME.value].iloc[0],
                last_trade_time=self.filtered_trades[TradeData.STR_TIME.value].iloc[-1],
                first_trade_price=self.filtered_trades[TradeData.PRICE.value].iloc[0],
                last_trade_price=self.filtered_trades[TradeData.PRICE.value].iloc[-1],
                pnl=pnl,
                turnover=turnover_value,
                executed_buys=executed_buys,
                executed_sells=executed_sells,
                num_buy_trades=len(buy_trades),
                num_sell_trades=len(sell_trades),
                total_buy_size=total_buy_size,
                total_sell_size=total_sell_size,
                avg_weighted_buy_price=avg_weighted_buy_price,
                avg_weighted_sell_price=avg_weighted_sell_price,
                max_long_position=max_long_position,
                max_short_position=max_short_position,
                largest_position=largest_position,
                position_type=position_type,
                exec_rate=exec_rate,
                global_symbol=symbol_val,
                exchange_symbol=symbol_val,
                exchange=exchange_val,
                product_type='spot' # Assuming spot, not available in DB data
            )

            # --- Chart Generation ---
            if gen_chart:
                self._generate_chart()
            
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

