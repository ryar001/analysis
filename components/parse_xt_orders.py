import pandas as pd
import numpy as np
import os


from common.Execution.order import OrderInfo,Side
from order_management_service.Utils.orders_db_const import ORDERS_DB_COL_TYPE_MAPPING
from order_management_service.Xt.xt_const import HistoricTradesDataInfo

# The keys of ORDERS_DB_COL_TYPE_MAPPING are the standard column names.
ORDER_COLUMNS = list(ORDERS_DB_COL_TYPE_MAPPING.keys())

def parse_xt_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses a DataFrame with XT exchange trade info to a standard orders format.

    The input DataFrame is expected to have columns from XT's spot trades CSV.
    Each row is a trade. If multiple trades exist for the same order ID, they
    are combined into a single record representing the filled order.

    Args:
        df (pd.DataFrame): DataFrame with XT trades data.

    Returns:
        pd.DataFrame: A DataFrame with standardized columns and unique order IDs.
    """
    parsed_df = pd.DataFrame()

    # --- Column Mapping and Transformations ---
    # Note: The input is trade data, so we are creating order-like records from trades.
    
    # Map from trade data to order fields
    parsed_df[OrderInfo.ORDER_ID] = df[HistoricTradesDataInfo.ORDER_ID]
    parsed_df[OrderInfo.SYMBOL] = df[HistoricTradesDataInfo.SYMBOL]
    parsed_df[OrderInfo.SIDE] = np.where(df[HistoricTradesDataInfo.ORDER_SIDE]=="BUY", Side.BUY, Side.SELL) 
    parsed_df[OrderInfo.CREATED_AT] = df[HistoricTradesDataInfo.TIME] * 10**3
    parsed_df[OrderInfo.PRICE] = df[HistoricTradesDataInfo.PRICE] # This is execution price, used as order price.
    parsed_df[OrderInfo.EXECUTED_SIZE] = df[HistoricTradesDataInfo.QUANTITY]
    
    # --- Inferred or Calculated Columns ---
    parsed_df[OrderInfo.AVG_PRICE] = df[HistoricTradesDataInfo.PRICE]  # For a single trade, avg_price is the trade price.
    parsed_df[OrderInfo.EXCHANGE] = 'xt'
    parsed_df[OrderInfo.STATE] = 'filled' # Trades are by definition filled.
    parsed_df[OrderInfo.ORIGINAL_SIZE] = df[HistoricTradesDataInfo.QUANTITY] # Assuming trade is a whole order.

    # --- Combine trades for the same order ID ---
    # XT trade history can contain multiple trades for a single order.
    # We group by order_id and aggregate the trades to represent a single order.
    if parsed_df[OrderInfo.ORDER_ID].duplicated().any():
        # For calculating weighted average price
        parsed_df['total_value'] = parsed_df[OrderInfo.PRICE] * parsed_df[OrderInfo.EXECUTED_SIZE]

        agg_spec = {
            OrderInfo.EXECUTED_SIZE: 'sum',
            OrderInfo.ORIGINAL_SIZE: 'sum',
            'total_value': 'sum',
            OrderInfo.CREATED_AT: 'min',
            OrderInfo.SYMBOL: 'first',
            OrderInfo.SIDE: 'first',
            OrderInfo.EXCHANGE: 'first',
            OrderInfo.STATE: 'first',
        }

        parsed_df = parsed_df.groupby(OrderInfo.ORDER_ID).agg(agg_spec).reset_index()

        # Calculate the weighted average price for the order
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_price = parsed_df['total_value'] / parsed_df[OrderInfo.EXECUTED_SIZE]
        
        parsed_df[OrderInfo.AVG_PRICE] = np.nan_to_num(avg_price)
        parsed_df[OrderInfo.PRICE] = parsed_df[OrderInfo.AVG_PRICE]

        parsed_df = parsed_df.drop(columns=['total_value'])

    # --- Add Missing Standard Columns ---
    # Add other standard columns, not present in the trade data, with null values.
    for col in ORDER_COLUMNS:
        if col not in parsed_df.columns:
            parsed_df[col] = np.nan

    # --- Data Types and Column Order ---
    # Ensure correct data types for numeric columns that might be 'object' due to NaNs
    numeric_cols = [
        OrderInfo.PRICE, OrderInfo.AVG_PRICE, OrderInfo.ORIGINAL_SIZE, OrderInfo.REMAINING_SIZE, OrderInfo.EXECUTED_SIZE,
        OrderInfo.REQUEST_RESPONSE_TIME, OrderInfo.GET_ORDER_RESPONSE_TIME
    ]
    for col in numeric_cols:
        if col in parsed_df.columns:
            parsed_df[col] = pd.to_numeric(parsed_df[col], errors='coerce')

    # Return dataframe with a consistent column order
    return parsed_df[ORDER_COLUMNS]

if __name__ == '__main__':
    # This block is for testing the parsing function.
    # It reads a sample CSV of XT trades, parses it, and prints the result.
    breakpoint()
    try:
        # Construct the absolute path to the sample data file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, '..', '..', 'z_downloads', 'spot_trades_eth_usdt.csv')

        if not os.path.exists(file_path):
            print(f"Error: Test file not found at '{file_path}'")
        else:
            print(f"Loading test data from '{file_path}'")
            sample_df = pd.read_csv(file_path)
            
            print("\n--- Original DataFrame Head ---")
            print(sample_df.head())
            
            # Parse the dataframe
            parsed_df = parse_xt_orders(sample_df)
            
            print("\n--- Parsed DataFrame Head ---")
            print(parsed_df.iloc[0])
            breakpoint()
            
            print("\n--- Parsed DataFrame Info ---")
            parsed_df.info()

    except Exception as e:
        print(f"\nAn error occurred during the test run: {e}")