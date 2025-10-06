import pandas as pd
from decimal import Decimal, InvalidOperation
from common.Execution.order import OrderInfo

def compare_orders(base_orders_df: pd.DataFrame, compare_orders_df: pd.DataFrame) -> dict:
    """
    Compares two DataFrames of orders and identifies discrepancies.

    The function compares orders based on their ID and checks for equality in
    symbol, price, executed size, and side.

    Args:
        base_orders_df (pd.DataFrame): The base DataFrame to compare from.
        compare_orders_df (pd.DataFrame): The DataFrame to compare against.

    Returns:
        dict: A dictionary of orders that failed comparison. The key is the
              order_id, and the value is a dict containing the 'base' and
              'compare' order rows. If an order from the base DataFrame is
              not found in the compare DataFrame, the 'compare' value will be None.
    """
    compare_fail_dict = {}
    breakpoint()
    # Ensure the index of compare_orders_df is the order ID for efficient lookups.
    if compare_orders_df.index.name != OrderInfo.ORDER_ID:
        compare_orders_indexed_df = compare_orders_df.set_index(OrderInfo.ORDER_ID)
    else:
        compare_orders_indexed_df = compare_orders_df

    for base_row in base_orders_df.itertuples():
        order_id = getattr(base_row, OrderInfo.ORDER_ID)
        is_mismatch = False
        
        try:
            # Find the corresponding order in the compare DataFrame
            compare_row = compare_orders_indexed_df.loc[order_id]
        except KeyError:
            # Order not found in the compare DataFrame, record as a failure
            compare_fail_dict[order_id] = {
                "base": base_row._asdict(),
                "compare": None
            }
            continue

        # --- Perform Comparisons ---

        # 1. Compare Symbol (string comparison)
        if getattr(base_row, OrderInfo.SYMBOL) != compare_row[OrderInfo.SYMBOL]:
            is_mismatch = True
            breakpoint()
        # 2. Compare Side (string comparison)
        if str(getattr(base_row, OrderInfo.SIDE).value) != str(compare_row[OrderInfo.SIDE]):
            is_mismatch = True
            breakpoint()

        # 3. Compare Price (Decimal comparison for precision)
        try:
            base_price = Decimal(str(getattr(base_row, OrderInfo.PRICE)))
            compare_price = Decimal(str(compare_row[OrderInfo.PRICE]))
            if base_price != compare_price:
                is_mismatch = True
                breakpoint()
        except (InvalidOperation, ValueError):
            # Fallback to string comparison if values are not valid for Decimal (e.g., NaN)
            if str(getattr(base_row, OrderInfo.PRICE)) != str(compare_row[OrderInfo.PRICE]):
                is_mismatch = True
                breakpoint()

        # 4. Compare Executed Size (Decimal comparison for precision)
        try:
            base_size = Decimal(str(getattr(base_row, OrderInfo.EXECUTED_SIZE)))
            compare_size = Decimal(str(compare_row[OrderInfo.EXECUTED_SIZE]))
            if base_size != compare_size:
                is_mismatch = True
                breakpoint()
        except (InvalidOperation, ValueError):
            # Fallback to string comparison if values are not valid for Decimal
            if str(getattr(base_row, OrderInfo.EXECUTED_SIZE)) != str(compare_row[OrderInfo.EXECUTED_SIZE]):
                is_mismatch = True
                

        # If any field mismatched, record the failure
        if is_mismatch:
            compare_fail_dict[order_id] = {
                "base": base_row._asdict(),
                "compare": compare_row.to_dict()
            }

    return compare_fail_dict

if __name__ == '__main__':
    import os
    import json
    from parse_xt_orders import parse_xt_orders

    # Define file paths relative to the script location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_csv_path = os.path.join(current_dir, 'spot_trades_trump_usdt.csv')
    compare_csv_path = os.path.join(current_dir, 'trump_usdt_240925__17_28.csv')

    # Load dataframes
    try:
        base_df_raw = pd.read_csv(base_csv_path)
        compare_df_raw = pd.read_csv(compare_csv_path)
    except FileNotFoundError as e:
        print(f"Error loading CSV file: {e}")
        print("Please ensure both 'spot_trades_trump_usdt.csv' and '../../trump_usdt_240925__17_28.csv' exist.")
        exit()

    # --- Pre-process base_df (from spot_trades) ---
    base_df = parse_xt_orders(base_df_raw)

    # --- Pre-process compare_df (from orders_history) ---
    # Rename columns to match OrderInfo constants, if they are not already.
    # The provided file seems to use snake_case, which matches OrderInfo.
    compare_df = compare_df_raw.rename(columns={
        'order_id': OrderInfo.ORDER_ID,
        'symbol': OrderInfo.SYMBOL,
        'side': OrderInfo.SIDE,
        'price': OrderInfo.PRICE,
        'executed_size': OrderInfo.EXECUTED_SIZE
    })

    # Run the comparison
    failed_orders = compare_orders(base_df, compare_df)

    # Print the results
    print("--- Starting Order Comparison Test ---")
    if failed_orders:
        print(f"\n--- Comparison Failed for {len(failed_orders)} orders: ---")
        # Using json.dumps for pretty printing the dictionaries
        print(json.dumps(failed_orders, indent=4))
    else:
        print("\n--- All orders matched successfully! ---")
    
    print("\n--- Order Comparison Test Finished ---")
