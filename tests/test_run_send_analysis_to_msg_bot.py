
import asyncio
import json
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
import sys

# Add project root to path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from analysis.run_send_analysis_to_msg_bot import main, STRATEGY_STATS_DB_COL_TYPE_MAPPING
from analysis.trades_analysis import Results
from common.db_utils_pandas import DbUtils

# Helper function to get DB info for integration tests
def get_db_info():
    db_json_path = os.path.join(os.path.dirname(__file__), '..', '..', 'common', 'db.json')
    if not os.path.exists(db_json_path):
        pytest.skip(f"DB config not found at {db_json_path}, skipping integration test.")
    with open(db_json_path) as f:
        return json.load(f)['DEFAULT']['Trading']

@pytest.fixture(scope="function")
def db_integration_fixture():
    """Creates and tears down a test table for strategy stats integration testing."""
    table_name = "test_strategy_stats"
    db_info = get_db_info()
    
    # This DbUtils instance is for setup and teardown.
    setup_util = DbUtils(
        db_user=db_info['user'], db_password=db_info['password'],
        db_host=db_info['host'], db_port=db_info['port'],
        db_name=db_info.get('dbname', 'Trading'), table_name=table_name
    )
    setup_util.start()

    try:
        # Assuming DbUtils has a method to execute raw SQL.
        # Drop table if it exists from a previous failed run.
        setup_util.execute_sql(f'DROP TABLE IF EXISTS "{table_name}";')
        
        # Create the test table based on the schema from the main script.
        cols = [f'"{col_name}" {col_type}' for col_name, col_type in STRATEGY_STATS_DB_COL_TYPE_MAPPING.items()]
        create_sql = f'CREATE TABLE "{table_name}" ({", ".join(cols)});'
        setup_util.execute_sql(create_sql)
        
        yield table_name
        
    finally:
        # Teardown: Drop the test table.
        if setup_util.conn:
            setup_util.execute_sql(f'DROP TABLE IF EXISTS "{table_name}";')
        setup_util.close()

@pytest.mark.asyncio
@patch('analysis.run_send_analysis_to_msg_bot.load_strategy_helper_async')
@patch('analysis.run_send_analysis_to_msg_bot.MsgBot')
@patch('analysis.run_send_analysis_to_msg_bot.TradesAnalysis')
@patch('analysis.run_send_analysis_to_msg_bot.PullDBData')
@patch('analysis.run_send_analysis_to_msg_bot.DbUtils')
@patch('builtins.open')
async def test_main_stores_strategy_stats(
    mock_open_builtin,
    mock_db_utils_class,
    mock_pull_db_data_class,
    mock_trades_analysis_class,
    mock_msg_bot_class,
    mock_load_helper
):
    """
    Test that the main function correctly calls store_strategy_stats with the right data.
    """
    settings_content = """
private:
  table_name: strategy_stats
  accounts:
    TestAccount:
      strategy_and_process_name:
        - strategy_name: TestStrategy
          process_name: TestProcess
          global_symbol: BTC/USDT
          exchange: XT_FX
  msg_bot_settings:
    bot_name: test_bot
    chat_id: 'id'
    api_name: 'api'
    url: 'url'
    title: 'title'
"""
    db_json_content = json.dumps({
        "DEFAULT": { "Trading": { "user": "test", "password": "test", "host": "localhost", "port": "5432", "dbname": "Trading" } }
    })

    mock_open_builtin.side_effect = [
        mock_open(read_data=settings_content).return_value, # for settings.yaml
        mock_open().return_value,                           # for the log file
        mock_open(read_data=db_json_content).return_value  # for db.json
    ]    # Configure mocks
    mock_orders_db = MagicMock()
    mock_stats_db = MagicMock()
    
    # This mock ensures that when DbUtils is called for the stats table, our mock is used.
    def db_utils_side_effect(*args, **kwargs):
        if kwargs.get('table_name') == 'strategy_stats':
            return mock_stats_db
        else:
            return mock_orders_db
    
    mock_db_utils_class.side_effect = db_utils_side_effect

    mock_pull_db_data_class.return_value.get_orders.return_value = pd.DataFrame({
        'strategy_order_id': ['1'], 'created_at': [pd.to_datetime('now', utc=True)], 'symbol': ['s'], 
        'side': ['buy'], 'price': [1], 'quantity': [1], 'fee': [0], 'exchange': ['e']
    })
    
    analysis_result = Results(
        start_datetime_utc='2023-01-01 00:00:00',
        earliest_trade_time='2023-01-01 10:00:00',
        last_trade_time='2023-01-01 10:05:00',
        global_symbol='BTC/USDT',
        exchange_symbol='BTC/USDT',
        exchange='XT_FX',
        product_type='spot',
        first_trade_price=20000.0,
        last_trade_price=20100.0,
        pnl=100.0,
        turnover=40100.0,
        executed_buys=1,
        executed_sells=1,
        num_buy_trades=1,
        num_sell_trades=1,
        total_buy_size=1.0,
        total_sell_size=1.0,
        avg_weighted_buy_price=20000.0,
        avg_weighted_sell_price=20100.0,
        max_long_position=1.0,
        max_short_position=0.0,
        largest_position=1.0,
        position_type='Long',
        exec_rate=1.0
    )
    mock_trades_analysis_class.return_value.run_analysis.return_value = (analysis_result, pd.DataFrame())
    mock_trades_analysis_class.return_value.parse_df.side_effect = lambda df: df

    # Run main
    await main()

    # Assertions
    mock_stats_db.start.assert_called_once()
    mock_orders_db.start.assert_called_once()

    # We expect insert_order to be called on our mock stats db.
    mock_stats_db.insert_order.assert_called_once()
    
    inserted_data = mock_stats_db.insert_order.call_args[1]['order_dict']
    assert inserted_data['account_name'] == 'TestAccount'
    assert inserted_data['strategy_name'] == 'TestStrategy'
    assert inserted_data['process_name'] == 'TestProcess'
    assert inserted_data['pnl'] == 100.0
    assert inserted_data['turnover'] == 40100.0
    assert inserted_data['num_buy_trades'] == 1
    assert 'timestamp' in inserted_data

    mock_stats_db.close.assert_called_once()
    mock_orders_db.close.assert_called_once()
    
    # Check that a message was sent
    mock_msg_bot_class.return_value.send_msg.assert_called_once()

@pytest.mark.integration
@pytest.mark.asyncio
@patch('analysis.run_send_analysis_to_msg_bot.load_strategy_helper_async')
@patch('analysis.run_send_analysis_to_msg_bot.MsgBot')
@patch('analysis.run_send_analysis_to_msg_bot.TradesAnalysis')
@patch('analysis.run_send_analysis_to_msg_bot.PullDBData')
@patch('builtins.open')
async def test_main_inserts_data_into_db(
    mock_open_builtin,
    mock_pull_db_data_class,
    mock_trades_analysis_class,
    mock_msg_bot_class,
    mock_load_helper,
    db_integration_fixture
):
    """
    Integration test: Verifies that main() correctly inserts analysis results into a live database.
    """
    test_table_name = db_integration_fixture
    
    # --- Arrange ---
    settings_content = f"""
private:
  table_name: {test_table_name}
  accounts:
    IntegrationTestAccount:
      strategy_and_process_name:
        - strategy_name: IntegrationStrategy
          process_name: IntegrationProcess
          global_symbol: ETH/USDT
          exchange: XT_FX
  msg_bot_settings: {{ bot_name: test, chat_id: 'id', api_name: 'api', url: 'url', title: 'title' }}
"""
    db_json_path = os.path.join(os.path.dirname(__file__), '..', '..', 'common', 'db.json')
    with open(db_json_path, 'r') as f:
        db_json_content = f.read()

    mock_open_builtin.side_effect = [
        mock_open(read_data=settings_content).return_value,
        mock_open().return_value, # logger
        mock_open(read_data=db_json_content).return_value
    ]

    mock_pull_db_data_class.return_value.get_orders.return_value = pd.DataFrame({
        'strategy_order_id': ['1'], 'created_at': [pd.to_datetime('now', utc=True)], 'symbol': ['s'], 
        'side': ['buy'], 'price': [100], 'quantity': [2], 'fee': [0.1], 'exchange': ['e']
    })
    
    analysis_result = Results(
        pnl=50.0, turnover=3050.0, total_buy_size=2.0,
        start_datetime_utc='2023-02-01 00:00:00', earliest_trade_time='2023-02-01 10:00:00',
        last_trade_time='2023-02-01 10:05:00', global_symbol='ETH/USDT', exchange_symbol='ETH/USDT',
        exchange='XT_FX', product_type='spot', first_trade_price=1500.0, last_trade_price=1550.0,
        executed_buys=1, executed_sells=0, num_buy_trades=1, num_sell_trades=0,
        total_sell_size=0.0, avg_weighted_buy_price=1525.0, avg_weighted_sell_price=0.0,
        max_long_position=2.0, max_short_position=0.0, largest_position=2.0,
        position_type='Long', exec_rate=1.0
    )
    mock_trades_analysis_class.return_value.run_analysis.return_value = (analysis_result, pd.DataFrame())
    mock_trades_analysis_class.return_value.parse_df.side_effect = lambda df: df

    # --- Act ---
    await main()

    # --- Assert ---
    db_info = get_db_info()
    verify_util = DbUtils(
        db_user=db_info['user'], db_password=db_info['password'],
        db_host=db_info['host'], db_port=db_info['port'],
        db_name=db_info.get('dbname', 'Trading'), table_name=test_table_name
    )
    verify_util.start()
    try:
        inserted_df = verify_util.get_table_as_df(where_clause="strategy_name = 'IntegrationStrategy'")
        assert not inserted_df.empty, "No data was inserted into the test database."
        assert len(inserted_df) == 1, "More than one row was inserted for the test strategy."
        
        inserted_row = inserted_df.iloc[0]
        assert inserted_row['account_name'] == 'IntegrationTestAccount'
        assert inserted_row['pnl'] == pytest.approx(50.0)
        assert inserted_row['global_symbol'] == 'ETH/USDT'
    finally:
        verify_util.close()


@pytest.mark.integration
@pytest.mark.asyncio
@patch('analysis.run_send_analysis_to_msg_bot.load_strategy_helper_async')
@patch('analysis.run_send_analysis_to_msg_bot.MsgBot')
@patch('analysis.run_send_analysis_to_msg_bot.TradesAnalysis')
@patch('analysis.run_send_analysis_to_msg_bot.PullDBData')
@patch('builtins.open')
async def test_main_inserts_data_into_db(
    mock_open_builtin,
    mock_pull_db_data_class,
    mock_trades_analysis_class,
    mock_msg_bot_class,
    mock_load_helper,
    db_integration_fixture
):
    """
    Integration test: Verifies that main() correctly inserts analysis results into a live database.
    """
    test_table_name = db_integration_fixture
    
    # --- Arrange ---
    settings_content = f"""
private:
  table_name: {test_table_name}
  accounts:
    IntegrationTestAccount:
      strategy_and_process_name:
        - strategy_name: IntegrationStrategy
          process_name: IntegrationProcess
          global_symbol: ETH/USDT
          exchange: XT_FX
  msg_bot_settings: {{ bot_name: test, chat_id: 'id', api_name: 'api', url: 'url', title: 'title' }}
"""
    db_json_path = os.path.join(os.path.dirname(__file__), '..', '..', 'common', 'db.json')
    with open(db_json_path, 'r') as f:
        db_json_content = f.read()

    mock_open_builtin.side_effect = [
        mock_open(read_data=settings_content).return_value,
        mock_open(read_data=db_json_content).return_value,
        mock_open().return_value # logger
    ]

    mock_pull_db_data_class.return_value.get_orders.return_value = pd.DataFrame({
        'strategy_order_id': ['1'], 'created_at': [pd.to_datetime('now', utc=True)], 'symbol': ['s'], 
        'side': ['buy'], 'price': [100], 'quantity': [2], 'fee': [0.1], 'exchange': ['e']
    })
    
    analysis_result = Results(
        pnl=50.0, turnover=3050.0, total_buy_size=2.0,
        start_datetime_utc='2023-02-01 00:00:00', earliest_trade_time='2023-02-01 10:00:00',
        last_trade_time='2023-02-01 10:05:00', global_symbol='ETH/USDT', exchange_symbol='ETH/USDT',
        exchange='XT_FX', product_type='spot', first_trade_price=1500.0, last_trade_price=1550.0,
        executed_buys=1, executed_sells=0, num_buy_trades=1, num_sell_trades=0,
        total_sell_size=0.0, avg_weighted_buy_price=1525.0, avg_weighted_sell_price=0.0,
        max_long_position=2.0, max_short_position=0.0, largest_position=2.0,
        position_type='Long', exec_rate=1.0
    )
    mock_trades_analysis_class.return_value.run_analysis.return_value = (analysis_result, pd.DataFrame())
    mock_trades_analysis_class.return_value.parse_df.side_effect = lambda df: df

    # --- Act ---
    await main()

    # --- Assert ---
    db_info = get_db_info()
    verify_util = DbUtils(
        db_user=db_info['user'], db_password=db_info['password'],
        db_host=db_info['host'], db_port=db_info['port'],
        db_name=db_info.get('dbname', 'Trading'), table_name=test_table_name
    )
    verify_util.start()
    try:
        inserted_df = verify_util.get_table_as_df(where_clause="strategy_name = 'IntegrationStrategy'")
        assert not inserted_df.empty, "No data was inserted into the test database."
        assert len(inserted_df) == 1, "More than one row was inserted for the test strategy."
        
        inserted_row = inserted_df.iloc[0]
        assert inserted_row['account_name'] == 'IntegrationTestAccount'
        assert inserted_row['pnl'] == pytest.approx(50.0)
        assert inserted_row['global_symbol'] == 'ETH/USDT'
    finally:
        verify_util.close()
