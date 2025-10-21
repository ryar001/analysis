import pandas as pd
from common.db_utils_pandas import DbUtils
from typing import Optional
from enum import Enum
from common.constants import ExchangeName


class PullDBData:
    def __init__(self, db_utils: DbUtils):
        """
        Initializes the PullDBData class.

        Args:
            db_utils (DbUtils): An instance of DbUtils.
        """
        self.db_utils = db_utils

    def get_orders(self,timestamp: Optional[int] = None,to_timestamp: Optional[int] = None,limit:Optional[int]=1000000,**kwargs) -> pd.DataFrame:
        """
        Gets orders from the database.

        Args:
            global_symbol (str, optional): The symbol to filter by. Defaults to None.
            timestamp (int, optional): The timestamp to filter by (in milliseconds). Defaults to None.
            to_timestamp (int, optional): The end timestamp to filter by (in milliseconds). Defaults to None.
            exchange (ExchangeName, optional): The exchange to filter by. Defaults to None.
            strategy_name (str, optional): The strategy name to filter by. Defaults to None.
            strategy_process_name (str, optional): The strategy process name to filter by. Defaults to None.
            limit (int, optional): The maximum number of rows to return. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the orders.
        """
        where_clauses = []
        params = {}
        self.db_utils.last_update_time_key = self.db_utils.last_update_time_key or 'timestamp'
        for key, value in kwargs.items():
            if value is not None and key in self.db_utils.db_col_type_mapping:
                where_clauses.append(f'"{key}" = :{key}')
                params[key] = value

        #filter by timestamp
        if timestamp is not None:
            where_clauses.append(f'"{self.db_utils.last_update_time_key}" >= :timestamp')
            params[self.db_utils.last_update_time_key] = timestamp
        if to_timestamp is not None:
            where_clauses.append(f'"{self.db_utils.last_update_time_key}" < :to_timestamp')
            params['to_timestamp'] = to_timestamp
        
        where_clause = " AND ".join(where_clauses) if where_clauses else None
        original_time_key = self.db_utils.last_update_time_key
        try:
            
            df = self.db_utils.get_table_as_df(where_clause=where_clause, params=params, limit=limit)
            # df = df[df[self.db_utils.last_update_time_key] >= timestamp]
            # df = df[df[self.db_utils.last_update_time_key] < to_timestamp]
        finally:
            self.db_utils.last_update_time_key = original_time_key
            
        return df
    

if __name__ == '__main__':
    import json
    import os

    def test_pull_db_data():
        try:
            # Assumes db.json is in the common directory, and we are running from asset_management root
            db_json_path = os.path.join(os.path.dirname(__file__), '..', '..', 'common', 'db.json')
            with open(db_json_path) as f:
                db_info = json.load(f)['DEFAULT']['Trading']
        except (FileNotFoundError, KeyError):
            print("Skipping test: db.json not found or configured incorrectly.")
            return

        test_table_name = "test_pull_db_data_orders"
        table_name = "orders_history_prod"
        dbu = DbUtils(
            db_user=db_info['user'],
            db_password=db_info['password'],
            db_host=db_info['host'],
            db_port=db_info['port'],
            db_name=db_info.get('dbname', 'Trading'), # Use dbname from json if present, else default
            table_name=table_name,
            primary_key="order_id",
            unique_pri_key=True,
            db_col_type_mapping={
                "strategy_order_id": "VARCHAR(255)",
                "exchange": "VARCHAR(255)",
                "symbol": "VARCHAR(255)",
                "order_id": "VARCHAR(255)",
                "strategy_connection_id": "VARCHAR(255)",
                "strategy_session_id": "VARCHAR(255)",
                "strategy_name": "VARCHAR(255)",
                "strategy_process_name": "VARCHAR(255)",
                "createdAt": "BIGINT",
                "closedAt": "BIGINT",
                "update_time": "BIGINT",
                "presend_msg_at": "BIGINT",
                "ws_ack_at": "BIGINT",
                "request_response_time": "FLOAT",
                "get_order_response_time": "FLOAT",
                "side": "VARCHAR(255)",
                "price": "FLOAT",
                "avg_price": "FLOAT",
                "original_size": "FLOAT",
                "remaining_size": "FLOAT",
                "executed_size": "FLOAT",
                "state": "VARCHAR(255)",
            }
        )
        dbu.start()
        # Clean up and prepare table
        # dbu.delete_table()
        dbu.init_tables()
        puller = PullDBData(dbu)
        df = puller.get_orders(global_symbol="FX-TRUMP/USDT")
        breakpoint()
        return
        # Test Data
        orders_data = [
            {'order_id': '1', 'symbol': 'FX-BTC/USDT', 'exchange': 'XT_FX', 'createdAt': 100, 'strategy_order_id': 's1', 'strategy_connection_id': 'sc1', 'strategy_session_id': 'ss1', 'strategy_name': 'sn1', 'strategy_process_name': 'spn1', 'closedAt': 0, 'update_time': 100, 'presend_msg_at': 99, 'ws_ack_at': 101, 'request_response_time': 2.0, 'get_order_response_time': 3.0, 'side': 'buy', 'price': 50000.0, 'avg_price': 50000.0, 'original_size': 1.0, 'remaining_size': 0.0, 'executed_size': 1.0, 'state': 'filled'},
            {'order_id': '2', 'symbol': 'FX-ETH/USDT', 'exchange': 'XT_FX', 'createdAt': 200, 'strategy_order_id': 's2', 'strategy_connection_id': 'sc1', 'strategy_session_id': 'ss1', 'strategy_name': 'sn1', 'strategy_process_name': 'spn1', 'closedAt': 0, 'update_time': 200, 'presend_msg_at': 199, 'ws_ack_at': 201, 'request_response_time': 2.0, 'get_order_response_time': 3.0, 'side': 'sell', 'price': 3000.0, 'avg_price': 3000.0, 'original_size': 10.0, 'remaining_size': 0.0, 'executed_size': 10.0, 'state': 'filled'},
            {'order_id': '3', 'symbol': 'FX-BTC/USDT', 'exchange': 'BINANCE', 'createdAt': 300, 'strategy_order_id': 's3', 'strategy_connection_id': 'sc2', 'strategy_session_id': 'ss2', 'strategy_name': 'sn2', 'strategy_process_name': 'spn2', 'closedAt': 0, 'update_time': 300, 'presend_msg_at': 299, 'ws_ack_at': 301, 'request_response_time': 2.0, 'get_order_response_time': 3.0, 'side': 'buy', 'price': 50001.0, 'avg_price': 50001.0, 'original_size': 2.0, 'remaining_size': 0.0, 'executed_size': 2.0, 'state': 'filled'},
            {'order_id': '4', 'symbol': 'FX-ETH/USDT', 'exchange': 'BINANCE', 'createdAt': 400, 'strategy_order_id': 's4', 'strategy_connection_id': 'sc2', 'strategy_session_id': 'ss2', 'strategy_name': 'sn2', 'strategy_process_name': 'spn2', 'closedAt': 0, 'update_time': 400, 'presend_msg_at': 399, 'ws_ack_at': 401, 'request_response_time': 2.0, 'get_order_response_time': 3.0, 'side': 'sell', 'price': 3001.0, 'avg_price': 3001.0, 'original_size': 5.0, 'remaining_size': 0.0, 'executed_size': 5.0, 'state': 'filled'},
            {'order_id': '5', 'symbol': 'FX-BTC/USDT', 'exchange': 'XT_FX', 'createdAt': 500, 'strategy_order_id': 's5', 'strategy_connection_id': 'sc1', 'strategy_session_id': 'ss1', 'strategy_name': 'sn1', 'strategy_process_name': 'spn1', 'closedAt': 0, 'update_time': 500, 'presend_msg_at': 499, 'ws_ack_at': 501, 'request_response_time': 2.0, 'get_order_response_time': 3.0, 'side': 'buy', 'price': 50002.0, 'avg_price': 50002.0, 'original_size': 1.0, 'remaining_size': 1.0, 'executed_size': 0.0, 'state': 'open'}
        ]
        dbu.bulk_insert(orders_data)

        puller = PullDBData(dbu)

        print("\n--- Testing get_orders ---")

        # 1. No filters
        df_all = puller.get_orders()
        print(f"1. No filters: Fetched {len(df_all)} records")
        assert len(df_all) == 5
        assert sorted(df_all['order_id'].tolist()) == ['1', '2', '3', '4', '5']

        # 2. Filter by symbol (FX-BTC/USDT)
        df_btc = puller.get_orders(symbol='FX-BTC/USDT')
        print(f"2. Filter by symbol='BTC-USDT': Fetched {len(df_btc)} records")
        assert len(df_btc) == 3
        assert sorted(df_btc['order_id'].tolist()) == ['1', '3', '5']

        # 3. Filter by exchange
        df_xt = puller.get_orders(exchange=ExchangeName.XT_FX)
        print(f"3. Filter by exchange='XT': Fetched {len(df_xt)} records")
        assert len(df_xt) == 3
        assert sorted(df_xt['order_id'].tolist()) == ['1', '2', '5']

        # 4. Filter by timestamp (from)
        df_from_ts = puller.get_orders(timestamp=250)
        print(f"4. Filter by timestamp>=250: Fetched {len(df_from_ts)} records")
        assert len(df_from_ts) == 3
        assert sorted(df_from_ts['order_id'].tolist()) == ['3', '4', '5']

        # 5. Filter by to_timestamp (until)
        df_to_ts = puller.get_orders(to_timestamp=350)
        print(f"5. Filter by to_timestamp<350: Fetched {len(df_to_ts)} records")
        assert len(df_to_ts) == 3
        assert sorted(df_to_ts['order_id'].tolist()) == ['1', '2', '3']

        # 6. Filter by timestamp and to_timestamp
        df_between_ts = puller.get_orders(timestamp=150, to_timestamp=450)
        print(f"6. Filter by 150<=timestamp<450: Fetched {len(df_between_ts)} records")
        assert len(df_between_ts) == 3
        assert sorted(df_between_ts['order_id'].tolist()) == ['2', '3', '4']

        # 7. Filter by multiple conditions
        df_combo = puller.get_orders(symbol='FX-ETH/USDT', exchange=ExchangeName.BINANCE)
        print(f"7. Filter by symbol='ETH-USDT' and exchange='BINANCE': Fetched {len(df_combo)} records")
        assert len(df_combo) == 1
        assert df_combo['order_id'].iloc[0] == '4'
        
        # 8. Filter by strategy_name
        df_sn1 = puller.get_orders(strategy_name='sn1')
        print(f"8. Filter by strategy_name='sn1': Fetched {len(df_sn1)} records")
        assert len(df_sn1) == 3
        assert sorted(df_sn1['order_id'].tolist()) == ['1', '2', '5']

        # 9. Filter by strategy_process_name
        df_spn2 = puller.get_orders(strategy_process_name='spn2')
        print(f"9. Filter by strategy_process_name='spn2': Fetched {len(df_spn2)} records")
        assert len(df_spn2) == 2
        assert sorted(df_spn2['order_id'].tolist()) == ['3', '4']

        print("All tests passed!")

        # Clean up
        dbu.delete_table()
        dbu.close()

    test_pull_db_data()