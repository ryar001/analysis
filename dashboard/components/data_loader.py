import json
import os
import sys
import yaml
import pandas as pd

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from common.db_utils_pandas import DbUtils

def get_strategy_stats_db_utils(db_path, settings_path):
    """
    Initializes DbUtils for the strategy_stats table.

    Args:
        db_path (str): Path to the db.json file.
        settings_path (str): Path to the send_to_lark_settings.yaml file.

    Returns:
        DbUtils: An instance of DbUtils.
    """
    with open(settings_path, 'r') as f:
        trade_settings = yaml.safe_load(f)
    
    with open(db_path, 'r') as f:
        db_config = json.load(f)

    table_name = trade_settings['private']['table_name']
    db_utils = DbUtils(db_config=db_config, table_name=table_name)
    return db_utils

def pull_db_data():
    """
    Fetches data from the strategy_stats table.

    Returns:
        pd.DataFrame: A DataFrame containing the strategy stats.
    """
    # Construct absolute paths to the configuration files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, '..', '..', '..', 'common', 'db.json')
    settings_path = os.path.join(current_dir, '..', '..', 'send_to_lark_settings.yaml')

    db_utils = get_strategy_stats_db_utils(db_path, settings_path)
    db_utils.start()
    
    # Fetch all data from the table
    df = db_utils.get_table_as_df()
    db_utils.close()
    return df

if __name__ == '__main__':
    # For testing purposes
    df = pull_db_data()
    print(df.head())