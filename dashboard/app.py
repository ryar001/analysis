import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from components.data_loader import pull_db_data
import yaml
import os
import sys
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.logging_utils import LoggingUtils

# --- Settings and Logging ---
SETTINGS_DIR = os.path.join(os.path.dirname(__file__), 'dashboard_settings')
SETTINGS_FILE = os.path.join(SETTINGS_DIR, 'dashboard_settings.yaml')

# Load settings from YAML file
with open(SETTINGS_FILE, 'r') as f:
    settings = yaml.safe_load(f)

# Setup logging
log_settings = settings.get('logging', {})
log_dir_relative = log_settings.get('log_dir', '../Logs')
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), log_dir_relative))
log_name = datetime.now().strftime(log_settings.get('log_name_format', 'dashboard_%d-%m-%y.log'))

logging_utils = LoggingUtils(
    log_dir=log_dir,
    log_name=log_name,
    log_level=log_settings.get('log_level', 'INFO'),
    print_output=log_settings.get('print_output', True)
)
logger = logging_utils.get_logger()
logger.info("Dashboard application started.")


# Initialize the Dash app
app = dash.Dash(__name__)
logger.info("Dash app initialized.")

# Load data
logger.info("Loading data...")
df = pull_db_data()
logger.info(f"Data loaded successfully. {len(df)} rows.")

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'],unit='us')

# Get unique values for dropdowns
account_names = df['account_name'].unique()
strategy_names = df['strategy_name'].unique()

# App layout
app.layout = html.Div([
    html.H1("Trading Dashboard"),
    
    html.Div([
        html.Div([
            html.Label("Date Range"),
            dcc.DatePickerRange(
                id='date-picker-range',
                min_date_allowed=df['timestamp'].min().date(),
                max_date_allowed=df['timestamp'].max().date(),
                start_date=df['timestamp'].min().date(),
                end_date=df['timestamp'].max().date()
            ),
        ], style={'width': '49%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Account Name"),
            dcc.Dropdown(
                id='account-name-dropdown',
                options=[{'label': i, 'value': i} for i in account_names],
                multi=True,
                placeholder="Select Account(s)"
            ),
        ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'})
    ]),
    
    html.Div([
        html.Label("Strategy Name"),
        dcc.Dropdown(
            id='strategy-name-dropdown',
            options=[{'label': i, 'value': i} for i in strategy_names],
            multi=True,
            placeholder="Select Strategy(s)"
        ),
    ]),
    
    dcc.Graph(id='pnl-timeseries'),
    
    html.H2("Strategy Summary"),
    dash_table.DataTable(
        id='strategy-summary-table',
        page_size=10,
        style_table={'overflowX': 'auto'}
    ),
    
    html.H2("Trade Volume Analysis"),
    dcc.Graph(id='trade-volume-bar'),
    dcc.Graph(id='trade-size-bar'),
    
    html.H2("Raw Data"),
    dash_table.DataTable(
        id='strategy-stats-table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        page_size=20,
        style_table={'overflowX': 'auto'}
    )
])

@app.callback(
    [
        Output('strategy-stats-table', 'data'),
        Output('pnl-timeseries', 'figure'),
        Output('strategy-summary-table', 'data'),
        Output('strategy-summary-table', 'columns'),
        Output('trade-volume-bar', 'figure'),
        Output('trade-size-bar', 'figure')
    ],
    [
        Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date'),
        Input('account-name-dropdown', 'value'),
        Input('strategy-name-dropdown', 'value')
    ]
)
def update_dashboard(start_date, end_date, selected_accounts, selected_strategies):
    logger.info(
        "Updating dashboard with filters: "
        f"start_date={start_date}, end_date={end_date}, "
        f"selected_accounts={selected_accounts}, selected_strategies={selected_strategies}"
    )
    filtered_df = df.copy()
    
    # Filter by date range
    if start_date and end_date:
        mask = (filtered_df['timestamp'] >= start_date) & (filtered_df['timestamp'] <= end_date)
        filtered_df = filtered_df.loc[mask]
        
    # Filter by account name
    if selected_accounts:
        filtered_df = filtered_df[filtered_df['account_name'].isin(selected_accounts)]
        
    # Filter by strategy name
    if selected_strategies:
        filtered_df = filtered_df[filtered_df['strategy_name'].isin(selected_strategies)]
        
    # Create PNL timeseries figure
    pnl_fig = px.line(filtered_df, x='timestamp', y='pnl', color='strategy_name', title='PNL Over Time')
    breakpoint()
    # Create Strategy Summary Table
    summary_df = filtered_df.groupby('strategy_name').agg(
        total_pnl=('pnl', 'sum'),
        total_turnover=('turnover', 'sum'),
        total_buy_trades=('num_buy_trades', 'sum'),
        total_sell_trades=('num_sell_trades', 'sum')
    ).reset_index()
    summary_columns = [{"name": i, "id": i} for i in summary_df.columns]
    
    # Create Trade Volume Analysis charts
    trade_volume_df = filtered_df.groupby('strategy_name')[['num_buy_trades', 'num_sell_trades']].sum().reset_index()
    trade_volume_fig = px.bar(trade_volume_df, x='strategy_name', y=['num_buy_trades', 'num_sell_trades'], title='Number of Buy/Sell Trades')
    
    trade_size_df = filtered_df.groupby('strategy_name')[['total_buy_size', 'total_sell_size']].sum().reset_index()
    trade_size_fig = px.bar(trade_size_df, x='strategy_name', y=['total_buy_size', 'total_sell_size'], title='Total Buy/Sell Size')
    
    logger.info("Dashboard updated successfully.")
    breakpoint()
    return filtered_df.to_dict('records'), pnl_fig, summary_df.to_dict('records'), summary_columns, trade_volume_fig, trade_size_fig

if __name__ == '__main__':
    logger.info("Starting dashboard server.")
    app.run(debug=True)
