# Project Planning

## Project: Trading Dashboard

**Objective:** Create a trading dashboard to visualize the strategy performance data stored in the database.

**Date:** 2025-10-02

### Architecture and Design

- **Framework:** The dashboard is built using the Dash framework, a Python library for creating interactive web applications.
- **Data Source:** The data is fetched from a PostgreSQL database.
- **Data Access:** A dedicated data loader module (`analysis/dashboard/components/data_loader.py`) is responsible for fetching data from the `strategy_stats` table. It uses the `common.db_utils_pandas.DbUtils` class for database interactions.
- **Modularity:** The application is structured into two main files:
    - `analysis/dashboard/app.py`: The main application file that contains the dashboard layout and callbacks.
    - `analysis/dashboard/components/data_loader.py`: A module for fetching data from the database.

### Features

- **Interactive Filters:** The dashboard provides filters for:
    - Date range
    - Account name
    - Strategy name
- **Visualizations:** The dashboard includes the following visualizations:
    - **PNL Timeseries:** A line chart showing Profit and Loss (PNL) over time for selected strategies and accounts.
    - **Strategy Summary Table:** A table that summarizes key metrics for each strategy, such as total PNL, turnover, and trade counts.
    - **Trade Volume Analysis:** Bar charts that visualize the number of buy/sell trades and the total buy/sell size for each strategy.
    - **Raw Data Table:** A table that displays the raw data from the `strategy_stats` table.

### How to Run

1.  Navigate to the `analysis/dashboard` directory:
    ```bash
    cd analysis/dashboard
    ```
2.  Run the `app.py` file:
    ```bash
    python app.py
    ```
3.  Open the URL provided in the terminal in a web browser to view the dashboard.