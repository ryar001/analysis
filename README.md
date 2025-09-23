# Asset Management Analysis

This project provides a set of tools for analyzing asset management data, with a focus on trade analysis. It allows users to pull trade data from a database, perform various analyses, and generate reports and charts.

## Features

*   **Trade Analysis:** Calculates key metrics such as PnL, trade volume, weighted average prices, and position sizes.
*   **Rolling Statistics:** Computes rolling statistics over a specified time period.
*   **Data Visualization:** Generates charts to visualize net position over time.
*   **Flexible Data Filtering:** Allows filtering of trade data by symbol, date range, exchange, and strategy.
*   **Report Generation:** Saves analysis results, filtered trades, and charts to files.

## How to Use

1.  **Configuration:**
    *   Ensure you have a `db.json` file in the `common` directory with the necessary database connection details. An example `db.json` might look like this:

        ```json
        {
            "DEFAULT": {
                "Trading": {
                    "user": "your_user",
                    "password": "your_password",
                    "host": "your_host",
                    "port": "your_port",
                    "dbname": "your_db"
                }
            }
        }
        ```

2.  **Run the Analysis:**
    *   Execute the `start_analysis.py` script from the `analysis` directory:

        ```bash
        python start_analysis.py
        ```

    *   The script will prompt you to enter various parameters for the analysis, such as:
        *   Global symbol (e.g., FX-BTC/USDT)
        *   Start and end timestamps
        *   Exchange
        *   Strategy name
        *   Row limit

3.  **Output:**
    *   The analysis results will be printed to the console.
    *   The following files will be saved in the `output_files` directory:
        *   `rolling_stats/rolling_stats__{timestamp}.csv`: Rolling statistics.
        *   `filters_trades/filtered_trades__{timestamp}.csv`: The filtered trade data used in the analysis.
        *   `filters_trades/filtered_trades_analysis__{timestamp}.csv`: A summary of the trade analysis.
        *   `position_timeseries_chart/position_timeseries_chart__{timestamp}.png`: A chart showing the net position over time.

## Future Implementations

*   **Market Statistics:**
    *   Integrate market data to provide broader context for trade analysis.
    *   Calculate and display key market statistics, such as:
        *   Market volatility
        *   Correlation with major indices
        *   Key support and resistance levels
    *   This will allow for a more comprehensive evaluation of trading strategy performance against market conditions.