Please provide the git diff output. The variable `${DIFF}` was not expanded.

## 2025-10-21

### Refactor

#### `trades_analysis.py`

- Updated default exchange from `phemex` to `xt` for all related table names and parameters.
- Adjusted default parameters in the `main` function for a new analysis run:
    - `account_id` set to `all`.
    - `days` set to `0`.
    - `parse_orders` enabled.
    - `plot_rolling_stats` and `send_position_timeseries_to_lark` disabled.
- Reduced pandas `display.max_rows` from 500 to 100.
- Commented out the call to `parse_phemex_orders`.

## 2025-10-10

### Refactor

#### `ai-tracker.sh`

- Improved the AI code review prompt to be more robust, adding explicit checks for errors and breakpoints.
- Added a step to abort the update process if the AI review finds any errors.

#### `trades_analysis.py`

- Consolidated the result instantiation logic to remove duplicated code.

### What's New

#### `send_to_lark_settings.yaml`

- Added `FX-USDC/USDT` to the market stats configuration for the `XT_FX` exchange.

#### `trades_analysis.py`

- The trade analysis title now includes the product type (e.g., spot).
- The summary now includes the number of executed buy and sell trades.



## 2025-09-23

### What's New

#### `ai-tracker.sh`

- Added a new script to automate tracking git changes, generating summaries, updating `UPDATES.md`, and creating commits.

#### `trades_analysis.py`

- Implemented the calculation and reporting of the overall order execution rate.
- Added logic to calculate and report the execution rate for specific time periods.

### Refactor

#### `trades_analysis.py`

- Replaced hardcoded column name strings with the `OrderInfo` enum for mapping trade data, improving code clarity and maintainability.
