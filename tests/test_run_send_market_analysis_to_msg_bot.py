import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import sys
import os
from unittest.mock import MagicMock

# Add project root to path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from analysis.run_send_market_analysis_to_msg_bot import get_latest_market_stats

@pytest.mark.asyncio
async def test_get_latest_market_stats_success():
    """Test get_latest_market_stats returns data when db call is successful."""
    # Arrange
    mock_db_utils = MagicMock()
    expected_df = pd.DataFrame([{'symbol': 'BTCUSDT', 'exchange': 'binance', 'timestamp': 1234567890}])
    mock_db_utils.get_table_as_df.return_value = expected_df
    
    # Act
    result_df = await get_latest_market_stats(
        db_utils=mock_db_utils,
        from_timestamp=1234567800,
        exchange='binance',
        symbol='BTCUSDT'
    )
    
    # Assert
    mock_db_utils.get_table_as_df.assert_called_once_with(
        where_clause='symbol = :symbol AND exchange = :exchange AND "timestamp" >= :from_timestamp ORDER BY "timestamp" DESC',
        params={'symbol': 'BTCUSDT', 'exchange': 'binance', 'from_timestamp': 1234567800},
        limit=1
    )
    assert_frame_equal(result_df, expected_df)

@pytest.mark.asyncio
async def test_get_latest_market_stats_empty():
    """Test get_latest_market_stats returns empty DataFrame when db returns empty."""
    # Arrange
    mock_db_utils = MagicMock()
    mock_db_utils.get_table_as_df.return_value = pd.DataFrame()
    
    # Act
    result_df = await get_latest_market_stats(
        db_utils=mock_db_utils,
        from_timestamp=1234567800,
        exchange='binance',
        symbol='BTCUSDT'
    )
    
    # Assert
    assert result_df.empty

@pytest.mark.asyncio
async def test_get_latest_market_stats_none():
    """Test get_latest_market_stats returns empty DataFrame when db returns None."""
    # Arrange
    mock_db_utils = MagicMock()
    mock_db_utils.get_table_as_df.return_value = None
    
    # Act
    result_df = await get_latest_market_stats(
        db_utils=mock_db_utils,
        from_timestamp=1234567800,
        exchange='binance',
        symbol='BTCUSDT'
    )
    
    # Assert
    assert result_df.empty

@pytest.mark.asyncio
async def test_get_latest_market_stats_exception():
    """Test get_latest_market_stats returns empty DataFrame on db exception."""
    # Arrange
    mock_db_utils = MagicMock()
    mock_db_utils.get_table_as_df.side_effect = Exception("DB Error")
    mock_logger = MagicMock()
    
    # Act
    result_df = await get_latest_market_stats(
        db_utils=mock_db_utils,
        from_timestamp=1234567800,
        exchange='binance',
        symbol='BTCUSDT',
        logger=mock_logger
    )
    
    # Assert
    assert result_df.empty
    mock_logger.error.assert_called_once()
    call_args = mock_logger.error.call_args[0][0]
    assert "Failed to fetch market stats for BTCUSDT@binance: DB Error" in call_args
