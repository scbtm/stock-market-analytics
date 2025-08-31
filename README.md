# Stock Market Analytics

A Python package for collecting and analyzing stock market data using yfinance with clean validation and processing.

## Features

- **Clean Data Collection**: Robust yfinance integration with error handling and validation
- **Flexible Configuration**: Pydantic-based collection plans with comprehensive validation
- **Data Processing**: Timeline continuity processing and schema enforcement
- **Library Interface**: Clean API for integration into your applications
- **Comprehensive Testing**: Full test suite with 53+ tests covering all functionality

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
# Clone the repository
git clone <repository-url>
cd stock-market-analytics

# Install dependencies
uv sync

# Install in development mode
uv pip install -e .
```

**NOTE**
In addition to the uv dependencies, at least for development and visualizations with hamilton, you need:
```bash
sudo apt install graphviz
```

## Quick Start

```python
from stock_market_analytics.data_collection import (
    YFinanceCollector, 
    YFinanceCollectionPlan, 
    ContinuousTimelineProcessor
)

# Create a collection plan
plan = YFinanceCollectionPlan(symbol="AAPL", period="1y")

# Collect data
collector = YFinanceCollector(plan)
data = collector.get_historical_data()

print(f"Collected {len(data)} records for AAPL")
print(data.head())

# Process for continuous timeline (optional)
processor = ContinuousTimelineProcessor("AAPL", data)
processed_data = processor.process()
```

## Package Structure

```
src/stock_market_analytics/
├── __init__.py
└── data_collection/               # Data collection module
    ├── __init__.py               # Main exports
    ├── collectors/               # Data collectors
    │   ├── __init__.py
    │   ├── base.py              # Protocol definitions
    │   └── yfinance_collector.py # YFinance implementation
    ├── models/                   # Data models
    │   ├── __init__.py
    │   └── collection_plans.py   # Pydantic validation models
    └── processors/               # Data processors
        ├── __init__.py
        └── timeline.py          # Timeline continuity processor
```

## API Reference

### YFinanceCollectionPlan

Pydantic model for validating collection parameters:

```python
# Valid configurations
plan1 = YFinanceCollectionPlan(symbol="AAPL", period="1y")
plan2 = YFinanceCollectionPlan(symbol="TSLA", start="2023-01-01", end="2023-12-31")
plan3 = YFinanceCollectionPlan(symbol="GOOGL", period="1mo", interval="1d")
```

**Validation Rules:**
- Either `start` OR `period` must be provided
- `start` must be before `end` when both provided
- Date format must be `YYYY-MM-DD`
- `interval` cannot be used with both `start` and `end`

### YFinanceCollector

Collects historical stock data from Yahoo Finance:

```python
collector = YFinanceCollector(plan)
data = collector.get_historical_data()

# Check collection status
print(f"Success: {collector.collection_successful}")
print(f"Empty data: {collector.collected_empty_data}")
print(f"Errors: {collector.errors_during_collection}")
```

**Returns:** Polars DataFrame with schema:
- `date`: pl.Date
- `symbol`: pl.Utf8  
- `open`, `high`, `low`, `close`: pl.Float64
- `volume`: pl.Int64

### ContinuousTimelineProcessor

Ensures data has continuous timeline (fills missing dates):

```python
processor = ContinuousTimelineProcessor("AAPL", data)
processed_data = processor.process()

print(f"Processing successful: {processor.processing_successful}")
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/unit/test_data_collection/test_collectors.py

# Run with coverage
uv run pytest --cov=stock_market_analytics
```

### Code Quality

```bash
# Format code
uv run ruff format

# Lint code  
uv run ruff check

# Type checking
uv run pyright
```

### Project Commands

```bash
# Install dependencies
uv add <package-name>

# Test imports
uv run python -c "from stock_market_analytics.data_collection import *"

# Run Python with project environment
uv run python -c "from stock_market_analytics import *"
```

## Dependencies

- **polars**: Fast DataFrame library for data processing
- **yfinance**: Yahoo Finance API client
- **pydantic**: Data validation using Python type annotations
- **pyarrow**: Required for pandas-polars conversion

## Entry Points

**Run batch data collection of the stocks in the tickers file**
The tickers file can be downloaded from the [Nasdaq website](https://www.nasdaq.com/market-activity/stocks/screener)
```sh
uv run batch-collect run
```