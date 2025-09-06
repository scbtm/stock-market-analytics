# Tests Documentation

This directory contains unit and integration tests for the stock market analytics project, properly organized by test type.

## Structure

```
tests/
├── __init__.py
├── conftest.py                          # Common test fixtures and configuration
├── README.md                            # This file
├── unit/                               # Pure unit tests (no external dependencies)
│   ├── test_config.py                  # Configuration object tests
│   ├── test_data_collection/           # Unit tests for data collection
│   │   ├── test_models.py             # Data models (YFinanceCollectionPlan)
│   │   ├── test_processors.py         # Data processors (ContinuousTimelineProcessor)
│   │   └── test_collectors_unit.py    # Basic collector initialization tests
│   ├── test_feature_engineering/       # Pure function tests
│   │   └── test_feature_pipeline.py   # Core feature pipeline functions
│   └── test_modeling/                  # Unit tests for modeling
│       ├── test_functions.py          # Modeling utility functions
│       └── test_processing.py         # Data processing functions
└── integration/                       # Integration tests (external API interactions)
    └── test_data_collection/          # Integration tests for data collection
        └── test_collectors.py         # YFinanceCollector API interaction tests
```

## Running Tests

### Prerequisites

Make sure you have pytest installed in your environment:

```bash
# If using uv (recommended)
uv add --dev pytest

# Or using pip
pip install pytest
```

### Running All Tests

From the project root directory:

```bash
# Run all tests (unit + integration)
uv run pytest tests/ -v

# Run all tests with coverage
uv run pytest tests/ -v --cov=src

# Run tests in parallel (if pytest-xdist is installed)
uv run pytest tests/ -v -n auto
```

### Running Tests by Type

```bash
# Run only unit tests (fast, no external dependencies)
uv run pytest tests/unit/ -v

# Run only integration tests (may use mocked external APIs)
uv run pytest tests/integration/ -v

# Run unit tests with coverage (recommended for development)
uv run pytest tests/unit/ -v --cov=src
```

### Running Specific Test Modules

```bash
# Unit tests
uv run pytest tests/unit/test_data_collection/ -v
uv run pytest tests/unit/test_feature_engineering/test_feature_pipeline.py -v
uv run pytest tests/unit/test_modeling/test_functions.py -v

# Integration tests  
uv run pytest tests/integration/test_data_collection/test_collectors.py -v
```


## Test Coverage

### Unit Tests (Pure, No External Dependencies)

**Configuration (`test_config.py`)**:
- ✅ Default configuration values
- ✅ Environment variable loading
- ✅ Configuration property calculations

**Data Collection Module**:
- **Models (`test_models.py`)**: `YFinanceCollectionPlan` validation and conversion
- **Processors (`test_processors.py`)**: `ContinuousTimelineProcessor` data processing workflows
- **Collectors (`test_collectors_unit.py`)**: Basic `YFinanceCollector` initialization and protocol compliance

**Feature Engineering (`test_feature_pipeline.py`)**:
- ✅ Data sorting and preprocessing functions
- ✅ Date-based feature generation
- ✅ Technical indicators (volatility, momentum, statistical)
- ✅ Ichimoku cloud features
- ✅ Missing value handling and data joins

**Modeling**:
- **Functions (`test_functions.py`)**: Conformal prediction functions, coverage metrics
- **Processing (`test_processing.py`)**: Data splitting and metadata functions

### Integration Tests (External API Interactions with Mocks)

**Data Collection (`test_collectors.py`)**:
- ✅ `YFinanceCollector` external API interactions (mocked)
- ✅ Success/failure scenarios with yfinance API
- ✅ Error handling and state management
- ✅ Data transformation and column mapping


## Test Data and Fixtures

Tests use pytest fixtures to provide consistent test data:

- `sample_valid_data`: Properly formatted continuous stock data
- `sample_data_with_gaps`: Stock data with missing dates
- `sample_data_wrong_schema`: Data with incorrect column types
- `empty_data`: Empty DataFrame for edge case testing

## Testing Philosophy

This project follows a **layered testing approach** with proper separation of concerns:

### Unit Tests (`tests/unit/`)
1. **Pure Functions**: Test individual functions in isolation with no external dependencies
2. **No Mocking**: Use real data and avoid mocking internal components
3. **Fast Execution**: Should run quickly for rapid feedback during development
4. **Happy Path + Edge Cases**: Cover core functionality and critical edge cases
5. **Minimal Complexity**: Focus on behavior, not implementation details

### Integration Tests (`tests/integration/`)
1. **External Interactions**: Test components that interact with external systems (APIs, databases)
2. **Mock External Systems**: Use mocks for external dependencies (yfinance API, file system)
3. **System Integration**: Test how components work together in realistic scenarios
4. **Error Handling**: Verify proper handling of external system failures

## Adding New Tests

### For Unit Tests (`tests/unit/`)
When testing pure functions with no external dependencies:

1. **Choose Unit Tests When**:
   - Testing individual functions or methods
   - No external API calls, file I/O, or database interactions
   - Can use real test data without mocking

2. **Structure**:
   - Create test class: `TestMyFunctionUnit`
   - 2-4 focused tests: happy path + key edge cases
   - Use descriptive names: `test_function_does_what_with_input_type`

### For Integration Tests (`tests/integration/`)
When testing components that interact with external systems:

1. **Choose Integration Tests When**:
   - Component makes external API calls (yfinance, web APIs)
   - Reads/writes files or databases
   - Tests interaction between multiple components
   - Need to mock external dependencies

2. **Structure**:
   - Create test class: `TestMyComponentIntegration`
   - Use `@patch` or `Mock` for external dependencies
   - Test both success and failure scenarios
   - Verify external calls are made correctly

### Quick Decision Guide
- **Pure function, no external calls** → `tests/unit/`
- **Uses external APIs, files, or databases** → `tests/integration/`
- **When in doubt** → Start with unit test, move to integration if mocking is needed 