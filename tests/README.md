# Tests Documentation

This directory contains comprehensive tests for the portfolio optimization project.

## Structure

```
tests/
├── __init__.py
├── conftest.py                 # Common test fixtures and configuration
├── README.md                   # This file
└── test_data_collection/       # Tests for data collection module
    ├── __init__.py
    ├── test_models.py          # Tests for data models (YFinanceCollectionPlan)
    ├── test_processors.py      # Tests for data processors (ContinuousTimelineProcessor)
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
# Run all tests
uv run pytest tests/ -v

# Run all tests with coverage
uv run pytest tests/ -v --cov=src

# Run tests in parallel (if pytest-xdist is installed)
uv run pytest tests/ -v -n auto
```

### Running Specific Test Modules

```bash
# Run only data collection tests
uv run pytest tests/test_data_collection/ -v

# Run only model tests
uv run pytest tests/test_data_collection/test_models.py -v

# Run only processor tests
uv run pytest tests/test_data_collection/test_processors.py -v
```


## Test Coverage

### Models (`test_models.py`)

Tests for `YFinanceCollectionPlan` model:
- ✅ Valid parameter combinations
- ✅ Date format validation
- ✅ Business rule validation (start/end dates, periods, intervals)
- ✅ Edge cases and boundary conditions
- ✅ Conversion to yfinance parameters
- ✅ All supported intervals and periods

### Processors (`test_processors.py`)

Tests for `ContinuousTimelineProcessor` class:
- ✅ Initialization and state management
- ✅ Schema validation and type conversion
- ✅ Data continuity checks
- ✅ Error handling for invalid data
- ✅ Date sorting and ordering
- ✅ Processing workflow and state transitions
- ✅ Different symbols and large date ranges

## Test Data and Fixtures

Tests use pytest fixtures to provide consistent test data:

- `sample_valid_data`: Properly formatted continuous stock data
- `sample_data_with_gaps`: Stock data with missing dates
- `sample_data_wrong_schema`: Data with incorrect column types
- `empty_data`: Empty DataFrame for edge case testing

## Best Practices

1. **Isolation**: Each test is independent and doesn't rely on external data
2. **Coverage**: Tests cover both happy path and error conditions
3. **Clarity**: Test names clearly describe what is being tested
4. **Fixtures**: Reusable test data through pytest fixtures
5. **Assertions**: Clear, specific assertions that test expected behavior

## Adding New Tests

When adding new tests:

1. Follow the existing naming convention (`test_*.py`)
2. Use descriptive test method names (`test_what_is_being_tested`)
3. Include docstrings explaining the test purpose
4. Use appropriate fixtures for test data
5. Test both success and failure scenarios
6. Keep tests simple and focused on a single aspect

## Future Extensions

The test structure is designed to easily accommodate additional modules:

- `test_collectors/` - For testing data collectors (with mocks)
- `test_integration/` - For integration tests
- `test_performance/` - For performance and load tests 