# Tests Documentation

This directory contains focused unit tests for the stock market analytics project.

## Structure

```
tests/
├── __init__.py
├── conftest.py                          # Common test fixtures and configuration
├── README.md                            # This file
└── unit/                               # Unit tests organized by module
    ├── test_config.py                  # Tests for configuration module
    ├── test_data_collection/           # Tests for data collection module
    │   ├── test_models.py             # Data models (YFinanceCollectionPlan)
    │   ├── test_processors.py         # Data processors (ContinuousTimelineProcessor)
    │   └── test_collectors.py         # Data collectors (YFinanceCollector)
    ├── test_feature_engineering/       # Tests for feature engineering
    │   └── test_feature_pipeline.py   # Core feature functions
    └── test_modeling/                  # Tests for modeling module
        └── test_functions.py          # Modeling utility functions
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

### Configuration (`test_config.py`)

Tests for configuration classes:
- ✅ Default configuration values
- ✅ Environment variable loading
- ✅ Configuration property calculations

### Data Collection Module

**Models (`test_models.py`)**:
- ✅ `YFinanceCollectionPlan` validation and conversion

**Processors (`test_processors.py`)**:
- ✅ `ContinuousTimelineProcessor` data processing

**Collectors (`test_collectors.py`)**:
- ✅ `YFinanceCollector` data retrieval and error handling

### Feature Engineering (`test_feature_pipeline.py`)

Tests for core feature functions:
- ✅ Data sorting and preprocessing
- ✅ Date-based feature generation
- ✅ Missing value handling

### Modeling (`test_functions.py`)

Tests for modeling utility functions:
- ✅ Conformal prediction functions
- ✅ Coverage and metric calculations

## Test Data and Fixtures

Tests use pytest fixtures to provide consistent test data:

- `sample_valid_data`: Properly formatted continuous stock data
- `sample_data_with_gaps`: Stock data with missing dates
- `sample_data_wrong_schema`: Data with incorrect column types
- `empty_data`: Empty DataFrame for edge case testing

## Testing Philosophy

This project follows a **simple and focused** approach to unit testing:

1. **Function-Level Testing**: Each test focuses on a single function or method
2. **Happy Path Coverage**: Tests verify core functionality with valid inputs
3. **Essential Edge Cases**: Only test critical edge cases (empty data, basic error conditions)
4. **Readable Tests**: Test names clearly describe what is being verified
5. **Minimal Complexity**: Avoid over-testing implementation details

## Adding New Tests

When adding new tests:

1. **Keep it Simple**: Focus on testing the core functionality
2. **One Function, Few Tests**: 2-4 tests per function (happy path + key edge cases)
3. **Clear Names**: Use descriptive test method names (`test_function_does_what`)
4. **Assume Good Data**: Don't test every possible invalid input scenario
5. **Focus on Behavior**: Test what the function should do, not how it does it

## Guidelines for New Functions

When you add a new function to the codebase:

1. Create a simple test class with the function name: `TestMyNewFunction`
2. Add 2-3 focused tests:
   - `test_basic_functionality()` - Core happy path
   - `test_handles_empty_data()` - Empty/edge case
   - `test_invalid_input()` - One key error condition (if needed)
3. Keep each test under 10 lines when possible 