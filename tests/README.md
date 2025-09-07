# Tests Documentation

This directory contains unit and integration tests for the stock market analytics project, organized to mirror the source code structure and maintain clear separation of concerns.

## Core Testing Rules

### 1. Mirror the Source Code Structure Exactly 📂
The `tests/` directory structure must **exactly mirror** the `src/` directory structure. For every source module at `src/stock_market_analytics/module/submodule/file.py`, there should be a corresponding test at `tests/unit/test_module/test_submodule/test_file.py`.

**Example:**
```
src/stock_market_analytics/modeling/pipeline_components/functions.py
tests/unit/test_modeling/test_pipeline_components/test_functions.py
```

This makes tests highly discoverable and eliminates confusion about where to add tests for any module.

### 2. Choose the Right Test Type 🎯
- **Unit Tests (`tests/unit/`)**: Test individual functions/classes in isolation with mocked dependencies
- **Integration Tests (`tests/integration/`)**: Test interactions between components or with external systems

**Decision criteria:**
- Has external dependencies (API calls, file I/O, database)? → Integration test
- Pure function with no external dependencies? → Unit test
- Tests component interactions? → Integration test

### 3. Unit Test Principles 🔬
- **Single Responsibility**: Each test focuses on one specific behavior
- **Fast Execution**: Should run in milliseconds, not seconds
- **Isolation**: Mock all external dependencies (APIs, file system, complex objects)
- **Comprehensive Edge Cases**: Test null inputs, empty collections, boundary conditions, error paths
- **Descriptive Names**: `test_function_does_what_when_condition`

### 4. Integration Test Principles 🔗
- **Real Interactions**: Test how components actually work together
- **Mock External Systems**: Mock APIs, databases, file systems - not internal components
- **Error Scenarios**: Verify proper handling of external system failures
- **Contract Testing**: Ensure data flows correctly between components

### 5. Effective Mocking Strategy 🎭
- **Mock at the Boundary**: Mock external systems, not internal components
- **Mock for Isolation**: Use mocks to isolate the unit under test
- **Don't Over-Mock**: In integration tests, let internal components interact naturally
- **Verify Behavior**: Assert on function calls and state changes, not just return values


## Structure

The test structure exactly mirrors the source code structure:

```
tests/
├── __init__.py
├── conftest.py                                    # Common test fixtures and configuration  
├── README.md                                      # This file
├── unit/                                         # Unit tests (isolated, fast)
│   ├── test_config.py                           # Tests src/stock_market_analytics/config.py
│   ├── test_data_collection/                    # Tests src/stock_market_analytics/data_collection/
│   │   ├── test_collection_steps.py            #   └── collection_steps.py
│   │   ├── test_collectors.py                   #   └── collectors/ (unit tests)
│   │   ├── test_data_quality.py                #   └── processors/data_quality.py
│   │   ├── test_models.py                       #   └── models/collection_plans.py
│   │   └── test_processors.py                   #   └── processors/timeline.py
│   ├── test_feature_engineering/               # Tests src/stock_market_analytics/feature_engineering/
│   │   ├── test_feature_pipeline.py            #   ├── feature_pipeline.py
│   │   └── test_feature_steps.py               #   └── feature_steps.py
│   └── test_modeling/                           # Tests src/stock_market_analytics/modeling/
│       ├── test_modeling_steps.py              #   ├── modeling_steps.py
│       ├── test_training_flow_cb.py             #   ├── training_flow_cb.py
│       └── test_pipeline_components/           #   └── pipeline_components/
│           ├── test_calibrators.py             #       ├── calibrators.py
│           ├── test_evaluators.py              #       ├── evaluators.py
│           ├── test_functions.py               #       ├── functions.py
│           ├── test_naive_baselines.py         #       ├── naive_baselines.py
│           ├── test_pipeline_factory.py        #       ├── pipeline_factory.py
│           └── test_predictors.py              #       └── predictors.py
└── integration/                                 # Integration tests (external dependencies)
    └── test_data_collection/                   # External API interactions
        └── test_collectors.py                  # YFinanceCollector API tests
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
uv run pytest tests/unit/test_modeling/test_pipeline_components/test_functions.py -v

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
- **Steps (`test_collection_steps.py`)**: Complete collection workflow functions with comprehensive mocking
- **Models (`test_models.py`)**: `YFinanceCollectionPlan` validation and conversion logic
- **Processors (`test_processors.py`)**: `ContinuousTimelineProcessor` data processing workflows
- **Data Quality (`test_data_quality.py`)**: Data validation rules and quality checking
- **Collectors (`test_collectors.py`)**: Basic `YFinanceCollector` initialization and protocol compliance

**Feature Engineering Module**:
- **Pipeline (`test_feature_pipeline.py`)**: Hamilton pipeline functions for feature creation
  - ✅ Data sorting and preprocessing functions
  - ✅ Date-based feature generation
  - ✅ Technical indicators (volatility, momentum, statistical)
  - ✅ Ichimoku cloud features
  - ✅ Missing value handling and data joins
- **Steps (`test_feature_steps.py`)**: Feature engineering workflow orchestration steps

**Modeling Module**:
- **Steps (`test_modeling_steps.py`)**: Complete modeling workflow orchestration
- **Training Flow (`test_training_flow_cb.py`)**: CatBoost-specific training flow (placeholder)
- **Pipeline Components**:
  - **Functions (`test_functions.py`)**: Conformal prediction, evaluation metrics, plotting utilities
  - **Calibrators (`test_calibrators.py`)**: Model calibration components (placeholder)
  - **Evaluators (`test_evaluators.py`)**: Model evaluation components (placeholder)
  - **Predictors (`test_predictors.py`)**: Prediction components (placeholder)
  - **Naive Baselines (`test_naive_baselines.py`)**: Baseline model implementations (placeholder)
  - **Pipeline Factory (`test_pipeline_factory.py`)**: Pipeline construction utilities (placeholder)

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