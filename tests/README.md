# Tests Documentation

This directory contains unit and integration tests for the stock market analytics project, organized to mirror the source code structure exactly.

## Structure

```
tests/
├── conftest.py                                    # Common test fixtures and configuration
├── unit/                                         # Unit tests (isolated, fast)
│   ├── test_config.py                           # Tests src/stock_market_analytics/config.py
│   ├── test_data_collection/                    # Tests src/stock_market_analytics/data_collection/
│   │   ├── test_collection_steps.py            #   ├── collection_steps.py
│   │   ├── test_collectors.py                   #   ├── collectors/
│   │   ├── test_data_quality.py                #   ├── processors/data_quality.py
│   │   ├── test_models.py                       #   ├── models/
│   │   └── test_processors.py                   #   └── processors/
│   ├── test_feature_engineering/               # Tests src/stock_market_analytics/feature_engineering/
│   │   ├── test_feature_pipeline.py            #   ├── feature_pipeline.py
│   │   └── test_feature_steps.py               #   └── feature_steps.py
│   ├── test_inference/                          # Tests src/stock_market_analytics/inference/
│   │   ├── test_inference_functions.py         #   ├── inference_functions.py
│   │   └── test_inference_steps.py             #   └── inference_steps.py
│   ├── test_modeling/                           # Tests src/stock_market_analytics/modeling/
│   │   ├── test_modeling_steps.py              #   ├── modeling_steps.py
│   │   ├── test_training_flow_cbm_qr.py        #   ├── training_flow_cbm_qr.py
│   │   └── test_model_factory/                 #   └── model_factory/
│   │       ├── test_calibration_functions.py   #       ├── calibration/calibration_functions.py
│   │       ├── test_calibrators.py             #       ├── calibration/calibrators.py
│   │       ├── test_data_management/           #       ├── data_management/
│   │       │   └── test_preprocessing.py       #       │   └── preprocessing.py
│   │       ├── test_estimation/                #       ├── estimation/
│   │       │   └── test_estimation_functions.py#       │   └── estimation_functions.py
│   │       ├── test_evaluators.py              #       ├── evaluation/evaluators.py
│   │       ├── test_functions.py               #       ├── evaluation/evaluation_functions.py
│   │       ├── test_naive_baselines.py         #       └── (legacy components)
│   │       ├── test_pipeline_factory.py
│   │       └── test_predictors.py
│   └── test_monitoring/                         # Tests src/stock_market_analytics/monitoring/
│       └── test_monitoring_metrics.py          #   └── monitoring_metrics.py
└── integration/                                 # Integration tests (external dependencies)
    ├── test_data_collection/                   # Data collection workflow integration
    │   ├── test_collection_steps.py            # Collection workflow step functions
    │   └── test_collectors.py                  # YFinance API integration (real API calls)
    ├── test_feature_engineering/               # Feature engineering workflow integration
    │   └── test_feature_steps.py               # Feature engineering step functions
    ├── test_inference/                         # Inference workflow integration
    │   └── test_inference_steps.py             # Inference workflow step functions
    ├── test_modeling/                          # Model training workflow integration
    │   └── test_modeling_steps.py              # Modeling workflow step functions
    └── test_monitoring/                        # Model monitoring workflow integration
        └── test_monitoring_steps.py            # Monitoring workflow step functions
```

## Testing Philosophy

### Unit Tests (`tests/unit/`)
- **Pure Functions**: Test individual functions in isolation
- **Minimal Mocking**: Avoid mocking internal components (antipattern)
- **Fast Execution**: Run in milliseconds for rapid feedback
- **Single Responsibility**: Each test focuses on one specific behavior

**❌ Avoid in Unit Tests:**
- Heavy mocking of internal components (e.g., YFinanceCollector, Hamilton driver)
- Multi-component workflow orchestration tests
- Tests that coordinate between multiple internal modules

### Integration Tests (`tests/integration/`)
- **Workflow Orchestration**: Test how multiple internal components work together
- **External Dependencies**: Mock external systems (W&B, file system) but NOT internal components
- **End-to-End Scenarios**: Test complete workflows from start to finish
- **Real YFinance API**: The only external dependency we don't mock

**✅ Good Candidates for Integration Tests:**
- W&B artifact downloading and model loading workflows
- Hamilton pipeline execution and feature generation workflows
- Multi-step data collection and processing workflows
- YFinance API integration (real API calls marked with `@pytest.mark.slow`)

## Running Tests

### All Tests
```bash
# Run all tests (unit + integration)
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --cov=src

# Skip slow tests (YFinance API) for faster feedback
uv run pytest tests/ -v -m "not slow"
```

### By Type
```bash
# Unit tests only (fast, no external dependencies)
uv run pytest tests/unit/ -v

# Integration tests only
uv run pytest tests/integration/ -v

# Only slow tests (real YFinance API calls)
uv run pytest tests/integration/ -v -m "slow"
```

### By Module
```bash
# Unit tests by module
uv run pytest tests/unit/test_data_collection/ -v
uv run pytest tests/unit/test_feature_engineering/ -v
uv run pytest tests/unit/test_modeling/ -v
uv run pytest tests/unit/test_inference/ -v
uv run pytest tests/unit/test_monitoring/ -v

# Integration tests by module
uv run pytest tests/integration/test_data_collection/ -v
uv run pytest tests/integration/test_feature_engineering/ -v
uv run pytest tests/integration/test_inference/ -v
uv run pytest tests/integration/test_modeling/ -v
uv run pytest tests/integration/test_monitoring/ -v
```

## Test Coverage Summary

### Unit Tests
- **Configuration**: Environment variables, property calculations
- **Data Collection**: CSV loading, validation, data quality, processing
- **Feature Engineering**: Hamilton pipeline functions, technical indicators
- **Modeling**: Model factory components, evaluation metrics, training flows
- **Inference**: Prediction utilities and functions
- **Monitoring**: Performance and drift metrics

### Integration Tests
- **Data Collection**: End-to-end collection workflow, real YFinance API integration
- **Feature Engineering**: Hamilton pipeline integration, complete feature building
- **Inference**: W&B artifact workflows, complete prediction pipelines
- **Modeling**: Time-series splitting, model training workflows
- **Monitoring**: Model loading, drift detection, performance evaluation

## Known Testing Antipatterns to Refactor

The following tests currently exist in unit tests but violate unit testing principles:

### Data Collection (`tests/unit/test_data_collection/test_collection_steps.py`)
- `TestCollectAndProcessSymbol` - Heavy mocking of internal components
- `TestUpdateHistoricalData` - Workflow orchestration tests

### Feature Engineering (`tests/unit/test_feature_engineering/test_feature_steps.py`)
- `TestCreateFeaturePipeline` - Mocks Hamilton driver (internal component)
- `TestBuildFeaturesFromData` - Complete workflow with heavy mocking

### Inference (`tests/unit/test_inference/test_inference_steps.py`)
- `test_download_and_load_model` - W&B workflow integration
- `test_get_inference_data_*` - Multi-component coordination

**Recommendation:** These should be refactored to focus on individual function behavior or moved to integration tests.

## Adding New Tests

### For Unit Tests
- **Pure function, no external calls** → `tests/unit/`
- Mirror source structure exactly: `src/module/file.py` → `tests/unit/test_module/test_file.py`
- Focus on individual function behavior
- Use real test data, avoid mocking internal components

### For Integration Tests
- **Uses external APIs, files, or databases** → `tests/integration/`
- **Workflow orchestration of multiple components** → `tests/integration/`
- Mock external systems only, let internal components interact naturally
- Test both success and failure scenarios