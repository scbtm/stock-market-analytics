# Stock Market Analytics

A production-ready machine learning pipeline for stock market analysis and predictive modeling, showcasing modern MLOps, software engineering, and data science best practices.

## About the Project

This project implements an end-to-end stock market analytics platform designed for predictive modeling of stock prices and market behavior. The system demonstrates enterprise-grade software engineering practices through:

- **Scalable Data Pipelines**: Robust data collection with real-time quality validation and feature engineering workflows
- **Production Architecture**: Modular design with clear separation of concerns
- **Type-Safe Configuration**: Centralized Pydantic-based configuration system with validation
- **Data Quality**: Real-time validation ensures only high-quality data reaches ML models
- **Code Quality**: Comprehensive testing, type checking, and code quality controls
- **MLOps Best Practices**: Versioned data flows, reproducible experiments, and automated validation

The platform focuses on creating reliable, maintainable, and scalable infrastructure for financial market analysis, emphasizing code quality over quick prototypes.

## Architecture Overview

The project follows a **3-layer modular architecture** designed for scalability, maintainability, and clean separation of concerns:

```
📁 stock-market-analytics/
├── src/stock_market_analytics/     # Core application code
│   ├── config.py                   # Centralized type-safe configuration
│   ├── main.py                     # CLI entry point  
│   ├── data_collection/            # Data ingestion pipeline
│   │   ├── collectors/             # 🔧 Core: Data collection logic
│   │   ├── processors/             # 🔧 Core: Data processing & validation  
│   │   ├── models/                 # 🔧 Core: Data models & schemas
│   │   ├── collection_steps.py     # 📋 Steps: Workflow step functions
│   │   └── batch_collection_flow.py # ⚙️ Flow: Orchestration layer
│   ├── feature_engineering/       # Feature computation pipeline
│   │   ├── feature_pipeline.py     # 🔧 Core: Hamilton feature functions
│   │   ├── feature_steps.py        # 📋 Steps: Feature workflow steps
│   │   └── feature_building_flow.py # ⚙️ Flow: Feature orchestration
│   └── modeling/                   # ML model training and evaluation
│       ├── pipeline_components/    # 🔧 Core: ML components & models
│       ├── modeling_steps.py       # 📋 Steps: Training workflow steps
│       └── training_flow_cb.py     # ⚙️ Flow: Training orchestration
├── tests/                          # Comprehensive test suite
├── Makefile                       # Development workflow automation
└── pyproject.toml                  # Dependencies and entry points
```

### 3-Layer Architecture Design

This architecture implements a clean **separation of concerns** across three distinct layers:

#### 🔧 **Core Layer** (Low-level)
**Purpose**: Contains the fundamental business logic and domain-specific functionality

- **Data Collection**: `collectors/`, `processors/`, `models/` - Raw data collection and validation logic
- **Feature Engineering**: `feature_pipeline.py` - Pure Hamilton functions for feature computation
- **Modeling**: `pipeline_components/` - CatBoost models, evaluators, calibrators

**Characteristics**:
- ✅ Pure business logic with minimal dependencies
- ✅ Highly testable and reusable components  
- ✅ No knowledge of orchestration or workflows
- ✅ Framework-agnostic (can work with any orchestration layer)

#### 📋 **Steps Layer** (Mid-level)
**Purpose**: Provides workflow step functions that combine core components for common use cases

- **Data Collection**: `collection_steps.py` - Functions like `load_tickers()`, `collect_and_process_symbol()`
- **Feature Engineering**: `feature_steps.py` - Functions like `build_features_from_data()`, `load_stock_data()`  
- **Modeling**: `modeling_steps.py` - Functions like `train_catboost_model()`, `evaluate_model()`

**Characteristics**:
- ✅ **Reusable**: Step functions can be used across different flows and scenarios
- ✅ **Simple**: Clean functions without complex class hierarchies
- ✅ **Testable**: Easy to unit test individual workflow steps
- ✅ **Focused**: Each step function has a single, clear responsibility

#### ⚙️ **Flow Layer** (High-level)  
**Purpose**: Metaflow orchestration that coordinates the entire workflow with logging and error handling

- **Orchestration**: `*_flow.py` files define the complete pipeline steps
- **Error Handling**: Centralized error handling and logging at the flow level
- **Scalability**: Leverages Metaflow for parallel processing and cloud scaling

**Characteristics**:
- ✅ **Clean & Focused**: Flows only handle orchestration, not business logic
- ✅ **Maintainable**: Easy to modify workflow steps without touching core logic
- ✅ **Observable**: All logging and monitoring happens at this level
- ✅ **Scalable**: Metaflow provides seamless local-to-cloud execution

### Architecture Benefits

This layered approach provides several key advantages:

🔄 **Reusability**: Step functions can be used across different flows, APIs, or batch jobs

🧪 **Testability**: Each layer can be independently tested
- Core: Unit tests for business logic
- Steps: Integration tests for workflow step functions  
- Flows: End-to-end pipeline tests

⚡ **Maintainability**: Changes are isolated to appropriate layers
- Business logic changes → Core layer
- Workflow step changes → Steps layer  
- Orchestration changes → Flow layer

📈 **Scalability**: Clean separation enables easy scaling strategies
- Core components can be optimized independently
- Steps can be deployed as microservices if needed
- Flows handle distributed execution automatically

🔧 **Flexibility**: Easy to swap implementations or add new flows
- Add new data sources → Core layer
- Add new workflow steps → Steps layer
- Add new workflows → Flow layer

## Code Structure

The project follows a modular architecture with **Metaflow workflows as primary entry points**, providing several key advantages:

### Metaflow Integration Benefits

1. **Reproducible Execution**: Every pipeline run is versioned and traceable
2. **Scalable Computing**: Easy transition from local to cloud execution
3. **Data Lineage**: Automatic tracking of data transformations and dependencies
4. **Fault Tolerance**: Built-in retry mechanisms and error handling
5. **Parallel Processing**: Native support for concurrent data processing

### Entry Points (via `pyproject.toml`)

```toml
[project.scripts]
stock-market-analytics = "stock_market_analytics.main:main"
batch-collect = "stock_market_analytics.data_collection.batch_collection_flow:BatchCollectionFlow"
build-features = "stock_market_analytics.feature_engineering.feature_building_flow:FeatureBuildingFlow"
train-model = "stock_market_analytics.modeling.training_flow_cb:TrainingFlow"
```

This design enables direct execution of complex workflows:
```bash
# Run data collection pipeline
uv run batch-collect run

# Execute feature engineering pipeline  
uv run build-features run

# Execute model training pipeline
uv run train-model run
```

### Data Collection Module

**Location**: `src/stock_market_analytics/data_collection/`

The data collection module exemplifies enterprise-grade data engineering patterns with production-ready data quality assurance:

#### Design Patterns & Architecture

1. **Abstract Base Classes**: `collectors/base.py` defines interfaces ensuring consistent collector implementations
2. **Strategy Pattern**: Pluggable collectors (YFinance, future: Alpha Vantage, IEX) with unified interfaces
3. **Data Validation**: Pydantic models (`models/collection_plans.py`) enforce schema validation and type safety
4. **Timeline Processing**: Intelligent incremental vs. full data refresh logic
5. **Real-time Quality Validation**: Comprehensive data quality checks applied during ingestion
6. **Metaflow Orchestration**: Parallel processing with automatic dependency management

#### Data Quality Validation System

The module features a sophisticated **DataQualityValidator** that ensures only high-quality data reaches downstream ML models:

**Quality Checks Applied**:
- **Price Consistency**: High ≥ Low validation for each trading period
- **Price Positivity**: All prices must be positive (no negative values)
- **Volume Validation**: Volume must be non-negative
- **OHLC Relationships**: Open/close prices within high/low bounds
- **Extreme Movement Detection**: Configurable thresholds flag potential data errors
- **Schema Completeness**: Required columns validation with null handling
- **Data Sufficiency**: Minimum data points validation

**Quality Assurance Benefits**:
- **ML Model Reliability**: Invalid data is automatically excluded from training datasets
- **Data Integrity**: Systematic validation prevents corrupted data propagation  
- **Configurable Rules**: Pydantic-based quality rules with sensible defaults
- **Detailed Reporting**: Comprehensive validation results with specific failure reasons
- **Graceful Degradation**: Quality failures are logged and tracked without pipeline crashes

#### Key Advantages

- **Type Safety**: Full Pydantic validation prevents runtime data errors
- **Data Quality**: Real-time validation ensures only reliable data for ML training
- **Extensibility**: Easy addition of new data sources and quality rules
- **Reliability**: Built-in error handling, retries, and comprehensive quality checks
- **Performance**: Parallel symbol processing with configurable batch sizes
- **Monitoring**: Detailed metadata tracking including quality validation results

### Feature Engineering Module  

**Location**: `src/stock_market_analytics/feature_engineering/`

The feature engineering pipeline leverages **Hamilton** for functional, declarative feature computation:

#### Hamilton Framework Benefits

1. **Functional Programming**: Pure functions enable easy testing and reasoning
2. **Dependency Resolution**: Automatic computation graph generation and optimization
3. **Visualization**: Built-in DAG visualization for pipeline understanding
4. **Caching**: Intelligent memoization prevents redundant computations
5. **Type Safety**: Full type checking and validation throughout the pipeline

#### Design Patterns & Architecture

- **Declarative Features**: Functions define features through clear input/output contracts
- **Composable Transforms**: Small, focused functions that combine into complex features
- **Configuration-Driven**: Centralized configuration system (`config.py`) controls all feature parameters
- **Pipeline Visualization**: Generated diagrams show feature dependencies and computation flow

#### Hamilton DAG Visualization

![Hamilton Feature Engineering Pipeline](src/stock_market_analytics/feature_engineering/features_diagram.png)

The Hamilton framework automatically generates this dependency graph showing the complete feature engineering pipeline. This visualization demonstrates:

- **Data Flow**: Clear progression from raw data through preprocessing to final features
- **Parallel Processing**: Independent feature computations that can run concurrently  
- **Configuration Points**: Parameters from the centralized `config.py` control window sizes and feature selection
- **Dependency Management**: Automatic resolution of feature dependencies and execution order
- **Type Safety**: All configuration parameters are validated through Pydantic models

### Adding New Features

The feature engineering pipeline uses Hamilton for functional, dependency-driven feature computation. To add a new feature, follow these steps:

#### Step 1: Add Feature Function to `feature_pipeline.py`

The feature engineering uses Hamilton framework with functions grouped by category. Add your new feature to the appropriate section in `src/stock_market_analytics/feature_engineering/feature_pipeline.py`.

For example, to add a new volatility feature, add it to the `volatility_features_df` function:

```python
def volatility_features_df(
    interpolated_df: pl.DataFrame,
    short_window: int,
    long_window: int,
) -> pl.DataFrame:
    """
    Compute volatility features using Hamilton framework.
    """
    return interpolated_df.with_columns([
        # ... existing features ...
        
        # Your new feature - use .over("symbol") to ensure no data leakage
        pl.col("log_returns_d")
        .rolling_std(short_window)
        .over("symbol")
        .alias("my_new_volatility_feature"),
    ])
```

**Key Requirements**:
- Add features to the appropriate category function (volatility, momentum, statistical, etc.)
- Return a DataFrame with all features in that category
- Use `.over("symbol")` for all rolling operations to prevent cross-symbol contamination
- Include proper type hints
- Features are automatically included in `df_features` through Hamilton's dependency resolution

#### Step 2: Update Feature Selection in Configuration

The `df_features` function automatically joins all feature categories. To include your new feature in the model, add it to the features list in `config.py`:

```python
# In src/stock_market_analytics/config.py
class ModelingConfig(BaseModel):
    features: list[str] = Field(default=[
        # ... existing features ...
        "my_new_volatility_feature",  # Add your feature name here
    ])
```

#### Step 3: Update Feature Parameters (Optional)

If your feature needs configurable parameters, add them to the centralized configuration in `src/stock_market_analytics/config.py`:

```python
class FeatureEngineeringConfig(BaseModel):
    my_window_size: int = 30  # Add your parameter here
```

#### Why This Structure?

This approach provides several advantages:

1. **Modular Design**: Features are grouped by category (volatility, momentum, etc.)
2. **Hamilton Integration**: Automatic dependency resolution and parallel execution
3. **No Data Leakage**: `.over("symbol")` ensures computations stay within symbols
4. **Type Safety**: Hamilton validates all function signatures and return types
5. **Performance**: Optimized computation graph with efficient data joins
6. **Maintainability**: Clear separation of concerns and easy feature management

#### Hamilton Visualization

After adding features, you can visualize the dependency graph. **Note**: Requires graphviz system package to be installed.

```bash
# Generate visualization of the complete feature pipeline
uv run python -c "
from hamilton import driver
from stock_market_analytics.feature_engineering import feature_pipeline
dr = driver.Builder().with_modules(feature_pipeline).build()
dr.visualize_execution(['df_features'], './features_graph.png', bypass_validation=True)
print('Hamilton dependency graph saved to: ./features_graph.png')
"
```

**Troubleshooting Visualization Issues**:
- If you get `ExecutableNotFound: failed to execute PosixPath('dot')`, install graphviz:
  ```bash
  sudo apt update && sudo apt install -y graphviz  # Ubuntu/Debian
  brew install graphviz                            # macOS
  ```
- Use `bypass_validation=True` to generate the graph without providing input data
- The generated diagram shows all feature dependencies and parallel processing opportunities

#### Updating the Official Feature Diagram

To regenerate the official Hamilton dependency graph shown in this README:

```bash
# Regenerate the official features_diagram.png
uv run python -c "
from hamilton import driver
from stock_market_analytics.feature_engineering import feature_pipeline
import os

# Create Hamilton driver
dr = driver.Builder().with_modules(feature_pipeline).build()

# Generate the visualization (overwrites existing diagram)
output_path = 'src/stock_market_analytics/feature_engineering/features_diagram.png'
dr.visualize_execution(['df_features'], output_path, bypass_validation=True)

print(f'Hamilton dependency graph updated: {output_path}')
print(f'File exists: {os.path.exists(output_path)}')
"
```

This command:
- Overwrites the existing `features_diagram.png` file
- Uses `bypass_validation=True` to avoid needing input data
- Updates the diagram referenced in this README
- Should be run after adding new features to keep documentation current

### Machine Learning Module

**Location**: `src/stock_market_analytics/modeling/`

The modeling module implements production-ready machine learning workflows for quantile regression on stock market data, featuring pre-configured hyperparameters and conformal prediction.

#### Framework & Architecture

The module leverages several enterprise-grade frameworks:

1. **CatBoost**: Gradient boosting framework optimized for:
   - **Multi-quantile regression**: Predicts uncertainty intervals (10th, 25th, 50th, 75th, 90th percentiles)
   - **Large datasets**: Efficient training on time series with hundreds of symbols
   - **Pre-configured parameters**: Optimized hyperparameters based on extensive testing

2. **Conformal Prediction**: Provides statistical guarantees:
   - **Coverage guarantee**: Ensures prediction intervals contain true values with specified probability
   - **Distribution-free**: Works regardless of underlying data distribution
   - **Post-hoc calibration**: Adjusts model predictions without retraining

3. **W&B Integration**: Experiment tracking and monitoring:
   - **Automated logging**: Tracks hyperparameters, metrics, and artifacts
   - **Model versioning**: Maintains history of model iterations
   - **Visualization**: Rich dashboards for experiment comparison

#### Design Patterns & Components

**Configuration-Driven Design**:
```python
# config.py - Centralized type-safe configuration
class ModelingConfig(BaseModel):
    features: list[str] = ["amihud_illiq", "rsi", "momentum", ...]  # Selected features
    quantiles: list[float] = [0.1, 0.25, 0.5, 0.75, 0.9]         # Prediction quantiles
    # ... other parameters with validation
```

**Hamilton Integration**: 
- **Data Processing**: Uses Hamilton for reproducible data splitting and preprocessing
- **Dependency Tracking**: Automatic resolution of data dependencies
- **Type Safety**: Full type checking throughout the pipeline

**Evaluation Framework**:
- **Multi-quantile metrics**: Pinball loss, coverage, interval width
- **Validation**: Time-aware splits preventing lookahead bias  
- **Conformal calibration**: Adjusts prediction intervals for target coverage

#### Key Workflows

##### Model Training (`train-model`)

Trains CatBoost model with optimized pre-configured hyperparameters:

```bash
# Train model with pre-configured hyperparameters  
export BASE_DATA_PATH="/path/to/your/data"
export WANDB_KEY="your_wandb_key"
uv run train-model run
```

**Training Process**:
1. **Data Loading**: Loads engineered features from parquet
2. **Splitting**: Time-aware train/validation/test splits
3. **Training**: Multi-quantile CatBoost with early stopping
4. **Calibration**: Conformal adjustment on validation set
5. **Evaluation**: Coverage and width metrics on test set
6. **Logging**: Results tracked in W&B for reproducibility

**Conformal Prediction Workflow**:
```python
# Get base model predictions (10th, 90th percentiles)
qlo_cal, qhi_cal = model.predict_quantiles(X_calibration)

# Compute conformal adjustment
qconf = conformal_adjustment(qlo_cal, qhi_cal, y_calibration, alpha=0.2)

# Apply to test predictions for 80% coverage guarantee
lo_adjusted, hi_adjusted = apply_conformal(qlo_test, qhi_test, qconf)
```

#### Evaluation Metrics

**Multi-quantile Performance**:
- **Pinball Loss**: Proper scoring rule for quantile predictions
- **Calibration Error**: Measures quantile prediction accuracy
- **Crossing Penalty**: Ensures monotonic quantile ordering

**Interval Quality**:
- **Coverage**: Percentage of true values within prediction intervals
- **Mean Width**: Average interval width (efficiency measure)

**Example Output**:
```
Training Metrics: {
  'loss': 0.0234,
  'pinball_mean': 0.0198,
  'coverage_10_90': 0.847,
  'calibration_error_mean': 0.012
}

Evaluation Metrics: {
  'coverage': 0.801,        # Close to target 80%
  'mean_width': 0.156,      # Tight intervals
  'pinball_loss': 0.0201    # Low prediction error
}
```

#### Environment Requirements

**Required Environment Variables**:
- `BASE_DATA_PATH`: Path to directory containing `stock_history_features.parquet`
- `WANDB_KEY`: Weights & Biases API key for experiment tracking

**Dependencies**:
- **CatBoost**: Multi-quantile regression model
- **Hamilton**: Data processing pipeline  
- **W&B**: Experiment tracking and visualization
- **Metaflow**: Pipeline orchestration and versioning


## CI/CD Pipeline

The project implements a comprehensive **Continuous Integration** pipeline focusing on code quality, security, and reliability:

### Makefile Automation

```makefile
verify: format lint typecheck test security-check
```

The CI process includes:

1. **Code Formatting**: Automated `ruff` formatting for consistent style
2. **Linting**: `ruff` linting catches common issues and enforces best practices  
3. **Type Checking**: `pyright` ensures type safety across the codebase
4. **Testing**: `pytest` with coverage reporting and HTML reports
5. **Security Auditing**: `pip-audit` scans for known vulnerabilities

### Pre-commit Integration

```yaml
repos:
  - repo: local
    hooks:
      - id: make-verify
        name: Run Makefile checks
        entry: uv run make verify
        language: system
        pass_filenames: false
```

**Benefits of this approach**:
- Prevents broken code from entering the repository
- Maintains consistent code quality across all contributions
- Automated security vulnerability detection
- Fast feedback loop for developers

### Quality Metrics

- **Code Coverage**: Comprehensive test coverage with HTML reporting
- **Type Coverage**: 100% type annotation coverage via pyright
- **Security**: Automated vulnerability scanning of all dependencies
- **Performance**: Lint rules optimized for performance-sensitive code

### Future CD Enhancements

- Pending

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for modern Python dependency management and provides a streamlined development experience.

### Prerequisites

- **Python 3.12+**: Required for modern type hints and performance optimizations
- **uv**: Fast Python package installer and resolver
- **graphviz**: Required for Hamilton pipeline visualization (system dependency)

### Quick Setup

```bash
# Clone the repository
git clone <repository-url>
cd stock-market-analytics

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (including development tools)
uv sync

# Install system dependencies for Hamilton visualizations
sudo apt update && sudo apt install -y graphviz  # On Ubuntu/Debian
# brew install graphviz                          # On macOS

# Note: If you encounter "ExecutableNotFound: failed to execute PosixPath('dot')" error,
# it means graphviz system package is not installed or not in PATH
```

### Verification

```bash
# Verify installation by running the full verification suite
uv run make verify
```

## Usage

### Quick Start

1. **Prepare your data**: Place ticker symbols in `/path/to/your/data/tickers.csv` with columns: `Symbol`, `Name`, `Country`, `IPO Year`, `Sector`, `Industry`. You can get a tickers file in the [Nasdaq stock screener](https://www.nasdaq.com/market-activity/stocks/screener?page=1&rows_per_page=25).

2. **Run data collection**:
```bash
# Set data directory (adjust path as needed)
export BASE_DATA_PATH="/path/to/your/data"

# Run the batch collection pipeline
uv run batch-collect run

# If the number of tickers is above 100, run
uv run batch-collect run --max-num-splits <number of tickers>
```
Note: ingesting the full history of 500 tickers takes about 5 minutes with 4 cores, thanks to easy parallelization by Metaflow.

3. **Generate features**:
```bash
# Run feature engineering pipeline
uv run build-features run
```
Note: building the features for 500 tickers on 4 cores takes about 30 seconds thanks to fast vectorial operations in Polars.

4. **Train ML model**:
```bash
# Set up Weights & Biases for experiment tracking
export WANDB_KEY="your_wandb_api_key"

# Train the model with pre-configured parameters
uv run train-model run
```
Note: Model training completes in under 5 minutes and includes conformal calibration for uncertainty quantification.

### Pipeline Execution Examples

#### Data Collection Pipeline

```bash
# Basic execution with default parameters
uv run batch-collect run

# View pipeline structure and steps
uv run batch-collect show

# Run with Metaflow's built-in resume capabilities
uv run batch-collect resume
```

The data collection pipeline will:
- Load ticker symbols from `data/tickers.csv`
- Check existing metadata for incremental updates
- Collect stock data in parallel for each symbol
- Validate and process the collected data
- Output consolidated data to `data/stocks_history.parquet`

#### Feature Engineering Pipeline

```bash
# Execute feature computation pipeline
uv run build-features run
```

The feature pipeline will:
- Load historical stock data
- Compute statistical moments (kurtosis, skewness, mean)
- Calculate technical indicators and momentum features
- Generate time series features with proper validation
- Output feature matrix to `data/stock_history_features.parquet`

#### Machine Learning Pipeline

```bash
# Set up environment for ML training
export BASE_DATA_PATH="/path/to/your/data"
export WANDB_KEY="your_wandb_api_key"

# Train model with pre-configured parameters
uv run train-model run
```

The ML pipeline will:
- **Model Training**:
  - Load engineered features from parquet
  - Split data chronologically (train/validation/test)
  - Train multi-quantile CatBoost regressor
  - Apply conformal prediction calibration
  - Evaluate coverage and prediction quality
  - Save trained model and metrics to W&B

**Expected Results**:
- **Coverage**: ~80% of true values within prediction intervals
- **Pinball Loss**: ~0.02 (quantile prediction error)
- **Training Time**: <5 minutes for complete model training and calibration

### Development Workflow

#### Running Quality Checks

```bash
# Run the complete verification suite
uv run make verify

# Individual checks
uv run make format      # Auto-format code with ruff
uv run make lint        # Check code quality
uv run make typecheck   # Verify type annotations
uv run make test        # Run test suite with coverage
uv run make security-check  # Audit dependencies
```

#### Testing

```bash
# Run all tests with coverage reporting
uv run make test

# View detailed coverage report
uv run make view-test-report  # Opens HTML coverage report
```

#### Pre-commit Setup

```bash
# Install pre-commit hooks (recommended)
uv run pre-commit install

# Now all commits will automatically run quality checks
git commit -m "Your changes"  # Runs make verify automatically
```

### Advanced Usage

#### Custom Configuration

The pipelines support configuration through environment variables and config files, making it easy to run locally or on the cloud:

```bash
# Custom data paths
export BASE_DATA_PATH="/path/to/your/data"

# Feature engineering parameters (modify config.py)
# Adjust window sizes, feature selections, etc. with type safety
```

#### Pipeline Debugging

```bash
# Run with debug mode for detailed logging
uv run batch-collect run --with DEBUG=1

# Resume from specific step (if pipeline fails)
uv run batch-collect resume

# Inspect intermediate artifacts
uv run metaflow <FLOW_NAME> inspect
```

## Development

### Project Standards

This project maintains high code quality standards:

- **Type Safety**: 100% type annotation coverage with pyright
- **Code Quality**: Enforced via ruff linting and formatting
- **Test Coverage**: Comprehensive test suite with coverage reporting
- **Security**: Automated vulnerability scanning
- **Documentation**: Inline docstrings and comprehensive README

### Architecture Decisions

Key design principles driving this implementation:

- **Modularity**: Clear separation between data collection, feature engineering, and ML modeling modules
- **Testability**: Pure functions and dependency injection enable comprehensive testing
- **Scalability**: Metaflow provides seamless local-to-cloud scaling
- **Maintainability**: Strong typing, comprehensive documentation, and automated quality checks
- **Extensibility**: Abstract base classes and configuration-driven design support easy enhancements
