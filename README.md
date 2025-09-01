# Stock Market Analytics

A production-ready machine learning pipeline for stock market analysis and predictive modeling, showcasing modern MLOps, software engineering, and data science best practices.

## About the Project

This project implements an end-to-end stock market analytics platform designed for predictive modeling of stock prices and market behavior. The system demonstrates enterprise-grade software engineering practices through:

- **Scalable Data Pipelines**: Robust data collection and feature engineering workflows
- **Production Architecture**: Modular design with clear separation of concerns
- **Quality Assurance**: Comprehensive testing, type checking, and code quality controls
- **MLOps Best Practices**: Versioned data flows, reproducible experiments, and automated validation

The platform focuses on creating reliable, maintainable, and scalable infrastructure for financial market analysis, emphasizing code quality over quick prototypes.

## Architecture Overview

```
📁 stock-market-analytics/
├── src/stock_market_analytics/     # Core application code
│   ├── data_collection/            # Data ingestion pipeline
│   └── feature_engineering/       # Feature computation pipeline
├── tests/                          # Comprehensive test suite
├── Makefile                       # Development workflow automation
└── pyproject.toml                  # Dependencies and entry points
```

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
batch-collect = "stock_market_analytics.data_collection.batch_collection_flow:BatchCollectionFlow"
build-features = "stock_market_analytics.feature_engineering.feature_building_flow:FeatureBuildingFlow"
```

This design enables direct execution of complex workflows:
```bash
# Run data collection pipeline
uv run batch-collect run

# Execute feature engineering pipeline  
uv run build-features run
```

### Data Collection Module

**Location**: `src/stock_market_analytics/data_collection/`

The data collection module exemplifies enterprise-grade data engineering patterns:

#### Design Patterns & Architecture

1. **Abstract Base Classes**: `collectors/base.py` defines interfaces ensuring consistent collector implementations
2. **Strategy Pattern**: Pluggable collectors (YFinance, future: Alpha Vantage, IEX) with unified interfaces
3. **Data Validation**: Pydantic models (`models/collection_plans.py`) enforce schema validation and type safety
4. **Timeline Processing**: Intelligent incremental vs. full data refresh logic
5. **Metaflow Orchestration**: Parallel processing with automatic dependency management

#### Key Advantages

- **Type Safety**: Full Pydantic validation prevents runtime data errors
- **Extensibility**: Easy addition of new data sources through base collector interface
- **Reliability**: Built-in error handling, retries, and data quality checks
- **Performance**: Parallel symbol processing with configurable batch sizes
- **Monitoring**: Comprehensive metadata tracking for operational visibility

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
- **Configuration-Driven**: External config files (`features_config.py`) control feature parameters
- **Pipeline Visualization**: Generated diagrams show feature dependencies and computation flow

#### Hamilton DAG Visualization

![Hamilton Feature Engineering Pipeline](src/stock_market_analytics/feature_engineering/features_diagram.png)

The Hamilton framework automatically generates this dependency graph showing the complete feature engineering pipeline. This visualization demonstrates:

- **Data Flow**: Clear progression from raw data through preprocessing to final features
- **Parallel Processing**: Independent feature computations that can run concurrently  
- **Configuration Points**: Input parameters that control window sizes and feature selection
- **Dependency Management**: Automatic resolution of feature dependencies and execution order


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
- **graphviz**: Required for Hamilton pipeline visualization

### Quick Setup

```bash
# Clone the repository
git clone <repository-url>
cd stock-market-analytics

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (including development tools)
uv sync

# Install system dependencies for visualizations
sudo apt install graphviz  # On Ubuntu/Debian
# brew install graphviz    # On macOS
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

# Feature engineering parameters (modify features_config.py)
# Adjust window sizes, feature selections, etc.
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

- **Modularity**: Clear separation between data collection, feature engineering, and future ML modules
- **Testability**: Pure functions and dependency injection enable comprehensive testing
- **Scalability**: Metaflow provides seamless local-to-cloud scaling
- **Maintainability**: Strong typing, comprehensive documentation, and automated quality checks
- **Extensibility**: Abstract base classes and configuration-driven design support easy enhancements
