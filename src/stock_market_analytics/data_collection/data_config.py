# Constants
TICKERS_FILE = "tickers.csv"
METADATA_FILE = "metadata.csv"
STOCKS_HISTORY_FILE = "stocks_history.parquet"

# Required columns for tickers file
REQUIRED_TICKER_COLUMNS = [
    "Symbol",
    "Name",
    "Country",
    "IPO Year",
    "Sector",
    "Industry",
]
TICKER_COLUMN_MAPPING = {
    col: col.replace(" ", "_").lower() for col in REQUIRED_TICKER_COLUMNS
}

# Required columns for metadata file
REQUIRED_METADATA_COLUMNS = ["symbol", "last_ingestion", "max_date_recorded", "status"]

data_config = dict(
    {
        "TICKERS_FILE": TICKERS_FILE,
        "METADATA_FILE": METADATA_FILE,
        "STOCKS_HISTORY_FILE": STOCKS_HISTORY_FILE,
        "REQUIRED_TICKER_COLUMNS": REQUIRED_TICKER_COLUMNS,
        "TICKER_COLUMN_MAPPING": TICKER_COLUMN_MAPPING,
        "REQUIRED_METADATA_COLUMNS": REQUIRED_METADATA_COLUMNS,
    }
)