from typing import Literal

from pydantic import BaseModel, Field


class DataQualityRules(BaseModel):
    """
    Pydantic model defining data quality validation rules for stock market data.

    This model specifies the quality checks that should be applied to stock price data
    to ensure data integrity and catch potential issues before data is stored.
    """

    # Price consistency rules
    check_high_low_consistency: bool = Field(
        default=True,
        description="Ensure high price is greater than or equal to low price",
    )

    check_price_positivity: bool = Field(
        default=True,
        description="Ensure all prices (open, high, low, close) are positive",
    )

    check_volume_non_negative: bool = Field(
        default=True, description="Ensure volume is non-negative"
    )

    # Price range validation
    check_extreme_price_movements: bool = Field(
        default=True,
        description="Flag potential data errors from extreme price movements",
    )

    max_daily_price_change_ratio: float = Field(
        default=10.0,
        description="Maximum allowed daily price change ratio (e.g., 10.0 means 1000% change)",
        gt=0.0,
    )

    # Data completeness rules
    check_required_columns: bool = Field(
        default=True, description="Ensure all required columns are present and non-null"
    )

    min_required_data_points: int = Field(
        default=1,
        description="Minimum number of data points required for validation",
        ge=1,
    )

    # Price relationship rules
    check_ohlc_relationships: bool = Field(
        default=True,
        description="Validate that open/close prices are within high/low range",
    )

    # Action on quality check failure
    on_failure: Literal["exclude", "warn", "fix"] = Field(
        default="exclude",
        description="Action to take when quality checks fail: exclude data, warn only, or attempt to fix",
    )

    # Tolerance settings
    price_tolerance: float = Field(
        default=1e-8,
        description="Tolerance for floating point price comparisons",
        gt=0.0,
    )


class DataQualityResult(BaseModel):
    """
    Result of data quality validation containing validation status and details.
    """

    is_valid: bool = Field(description="Whether the data passed all quality checks")

    failed_checks: list[str] = Field(
        default_factory=list, description="List of failed validation check names"
    )

    warnings: list[str] = Field(
        default_factory=list, description="List of validation warnings"
    )

    total_rows_checked: int = Field(
        default=0, description="Total number of rows that were validated"
    )

    rows_with_issues: int = Field(
        default=0, description="Number of rows that had quality issues"
    )

    validation_details: dict[str, bool] = Field(
        default_factory=dict, description="Detailed results for each validation rule"
    )
