import polars as pl

from stock_market_analytics.data_collection.models.data_quality_rules import (
    DataQualityResult,
    DataQualityRules,
)


class DataQualityValidator:
    """
    Processor that validates data quality for stock market data.

    This processor applies configurable data quality rules to ensure the integrity
    of stock price data before it's stored in the consolidated dataset.
    """

    def __init__(
        self, symbol: str, data: pl.DataFrame, rules: DataQualityRules | None = None
    ):
        self.symbol = symbol
        self.data = data
        self.rules = rules or DataQualityRules()  # Use default rules if none provided
        self.validation_successful = False
        self.validation_result: DataQualityResult | None = None

    def validate(self) -> pl.DataFrame | None:
        """
        Validate the data against configured quality rules.

        Returns:
            pl.DataFrame or None: Validated data if all checks pass, None otherwise
        """
        if self.data.is_empty():
            self.validation_successful = False
            self.validation_result = DataQualityResult(
                is_valid=False, failed_checks=["empty_data"], total_rows_checked=0
            )
            return None

        validation_details = {}
        failed_checks = []
        warnings = []
        rows_with_issues = 0

        # Check 1: Required columns present and non-null (do this first, before other checks)
        if self.rules.check_required_columns:
            check_passed = self._check_required_columns()
            validation_details["required_columns"] = check_passed
            if not check_passed:
                failed_checks.append("required_columns")
                # If required columns are missing, skip other checks
                self.validation_result = DataQualityResult(
                    is_valid=False,
                    failed_checks=failed_checks,
                    warnings=warnings,
                    total_rows_checked=len(self.data),
                    rows_with_issues=0,
                    validation_details=validation_details,
                )
                self.validation_successful = False
                return None

        try:
            # Check 2: Price positivity
            if self.rules.check_price_positivity:
                check_passed, issue_count = self._check_price_positivity()
                validation_details["price_positivity"] = check_passed
                if not check_passed:
                    failed_checks.append("price_positivity")
                    rows_with_issues += issue_count

            # Check 3: High >= Low consistency
            if self.rules.check_high_low_consistency:
                check_passed, issue_count = self._check_high_low_consistency()
                validation_details["high_low_consistency"] = check_passed
                if not check_passed:
                    failed_checks.append("high_low_consistency")
                    rows_with_issues += issue_count

            # Check 4: Volume non-negative
            if self.rules.check_volume_non_negative:
                check_passed, issue_count = self._check_volume_non_negative()
                validation_details["volume_non_negative"] = check_passed
                if not check_passed:
                    failed_checks.append("volume_non_negative")
                    rows_with_issues += issue_count

            # Check 5: OHLC relationships
            if self.rules.check_ohlc_relationships:
                check_passed, issue_count = self._check_ohlc_relationships()
                validation_details["ohlc_relationships"] = check_passed
                if not check_passed:
                    failed_checks.append("ohlc_relationships")
                    rows_with_issues += issue_count

            # Check 6: Extreme price movements
            if self.rules.check_extreme_price_movements:
                check_passed, issue_count, warning_messages = (
                    self._check_extreme_price_movements()
                )
                validation_details["extreme_price_movements"] = check_passed
                if not check_passed:
                    failed_checks.append("extreme_price_movements")
                    rows_with_issues += issue_count
                warnings.extend(warning_messages)

            # Check 7: Minimum data points
            if len(self.data) < self.rules.min_required_data_points:
                validation_details["min_data_points"] = False
                failed_checks.append("min_data_points")
            else:
                validation_details["min_data_points"] = True

            # Create validation result
            is_valid = len(failed_checks) == 0
            self.validation_result = DataQualityResult(
                is_valid=is_valid,
                failed_checks=failed_checks,
                warnings=warnings,
                total_rows_checked=len(self.data),
                rows_with_issues=rows_with_issues,
                validation_details=validation_details,
            )

            self.validation_successful = is_valid
            return self.data if is_valid else None

        except Exception as e:
            self.validation_successful = False
            self.validation_result = DataQualityResult(
                is_valid=False,
                failed_checks=["validation_error"],
                warnings=[f"Validation error: {str(e)}"],
                total_rows_checked=len(self.data),
            )
            return None

    def _check_required_columns(self) -> bool:
        """Check that all required columns are present and have non-null values."""
        required_columns = ["date", "symbol", "open", "high", "low", "close", "volume"]

        # Check if columns exist
        missing_columns = set(required_columns) - set(self.data.columns)
        if missing_columns:
            return False

        # Check for completely null columns (excluding allowed nulls in continuous timeline)
        for col in ["open", "high", "low", "close", "volume"]:
            if col in self.data.columns and self.data[col].is_not_null().sum() == 0:
                return False

        return True

    def _check_price_positivity(self) -> tuple[bool, int]:
        """Check that all prices are positive where not null."""
        price_columns = ["open", "high", "low", "close"]
        issue_count = 0

        for col in price_columns:
            # Count non-null negative or zero values
            negative_count = (
                self.data.filter(pl.col(col).is_not_null())
                .filter(pl.col(col) <= 0)
                .shape[0]
            )
            issue_count += negative_count

        return issue_count == 0, issue_count

    def _check_high_low_consistency(self) -> tuple[bool, int]:
        """Check that high >= low where both are not null."""
        issue_count = (
            self.data.filter(pl.col("high").is_not_null() & pl.col("low").is_not_null())
            .filter(pl.col("high") < pl.col("low") - self.rules.price_tolerance)
            .shape[0]
        )

        return issue_count == 0, issue_count

    def _check_volume_non_negative(self) -> tuple[bool, int]:
        """Check that volume is non-negative where not null."""
        issue_count = (
            self.data.filter(pl.col("volume").is_not_null())
            .filter(pl.col("volume") < 0)
            .shape[0]
        )

        return issue_count == 0, issue_count

    def _check_ohlc_relationships(self) -> tuple[bool, int]:
        """Check that open and close prices are within high/low range."""
        # Check open within high/low range
        open_issues = (
            self.data.filter(
                pl.col("open").is_not_null()
                & pl.col("high").is_not_null()
                & pl.col("low").is_not_null()
            )
            .filter(
                (pl.col("open") > pl.col("high") + self.rules.price_tolerance)
                | (pl.col("open") < pl.col("low") - self.rules.price_tolerance)
            )
            .shape[0]
        )

        # Check close within high/low range
        close_issues = (
            self.data.filter(
                pl.col("close").is_not_null()
                & pl.col("high").is_not_null()
                & pl.col("low").is_not_null()
            )
            .filter(
                (pl.col("close") > pl.col("high") + self.rules.price_tolerance)
                | (pl.col("close") < pl.col("low") - self.rules.price_tolerance)
            )
            .shape[0]
        )

        total_issues = open_issues + close_issues
        return total_issues == 0, total_issues

    def _check_extreme_price_movements(self) -> tuple[bool, int, list[str]]:
        """Check for extreme price movements that might indicate data errors."""
        warnings = []
        issue_count = 0

        # Calculate daily price changes (using close prices)
        data_with_changes = (
            self.data.filter(pl.col("close").is_not_null())
            .sort("date")
            .with_columns(pl.col("close").pct_change().alias("daily_change"))
        )

        if data_with_changes.is_empty():
            return True, 0, warnings

        # Find extreme movements
        extreme_movements = data_with_changes.filter(
            pl.col("daily_change").abs() > self.rules.max_daily_price_change_ratio
        )

        issue_count = extreme_movements.shape[0]

        if issue_count > 0:
            # Create warning messages for extreme movements
            for row in extreme_movements.iter_rows(named=True):
                date = row["date"]
                change = row["daily_change"] * 100  # Convert to percentage
                warnings.append(
                    f"Extreme price movement on {date}: {change:.1f}% change"
                )

        return issue_count == 0, issue_count, warnings

    def get_validation_summary(self) -> str:
        """Get a human-readable summary of the validation results."""
        if not self.validation_result:
            return "No validation performed"

        result = self.validation_result
        summary = f"Data Quality Validation for {self.symbol}:\n"
        summary += f"  Status: {'PASSED' if result.is_valid else 'FAILED'}\n"
        summary += f"  Rows checked: {result.total_rows_checked}\n"
        summary += f"  Rows with issues: {result.rows_with_issues}\n"

        if result.failed_checks:
            summary += f"  Failed checks: {', '.join(result.failed_checks)}\n"

        if result.warnings:
            summary += f"  Warnings: {len(result.warnings)}\n"
            for warning in result.warnings[:3]:  # Show first 3 warnings
                summary += f"    - {warning}\n"
            if len(result.warnings) > 3:
                summary += f"    ... and {len(result.warnings) - 3} more warnings\n"

        return summary
