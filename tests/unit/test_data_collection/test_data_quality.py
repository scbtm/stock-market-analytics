import pytest
import polars as pl
from datetime import date

from stock_market_analytics.data_collection.processors.data_quality import DataQualityValidator
from stock_market_analytics.data_collection.models.data_quality_rules import (
    DataQualityRules,
    DataQualityResult,
)


class TestDataQualityValidator:
    """Test cases for DataQualityValidator class."""

    def create_valid_sample_data(self) -> pl.DataFrame:
        """Create a valid sample stock data DataFrame for testing."""
        return pl.DataFrame({
            "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "open": [100.0, 102.0, 101.0],
            "high": [105.0, 104.0, 103.0],
            "low": [98.0, 99.0, 100.0],
            "close": [102.0, 101.0, 102.0],
            "volume": [1000000, 1200000, 800000],
        })

    def test_valid_data_passes_all_checks(self):
        """Test that valid data passes all quality checks."""
        data = self.create_valid_sample_data()
        validator = DataQualityValidator("AAPL", data)
        
        result_data = validator.validate()
        
        assert result_data is not None
        assert validator.validation_successful is True
        assert validator.validation_result is not None
        assert validator.validation_result.is_valid is True
        assert len(validator.validation_result.failed_checks) == 0

    def test_empty_data_fails_validation(self):
        """Test that empty data fails validation."""
        empty_data = pl.DataFrame(schema={
            "date": pl.Date,
            "symbol": pl.Utf8,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
        })
        
        validator = DataQualityValidator("AAPL", empty_data)
        result_data = validator.validate()
        
        assert result_data is None
        assert validator.validation_successful is False
        assert "empty_data" in validator.validation_result.failed_checks

    def test_high_less_than_low_fails_validation(self):
        """Test that data where high < low fails validation."""
        data = self.create_valid_sample_data()
        # Make high less than low for one row
        data = data.with_columns(
            pl.when(pl.col("date") == date(2023, 1, 2))
            .then(95.0)  # low is 99.0, so this violates high >= low
            .otherwise(pl.col("high"))
            .alias("high")
        )
        
        validator = DataQualityValidator("AAPL", data)
        result_data = validator.validate()
        
        assert result_data is None
        assert validator.validation_successful is False
        assert "high_low_consistency" in validator.validation_result.failed_checks
        assert validator.validation_result.rows_with_issues > 0

    def test_negative_prices_fail_validation(self):
        """Test that negative prices fail validation."""
        data = self.create_valid_sample_data()
        # Make one price negative
        data = data.with_columns(
            pl.when(pl.col("date") == date(2023, 1, 2))
            .then(-10.0)
            .otherwise(pl.col("open"))
            .alias("open")
        )
        
        validator = DataQualityValidator("AAPL", data)
        result_data = validator.validate()
        
        assert result_data is None
        assert validator.validation_successful is False
        assert "price_positivity" in validator.validation_result.failed_checks

    def test_negative_volume_fails_validation(self):
        """Test that negative volume fails validation."""
        data = self.create_valid_sample_data()
        # Make volume negative
        data = data.with_columns(
            pl.when(pl.col("date") == date(2023, 1, 2))
            .then(-1000)
            .otherwise(pl.col("volume"))
            .alias("volume")
        )
        
        validator = DataQualityValidator("AAPL", data)
        result_data = validator.validate()
        
        assert result_data is None
        assert validator.validation_successful is False
        assert "volume_non_negative" in validator.validation_result.failed_checks

    def test_ohlc_relationships_validation(self):
        """Test that open/close prices outside high/low range fail validation."""
        data = self.create_valid_sample_data()
        # Make open price higher than high price
        data = data.with_columns(
            pl.when(pl.col("date") == date(2023, 1, 2))
            .then(110.0)  # high is 104.0, so this violates open <= high
            .otherwise(pl.col("open"))
            .alias("open")
        )
        
        validator = DataQualityValidator("AAPL", data)
        result_data = validator.validate()
        
        assert result_data is None
        assert validator.validation_successful is False
        assert "ohlc_relationships" in validator.validation_result.failed_checks

    def test_extreme_price_movements_detected(self):
        """Test that extreme price movements are detected."""
        data = self.create_valid_sample_data()
        # Create extreme price movement (1000% increase)
        data = data.with_columns(
            pl.when(pl.col("date") == date(2023, 1, 2))
            .then(1000.0)  # 900% increase from 102.0
            .otherwise(pl.col("close"))
            .alias("close")
        )
        
        # Set low max_daily_price_change_ratio to trigger the check
        rules = DataQualityRules(max_daily_price_change_ratio=5.0)
        validator = DataQualityValidator("AAPL", data, rules)
        result_data = validator.validate()
        
        assert result_data is None
        assert validator.validation_successful is False
        assert "extreme_price_movements" in validator.validation_result.failed_checks
        assert len(validator.validation_result.warnings) > 0

    def test_missing_required_columns_fails_validation(self):
        """Test that data missing required columns fails validation."""
        # Create data without 'close' column
        incomplete_data = pl.DataFrame({
            "date": [date(2023, 1, 1)],
            "symbol": ["AAPL"],
            "open": [100.0],
            "high": [105.0],
            "low": [98.0],
            # Missing 'close' and 'volume'
        })
        
        validator = DataQualityValidator("AAPL", incomplete_data)
        result_data = validator.validate()
        
        assert result_data is None
        assert validator.validation_successful is False
        assert "required_columns" in validator.validation_result.failed_checks

    def test_minimum_data_points_validation(self):
        """Test that insufficient data points fail validation."""
        data = self.create_valid_sample_data().head(1)  # Only one row
        
        rules = DataQualityRules(min_required_data_points=3)
        validator = DataQualityValidator("AAPL", data, rules)
        result_data = validator.validate()
        
        assert result_data is None
        assert validator.validation_successful is False
        assert "min_data_points" in validator.validation_result.failed_checks

    def test_custom_rules_configuration(self):
        """Test that custom rules configuration works correctly."""
        data = self.create_valid_sample_data()
        
        # Create rules that disable some checks
        rules = DataQualityRules(
            check_high_low_consistency=False,
            check_volume_non_negative=False,
            min_required_data_points=1
        )
        
        validator = DataQualityValidator("AAPL", data, rules)
        result_data = validator.validate()
        
        assert result_data is not None
        assert validator.validation_successful is True
        
        # These checks should not appear in validation_details since they're disabled
        assert "high_low_consistency" not in validator.validation_result.validation_details
        assert "volume_non_negative" not in validator.validation_result.validation_details

    def test_data_with_nulls_validates_correctly(self):
        """Test that data with null values (from continuous timeline) validates correctly."""
        data = pl.DataFrame({
            "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "open": [100.0, None, 101.0],  # Null value (weekend/holiday)
            "high": [105.0, None, 103.0],
            "low": [98.0, None, 100.0],
            "close": [102.0, None, 102.0],
            "volume": [1000000, None, 800000],
        })
        
        validator = DataQualityValidator("AAPL", data)
        result_data = validator.validate()
        
        # Should pass validation despite null values
        assert result_data is not None
        assert validator.validation_successful is True

    def test_validation_summary_generation(self):
        """Test that validation summary is generated correctly."""
        data = self.create_valid_sample_data()
        validator = DataQualityValidator("AAPL", data)
        validator.validate()
        
        summary = validator.get_validation_summary()
        
        assert "AAPL" in summary
        assert "PASSED" in summary
        assert "Rows checked: 3" in summary

    def test_failed_validation_summary(self):
        """Test validation summary for failed validation."""
        data = self.create_valid_sample_data()
        # Make data invalid
        data = data.with_columns(pl.lit(-10.0).alias("open"))  # Negative price
        
        validator = DataQualityValidator("AAPL", data)
        validator.validate()
        
        summary = validator.get_validation_summary()
        
        assert "FAILED" in summary
        assert "price_positivity" in summary

    def test_price_tolerance_setting(self):
        """Test that price tolerance setting works correctly."""
        # Create data where high is very slightly less than low
        data = pl.DataFrame({
            "date": [date(2023, 1, 1)],
            "symbol": ["AAPL"], 
            "open": [100.0],
            "high": [99.999999],  # Slightly less than low
            "low": [100.0],
            "close": [100.0],
            "volume": [1000000],
        })
        
        # Should pass with lenient tolerance
        lenient_rules = DataQualityRules(price_tolerance=1e-5)
        validator = DataQualityValidator("AAPL", data, lenient_rules)
        result_data = validator.validate()
        assert result_data is not None
        
        # Should fail with very strict tolerance
        strict_rules = DataQualityRules(price_tolerance=1e-8)
        validator_strict = DataQualityValidator("AAPL", data, strict_rules)
        result_data_strict = validator_strict.validate()
        assert result_data_strict is None

    def test_validation_result_structure(self):
        """Test that DataQualityResult structure is correct."""
        data = self.create_valid_sample_data()
        validator = DataQualityValidator("AAPL", data)
        validator.validate()
        
        result = validator.validation_result
        
        assert isinstance(result, DataQualityResult)
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.failed_checks, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.total_rows_checked, int)
        assert isinstance(result.rows_with_issues, int)
        assert isinstance(result.validation_details, dict)
        
        assert result.total_rows_checked == 3
        assert result.rows_with_issues == 0
        assert result.is_valid is True

    def test_exception_handling_in_validation(self):
        """Test that validation handles exceptions gracefully."""
        # Create data with correct columns but that will cause validation errors
        malformed_data = pl.DataFrame({
            "date": [date(2023, 1, 1)],
            "symbol": ["AAPL"],
            "open": [float("nan")],  # NaN values can cause issues
            "high": [float("inf")],  # Infinite values can cause issues
            "low": [98.0],
            "close": [102.0],
            "volume": [1000000],
        })
        
        validator = DataQualityValidator("AAPL", malformed_data)
        result_data = validator.validate()
        
        # This should either fail validation due to invalid values or handle gracefully
        assert result_data is None
        assert validator.validation_successful is False
        assert validator.validation_result is not None
        # Should fail on some validation check (could be price_positivity or other checks)
        assert len(validator.validation_result.failed_checks) > 0