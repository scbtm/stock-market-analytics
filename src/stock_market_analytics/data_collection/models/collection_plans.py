from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class YFinanceCollectionPlan(BaseModel):
    """
    Pydantic model for YFinance data collection plan with validation logic.

    This model handles validation for YFinance historical data collection parameters,
    ensuring that the combination of parameters is valid according to YFinance API requirements.
    """

    symbol: str = Field(description="The symbol of the stock to collect data for")
    start: str | None = None
    end: str | None = None
    interval: (
        Literal[
            "1m",
            "2m",
            "5m",
            "15m",
            "30m",
            "60m",
            "90m",
            "1h",
            "1d",
            "5d",
            "1wk",
            "1mo",
            "3mo",
        ]
        | None
    ) = None
    period: (
        Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        | None
    ) = None

    @field_validator("start", "end")
    @classmethod
    def validate_date_format(cls, v: str | None) -> str | None:
        """Validate that date strings can be parsed as dates."""
        if v is None:
            return v

        try:
            # Try to parse the date to ensure it's valid
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError as e:
            raise ValueError(f"Date must be in YYYY-MM-DD format, got: {v}") from e

    @model_validator(mode="after")
    def validate_collection_plan(self) -> "YFinanceCollectionPlan":
        """
        Validate the entire collection plan to ensure parameter combinations are valid.

        Validation rules:
        1. At least one parameter must be provided
        2. Either start OR period must be provided (but not both)
        3. end and period cannot both be provided
        4. start must be before end when both are provided
        5. When using start and end, interval should not be provided
           (YFinance will use default interval)
        """

        # Rule 1: At least one parameter must be provided
        if all(v is None for v in [self.start, self.end, self.interval, self.period]):
            raise ValueError(
                "At least one parameter (start, end, interval, or period) must be provided"
            )

        # Rule 2: Either start OR period must be provided, but not both
        if self.start is None and self.period is None:
            raise ValueError("Either start or period must be provided")

        if self.start is not None and self.period is not None:
            raise ValueError("start and period cannot both be provided")

        # Rule 3: end and period cannot both be provided
        if self.end is not None and self.period is not None:
            raise ValueError("end and period cannot both be provided")

        # Rule 4: start must be before end when both are provided
        if self.start is not None and self.end is not None:
            start_dt = datetime.strptime(self.start, "%Y-%m-%d").date()
            end_dt = datetime.strptime(self.end, "%Y-%m-%d").date()

            if start_dt >= end_dt:
                raise ValueError("start must be before end")

        # Rule 5: When using start and end, interval should not be provided
        if (
            self.start is not None
            and self.end is not None
            and self.interval is not None
        ):
            raise ValueError(
                "interval should not be provided when using both start and end"
            )

        return self

    def to_yfinance_params(self) -> dict[str, str]:
        """
        Convert the collection plan to parameters suitable for yfinance.

        Returns:
            dict: Parameters that can be passed to yfinance Ticker.history()
        """
        params = {}
        if self.start is not None:
            params["start"] = self.start
        if self.end is not None:
            params["end"] = self.end
        if self.interval is not None:
            params["interval"] = self.interval
        if self.period is not None:
            params["period"] = self.period

        return params
