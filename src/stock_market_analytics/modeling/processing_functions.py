from typing import Any

import pandas as pd

from stock_market_analytics.modeling.pipeline_components.configs import modeling_config

TARGET = modeling_config["TARGET"]


def split_data(
    df: pd.DataFrame,
    time_span: int,
) -> pd.DataFrame:
    """
    Split the DataFrame chronologically into training, validation, and testing sets
    """

    # Test with the most recent 6 months of data
    df["fold"] = "train"  # Initialize fold column
    df.loc[df["date"] >= df["date"].max() - pd.Timedelta(days=time_span), "fold"] = (
        "test"
    )

    min_test_date = df[df["fold"] == "test"]["date"].min()

    min_val_date = min_test_date - pd.Timedelta(days=time_span)

    # Validation is 6 months prior to test
    df.loc[(df["date"] < min_test_date) & (df["date"] >= min_val_date), "fold"] = (
        "validation"
    )

    return df


def metadata(
    split_data: pd.DataFrame,
) -> dict[str, Any]:
    """
    Get metadata for the training, validation, and test sets.
    """
    training_start, training_end = (
        split_data[split_data["fold"] == "train"]["date"].min(),
        split_data[split_data["fold"] == "train"]["date"].max(),
    )
    validation_start, validation_end = (
        split_data[split_data["fold"] == "validation"]["date"].min(),
        split_data[split_data["fold"] == "validation"]["date"].max(),
    )
    test_start, test_end = (
        split_data[split_data["fold"] == "test"]["date"].min(),
        split_data[split_data["fold"] == "test"]["date"].max(),
    )

    training_n_rows = split_data[split_data["fold"] == "train"].shape[0]
    validation_n_rows = split_data[split_data["fold"] == "validation"].shape[0]
    test_n_rows = split_data[split_data["fold"] == "test"].shape[0]

    # Nice string format
    date_of_run = pd.Timestamp.now()

    columns = split_data.columns.tolist()

    metadata_info = {
        "date_of_run": date_of_run,
        "training_start": training_start,
        "training_end": training_end,
        "training_n_rows": training_n_rows,
        "validation_start": validation_start,
        "validation_end": validation_end,
        "validation_n_rows": validation_n_rows,
        "test_start": test_start,
        "test_end": test_end,
        "test_n_rows": test_n_rows,
        "columns": columns,
    }

    return metadata_info


def modeling_datasets(
    split_data: pd.DataFrame,
    features: list[str],
    target: str = TARGET,
) -> dict[str, Any]:
    """
    Prepare modeling datasets.
    """

    xtrain, ytrain = (
        split_data[split_data["fold"] == "train"][features],
        split_data[split_data["fold"] == "train"][target],
    )
    xval, yval = (
        split_data[split_data["fold"] == "validation"][features],
        split_data[split_data["fold"] == "validation"][target],
    )
    xtest, ytest = (
        split_data[split_data["fold"] == "test"][features],
        split_data[split_data["fold"] == "test"][target],
    )

    modeling_data = {
        "xtrain": xtrain,
        "ytrain": ytrain,
        "xval": xval,
        "yval": yval,
        "xtest": xtest,
        "ytest": ytest,
    }

    return modeling_data
