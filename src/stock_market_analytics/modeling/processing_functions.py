import pandas as pd
from catboost import Pool
from hamilton.function_modifiers import extract_fields


@extract_fields(
    dict({"train": pd.DataFrame, "validation": pd.DataFrame, "test": pd.DataFrame})
)
def split_data(
    df: pd.DataFrame,
    time_span: int,
) -> dict[str, pd.DataFrame]:
    """
    Split the DataFrame chronologically into training, validation, and testing sets
    """

    # Test with the most recent 6 months of data
    test = df[df["date"] >= df["date"].max() - pd.Timedelta(days=time_span)]

    min_test_date = test["date"].min()

    # Validation is 6 months prior to test
    validation = df[
        (df["date"] < min_test_date)
        & (df["date"] >= min_test_date - pd.Timedelta(days=time_span))
    ]
    min_val_date = validation["date"].min()

    # Training data is anything before validation
    train = df[df["date"] < min_val_date]

    return dict({"train": train, "validation": validation, "test": test})  # type: ignore


def metadata(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """
    Get metadata for the training, validation, and test sets.
    """
    training_start, training_end = train["date"].min(), train["date"].max()
    validation_start, validation_end = (
        validation["date"].min(),
        validation["date"].max(),
    )
    test_start, test_end = test["date"].min(), test["date"].max()

    training_n_rows = train.shape[0]
    validation_n_rows = validation.shape[0]
    test_n_rows = test.shape[0]

    # Nice string format
    date_of_run = pd.Timestamp.now()

    features = train.columns.tolist()

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
        "features": features,
    }

    return metadata_info


def pools(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> dict[str, Pool]:
    """
    Create CatBoost Pools for training, validation, and test sets.
    """

    xtrain = train[features]
    xvalidation = validation[features]
    xtest = test[features]

    train_pool = Pool(data=xtrain, label=train["y_log_returns"], feature_names=features)
    validation_pool = Pool(
        data=xvalidation, label=validation["y_log_returns"], feature_names=features
    )
    test_pool = Pool(data=xtest, label=test["y_log_returns"], feature_names=features)

    return {
        "train_pool": train_pool,
        "validation_pool": validation_pool,
        "test_pool": test_pool,
    }
