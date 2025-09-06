from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stock_market_analytics.modeling.pipeline_components.configs import (
    cb_model_params,
    modeling_config,
    pca_group_params,
)
from stock_market_analytics.modeling.pipeline_components.naive_baselines import (
    HistoricalQuantileBaseline,
)
from stock_market_analytics.modeling.pipeline_components.predictors import (
    CatBoostMultiQuantileModel,
)

# ------------------------------- PCA for feature groups -------------------------------
feature_groups = modeling_config["FEATURE_GROUPS"]

pca_groups = {
    group.lower() + "_pca": PCA(**pca_group_params[group]) for group in feature_groups
}

scaler_groups = {
    group.lower() + "_scaler": Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    for group in feature_groups
}

# ------------------------------- Transformation Pipeline -------------------------------
scalers = ColumnTransformer(
    transformers=[
        (name, scaler, feature_groups[group])
        for group, name, scaler in zip(
            feature_groups, scaler_groups.keys(), scaler_groups.values(), strict=False
        )
    ],
    remainder="drop",  # Drop any features not specified in the transformers
    verbose_feature_names_out="{feature_name}__scaled",
    verbose=False,
).set_output(transform="pandas")

dimensionality_reducers = ColumnTransformer(
    transformers=[
        (
            name,
            pca,
            [f"{feature_name}__scaled" for feature_name in feature_groups[group]],
        )
        for group, name, pca in zip(
            feature_groups, pca_groups.keys(), pca_groups.values(), strict=False
        )
    ],
    remainder="drop",  # Drop any features not specified in the transformers
    verbose_feature_names_out=True,
    verbose=False,
).set_output(transform="pandas")

transformation_pipeline = Pipeline(
    steps=[("scalers", scalers), ("dimensionality_reducers", dimensionality_reducers)]
)

# ------------------------------- Full Pipeline -------------------------------

QUANTILES = modeling_config["QUANTILES"]

quantile_regressor = CatBoostMultiQuantileModel(quantiles=QUANTILES, **cb_model_params)

# transformation_pipeline = Pipeline(steps=[
#     ("scaler", StandardScaler()),
#     ("pca", PCA(**pca_params))
# ])


pipeline = Pipeline(
    steps=[
        ("transformations", transformation_pipeline),
        ("quantile_regressor", quantile_regressor),
    ]
)

# Baseline pipelines for comparison
baseline_pipelines = {
    "historical": Pipeline(
        steps=[
            ("transformations", transformation_pipeline),
            ("quantile_regressor", HistoricalQuantileBaseline(quantiles=QUANTILES)),
        ]
    ),
    # "linear": Pipeline(steps=[
    #     ("transformations", transformation_pipeline),
    #     ("quantile_regressor", LinearQuantileBaseline(quantiles=QUANTILES))
    # ]),
    # "ensemble": Pipeline(steps=[
    #     ("transformations", transformation_pipeline),
    #     ("quantile_regressor", NaiveQuantileEnsemble(quantiles=QUANTILES))
    # ])
}


def get_pipeline(model_type: str = "catboost") -> Pipeline:
    """
    Returns the machine learning pipeline for stock market analytics.

    Parameters
    ----------
    model_type : str, default="catboost"
        Type of model to use. Options:
        - "catboost": CatBoost MultiQuantile Regressor (default)
        - "historical": Historical quantile baseline
        - "linear": Linear quantile regression baseline
        - "ensemble": Ensemble of naive baselines

    Returns
    -------
    Pipeline: A scikit-learn Pipeline object.
    """
    if model_type == "catboost":
        return pipeline
    elif model_type in baseline_pipelines:
        return baseline_pipelines[model_type]
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Available options: 'catboost', 'historical', 'linear', 'ensemble'"
        )


def get_baseline_pipeline(baseline_type: str = "ensemble") -> Pipeline:
    """
    Returns a baseline pipeline for comparison with the main CatBoost model.

    Parameters
    ----------
    baseline_type : str, default="ensemble"
        Type of baseline. Options: "historical", "linear", "ensemble"

    Returns
    -------
    Pipeline: A baseline pipeline for comparison.
    """
    if baseline_type not in baseline_pipelines:
        raise ValueError(
            f"Unknown baseline_type: {baseline_type}. "
            f"Available options: {list(baseline_pipelines.keys())}"
        )
    return baseline_pipelines[baseline_type]
