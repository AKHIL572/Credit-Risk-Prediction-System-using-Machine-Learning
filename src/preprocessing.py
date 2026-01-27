from typing import Tuple, List

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def get_feature_types(
    X: pd.DataFrame
) -> Tuple[List[str], List[str]]:
    """
    Separate numeric and categorical columns.

    Parameters
    ----------
    X : pd.DataFrame

    Returns
    -------
    num_cols : List[str]
    cat_cols : List[str]
    """
    num_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    cat_cols = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    return num_cols, cat_cols


def build_preprocessor(
    num_cols: List[str],
    cat_cols: List[str]
) -> ColumnTransformer:
    """
    Build preprocessing pipeline.

    Numeric:
    - median imputation
    - standard scaling

    Categorical:
    - most frequent imputation
    - one-hot encoding (sparse)

    Returns
    -------
    ColumnTransformer
    """

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols)
        ],
        remainder="drop"
    )

    return preprocessor
