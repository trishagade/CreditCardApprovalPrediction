"""
data_preprocessing.py
----------------------
Builds the preprocessing pipeline (imputation + scaling for numeric columns,
one-hot encoding for categorical columns) and a helper to split the dataset.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from feature_engineering import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_COLUMNS),
        ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
    ])
    return preprocessor


def get_train_test_split(df, test_size=0.2, random_state=42):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
