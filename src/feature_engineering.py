"""
feature_engineering.py
-----------------------
Loads application_record.csv + credit_record.csv, merges them, engineers
features, and derives the binary target label (APPROVED / REJECTED).

Key step (Scenario 2 in the project brief): the multi-class STATUS payment
codes (0-5, C, X) are converted into a BINARY high-risk flag per applicant:
    - Any month with STATUS in {'2','3','4','5'}  (60+ days past due) => risky
    - Otherwise => not risky

An applicant is labelled REJECTED (target = 0) if they are flagged risky,
and APPROVED (target = 1) otherwise.
"""

import numpy as np
import pandas as pd


RISKY_STATUS_CODES = {"2", "3", "4", "5"}


def load_raw_data(app_path: str, credit_path: str):
    app_df = pd.read_csv(app_path)
    credit_df = pd.read_csv(credit_path)
    return app_df, credit_df


def derive_target_label(credit_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multi-class STATUS history per applicant into a binary label.

    Returns a DataFrame indexed by ID with a single column: TARGET
        TARGET = 0 -> Rejected / high risk (had a 60+ day delinquency)
        TARGET = 1 -> Approved / low risk
    """
    credit_df = credit_df.copy()
    credit_df["IS_RISKY_MONTH"] = credit_df["STATUS"].astype(str).isin(RISKY_STATUS_CODES)

    # Require at least 2 delinquent (60+ days past due) months before
    # labelling an applicant high-risk, rather than flagging on a single
    # isolated late payment.
    risky_month_count = credit_df.groupby("ID")["IS_RISKY_MONTH"].sum()
    risk_flag = risky_month_count >= 2
    target = (~risk_flag).astype(int)  # 1 = approved (not risky), 0 = rejected (risky)
    target.name = "TARGET"
    return target.reset_index()


def engineer_features(app_df: pd.DataFrame) -> pd.DataFrame:
    df = app_df.copy()

    # Age in years (DAYS_BIRTH is negative)
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).round(1)

    # Employment years; pensioners carry the sentinel 365243 -> treat as 0 years employed
    df["IS_PENSIONER"] = (df["DAYS_EMPLOYED"] > 0).astype(int)
    df["EMPLOYMENT_YEARS"] = np.where(
        df["DAYS_EMPLOYED"] > 0, 0, (-df["DAYS_EMPLOYED"] / 365.25)
    ).round(1)

    # Income per family member (affordability proxy)
    df["INCOME_PER_FAMILY_MEMBER"] = (df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].clip(lower=1)).round(2)

    # Simple income band feature
    df["INCOME_BAND"] = pd.cut(
        df["AMT_INCOME_TOTAL"],
        bins=[0, 100000, 200000, 350000, np.inf],
        labels=["Low", "Medium", "High", "Very High"],
    )

    # Drop raw day-based / redundant columns no longer needed
    df = df.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED", "FLAG_MOBIL"])

    return df


def build_dataset(app_path: str, credit_path: str) -> pd.DataFrame:
    """Full pipeline: load, engineer features, merge with binary target."""
    app_df, credit_df = load_raw_data(app_path, credit_path)

    features_df = engineer_features(app_df)
    target_df = derive_target_label(credit_df)

    merged = features_df.merge(target_df, on="ID", how="inner")
    merged = merged.drop_duplicates(subset="ID").reset_index(drop=True)
    return merged


FEATURE_COLUMNS = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN",
    "AMT_INCOME_TOTAL", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "FLAG_WORK_PHONE",
    "FLAG_PHONE", "FLAG_EMAIL", "OCCUPATION_TYPE", "CNT_FAM_MEMBERS",
    "AGE_YEARS", "IS_PENSIONER", "EMPLOYMENT_YEARS",
    "INCOME_PER_FAMILY_MEMBER",
]

CATEGORICAL_COLUMNS = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
]

NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]

TARGET_COLUMN = "TARGET"


if __name__ == "__main__":
    dataset = build_dataset("../data/application_record.csv", "../data/credit_record.csv")
    print(dataset.shape)
    print(dataset["TARGET"].value_counts(normalize=True))
    dataset.to_csv("../data/processed_dataset.csv", index=False)
    print("Saved processed_dataset.csv")
