"""
generate_dataset.py
--------------------
Generates a SYNTHETIC dataset that mirrors the structure of the well-known
Kaggle "Credit Card Approval Prediction" dataset (application_record.csv +
credit_record.csv). Use this if you don't already have the real dataset.

If you have the real dataset, simply drop application_record.csv and
credit_record.csv into this `data/` folder and skip running this script.

Run:
    python generate_dataset.py
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N_APPLICANTS = 20000
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_application_record(n=N_APPLICANTS):
    ids = np.arange(5000000, 5000000 + n)

    gender = np.random.choice(["M", "F"], size=n, p=[0.45, 0.55])
    own_car = np.random.choice(["Y", "N"], size=n, p=[0.4, 0.6])
    own_realty = np.random.choice(["Y", "N"], size=n, p=[0.6, 0.4])
    cnt_children = np.random.choice([0, 1, 2, 3, 4], size=n, p=[0.55, 0.2, 0.15, 0.07, 0.03])

    # Income (log-normal for realism)
    income = np.round(np.random.lognormal(mean=11.2, sigma=0.5, size=n), 2)
    income = np.clip(income, 27000, 1600000)

    income_type = np.random.choice(
        ["Working", "Commercial associate", "Pensioner", "State servant", "Student"],
        size=n, p=[0.5, 0.23, 0.17, 0.08, 0.02]
    )
    education_type = np.random.choice(
        ["Secondary / secondary special", "Higher education", "Incomplete higher",
         "Lower secondary", "Academic degree"],
        size=n, p=[0.65, 0.24, 0.07, 0.03, 0.01]
    )
    family_status = np.random.choice(
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        size=n, p=[0.6, 0.18, 0.09, 0.08, 0.05]
    )
    housing_type = np.random.choice(
        ["House / apartment", "With parents", "Municipal apartment",
         "Rented apartment", "Office apartment", "Co-op apartment"],
        size=n, p=[0.72, 0.1, 0.08, 0.06, 0.02, 0.02]
    )

    # Age in days before "today" (negative, as in the original dataset)
    age_years = np.random.normal(loc=42, scale=11, size=n).clip(20, 70)
    days_birth = -(age_years * 365.25).astype(int)

    # Employment: pensioners get a large positive placeholder (365243) like the real dataset
    is_pensioner = income_type == "Pensioner"
    emp_years = np.random.exponential(scale=6, size=n).clip(0, 45)
    days_employed = -(emp_years * 365.25).astype(int)
    days_employed = np.where(is_pensioner, 365243, days_employed)

    flag_mobil = np.ones(n, dtype=int)
    flag_work_phone = np.random.choice([0, 1], size=n, p=[0.75, 0.25])
    flag_phone = np.random.choice([0, 1], size=n, p=[0.6, 0.4])
    flag_email = np.random.choice([0, 1], size=n, p=[0.85, 0.15])

    occupation_type = np.random.choice(
        ["Laborers", "Core staff", "Sales staff", "Managers", "Drivers",
         "High skill tech staff", "Accountants", "Medicine staff",
         "Cooking staff", "Security staff", "Cleaning staff", "Private service staff"],
        size=n
    )

    cnt_fam_members = (cnt_children + np.where(family_status == "Single / not married", 1, 2)).clip(1, 9)

    df = pd.DataFrame({
        "ID": ids,
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": own_car,
        "FLAG_OWN_REALTY": own_realty,
        "CNT_CHILDREN": cnt_children,
        "AMT_INCOME_TOTAL": income,
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education_type,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing_type,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "FLAG_MOBIL": flag_mobil,
        "FLAG_WORK_PHONE": flag_work_phone,
        "FLAG_PHONE": flag_phone,
        "FLAG_EMAIL": flag_email,
        "OCCUPATION_TYPE": occupation_type,
        "CNT_FAM_MEMBERS": cnt_fam_members,
    })
    return df


def generate_credit_record(app_df):
    """
    Simulate monthly credit-history records per applicant.
    STATUS codes (as in the original dataset):
        0  : 1-29 days past due
        1  : 30-59 days past due
        2  : 60-89 days past due
        3  : 90-119 days past due
        4  : 120-149 days past due
        5  : Overdue > 150 days / written off
        C  : paid off that month
        X  : no loan that month
    """
    records = []

    # Build a hidden "risk propensity" that is CORRELATED with real financial
    # signals (income, employment length, age, dependents) so the trained
    # models have genuine patterns to learn, rather than pure noise.
    income_norm = (app_df["AMT_INCOME_TOTAL"] - app_df["AMT_INCOME_TOTAL"].min()) / (
        app_df["AMT_INCOME_TOTAL"].max() - app_df["AMT_INCOME_TOTAL"].min()
    )
    emp_years_tmp = np.where(app_df["DAYS_EMPLOYED"] > 0, 25, -app_df["DAYS_EMPLOYED"] / 365.25)
    emp_norm = np.clip(emp_years_tmp / 25, 0, 1)
    age_years_tmp = -app_df["DAYS_BIRTH"] / 365.25
    age_norm = np.clip((age_years_tmp - 20) / 50, 0, 1)
    children_penalty = np.clip(app_df["CNT_CHILDREN"] / 4, 0, 1)

    risk_score = (
        0.45 * (1 - income_norm)
        + 0.30 * (1 - emp_norm)
        + 0.15 * (1 - age_norm)
        + 0.10 * children_penalty
    )
    noise = np.random.normal(0, 0.12, size=len(app_df))
    risk_propensity = np.clip(risk_score + noise, 0.02, 0.95).to_numpy() if hasattr(risk_score + noise, "to_numpy") else np.clip(risk_score + noise, 0.02, 0.95)

    for idx, (_, row) in enumerate(app_df.iterrows()):
        n_months = np.random.randint(6, 30)
        p_bad = risk_propensity[idx]
        # Strongly separate low-risk vs high-risk applicants: square the
        # propensity so only genuinely risky profiles accumulate delinquent
        # months at meaningful rates.
        p_risky_month = 0.02 + 0.20 * (p_bad ** 2)
        for m in range(n_months):
            months_balance = -m
            r = np.random.rand()
            if r < p_risky_month:
                # A delinquent month; severity also scales with p_bad
                sev = np.random.rand()
                if sev < 0.4:
                    status = "2"
                elif sev < 0.7:
                    status = "3"
                elif sev < 0.9:
                    status = "4"
                else:
                    status = "5"
            elif r < p_risky_month + 0.05:
                status = "1"
            elif r < p_risky_month + 0.10:
                status = "0"
            elif r < p_risky_month + 0.55:
                status = "C"
            else:
                status = "X"
            records.append((row["ID"], months_balance, status))

    return pd.DataFrame(records, columns=["ID", "MONTHS_BALANCE", "STATUS"])


if __name__ == "__main__":
    app_df = generate_application_record()
    credit_df = generate_credit_record(app_df)

    app_path = os.path.join(OUT_DIR, "application_record.csv")
    credit_path = os.path.join(OUT_DIR, "credit_record.csv")

    app_df.to_csv(app_path, index=False)
    credit_df.to_csv(credit_path, index=False)

    print(f"Saved {len(app_df)} applicant records -> {app_path}")
    print(f"Saved {len(credit_df)} credit history rows -> {credit_path}")
