"""
app.py
-------
Flask web application for real-time credit card approval prediction.

Run:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model", "best_model.pkl")

app = Flask(__name__)

# Load the trained pipeline (preprocessing + classifier) once at startup
model = joblib.load(MODEL_PATH)

INCOME_TYPES = ["Working", "Commercial associate", "Pensioner", "State servant", "Student"]
EDUCATION_TYPES = ["Secondary / secondary special", "Higher education", "Incomplete higher",
                    "Lower secondary", "Academic degree"]
FAMILY_STATUSES = ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"]
HOUSING_TYPES = ["House / apartment", "With parents", "Municipal apartment",
                  "Rented apartment", "Office apartment", "Co-op apartment"]
OCCUPATION_TYPES = ["Laborers", "Core staff", "Sales staff", "Managers", "Drivers",
                     "High skill tech staff", "Accountants", "Medicine staff",
                     "Cooking staff", "Security staff", "Cleaning staff", "Private service staff"]


def build_feature_row(form) -> pd.DataFrame:
    """Convert user-friendly form inputs into the exact engineered feature
    row the trained pipeline expects (mirrors src/feature_engineering.py)."""

    gender = form.get("gender")
    own_car = form.get("own_car")
    own_realty = form.get("own_realty")
    cnt_children = int(form.get("cnt_children", 0))
    income = float(form.get("income"))
    income_type = form.get("income_type")
    education_type = form.get("education_type")
    family_status = form.get("family_status")
    housing_type = form.get("housing_type")
    flag_work_phone = int(form.get("flag_work_phone", 0))
    flag_phone = int(form.get("flag_phone", 0))
    flag_email = int(form.get("flag_email", 0))
    occupation_type = form.get("occupation_type")
    cnt_fam_members = int(form.get("cnt_fam_members", 1))
    age_years = float(form.get("age_years"))
    employment_years = float(form.get("employment_years", 0))
    is_pensioner = 1 if income_type == "Pensioner" else 0
    if is_pensioner:
        employment_years = 0.0

    income_per_family_member = round(income / max(cnt_fam_members, 1), 2)

    row = {
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": own_car,
        "FLAG_OWN_REALTY": own_realty,
        "CNT_CHILDREN": cnt_children,
        "AMT_INCOME_TOTAL": income,
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education_type,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing_type,
        "FLAG_WORK_PHONE": flag_work_phone,
        "FLAG_PHONE": flag_phone,
        "FLAG_EMAIL": flag_email,
        "OCCUPATION_TYPE": occupation_type,
        "CNT_FAM_MEMBERS": cnt_fam_members,
        "AGE_YEARS": age_years,
        "IS_PENSIONER": is_pensioner,
        "EMPLOYMENT_YEARS": employment_years,
        "INCOME_PER_FAMILY_MEMBER": income_per_family_member,
    }
    return pd.DataFrame([row])


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        income_types=INCOME_TYPES,
        education_types=EDUCATION_TYPES,
        family_statuses=FAMILY_STATUSES,
        housing_types=HOUSING_TYPES,
        occupation_types=OCCUPATION_TYPES,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        X = build_feature_row(request.form)
        proba = model.predict_proba(X)[0]
        pred = int(model.predict(X)[0])  # 1 = Approved, 0 = Rejected

        result = "Approved" if pred == 1 else "Rejected"
        confidence = round(float(proba[pred]) * 100, 2)

        return render_template(
            "result.html",
            result=result,
            confidence=confidence,
            approved=(pred == 1),
        )
    except Exception as exc:
        return render_template("result.html", error=str(exc)), 400


if __name__ == "__main__":
    app.run(debug=True)
