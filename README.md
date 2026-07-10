# Credit Card Approval Prediction

Automates the credit card approval decision using machine learning. The
system evaluates an applicant's financial and demographic profile and
predicts **Approved** or **Rejected**, the way a bank credit analyst would.

Four classifiers are trained and compared: **Logistic Regression**,
**Decision Tree**, **Random Forest**, and **XGBoost** (Gradient Boosting).
The best-performing model is saved and served through a **Flask** web
application, with an optional **IBM Watson Machine Learning** deployment
pipeline for cloud hosting.

---

## Project Structure

```
CreditCardApprovalPrediction/
├── README.md
├── requirements.txt
├── data/
│   ├── generate_dataset.py        # Creates a synthetic dataset (schema-matched to the
│   │                               # Kaggle "Credit Card Approval Prediction" dataset)
│   ├── application_record.csv     # Applicant demographic / financial data
│   ├── credit_record.csv          # Monthly payment-status history
│   └── processed_dataset.csv      # Output of the feature engineering step
├── notebooks/
│   └── Credit_Card_Approval_Prediction.ipynb   # Full EDA + training walkthrough
├── src/
│   ├── feature_engineering.py     # Merges data, engineers features, derives binary target
│   ├── data_preprocessing.py      # ColumnTransformer (scaling + one-hot encoding)
│   └── train_models.py            # Trains & compares all 4 models, saves the best one
├── models/
│   ├── best_model.pkl             # Saved best pipeline (preprocessing + classifier)
│   ├── best_model_info.json       # Which model won, and its metrics
│   ├── model_comparison.csv
│   └── model_comparison.png       # Bar chart comparing all 4 models
├── flask_app/
│   ├── app.py                     # Flask web application
│   ├── templates/
│   │   ├── index.html             # Applicant data entry form
│   │   └── result.html            # Prediction result page
│   ├── static/
│   │   └── style.css
│   └── model/
│       └── best_model.pkl         # Copy of the best model used by the app
└── deployment/
    └── ibm_watson_deploy.py       # IBM Watson ML cloud deployment script
```

---

## 1. Setup

```bash
# (Recommended) create a virtual environment first
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**System requirements:** Python 3.8+, 4 GB RAM minimum (8 GB recommended),
~2 GB free disk space, internet connection for package installs and cloud
deployment. Anaconda Navigator + Jupyter Notebook recommended for the
notebook; VS Code or PyCharm for the Flask code.

---

## 2. Get the Data

No dataset was supplied with this project, so a **synthetic dataset**
(same schema as the popular Kaggle "Credit Card Approval Prediction"
dataset: `application_record.csv` + `credit_record.csv`) is included and
was used to build and validate the full pipeline end-to-end.

- To regenerate it (or produce a fresh sample): `python data/generate_dataset.py`
- To use the **real** Kaggle dataset instead, download it from
  https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction
  and drop `application_record.csv` and `credit_record.csv` into `data/`
  (same column names — the pipeline will work unchanged).

---

## 3. Explore & Train (Notebook)

```bash
jupyter notebook notebooks/Credit_Card_Approval_Prediction.ipynb
```

Walks through EDA, feature engineering (including the binary conversion of
multi-class payment-status codes described in Scenario 2), preprocessing,
training all 4 models, comparing them, and saving the best one.

## 3b. Train from the Command Line

```bash
cd src
python train_models.py
```

This prints a metrics comparison table for all 4 models, saves
`models/model_comparison.png` + `.csv`, and writes the best model to both
`models/best_model.pkl` and `flask_app/model/best_model.pkl`.

> **Note on XGBoost:** if the `xgboost` package isn't installed, the script
> automatically substitutes scikit-learn's `GradientBoostingClassifier` so
> the pipeline still runs end-to-end. Install `xgboost` for the real
> gradient-boosted-tree implementation.

---

## 4. Run the Web Application

```bash
cd flask_app
python app.py
```

Open **http://127.0.0.1:5000** in your browser. Fill in the applicant's
details (income, employment history, credit history related fields,
demographics) and submit to get an instant **Approved / Rejected**
prediction with a confidence score.

### Usage Scenarios
- **Scenario 1 — Automated Screening:** a credit analyst enters an
  applicant's profile and gets an instant prediction to prioritize review.
- **Scenario 2 — High-Risk Identification:** the feature-engineering
  pipeline (`src/feature_engineering.py`) converts multi-class payment
  status history into a binary high-risk flag, so applicants with a
  pattern of 60+ day delinquencies are clearly classified as high risk.
- **Scenario 3 — Self-Service Eligibility Check:** a prospective customer
  uses the same form to check their own eligibility before formally
  applying.

---

## 5. (Optional) Deploy to IBM Watson Machine Learning

```bash
pip install ibm-watson-machine-learning

export IBM_WATSON_API_KEY="your-ibm-cloud-api-key"
export IBM_WATSON_LOCATION="us-south"        # or your region
export IBM_WATSON_SPACE_ID="your-deployment-space-id"

python deployment/ibm_watson_deploy.py
```

This stores `models/best_model.pkl` as a WML model asset, creates an online
deployment, and prints a scoring URL you can call for cloud-hosted,
real-time predictions (e.g. from the Flask app or any other client).

---

## Model Comparison (on the included synthetic dataset)

See `models/model_comparison.png` / `models/model_comparison.csv` for the
full table — it's regenerated every time you run `train_models.py`.
Accuracy, precision, recall, F1-score, and ROC-AUC are reported for each
of the four algorithms, and the model with the highest F1-score is
selected automatically as `best_model.pkl`.

**Note:** because the included data is synthetic, the absolute metric
values are illustrative rather than production-grade. Swap in the real
dataset (see step 2) for meaningful results.

---

## Skills / Tech Stack

XGBoost · Machine Learning Algorithms · Decision Tree Learning ·
NumPy · Python · Scikit-Learn · Matplotlib · Flask · IBM Watson Machine
Learning
