"""
ibm_watson_deploy.py
----------------------
Deploys the trained best_model.pkl to IBM Watson Machine Learning so it can
serve real-time predictions from the cloud.

PREREQUISITES
--------------
1. pip install ibm-watson-machine-learning
2. An IBM Cloud account with a Watson Machine Learning service instance.
3. Your credentials:
     - IBM Cloud API key
     - WML instance location (e.g. "us-south")
     - Deployment space ID (create one in the Watson Studio / WML console)

USAGE
-----
Set the environment variables below (or edit the CONFIG block), then run:

    export IBM_WATSON_API_KEY="your-api-key"
    export IBM_WATSON_LOCATION="us-south"
    export IBM_WATSON_SPACE_ID="your-space-id"
    python ibm_watson_deploy.py

This will:
    1. Authenticate with IBM Watson Machine Learning.
    2. Store (upload) the scikit-learn pipeline as a WML model asset.
    3. Create an online deployment for real-time scoring.
    4. Print the scoring endpoint URL you can call from the Flask app or
       any other client for cloud-hosted predictions.
"""

import os
import sys
import joblib

try:
    from ibm_watson_machine_learning import APIClient
except ImportError:
    print(
        "The 'ibm-watson-machine-learning' package is not installed.\n"
        "Install it with:\n\n    pip install ibm-watson-machine-learning\n"
    )
    sys.exit(1)

# ----------------------------- CONFIG -----------------------------------
IBM_WATSON_API_KEY = os.environ.get("IBM_WATSON_API_KEY", "<YOUR_IBM_CLOUD_API_KEY>")
IBM_WATSON_LOCATION = os.environ.get("IBM_WATSON_LOCATION", "us-south")
IBM_WATSON_SPACE_ID = os.environ.get("IBM_WATSON_SPACE_ID", "<YOUR_DEPLOYMENT_SPACE_ID>")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")
MODEL_NAME = "credit-card-approval-model"
DEPLOYMENT_NAME = "credit-card-approval-deployment"
# --------------------------------------------------------------------------


def get_wml_credentials():
    return {
        "url": f"https://{IBM_WATSON_LOCATION}.ml.cloud.ibm.com",
        "apikey": IBM_WATSON_API_KEY,
    }


def main():
    if "<YOUR" in IBM_WATSON_API_KEY or "<YOUR" in IBM_WATSON_SPACE_ID:
        print(
            "Please set IBM_WATSON_API_KEY and IBM_WATSON_SPACE_ID "
            "(as environment variables, or directly in this file) before running."
        )
        sys.exit(1)

    print("Authenticating with IBM Watson Machine Learning...")
    client = APIClient(get_wml_credentials())
    client.set.default_space(IBM_WATSON_SPACE_ID)

    print(f"Loading trained pipeline from {MODEL_PATH} ...")
    model = joblib.load(MODEL_PATH)

    # Identify the scikit-learn software specification WML should use.
    sw_spec_uid = client.software_specifications.get_uid_by_name("runtime-23.1-py3.10")

    print("Storing model in Watson Machine Learning repository...")
    model_meta_props = {
        client.repository.ModelMetaNames.NAME: MODEL_NAME,
        client.repository.ModelMetaNames.TYPE: "scikit-learn_1.1",
        client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_uid,
    }
    stored_model_details = client.repository.store_model(
        model=model, meta_props=model_meta_props
    )
    model_uid = client.repository.get_model_id(stored_model_details)
    print(f"Model stored. Model UID: {model_uid}")

    print("Creating online deployment...")
    deployment_meta_props = {
        client.deployments.ConfigurationMetaNames.NAME: DEPLOYMENT_NAME,
        client.deployments.ConfigurationMetaNames.ONLINE: {},
    }
    deployment_details = client.deployments.create(model_uid, meta_props=deployment_meta_props)
    deployment_uid = client.deployments.get_id(deployment_details)

    scoring_url = client.deployments.get_scoring_href(deployment_details)
    print("\n=== Deployment successful ===")
    print(f"Deployment UID : {deployment_uid}")
    print(f"Scoring URL    : {scoring_url}")
    print(
        "\nYou can now send POST requests with applicant feature payloads "
        "to this scoring URL for real-time cloud predictions."
    )


def sample_scoring_payload():
    """Example of the payload structure WML expects for online scoring."""
    return {
        "input_data": [
            {
                "fields": [
                    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN",
                    "AMT_INCOME_TOTAL", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
                    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "FLAG_WORK_PHONE",
                    "FLAG_PHONE", "FLAG_EMAIL", "OCCUPATION_TYPE", "CNT_FAM_MEMBERS",
                    "AGE_YEARS", "IS_PENSIONER", "EMPLOYMENT_YEARS",
                    "INCOME_PER_FAMILY_MEMBER",
                ],
                "values": [
                    ["F", "Y", "Y", 1, 250000, "Working", "Higher education",
                     "Married", "House / apartment", 1, 1, 1, "Core staff", 3,
                     34, 0, 5, 83333.33],
                ],
            }
        ]
    }


if __name__ == "__main__":
    main()
