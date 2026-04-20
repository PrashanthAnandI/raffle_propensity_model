#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import mlflow
import mlflow.xgboost
import json
from preprocess import preprocess_data
from config import MLFLOW_MODEL_NAME, MODEL_STAGE

class ModelService:

    def __init__(self):

        
        #  Load specific model version        

        self.model_name = MLFLOW_MODEL_NAME
        self.model_version = MODEL_STAGE

        model_uri = f"models:/{self.model_name}/{self.model_version}"
        print(f"Loading model from {model_uri}")

        self.model = mlflow.xgboost.load_model(model_uri)

        
        client = mlflow.tracking.MlflowClient()

        model_version_details = client.get_model_version(
            name=self.model_name,
            version=self.model_version
        )

        run_id = model_version_details.run_id

        print(f"Fetching artifacts from run_id: {run_id}")

        local_path = client.download_artifacts(
            run_id,
            "cap_values.json"
        )

        with open(local_path, "r") as f:
            self.cap_values = json.load(f)

        print("Cap values loaded successfully.")

  
    #  Prediction
    def predict(self, df):

        # Apply preprocessing using stored caps
        df_processed = preprocess_data(df, self.cap_values)

        selected_features = [
            'TOTAL_TICKET_SALES',
            'NUMBER_OF_TICKET_RESPONSES',
            'AGE',
            'MEMBERSHIP_STATUS',
            'Raffle_Recency_Months',
            'Raffle_Tenure_Months',
            'Other_gift_Tenure_Months',
            'TOTAL_DONATIONS',
            'NUMBER_OF_DONATIONS_ONLY',
            'PERCENTAGE_FROM_DONATIONS',
            'PROXIMITY_CLASSIFICATION',
            'ANY_GIFT_RECENCY',
            'OTHER_RECENT_AND_NO_RAFFLE',
            'SEX'
        ]

        X = df_processed[selected_features]

        # Get probability scores
        probs = self.model.predict_proba(X)[:, 1]

        # Attach back to original dataframe
        df["DONATION_PROBABILITY"] = probs

        # Return only required output columns
        return df[["ID", "CONSTITUENT_ID", "DONATION_PROBABILITY"]]

