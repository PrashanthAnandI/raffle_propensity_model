#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
import pandas as pd
import os

from snowflake_client import fetch_data
from model_service import ModelService

app = FastAPI()

# Initialize model once at startup
model_service = ModelService()


class MailDateRequest(BaseModel):
    mail_date: str


@app.post("/score")
def score_raffle(request: MailDateRequest):

    mail_date = request.mail_date

    print(f"Received mail_date: {mail_date}")

    
    #  Fetch data from Snowflake
    
    df = fetch_data(mail_date)

    if df.empty:
        return {"message": "No data returned for given mail_date."}

    print("Data fetched successfully.")

   
    ##Predict using MLflow model
    
    scored_df = model_service.predict(df)

    print("Prediction completed.")

   # 
    #  output
   
    #output_file = "scored_output.csv"
    #scored_df.to_csv(output_file, index=False)
    return scored_df.to_dict(orient="records")
    #return FileResponse(
        #path=output_file,
        #media_type="text/csv",
        #filename="scored_output.csv"
   # )

