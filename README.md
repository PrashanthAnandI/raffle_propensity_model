# raffle_propensity_model
This project demonstrates an end-to-end machine learning pipeline built to predict the likelihood of donation for raffle campaign supporters.The solution combines data extraction, feature engineering, model serving, and API deployment into a single, production-style workflow.

Tech Stack:
Python
FastAPI
MLflow
XGBoost
Snowflake
Pandas / NumPy
Docker

What This Project Does:
Given a mail_date, the API:
Fetches relevant supporter data from snowflake
Applies preprocessing and feature engineering
Loads a trained model from MLflow
Returns a probability score indicating likelihood of donation

