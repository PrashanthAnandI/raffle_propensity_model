## raffle_propensity_model
This project demonstrates an end-to-end machine learning pipeline built to predict the likelihood of donation for raffle campaign supporters.The solution combines data extraction, feature engineering, model serving, and API deployment into a single, production-style workflow.

## Tech Stack:
- Python
- FastAPI
- MLflow
- XGBoost
- Snowflake
- Pandas / NumPy
- Docker

## What This Project Does:
Given a mail_date, the API:
- Fetches relevant supporter data from snowflake
- Applies preprocessing and feature engineering
- Loads a trained model from MLflow
- Returns a probability score indicating likelihood of donation

```markdown id="1y8mkk"
## Project Structure
raffle_propensity_model/
│── src/
│   ├── app.py
│   ├── config.py
│   ├── model_service.py
│   ├── preprocess.py
│   ├── snowflake_client.py
│
│── Dockerfile
│── requirements.txt
│── README.md

## How to Run Locally
1. Install dependencies
pip install -r requirements.txt
2. Start the API
uvicorn src.app:app --reload
3. Open API Docs
http://127.0.0.1:8000/docs

## API Example
Request
{
  "mail_date": "01-JAN-2024"
}

Response

{
Supporter ID: 123,
Donation Probability: 0.85
}

## Notes
Real data and queries are not included due to confidentiality.
Snowflake connection requires internal credentials.
This repository focuses on code structure and modeling approach.
