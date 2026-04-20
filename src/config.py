#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os

SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USER"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "authenticator": "externalbrowser",
    "role": os.getenv("SNOWFLAKE_ROLE"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DB"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
}

MLFLOW_MODEL_NAME = "raffle_model"
MODEL_STAGE = "Production"

