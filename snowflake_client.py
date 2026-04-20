#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# snowflake_client.py

import snowflake.connector
import pandas as pd
from config import SNOWFLAKE_CONFIG


def get_connection():
    return snowflake.connector.connect(**SNOWFLAKE_CONFIG)


def fetch_data(mail_date: str):
# Actual query removed for confidentiality
    query = """
    "SELECT * FROM YOUR_TABLE WHERE mail_date = %(mail_date)s"
    """

    conn = get_connection()

    df = pd.read_sql(
        query,
        conn,
        params={"mail_date": mail_date}
    )

    conn.close()

    return df

