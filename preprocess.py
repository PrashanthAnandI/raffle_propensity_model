#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


def preprocess_data(df, cap_values_dict):

    df = df.copy()

    # -------------------------
    # Date Handling
    # -------------------------
    date_cols = [
        "LATEST_RAFFLE_GIFTDATE",
        "FIRST_RAFFLE_GIFTDATE",
        "FIRSTOTHERGIFTDATE",
        "LATESTOTHERGIFTDATE",
        "MAIL_DATE",
        "LATEST_EVENT_DATE"
    ]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["Has_Raffle_History"] = df["LATEST_RAFFLE_GIFTDATE"].notna().astype(int)
    df["Has_other_gifts"] = df["LATESTOTHERGIFTDATE"].notna().astype(int)
    df["is_event_participated"] = df["LATEST_EVENT_DATE"].notna().astype(int)

    # -------------------------
    # Numeric Null Handling
    # -------------------------
    numeric_zero_cols = [
        "TOTAL_RAFFLE_GIFT","AVG_RAFFLE_GIFT","LARGEST_RAFFLE_GIFT",
        "TOTAL_TICKET_SALES","AVG_TICKET_SALES",
        "TOTAL_DONATIONS","AVG_DONATION",
        "NUMBER_OF_TICKET_RESPONSES","NUMBER_OF_DONATIONS_ONLY",
        "PERCENTAGE_FROM_TICKETS","TOTAL_OTHER_GIFT","AVG_OTHER_GIFT",
        "NUMBER_OF_EVENTS"
    ]

    df[numeric_zero_cols] = df[numeric_zero_cols].fillna(0)

    df[["ACORN_E6_CAT","ACORN_E6_GRP","MEMBERSHIP_STATUS"]] =         df[["ACORN_E6_CAT","ACORN_E6_GRP","MEMBERSHIP_STATUS"]].fillna("Unknown")

    df["AGE"] = df["AGE"].fillna(df["AGE"].mean())

    # -------------------------
    # Recency Features
    # -------------------------
    df["Raffle_Recency_Months"] = (
        (df["MAIL_DATE"] - df["LATEST_RAFFLE_GIFTDATE"]).dt.days / 30.44
    )

    df["Raffle_Tenure_Months"] = (
        (df["MAIL_DATE"] - df["FIRST_RAFFLE_GIFTDATE"]).dt.days / 30.44
    )

    df["Other_gift_Recency_Months"] = (
        (df["MAIL_DATE"] - df["LATESTOTHERGIFTDATE"]).dt.days / 30.44
    )

    df["Other_gift_Tenure_Months"] = (
        (df["MAIL_DATE"] - df["FIRSTOTHERGIFTDATE"]).dt.days / 30.44
    )

    df["Event_Recency_Months"] = (
        (df["MAIL_DATE"] - df["LATEST_EVENT_DATE"]).dt.days / 30.44
    )

    df = df.drop(columns=date_cols)

    # -------------------------
    # APPLY CAPS (from MLflow)
    # -------------------------
    for col, cap in cap_values_dict.items():
        df[col] = df[col].clip(upper=cap)

    # -------------------------
    # Fill recency null logic
    # -------------------------
    df["Raffle_Recency_Months"] = df["Raffle_Recency_Months"].fillna(999)
    df["Raffle_Tenure_Months"] = df["Raffle_Tenure_Months"].fillna(0)
    df["Other_gift_Recency_Months"] = df["Other_gift_Recency_Months"].fillna(999)
    df["Other_gift_Tenure_Months"] = df["Other_gift_Tenure_Months"].fillna(0)
    df["Event_Recency_Months"] = df["Event_Recency_Months"].fillna(999)

    # -------------------------
    # Feature Engineering
    # -------------------------
    df["ANY_GIFT_RECENCY"] = df[
        ["Raffle_Recency_Months","Other_gift_Recency_Months"]
    ].min(axis=1)

    df["OTHER_RECENT_AND_NO_RAFFLE"] = (
        (df["Other_gift_Recency_Months"] <= 12) &
        (df["Raffle_Tenure_Months"] == 0)
    ).astype(int)

    df["IS_REGULAR_GIVER"] = df["IS_REGULAR_GIVER"].map({"Y":1,"N":0}).fillna(0)

    proximity_map = {
        "No Proximity Coded":0,
        "Further Proximity":1,
        "Close Proximity":2
    }

    df["PROXIMITY_CLASSIFICATION"] =         df["PROXIMITY_CLASSIFICATION"].map(proximity_map).fillna(0)

    membership_map = {"Unknown":0,"Dropped":1,"Active":2}
    df["MEMBERSHIP_STATUS"] = df["MEMBERSHIP_STATUS"].map(membership_map).fillna(0)

    df["HAS_GIVING"] = (
        df["TOTAL_DONATIONS"] + df["TOTAL_TICKET_SALES"] > 0
    ).astype(int)

    df["PERCENTAGE_FROM_DONATIONS"] = np.where(
        df["HAS_GIVING"]==1,
        100 - df["PERCENTAGE_FROM_TICKETS"],
        0
    )

    df["Loyalty_Score"] = np.log1p(df["Raffle_Tenure_Months"])
    df["Raffle_Recency_Months"] = np.log1p(df["Raffle_Recency_Months"])

    df = df.drop(columns=["LAST_DONATION_TYPE3","FIRST_DONATION_TYPE3"], errors="ignore")

    return df

