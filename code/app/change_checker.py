
import pandas as pd
from datetime import datetime

def load_cmdb(cmdb_path):
    return pd.read_csv(cmdb_path)

def load_changes(change_path):
    return pd.read_csv(change_path)

def get_related_changes(app, incident_date, change_df):
    try:
        incident_dt = datetime.strptime(incident_date, "%Y-%m-%d")
    except:
        return []

    related_changes = change_df[
        (change_df["app"] == app) &
        (pd.to_datetime(change_df["date"], errors='coerce') <= incident_dt)
    ]
    return related_changes.to_dict(orient="records")
