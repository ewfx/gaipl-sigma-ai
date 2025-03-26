import pandas as pd

def load_incident_data(filepath):
    df = pd.read_csv(filepath)
    df.fillna("", inplace=True)

    df["combined_text"] = (
        "Description: " + df["description"] + ". " +
        "Category: " + df["category"] + ". " +
        "Urgency: " + df["urgency"] + ". " +
        "CI ID: " + df["ci_id"] + ". " +
        "CR Number: " + df["cr_number"] + ". " +
        "Resolution: " + df["resolution"] + ". " +
        "Tags: " + df["tags"] + ". " +
        "Cause: " + df["cause"]
    )

    return df
