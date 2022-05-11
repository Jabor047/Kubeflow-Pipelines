import io
import joblib
import argparse
import pandas as pd
from requests import get


def download_data() -> pd.DataFrame:

    url = "https://storage.googleapis.com/< GCS BUCKET >/data/telco.csv"
    s = get(url).content
    df = pd.read_csv(io.StringIO(s.decode("utf-8")))

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    df_fill_nan = df.drop(["Bearer Id", "IMSI", "MSISDN/Number", "IMEI", "Start", "End", "Last Location Name",
                           "Handset Manufacturer", "Handset Type"], axis=1)

    # fill all the null values with the mean of the column they occur in
    df_fill_nan = df_fill_nan.apply(lambda x: x.fillna(x.median()), axis=1)

    # replace the columns where values have been filled in
    for col in df_fill_nan.columns.to_list():
        df[col] = df_fill_nan[col]

    return df

def feature_prep(df: pd.DataFrame, exp_data_path: str, eng_data_path: str):

    # this sums up all the values in a dataframe col
    def sum_agg(dataframe: pd.DataFrame, col: str) -> pd.DataFrame:
        s = dataframe.groupby("MSISDN/Number")[col].agg("sum").sort_values(ascending=False)
        df = pd.DataFrame({"MSISDN/Number": s.index, col: s.values})

        return df

    # combine both the download and upload data cols in bytes
    df["Total traffic"] = df["Total DL (Bytes)"] + df["Total UL (Bytes)"]

    # get total duration and traffic for each use and merhe them into one dataframe
    df_dur_ms = sum_agg(df, "Dur. (ms)")
    df_total_bytes = sum_agg(df, "Total traffic")
    df_engagement = pd.merge(df_dur_ms, df_total_bytes, on="MSISDN/Number")

    # get the number of sessions each user has
    session_series = df.groupby("MSISDN/Number")["Dur. (ms)"].count().sort_values(ascending=False)
    df_sess_freq = pd.DataFrame({"MSISDN/Number": session_series.index, "sessions freq": session_series.values})

    # merge the engagement dataframe and sessions frequency dataframe
    df_engagement = pd.merge(df_engagement, df_sess_freq, on="MSISDN/Number")

    # Summing the Uploads and Downloads columns to get the Total data columns
    df["Avg RTT"] = df["Avg RTT DL (ms)"] + df["Avg RTT UL (ms)"]
    df["Avg Bearer TP (kbps)"] = df["Avg Bearer TP DL (kbps)"] + df["Avg Bearer TP UL (kbps)"]
    df["TCP Retrans. Vol (Bytes)"] = df["TCP DL Retrans. Vol (Bytes)"] + df["TCP UL Retrans. Vol (Bytes)"]

    # select the required columns for experience analysis
    df_experience = df[["MSISDN/Number", "Avg RTT", "Avg Bearer TP (kbps)", "TCP Retrans. Vol (Bytes)"]]

    joblib.dump(df_experience, exp_data_path)
    joblib.dump(df_engagement, eng_data_path)

if __name__ == "__main__":
    print("Feature preparation started")
    df = download_data()
    df = clean_data(df)

    parser = argparse.ArgumentParser()
    parser.add_argument("--experience_data", type=str, help="Path to experience data")
    parser.add_argument("--engagement_data", type=str, help="Path to engagement data")
    args = parser.parse_args()

    feature_prep(df, args.experience_data, args.engagement_data)
    print("Feature preparation complete")
