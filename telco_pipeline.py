import kfp
from kfp import dsl
from typing import NamedTuple

def download_data() -> str:
    import pandas as pd
    from requests import get
    import io

    url = "https://storage.googleapis.com/datatonic-mlops/telco.csv"
    s = get(url).content
    df = pd.read_csv(io.StringIO(s.decode("utf-8")))

    data_path = "/data/telco.csv"
    df.to_csv(data_path)

    return data_path


def clean_data(input_path: str) -> str:
    import pandas as pd

    df = pd.read_csv(input_path)
    df_fill_nan = df.drop(["Bearer Id", "IMSI", "MSISDN/Number", "IMEI", "Start", "End", "Last Location Name",
                           "Handset Manufacturer", "Handset Type"], axis=1)

    # fill all the null values with the mean of the column they occur in
    df_fill_nan = df_fill_nan.apply(lambda x: x.fillna(x.mean()), axis=1)

    # replace the columns where values have been filled in
    for col in df_fill_nan.columns.to_list():
        df[col] = df_fill_nan[col]

    clean_data_path = "/data/clean_telco.csv"
    df.to_csv(clean_data_path)

    return clean_data_path

def feature_prep(input_path: str) -> NamedTuple("feature_paths", [("exp_path", str), ("eng_path", str)]):
    import pandas as pd

    # this sums up all the values in a dataframe col
    def sum_agg(dataframe: pd.DataFrame, col: str) -> pd.DataFrame:
        s = dataframe.groupby("MSISDN/Number")[col].agg("sum").sort_values(ascending=False)
        df = pd.DataFrame({"MSISDN/Number": s.index, col: s.values})

        return df

    df = pd.read_csv(input_path)

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

    # convert bytes and milliseconds to megabytes and seconds
    df_engagement["Total traffic"] = df_engagement["Total traffic"] / 1000000
    df_engagement["Dur"] = df_engagement["Dur. (ms)"] / 1000
    df_engagement.drop(["Dur. (ms)"], axis=1, inplace=True)

    # Summing the Uploads and Downloads columns to get the Total data columns
    df["Avg RTT"] = df["Avg RTT DL (ms)"] + df["Avg RTT UL (ms)"]
    df["Avg Bearer TP (mbps)"] = df["Avg Bearer TP DL (kbps)"] + df["Avg Bearer TP UL (kbps)"]
    df["TCP Retrans. Vol (MegaBytes)"] = df["TCP DL Retrans. Vol (Bytes)"] + df["TCP UL Retrans. Vol (Bytes)"]

    # Turning the Data columns into MegaBytes
    df["Avg Bearer TP (mbps)"] = df["Avg Bearer TP (mbps)"] / 1000
    df["TCP Retrans. Vol (MegaBytes)"] = df["TCP Retrans. Vol (MegaBytes)"] / 1000000

    df_experience = df[["MSISDN/Number", "Avg RTT", "Avg Bearer TP (mbps)", "TCP Retrans. Vol (MegaBytes)"]]

    df_exp_path = "/data/experience.csv"
    df_eng_path = "/data/engagement.csv"

    df_engagement.to_csv(df_eng_path)
    df_experience.to_csv(df_exp_path)

    from collections import namedtuple
    feature_paths = namedtuple("feature_paths", ["exp_path", "eng_path"])
    return feature_paths
