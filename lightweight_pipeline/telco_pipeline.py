import kfp
from kfp import dsl
from typing import NamedTuple

def download_data() -> str:
    import pandas as pd
    from requests import get
    import io

    url = "https://storage.googleapis.com/<GCS Bucket>/data/telco.csv"
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
    df_fill_nan = df_fill_nan.apply(lambda x: x.fillna(x.median()), axis=1)

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

    # Summing the Uploads and Downloads columns to get the Total data columns
    df["Avg RTT"] = df["Avg RTT DL (ms)"] + df["Avg RTT UL (ms)"]
    df["Avg Bearer TP (kbps)"] = df["Avg Bearer TP DL (kbps)"] + df["Avg Bearer TP UL (kbps)"]
    df["TCP Retrans. Vol (Bytes)"] = df["TCP DL Retrans. Vol (Bytes)"] + df["TCP UL Retrans. Vol (Bytes)"]

    # select the required columns for experience analysis
    df_experience = df[["MSISDN/Number", "Avg RTT", "Avg Bearer TP (kbps)", "TCP Retrans. Vol (Bytes)"]]

    df_exp_path = "/data/experience.csv"
    df_eng_path = "/data/engagement.csv"

    df_engagement.to_csv(df_eng_path)
    df_experience.to_csv(df_exp_path)

    # convert the feature paths to a named tuple
    from collections import namedtuple
    feature_paths = namedtuple("feature_paths", ["exp_path", "eng_path"])
    return feature_paths(df_eng_path, df_eng_path)

def find_eng_and_exp_score(exp_path: str, eng_path: str) -> str:
    import joblib
    import json
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import normalize
    from sklearn.cluster import KMeans
    from google.cloud import storage
    from requests import get

    url = "https://storage.googleapis.com/<GCS Bucket>/kubeflow-tutorials-340813-d338387fc0f4.json"
    s = get(url).content

    serv_acc_json = json.loads(s)
    with open("/data/service_account.json", 'w') as f:
        json.dump(serv_acc_json, f)

    storage_client = storage.Client.from_service_account_json("/data/service_account.json")
    bucket = storage_client.bucket("<GCS Bucket>")
    eng_blob = bucket.blob("models/sklearn/engagement/001/model.pkl")
    exp_blob = bucket.blob("models/sklearn/experience/001/model.pkl")

    df_experience = pd.read_csv(exp_path)
    df_engagement = pd.read_csv(eng_path)

    # normalize the columns to be entered in to the clustering algo
    def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        df.drop("MSISDN/Number", axis=1, inplace=True)
        for col in df.columns.to_list():
            df[col] = np.transpose(normalize([np.array(df[col])]))

        return df

    df_experience_norm = normalize_df(df_experience.copy())
    df_engagement_norm = normalize_df(df_engagement.copy())

    # cluster the engagement and experience into 3 clusters each
    engagement_kmeans = KMeans(n_clusters=3, random_state=42).fit(df_engagement_norm)
    experience_kmeans = KMeans(n_clusters=3, random_state=42).fit(df_experience_norm)

    joblib.dump(engagement_kmeans, "/data/engagement_kmeans.pkl")
    joblib.dump(experience_kmeans, "/data/experience_kmeans.pkl")
    eng_blob.upload_from_filename("/data/engagement_kmeans.pkl")
    exp_blob.upload_from_filename("/data/experience_kmeans.pkl")

    least_eng_cluster = engagement_kmeans.cluster_centers_[0]
    least_exp_cluster = experience_kmeans.cluster_centers_[0]

    # finding the eculidean distance between each data point and the least engaged cluster
    engagement_score = []
    for row in df_engagement_norm.to_numpy():
        eng_score = np.linalg.norm(row - least_eng_cluster)
        engagement_score.append(eng_score)

    # finding the eculidean distance between each data point and the least experience cluster
    experience_score = []
    for row in df_experience_norm.to_numpy():
        exp_score = np.linalg.norm(row - least_exp_cluster)
        experience_score.append(exp_score)

    # add the engagement and experience scores to normalized dataframes
    df_engagement_norm["Engagement Score"] = np.transpose(np.array(engagement_score))
    df_experience_norm["Experience Score"] = np.transpose(np.array(experience_score))

    # add the unique identifier for merging
    df_experience_norm["MSISDN/Number"] = df_experience["MSISDN/Number"]
    df_engagement_norm["MSISDN/Number"] = df_engagement["MSISDN/Number"]

    # merging the two dataframes
    df = pd.merge(df_engagement_norm, df_experience_norm, on="MSISDN/Number")

    sat_data_path = "/data/satisfication.csv"
    df.to_csv(sat_data_path)

    return sat_data_path

def find_satisfaction(input_path: str):
    import joblib
    import json
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans
    from google.cloud import storage
    from requests import get

    url = "https://storage.googleapis.com/<GCS Bucket>/kubeflow-tutorials-340813-d338387fc0f4.json"
    s = get(url).content

    serv_acc_json = json.loads(s)
    with open("/data/service_account.json", 'w') as f:
        json.dump(serv_acc_json, f)

    storage_client = storage.Client.from_service_account_json("/data/service_account.json")
    bucket = storage_client.bucket("<GCS Bucket>")
    blob = bucket.blob("models/sklearn/satisfaction/001/model.pkl")

    df = pd.read_csv(input_path)
    satisfaction_kmeans = KMeans(n_clusters=2, random_state=42).fit(df[["Engagement Score", "Experience Score"]])
    df["Satisfaction"] = np.transpose(satisfaction_kmeans.labels_)
    joblib.dump(satisfaction_kmeans, "/data/satisfaction_kmeans.pkl")
    blob.upload_from_filename("/data/satisfaction_kmeans.pkl")

    print(df[["MSISDN/Number", "Satisfaction"]])

@dsl.pipeline(name="telco_pipeline",
              description="lightweight component Telco pipeline for the presentation")
def telco_pipeline():

    data_op = dsl.VolumeOp(name="create-pvc",
                           resource_name="data-volume",
                           size="2Gi",
                           modes=dsl.VOLUME_MODE_RWO)

    download_data_op = kfp.components.func_to_container_op(download_data,
                                                           packages_to_install=["pandas", "requests"])
    clean_data_op = kfp.components.func_to_container_op(clean_data, packages_to_install=["pandas"])
    feature_prep_op = kfp.components.func_to_container_op(feature_prep, packages_to_install=["pandas"])
    find_eng_and_exp_score_op = kfp.components.func_to_container_op(find_eng_and_exp_score,
                                                                    packages_to_install=["pandas", "numpy",
                                                                                         "scikit-learn",
                                                                                         "google-cloud-storage",
                                                                                         "requests"])
    find_satisfaction_op = kfp.components.func_to_container_op(find_satisfaction,
                                                               packages_to_install=["pandas", "numpy",
                                                                                    "scikit-learn",
                                                                                    "google-cloud-storage",
                                                                                    "requests"])

    step1 = download_data_op().add_pvolumes({"/data": data_op.volume})
    step2 = clean_data_op(step1.output).add_pvolumes({"/data": data_op.volume})
    step3 = feature_prep_op(step2.output).add_pvolumes({"/data": data_op.volume})
    step4 = find_eng_and_exp_score_op(step3.outputs["exp_path"], step3.outputs["eng_path"])\
        .add_pvolumes({"/data": data_op.volume})
    step5 = find_satisfaction_op(step4.output).add_pvolumes({"/data": data_op.volume})

    kf_serve = kfp.components.load_component_from_url("https://raw.githubusercontent.com/kubeflow/pipelines/master/comp"
                                                      "onents/kubeflow/kfserving/component.yaml")
    kf_serve_op = kf_serve(
        action="apply",
        model_uri="gs://<GCS Bucket>/models/sklearn/satisfaction",
        model_name="satisfactionkmeans",
        namespace="gkkarobia",
        framework="sklearn",
        watch_timeout="300"
    )
    kf_serve_op.after(step5)

kfp.compiler.Compiler().compile(telco_pipeline, "telco_pipeline.zip")

kubeflow_gateway_endpoint = "localhost:7777"
authservice_session_cookie = "MTY1Mjg2MTY4MnxOd3dBTkZnM1dVdFVRMFEwTkVWR1dVWlZWbEpYVmxBMVNrUlpTRXhTUkZoV05ETkVTalpaUWtaU"
"E5GaExXbFZKVlU0MlNWZE9XbEU9fCF_YdjoAAYppXzd0de2fRiN9xrLret1r5AhyO25J0XO"
namespace = "gkkarobia"

client = kfp.Client(f"http://{kubeflow_gateway_endpoint}/pipeline",
                    cookies=f"authservice_session={authservice_session_cookie}")

experiment = client.create_experiment("telco", namespace=namespace)
print(client.list_experiments(namespace=namespace))

run = client.run_pipeline(experiment_id=experiment.id, job_name="telco_pipeline",
                          pipeline_package_path="telco_pipeline.zip")
