import joblib
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans


def find_eng_and_exp_score(experience_data, engagement_data, engagement_model, experience_model):

    df_experience = joblib.load(experience_data)
    df_engagement = joblib.load(engagement_data)

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

    joblib.dump(engagement_kmeans, engagement_model)
    joblib.dump(experience_kmeans, experience_model)

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

    return df

def find_satisfaction(df, satisfaction_model):

    satisfaction_kmeans = KMeans(n_clusters=2, random_state=42).fit(df[["Engagement Score", "Experience Score"]])
    df["Satisfaction"] = np.transpose(satisfaction_kmeans.labels_)
    joblib.dump(satisfaction_kmeans, satisfaction_model)

    print(df[["MSISDN/Number", "Satisfaction"]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experience_data", type=str, help="Path to experience data")
    parser.add_argument("--engagement_data", type=str, help="Path to engagement data")
    parser.add_argument("--engagement_model", type=str, help="Path to engagement model")
    parser.add_argument("--experience_model", type=str, help="Path to experience model")
    parser.add_argument("--satisfaction_model", type=str, help="Path to satisfaction model")
    args = parser.parse_args()

    df = find_eng_and_exp_score(args.experience_data, args.engagement_data, args.engagement_model,
                                args.experience_model)
    find_satisfaction(df, args.satisfaction_model)
