from EHRProcessor import EHRProcessor
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pickle
import time

class PatientClusteringProcessor:
    def __init__(self, data_file, abbreviations_file):
        self.data_file = data_file
        self.abbreviations_file = abbreviations_file
        self.ehr_processor = EHRProcessor(data_file, abbreviations_file)

    def run(self):
        self.ehr_processor.run()
        datasets = ['data/train_data.csv', 'data/test_data.csv', 'data/validate_data.csv']
        for dataset in datasets:
            print(f"Processing {dataset}")
            df = pd.read_csv(dataset)  # Ensure this line correctly loads the CSV into a DataFrame
            cluster_df = self.patient_clustering_processing(df)
            name = dataset.split('.')[0]
            cluster_df.to_csv(f"{name}_cluster_df.csv", index=False)

    def patient_clustering_processing(self, df):
        # Preprocessing -- adding back together input similarity columns
        df["Chief complaint"] = df["Chief complaint"].astype(str)
        df["Allergies"] = df["Allergies"].astype(str)
        df["Major surgical or invasive procedure"] = df["Major surgical or invasive procedure"].astype(str)
        df["History of present illness"] = df["History of present illness"].astype(str)
        df["Past medical history"] = df["Past medical history"].astype(str)
        df["Medications on admission"] = df["Medications on admission"].astype(str)
        df["Discharge medications"] = df["Discharge medications"].astype(str)
        df["Discharge diagnosis"] = df["Discharge diagnosis"].astype(str)

        # Final text column for sim index calcs
        df["medical_summary"] = df["Chief complaint"] + df["Allergies"] + df["Major surgical or invasive procedure"] + df["History of present illness"] + \
                                df["Past medical history"] + df["Medications on admission"] + df["Discharge medications"] + df["Discharge diagnosis"]

        cluster_train_df = df[["subject_id", "medical_summary"]]
        cluster_train_df = cluster_train_df.sort_values(by=["subject_id"]).reset_index(drop=True)

        # vectorization of the medical summary text
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(cluster_train_df["medical_summary"].values.astype('U'))

        # Perform clustering and save model object
        start = time.time()
        n_clusters = 50
        max_iter = 200
        print(f"Performing KMeans Clustering on {n_clusters} clusters...")
        kmeans_model = KMeans(n_clusters=n_clusters, max_iter=max_iter).fit(X)
        with open("model_objects/kmeans_cluster_model.pkl", "wb") as f:
            pickle.dump(kmeans_model, f)
        print("Model fitted and saved.")
        end = time.time()
        print("Clustering execution time: ", end - start)

        # Save cluster values to a new column in original dataset
        df["Cluster"] = kmeans_model.labels_.tolist()

        # Save final preprocessed dataset with clustering values (and dropping extra repetitive medical summary combined column)
        # NOTE: Keep original df with medical_summary column for patient similarity calculations

        # Number of patients per cluster
        df.Cluster.value_counts()

        cluster_df = df[['subject_id', "medical_summary", 'Cluster']].copy()

        # df_save -> original
        return cluster_df

if __name__ == '__main__':
    data_file = 'data/discharge.csv'
    abbreviations_file = 'abbreviations.json'
    processor = PatientClusteringProcessor(data_file, abbreviations_file)
    processor.run()