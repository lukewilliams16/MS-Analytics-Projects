import streamlit as st
import pandas as pd
import medspacy
from TargetMatcher import load_nlp 
from spacy_streamlit import visualize_ner
from medspacy.visualization import visualize_ent
import medspacy
from medspacy.ner import TargetRule
from spacy_streamlit import visualize_ner
from medspacy.ner import TargetRule
from medspacy.visualization import visualize_ent
import pandas as pd
import time
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.cluster import KMeans 
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import re 

@st.cache_data
def load_nlp():
    nlp = medspacy.load()
    target_matcher = nlp.get_pipe("medspacy_target_matcher")
    
    target_rules = [
    # Heart Disease
        TargetRule("coronary artery disease", "HEART_DISEASE"),
        TargetRule("myocardial infarction", "HEART_DISEASE"),
        TargetRule("heart disease", "HEART_DISEASE"),
        TargetRule("heart failure", "HEART_DISEASE"),
        TargetRule("arrhythmia", "HEART_DISEASE"),
        TargetRule("atrial fibrillation", "HEART_DISEASE"),
        TargetRule("cardiomyopathy", "HEART_DISEASE"),
        TargetRule("congenital heart disease", "HEART_DISEASE"),
        TargetRule("aortic stenosis", "HEART_DISEASE"),
        TargetRule("aortic valve stenosis", "HEART_DISEASE"),
        TargetRule("pericarditis", "HEART_DISEASE"),
        TargetRule("atrial septal defect", "HEART_DISEASE"),
        TargetRule("ventricular septal defect", "HEART_DISEASE"),
        TargetRule("patent ductus arteriosus", "HEART_DISEASE"),
        TargetRule("ventricular septal rupture", "HEART_DISEASE"),
        TargetRule("coronary angiography", "HEART_DISEASE"),
        TargetRule("cardiac catheterization", "HEART_DISEASE"),
        TargetRule("angioplasty with stent placement", "HEART_DISEASE"),
        TargetRule("coronary artery bypass grafting", "HEART_DISEASE"),
        TargetRule("valve repair", "HEART_DISEASE"),
        TargetRule("valve replacement", "HEART_DISEASE"),
        TargetRule("heart transplantation", "HEART_DISEASE"),
        TargetRule("implantation of pacemaker", "HEART_DISEASE"),
        TargetRule("implantation of defibrillator", "HEART_DISEASE"),
        TargetRule("cardiac rehabilitation", "HEART_DISEASE"),

        # Cancer
        TargetRule("malignant neoplasm", "CANCER"),
        TargetRule("breast cancer", "CANCER"),
        TargetRule("lung cancer", "CANCER"),
        TargetRule("colorectal cancer", "CANCER"),
        TargetRule("leukemia", "CANCER"),
        TargetRule("lymphoma", "CANCER"),
        TargetRule("metastasis", "CANCER"),
        TargetRule("sarcoma", "CANCER"),
        TargetRule("cervical cancer", "CANCER"),
        TargetRule("carcinoma", "CANCER"),
        TargetRule("carcinoma in situ", "CANCER"),
        TargetRule("paraneoplastic syndrome", "CANCER"),
        TargetRule("chemotherapy", "CANCER"),
        TargetRule("radiotherapy", "CANCER"),
        TargetRule("immunotherapy", "CANCER"),
        TargetRule("targeted therapy", "CANCER"),
        TargetRule("hormone therapy", "CANCER"),
        TargetRule("stem cell transplant", "CANCER"),
        TargetRule("checkpoint inhibitor", "CANCER"),
        TargetRule("monoclonal antibody therapy", "CANCER"),
        TargetRule("tyrosine kinase inhibitor", "CANCER"),
        TargetRule("biologic therapy", "CANCER"),
        TargetRule("cancer vaccine", "CANCER"),
        TargetRule("targeted molecular therapy", "CANCER"),
        TargetRule("radiosensitizer", "CANCER"),
        TargetRule("immune checkpoint blockade", "CANCER"),
        TargetRule("oncolytic virus therapy", "CANCER"),
        TargetRule("proton therapy", "CANCER"),
        TargetRule("cytokine therapy", "CANCER"),
        TargetRule("neoadjuvant therapy", "CANCER"),
        TargetRule("adjuvant therapy", "CANCER"),
        TargetRule("targeted drug delivery", "CANCER"),
        TargetRule("cancer", "CANCER"),
        TargetRule("biops", "CANCER"),
        TargetRule("mastectomy", "CANCER"),
        TargetRule("lumpectomy", "CANCER"),
        TargetRule("radiotherapy", "CANCER"),
        TargetRule("chemotherapy", "CANCER"),
        TargetRule("immunotherapy", "CANCER"),
        TargetRule("targeted therapy", "CANCER"),
        TargetRule("hormone therapy", "CANCER"),
        TargetRule("stem cell transplant", "CANCER"),

        # Chronic Lung Disease
        TargetRule("chronic obstructive pulmonary disease", "LUNG_DISEASE"),
        TargetRule("lung disease", "LUNG_DISEASE"),
        TargetRule("emphysema", "LUNG_DISEASE"),
        TargetRule("chronic bronchitis", "LUNG_DISEASE"),
        TargetRule("asthma", "LUNG_DISEASE"),
        TargetRule("pulmonary fibrosis", "LUNG_DISEASE"),
        TargetRule("cystic fibrosis", "LUNG_DISEASE"),
        TargetRule("occupational lung disease", "LUNG_DISEASE"),
        TargetRule("inhaled corticosteroid", "LUNG_DISEASE"),
        TargetRule("pulmonary artery hypertension", "LUNG_DISEASE"),
        TargetRule("bronchiectasis", "LUNG_DISEASE"),
        TargetRule("incentive spirometry", "LUNG_DISEASE"),
        TargetRule("spirometry", "LUNG_DISEASE"),
        TargetRule("bronchoscopy", "LUNG_DISEASE"),
        TargetRule("lung volume reduction surgery", "LUNG_DISEASE"),
        TargetRule("lung transplantation", "LUNG_DISEASE"),
        TargetRule("pulmonary rehabilitation", "LUNG_DISEASE"),
        TargetRule("continuous positive airway pressure", "LUNG_DISEASE"),
        TargetRule("mechanical ventilation", "LUNG_DISEASE"),

        # Stroke
        TargetRule("ischemic stroke", "STROKE"),
        TargetRule("hemorrhagic stroke", "STROKE"),
        TargetRule("transient ischemic attack", "STROKE"),
        TargetRule("subarachnoid hemorrhage", "STROKE"),
        TargetRule("cerebral aneurysm", "STROKE"),
        TargetRule("vascular malformation", "STROKE"),
        TargetRule("carotid endarterectomy", "STROKE"),
        TargetRule("intravenous thrombolysis", "STROKE"),
        TargetRule("mechanical thrombectomy", "STROKE"),
        TargetRule("antiplatelet therapy", "STROKE"),
        TargetRule("anticoagulation therapy", "STROKE"),
        TargetRule("neurosurgery", "STROKE"),
        TargetRule("rehabilitation therapy", "STROKE"),
        TargetRule("stroke", "STROKE"),

        # Alzheimer's Disease
        TargetRule("Alzheimer's disease", "ALZHEIMERS"),
        TargetRule("dementia", "ALZHEIMERS"),
        TargetRule("memory loss", "ALZHEIMERS"),
        TargetRule("cognitive decline", "ALZHEIMERS"),
        TargetRule("amyloid plaques", "ALZHEIMERS"),
        TargetRule("neurofibrillary tangles", "ALZHEIMERS"),
        TargetRule("cholinesterase inhibitors", "ALZHEIMERS"),
        TargetRule("memantine", "ALZHEIMERS"),
        TargetRule("cognitive stimulation therapy", "ALZHEIMERS"),
        TargetRule("reminiscence therapy", "ALZHEIMERS"),
        TargetRule("aromatherapy", "ALZHEIMERS"),
        TargetRule("music therapy", "ALZHEIMERS"),
        TargetRule("occupational therapy", "ALZHEIMERS"),

        # Diabetes
        TargetRule("diabetes mellitus", "DIABETES"),
        TargetRule("type 1 diabetes", "DIABETES"),
        TargetRule("type 2 diabetes", "DIABETES"),
        TargetRule("insulin", "DIABETES"),
        TargetRule("glucose monitoring", "DIABETES"),
        TargetRule("hemoglobin A1c", "DIABETES"),
        TargetRule("metformin", "DIABETES"),
        TargetRule("insulin therapy", "DIABETES"),
        TargetRule("sulfonylureas", "DIABETES"),
        TargetRule("DPP-4 inhibitors", "DIABETES"),
        TargetRule("GLP-1 receptor agonists", "DIABETES"),
        TargetRule("SGLT-2 inhibitors", "DIABETES"),
        TargetRule("thiazolidinediones", "DIABETES"),
        TargetRule("alpha-glucosidase inhibitors", "DIABETES"),
        TargetRule("insulin pump", "DIABETES"),
        TargetRule("continuous glucose monitoring", "DIABETES"),
        TargetRule("diabetes education", "DIABETES"),
        TargetRule("pancreas transplantation", "DIABETES"),
        TargetRule("islet cell transplantation", "DIABETES"),
        TargetRule("glucose tolerance test", "DIABETES"),
        TargetRule("hemoglobin A1c test", "DIABETES"),
        TargetRule("fasting blood glucose test", "DIABETES"),
        TargetRule("insulin tolerance test", "DIABETES"),
        TargetRule("oral glucose tolerance test", "DIABETES"),
        TargetRule("random blood glucose test", "DIABETES"),
        TargetRule("urine glucose test", "DIABETES"),
        TargetRule("renal artery stenosis", "DIABETES"),
        TargetRule("nephropathy", "DIABETES"),
        TargetRule("neuropathy", "DIABETES"),
        TargetRule("retinopathy", "DIABETES"),
        TargetRule("peripheral vascular disease", "DIABETES"),
        TargetRule("diabetic foot", "DIABETES"),
        TargetRule("diabetes", "DIABETES"),

        # Chronic Kidney Disease (CHRONIC_KIDNEY_DISEASE)
        TargetRule("chronic kidney disease", "CHRONIC_KIDNEY_DISEASE"),
        TargetRule("kidney disease", "CHRONIC_KIDNEY_DISEASE"),
        TargetRule("acute kidney injury", "CHRONIC_KIDNEY_DISEASE"),
        TargetRule("glomerulonephritis", "CHRONIC_KIDNEY_DISEASE"),
        TargetRule("kidney stones", "CHRONIC_KIDNEY_DISEASE"),
        TargetRule("end-stage renal disease", "CHRONIC_KIDNEY_DISEASE"),
        TargetRule("dialysis", "CHRONIC_KIDNEY_DISEASE"),
        TargetRule("hemodialysis", "CHRONIC_KIDNEY_DISEASE"),
        TargetRule("peritoneal dialysis", "CHRONIC_KIDNEY_DISEASE"),
        TargetRule("kidney transplantation", "CHRONIC_KIDNEY_DISEASE"),
        TargetRule("urinalysis", "CHRONIC_KIDNEY_DISEASET"),
        TargetRule("kidney biopsy", "CHRONIC_KIDNEY_DISEASE"),
        TargetRule("cystoscopy", "CHRONIC_KIDNEY_DISEASE"),
        TargetRule("nephrectomy", "CHRONIC_KIDNEY_DISEASE"),
        TargetRule("renal artery angioplasty", "CHRONIC_KIDNEY_DISEASE"),
        TargetRule("ureteroscopy", "CHRONIC_KIDNEY_DISEASE"),
    ]
    target_matcher.add(target_rules)
    return nlp

color_map = {
    'HEART_DISEASE': '#FF6666',  # red variations
    'ALZHEIMERS': '#6666FF',     # blue variations
    'CANCER': '#FF66FF',         # pink variations
    'LUNG_DISEASE': '#66CCFF',   # light blue variations
    'DIABETES': '#FFFF66',       # yellow variations
    'CHRONIC_KIDNEY_DISEASE': '#66FF66',  # green variations
}

colors = {
    "colors" :{
    'DIABETES_ANATOMY': '#FFFF99',
    'HEART_DISEASE_PROCEDURE': '#FF9999',
    'KIDNEY_CANCER': '#FF99FF',
    'LUNG_DISEASE_TREATMENT': '#99D6FF',
    'CANCER_MEDICATION': '#FF99FF',
    'LUNG_DISEASE_MEDICATION': '#99D6FF',
    'HEART_DISEASE_VITAL': '#FF9999',
    'HEART_DISEASE_MEDICATION': '#FF9999',
    'DIABETES': '#FFFF99',
    'DIABETES_TREATMENT': '#FFFF99',
    'KIDNEY_FUNCTION_TEST': '#99FF99',
    'KIDNEY_SURGERY': '#99FF99',
    'KIDNEY_DISEASE_COMPLICATION': '#99FF99',
    'CANCER_VITAL': '#FF99FF',
    'KIDNEY_INFECTION': '#99FF99',
    'STROKE_MEDICATION': '#FF9999',
    'CANCER_PROCEDURE': '#FF99FF',
    'STROKE': '#FF6666',
    'DIABETES_COMPLICATION': '#FFFF99',
    'LUNG_DISEASE': '#66CCFF',
    'DIABETES_FUNCTION_TEST': '#FFFF99',
    'DIABETES_PROCEDURE': '#FFFF99',
    'KIDNEY_DIALYSIS': '#99FF99',
    'LUNG_DISEASE_VITAL': '#66CCFF',
    'KIDNEY_DISEASE_PROCEDURE': '#99FF99',
    'ALZHEIMERS': '#6666FF',
    'CANCER_MEDIC': '#FF99FF',
    'CANCER': '#FF66FF',
    'ALZHEIMERS_VITAL': '#6666FF',
    'KIDNEY_DISEASE_VITAL': '#99FF99',
    'HEART_DISEASE': '#FF6666',
    'STROKE_TREATMENT': '#FF6666',
    'DIABETES_VITAL': '#FFFF99',
    'KIDNEY_DISEASE_MEDICATION': '#99FF99',
    'KIDNEY_DISEASE_TREATMENT': '#99FF99',
    'CANCER_TREATMENT': '#FF99FF',
    'KIDNEY_DISEASE': '#66FF66',
    'KIDNEY_FAILURE': '#66FF66',
    'LUNG_DISEASE_PROCEDURE': '#66CCFF',
    'ALZHEIMERS_PROCEDURE': '#6666FF',
    'DIABETES_MEDICATION': '#FFFF99',
    'ALZHEIMERS_FUNCTION_TEST': '#6666FF',
    'ALZHEIMERS_TREATMENT': '#6666FF',
    'KIDNEY_ANATOMY': '#99FF99',
    'ALZHEIMERS_MEDICATION': '#6666FF'
    }
}

def extract_top_sim_patients_live(patient_id, df, nlp) -> pd.DataFrame:
    """
    Identifies the top 10 similar patients to the provided patient ID.
    Calls to live_calculate_sim_index.

    Parameters:
    patient_id (int): The base patient ID the user is exploring (and wanting to find similar patients to!)
    df (pandas Dataframe): The lookup medical data (full data - will be subset by cluster based on provided patient ID)
    nlp: Loaded Medspacy model

    Returns:
    sim_index_df (pandas Dataframe): A dataframe with two columns: (1) Similar Patient ID (the top 10 patients requested by the user) 
                                    and (2) The similarity scores for each of those patients.
    """
    nlp = spacy.load("en_core_web_md")
    start = time.time()
    print("Calculating patient similarities... May take a minute or two...")

    # identify patient's cluster to subset the data
    patient_cluster = df.loc[df["subject_id"] == patient_id, "Cluster"].iloc[0]
    
    cluster_df = df.loc[df["Cluster"] == patient_cluster]
    
    # calculate similarities for the most similar patients
    keys = list(cluster_df.subject_id)
    keys.remove(patient_id)
    # create empty lists to store similarity values
    sims = [0.0 for _ in range(len(keys))]
    
    # create final Pandas DF
    sim_index_df = pd.DataFrame({"similar_patient_id": keys, "similarity_score": sims})
        
    # create base text for given patient id
    text = nlp.pipe(list(cluster_df["medical_summary"]))
    ref_df = pd.DataFrame({"subject_id": keys + [patient_id], "text": list(text)}) 
    text1 = ref_df.loc[ref_df["subject_id"] == patient_id, "text"].iloc[0]
        
    for patient in sim_index_df["similar_patient_id"]:
        # comparison text for all other patient ids
        text2 = ref_df.loc[ref_df["subject_id"] == patient, "text"].iloc[0]
        # calculate cosine similarity
        sim_index_df.loc[sim_index_df["similar_patient_id"] == patient, "similarity_score"] = text1.similarity(text2)

    # sort and return the top 10 patients specified
    sim_index_df = sim_index_df.sort_values(by=["similarity_score"], ascending=False).head(10).reset_index().drop(["index"], axis=1)

    end = time.time()
    print("Extract top similar patients execution time: ", end-start)
    
    return sim_index_df

@st.cache_data
def load_data():
    nlp = load_nlp()
    cluster_df = pd.read_csv('data/train_data_cluster_df.csv')
    df = pd.read_csv('data/train_data.csv')
    df = clean_service_type(df.copy())
    test_df = pd.read_csv('metrics_data/training_answer.csv')
    print(df.shape)
    #cluster_df = pd.read_csv('data/cluster_train_data.csv')
    return nlp, df, cluster_df, test_df

def process_text(nlp, text):
    doc = nlp(text)
    return doc

### ADD THIS FUNCTION TO EHR PROCESSOR & RUN TRANSFORM SEPARATELY ON EXISTING DATASETS ###
def clean_service_type(df):

    valid_service_types = ["orthopaedics", "medicine", "neurology", "surgery", "gynecology",
                           "cardiothoracic", "urology", "neurosurgery", "plastic", "psychiatry",
                           "podiatry", "otolaryngology", "emergency", "dental", "anesthesiology",
                           "radiology", "ophthalmology"]

    df["Service"] = df["Service"].astype(str)

    new_service_type = []
    for service_type in df["Service"]:
        # only replace if not already a valid type (after stripping trailing/leading whitespace)
        if service_type.strip() not in valid_service_types:
            # finds first mention of valid service types in longer string
            g = re.search(r"(?=(" + '|'.join(valid_service_types) + r"))", service_type)
            # if match is found, replace big string with the one word service type
            if g:
                new = g.group(0)
                # if blank entry, replace with "not specified"
                if not new:
                    new_service_type.append("not specified")
                else:
                    new_service_type.append(new)
            # if match not found, then the service type is categorized as "other"
            else:
                new_service_type.append("other")
        # keep if valid type
        else:
            new_service_type.append(service_type.strip())

    # replace existing Service column with new Service types
    updated_service_df = df.drop(columns="Service")
    updated_service_df["Service"] = new_service_type

    return updated_service_df

def main(colors):
    st.title('MedMap⚕️')
    st.header('The NLP-Driven Healthcare Visualization Tool')
    st.caption('Team 050: Stefan Lehman, Kelly Lewis, Ruby Truong, Ying Wang, Luke Williams')
    
    MainTab, Clustering, Visual, About = st.tabs(["Report","Patient Matcher","Visualization", "About"])
    nlp, df, cluster_df, test_df = load_data()
    print(type(cluster_df))

    with MainTab:
        st.write('The NLP-Driven Healthcare Visualization Tool')
        # Create a dropdown for selecting the sample index
        # index = st.selectbox("Select a sample index", options=range(len(df)), index=0)
        subject_ids = df['subject_id'].unique()

        selected_subject_id = st.selectbox("Select a subject ID", options=subject_ids, key=0)
        
        # Find the index of the selected subject_id in the dataframe
        index = df.index[df['subject_id'] == selected_subject_id].tolist()[0]
        colors = colors["colors"]
        
        st.markdown("<p style='font-size:20px; text-decoration: underline;'>Below are the details about this patient:</p>", unsafe_allow_html=True)
        
        # Display each column's content and overlay visuals for recognized entities
        for col in df.columns:
            cell_value = df.iloc[index][col]
            if isinstance(cell_value, str):
                #st.markdown(f"**{col}:**")
                doc = process_text(nlp, cell_value)
                labels = list({ent.label_ for ent in doc.ents})  # get the labels of recognized entities

                if labels:  # check if there are any recognized entities
                    visualize_ner(doc, labels=labels, title=f"{col}", key=f"{col}_{index}", colors=colors)
                else:
                    st.write(cell_value)  # display the original text if no entities are found
            else:
                st.markdown(f"**{col}:** {cell_value}")

            st.markdown("---")

            
    with Clustering: 
        st.title('Patient Matcher')
        st.markdown(''':blue[Disclaimer: Calculating patient similarities... May take a minute or two...]''')
        st.markdown("---")
        st.markdown('Identified similar patients to the provided patient ID based on Semantic/Language Similarities')
        st.markdown('Using the selected ID, you can look up the most similar patients to enable physicians to quickly identify treatment plans per cohort.')
        cluster_subject_ids = cluster_df['subject_id'].unique()
        print(cluster_df.columns)
        selected_cluster_subject_id = st.selectbox("Select a subject ID", options=cluster_subject_ids, key=1)
        new_df = extract_top_sim_patients_live(selected_cluster_subject_id, cluster_df, nlp)
        #st.dataframe(new_df)
        st.table(new_df)
        ls_sim = new_df['similar_patient_id'].tolist()
        st.markdown("---")
        st.title('Select Similar Patient ID')
        selected_sim_subject_id = st.selectbox("Select a similar subject ID", options=ls_sim, key=3)
        index = df.index[df['subject_id'] == selected_sim_subject_id].tolist()[0]
        
        cols_ls = ['Chief complaint','Major surgical or invasive procedure', 'History of present illness', 'Past medical history', 'Discharge medications', 'Discharge disposition', 'Discharge diagnosis']
        for col in cols_ls:
            cell_value = df.iloc[index][col]
            if isinstance(cell_value, str):
                #st.markdown(f"**{col}:**")
                doc = process_text(nlp, cell_value)
                labels = list({ent.label_ for ent in doc.ents})  # get the labels of recognized entities
                # print(labels)
                if labels:  # check if there are any recognized entities
                    visualize_ner(doc, labels=labels, title=f"{col}", key=f"{col}_{index}", colors=colors)
                else:
                    st.write(cell_value)  # display the original text if no entities are found
            else:
                st.markdown(f"**{col}:** {cell_value}")

            st.markdown("---")
        
    with Visual:
        st.write('Supplemental Visuals Profiling the EHR Data')
        st.markdown("This section provides some supplemental visuals to explore and summarize the "
                "current EHR data.")
        # Vis 1: Bar Graph"
        chronic_diseases = ["HEART_DISEASE", "CANCER", "LUNG_DISEASE", "STROKE", "ALZHEIMERS", "DIABETES",
                             "CHRONIC_KIDNEY_DISEASE"]

        nos = []
        yeses = []
        for col in chronic_diseases:
            print(test_df.columns)
            yeses.append(test_df.loc[test_df[col] == "Yes", col].count())
            nos.append(test_df.loc[test_df[col] == "No", col].count())

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=chronic_diseases,
            x=yeses,
            name="yes",
            orientation="h",
            marker_color="rgb(135, 206, 235, 1)" #skyblue
        ))

        fig.add_trace(go.Bar(
            y=chronic_diseases,
            x=nos,
            name="no",
            orientation="h",
            marker_color="rgb(128, 128, 128, 1)" #gray
        ))

        fig.update_layout(title="Breakdown of Chronic Disease Across All Patient EHRs",
                          xaxis_title="Count of Patients", barmode="stack")

        st.markdown("The following horizontal bar plot shows the number of patients with and without the specified condition "
                     "across the EHR data. Patients who have the condition are marked 'yes' and patients who don't are marked "
                     "'no'.")
        st.plotly_chart(fig)
        ## "Visualization 2: histogram"
        service_types = df["Service"].unique()
        selected_service_type = st.selectbox("Select a Service type", options=service_types, key=2)

        fig = px.histogram(df[df["Service"] == selected_service_type], x="num_prior_office_visits",
                           title=f"Histogram of Prior Office Visits for {selected_service_type}",
                           color="gender", color_discrete_map={"f": "mediumpurple", "m": "skyblue"})

        st.markdown("The following histogram shows the number of prior office visits for each patient by gender "
                    "based on the selected Service type (orthopaedics, neurology, medicine, etc.).")
        st.plotly_chart(fig)
    
    with About:
        with st.container():
            st.markdown("""- This tool uses advanced Natural Language Processing (NLP) techniques to transform raw clinical data into interactive visual reports for healthcare professionals and stakeholders, simplifying time-consuming interactions with EHRs and improving clinical productivity.""")
            
            st.markdown("""- It also enables users to draw comparisons among patients with similar conditions, enhancing clinical decision-making and personalized care.""")

if __name__ == "__main__":
    main(colors)