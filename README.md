# MS-Analytics-Projects
A portfolio of advanced academic projects completed during the Georgia Tech Master of Science in Analytics program. 

## 1) [MEDMAP: The NLP-Driven Healthcare Visualization Tool](Healthcare-NLP-Decision-Support/)

**[🎬 WATCH DEMO](https://www.youtube.com/watch?v=sdoLxDfRqGM) Healthcare NLP Overview and Results Walkthrough**

**FINAL REPORT: [Technical Report](Healthcare-NLP-Decision-Support/Docs/team050report.pdf) and [Final Presentation Slides](Healthcare-NLP-Decision-Support/Docs/team050poster.pdf)**
<details>
<summary>View Detailed Project Description (Click to Expand)</summary>
	
### DESCRIPTION ###
MedMap is a user-friendly NLP-driven interactive healthcare visualization tool that enables healthcare providers to quickly access patient information with key word highlighting and draw comparisons among patients with similar conditions to enhance clinical decision-making and targeted treatment plans. MedMap aims to simplify time-consuming interactions with Electronic Health Records (EHRs) to improve clinical productivity and patient and physician satisfaction.

MedMap contains three primary features in the tool's user-interface split out by tab: 

(a) a report tab that displays a patient's highlighted medical data based on the MedSpacy NLP model, 

(b) a patient matcher tab that displays the most similar patients to the selected patient and additional similar patient data, and 

(c) a visualization tab that displays summary information including chronic disease breakdowns, types of medical service, and number of prior office visits to characterize the provided EHR data.

### INSTALLATION ###
In a Conda environment:
1) Make a new conda environment:
	conda create -n myenvname
2) Activate the new environment:
	conda activate myenvname
3) Install pip for conda:
	conda install pip
4) Install medspacy:
	pip install spacy medspacy
	python -m spacy download en_core_web_md
If this doesn't work, try installing the wheel package:
pip install https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl
5) Install the provided requirements.txt file (navigate to the main code location):
	pip install -r requirements.txt

In any IDE (such as Pycharm, etc.):
1) Create a new environment.
2) Install medspacy using pip or the wheel provided above.
3) Install provided requirements.txt file as described above.

These packages should be compatible with Python 3.9-3.11. If your version of Python isn't working, try one of the other versions.

### EXECUTION ###

COMPONENTS

-[data folder](Healthcare-NLP-Decision-Support/Code/data/): contains the input data file and will store output data as the pipeline is executed

-[model_objects folder](Healthcare-NLP-Decision-Support/Model_Objectives/): contains the clustering model object

-[abbreviations.json](Healthcare-NLP-Decision-Support/Code/abbreviations.json): a mapping file to consolidate spelled-out words and abbreviations in the clinical text data

-[requirements.txt](Healthcare-NLP-Decision-Support/Code/requirements.txt): contains required packages for MedMap to run

-[EHRExecutor.py](Healthcare-NLP-Decision-Support/Code/EHRExecutor.py): Computes clusters for PatientMatcher feature of MedMap (imported by ehr_runner notebook)

-[EHRProcessor.py](Healthcare-NLP-Decision-Support/Code/EHRProcessor.py): Performs NLP pre-processing and transformations on the data to get it ready for modeling (imported by ehr_runner notebook)

-[ehr_runner_and_clustering.ipynb](Healthcare-NLP-Decision-Support/Code/ehr_runner_and_clustering.ipynb): Jupyter notebook that runs EHRExecutor.py and EHRProcessor.py to generate required data and model objects as required inputs for the MedMap UI

-[TargetMatcher.py](Healthcare-NLP-Decision-Support/Code/TargetMatcher.py): Contains custom chronic disease rules for MedSpacy (imported by SpacyUI.py)

-[SpacyUI.py](Healthcare-NLP-Decision-Support/Code/SpacyUI.py): Spins up the UI for the MedMap Tool (final product)

HOW TO RUN

1) Run ehr_runner_and_clustering.ipynb to clean, preprocess, and generate train/test data and compute clusters for the PatientMatcher feature. This will call to the EHRExecutor.py and EHRProcessor.py files. All required outputs will show up in the data and model_objects folders and these will be loaded in appropriately when the SpacyUI is executed.
   
2) Run SpacyUI.py to spin up the MedMap tool in a UI window.
   
--> a) Open a terminal.

--> b) Navigate to the folder this py file is contained in.

--> c) Type the command: streamlit run SpacyUI.py

3) A new browser window will open with a "Running..." icon in the upper right corner. Wait a few seconds for all the tabs to show up. 

NAVIGATING THE MEDMAP UI

--> Tab 1: Report -- This tab allows the user to type in a Patient ID and displays that patient's clinical data using the MedSpacy highlighting feature for easy access of key information and visual skimming.

--> Tab 2: Patient Matcher -- This tool allows the user to type in a Patient ID and displays the most similar patients (computed using a clustering algorithm, TFIDF vectorizaton, and cosine similarity). Another dropdown allows the user to select a similar patient and displays that similar patient's clinical data on the Patient Matcher tab to compare to the main patient's clinical data on the Report tab.

--> Tab 3: Visualization -- This tab provides two primary visuals: (1) A bar graph of chronic disease breakdown within the provided EHR data showing counts of patients with and without these conditions and (2) A histogram of number of prior office visits broken out by gender with an option for the user to select different Service types.

--> Tab 4: About -- This tab provides a description of each feature of the MedMap tool.

</details>

## 2) [Weather-Driven Sentiment Analysis using Machine Learning](BAM/BAM_Final_Report.pdf)
**Balyasny Asset Management (BAM) Final Report**

*Note: Due to firm confidentiality, this document contains the final findings, methodology, and results, but **does not include the proprietary code or raw data***.

