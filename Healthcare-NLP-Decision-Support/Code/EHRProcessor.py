import pandas as pd
import re
import json
from sklearn.model_selection import train_test_split
import sqlite3
import os


class EHRProcessor:
    def __init__(self, data_file='data/ehr.csv', abbreviations_file='abbreviations.json'):
        with open(abbreviations_file, 'r') as f:
            self.abbreviations = json.load(f)
        self.abbreviations_lower = {k.lower(): v.lower() for k, v in self.abbreviations.items()}
        self.data_file = data_file
        self.abbreviations_file = abbreviations_file
        self.pattern_family_history = r"(?:family history(?: history)?)(?::)?(?:.*?)(?=\n\S|$)"
        self.pattern_physical_exam = r"physical _*|physical exam: physical exam:|physical examination: physical examination:|physical examination:|physical exam:"

    def parse_ehr_text(self, ehr_text: str) -> dict:
        """
        Parses the unstructured Electronic Health Record (EHR) text into structured data.

        Parameters:
        ehr_text (str): Unstructured text of the Electronic Health Record (EHR).

        Returns:
        dict: Structured data extracted from the EHR text. Each section of the EHR is represented as a key-value pair
              in the dictionary, with the section name as the key and its corresponding content as the value.
        """
        ehr_text_lower = ehr_text.lower()

        # mapping of common variations and typos of sections in ehr
        section_mapping = {
            "chief complaint": "Chief Complaint",
            "major surgical or invasive procedure": "Major Surgical or Invasive Procedure",
            "history of present illness": "History of Present Illness",
            "past medical history": "Past Medical History",
            "medications on admission": "Medications on Admission",
            "discharge medications": "Discharge Medications",
            "discharge disposition": "Discharge Disposition",
            "discharge diagnosis": "Discharge Diagnosis",
            "discharge condition": "Discharge Condition",
            "discharge instructions": "Discharge Instructions",
            "followup instructions": "Followup Instructions",
            "allergies": "Allergies",
            "service": "Service"
        }

        # regex pattern for splitting the sections
        section_headers_pattern = r"\n(" + "|".join(section_mapping.keys()) + "):"
        sections = re.split(section_headers_pattern, ehr_text_lower, flags=re.IGNORECASE)
        structured_data = {}
        current_section = "De_Identified"

        for i, section in enumerate(sections):
            # even indices contain the headers
            if i % 2 == 0:
                section_text = re.sub(r'\s+', ' ', section.strip())

                if section_text:
                    # adding to dictionary if there's any content
                    structured_data[current_section] = section_text
            else:
                # odd indices contain the content per section
                current_section = section_mapping.get(section.strip(), section.strip()).capitalize()
                if current_section not in structured_data:
                    structured_data[current_section] = ""

        # Extract family history from the text
        cleaned_text = re.split(self.pattern_family_history, ehr_text_lower)
        fam_part = cleaned_text[-1].strip()

        # Remove physical examination section
        cleaned_text = re.split(self.pattern_physical_exam, fam_part)
        family_history_text = cleaned_text[0].strip()

        structured_data["Family History"] = family_history_text

        return structured_data

    def replace_abbreviations_with_regex(self, text):
        """
        Replace abbreviations in the given text with their full forms based on the provided dictionary.

        This function searches the text for any words that match keys in the `abbreviations` dictionary from json
        (case-insensitive) and replaces them with the corresponding values from the dictionary. If the input
        is not a string, it returns the input as is.

        Returns:
        str: The text with abbreviations replaced by their full forms as specified in `abbreviations` dictioanry.
        """
        if not isinstance(text, str):
            return text
        pattern = re.compile(r'(?i)\b(?:' + '|'.join(map(re.escape, self.abbreviations_lower.keys())) + r')\b')

        return pattern.sub(lambda m: self.abbreviations_lower[m.group().lower()], text)
    
    def process_text(self, text):
        if pd.isna(text) or text is None:
            return ""  # Replace with an empty string or a suitable default value
        else:
            return str(text)
    
    def remove_family_history(self, text):
        # Remove family history section
        cleaned_text = re.split(self.pattern_family_history, text)
        first_part = cleaned_text[0].strip()

        # Remove physical examination section
        cleaned_text = re.split(self.pattern_physical_exam, text)
        second_part = cleaned_text[-1].strip()

        if first_part:
            if second_part not in first_part:
                return first_part + ' ' + "physical examination: " + second_part
            else:
                return first_part
        else:
            return "physical examination: " + second_part if second_part else ''
    
    def apply_abbreviation_converter(self, master_ehr_df):
        """
        Applies abbreviation replacement to specified columns in a dataframe.

        Parameters:
        master_ehr_df (pd.DataFrame): The dataframe to process.
        """
        columns_to_replace = [
            'note_id', 'subject_id', 'hadm_id', 'note_type', 'Service', 'Allergies', 'Chief complaint',
            'Major surgical or invasive procedure', 'History of present illness', 'Past medical history',
            'Medications on admission', 'Discharge medications', 'Discharge disposition', 'Discharge diagnosis',
            'Discharge condition', 'Discharge instructions', 'Followup instructions', 'num_prior_office_visits',
            'gender', 'Family History'
        ]

        for column in columns_to_replace:
            if column in master_ehr_df.columns:
                master_ehr_df[column] = master_ehr_df[column].astype(str)
                master_ehr_df[column] = master_ehr_df[column].apply(self.process_text)
                if column == 'Past medical history':
                    master_ehr_df[column] = master_ehr_df[column].apply(self.remove_family_history)
                master_ehr_df[column] = master_ehr_df[column].apply(self.replace_abbreviations_with_regex)

        master_ehr_df.to_csv('data/ehr_sample_w_sections_and_abbreviations.csv', index=False)
        return master_ehr_df

    def create_master_ehr_dataframe(self, csv_file_path: str) -> pd.DataFrame:
        """
        Creates a master DataFrame of Electronic Health Records (EHR) from a CSV file containing unstructured text data.

        Parameters:
        csv_file_path (str): The file path to the CSV file containing EHR data.

        Returns:
        pandas.DataFrame: A DataFrame containing structured EHR data. Each row corresponds to an EHR entry, with columns
                          representing different sections of the EHR such as chief complaint, medical history, medications, etc.
                          The 'note_id' column is used as the primary key to join the sections with the original EHR entries.
        """
        ehr_full_df = pd.read_csv(csv_file_path)

        parsed_ehr_dicts = ehr_full_df['text'].apply(self.parse_ehr_text)
        parsed_ehr_section_df = pd.DataFrame(list(parsed_ehr_dicts))
        # using content from note_id to map to original dataframe with the parsed_ehr_dicts
        parsed_ehr_section_df['note_id'] = ehr_full_df['note_id']

        # merging sections
        master_ehr_df = pd.merge(ehr_full_df, parsed_ehr_section_df, on='note_id', how='outer')
        master_ehr_df.drop('text', axis=1, inplace=True)

        return master_ehr_df

    def clean_data(self, master_ehr_df) -> pd.DataFrame:
        """
        Creates a cleaned DataFrame of Electronic Health Records (EHR) from a master Dataframe containing parsed and structured data.

        Parameters:
        master_ehr_df (pd.Dataframe): Master DataFrame containing EHR data.

        Returns:
        cleaned_ehr_df (pd.DataFrame): A DataFrame containing cleaned structured EHR data. Irrelevant or redacted columns are removed and only the latest EHR visit is kept per patient, with the addition of a column with the number of  =prior visits.
        """
        # identifying the total number of office visits
        office_visits_df = master_ehr_df.groupby('subject_id')['note_seq'].count()

        # identifying latest office visit per patient
        reduced_indices = master_ehr_df.groupby('subject_id')['note_seq'].idxmax()

        # reduce dataset to one office visit (latest) per patient
        reduced_df = master_ehr_df.loc[reduced_indices.values]
        reduced_df["num_prior_office_visits"] = office_visits_df.values - 1

        df = reduced_df.drop(columns=["note_seq", "charttime", "storetime"])

        # extract gender from De_Identified column
        genders = []
        for text in df["De_Identified"]:
            g = re.match('[\w\W]+sex:\s*([a-z]+)', text)
            if g:
                gender = g.group(1)
                genders.append(gender)
            else:
                genders.append("")

        # create new gender column and drop old de_identified column since all other info is redacted
        df["gender"] = genders
        cleaned_ehr_df = df.drop(columns=["De_Identified"])

        return cleaned_ehr_df
    
    def create_sqlite_table(self, df, table_name, conn):
        df.to_sql(table_name, conn, if_exists='replace', index=False)
    
    def create_sqlite_db(self, data, db_file):
        conn = sqlite3.connect(db_file)
        self.create_sqlite_table(data, 'data', conn)
        conn.commit()
        conn.close()
    
    def create_split(self, df):
        train_data, test_data = train_test_split(df, test_size=0.3, random_state=6242)
        test_data, validation_data = train_test_split(test_data, test_size=0.5, random_state=6242)
        train_data.to_csv('data/train_data.csv', index=False)
        print('Training data size:', train_data.shape[0])
        test_data.to_csv('data/test_data.csv', index=False)
        print('Testing data size:', test_data.shape[0])
        validation_data.to_csv('data/validate_data.csv', index=False)
        print('Validation data size:', validation_data.shape[0])
        
        train_db_file = 'data/train_data.db'
        test_db_file = 'data/test_data.db'
        validation_db_file = 'data/validation_data.db'
        
        os.makedirs('data', exist_ok=True)
        
        self.create_sqlite_db(train_data, train_db_file)
        print(f'Splitting training data, saving to {train_db_file}')
        self.create_sqlite_db(test_data, test_db_file)
        print(f'Splitting testing data, saving to {test_db_file}')
        self.create_sqlite_db(validation_data, validation_db_file)
        print(f'Splitting validation data, saving to {validation_db_file}')
        print('Created split')
        
    def run(self):
        self.master_ehr_df = self.create_master_ehr_dataframe(self.data_file)
        self.cleaned_ehr_df = self.clean_data(self.master_ehr_df)
        self.cleaned_ehr_w_abbreviations_df = self.apply_abbreviation_converter(self.cleaned_ehr_df)
        self.create_split(self.cleaned_ehr_w_abbreviations_df)

if __name__ == '__main__':
    data_file = 'data/ehr_data.csv'
    abbreviations_file = 'abbreviations.json'
    ehr_processor = EHRProcessor(data_file, abbreviations_file)
    ehr_processor.run()