import spacy
from spacy.pipeline import EntityRuler
from sklearn.model_selection import train_test_split
from spacy.tokens import DocBin
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import requests
import wikipediaapi
import streamlit as st

def get_compound_info(compound_name):
    """
    Query the PubChem API to check if a compound exists.

    This function sends a request to the PubChem API for a given compound name.
    It returns 1 if the compound is found (status code 200) and 0 otherwise.

    Parameters:
    compound_name (str): The name of the compound to query.

    Returns:
    int: 1 if the compound is found in PubChem, 0 if not.
    """
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
    response = requests.get(f"{base_url}/{compound_name}/JSON")
    return 1 if response.status_code == 200 else 0

def query_pubchem(row):
    """
    Apply the get_compound_info function to a DataFrame row.

    This function is intended to be used with the DataFrame.apply() method.
    It takes a row of a DataFrame, extracts the 'Entity' field, and queries
    the PubChem API to check if this entity is a known chemical compound.

    Parameters:
    row (pd.Series): A row of a pandas DataFrame.

    Returns:
    int: 1 if the entity is found in PubChem, 0 if not.
    """
    return get_compound_info(row['Entity'])

def get_wikipedia_article(page_title):
    """
    Retrieve the text of a Wikipedia article based on its title.

    This function uses the wikipediaapi library to fetch the content of a
    Wikipedia article. It returns the text of the article if it exists,
    otherwise it returns None.

    Parameters:
    page_title (str): The title of the Wikipedia article to retrieve.

    Returns:
    str or None: The text of the Wikipedia article, or None if the page does not exist.
    """
    page = wiki_wiki.page(page_title)
    return page.text if page.exists() else None

def run_chemner(text):
    """
    Perform Named Entity Recognition (NER) with ChemNER model on a given text.

    This function uses a pre-loaded NLP model to identify and extract entities
    from a given text. The entities are returned along with their labels.

    Parameters:
    text (str): The text on which to perform NER.

    Returns:
    list of tuples: A list where each tuple contains an entity and its corresponding label.
    """
    doc = chemner(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def update_chemical_compound(row):
    """
    Update the 'Chemical Compound' column based on the entity's match with specific compound types.

    This function checks if the entity in a given row matches any predefined chemical compound types
    (like alkane, alkene, etc.) or their plural forms. If a match is found, it sets the 'Chemical Compound' 
    value to 1. If no match is found, it retains the original value from the 'Chemical Compound' column.

    Parameters:
    row (pd.Series): A row from a pandas DataFrame, expected to contain 'Entity' and 'Correct' columns.

    Returns:
    int: 1 if the entity matches a predefined chemical compound type, the original 'Correct' value otherwise.
    """
    entity = row['Entity'].lower()
    if any(compound in entity for compound in chemical_compounds + [c + 's' for c in chemical_compounds]):
        return 1
    return row['Chemical Compound']

#Initialize ChemNER Model
chemner = spacy.load("en_chemner")

# Define user agent for Wikipedia API
user_agent = "ChemNER/1.0 (victor.murcia@wsu.edu)"

# Initialize Wikipedia API and spaCy
wiki_wiki = wikipediaapi.Wikipedia(user_agent,'en')

# List of specific chemical compound types
chemical_compounds = ['alkane', 'alkene', 'alkyne', 'ketone', 'aldehyde', 'alcohol', 'carboxylic acid']

# Streamlit app layout
st.title("ChemNER: An NER Model for Extracting Chemical Compounds From Text")

# User input for text or Wikipedia URL
user_input = st.text_input("Enter text or Wikipedia article URL:")

# Button to run the process
if st.button("Run NER"):
    # Check if input is a Wikipedia URL or text
    if "wikipedia.org" in user_input:
        text = get_wikipedia_article(user_input)
    else:
        text = user_input

    #Check whether an article was found for the given search term
    if text:
        # Perform NER on the article content
        entities = run_chemner(text)
        st.write('Article found')
    else:
        st.write('Article not found')

    # Create DataFrame to hold entities and their labels
    df = pd.DataFrame(entities, columns=['Entity', 'Label'])

    # Removing duplicate entries in the DataFrame for faster queries to PubChem
    df_unique = df.drop_duplicates(subset=['Entity']).reset_index(drop=True)

    # Query PubChem using each Entity to classify to determine whether they are a chemical compound
    df_unique['Chemical Compound'] = df_unique.apply(query_pubchem, axis=1)

    # Ensure that chemical classes are classified as compounds as well
    df_unique['Chemical Compound'] = df_unique.apply(update_chemical_compound, axis=1)

    #Merge the original dataframe with the query dataframe
    df_merged = pd.merge(df, df_unique[['Entity', 'Chemical Compound']], on='Entity', how='left')

    # Dropping rows where 'Chemical Compound' is 0
    df_merged = df_merged[df_merged['Chemical Compound'] != 0].reset_index()

    # Display the final DataFrame
    st.write(df)
