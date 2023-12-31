import spacy
from spacy.pipeline import EntityRuler
from sklearn.model_selection import train_test_split
from spacy.tokens import DocBin
from spacy import displacy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import requests
import wikipediaapi
import streamlit as st
import urllib.parse

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

def get_wikipedia_article(input_string):
    """
    Retrieve the text of a Wikipedia article based on its title or URL.

    If the input is a valid Wikipedia article URL, the function parses the title
    from the URL. Otherwise, it treats the input as a page title. It then uses
    the wikipediaapi library to fetch the content of the Wikipedia article.
    It returns the text of the article if it exists, otherwise it returns None.

    Parameters:
    input_string (str): The title or URL of the Wikipedia article to retrieve.

    Returns:
    str or None: The text of the Wikipedia article, or None if the page does not exist.
    """
    if input_string.startswith("http://") or input_string.startswith("https://"):
        # Input is a URL, extract the title
        parsed_url = urllib.parse.urlparse(input_string)
        if 'wikipedia.org' not in parsed_url.netloc:
            raise ValueError("URL does not belong to Wikipedia.")
        
        title = parsed_url.path.split('/')[-1]
        title = urllib.parse.unquote(title)  # Decode URL-encoded title
    else:
        # Input is assumed to be a page title
        title = input_string

    # Fetch the article text
    page = wiki_wiki.page(title)
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

# Function to convert DataFrame to CSV
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')
    
def filter_doc_for_chemical_compounds(doc, df_filtered):
    """
    Create a new spaCy Doc containing only entities identified as chemical compounds.

    Parameters:
    doc (spacy.Doc): Original spaCy Doc object with NER annotations.
    df_filtered (pd.DataFrame): DataFrame containing filtered entities identified as chemical compounds.

    Returns:
    spacy.Doc: New spaCy Doc object with only chemical compound entities.
    """
    from spacy.tokens import Span

    # Filter entities based on DataFrame
    filtered_entities = []
    for ent in doc.ents:
        if ent.text in df_filtered['Entity'].values:
            filtered_entities.append(Span(doc, ent.start, ent.end, label=ent.label_))

    # Create a new Doc with the filtered entities
    filtered_doc = spacy.tokens.Doc(doc.vocab, words=[t.text for t in doc], spaces=[t.whitespace_ for t in doc])
    filtered_doc.ents = filtered_entities
    return filtered_doc

def visualize_ner(filtered_doc):
    # Define colors for chemical entity types
    entity_colors = {
        "ALKANE": "linear-gradient(90deg, #ffadad, #ffd6a5)",
        "ALKENE": "linear-gradient(90deg, #fdffb6, #caffbf)",
        "ALKYNE": "linear-gradient(90deg, #9bf6ff, #a0c4ff)",
        "ALDEHYDE": "linear-gradient(90deg, #bdb2ff, #ffc6ff)",
        "KETONE": "linear-gradient(90deg, #fffffc, #a5f2e9)",
        "ALCOHOL": "linear-gradient(90deg, #ffafcc, #f9c6aa)",
        "C_ACID": "linear-gradient(90deg, #8ecae6, #219ebc)"
    }
    options = {"ents": list(entity_colors.keys()), "colors": entity_colors}
    html = displacy.render(filtered_doc, style="ent", options=options)
    return html

def create_and_save_count_plot(df, column_name, filename='plot.png', width=6, height=4):
    plt.figure(figsize=(width, height))
    sns.countplot(x=column_name, data=df, palette='rocket')
    plt.title('Count of NER Labels')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()  # Close the plot to free up memory
    
#Initialize ChemNER Model
chemner = spacy.load("en_chemner")

# Define user agent for Wikipedia API
user_agent = "ChemNER/1.0 (victor.murcia@wsu.edu)"

# Initialize Wikipedia API and spaCy
wiki_wiki = wikipediaapi.Wikipedia(user_agent,'en')

# List of specific chemical compound types
chemical_compounds = ['alkane', 'alkene', 'alkyne', 'ketone', 'aldehyde', 'alcohol', 'carboxylic acid']

#Set page layout
st.set_page_config(layout="wide")

# Streamlit app layout
st.title("ChemNER: An NER Model for Extracting Chemical Compounds From Text")

# Checkbox for user to choose between raw input or Wikipedia URL
is_wikipedia_url = st.checkbox("Fetch text from Wikipedia URL")

# User input for text or Wikipedia URL based on checkbox
if is_wikipedia_url:
    user_input = st.text_input("Enter Wikipedia article URL:")
else:
    user_input = st.text_area("Enter text directly:")

# Button to run the process
if st.button("Run NER"):
    if is_wikipedia_url:
        # Fetch text from Wikipedia if checkbox is selected
        text = get_wikipedia_article(user_input)
    else:
        # Use raw user input
        text = user_input

    #Check whether an article was found for the given search term
    if text:
        # Perform NER on the article content
        entities = run_chemner(text)
        #html = visualize_ner(text)
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

    # Creating two columns for side by side display
    col1, col2 = st.columns(2)

    with col1:
        st.header("NER Dataframe")
        # Display the final DataFrame
        st.write(df_merged)

    with col2:
        st.header("NER Label Distribution")
        
        # Generate and display the plot
        # Generate and save the plot as an image
        create_and_save_count_plot(df_merged, 'Label', filename='plot.png', width=10, height=8)
        st.image('plot.png',width=500)

    st.title("NER Visualization")
    # Create a filtered Doc for visualization
    doc = chemner(text)
    filtered_doc = filter_doc_for_chemical_compounds(doc, df_merged)
    html = visualize_ner(filtered_doc)
    st.markdown(html, unsafe_allow_html=True)

    # After processing and obtaining results_df
if 'df_merged' in locals():  # Check if results_df exists
    csv = convert_df_to_csv(df_merged)
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name='chemner_results.csv',
        mime='text/csv',
    )
