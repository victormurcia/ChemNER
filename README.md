# ChemNER
A custom NER model for extracting and labeling chemical compounds from text

# Installation
!pip install https://huggingface.co/victormurcia/en_chemner/resolve/main/en_chemner-any-py3-none-any.whl

## Using spacy.load().
import spacy
nlp = spacy.load("en_chemner")

## Importing as module.
import en_chemner
nlp = en_chemner.load()
