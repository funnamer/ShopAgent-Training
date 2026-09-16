#!/bin/bash
# Install Python Dependencies
pip install -r requirements.txt;

# Download spaCy large NLP model
python -m spacy download zh_core_web_sm
