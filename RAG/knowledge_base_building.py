import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import os

os.chdir('RAG/')

with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

knowledge_base_file = config['knowledge_base_file']
processed_data_file = config['processed_data_file']
faiss_index_file = config['faiss_index_file']
keyword_column = config['keyword_column']
description_column = config['description_column']
command_column = config['command_column']
sheet_name = config['sheet_name']

def ensure_dir_exists(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

ensure_dir_exists(processed_data_file)
ensure_dir_exists(os.path.dirname(faiss_index_file))

try:
    data_df = pd.read_excel(knowledge_base_file, sheet_name=sheet_name)
except Exception as e:
    print(f"Error reading the sheet {sheet_name} from the Excel file {knowledge_base_file}: {e}")
    raise

def preprocess(text):
    if config['preprocessing'].get('strip', True):
        text = text.strip()
    if config['preprocessing'].get('lowercase', False):
        text = text.lower()
    return text

for column in [keyword_column, description_column, command_column]:
    if column in data_df.columns:
        data_df[column] = data_df[column].astype(str).apply(preprocess)
    else:
        print(f"Warning: Column '{column}' does not exist in the sheet '{sheet_name}'.")
        raise KeyError(f"Column '{column}' does not exist in the sheet '{sheet_name}'.")

data = data_df.to_dict(orient='records')

try:
    model = SentenceTransformer(config['model_name'])
except Exception as e:
    print(f"Error loading Sentence-BERT model {config['model_name']}: {e}")
    raise

descriptions = [item[description_column] for item in data]
try:
    embeddings = model.encode(descriptions, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
except Exception as e:
    print(f"Error generating embeddings: {e}")
    raise

for item, embedding in zip(data, embeddings):
    item['embedding'] = embedding.tolist()

dimension = embeddings.shape[1]
try:
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
except Exception as e:
    print(f"Error building FAISS vector index: {e}")
    raise

try:
    faiss.write_index(index, faiss_index_file)
except Exception as e:
    print(f"Error saving FAISS index to {faiss_index_file}: {e}")
    raise

try:
    with open(processed_data_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
except Exception as e:
    print(f"Error saving processed data to {processed_data_file}: {e}")
    raise

print("The knowledge base has been successfully generated.")
