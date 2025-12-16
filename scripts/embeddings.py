from importlib.metadata import metadata
import json , pickle , numpy as np , faiss
from sentence_transformers import SentenceTransformer

DATA_FILE = 'data/cleaned_catalog.json'
METADATA_FILE = 'data/metadata.pkl'
FAISS_INDEX_FILE = 'data/faiss.index'

def main():
    
    #Here loading the data
    with open(DATA_FILE , 'r', encoding='utf-8') as f:
        data = json.load(f)
    

    texts = [item['text'] for item in data]

     #Here loading the embedding model 
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')

    print("Generating embeddings...")
    embeddings = MODEL.encode(texts, show_progress_bar=True)

    
    embeddings = np.array(embeddings).astype('float32')

    #create Faiss index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print("Total vectors in index: " , index.ntotal)
    
    faiss.write_index(index , FAISS_INDEX_FILE)

    metadata = [
        {
            "name" : item['name'],
            "url" : item['url'],
            "test_type" : item['test_type'],
            "duration" : item['duration'],
            "adaptive_support" : item['adaptive_support'],
            "remote_support" : item['remote_support']
        }
        for item in data
    ]

    with open(METADATA_FILE , 'wb') as f:
        pickle.dump(metadata , f)

    print("Embeddings and metadata saved successfully.")

if __name__ == '__main__':
    main()
