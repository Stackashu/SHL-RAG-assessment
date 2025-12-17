# SHL-RAG-Assessment

This project is a Recommendation System for SHL Assessments, in which we have to use the data just by scraping it from the SHL produc catalog page then making its embeddings which will be act as a vector in the vector DB and then using LLM of langchain_google_genai to generate or retriveve the info.

## Features
- **Scraping**: Automated scraper to fetch assessment data from SHL website by using beautifulsoup.
- **Data Cleaning**: Normalizes and structures the raw data.
- **Vector Search**: Uses FAISS and Sentence Transformers for semantic search.
- **RAG Integration**: Uses Google Gemini (via LangChain) to generate personalized explanations for recommendations.
- **FastAPI**: High-performance backend API.


## Setup


3. **Environment Setup**:
   Must use below api key
   GEMINI_API_KEY=your_actual_api_key_here

## Pipeline Execution

Follow these steps to rebuild the data from scratch:

### 1. Scraping
Fetch the latest assessment data from SHL.

python scraper/scrape_shl.py
*Output: `data/catalog.json` (or `raw_catalog.json` depending on script version)*

### 2. Data Cleaning
Normalize the scraped data for embedding.
python scripts/cleaning_data.py
*Output: `data/cleaned_catalog.json`*

### 3. Embeddings
Generate vector embeddings for the cleaned data.
python scripts/embeddings.py
*Output: `data/faiss.index` and `data/metadata.pkl`*

## Running the Application

Start the backend server:
uvicorn app:app --reload
The API will run at `http://127.0.0.1:8000` 

## Usage

**Endopoint**: `GET /health` // this is for healthcheck

**Endpoint**: `POST /recommend`

**Request**:
```json
{
  "query": "I need a test for a Python developer"
}
```

**Response**:
```json
{
  "recommendations": [ ... ],
  "ai_analysis": "Based on your request..."
}
```

