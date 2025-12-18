# SHL-RAG-Assessment

This project is a Recommendation System for SHL Assessments, in which we have to use the data just by scraping it from the SHL produc catalog page then making its embeddings which will be act as a vector in the vector DB and then using LLM of langchain_google_genai to generate or retriveve the info.

# Response 

First Request maybe slower due to cold start.

## Frontend

Frontend is a simple html page with some css and js.
[**LINK**](https://shl-rag-assessment-1.onrender.com)

## Features
- **Scraping**: Automated scraper to fetch assessment data from SHL website by using beautifulsoup.
- **Data Cleaning**: Normalizes and structures the raw data.
- **Vector Search**: Uses FAISS and Sentence Transformers for semantic search.
- **RAG Integration**: Uses Google Gemini (via LangChain) to generate personalized explanations for recommendations.
- **FastAPI**: High-performance backend API.
- **Deployment**: Render is used for deployment and which automatically  deployes latest code

## Setup

1. **Clone the repository** (or navigate to the project directory).

2. **Install Dependencies**:
   pip install -r requirements.txt

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

**Endopoint**: `GET https://shl-rag-assessment-c30j.onrender.com/health` # this is for healthcheck


**Response**:
```json
{
  "status": "ok"
}
```

**Endpoint**: `POST https://shl-rag-assessment-c30j.onrender.com/recommend` # this is for recommendations

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

#FIXED

1. Fixed the of memory full on render.com
2. using Qdrant for vector DB which stores all the embeddings online 