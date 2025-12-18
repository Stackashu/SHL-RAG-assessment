from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.retriever import search
from backend.llm import generate_response

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query is empty")

    results = search(req.query, top_k=5)

    # 🔒 LLM should NEVER break API
    try:
        llm_answer = generate_response(req.query, results)
    except Exception as e:
        print("LLM error:", e)
        llm_answer = "AI explanation temporarily unavailable."

    response = []
    for r in results:
        response.append({
            "name": r.get("name", ""),
            "url": r.get("url", ""),
            "test_type": r.get("test_type", ""),
            "duration": r.get("duration", ""),
            "adaptive_support": r.get("adaptive_support", ""),
            "remote_support": r.get("remote_support", ""),
            "description": r.get("description", ""),
        })

    return {
        "recommendations": response  }
