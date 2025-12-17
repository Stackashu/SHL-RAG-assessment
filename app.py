from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.retriever import search

app = FastAPI()

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
    
    # Generate natural language response
    from backend.llm import generate_response
    llm_answer = generate_response(req.query, results)

    response = []
    for r in results:
        response.append({
            "name": r.get("name"),
            "url": r.get("url"),
            "test_type": r.get("test_type") or "Unknown",
            "description": r.get("description") or "Not available",
            "duration": r.get("duration") or "Not specified",
            "adaptive_support": r.get("adaptive_support") or "Unknown",
            "remote_support": r.get("remote_support") or "Unknown"

        })

    return {"recommendations": response, "ai_analysis": llm_answer}
