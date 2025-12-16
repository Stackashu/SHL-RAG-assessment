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

    results = search(req.query, top_k=10)

    response = []
    for r in results:
        response.append({
            "assessment_name": r["name"],
            "assessment_url": r["url"]
        })

    return {"recommendations": response}
