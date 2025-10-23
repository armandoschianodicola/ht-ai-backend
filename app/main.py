from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_pipeline import run_rag

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_rag(request: QueryRequest):
    answer = run_rag(request.question)
    return {"answer": answer}
