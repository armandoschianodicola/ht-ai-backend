from fastapi import APIRouter

from models.requests.query import QueryRequest
from services.rag_pipeline import run_rag

router = APIRouter()


@router.post("/query")
async def query_rag(request: QueryRequest):
    answer = await run_rag(request.question)
    return {"answer": answer}
