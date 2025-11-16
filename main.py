from fastapi import FastAPI

from routers.rag import router as rag_router
from services.rag_pipeline import preload_rag

app = FastAPI(title="HT AI Backend")

@app.on_event("startup")
async def startup_event():
    await preload_rag()

# register routes
app.include_router(rag_router, prefix="/rag")
