from contextlib import asynccontextmanager

from fastapi import FastAPI

from routers.rag import router as rag_router
from services.rag_pipeline import preload_rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    await preload_rag()
    yield

app = FastAPI(title="HT AI Backend", lifespan=lifespan)

# register routes
app.include_router(rag_router, prefix="/rag")
