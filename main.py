from fastapi import FastAPI

from routers.rag import router as rag_router

app = FastAPI(title="HT AI Backend")


# register routes
app.include_router(rag_router, prefix="/rag")
