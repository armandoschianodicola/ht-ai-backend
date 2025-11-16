import os

from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vectorstore")
    DOCS_PATH = os.getenv("DOCS_PATH", "./data/docs")


settings = Settings()
