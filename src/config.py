from dataclasses import dataclass
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

@dataclass
class Config:
    data_path: str = "assets\dataset.json"
    model_name: str = "all-MiniLM-L6-v2"
    persist_directory: str = "./chroma_db"
    collection_name: str = "example_collection"
    hard_thr: float = 0.6
    c: float = 0.9
    init_const: float = 1.5
    normalize: bool = True

    # DI Selection
    # Options: "local", "gemini"
    embedding_type: str = "local"

    # Gemini Config
    gemini_api_key: str = os.getenv("GOOGLE_API_KEY")
    gemini_model_name: str = "gemini-embedding-2-preview"