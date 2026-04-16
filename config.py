from dataclasses import dataclass


@dataclass
class Config:
    model_name: str = "all-MiniLM-L6-v2"
    persist_directory: str = "./chroma_langchain_db"
    collection_name: str = "example_collection"
    hard_thr: float = 0.6
    c: float = 0.9
    init_const: float = 1.5
    normalize: bool = True