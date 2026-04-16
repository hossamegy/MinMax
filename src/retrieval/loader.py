import json
from interfaces.baseLoader import BaseLoader

class JsonLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self, file_path: str = None):
        target_path = file_path if file_path else self.file_path
        with open(target_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data