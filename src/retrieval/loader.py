import json
from src.interfaces.baseLoader import BaseLoader


class JsonLoader(BaseLoader):
    def load(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data