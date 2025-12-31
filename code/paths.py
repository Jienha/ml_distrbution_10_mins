import json
import os
from pathlib import Path

class PathRegistry():
    def paths():
        # project root (works both locally and in Docker)
        BASE_DIR = Path(__file__).resolve().parent.parent

        CONFIG_PATH = os.path.join(BASE_DIR, "path_registry.json")

        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)

        PATHS = config["paths"]
        return PATHS