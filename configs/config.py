from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class Config:
    video_directory: Path
    video_files: Optional[List[Path]]
    data_directory: Path
    data_name: Optional[str]

    @classmethod
    def load_config(cls):
        return yaml.safe_load(Path("config.yaml").read_text())


CONFIG = Config.load_config()

if __name__ == "__main__":
    print(CONFIG)
