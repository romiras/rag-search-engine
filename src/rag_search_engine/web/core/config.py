import yaml
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class SearchSettings(BaseModel):
    threshold: float = 0.6
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


class StorageSettings(BaseModel):
    db_path: str = "data/local_index.db"


class LoggingSettings(BaseModel):
    log_path: str = "logs/search.log"
    level: str = "INFO"


class IndexingSettings(BaseModel):
    batch_size: int = 32
    include_paths: List[str] = Field(default_factory=list)


class AppSettings(BaseModel):
    search: SearchSettings = Field(default_factory=SearchSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    indexing: IndexingSettings = Field(default_factory=IndexingSettings)
    debug: bool = False

    @classmethod
    def load(cls, path: str = "config.yaml") -> "AppSettings":
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)


_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    global _settings
    if _settings is None:
        _settings = AppSettings.load()
    return _settings
