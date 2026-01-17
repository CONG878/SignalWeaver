import yaml
from pathlib import Path
from typing import Any, Dict

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """YAML 설정 파일을 로드합니다."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def get_path(base_path: str, reference_date: str) -> Path:
    """기준일이 포함된 경로를 생성합니다."""
    return Path(base_path) / f"dataset_{reference_date}.parquet"