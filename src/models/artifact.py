"""
Purpose:
    - 모델 파일(아티팩트) 저장 규칙을 통일
    - 실험/운영에서 사용된 모델 메타데이터를 registry로 관리
    - LightGBM → 후보선정 → GRU 파이프라인 연결을 안정화

Design scope (현재 단계):
    - 파일 기반(JSON) registry
    - MLflow 등 외부 시스템 도입 전의 경량 구현
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


# ---------------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------------

DEFAULT_MODEL_DIR = Path("data/04_models")
REGISTRY_FILE = DEFAULT_MODEL_DIR / "registry.json"


# ---------------------------------------------------------------------
# 내부 유틸
# ---------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _hash_dict(d: Dict[str, Any]) -> str:
    """파라미터 dict → 짧은 해시값"""
    dumped = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(dumped.encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def save_model_artifact(
    *,
    model_name: str,
    model_version: str,
    model_object: Any,
    metadata: Dict[str, Any],
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> Path:
    """
    모델 아티팩트 저장 + registry 업데이트

    Naming rule
    -----------
    {model_name}/{YYYYMMDD}_{model_version}_{param_hash}.pkl
    """

    _ensure_dir(model_dir / model_name)

    param_hash = _hash_dict(metadata.get("hyperparameters", {}))
    date_str = datetime.now().strftime("%Y%m%d")

    artifact_path = (
        model_dir
        / model_name
        / f"{date_str}_{model_version}_{param_hash}.pkl"
    )

    # 실제 모델 저장은 model_object가 담당
    model_object.save(str(artifact_path))

    # registry entry
    entry = {
        "model_name": model_name,
        "model_version": model_version,
        "artifact_path": str(artifact_path),
        "created_at": datetime.now().isoformat(),
        "metadata": metadata,
    }

    _update_registry(entry, model_dir)

    return artifact_path


def load_registry(model_dir: Path = DEFAULT_MODEL_DIR) -> Dict[str, Any]:
    """
    registry 전체 로드
    """
    if not REGISTRY_FILE.exists():
        return {"models": []}

    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _update_registry(entry: Dict[str, Any], model_dir: Path) -> None:
    _ensure_dir(model_dir)

    registry = load_registry(model_dir)
    registry.setdefault("models", []).append(entry)

    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------
# Query helpers (optional but useful)
# ---------------------------------------------------------------------

def find_models(
    *,
    model_name: str,
    model_version: str | None = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> list[Dict[str, Any]]:
    """
    registry에서 조건에 맞는 모델 검색
    """
    registry = load_registry(model_dir)

    results = []
    for m in registry.get("models", []):
        if m["model_name"] != model_name:
            continue
        if model_version and m["model_version"] != model_version:
            continue
        results.append(m)

    return results


# ---------------------------------------------------------------------
# Usage example (documentation only)
# ---------------------------------------------------------------------
"""
from src.models.artifact import save_model_artifact

artifact_path = save_model_artifact(
    model_name="lightgbm",
    model_version="v1",
    model_object=lgbm_model,
    metadata={
        "feature_list": features,
        "training_period": "2019-01-01~2024-12-31",
        "hyperparameters": params,
        "data_version": "2025-01-10",
    },
)
"""