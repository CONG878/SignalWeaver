"""
Purpose:
    - 모델 파일(아티팩트) 저장 규칙을 통일
    - 실험/운영에서 사용된 모델 메타데이터를 registry로 관리
    - LightGBM → 후보선정 → GRU 파이프라인 연결을 안정화

Design scope (현재 단계):
    - 파일 기반(JSON) registry
    - MLflow 등 외부 시스템 도입 전의 경량 구현

✨ H1+H2 패치 (2026-02-08):
✨ v3.9.2 패치 (2026-03-16): param_hash 문서화
- save_model_artifact() Notes 섹션: hyperparameters 키 누락 시 hash 충돌 경고 추가
    - ProjectPaths 클래스 사용
    - 중복 경로 문제 해결: model_dir/{model_name}/ 구조에서
      model_dir이 이미 {model_name}을 포함한 경우 중복 방지
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

DEFAULT_MODEL_DIR = Path("data/03_training")  # ✨ H1 패치: 경로 변경
REGISTRY_FILE_NAME = "registry.json"


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
    model_dir: Path = None,
) -> Path:
    """
    모델 아티팩트 저장 + registry 업데이트

    Naming rule
    -----------
    {model_dir}/{YYYYMMDD}_{model_version}_{param_hash}.pkl
    
    ✨ H2 패치: 중복 경로 방지
    - model_dir이 이미 model_name을 포함하면 중복 생성 안 함
    
    Parameters
    ----------
    model_name : str
        모델명 (예: "lightgbm_multi")
    model_version : str
        버전 (예: "v1_20260206")
    model_object : Any
        저장할 모델 객체 (.save() 메서드 필요)
    metadata : dict
        메타데이터
    model_dir : Path, optional
        저장 디렉토리 (None이면 DEFAULT_MODEL_DIR 사용)
    
    Returns
    -------
    Path
        저장된 파일 경로

    Notes
    -----
    **param_hash 생성 규칙**
    파일명의 `{param_hash}`는 ``metadata.get("hyperparameters", {})``를
    MD5 해싱하여 생성합니다 (앞 8자리).

    .. warning::
        호출 시 ``metadata``에 ``hyperparameters`` 키가 없으면
        빈 dict ``{}``가 해싱되어 **모든 모델의 hash가 동일**해집니다.
        이 경우 ``registry.json``에서 모델을 구분할 수 없게 됩니다.
        반드시 아래와 같이 명시적으로 전달하십시오.

        >>> save_model_artifact(
        ...     metadata={
        ...         "test_metrics": ...,
        ...         "target_columns": ...,
        ...         "hyperparameters": train_cfg.get("lgbm_params", {}),  # 필수
        ...     }
        ... )
    """
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR / model_name
    
    # ✨ 모델 분리: model_dir이 이미
    # 예: model_dir = "data/03_training/20260206/{model_name}"
    # 이므로 추가 검사 없이 그대로 사용합니다.
    save_dir = model_dir
    _ensure_dir(save_dir)

    param_hash = _hash_dict(metadata.get("hyperparameters", {}))
    date_str = datetime.now().strftime("%Y%m%d")

    artifact_path = save_dir / f"{date_str}_{model_version}_{param_hash}.pkl"

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

    _update_registry(entry, save_dir)

    return artifact_path


def load_registry(model_dir: Path = None) -> Dict[str, Any]:
    """
    registry 전체 로드
    """
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    
    registry_file = model_dir / REGISTRY_FILE_NAME
    
    if not registry_file.exists():
        return {"models": []}

    with open(registry_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _update_registry(entry: Dict[str, Any], model_dir: Path) -> None:
    """registry 업데이트 (내부 함수)"""
    _ensure_dir(model_dir)

    registry_file = model_dir / REGISTRY_FILE_NAME
    
    # 기존 registry 로드
    if registry_file.exists():
        with open(registry_file, "r", encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = {"models": []}
    
    registry.setdefault("models", []).append(entry)

    with open(registry_file, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------
# Query helpers (optional but useful)
# ---------------------------------------------------------------------

def find_models(
    *,
    model_name: str,
    model_version: str | None = None,
    model_dir: Path = None,
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
✨ H2 패치 적용 예시:

from src.utils.config import load_config, ProjectPaths
from src.models.artifact import save_model_artifact

cfg = load_config()
paths = ProjectPaths.from_config(cfg)

# ProjectPaths를 사용한 저장
artifact_path = save_model_artifact(
    model_name="lightgbm_multi",
    model_version="v1",
    model_object=lgbm_model,
    metadata={
        "feature_list": features,
        "training_period": "2019-01-01~2024-12-31",
        "hyperparameters": params,
        "data_version": "2025-01-10",
    },
    model_dir=paths.get_model_dir("lightgbm_multi")  # ← 이미 model_name 포함
)

결과:
data/03_training/20260206/lightgbm_multi/20260206_v1_abc123.pkl
                                          ^^^^^^^^^^^^^^^^^^^^^^
                                          중복 없음!
"""
