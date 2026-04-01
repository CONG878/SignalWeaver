"""
Purpose:
    - 모델 파일(아티팩트) 저장 규칙을 통일
    - 실험/운영에서 사용된 모델 메타데이터를 registry로 관리

Design scope (현재 단계):
    - 파일 기반(JSON) registry
    - MLflow 등 외부 시스템 도입 전의 경량 구현

## 버전
- v4.0.0: 모델 파일명에서 'v1' 제거. registry.json이 버전 관리를 담당하므로 중복.
          파일명: {YYYYMMDD}_{param_hash}.pkl  (구: {YYYYMMDD}_v1_{param_hash}.pkl)
- v3.9.2: param_hash 문서화 (hyperparameters 키 누락 시 hash 충돌 경고)
- H2 패치: ProjectPaths 클래스 사용
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


DEFAULT_MODEL_DIR = Path("data/03_training")
REGISTRY_FILE_NAME = "registry.json"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _hash_dict(d: Dict[str, Any]) -> str:
    """파라미터 dict → 짧은 해시값 (앞 8자리)."""
    dumped = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(dumped.encode("utf-8")).hexdigest()[:8]


def save_model_artifact(
    *,
    model_name: str,
    model_version: str,
    model_object: Any,
    metadata: Dict[str, Any],
    model_dir: Path = None,
) -> Path:
    """
    모델 아티팩트 저장 + registry 업데이트.

    Naming rule (v4.0.0)
    --------------------
    {model_dir}/{YYYYMMDD}_{param_hash}.pkl

    v4.0.0 변경: 'v1' 접미사 제거.
    registry.json이 버전 관리를 담당하므로 파일명의 버전 표기는 불필요.

    Parameters
    ----------
    model_name : str
    model_version : str
    model_object : Any
        저장할 모델 객체 (.save() 메서드 필요)
    metadata : dict
        메타데이터.

    Notes
    -----
    **param_hash 생성 규칙**
    ``metadata.get("hyperparameters", {})``를 MD5 해싱 (앞 8자리).

    .. warning::
        ``metadata``에 ``hyperparameters`` 키가 없으면 빈 dict ``{}``가 해싱되어
        **모든 모델의 hash가 동일**해집니다. 반드시 명시적으로 전달하세요.

        >>> save_model_artifact(
        ...     metadata={
        ...         "test_metrics": ...,
        ...         "hyperparameters": train_cfg.get("lgbm_params", {}),  # 필수
        ...     }
        ... )
    """
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR / model_name

    save_dir = model_dir
    _ensure_dir(save_dir)

    param_hash = _hash_dict(metadata.get("hyperparameters", {}))
    date_str   = datetime.now().strftime("%Y%m%d")

    # v4.0.0: v1 제거 → {YYYYMMDD}_{param_hash}.pkl
    artifact_path = save_dir / f"{date_str}_{param_hash}.pkl"

    model_object.save(str(artifact_path))

    entry = {
        "model_name":    model_name,
        "model_version": model_version,
        "artifact_path": str(artifact_path),
        "created_at":    datetime.now().isoformat(),
        "metadata":      metadata,
    }

    _update_registry(entry, save_dir)

    return artifact_path


def load_registry(model_dir: Path = None) -> Dict[str, Any]:
    """registry 전체 로드."""
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    registry_file = model_dir / REGISTRY_FILE_NAME

    if not registry_file.exists():
        return {"models": []}

    with open(registry_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _update_registry(entry: Dict[str, Any], model_dir: Path) -> None:
    _ensure_dir(model_dir)

    registry_file = model_dir / REGISTRY_FILE_NAME

    if registry_file.exists():
        with open(registry_file, "r", encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = {"models": []}

    registry.setdefault("models", []).append(entry)

    with open(registry_file, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def find_models(
    *,
    model_name: str,
    model_version: str | None = None,
    model_dir: Path = None,
) -> list[Dict[str, Any]]:
    """registry에서 조건에 맞는 모델 검색."""
    registry = load_registry(model_dir)

    results = []
    for m in registry.get("models", []):
        if m["model_name"] != model_name:
            continue
        if model_version and m["model_version"] != model_version:
            continue
        results.append(m)

    return results
