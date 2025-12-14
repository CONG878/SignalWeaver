"""
Purpose:
    모든 예측 모델(LightGBM, GRU, 이후 확장 모델)이 반드시 구현해야 하는
    최소 공통 인터페이스를 정의한다.

Design principles:
    - 학습/추론/저장/로드의 호출 규약을 통일
    - 파이프라인 레벨에서는 모델 종류를 의식하지 않음
    - 실험 재현성과 아티팩트 관리에 필요한 메타데이터 제공
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd


class ModelBase(ABC):
    """
    모든 모델의 공통 베이스 클래스

    LightGBM, GRU, Transformer 등 모델 종류와 무관하게
    동일한 호출 패턴을 강제한다.
    """

    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Core lifecycle methods
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        모델 학습

        Parameters
        ----------
        X : DataFrame
            학습용 feature matrix
        y : Series
            타깃 벡터
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs) -> pd.Series:
        """
        예측 수행

        Returns
        -------
        Series
            예측 결과 (score, return 등)
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @abstractmethod
    def save(self, path: str) -> None:
        """
        모델 아티팩트 저장
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "ModelBase":
        """
        저장된 모델 로드
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Metadata / Introspection
    # ------------------------------------------------------------------

    @abstractmethod
    def get_meta(self) -> Dict[str, Any]:
        """
        모델 메타데이터 반환

        포함 권장 항목
        ----------------
        - model_name
        - model_version
        - feature_list
        - training_period
        - hyperparameters
        - data_version
        """
        raise NotImplementedError

    def validate_fitted(self) -> None:
        """
        학습 여부 검증
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet")


# ----------------------------------------------------------------------
# Example usage (documentation only)
# ----------------------------------------------------------------------
"""
from src.models.base import ModelBase

class LightGBMModel(ModelBase):
    def fit(self, X, y, **kwargs):
        ...
        self.is_fitted = True

    def predict(self, X, **kwargs):
        self.validate_fitted()
        ...

    def save(self, path: str):
        ...

    @classmethod
    def load(cls, path: str):
        ...

    def get_meta(self):
        return {...}
"""