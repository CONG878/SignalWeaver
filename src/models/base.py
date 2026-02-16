"""
Purpose:
    모든 예측 모델이 구현해야 하는 공통 인터페이스 정의
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Union
import pandas as pd

class ModelBase(ABC):
    """
    모든 모델의 공통 베이스 클래스
    """

    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame], **kwargs) -> None:
        """
        모델 학습

        Parameters
        ----------
        X : DataFrame
            학습용 feature matrix
        y : Series or DataFrame
            타깃 벡터 (Series) 또는 타깃 매트릭스 (DataFrame - Multi-output용)
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        예측 수행
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "ModelBase":
        raise NotImplementedError

    @abstractmethod
    def get_meta(self) -> Dict[str, Any]:
        raise NotImplementedError

    def validate_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet")