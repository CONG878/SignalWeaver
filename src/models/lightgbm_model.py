"""
Purpose:
    - LightGBM 기반 1차 예측 모델 구현체
    - ModelBase 인터페이스를 구현
    - 후보 종목 선정 단계의 입력(score)을 생성

Design notes:
    - 시계열 split / walk-forward 로직은 trainer 쪽에서 담당
    - 이 클래스는 "한 번의 학습/예측"에만 집중
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List
import pandas as pd
import lightgbm as lgb
import pickle

from src.models.base import ModelBase


class LightGBMModel(ModelBase):
    """
    LightGBM 회귀/분류 모델 래퍼
    """

    def __init__(
        self,
        model_version: str,
        params: Dict[str, Any],
        feature_list: List[str],
        task: str = "regression",
    ):
        super().__init__(model_name="lightgbm", model_version=model_version)

        self.params = params
        self.feature_list = feature_list
        self.task = task
        self.model: Optional[lgb.Booster] = None

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        LightGBM 학습

        Parameters
        ----------
        X : DataFrame
            feature matrix (feature_list 기준)
        y : Series
            target
        """

        train_data = lgb.Dataset(X[self.feature_list], label=y)

        self.model = lgb.train(
            params=self.params,
            train_set=train_data,
            **kwargs,
        )

        self.is_fitted = True

    def predict(self, X: pd.DataFrame, **kwargs) -> pd.Series:
        """
        예측 수행
        """
        self.validate_fitted()

        preds = self.model.predict(X[self.feature_list])
        return pd.Series(preds, index=X.index, name="score_lgbm")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        모델 저장
        """
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "params": self.params,
                    "feature_list": self.feature_list,
                    "task": self.task,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "LightGBMModel":
        """
        모델 로드
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)

        inst = cls(
            model_version="loaded",
            params=obj["params"],
            feature_list=obj["feature_list"],
            task=obj.get("task", "regression"),
        )
        inst.model = obj["model"]
        inst.is_fitted = True

        return inst

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_meta(self) -> Dict[str, Any]:
        """
        모델 메타데이터 반환
        """
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "task": self.task,
            "feature_list": self.feature_list,
            "hyperparameters": self.params,
        }


# ----------------------------------------------------------------------
# Usage example (documentation only)
# ----------------------------------------------------------------------
"""
from src.models.lightgbm_model import LightGBMModel

model = LightGBMModel(
    model_version="v1",
    params={"objective": "regression", "learning_rate": 0.01},
    feature_list=["rsi_14", "ma_20", "volatility_20"],
)

model.fit(X_train, y_train)
scores = model.predict(X_test)
"""