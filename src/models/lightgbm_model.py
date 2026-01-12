"""
Purpose:
    - LightGBM 기반 1차 예측 모델 구현체 (Enhanced)
    - ModelBase 인터페이스 구현
    - Categorical Feature (ticker) 지원 추가
    - 전체 통합 데이터셋 학습 가능

Design notes:
    - 시계열 split / walk-forward 로직은 trainer에서 담당
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
    LightGBM 회귀 모델 래퍼 (Categorical Feature 지원)
    """

    def __init__(
        self,
        model_version: str,
        params: Dict[str, Any],
        feature_list: List[str],
        categorical_features: Optional[List[str]] = None,
        task: str = "regression",
    ):
        super().__init__(model_name="lightgbm", model_version=model_version)

        self.params = params
        self.feature_list = feature_list
        self.categorical_features = categorical_features or []
        self.task = task
        self.model: Optional[lgb.Booster] = None

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        eval_set: Optional[List[tuple]] = None,
        **kwargs
    ) -> None:
        """
        LightGBM 학습

        Parameters
        ----------
        X : DataFrame
            feature matrix (feature_list 기준)
        y : Series
            target
        eval_set : List[tuple], optional
            검증 세트 [(X_valid, y_valid), ...]
        **kwargs : dict
            lgb.train()에 전달할 추가 인자
            - num_boost_round
            - callbacks
            - categorical_feature 등
        """
        # Categorical feature 인덱스 변환
        cat_indices = []
        if self.categorical_features:
            cat_indices = [
                i for i, col in enumerate(self.feature_list) 
                if col in self.categorical_features
            ]

        # Train Dataset 생성
        train_data = lgb.Dataset(
            X[self.feature_list], 
            label=y,
            categorical_feature=cat_indices if cat_indices else 'auto'
        )

        # Valid Datasets 준비
        valid_sets = [train_data]
        valid_names = ['train']
        
        if eval_set:
            for i, (X_valid, y_valid) in enumerate(eval_set, 1):
                valid_data = lgb.Dataset(
                    X_valid[self.feature_list],
                    label=y_valid,
                    reference=train_data,
                    categorical_feature=cat_indices if cat_indices else 'auto'
                )
                valid_sets.append(valid_data)
                valid_names.append(f'valid{i}')

        # 학습
        self.model = lgb.train(
            params=self.params,
            train_set=train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            **kwargs,
        )

        self.is_fitted = True

    def predict(self, X: pd.DataFrame, **kwargs) -> pd.Series:
        """
        예측 수행
        """
        self.validate_fitted()

        preds = self.model.predict(X[self.feature_list], **kwargs)
        return pd.Series(preds, index=X.index, name="prediction")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        모델 저장 (pickle)
        """
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "params": self.params,
                    "feature_list": self.feature_list,
                    "categorical_features": self.categorical_features,
                    "task": self.task,
                    "model_version": self.model_version,
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
            model_version=obj.get("model_version", "loaded"),
            params=obj["params"],
            feature_list=obj["feature_list"],
            categorical_features=obj.get("categorical_features", []),
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
        meta = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "task": self.task,
            "feature_list": self.feature_list,
            "categorical_features": self.categorical_features,
            "hyperparameters": self.params,
        }
        
        # Feature Importance 추가 (학습 후)
        if self.is_fitted and self.model:
            importance = self.model.feature_importance(importance_type='gain')
            meta["feature_importance"] = dict(zip(self.feature_list, importance))
        
        return meta


# ----------------------------------------------------------------------
# Usage example (documentation only)
# ----------------------------------------------------------------------
"""
from src.models.lightgbm_model import LightGBMModel

# 전체 통합 모델 (ticker를 categorical feature로)
model = LightGBMModel(
    model_version="v1_unified",
    params={
        "objective": "regression",
        "learning_rate": 0.05,
        "num_leaves": 31,
    },
    feature_list=["feature_ma_5", "feature_rsi_14", "ticker"],
    categorical_features=["ticker"]  # 종목 코드
)

# 학습
model.fit(
    X_train, 
    y_train,
    eval_set=[(X_valid, y_valid)],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50)]
)

# 예측
scores = model.predict(X_test)

# 저장
model.save("models/unified_model.pkl")
"""