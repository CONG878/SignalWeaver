"""
Purpose:
    - Scikit-learn 기반 RandomForest Multi-output 예측 모델 구현체
    - MultiOutputRegressor를 사용하여 다중 타깃 동시 학습 수행
    - ModelBase 인터페이스 준수
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from src.models.base import ModelBase


class RandomForestMultiModel(ModelBase):
    """
    RandomForest + MultiOutputRegressor 모델 래퍼
    
    특징:
    - fit() 호출 시 DataFrame 형태의 y를 받아 한 번에 학습합니다.
    """

    def __init__(
        self,
        model_version: str,
        params: Dict[str, Any],
        feature_list: List[str],
        task: str = "regression",
    ):
        super().__init__(model_name="randomforest_multi", model_version=model_version)

        self.params = params
        self.feature_list = feature_list
        self.task = task
        self.target_columns: List[str] = []
        
        # 기본 Estimator 설정 (RandomForest)
        # n_jobs=-1로 병렬 처리를 활성화하여 속도 향상 권장
        rf_params = params.copy()
        if "n_jobs" not in rf_params:
            rf_params["n_jobs"] = -1
            
        base_estimator = RandomForestRegressor(**rf_params)
        self.model = MultiOutputRegressor(base_estimator)

    def fit(
        self, 
        X: pd.DataFrame, 
        y: Union[pd.Series, pd.DataFrame], 
        **kwargs
    ) -> None:
        """
        모델 학습
        """
        if isinstance(y, pd.Series):
            y = y.to_frame()
            
        self.target_columns = y.columns.tolist()

        # MultiOutputRegressor는 DataFrame y를 지원함
        self.model.fit(X[self.feature_list], y)
        self.is_fitted = True

    def predict(
        self, X: pd.DataFrame, target_name: Optional[str] = None, **kwargs
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        예측 수행
        """
        self.validate_fitted()
        
        # 전체 예측 (numpy array 반환됨)
        preds = self.model.predict(X[self.feature_list])
        
        # DataFrame으로 변환
        pred_df = pd.DataFrame(preds, index=X.index, columns=self.target_columns)
        
        # 특정 타깃만 요청된 경우 (Trainer 호환성)
        if target_name:
            if target_name not in self.target_columns:
                raise ValueError(f"Target '{target_name}' not found in trained columns.")
            return pred_df[target_name]

        return pred_df

    def save(self, path: str) -> None:
        """모델 저장 (pickle)"""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "target_columns": self.target_columns,
                    "params": self.params,
                    "feature_list": self.feature_list,
                    "task": self.task,
                    "model_version": self.model_version,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "RandomForestMultiModel":
        """모델 로드"""
        with open(path, "rb") as f:
            obj = pickle.load(f)

        inst = cls(
            model_version=obj.get("model_version", "loaded"),
            params=obj["params"],
            feature_list=obj["feature_list"],
            task=obj.get("task", "regression"),
        )
        
        inst.model = obj["model"]
        inst.target_columns = obj.get("target_columns", [])
        inst.is_fitted = True
        return inst

    def get_meta(self) -> Dict[str, Any]:
        """메타데이터 반환"""
        meta = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "task": self.task,
            "feature_list": self.feature_list,
            "hyperparameters": self.params,
            "target_columns": self.target_columns,
            "num_outputs": len(self.target_columns)
        }
        
        # Feature Importance 평균 계산
        if self.is_fitted:
            try:
                importances = []
                for estimator in self.model.estimators_:
                    importances.append(estimator.feature_importances_)
                
                avg_imp = np.mean(importances, axis=0)
                meta["feature_importance"] = dict(zip(self.feature_list, avg_imp))
            except AttributeError:
                # 일부 estimator는 feature_importances_가 없을 수 있음
                meta["feature_importance"] = {}
        
        return meta