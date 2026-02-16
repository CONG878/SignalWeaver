"""
Purpose:
    - 여러 예측 모델을 하나로 묶어 단일 모델처럼 동작하게 하는 Wrapper 클래스
    - 04단계(Recursive Extension) 파이프라인 수정 없이 앙상블 적용 가능
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from src.models.base import ModelBase


class EnsembleModel(ModelBase):
    """
    학습된 다수의 모델과 최적 가중치를 결합한 앙상블 모델
    """

    def __init__(
        self,
        model_version: str,
        models: List[ModelBase],
        weights: List[float],
    ):
        super().__init__(model_name="ensemble", model_version=model_version)

        self.models = models
        self.weights = weights
        
        # 하위 모델(LGBM 등)의 메타데이터를 상속
        self.target_columns = self.models[0].target_columns
        self.feature_list = self.models[0].feature_list
        
        # 앙상블 모델은 이미 03b 단계에서 조립되므로 항상 fitted 상태임
        self.is_fitted = True

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame], **kwargs) -> None:
        raise NotImplementedError(
            "EnsembleModel은 03b 단계에서 사전 학습된 모델들을 조립하여 생성되므로, "
            "fit()을 직접 호출하지 않습니다."
        )

    def predict(
        self, X: pd.DataFrame, target_name: Optional[str] = None, **kwargs
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        각 하위 모델의 예측값을 구한 뒤 가중 평균(Weighted Average)을 산출하여 반환
        - DataFrame과 Series 반환 모두 완벽하게 지원 (pandas 연산의 브로드캐스팅 활용)
        """
        self.validate_fitted()
        
        blended_pred = None
        
        for model, weight in zip(self.models, self.weights):
            # 개별 모델 예측 (단일 타깃이면 Series, 전체면 DataFrame 반환)
            pred = model.predict(X, target_name=target_name, **kwargs)
            
            if blended_pred is None:
                blended_pred = pred * weight
            else:
                blended_pred += pred * weight
                
        return blended_pred

    def save(self, path: str) -> None:
        """모델 및 가중치 저장"""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "models": self.models,
                    "weights": self.weights,
                    "target_columns": self.target_columns,
                    "feature_list": self.feature_list,
                    "model_version": self.model_version,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "EnsembleModel":
        """모델 로드"""
        with open(path, "rb") as f:
            obj = pickle.load(f)

        inst = cls(
            model_version=obj["model_version"],
            models=obj["models"],
            weights=obj["weights"],
        )
        inst.target_columns = obj["target_columns"]
        inst.feature_list = obj["feature_list"]
        return inst

    def get_meta(self) -> Dict[str, Any]:
        """앙상블 메타데이터 반환"""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "weights": self.weights,
            "target_columns": self.target_columns,
            "sub_models": [m.get_meta() for m in self.models]
        }