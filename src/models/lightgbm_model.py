"""
Purpose:
    - LightGBM 기반 Multi-output 예측 모델 구현체 (Refactored)
    - 단일 시점(Scalar) 및 다중 시점(Vector) 예측을 모두 지원
    - ModelBase 인터페이스 준수
    - 내부적으로 Target 컬럼별 개별 Booster를 관리하는 'Chained Regression' 방식 적용

Design notes:
    - fit(): y가 DataFrame(다중 컬럼)일 경우, 컬럼별로 루프를 돌며 개별 모델을 학습
    - predict(): 학습된 모든 모델의 예측값을 모아 DataFrame으로 반환
    - save/load: 여러 개의 Booster 객체를 리스트/딕셔너리 형태로 직렬화
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List, Union
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from tqdm import tqdm

from src.models.base import ModelBase


class LightGBMModel(ModelBase):
    """
    LightGBM Multi-output 회귀 모델 래퍼
    
    특징:
    - y로 Series(단일)가 들어오면 기존처럼 동작
    - y로 DataFrame(다중)이 들어오면 컬럼 수만큼 모델을 생성 (Multi-output)
    """

    def __init__(
        self,
        model_version: str,
        params: Dict[str, Any],
        feature_list: List[str],
        categorical_features: Optional[List[str]] = None,
        task: str = "regression",
    ):
        super().__init__(model_name="lightgbm_multi", model_version=model_version)

        self.params = params
        self.feature_list = feature_list
        self.categorical_features = categorical_features or []
        self.task = task
        
        # [변경] 단일 모델 대신 여러 모델을 관리하기 위한 컨테이너
        # Key: Target 컬럼명, Value: 학습된 lgb.Booster 객체
        self.models: Dict[str, lgb.Booster] = {}
        self.target_columns: List[str] = []

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    def fit(
        self, 
        X: pd.DataFrame, 
        y: Union[pd.Series, pd.DataFrame], 
        eval_set: Optional[List[tuple]] = None,
        verbose: bool = False,
        **kwargs
    ) -> None:
        """
        LightGBM 학습 (Multi-output 지원)

        Parameters
        ----------
        X : DataFrame
            feature matrix
        y : Series or DataFrame
            target vector(s). 
            DataFrame일 경우 컬럼 개수만큼 모델이 학습됨 (예: t+1 ~ t+5)
        eval_set : List[tuple], optional
            [(X_valid, y_valid), ...]
            y_valid 역시 y와 동일한 형태여야 함
        """
        # 1. 특정 식별자(ID) 추출
        target_id = kwargs.pop('target_name', None)

        if isinstance(y, pd.Series):
            y = y.to_frame()
        
        # 2. 루프 대상 결정
        # target_id가 있으면 그 이름으로 저장하되, 
        # 실제 데이터는 y의 첫 번째 컬럼(target_log_close)을 사용함
        loop_targets = [target_id] if target_id else y.columns

        for col in loop_targets:
            # [수정] 데이터 추출 로직: 
            # col 이름이 y에 있으면 쓰고, 없으면 y의 첫 번째 컬럼을 가져옴
            y_col = y[col] if col in y.columns else y.iloc[:, 0]
            
            # 메타데이터 업데이트
            if col not in self.target_columns:
                self.target_columns.append(col)

            train_ds = lgb.Dataset(X[self.feature_list], label=y_col)
            
            valid_sets = [train_ds]; valid_names = ['train']
            if eval_set:
                for i, (X_val, y_val) in enumerate(eval_set, 1):
                    # [수정] 검증 데이터도 동일하게 첫 번째 컬럼 참조
                    if isinstance(y_val, pd.DataFrame):
                        y_val_c = y_val[col] if col in y_val.columns else y_val.iloc[:, 0]
                    else:
                        y_val_c = y_val
                        
                    valid_ds = lgb.Dataset(X_val[self.feature_list], label=y_val_c, reference=train_ds)
                    valid_sets.append(valid_ds); valid_names.append(f'valid{i}')

            # 3. 학습 및 저장
            booster = lgb.train(
                params=self.params,
                train_set=train_ds,
                valid_sets=valid_sets,
                valid_names=valid_names,
                **kwargs,
            )
            self.models[col] = booster

        self.is_fitted = True

    def predict(self, X: pd.DataFrame, target_name: Optional[str] = None, **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """
        target_name을 지정하면 해당 Horizon만 예측하여 Series 반환 (효율성)
        지정하지 않으면 모든 Horizon을 예측하여 DataFrame 반환
        """
        self.validate_fitted()
        
        # 1. 단일 타깃 예측 (Trainer의 루프 최적화용)
        if target_name:
            if target_name not in self.models:
                raise ValueError(f"Model for target '{target_name}' not found.")
            
            pred = self.models[target_name].predict(X[self.feature_list], **kwargs)
            return pd.Series(pred, index=X.index, name=target_name)

        # 2. 전체 타깃 예측
        results = {}
        for col, booster in self.models.items():
            results[col] = booster.predict(X[self.feature_list], **kwargs)
            
        return pd.DataFrame(results, index=X.index)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        모델 저장 (pickle)
        - 여러 개의 booster를 딕셔너리 형태로 저장
        """
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "models": self.models,  # [변경] 단일 model -> models dict
                    "target_columns": self.target_columns,
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
        
        # [변경] models 복원
        inst.models = obj["models"]
        inst.target_columns = obj.get("target_columns", [])
        
        # 구버전 호환성 (단일 모델 파일인 경우)
        if "model" in obj and not inst.models:
            inst.models = {"single_output": obj["model"]}
            inst.target_columns = ["single_output"]

        inst.is_fitted = True
        return inst

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_meta(self) -> Dict[str, Any]:
        """
        모델 메타데이터 반환
        - Feature Importance는 첫 번째 모델(가장 가까운 시점)을 기준으로 하거나, 평균을 낼 수 있음
        - 여기서는 '평균 Importance'를 제공
        """
        meta = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "task": self.task,
            "feature_list": self.feature_list,
            "categorical_features": self.categorical_features,
            "hyperparameters": self.params,
            "target_columns": self.target_columns,
            "num_outputs": len(self.models)
        }
        
        # Feature Importance 집계 (모든 Horizon 모델의 평균)
        if self.is_fitted and self.models:
            importances = []
            for booster in self.models.values():
                imp = booster.feature_importance(importance_type='gain')
                importances.append(imp)
            
            # 평균 계산
            avg_imp = np.mean(importances, axis=0)
            meta["feature_importance"] = dict(zip(self.feature_list, avg_imp))
        
        return meta