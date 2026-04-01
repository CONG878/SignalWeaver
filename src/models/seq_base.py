"""
SeqModelBase — 시계열 전용 모델 베이스 클래스

## 설계 원칙
- ModelBase를 상속하여 전체 모델 계층에 편입
- fit() / predict() 시그니처를 3D 시퀀스 입력으로 재정의
- save() / load()는 디렉토리 단위 (weights.pt + config.json)
- EnsembleModel과의 직접 혼합은 v4.1.0에서 처리 예정

## Tabular ModelBase와의 차이
    ModelBase.fit(X: pd.DataFrame, y: ...)
    SeqModelBase.fit(X_seq: np.ndarray, y_seq: np.ndarray, ...)
    →  입력이 (N, seq_len, n_features) 3D NumPy 배열

## v4.0.0
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.models.base import ModelBase


class SeqModelBase(ModelBase):
    """
    시계열 전용 모델 베이스 클래스.

    Parameters
    ----------
    model_name : str
    model_version : str
    seq_len : int
        입력 시퀀스 길이 (거래일). config.sequence.seq_len 값을 그대로 사용.
    forecast_horizon : int
        예측 대상 기간. config.sequence.forecast_horizon 값.
    feature_list : List[str]
        학습/예측에 사용할 피처 컬럼명 목록.
    target_type : str
        "log_return_1d" (권장) 또는 "log_close".
    """

    def __init__(
        self,
        model_name: str,
        model_version: str,
        seq_len: int,
        forecast_horizon: int,
        feature_list: List[str],
        target_type: str = "log_return_1d",
    ):
        super().__init__(model_name=model_name, model_version=model_version)
        self.seq_len          = seq_len
        self.forecast_horizon = forecast_horizon
        self.feature_list     = feature_list
        self.target_type      = target_type

        # 타겟 컬럼명 — Tabular 트랙과 동일한 규격으로 자동 생성
        # 예: ["target_log_return_1d_h1", ..., "target_log_return_1d_h20"]
        base = "target_log_return_1d" if target_type == "log_return_1d" else "target_log_close"
        self.target_columns: List[str] = [
            f"{base}_h{h}" for h in range(1, forecast_horizon + 1)
        ]

    # ------------------------------------------------------------------
    # 추상 메서드 재정의 — 시그니처 명시
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X_seq: np.ndarray,
        y_seq: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs,
    ) -> None:
        """
        시퀀스 데이터로 모델 학습.

        Parameters
        ----------
        X_seq : np.ndarray, shape (N, seq_len, n_features)
            입력 시퀀스 텐서.
        y_seq : np.ndarray, shape (N, forecast_horizon)
            타겟 행렬. 각 열은 t+1, t+2, ..., t+forecast_horizon 시점의 타겟값.
        eval_set : (X_val, y_val), optional
            Early stopping용 검증 데이터.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        X_seq: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        예측 수행.

        Parameters
        ----------
        X_seq : np.ndarray, shape (N, seq_len, n_features)

        Returns
        -------
        np.ndarray, shape (N, forecast_horizon)
            각 열이 t+1, ..., t+forecast_horizon 시점의 예측값.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, dir_path: str) -> None:
        """
        디렉토리에 저장.

        저장 파일 구성:
          {dir_path}/weights.pt    — PyTorch state_dict
          {dir_path}/config.json   — 아키텍처 파라미터 및 메타데이터
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, dir_path: str) -> "SeqModelBase":
        """디렉토리에서 모델 복원."""
        raise NotImplementedError

    @abstractmethod
    def get_meta(self) -> Dict[str, Any]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # ModelBase의 Tabular 전용 predict 시그니처 충돌 방지
    # ------------------------------------------------------------------

    def _predict_tabular(self, X, **kwargs):
        """
        ModelBase.predict(X: pd.DataFrame) 시그니처 호환용.
        SeqModelBase는 DataFrame 입력을 지원하지 않습니다.
        이 메서드를 직접 호출하지 마세요.
        """
        raise TypeError(
            f"{self.__class__.__name__}은 DataFrame 입력을 지원하지 않습니다. "
            f"predict(X_seq: np.ndarray) 를 사용하세요."
        )

    # ------------------------------------------------------------------
    # 공통 유틸
    # ------------------------------------------------------------------

    def validate_input_shape(self, X_seq: np.ndarray) -> None:
        """입력 텐서 shape 검증."""
        if X_seq.ndim != 3:
            raise ValueError(
                f"X_seq는 3D 텐서여야 합니다. "
                f"현재 shape: {X_seq.shape}  (N, seq_len, n_features) 형태로 전달하세요."
            )
        if X_seq.shape[1] != self.seq_len:
            raise ValueError(
                f"seq_len 불일치: 모델={self.seq_len}, 입력={X_seq.shape[1]}"
            )
        if X_seq.shape[2] != len(self.feature_list):
            raise ValueError(
                f"n_features 불일치: 모델={len(self.feature_list)}, 입력={X_seq.shape[2]}"
            )
