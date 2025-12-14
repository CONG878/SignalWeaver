"""
Purpose:
    - Walk-forward (rolling window) 방식의 학습/검증 오케스트레이션
    - ModelBase 인터페이스를 따르는 모든 모델(LightGBM, GRU 등)을 동일하게 처리
    - 현재 단계에서는 LightGBM 1차 예측을 주 용도로 설계

Design principles:
    - 데이터 분할 책임은 Trainer가 전담
    - 모델 객체는 단일 fit/predict에만 집중
    - 실험 재현성과 확장을 위해 side-effect 최소화
"""

from __future__ import annotations

from typing import Iterator, Tuple, Dict, Any, List
from datetime import timedelta
import pandas as pd

from src.models.base import ModelBase


# ---------------------------------------------------------------------
# Walk-forward split generator
# ---------------------------------------------------------------------

def walk_forward_split(
    df: pd.DataFrame,
    *,
    date_col: str,
    train_window: int,
    test_window: int,
    step: int,
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Walk-forward 방식으로 (train_df, test_df) 쌍을 생성

    Parameters
    ----------
    df : DataFrame
        시계열 데이터 (date_col 기준 정렬 필요)
    date_col : str
        날짜 컬럼명
    train_window : int
        학습 윈도우 길이 (일 단위)
    test_window : int
        테스트 윈도우 길이 (일 단위)
    step : int
        이동 간격 (일 단위)
    """

    dates = df[date_col].sort_values().unique()

    start_idx = 0
    while True:
        train_start = dates[start_idx]
        train_end_idx = start_idx + train_window
        test_end_idx = train_end_idx + test_window

        if test_end_idx > len(dates):
            break

        train_dates = dates[start_idx:train_end_idx]
        test_dates = dates[train_end_idx:test_end_idx]

        train_df = df[df[date_col].isin(train_dates)]
        test_df = df[df[date_col].isin(test_dates)]

        yield train_df, test_df

        start_idx += step


# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------

class WalkForwardTrainer:
    """
    Walk-forward 학습/평가 관리자
    """

    def __init__(
        self,
        *,
        model: ModelBase,
        feature_cols: List[str],
        target_col: str,
        date_col: str = "date",
    ):
        self.model = model
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.date_col = date_col

    def run(
        self,
        df: pd.DataFrame,
        *,
        train_window: int,
        test_window: int,
        step: int,
        fit_kwargs: Dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Walk-forward 학습 실행

        Returns
        -------
        DataFrame
            각 split별 예측 결과 누적
        """

        fit_kwargs = fit_kwargs or {}
        results = []

        for i, (train_df, test_df) in enumerate(
            walk_forward_split(
                df,
                date_col=self.date_col,
                train_window=train_window,
                test_window=test_window,
                step=step,
            )
        ):
            # 학습
            self.model.fit(
                train_df[self.feature_cols],
                train_df[self.target_col],
                **fit_kwargs,
            )

            # 예측
            scores = self.model.predict(test_df[self.feature_cols])

            fold_result = test_df[[self.date_col, "ticker"]].copy()
            fold_result["score"] = scores.values
            fold_result["fold"] = i

            results.append(fold_result)

        return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------
# Usage example (documentation only)
# ---------------------------------------------------------------------
"""
from src.models.lightgbm_model import LightGBMModel
from src.modeling.trainer import WalkForwardTrainer

model = LightGBMModel(
    model_version="v1",
    params={"objective": "regression", "learning_rate": 0.01},
    feature_list=features,
)

trainer = WalkForwardTrainer(
    model=model,
    feature_cols=features,
    target_col="target_lgbm",
)

pred_df = trainer.run(
    df=dataset,
    train_window=252 * 3,
    test_window=5,
    step=5,
)
"""