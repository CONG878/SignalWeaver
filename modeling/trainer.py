"""
Purpose:
    - Walk-forward (rolling window) 방식의 학습/검증 오케스트레이션
    - 날짜 기준 분할 지원 (전체 통합 데이터셋용)
    - ModelBase 인터페이스를 따르는 모든 모델 처리

Design principles:
    - 데이터 분할 책임은 Trainer가 전담
    - 모델 객체는 단일 fit/predict에만 집중
    - 실험 재현성과 확장을 위해 side-effect 최소화
"""

from __future__ import annotations

from typing import Iterator, Tuple, Dict, Any, List, Optional
from datetime import timedelta
import pandas as pd
import numpy as np

from src.models.base import ModelBase


# ---------------------------------------------------------------------
# Walk-forward split generator (Date-based)
# ---------------------------------------------------------------------

def walk_forward_split_by_date(
    df: pd.DataFrame,
    *,
    date_col: str,
    train_end: str,
    valid_window_days: int,
    test_window_days: int,
    num_valid: int = 1,
) -> Tuple[pd.DataFrame, List[pd.DataFrame], pd.DataFrame]:
    """
    날짜 기준 Walk-forward 분할 (전체 통합 데이터셋용)
    
    Parameters
    ----------
    df : DataFrame
        전체 데이터셋 (date_col 기준 정렬 필요)
    date_col : str
        날짜 컬럼명
    train_end : str
        학습 종료일 (예: '2024-01-01')
    valid_window_days : int
        각 검증 구간 길이 (거래일 기준)
    test_window_days : int
        테스트 구간 길이
    num_valid : int
        검증 구간 개수
        
    Returns
    -------
    train_df : DataFrame
        학습 데이터
    valid_dfs : List[DataFrame]
        검증 데이터 리스트
    test_df : DataFrame
        테스트 데이터
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # 날짜 변환
    train_end_date = pd.to_datetime(train_end)
    
    # 학습 데이터
    train_df = df[df[date_col] < train_end_date].copy()
    
    # 검증 구간 시작일
    valid_start = train_end_date
    
    # 검증 데이터 생성
    valid_dfs = []
    for i in range(num_valid):
        # i번째 검증 구간의 날짜 범위
        valid_dates = pd.date_range(
            start=valid_start,
            periods=valid_window_days,
            freq='B'  # 영업일 기준
        )
        
        valid_df = df[df[date_col].isin(valid_dates)].copy()
        
        if not valid_df.empty:
            valid_dfs.append(valid_df)
        
        # 다음 검증 구간 시작일
        valid_start = valid_dates[-1] + timedelta(days=1)
    
    # 테스트 데이터
    test_start = valid_start
    test_dates = pd.date_range(
        start=test_start,
        periods=test_window_days,
        freq='B'
    )
    test_df = df[df[date_col].isin(test_dates)].copy()
    
    return train_df, valid_dfs, test_df


# ---------------------------------------------------------------------
# Trainer (Enhanced)
# ---------------------------------------------------------------------

class WalkForwardTrainer:
    """
    Walk-forward 학습/평가 관리자 (날짜 기준 분할 지원)
    """

    def __init__(
        self,
        *,
        model: ModelBase,
        feature_cols: List[str],
        target_col: str,
        date_col: str = "date",
        categorical_features: Optional[List[str]] = None,
    ):
        self.model = model
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.date_col = date_col
        self.categorical_features = categorical_features or []

    def run(
        self,
        df: pd.DataFrame,
        *,
        train_end: str,
        valid_window_days: int,
        test_window_days: int,
        num_valid: int = 1,
        fit_kwargs: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Walk-forward 학습 실행 (날짜 기준)

        Parameters
        ----------
        df : DataFrame
            전체 데이터셋
        train_end : str
            학습 종료일 (예: '2024-01-01')
        valid_window_days : int
            검증 구간 길이 (거래일)
        test_window_days : int
            테스트 구간 길이
        num_valid : int
            검증 구간 개수
        fit_kwargs : dict, optional
            model.fit()에 전달할 추가 인자

        Returns
        -------
        dict
            {
                'train_metrics': {...},
                'valid_metrics': [{...}, ...],
                'test_metrics': {...},
                'test_predictions': DataFrame
            }
        """
        fit_kwargs = fit_kwargs or {}

        # 1. 데이터 분할
        train_df, valid_dfs, test_df = walk_forward_split_by_date(
            df,
            date_col=self.date_col,
            train_end=train_end,
            valid_window_days=valid_window_days,
            test_window_days=test_window_days,
            num_valid=num_valid,
        )

        print(f"📊 Data Split:")
        print(f"   Train: {len(train_df):,} rows ({train_df[self.date_col].min()} ~ {train_df[self.date_col].max()})")
        print(f"   Valid: {num_valid} folds x {valid_window_days} days")
        print(f"   Test:  {len(test_df):,} rows ({test_df[self.date_col].min()} ~ {test_df[self.date_col].max()})")

        # 2. 학습
        print("\n🔨 Training model...")
        
        # 검증 세트 준비
        eval_set = [(valid_df[self.feature_cols], valid_df[self.target_col]) 
                    for valid_df in valid_dfs]
        
        self.model.fit(
            train_df[self.feature_cols],
            train_df[self.target_col],
            eval_set=eval_set,
            **fit_kwargs,
        )

        # 3. 평가
        results = {
            'train_metrics': self._evaluate(train_df),
            'valid_metrics': [self._evaluate(vdf) for vdf in valid_dfs],
            'test_metrics': self._evaluate(test_df),
            'test_predictions': self._predict_with_metadata(test_df),
        }

        return results

    def _evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        모델 평가 지표 계산
        """
        y_true = df[self.target_col].values
        y_pred = self.model.predict(df[self.feature_cols]).values

        # 결측치 제거
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return {
                'rmse': np.nan,
                'mae': np.nan,
                'r2': np.nan,
                'samples': 0
            }

        # 지표 계산
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        
        # R² (결정계수)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'samples': len(y_true)
        }

    def _predict_with_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        예측 + 메타데이터 결합
        """
        result = df[[self.date_col, 'ticker']].copy()
        result['y_true'] = df[self.target_col].values
        result['y_pred'] = self.model.predict(df[self.feature_cols]).values
        result['abs_error'] = np.abs(result['y_true'] - result['y_pred'])

        return result


# ---------------------------------------------------------------------
# Usage example (documentation only)
# ---------------------------------------------------------------------
"""
from src.models.lightgbm_model import LightGBMModel
from src.modeling.trainer import WalkForwardTrainer
import lightgbm as lgb

# 모델 초기화
model = LightGBMModel(
    model_version="v1",
    params={"objective": "regression", "learning_rate": 0.05},
    feature_list=feature_cols,
    categorical_features=["ticker"]
)

# Trainer 초기화
trainer = WalkForwardTrainer(
    model=model,
    feature_cols=feature_cols,
    target_col="target_log_return",
    categorical_features=["ticker"]
)

# 학습 실행
results = trainer.run(
    df=dataset,
    train_end="2024-01-01",
    valid_window_days=46,
    test_window_days=61,
    num_valid=4,
    fit_kwargs={
        'num_boost_round': 1000,
        'callbacks': [lgb.early_stopping(50), lgb.log_evaluation(100)]
    }
)

# 결과 확인
print(results['test_metrics'])
"""