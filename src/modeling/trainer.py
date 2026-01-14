# src/modeling/trainer.py

from __future__ import annotations
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from src.models.base import ModelBase
import copy

class WalkForwardTrainer:
    """
    Rolling Window 방식의 Walk-Forward 학습/검증 관리자
    - 매 검증 구간마다 학습 데이터를 이동(Rolling)시키며 모델을 재학습합니다.
    - Test Set은 마지막 검증 구간 다음의 N+1번째 구간으로 취급합니다.
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
        Rolling Window Walk-Forward 실행

        Parameters
        ----------
        df : DataFrame (전체 데이터)
        train_end : str (첫 번째 Fold의 학습 종료일)
        valid_window_days : int (검증 윈도우 크기 - 거래일 기준)
        test_window_days : int (테스트 윈도우 크기 - 거래일 기준)
        num_valid : int (검증 Fold 횟수)
        """
        fit_kwargs = fit_kwargs or {}
        
        # 날짜 정렬 및 초기화
        df = df.sort_values(self.date_col).reset_index(drop=True)
        all_dates = pd.to_datetime(df[self.date_col].unique())
        all_dates = np.sort(all_dates)
        
        # 첫 학습 종료일의 인덱스 찾기
        train_end_dt = pd.to_datetime(train_end)
        if train_end_dt not in all_dates:
            # train_end가 거래일이 아니면 가장 가까운 이전 거래일 찾기
            train_end_idx = np.searchsorted(all_dates, train_end_dt)
        else:
            train_end_idx = np.where(all_dates == train_end_dt)[0][0] + 1

        # 결과 저장소
        valid_metrics_history = []
        test_metrics = {}
        test_predictions = pd.DataFrame()
        
        # 총 반복 횟수: 검증 횟수(num_valid) + 테스트(1)
        # 테스트는 마지막 검증 직후의 구간을 의미
        total_folds = num_valid + 1
        
        print(f"🚀 Starting Rolling Window Training (Total {total_folds} folds)")
        print(f"   - Initial Train End: {train_end}")
        print(f"   - Window Size: {valid_window_days} days (Test: {test_window_days} days)")
        
        # 초기 학습 데이터 크기 (Rolling Window를 위해 고정)
        # Start Index는 0에서 시작
        current_train_start_idx = 0
        current_train_end_idx = train_end_idx
        
        final_model = None

        for i in range(total_folds):
            is_test_fold = (i == num_valid) # 마지막 루프는 Test Fold
            
            # 1. 구간 설정
            window_size = test_window_days if is_test_fold else valid_window_days
            
            # 검증/테스트 구간의 끝 인덱스
            eval_end_idx = current_train_end_idx + window_size
            
            if eval_end_idx > len(all_dates):
                raise ValueError(f"데이터 부족: Fold {i+1}을 위한 데이터가 모자랍니다.")
            
            # 날짜 기준으로 데이터 슬라이싱
            train_dates = all_dates[current_train_start_idx : current_train_end_idx]
            eval_dates = all_dates[current_train_end_idx : eval_end_idx]
            
            train_df = df[df[self.date_col].isin(train_dates)]
            eval_df = df[df[self.date_col].isin(eval_dates)]
            
            fold_name = "TEST" if is_test_fold else f"Valid-{i+1}"
            print(f"\n[Fold {i+1}/{total_folds}] {fold_name}")

            # .date() 에러 해결을 위해 pd.Timestamp로 변환 후 출력
            t_start = pd.Timestamp(train_dates[0]).date()
            t_end = pd.Timestamp(train_dates[-1]).date()
            e_start = pd.Timestamp(eval_dates[0]).date()
            e_end = pd.Timestamp(eval_dates[-1]).date()
            
            print(f"   Train: {t_start} ~ {t_end} ({len(train_df):,} rows)")
            print(f"   Eval : {e_start} ~ {e_end} ({len(eval_df):,} rows)")
            
            # 2. 모델 재설정 (이전 학습 상태 초기화)
            # 주의: model 객체가 stateful하다면 reset하거나 새로 생성해야 함.
            # LightGBMModel은 fit 호출 시 보통 내부 부스터를 새로 만듭니다.
            # 더 확실하게 하기 위해 fit 내에서 부스터가 초기화되는지 확인 필요.
            # 여기서는 모델 인스턴스는 유지하되 fit으로 덮어쓰기 합니다.
            
            # 3. 학습
            # Valid Fold인 경우: eval_set을 사용하여 early stopping 가능
            # Test Fold인 경우: Test set을 eval_set으로 쓰면 data leakage는 아니지만(미래 데이터 X),
            # 보통 Test시에는 Full Train 후 예측만 수행. 하지만 일관성을 위해 여기선 eval_set 사용.
            
            self.model.fit(
                train_df[self.feature_cols],
                train_df[self.target_col],
                eval_set=[(eval_df[self.feature_cols], eval_df[self.target_col])],
                **fit_kwargs
            )
            
            # 4. 평가 및 기록
            metrics = self._evaluate(eval_df)
            
            if is_test_fold:
                test_metrics = metrics
                test_predictions = self._predict_with_metadata(eval_df)
                final_model = self.model # 마지막 모델 저장
                print(f"   ✅ {fold_name} RMSE: {metrics['rmse']:.6f}")
            else:
                valid_metrics_history.append(metrics)
                print(f"   ✅ {fold_name} RMSE: {metrics['rmse']:.6f}")
            
            # 5. 다음 Fold를 위해 윈도우 슬라이딩 (Rolling Window)
            # 학습 시작일과 종료일을 모두 Window 크기만큼 뒤로 밈
            # 단, Test Fold 직전에는 Test Window가 아니라 Valid Window만큼 밀어야 자연스러운 연속성 유지
            # 사용자의 의도: "검증 단계가 올라갈 때마다 검증 단위 기간만큼 슬라이드"
            
            shift_step = valid_window_days
            current_train_start_idx += shift_step
            current_train_end_idx += shift_step
            
        return {
            'train_metrics': self._evaluate(train_df), # 마지막 Fold의 Train metrics
            'valid_metrics': valid_metrics_history,
            'test_metrics': test_metrics,
            'test_predictions': test_predictions,
            'final_model': final_model # 저장용
        }

    def _evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        y_true = df[self.target_col].values
        y_pred = self.model.predict(df[self.feature_cols]).values
        
        # 결측 제거
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true, y_pred = y_true[mask], y_pred[mask]
        
        if len(y_true) == 0: return {'rmse': np.nan, 'r2': np.nan}
        
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        
        return {'rmse': rmse, 'r2': r2, 'samples': len(y_true)}

    def _predict_with_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df[[self.date_col, 'ticker']].copy()
        result['y_true'] = df[self.target_col].values
        result['y_pred'] = self.model.predict(df[self.feature_cols]).values
        return result