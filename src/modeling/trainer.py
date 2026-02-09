# src/modeling/trainer.py

from __future__ import annotations
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from src.models.base import ModelBase
import copy

class WalkForwardTrainer:
    def __init__(
        self,
        *,
        model: ModelBase,
        feature_cols: List[str],
        target_col_name: str = "target_log_close", # 베이스 타깃명
        horizons: List[int] = [1],                 # 시차 리스트
        date_col: str = "date",
        categorical_features: Optional[List[str]] = None,
        base_price_col: str = "close"
    ):
        self.model = model
        self.feature_cols = feature_cols
        self.target_col_name = target_col_name
        self.horizons = horizons
        self.date_col = date_col
        self.categorical_features = categorical_features or []
        self.base_price_col = base_price_col

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
        Target-Centric Walk-Forward 학습 및 검증 실행
        - 02단계에서 생성된 공통 타깃(self.target_col_name)을 사용합니다.
        - 각 Horizon별로 전체 데이터 시프트 후 슬라이싱하여 경계면 데이터 손실을 방지합니다.
        """
        fit_kwargs = fit_kwargs or {}
        
        # 모델 및 트레이너의 타깃 식별자 생성 (예: target_log_close_h1, ...)
        target_cols = [f"{self.target_col_name}_h{h}" for h in self.horizons]
        
        # 데이터 정렬 및 날짜 타입 보정
        df_run = df.copy().sort_values([self.date_col, 'ticker'])
        df_run[self.date_col] = pd.to_datetime(df_run[self.date_col])
        
        # 02단계에서 생성된 베이스 타깃 컬럼 존재 여부 확인
        if self.target_col_name not in df_run.columns:
            raise KeyError(f"'{self.target_col_name}' 컬럼이 데이터셋에 없습니다. 02단계를 먼저 실행하세요.")

        # 날짜 인덱싱 설정
        all_dates = np.sort(df_run[self.date_col].unique())
        train_end_val = pd.to_datetime(train_end).to_datetime64()
        train_end_idx = np.searchsorted(all_dates, train_end_val)
        if train_end_idx < len(all_dates) and all_dates[train_end_idx] == train_end_val:
            train_end_idx += 1

        valid_metrics_history = []
        test_metrics = {}
        test_predictions = pd.DataFrame()
        final_model = None
        
        current_train_start_idx = 0
        current_train_end_idx = train_end_idx
        total_folds = num_valid + 1

        print(f"🚀 Starting Target-Centric Multi-horizon Training")
        print(f"   - Base Target: {self.target_col_name}")
        print(f"   - Horizons: {self.horizons}")

        for i in range(total_folds):
            is_test_fold = (i == num_valid)
            window_size = test_window_days if is_test_fold else valid_window_days
            eval_end_idx = current_train_end_idx + window_size
            
            if eval_end_idx > len(all_dates):
                print(f"⚠️ 데이터 부족으로 Fold {i+1}에서 조기 종료합니다.")
                break
            
            train_dates = all_dates[current_train_start_idx : current_train_end_idx]
            eval_dates = all_dates[current_train_end_idx : eval_end_idx]
            
            fold_name = "TEST" if is_test_fold else f"Valid-{i+1}"
            print(f"\n[{fold_name}] Training horizons...")

            # 각 시차(Horizon)별 독립 모델 학습
            for h in self.horizons:
                target_id = f"{self.target_col_name}_h{h}"
                
                # 1. 전체 데이터 시프트 (Shift-then-Slice)
                # 공통 타깃인 target_log_close는 고정하고 피처만 과거로 시프트
                work_cols = [self.date_col, 'ticker', self.target_col_name] + self.feature_cols
                temp_df = df_run[work_cols].copy()
                
                for f in self.feature_cols:
                    temp_df[f] = temp_df.groupby('ticker')[f].shift(h)
                
                # 2. 날짜 슬라이싱
                train_df = temp_df[temp_df[self.date_col].isin(train_dates)].dropna()
                eval_df = temp_df[temp_df[self.date_col].isin(eval_dates)].dropna()

                # 3. 개별 모델 Fit (y는 시프트되지 않은 오늘의 정답 사용)
                self.model.fit(
                    train_df[self.feature_cols],
                    train_df[[self.target_col_name]], 
                    eval_set=[(eval_df[self.feature_cols], eval_df[[self.target_col_name]])],
                    target_name=target_id,
                    **fit_kwargs
                )

            # 4. Fold 평가 및 예측 결과 기록
            metrics = self._evaluate(df_run, eval_dates, self.horizons)
            
            if is_test_fold:
                test_metrics = metrics
                test_predictions = self._predict_with_metadata(df_run, eval_dates, self.horizons)
                final_model = copy.deepcopy(self.model)
            else:
                valid_metrics_history.append(metrics)
            
            print(f"   ✅ {fold_name} Avg RMSE: {metrics['avg_rmse']:.6f}")
            
            # 윈도우 이동
            shift_step = valid_window_days
            current_train_start_idx += shift_step
            current_train_end_idx += shift_step

        return {
            'valid_metrics': valid_metrics_history,
            'test_metrics': test_metrics,
            'test_predictions': test_predictions,
            'final_model': final_model
        }

    def _evaluate(self, df_run: pd.DataFrame, eval_dates: np.ndarray, horizons: List[int]) -> Dict[str, float]:
        """
        🔧 수정: 모든 horizon에서 공통으로 유효한 인덱스만 사용하여 평가
        
        문제: 각 horizon마다 shift + dropna 하면 유효 행이 달라져 길이 불일치
        해결: 모든 horizon에 대해 교집합 인덱스를 먼저 찾고, 그것만 사용
        """
        rmses = []
        cols_needed = [self.date_col, 'ticker', self.target_col_name] + self.feature_cols
        base_df = df_run[cols_needed].copy()
        
        # 1️⃣ 모든 horizon에서 공통으로 유효한 인덱스 찾기
        valid_indices = None
        
        for h in horizons:
            temp_df = base_df.copy()
            for f in self.feature_cols:
                temp_df[f] = temp_df.groupby('ticker')[f].shift(h)
            
            # eval_dates 슬라이싱
            eval_slice = temp_df[temp_df[self.date_col].isin(eval_dates)]
            
            # 이 horizon에서 유효한 인덱스 (NaN 없는 행)
            h_valid_indices = eval_slice.dropna().index
            
            # 교집합 업데이트
            if valid_indices is None:
                valid_indices = h_valid_indices
            else:
                valid_indices = valid_indices.intersection(h_valid_indices)
        
        # 유효 인덱스가 없으면 조기 종료
        if valid_indices is None or len(valid_indices) == 0:
            return {'avg_rmse': np.nan, 'samples': 0}
        
        # 2️⃣ 공통 인덱스로 각 horizon 평가
        for h in horizons:
            target_id = f"{self.target_col_name}_h{h}"
            
            temp_df = base_df.copy()
            for f in self.feature_cols:
                temp_df[f] = temp_df.groupby('ticker')[f].shift(h)
            
            # ✅ 공통 인덱스만 사용 (길이 일치 보장)
            eval_df = temp_df.loc[valid_indices]
            
            y_true = eval_df[self.target_col_name].values
            y_pred = self.model.predict(eval_df[self.feature_cols], target_name=target_id).values
            
            rmses.append(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            
        return {'avg_rmse': np.mean(rmses) if rmses else np.nan, 'samples': len(valid_indices)}

    def _predict_with_metadata(self, df_run: pd.DataFrame, eval_dates: np.ndarray, horizons: List[int]) -> pd.DataFrame:
        """
        🔧 수정: 공통 유효 인덱스만 사용하여 예측 메타데이터 생성
        
        _evaluate와 동일한 로직 적용
        """
        # 1️⃣ 모든 horizon에서 공통 유효 인덱스 찾기
        cols_needed = [self.date_col, 'ticker'] + self.feature_cols
        full_base = df_run[cols_needed].copy()
        
        valid_indices = None
        
        for h in horizons:
            temp_full = full_base.copy()
            for f in self.feature_cols:
                temp_full[f] = temp_full.groupby('ticker')[f].shift(h)
            
            eval_slice = temp_full[temp_full[self.date_col].isin(eval_dates)]
            h_valid_indices = eval_slice.dropna().index
            
            if valid_indices is None:
                valid_indices = h_valid_indices
            else:
                valid_indices = valid_indices.intersection(h_valid_indices)
        
        if valid_indices is None or len(valid_indices) == 0:
            return pd.DataFrame()
        
        # 2️⃣ 결과용 뼈대 (공통 인덱스만)
        eval_base = df_run.loc[valid_indices].copy()
        result = eval_base[[self.date_col, 'ticker', self.base_price_col, self.target_col_name]].copy()
        
        # 3️⃣ 각 horizon 예측
        for h in horizons:
            target_id = f"{self.target_col_name}_h{h}"
            
            temp_full = full_base.copy()
            for f in self.feature_cols:
                temp_full[f] = temp_full.groupby('ticker')[f].shift(h)
            
            # ✅ 공통 인덱스만 사용
            eval_slice = temp_full.loc[valid_indices]
            
            # 예측 수행
            preds = self.model.predict(eval_slice[self.feature_cols], target_name=target_id)
            
            # 결과 저장
            result[f'pred_{target_id}'] = preds.values
            result[f'true_{target_id}'] = eval_base[self.target_col_name].values
            
        return result
