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
        target_col_prefix: str = "target_log_close",
        date_col: str = "date",
        categorical_features: Optional[List[str]] = None,
        base_price_col: str = "close"
    ):
        self.model = model
        self.feature_cols = feature_cols
        self.target_col_prefix = target_col_prefix
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
        fit_kwargs = fit_kwargs or {}
        target_cols = getattr(self.model, 'target_columns', [self.target_col_prefix])
        
        # 초기화: 데이터 정렬 및 날짜 타입 보정
        df_run = df.copy().sort_values([self.date_col, 'ticker'])
        df_run[self.date_col] = pd.to_datetime(df_run[self.date_col])
        
        # 타깃 생성 (현재 시점 가격)
        for col in target_cols:
            df_run[col] = np.log(df_run[self.base_price_col])

        # 날짜 인덱싱
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

        print(f"🚀 Starting Target-Centric Training (Corrected Logic)")

        for i in range(total_folds):
            is_test_fold = (i == num_valid)
            window_size = test_window_days if is_test_fold else valid_window_days
            eval_end_idx = current_train_end_idx + window_size
            
            if eval_end_idx > len(all_dates):
                break
            
            train_dates = all_dates[current_train_start_idx : current_train_end_idx]
            eval_dates = all_dates[current_train_end_idx : eval_end_idx]
            
            fold_name = "TEST" if is_test_fold else f"Valid-{i+1}"
            print(f"\n[Fold {i+1}] {fold_name} ({len(train_dates)} train days, {len(eval_dates)} eval days)")

            # [핵심 수정] Horizon 별로 "전체 데이터 시프트 -> 슬라이싱" 수행
            for h_idx, col in enumerate(target_cols, 1):
                # 1. 필요한 컬럼만으로 작업용 DF 생성 (메모리 최적화)
                #    전체 기간에 대해 Shift를 수행해야 슬라이싱 경계면의 데이터를 살릴 수 있음
                work_cols = [self.date_col, 'ticker', col] + self.feature_cols
                temp_df = df_run[work_cols].copy()
                
                # 2. 피처 시프트 (전체 데이터 대상)
                for f in self.feature_cols:
                    temp_df[f] = temp_df.groupby('ticker')[f].shift(h_idx)
                
                # 3. 날짜 기준으로 슬라이싱
                train_df = temp_df[temp_df[self.date_col].isin(train_dates)]
                eval_df = temp_df[temp_df[self.date_col].isin(eval_dates)]
                
                # 4. 결측 제거 (이제 경계면 데이터는 보존되고, 진짜 결측만 제거됨)
                train_df = train_df.dropna()
                eval_df = eval_df.dropna()

                # 5. 모델 학습 (해당 Horizon만)
                self.model.fit(
                    train_df[self.feature_cols],
                    train_df[[col]], # DataFrame 형태로 전달
                    eval_set=[(eval_df[self.feature_cols], eval_df[[col]])],
                    **fit_kwargs
                )

            # 평가 및 예측
            metrics = self._evaluate(df_run, eval_dates, target_cols)
            
            if is_test_fold:
                test_metrics = metrics
                test_predictions = self._predict_with_metadata(df_run, eval_dates, target_cols)
                final_model = copy.deepcopy(self.model)
                print(f"   ✅ {fold_name} RMSE: {metrics['avg_rmse']:.6f}")
            else:
                valid_metrics_history.append(metrics)
                print(f"   ✅ {fold_name} RMSE: {metrics['avg_rmse']:.6f}")
            
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

    def _evaluate(self, df_run, eval_dates, target_cols) -> Dict[str, float]:
        rmses = []
        # 평가 시에도 Shift-then-Slice 로직 적용
        cols_needed = [self.date_col, 'ticker'] + self.feature_cols + target_cols
        base_df = df_run[cols_needed].copy() # 전체 데이터 복사본 (피처 원본)
        
        for h_idx, col in enumerate(target_cols, 1):
            temp_df = base_df.copy()
            for f in self.feature_cols:
                temp_df[f] = temp_df.groupby('ticker')[f].shift(h_idx)
            
            # 슬라이싱 & Dropna
            eval_df = temp_df[temp_df[self.date_col].isin(eval_dates)].dropna(subset=self.feature_cols + [col])
            
            if len(eval_df) == 0: continue
            
            y_true = eval_df[col].values
            # target_name을 사용하여 해당 Horizon 모델로만 예측
            y_pred = self.model.predict(eval_df[self.feature_cols], target_name=col).values
            
            rmses.append(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            
        return {'avg_rmse': np.mean(rmses) if rmses else np.nan, 'samples': len(eval_dates)}

    def _predict_with_metadata(self, df_run, eval_dates, target_cols) -> pd.DataFrame:
        # 결과용 뼈대: 평가 날짜의 원본 데이터
        eval_base = df_run[df_run[self.date_col].isin(eval_dates)].copy()
        result = eval_base[[self.date_col, 'ticker', self.base_price_col]].copy()
        
        # 예측용 전체 데이터 준비
        cols_needed = [self.date_col, 'ticker'] + self.feature_cols
        full_base = df_run[cols_needed].copy()
        
        for h_idx, col in enumerate(target_cols, 1):
            # 전체 데이터 시프트
            temp_full = full_base.copy()
            for f in self.feature_cols:
                temp_full[f] = temp_full.groupby('ticker')[f].shift(h_idx)
            
            # 평가 구간 슬라이싱 (Dropna 하지 않음 -> 결측이면 NaN으로 남겨서 예측 시도 or 채움)
            # 여기서는 dropna를 하면 행이 사라져서 result와 인덱스 매칭이 깨짐.
            # 하지만 shift로 인해 앞부분이 NaN이면 예측 불가.
            # -> lightgbm은 NaN 예측 가능. 따라서 dropna 없이 진행.
            eval_slice = temp_full[temp_full[self.date_col].isin(eval_dates)]
            
            # 인덱스 정렬 보장
            # eval_slice와 result는 같은 날짜/종목 순서여야 함 (sort_values 되어 있다고 가정)
            
            preds = self.model.predict(eval_slice[self.feature_cols], target_name=col)
            
            # 결과 할당 (Index alignment)
            result[f'pred_{col}'] = preds.values
            result[f'true_{col}'] = eval_base[col].values if col in eval_base else np.log(result[self.base_price_col])

        return result