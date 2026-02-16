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
        target_col_name: str = "target_log_close",
        horizons: List[int] = [1],
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
        Vectorized Walk-Forward 학습 실행
        - 피처 행렬 X는 고정하고, 타깃 y를 여러 시점(Target shifting)으로 만들어 한 번에 학습합니다.
        - 모델 종류(LGBM, RF)에 상관없이 단일 fit 호출을 보장합니다.
        """
        fit_kwargs = fit_kwargs or {}
        
        # 1. 데이터 전처리: Multi-output 타깃 생성
        # X(t)에 대해 y(t+h1), y(t+h2)... 를 매칭 (Look-ahead Target)
        df_run = df.copy().sort_values([self.date_col, 'ticker'])
        df_run[self.date_col] = pd.to_datetime(df_run[self.date_col])

        if self.target_col_name not in df_run.columns:
            raise KeyError(f"'{self.target_col_name}' 컬럼이 없습니다.")

        # 타깃 컬럼명 리스트 생성
        target_cols = []
        for h in self.horizons:
            col_name = f"{self.target_col_name}_h{h}"
            target_cols.append(col_name)
            # 미래의 값을 현재 행으로 당겨옴 (Shift Backwards)
            df_run[col_name] = df_run.groupby('ticker')[self.target_col_name].shift(-h)

        # 2. 날짜 인덱싱 설정
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

        print(f"🚀 Starting Vectorized Multi-horizon Training")
        print(f"   - Base Target: {self.target_col_name}")
        print(f"   - Horizons: {self.horizons} (Target Cols: {target_cols})")

        for i in range(total_folds):
            is_test_fold = (i == num_valid)
            window_size = test_window_days if is_test_fold else valid_window_days
            eval_end_idx = current_train_end_idx + window_size
            
            if eval_end_idx > len(all_dates):
                print(f"⚠️ 데이터 부족으로 Fold {i+1} 조기 종료.")
                break
            
            train_dates = all_dates[current_train_start_idx : current_train_end_idx]
            eval_dates = all_dates[current_train_end_idx : eval_end_idx]
            
            fold_name = "TEST" if is_test_fold else f"Valid-{i+1}"
            print(f"\n[{fold_name}] Training...")

            # 3. 데이터셋 분할 (NaN 제거 포함)
            # 모든 타깃(h1~hN)이 유효한 행만 학습에 사용 (Intersection)
            cols_needed = [self.date_col, 'ticker'] + self.feature_cols + target_cols
            temp_df = df_run[cols_needed].copy()
            
            train_df = temp_df[temp_df[self.date_col].isin(train_dates)].dropna()
            eval_df = temp_df[temp_df[self.date_col].isin(eval_dates)].dropna()

            if train_df.empty:
                raise ValueError("학습 데이터가 없습니다. 날짜 범위나 시프트 로직을 확인하세요.")

            # 4. 모델 학습 (단 1회 호출!)
            # DataFrame 형태의 y를 전달
            self.model.fit(
                X=train_df[self.feature_cols],
                y=train_df[target_cols],
                eval_set=[(eval_df[self.feature_cols], eval_df[target_cols])],
                **fit_kwargs
            )

            # 5. 평가 및 예측
            metrics = self._evaluate(eval_df, target_cols)
            
            if is_test_fold:
                test_metrics = metrics
                # 테스트셋에 대해서는 전체 데이터(NaN 포함 가능)에 대해 예측 수행
                full_eval_slice = temp_df[temp_df[self.date_col].isin(eval_dates)]
                test_predictions = self._predict_with_metadata(full_eval_slice, target_cols)
                final_model = copy.deepcopy(self.model)
            else:
                valid_metrics_history.append(metrics)
            
            print(f"   ✅ {fold_name} Avg RMSE: {metrics['avg_rmse']:.6f}")
            
            # 윈도우 이동
            current_train_start_idx += valid_window_days
            current_train_end_idx += valid_window_days

        return {
            'valid_metrics': valid_metrics_history,
            'test_metrics': test_metrics,
            'test_predictions': test_predictions,
            'final_model': final_model
        }

    def _evaluate(self, eval_df: pd.DataFrame, target_cols: List[str]) -> Dict[str, float]:
        """
        Vectorized 평가
        """
        if eval_df.empty:
            return {'avg_rmse': np.nan, 'samples': 0}
            
        # 예측 (DataFrame 반환)
        preds_df = self.model.predict(eval_df[self.feature_cols])
        
        # DataFrame 반환을 보장 (일부 모델이 numpy를 줄 수도 있음)
        if isinstance(preds_df, np.ndarray):
            preds_df = pd.DataFrame(preds_df, index=eval_df.index, columns=target_cols)
            
        rmses = []
        for col in target_cols:
            y_true = eval_df[col].values
            y_pred = preds_df[col].values
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            rmses.append(rmse)
            
        return {'avg_rmse': np.mean(rmses) if rmses else np.nan, 'samples': len(eval_df)}

    def _predict_with_metadata(self, eval_df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
        """
        메타데이터를 포함한 예측 결과 생성
        """
        if eval_df.empty:
            return pd.DataFrame()
            
        # 예측 수행
        preds = self.model.predict(eval_df[self.feature_cols])
        
        if isinstance(preds, np.ndarray):
            preds = pd.DataFrame(preds, index=eval_df.index, columns=target_cols)
            
        # 결과 조립
        result = eval_df[[self.date_col, 'ticker']].copy()
        
        # Base price가 있다면 추가 (참조용)
        # 주의: Shift된 데이터셋이 아니라 원본 df_run에서 가져오는 것이 안전하나,
        # 여기서는 eval_df가 이미 필요한 컬럼을 가지고 있다고 가정하지 않고 원본 매핑을 피함
        
        for col in target_cols:
            result[f'pred_{col}'] = preds[col].values
            # 정답지(True)가 NaN일 수도 있음 (미래 데이터인 경우)
            result[f'true_{col}'] = eval_df[col].values
            
        return result