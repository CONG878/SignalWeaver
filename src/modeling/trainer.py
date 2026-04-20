# src/modeling/trainer.py

from __future__ import annotations
import warnings
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import copy
import scipy.stats as stats
from src.models.base import ModelBase
from src.utils.integration import reconstruct_log_close

_VALID_TARGET_TYPES = ("log_close", "log_return_1d")


class WalkForwardTrainer:
    def __init__(
        self,
        *,
        model: ModelBase,
        feature_cols: List[str],
        target_col_name: str = "target_log_close",
        target_type: str = "log_close",
        horizons: List[int] = [1],
        date_col: str = "date",
        categorical_features: Optional[List[str]] = None,
        base_price_col: str = "close",
        integration_order: int = 1,
    ):
        """
        Multi-horizon Walk-Forward 학습을 관리하는 Trainer 클래스.

        Parameters
        ----------
        target_col_name : str
            02단계 dataset의 기준 타겟 컬럼명
        target_type : str
            'log_close'     : log(close(t+n)) 절대 가격 직접 예측
            'log_return_1d' : 1일 당일 등락률 예측
        integration_order : int, default=1
            log_return_1d 모드에서 log-close 역산 시 사용할 수치 적분 차수.
            0 — 직사각형 (v3.9.0)
            1 — 사다리꼴  (v3.9.1, 기본값)
            2 — Adams-Moulton 2-step (v4.1.0)
        """
        if target_type == "log_return":
            raise ValueError(
                "[REMOVED v3.9.0] 'log_return' 모드는 정식 폐기되었습니다. "
                "'log_close' 또는 'log_return_1d'를 사용하세요."
            )
        if target_type not in _VALID_TARGET_TYPES:
            raise ValueError(f"target_type은 {_VALID_TARGET_TYPES} 중 하나여야 합니다.")

        self.model = model
        self.feature_cols = feature_cols
        self.target_col_name = target_col_name
        self.target_type = target_type
        self.horizons = horizons
        self.date_col = date_col
        self.categorical_features = categorical_features or []
        self.base_price_col = base_price_col
        self.integration_order = integration_order

    def run(
        self,
        df: pd.DataFrame,
        *,
        train_end: str,
        valid_window_days: int,
        test_window_days: int,
        fit_kwargs: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        2-Fold Walk-Forward 학습 실행.
        """
        fit_kwargs = fit_kwargs or {}

        df_run = df.copy().sort_values([self.date_col, 'ticker'])
        df_run[self.date_col] = pd.to_datetime(df_run[self.date_col])

        if self.target_col_name not in df_run.columns:
            raise KeyError(
                f"'{self.target_col_name}' 컬럼이 데이터에 없습니다."
            )

        target_cols = self._build_target_cols(df_run)

        all_dates = np.sort(df_run[self.date_col].unique())
        train_end_val = pd.to_datetime(train_end).to_datetime64()
        train_end_idx = np.searchsorted(all_dates, train_end_val)
        if train_end_idx < len(all_dates) and all_dates[train_end_idx] == train_end_val:
            train_end_idx += 1

        G = max(self.horizons)

        val_train_start_idx = 0
        val_train_end_idx   = train_end_idx - G
        val_eval_start_idx  = train_end_idx
        val_eval_end_idx    = train_end_idx + valid_window_days

        test_train_start_idx = valid_window_days
        test_train_end_idx   = train_end_idx + valid_window_days - G
        test_eval_start_idx  = train_end_idx + valid_window_days
        test_eval_end_idx    = test_eval_start_idx + test_window_days

        if val_train_end_idx <= val_train_start_idx:
            raise ValueError(f"embargo_gap_days={G}가 너무 큽니다.")
        if val_eval_end_idx > len(all_dates):
            raise ValueError("검증 폴드 데이터 부족.")
        if test_eval_end_idx > len(all_dates):
            raise ValueError("테스트 폴드 데이터 부족.")

        val_train_dates  = all_dates[val_train_start_idx : val_train_end_idx]
        val_eval_dates   = all_dates[val_eval_start_idx  : val_eval_end_idx]
        test_train_dates = all_dates[test_train_start_idx : test_train_end_idx]
        test_eval_dates  = all_dates[test_eval_start_idx  : test_eval_end_idx]

        print(f"🚀 Walk-Forward Training (2-Fold)")
        print(f"   Target  : {self.target_col_name}  |  "
              f"Type: {self.target_type}  |  Horizons: {self.horizons}  |  "
              f"integration_order: {self.integration_order}")
        print(f"   Embargo : {G} 거래일")

        _ref_col = 'target_log_close'
        _extra   = []
        if self.target_type == "log_return_1d":
            if _ref_col in df_run.columns:
                _extra.append(_ref_col)
            if self.target_col_name in df_run.columns:
                _extra.append(self.target_col_name)
            # order=2: lag1 앵커 컬럼
            if self.integration_order == 2:
                lag1_col = 'target_log_return_1d_lag1'
                if lag1_col not in df_run.columns:
                    raise KeyError(
                        f"integration_order=2에는 '{lag1_col}' 컬럼이 필요합니다. "
                        "02단계(02_build_dataset.ipynb)를 재실행하세요."
                    )
                _extra.append(lag1_col)

        cols_needed = [self.date_col, 'ticker'] + self.feature_cols + target_cols + _extra
        temp_df = df_run[cols_needed].copy()

        # ── FOLD 1: 검증 폴드 ──────────────────────────────────────
        print(f"\n[검증 폴드] 학습 중...")
        val_train_df = temp_df[temp_df[self.date_col].isin(val_train_dates)].dropna()
        val_eval_df  = temp_df[temp_df[self.date_col].isin(val_eval_dates)].dropna()

        self.model.fit(
            X=val_train_df[self.feature_cols],
            y=val_train_df[target_cols],
            eval_set=[(val_eval_df[self.feature_cols], val_eval_df[target_cols])],
            **fit_kwargs
        )

        valid_metrics   = self._evaluate(val_eval_df, target_cols)
        full_val_slice  = temp_df[temp_df[self.date_col].isin(val_eval_dates)]
        val_predictions = self._predict_with_metadata(full_val_slice, target_cols, fold='valid')

        print(f"   ✅ 검증 RMS RMSE: {valid_metrics['rms_rmse']:.6f}  "
              f"| Avg IC: {valid_metrics['avg_ic']:.4f}")

        # ── FOLD 2: 테스트 폴드 ──────────────────────────────────────
        print(f"\n[테스트 폴드] 학습 중...")
        test_model = copy.deepcopy(self.model)
        test_model.is_fitted = False

        test_train_df = temp_df[temp_df[self.date_col].isin(test_train_dates)].dropna()
        test_eval_df  = temp_df[temp_df[self.date_col].isin(test_eval_dates)].dropna()

        test_model.fit(
            X=test_train_df[self.feature_cols],
            y=test_train_df[target_cols],
            eval_set=[(test_eval_df[self.feature_cols], test_eval_df[target_cols])],
            **fit_kwargs
        )

        _prev_model = self.model
        self.model  = test_model
        test_metrics    = self._evaluate(test_eval_df, target_cols)
        full_test_slice = temp_df[temp_df[self.date_col].isin(test_eval_dates)]
        test_predictions = self._predict_with_metadata(full_test_slice, target_cols, fold='test')
        self.model = _prev_model

        print(f"   ✅ 테스트 RMS RMSE: {test_metrics['rms_rmse']:.6f}  "
              f"| Avg IC: {test_metrics['avg_ic']:.4f}")

        return {
            'valid_metrics'     : valid_metrics,
            'val_predictions'   : val_predictions,
            'test_metrics'      : test_metrics,
            'test_predictions'  : test_predictions,
            'final_model'       : test_model,
            'target_cols'       : target_cols,
            'target_type'       : self.target_type,
            'embargo_gap_days'  : G,
            'integration_order' : self.integration_order,
        }

    # ──────────────────────────────────────────────────────────────
    # 타겟 생성
    # ──────────────────────────────────────────────────────────────

    def _build_target_cols(self, df_run: pd.DataFrame) -> List[str]:
        target_cols = []
        if self.target_type == "log_return_1d":
            for h in self.horizons:
                col_name = f"target_log_return_1d_h{h}"
                df_run[col_name] = df_run.groupby('ticker')[self.target_col_name].shift(-h)
                target_cols.append(col_name)
        else:
            for h in self.horizons:
                col_name = f"{self.target_col_name}_h{h}"
                df_run[col_name] = df_run.groupby('ticker')[self.target_col_name].shift(-h)
                target_cols.append(col_name)
        return target_cols

    # ──────────────────────────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────────────────────────

    def _evaluate(self, eval_df: pd.DataFrame, target_cols: List[str]) -> Dict[str, float]:
        if eval_df.empty:
            return {'rms_rmse': np.nan, 'avg_rmse': np.nan, 'avg_ic': np.nan,
                    'per_horizon': {}, 'samples': 0}

        preds_df = self.model.predict(eval_df[self.feature_cols])
        if isinstance(preds_df, np.ndarray):
            preds_df = pd.DataFrame(preds_df, index=eval_df.index, columns=target_cols)

        temp_eval = eval_df[[self.date_col]].copy()
        for col in target_cols:
            temp_eval[f'pred_{col}'] = preds_df[col].values
            temp_eval[f'true_{col}'] = eval_df[col].values

        if self.target_type == "log_return_1d" and 'target_log_close' in eval_df.columns:
            log_close_base = eval_df['target_log_close'].values
            sorted_cols    = sorted(target_cols, key=lambda c: int(c.split('_h')[-1]))
            delta_y_t      = eval_df[self.target_col_name].values

            # order=2 앵커
            delta_y_t_minus_1 = (
                eval_df['target_log_return_1d_lag1'].values
                if self.integration_order == 2
                else None
            )

            for idx, col in enumerate(sorted_cols):
                pred_delta_h = preds_df[col].values
                true_delta_h = eval_df[col].values

                cum_pred = sum(preds_df[c].values for c in sorted_cols[:idx + 1])
                cum_true = sum(eval_df[c].values  for c in sorted_cols[:idx + 1])

                # order=2: delta_y_h_minus_1
                if self.integration_order == 2:
                    pred_delta_h_minus_1 = (
                        preds_df[sorted_cols[idx - 1]].values if idx > 0
                        else delta_y_t  # 청크 첫 스텝: delta_y_t로 fallback
                    )
                    true_delta_h_minus_1 = (
                        eval_df[sorted_cols[idx - 1]].values if idx > 0
                        else delta_y_t
                    )
                else:
                    pred_delta_h_minus_1 = None
                    true_delta_h_minus_1 = None

                temp_eval[f'pred_{col}'] = reconstruct_log_close(
                    log_close_base, cum_pred, delta_y_t, pred_delta_h,
                    delta_y_t_minus_1=delta_y_t_minus_1,
                    delta_y_h_minus_1=pred_delta_h_minus_1,
                    order=self.integration_order,
                )
                temp_eval[f'true_{col}'] = reconstruct_log_close(
                    log_close_base, cum_true, delta_y_t, true_delta_h,
                    delta_y_t_minus_1=delta_y_t_minus_1,
                    delta_y_h_minus_1=true_delta_h_minus_1,
                    order=self.integration_order,
                )

        per_horizon = {}
        for col in target_cols:
            y_true = temp_eval[f'true_{col}'].values
            y_pred = temp_eval[f'pred_{col}'].values
            mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
            rmse = np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)) if mask.any() else np.nan

            daily_ics = []
            for date, group in temp_eval.groupby(self.date_col):
                if len(group) > 1:
                    g_true = group[f'true_{col}'].values
                    g_pred = group[f'pred_{col}'].values
                    g_mask = ~np.isnan(g_true) & ~np.isnan(g_pred)
                    if g_mask.sum() > 1:
                        corr, _ = stats.spearmanr(g_true[g_mask], g_pred[g_mask])
                        if not np.isnan(corr):
                            daily_ics.append(corr)

            ic_mean = np.mean(daily_ics) if daily_ics else np.nan
            ic_std  = np.std(daily_ics)  if daily_ics else np.nan
            icir    = (ic_mean / ic_std) if (ic_std and ic_std > 0) else np.nan
            per_horizon[col] = {'rmse': rmse, 'ic_mean': ic_mean, 'icir': icir}

        valid_rmses = [v['rmse']    for v in per_horizon.values() if not np.isnan(v['rmse'])]
        valid_ics   = [v['ic_mean'] for v in per_horizon.values() if not np.isnan(v['ic_mean'])]

        return {
            'rms_rmse'    : float(np.sqrt(np.mean(np.square(valid_rmses)))) if valid_rmses else np.nan,
            'avg_rmse'    : float(np.mean(valid_rmses)) if valid_rmses else np.nan,
            'avg_ic'      : float(np.mean(valid_ics))   if valid_ics   else np.nan,
            'per_horizon' : per_horizon,
            'samples'     : int(eval_df[target_cols].notna().all(axis=1).sum()),
        }

    def _predict_with_metadata(
        self,
        eval_df: pd.DataFrame,
        target_cols: List[str],
        fold: str = 'unknown'
    ) -> pd.DataFrame:
        if eval_df.empty:
            return pd.DataFrame()

        preds = self.model.predict(eval_df[self.feature_cols])
        if isinstance(preds, np.ndarray):
            preds = pd.DataFrame(preds, index=eval_df.index, columns=target_cols)

        result = eval_df[[self.date_col, 'ticker']].copy().reset_index(drop=True)
        preds  = preds.reset_index(drop=True)

        for col in target_cols:
            result[f'pred_{col}'] = preds[col].values
            result[f'true_{col}'] = eval_df[col].reset_index(drop=True).values

        result['fold'] = fold
        return result
