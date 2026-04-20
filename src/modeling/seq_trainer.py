"""
SeqTrainer — Seq 모델 전용 학습 관리자

## v4.1.0 변경
- reconstruct_log_close(order=) 지원 (integration_order 파라미터 추가)
- trapezoidal.py → integration.py 참조 변경
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader

from src.models.seq_base import SeqModelBase
from src.data_loader.seq_builder import SeqDataset, split_by_date
from src.utils.integration import reconstruct_log_close


class SeqTrainer:
    """
    Seq 모델 Walk-Forward 학습 관리자.

    Parameters
    ----------
    model : SeqModelBase
    feature_cols : List[str]
    target_col : str
    seq_len : int
    forecast_horizon : int
    stride : int
    date_col : str
    integration_order : int, default=1
        log_return_1d 역산 시 수치 적분 차수.
        0 — 직사각형 / 1 — 사다리꼴 / 2 — Adams-Moulton 2-step
    """

    def __init__(
        self,
        *,
        model: SeqModelBase,
        feature_cols: List[str],
        target_col: str = "target_log_return_1d",
        seq_len: int,
        forecast_horizon: int,
        stride: int = 1,
        date_col: str = "date",
        integration_order: int = 1,
    ):
        self.model             = model
        self.feature_cols      = feature_cols
        self.target_col        = target_col
        self.seq_len           = seq_len
        self.forecast_horizon  = forecast_horizon
        self.stride            = stride
        self.date_col          = date_col
        self.integration_order = integration_order

    def run(
        self,
        df: pd.DataFrame,
        *,
        train_end: str,
        valid_window_days: int,
        test_window_days: int,
        n_folds: int = 1,
        resume: bool = False,
        fit_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Walk-Forward 학습 실행."""
        fit_kwargs  = fit_kwargs or {}
        embargo_gap = self.forecast_horizon

        # order=2 앵커 컬럼 존재 확인
        if self.integration_order == 2:
            lag1_col = 'target_log_return_1d_lag1'
            if lag1_col not in df.columns:
                raise KeyError(
                    f"integration_order=2에는 '{lag1_col}' 컬럼이 필요합니다. "
                    "02단계(02_build_dataset.ipynb)를 재실행하세요."
                )

        print(f"🔨 시퀀스 인덱스 구성 중 (seq_len={self.seq_len}, "
              f"horizon={self.forecast_horizon}, stride={self.stride})...")

        splits = split_by_date(
            df=df,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            seq_len=self.seq_len,
            forecast_horizon=self.forecast_horizon,
            stride=self.stride,
            train_end=train_end,
            valid_window_days=valid_window_days,
            test_window_days=test_window_days,
            embargo_gap=embargo_gap,
            date_col=self.date_col,
        )

        ds_train = splits["ds_train"]
        ds_val   = splits["ds_val"]
        ds_test  = splits["ds_test"]
        d        = splits["dates"]

        print(f"   훈련 샘플: {len(ds_train):,}  검증: {len(ds_val):,}  테스트: {len(ds_test):,}")
        print(f"\n🚀 Seq Walk-Forward Training ({n_folds}-Fold)")
        print(f"   Target  : {self.target_col}  |  Horizon: {self.forecast_horizon}  |  "
              f"seq_len: {self.seq_len}  |  integration_order: {self.integration_order}")

        if n_folds == 1:
            return self._run_1fold(df, ds_train, ds_val, ds_test, d, resume, fit_kwargs, embargo_gap)
        elif n_folds == 2:
            return self._run_2fold(df, splits, d, resume, fit_kwargs, embargo_gap)
        else:
            raise ValueError(f"n_folds는 1 또는 2여야 합니다. 현재: {n_folds}")

    # ------------------------------------------------------------------
    # 1-Fold
    # ------------------------------------------------------------------

    def _run_1fold(self, df, ds_train, ds_val, ds_test, d, resume, fit_kwargs, embargo_gap):
        print(f"\n   [1-Fold]")
        print(f"   훈련: ~ {d['train_end']}  검증(ES): {d['val_start']} ~ {d['val_end']}")
        print(f"   테스트: {d['test_start']} ~ {d['test_end']}")

        self.model.fit(ds_train, eval_set=ds_val, resume=resume, **fit_kwargs)

        test_metrics, test_predictions = self._evaluate_and_build(ds_test, df, fold="test")
        self.model._test_rmse = float(test_metrics["avg_rmse"])

        print(f"   ✅ 테스트 Avg RMSE: {test_metrics['avg_rmse']:.6f}  "
              f"| Avg IC: {test_metrics['avg_ic']:.4f}")

        return {
            "test_metrics":      test_metrics,
            "test_predictions":  test_predictions,
            "val_predictions":   None,
            "valid_metrics":     None,
            "final_model":       self.model,
            "target_cols":       self.model.target_columns,
            "target_type":       self.model.target_type,
            "embargo_gap_days":  embargo_gap,
            "integration_order": self.integration_order,
        }

    # ------------------------------------------------------------------
    # 2-Fold
    # ------------------------------------------------------------------

    def _run_2fold(self, df, splits, d, resume, fit_kwargs, embargo_gap):
        import copy

        ds_train = splits["ds_train"]
        ds_val   = splits["ds_val"]
        ds_test  = splits["ds_test"]

        from src.data_loader.seq_builder import SeqDataset
        all_trading_dates = pd.DatetimeIndex(
            sorted(pd.to_datetime(df[self.date_col].unique()))
        )
        ds_train2 = SeqDataset(
            df=df,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            seq_len=self.seq_len,
            forecast_horizon=self.forecast_horizon,
            stride=self.stride,
            date_filter=(all_trading_dates[0], pd.Timestamp(d["val_end"])),
        )

        self.model.fit(ds_train, eval_set=ds_val, resume=resume, **fit_kwargs)
        valid_metrics, val_predictions = self._evaluate_and_build(ds_val, df, fold="valid")
        print(f"   ✅ 검증 Avg RMSE: {valid_metrics['avg_rmse']:.6f} | IC: {valid_metrics['avg_ic']:.4f}")

        test_model = copy.deepcopy(self.model)
        test_model.is_fitted = False
        test_model.net = None
        test_model.fit(ds_train2, eval_set=ds_test, resume=False, **fit_kwargs)

        _prev = self.model
        self.model = test_model
        test_metrics, test_predictions = self._evaluate_and_build(ds_test, df, fold="test")
        self.model = _prev

        test_model._val_rmse  = float(valid_metrics["avg_rmse"])
        test_model._test_rmse = float(test_metrics["avg_rmse"])
        print(f"   ✅ 테스트 Avg RMSE: {test_metrics['avg_rmse']:.6f} | IC: {test_metrics['avg_ic']:.4f}")

        return {
            "valid_metrics":     valid_metrics,
            "val_predictions":   val_predictions,
            "test_metrics":      test_metrics,
            "test_predictions":  test_predictions,
            "final_model":       test_model,
            "target_cols":       self.model.target_columns,
            "target_type":       self.model.target_type,
            "embargo_gap_days":  embargo_gap,
            "integration_order": self.integration_order,
        }

    # ------------------------------------------------------------------
    # 평가 + predictions 생성 (공통)
    # ------------------------------------------------------------------

    def _evaluate_and_build(self, ds, df_orig, fold):
        self.model.net.eval()
        all_preds, all_trues = [], []

        loader = DataLoader(ds, batch_size=self.model.batch_size, shuffle=False)
        with torch.no_grad():
            for X_b, y_b in loader:
                pred = self.model.net(X_b.to(self.model.device)).cpu().numpy()
                all_preds.append(pred)
                all_trues.append(y_b.numpy())

        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_trues, axis=0)
        meta   = ds.get_meta()

        metrics = self._compute_metrics(y_pred, y_true, meta, df_orig)

        result = meta.copy().reset_index(drop=True)
        result["fold"]   = str(fold)
        result["ticker"] = result["ticker"].astype(str)

        base = ("target_log_return_1d" if self.model.target_type == "log_return_1d"
                else "target_log_close")
        for h in range(self.forecast_horizon):
            col = f"{base}_h{h+1}"
            result[f"pred_{col}"] = y_pred[:, h].copy()
            result[f"true_{col}"] = y_true[:, h].copy()

        return metrics, result

    def _compute_metrics(self, y_pred, y_true, meta, df_orig):
        if self.model.target_type == "log_return_1d":
            y_pred, y_true = self._to_log_close(y_pred, y_true, meta, df_orig)

        per_horizon = {}
        for h in range(self.forecast_horizon):
            p, t = y_pred[:, h], y_true[:, h]
            mask = ~np.isnan(p) & ~np.isnan(t)
            if mask.sum() < 2:
                per_horizon[h+1] = {"rmse": np.nan, "ic_mean": np.nan}
                continue
            rmse = float(np.sqrt(np.mean((p[mask] - t[mask]) ** 2)))
            ic_vals = []
            meta_r = meta.reset_index(drop=True)
            for _, grp in meta_r.groupby("date"):
                gi = grp.index.values
                if len(gi) < 3: continue
                gm = ~np.isnan(p[gi]) & ~np.isnan(t[gi])
                if gm.sum() >= 3:
                    ic, _ = stats.spearmanr(p[gi][gm], t[gi][gm])
                    if not np.isnan(ic): ic_vals.append(ic)
            per_horizon[h+1] = {"rmse": rmse,
                                 "ic_mean": float(np.mean(ic_vals)) if ic_vals else np.nan}

        rmses = [v["rmse"]    for v in per_horizon.values() if not np.isnan(v["rmse"])]
        ics   = [v["ic_mean"] for v in per_horizon.values() if not np.isnan(v["ic_mean"])]
        return {
            "avg_rmse": float(np.mean(rmses)) if rmses else np.nan,
            "avg_ic":   float(np.mean(ics))   if ics   else np.nan,
            "per_horizon": per_horizon,
            "samples":     len(y_pred),
        }

    def _to_log_close(self, y_pred, y_true, meta, df_orig):
        ref_cols = ["date", "ticker", "target_log_close", "target_log_return_1d"]
        if self.integration_order == 2:
            ref_cols.append("target_log_return_1d_lag1")

        ref = df_orig[ref_cols].copy()
        ref["date"] = pd.to_datetime(ref["date"])
        mc = meta.copy().reset_index(drop=True)
        mc["date"] = pd.to_datetime(mc["date"])
        merged = mc.merge(ref, on=["date", "ticker"], how="left")

        lc_base   = merged["target_log_close"].values
        delta_y_t = merged["target_log_return_1d"].values
        delta_y_t_minus_1 = (
            merged["target_log_return_1d_lag1"].values
            if self.integration_order == 2 else None
        )

        po = np.empty_like(y_pred)
        to = np.empty_like(y_true)

        for h in range(self.forecast_horizon):
            cp = np.sum(y_pred[:, :h+1], axis=1)
            ct = np.sum(y_true[:, :h+1], axis=1)

            # order=2: delta_y_h_minus_1
            if self.integration_order == 2:
                pred_dhm1 = y_pred[:, h-1] if h > 0 else delta_y_t
                true_dhm1 = y_true[:, h-1] if h > 0 else delta_y_t
            else:
                pred_dhm1 = None
                true_dhm1 = None

            po[:, h] = reconstruct_log_close(
                lc_base, cp, delta_y_t, y_pred[:, h],
                delta_y_t_minus_1=delta_y_t_minus_1,
                delta_y_h_minus_1=pred_dhm1,
                order=self.integration_order,
            )
            to[:, h] = reconstruct_log_close(
                lc_base, ct, delta_y_t, y_true[:, h],
                delta_y_t_minus_1=delta_y_t_minus_1,
                delta_y_h_minus_1=true_dhm1,
                order=self.integration_order,
            )

        return po, to
