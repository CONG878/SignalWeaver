"""
SeqTrainer — Seq 모델 전용 학습 관리자

## v4.0.0 rev2 변경
- SeqDataset(on-the-fly) 사용 → 전체 텐서 미리 적재 없음
- split_by_date()로 SeqDataset을 train/val/test로 직접 분할
- GRUModel.fit()에 resume 파라미터 전달 지원
- n_folds: 1(기본) / 2(앙상블 대비, v4.1.0)
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
from src.utils.trapezoidal import trapezoid_log_close


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
    ):
        self.model            = model
        self.feature_cols     = feature_cols
        self.target_col       = target_col
        self.seq_len          = seq_len
        self.forecast_horizon = forecast_horizon
        self.stride           = stride
        self.date_col         = date_col

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
        """
        Walk-Forward 학습 실행.

        Parameters
        ----------
        df : pd.DataFrame
        train_end : str
        valid_window_days : int
        test_window_days : int
        n_folds : int, default=1
            1: 1-Fold (권장). val_predictions 미생성.
            2: 2-Fold. val_predictions 생성 (v4.1.0 앙상블 대비).
        resume : bool, default=False
            True: checkpoint.pt에서 이어서 학습.
            False: 처음부터 학습.
        fit_kwargs : dict, optional
        """
        fit_kwargs  = fit_kwargs or {}
        embargo_gap = self.forecast_horizon

        # ── 날짜 분할 및 SeqDataset 생성 ────────────────────────
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
        print(f"   Target  : {self.target_col}  |  Horizon: {self.forecast_horizon}  |  seq_len: {self.seq_len}")
        print(f"   Embargo : {embargo_gap} 거래일")
        print(f"   resume  : {resume}")

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

        print(f"\n[학습 중...]")
        self.model.fit(
            ds_train, eval_set=ds_val,
            resume=resume, **fit_kwargs
        )

        test_metrics, test_predictions = self._evaluate_and_build(ds_test, df, fold="test")

        self.model._test_rmse = float(test_metrics["avg_rmse"])

        print(
            f"   ✅ 테스트 Avg RMSE: {test_metrics['avg_rmse']:.6f}  "
            f"| Avg IC: {test_metrics['avg_ic']:.4f}  "
            f"(samples: {test_metrics['samples']:,})"
        )

        return {
            "test_metrics":     test_metrics,
            "test_predictions": test_predictions,
            "val_predictions":  None,
            "valid_metrics":    None,
            "final_model":      self.model,
            "target_cols":      self.model.target_columns,
            "target_type":      self.model.target_type,
            "embargo_gap_days": embargo_gap,
        }

    # ------------------------------------------------------------------
    # 2-Fold
    # ------------------------------------------------------------------

    def _run_2fold(self, df, splits, d, resume, fit_kwargs, embargo_gap):
        import copy

        ds_train = splits["ds_train"]
        ds_val   = splits["ds_val"]
        ds_test  = splits["ds_test"]

        # Fold 2용 훈련 데이터: val 기간까지 포함 (롤링)
        # split_by_date에서 ds_train은 embargo_end까지만이므로
        # Fold 2는 val_end까지를 훈련으로 사용
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

        print(f"\n   [검증 폴드]")
        print(f"   훈련: ~ {d['train_end']}  검증: {d['val_start']} ~ {d['val_end']}")
        print(f"\n[검증 폴드] 학습 중...")
        self.model.fit(ds_train, eval_set=ds_val, resume=resume, **fit_kwargs)

        valid_metrics, val_predictions = self._evaluate_and_build(ds_val, df, fold="valid")
        print(f"   ✅ 검증 Avg RMSE: {valid_metrics['avg_rmse']:.6f} | IC: {valid_metrics['avg_ic']:.4f}")

        print(f"\n   [테스트 폴드]")
        print(f"   훈련: ~ {d['val_end']}  테스트: {d['test_start']} ~ {d['test_end']}")
        test_model = copy.deepcopy(self.model)
        test_model.is_fitted = False
        test_model.net = None
        print(f"\n[테스트 폴드] 학습 중...")
        test_model.fit(ds_train2, eval_set=ds_test, resume=False, **fit_kwargs)

        _prev = self.model
        self.model = test_model
        test_metrics, test_predictions = self._evaluate_and_build(ds_test, df, fold="test")
        self.model = _prev

        test_model._val_rmse  = float(valid_metrics["avg_rmse"])
        test_model._test_rmse = float(test_metrics["avg_rmse"])
        print(f"   ✅ 테스트 Avg RMSE: {test_metrics['avg_rmse']:.6f} | IC: {test_metrics['avg_ic']:.4f}")

        return {
            "valid_metrics":    valid_metrics,
            "val_predictions":  val_predictions,
            "test_metrics":     test_metrics,
            "test_predictions": test_predictions,
            "final_model":      test_model,
            "target_cols":      self.model.target_columns,
            "target_type":      self.model.target_type,
            "embargo_gap_days": embargo_gap,
        }

    # ------------------------------------------------------------------
    # 평가 + predictions 생성 (공통)
    # ------------------------------------------------------------------

    def _evaluate_and_build(
        self,
        ds: SeqDataset,
        df_orig: pd.DataFrame,
        fold: str,
    ) -> tuple:
        """DataLoader로 예측 → 평가지표 + predictions DataFrame 반환."""
        self.model.net.eval()
        all_preds, all_trues = [], []

        loader = DataLoader(ds, batch_size=self.model.batch_size, shuffle=False)
        with torch.no_grad():
            for X_b, y_b in loader:
                pred = self.model.net(X_b.to(self.model.device)).cpu().numpy()
                all_preds.append(pred)
                all_trues.append(y_b.numpy())

        y_pred = np.concatenate(all_preds, axis=0)  # (N, forecast_horizon)
        y_true = np.concatenate(all_trues, axis=0)
        meta   = ds.get_meta()

        # 평가
        metrics = self._compute_metrics(y_pred, y_true, meta, df_orig)

        # predictions DataFrame
        result = meta.copy().reset_index(drop=True)
        result["fold"] = str(fold)  # object 타입을 명확한 문자열로 보장
        result["ticker"] = result["ticker"].astype(str) # 혹시 모를 타입 꼬임 방지
        
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
            "avg_rmse":    float(np.mean(rmses)) if rmses else np.nan,
            "avg_ic":      float(np.mean(ics))   if ics   else np.nan,
            "per_horizon": per_horizon,
            "samples":     len(y_pred),
        }

    def _to_log_close(self, y_pred, y_true, meta, df_orig):
        ref = df_orig[["date", "ticker", "target_log_close", "target_log_return_1d"]].copy()
        ref["date"] = pd.to_datetime(ref["date"])
        mc = meta.copy().reset_index(drop=True)
        mc["date"] = pd.to_datetime(mc["date"])
        merged = mc.merge(ref, on=["date", "ticker"], how="left")
        lc_base  = merged["target_log_close"].values
        delta_yt = merged["target_log_return_1d"].values
        po, to = np.empty_like(y_pred), np.empty_like(y_true)
        for h in range(self.forecast_horizon):
            cp = np.sum(y_pred[:, :h+1], axis=1)
            ct = np.sum(y_true[:, :h+1], axis=1)
            po[:, h] = trapezoid_log_close(lc_base, cp, delta_yt, y_pred[:, h])
            to[:, h] = trapezoid_log_close(lc_base, ct, delta_yt, y_true[:, h])
        return po, to
