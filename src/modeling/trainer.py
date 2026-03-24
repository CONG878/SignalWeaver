# src/modeling/trainer.py

from __future__ import annotations
import warnings
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import copy
import scipy.stats as stats
from src.models.base import ModelBase
from src.utils.trapezoidal import trapezoid_log_close

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
        base_price_col: str = "close"
    ):
        """
        Multi-horizon Walk-Forward 학습을 관리하는 Trainer 클래스.

        Parameters
        ----------
        target_col_name : str
            02단계 dataset의 기준 타겟 컬럼명 (예: 'target_log_close', 'target_log_return_1d')
        target_type : str
            'log_close'     : log(close(t+n)) 절대 가격 직접 예측 (기본값)
            'log_return_1d' : log1p(change_pct(t+n)) 1일 당일 등락률 예측 (v3.9.1 사다리꼴 보정 적용)
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
        2-Fold Walk-Forward 학습을 실행합니다. (Look-ahead 편향 방지를 위해 Embargo Gap 자동 적용)

        [검증 폴드] 훈련: [0 ~ E-G] / 검증: [E ~ E+V]
        [테스트 폴드] 훈련: [V ~ E+V-G] / 테스트: [E+V ~ E+V+T]
        * E: train_end, V/T: valid/test window, G: embargo_gap (max horizons)

        Returns
        -------
        Dict[str, Any]: metrics, predictions, final_model 등 평가 및 산출물 딕셔너리
        """
        fit_kwargs = fit_kwargs or {}

        # ──────────────────────────────────────────
        # 1. 타겟 컬럼 생성
        # ──────────────────────────────────────────
        df_run = df.copy().sort_values([self.date_col, 'ticker'])
        df_run[self.date_col] = pd.to_datetime(df_run[self.date_col])

        if self.target_col_name not in df_run.columns:
            raise KeyError(
                f"'{self.target_col_name}' 컬럼이 데이터에 없습니다. "
                f"02단계에서 해당 타겟 컬럼이 생성됐는지 확인하세요.\n"
                f"  log_close     모드: target_log_close\n"
                f"  log_return_1d 모드: target_log_return_1d"
            )

        target_cols = self._build_target_cols(df_run)

        # ──────────────────────────────────────────
        # 2. 날짜 인덱스 계산
        # ──────────────────────────────────────────
        all_dates = np.sort(df_run[self.date_col].unique())
        train_end_val = pd.to_datetime(train_end).to_datetime64()
        train_end_idx = np.searchsorted(all_dates, train_end_val)
        if train_end_idx < len(all_dates) and all_dates[train_end_idx] == train_end_val:
            train_end_idx += 1  # train_end 당일 포함

        G = max(self.horizons)  # embargo gap = max horizon (자동 계산)

        # 검증 폴드
        val_train_start_idx = 0
        val_train_end_idx   = train_end_idx - G
        val_eval_start_idx  = train_end_idx
        val_eval_end_idx    = train_end_idx + valid_window_days

        # 테스트 폴드 (훈련 구간을 valid_window_days만큼 롤링)
        test_train_start_idx = valid_window_days
        test_train_end_idx   = train_end_idx + valid_window_days - G
        test_eval_start_idx  = train_end_idx + valid_window_days
        test_eval_end_idx    = test_eval_start_idx + test_window_days

        if val_train_end_idx <= val_train_start_idx:
            raise ValueError(
                f"embargo_gap_days={G}가 너무 커서 검증 폴드 훈련 데이터가 없습니다."
            )
        if val_eval_end_idx > len(all_dates):
            raise ValueError(
                f"검증 폴드 데이터 부족: 필요 {val_eval_end_idx}일, "
                f"가용 {len(all_dates)}일"
            )
        if test_eval_end_idx > len(all_dates):
            raise ValueError(
                f"테스트 폴드 데이터 부족: 필요 {test_eval_end_idx}일, "
                f"가용 {len(all_dates)}일"
            )

        # ──────────────────────────────────────────
        # 3. 날짜 배열 추출 및 기간 출력
        # ──────────────────────────────────────────
        val_train_dates  = all_dates[val_train_start_idx : val_train_end_idx]
        val_eval_dates   = all_dates[val_eval_start_idx  : val_eval_end_idx]
        test_train_dates = all_dates[test_train_start_idx : test_train_end_idx]
        test_eval_dates  = all_dates[test_eval_start_idx  : test_eval_end_idx]

        print(f"🚀 Walk-Forward Training (2-Fold)")
        print(f"   Target  : {self.target_col_name}  |  "
              f"Type: {self.target_type}  |  Horizons: {self.horizons}")
        print(f"   Embargo : {G} 거래일 (train_end - {G}d → train_end 구간 제거)")
        print(f"\n   [검증 폴드]")
        print(f"   훈련: {pd.Timestamp(val_train_dates[0]).date()} ~ "
              f"{pd.Timestamp(val_train_dates[-1]).date()} "
              f"({len(val_train_dates)} 거래일)  ← embargo {G}일 제거됨")
        print(f"   검증: {pd.Timestamp(val_eval_dates[0]).date()} ~ "
              f"{pd.Timestamp(val_eval_dates[-1]).date()} "
              f"({len(val_eval_dates)} 거래일)")
        print(f"\n   [테스트 폴드]")
        print(f"   훈련: {pd.Timestamp(test_train_dates[0]).date()} ~ "
              f"{pd.Timestamp(test_train_dates[-1]).date()} "
              f"({len(test_train_dates)} 거래일)  ← embargo {G}일 제거됨")
        print(f"   테스트: {pd.Timestamp(test_eval_dates[0]).date()} ~ "
              f"{pd.Timestamp(test_eval_dates[-1]).date()} "
              f"({len(test_eval_dates)} 거래일)")

        # ──────────────────────────────────────────
        # 4. 공통 전처리: 필요한 컬럼만 추출
        # ──────────────────────────────────────────
        # log_return_1d 모드: 보고 지표(RMSE/IC)를 로그 종가 스케일로 통일하기 위해
        #   - target_log_close : 기준값 y(t)
        #   - target_col_name  : Δy(t) 앵커 (사다리꼴 보정용, ✨ v3.9.1)
        _ref_col = 'target_log_close'
        _extra   = []
        if self.target_type == "log_return_1d":
            if _ref_col in df_run.columns:
                _extra.append(_ref_col)
            # ✨ v3.9.1: Δy(t) 앵커 컬럼 추가 (사다리꼴 보정)
            if self.target_col_name in df_run.columns:
                _extra.append(self.target_col_name)

        cols_needed = [self.date_col, 'ticker'] + self.feature_cols + target_cols + _extra
        temp_df = df_run[cols_needed].copy()

        # ══════════════════════════════════════════
        # FOLD 1: 검증 폴드
        # ══════════════════════════════════════════
        print(f"\n[검증 폴드] 학습 중...")

        val_train_df = temp_df[temp_df[self.date_col].isin(val_train_dates)].dropna()
        val_eval_df  = temp_df[temp_df[self.date_col].isin(val_eval_dates)].dropna()

        if val_train_df.empty:
            raise ValueError("검증 폴드: 훈련 데이터가 없습니다.")
        if val_eval_df.empty:
            raise ValueError("검증 폴드: 검증 데이터가 없습니다.")

        self.model.fit(
            X=val_train_df[self.feature_cols],
            y=val_train_df[target_cols],
            eval_set=[(val_eval_df[self.feature_cols], val_eval_df[target_cols])],
            **fit_kwargs
        )

        valid_metrics = self._evaluate(val_eval_df, target_cols)
        full_val_slice = temp_df[temp_df[self.date_col].isin(val_eval_dates)]
        val_predictions = self._predict_with_metadata(full_val_slice, target_cols, fold='valid')

        print(f"   ✅ 검증 RMS RMSE: {valid_metrics['rms_rmse']:.6f}  "
              f"| Avg RMSE: {valid_metrics['avg_rmse']:.6f}  "
              f"| Avg IC: {valid_metrics['avg_ic']:.4f}  "
              f"(samples: {valid_metrics['samples']:,})")

        # ══════════════════════════════════════════
        # FOLD 2: 테스트 폴드
        # ══════════════════════════════════════════
        print(f"\n[테스트 폴드] 학습 중...")

        test_model = copy.deepcopy(self.model)
        test_model.is_fitted = False

        test_train_df = temp_df[temp_df[self.date_col].isin(test_train_dates)].dropna()
        test_eval_df  = temp_df[temp_df[self.date_col].isin(test_eval_dates)].dropna()

        if test_train_df.empty:
            raise ValueError("테스트 폴드: 훈련 데이터가 없습니다.")
        if test_eval_df.empty:
            raise ValueError("테스트 폴드: 테스트 데이터가 없습니다.")

        test_model.fit(
            X=test_train_df[self.feature_cols],
            y=test_train_df[target_cols],
            eval_set=[(test_eval_df[self.feature_cols], test_eval_df[target_cols])],
            **fit_kwargs
        )

        _prev_model = self.model
        self.model = test_model
        test_metrics = self._evaluate(test_eval_df, target_cols)
        full_test_slice = temp_df[temp_df[self.date_col].isin(test_eval_dates)]
        test_predictions = self._predict_with_metadata(full_test_slice, target_cols, fold='test')
        self.model = _prev_model

        print(f"   ✅ 테스트 RMS RMSE: {test_metrics['rms_rmse']:.6f}  "
              f"| Avg RMSE: {test_metrics['avg_rmse']:.6f}  "
              f"| Avg IC: {test_metrics['avg_ic']:.4f}  "
              f"(samples: {test_metrics['samples']:,})")

        return {
            'valid_metrics'    : valid_metrics,
            'val_predictions'  : val_predictions,
            'test_metrics'     : test_metrics,
            'test_predictions' : test_predictions,
            'final_model'      : test_model,
            'target_cols'      : target_cols,
            'target_type'      : self.target_type,
            'embargo_gap_days' : G,
        }

    # ──────────────────────────────────────────────────────────────
    # 타겟 생성
    # ──────────────────────────────────────────────────────────────

    def _build_target_cols(self, df_run: pd.DataFrame) -> List[str]:
        """
        target_type에 따라 horizon별 타겟 컬럼을 df_run에 인플레이스로 추가하고
        컬럼명 리스트를 반환.

        log_close 모드 (기본값):
            target_log_close_h{n}(t) = log(close(t+n))

        log_return_1d 모드 (v3.9.0 신규):
            target_log_return_1d_h{n}(t) = log1p(change_pct(t+n))
            = t+n 시점의 당일 등락률 로그값 (shift(-n) 적용)
            원본 컬럼: 02단계에서 np.log1p(df['change_pct'])로 생성된 target_log_return_1d
        """
        target_cols = []

        if self.target_type == "log_return_1d":
            for h in self.horizons:
                col_name = f"target_log_return_1d_h{h}"
                df_run[col_name] = df_run.groupby('ticker')[self.target_col_name].shift(-h)
                target_cols.append(col_name)

        else:  # log_close (기본값, 권장)
            for h in self.horizons:
                col_name = f"{self.target_col_name}_h{h}"
                df_run[col_name] = df_run.groupby('ticker')[self.target_col_name].shift(-h)
                target_cols.append(col_name)

        return target_cols

    # ──────────────────────────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────────────────────────

    def _evaluate(self, eval_df: pd.DataFrame, target_cols: List[str]) -> Dict[str, float]:
        """Horizon별 RMSE, IC, ICIR 및 평균 지표를 계산합니다."""
        if eval_df.empty:
            return {'rms_rmse': np.nan, 'avg_rmse': np.nan, 'avg_ic': np.nan, 'per_horizon': {}, 'samples': 0}

        preds_df = self.model.predict(eval_df[self.feature_cols])
        if isinstance(preds_df, np.ndarray):
            preds_df = pd.DataFrame(preds_df, index=eval_df.index, columns=target_cols)

        temp_eval = eval_df[[self.date_col]].copy()
        for col in target_cols:
            temp_eval[f'pred_{col}'] = preds_df[col].values
            temp_eval[f'true_{col}'] = eval_df[col].values

        # ── log_return_1d: 사다리꼴 보정을 통한 로그 종가 스케일 환산 (v3.9.1) ──
        if self.target_type == "log_return_1d" and 'target_log_close' in eval_df.columns:
            log_close_base = eval_df['target_log_close'].values
            sorted_cols    = sorted(target_cols, key=lambda c: int(c.split('_h')[-1]))
            delta_y_t      = eval_df[self.target_col_name].values  # 사다리꼴 앵커

            for idx, col in enumerate(sorted_cols):
                pred_delta_h = preds_df[col].values
                true_delta_h = eval_df[col].values

                cum_pred = sum(preds_df[c].values for c in sorted_cols[:idx + 1])
                cum_true = sum(eval_df[c].values  for c in sorted_cols[:idx + 1])

                temp_eval[f'pred_{col}'] = trapezoid_log_close(
                    log_close_base, cum_pred, delta_y_t, pred_delta_h
                )
                temp_eval[f'true_{col}'] = trapezoid_log_close(
                    log_close_base, cum_true, delta_y_t, true_delta_h
                )
        # ──────────────────────────────────────────────────────────────────

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
        """메타데이터 포함 예측 결과 생성"""
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
