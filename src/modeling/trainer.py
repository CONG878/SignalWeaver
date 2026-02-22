# src/modeling/trainer.py
# v3.7.0 변경사항:
#   1. target_type="log_return" → DeprecationWarning 추가, 기본값 "log_close"로 롤백
#   2. Embargo gap 도입 (run 메서드)
#      → G = max(self.horizons) 자동 계산 (별도 파라미터 없음)
#      → val/test 훈련 샘플 끝을 G만큼 앞당기고, eval 윈도우는 그대로 유지

from __future__ import annotations
import warnings
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from src.models.base import ModelBase
import copy
import scipy.stats as stats


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
        파라미터
        -------
        target_col_name : str
            02단계 dataset의 기준 타겟 컬럼명.
            (기본값: "target_log_close")
        target_type : str
            "log_close"  : target_log_close_h{n}(t) = log_close(t+n)          [기본값, 권장]
            "log_return" : target_log_return_h{n}(t) = log_close(t+n) - log_close(t)
                           ⚠️ DEPRECATED (v3.7.0): 피처 확장 후 log_close 대비 성능 열위.
                           원인 분석 및 개선 완료 전까지 log_close를 사용하세요.
        """
        if target_type not in ("log_return", "log_close"):
            raise ValueError(
                f"target_type은 'log_return' 또는 'log_close'이어야 합니다. "
                f"받은 값: '{target_type}'"
            )

        # ⚠️ v3.7.0: log_return deprecated
        if target_type == "log_return":
            warnings.warn(
                "[DEPRECATED v3.7.0] target_type='log_return'은 deprecated 상태입니다. "
                "피처 확장 이후 log_close 대비 성능이 열위로 확인되어 롤백되었습니다. "
                "원인 분석 및 개선 완료 전까지 target_type='log_close'를 사용하세요. "
                "재도입 여부는 향후 버전에서 결정됩니다.",
                DeprecationWarning,
                stacklevel=2
            )

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
        2-Fold Walk-Forward 학습 실행

        Embargo gap = max(self.horizons) 자동 계산.
        horizons 변경 시 gap이 자동으로 연동되므로 별도 설정 불필요.

        구조 (G = max(horizons)):
            [검증 폴드]
              실제 훈련: [0,     E-G]    ← embargo gap만큼 끝을 앞당김
              embargo:   [E-G,   E]      ← 검증 타겟과 날짜 겹침 구간 제거
              검증:      [E,     E+V]    ← 윈도우 크기 변화 없음

            [테스트 폴드]
              실제 훈련: [V,     E+V-G]  ← 동일 구조
              embargo:   [E+V-G, E+V]
              테스트:    [E+V,   E+V+T]  ← 윈도우 크기 변화 없음

        E = train_end, V = valid_window_days, T = test_window_days, G = max(horizons)

        ※ 훈련 샘플 손실(G일)은 01단계에서 수집 기간을 앞당겨 보완합니다.

        반환
        ----
        dict with keys:
            'valid_metrics'     : 검증 폴드 지표 (avg_rmse, avg_ic, per_horizon, samples)
            'val_predictions'   : 검증 폴드 예측값 DataFrame (앙상블 가중치 최적화용)
            'test_metrics'      : 테스트 폴드 지표
            'test_predictions'  : 테스트 폴드 예측값 DataFrame (최종 평가 전용)
            'final_model'       : 테스트 폴드 학습 모델
            'target_cols'       : 생성된 타겟 컬럼명 리스트 (04단계 역산에 필요)
            'target_type'       : 사용된 타겟 타입
            'embargo_gap_days'  : 적용된 embargo gap (= max(horizons), 거래일)
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
                f"02단계에서 target_log_close 컬럼이 생성됐는지 확인하세요."
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
        val_train_end_idx   = train_end_idx - G      # ← 실제 학습 샘플 끝 (embargo 적용)
        val_eval_start_idx  = train_end_idx           # ← 검증 시작 (embargo gap 이후)
        val_eval_end_idx    = train_end_idx + valid_window_days

        # 테스트 폴드 (훈련 구간을 valid_window_days만큼 롤링)
        test_train_start_idx = valid_window_days
        test_train_end_idx   = train_end_idx + valid_window_days - G  # ← embargo 적용
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
        val_eval_dates   = all_dates[val_eval_start_idx  : val_eval_end_idx]   # ← 분리됨
        test_train_dates = all_dates[test_train_start_idx : test_train_end_idx]
        test_eval_dates  = all_dates[test_eval_start_idx  : test_eval_end_idx]  # ← 분리됨

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
        cols_needed = [self.date_col, 'ticker'] + self.feature_cols + target_cols
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

        print(f"   ✅ 검증 Avg RMSE: {valid_metrics['avg_rmse']:.6f}  "
              f"| Avg IC: {valid_metrics['avg_ic']:.4f}  "
              f"(samples: {valid_metrics['samples']:,})")

        # ══════════════════════════════════════════
        # FOLD 2: 테스트 폴드
        # ══════════════════════════════════════════
        print(f"\n[테스트 폴드] 학습 중...")

        # 검증 폴드 가중치와 독립된 새 모델로 재학습
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

        # test_model로 평가 수행 (self.model 임시 교체)
        _prev_model = self.model
        self.model = test_model
        test_metrics = self._evaluate(test_eval_df, target_cols)
        full_test_slice = temp_df[temp_df[self.date_col].isin(test_eval_dates)]
        test_predictions = self._predict_with_metadata(full_test_slice, target_cols, fold='test')
        self.model = _prev_model  # 원복

        print(f"   ✅ 테스트 Avg RMSE: {test_metrics['avg_rmse']:.6f}  "
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

        log_close 모드 (기본값, v3.7.0~):
            target_log_close_h{n}(t) = log_close(t+n)

        log_return 모드 (DEPRECATED v3.7.0):
            prefix = "target_log_return"
            target_log_return_h{n}(t) = log_close(t+n) - log_close(t)
        """
        target_cols = []

        if self.target_type == "log_return":
            # "target_log_close" → "target_log_return"
            prefix = self.target_col_name.replace("log_close", "log_return")

            for h in self.horizons:
                col_name = f"{prefix}_h{h}"
                future_log_close = df_run.groupby('ticker')[self.target_col_name].shift(-h)
                df_run[col_name] = future_log_close - df_run[self.target_col_name]
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
        """Horizon별 RMSE, IC, ICIR 및 평균 지표 계산"""
        if eval_df.empty:
            return {'avg_rmse': np.nan, 'avg_ic': np.nan, 'per_horizon': {}, 'samples': 0}

        preds_df = self.model.predict(eval_df[self.feature_cols])
        if isinstance(preds_df, np.ndarray):
            preds_df = pd.DataFrame(preds_df, index=eval_df.index, columns=target_cols)

        temp_eval = eval_df[[self.date_col]].copy()
        for col in target_cols:
            temp_eval[f'pred_{col}'] = preds_df[col].values
            temp_eval[f'true_{col}'] = eval_df[col].values

        per_horizon = {}
        for col in target_cols:
            # 1. RMSE
            y_true = temp_eval[f'true_{col}'].values
            y_pred = temp_eval[f'pred_{col}'].values
            mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
            rmse = np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)) if mask.any() else np.nan

            # 2. Daily Cross-Sectional IC
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

            per_horizon[col] = {
                'rmse'   : rmse,
                'ic_mean': ic_mean,
                'icir'   : icir
            }

        valid_rmses = [v['rmse']    for v in per_horizon.values() if not np.isnan(v['rmse'])]
        valid_ics   = [v['ic_mean'] for v in per_horizon.values() if not np.isnan(v['ic_mean'])]

        return {
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
