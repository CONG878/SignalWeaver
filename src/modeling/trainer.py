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
        target_type: str = "log_return",
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
            log_return 모드에서는 이 컬럼이 log_close 값을 담고 있어야 함.
            (기본값: "target_log_close")
        target_type : str
            "log_return" : target_log_return_h{n}(t) = log_close(t+n) - log_close(t)
                           오늘 대비 n일 후의 누적 로그 수익률. 정상(stationary) 시계열.
            "log_close"  : target_log_close_h{n}(t) = log_close(t+n)
                           레거시 모드. 비정상 시계열이므로 권장하지 않음.
        """
        if target_type not in ("log_return", "log_close"):
            raise ValueError(
                f"target_type은 'log_return' 또는 'log_close'이어야 합니다. "
                f"받은 값: '{target_type}'"
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

        구조:
            [검증 폴드] 훈련: [0, E],    검증: [E, E+V]
                        → early stopping 기준 + val_predictions 저장 (앙상블 가중치용)
            [테스트 폴드] 훈련: [V, E+V], 테스트: [E+V, E+V+T]
                        → 최종 성능 평가 + test_predictions 저장 + final_model

        반환
        ----
        dict with keys:
            'valid_metrics'     : 검증 폴드 지표 (avg_rmse, per_horizon, samples)
            'val_predictions'   : 검증 폴드 예측값 DataFrame (앙상블 가중치 최적화용)
            'test_metrics'      : 테스트 폴드 지표
            'test_predictions'  : 테스트 폴드 예측값 DataFrame (최종 평가 전용)
            'final_model'       : 테스트 폴드 학습 모델
            'target_cols'       : 생성된 타겟 컬럼명 리스트 (04단계 역산에 필요)
            'target_type'       : 사용된 타겟 타입
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

        # 검증 폴드
        val_train_start_idx = 0
        val_train_end_idx   = train_end_idx
        val_eval_end_idx    = train_end_idx + valid_window_days

        # 테스트 폴드 (훈련 구간을 valid_window_days만큼 롤링)
        test_train_start_idx = valid_window_days
        test_train_end_idx   = train_end_idx + valid_window_days
        test_eval_end_idx    = test_train_end_idx + test_window_days

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
        val_eval_dates   = all_dates[val_train_end_idx   : val_eval_end_idx]
        test_train_dates = all_dates[test_train_start_idx : test_train_end_idx]
        test_eval_dates  = all_dates[test_train_end_idx   : test_eval_end_idx]

        print(f"🚀 Walk-Forward Training (2-Fold)")
        print(f"   Target  : {self.target_col_name}  |  "
              f"Type: {self.target_type}  |  Horizons: {self.horizons}")
        print(f"\n   [검증 폴드]")
        print(f"   훈련: {pd.Timestamp(val_train_dates[0]).date()} ~ "
              f"{pd.Timestamp(val_train_dates[-1]).date()} "
              f"({len(val_train_dates)} 거래일)")
        print(f"   검증: {pd.Timestamp(val_eval_dates[0]).date()} ~ "
              f"{pd.Timestamp(val_eval_dates[-1]).date()} "
              f"({len(val_eval_dates)} 거래일)")
        print(f"\n   [테스트 폴드]")
        print(f"   훈련: {pd.Timestamp(test_train_dates[0]).date()} ~ "
              f"{pd.Timestamp(test_train_dates[-1]).date()} "
              f"({len(test_train_dates)} 거래일)")
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
              f"(samples: {test_metrics['samples']:,})")

        return {
            'valid_metrics'    : valid_metrics,
            'val_predictions'  : val_predictions,
            'test_metrics'     : test_metrics,
            'test_predictions' : test_predictions,
            'final_model'      : test_model,
            'target_cols'      : target_cols,   # 04단계 역산에서 컬럼명 참조용
            'target_type'      : self.target_type,
        }

    # ──────────────────────────────────────────────────────────────
    # 타겟 생성
    # ──────────────────────────────────────────────────────────────

    def _build_target_cols(self, df_run: pd.DataFrame) -> List[str]:
        """
        target_type에 따라 horizon별 타겟 컬럼을 df_run에 인플레이스로 추가하고
        컬럼명 리스트를 반환.

        log_return 모드:
            prefix = "target_log_return"  (target_col_name의 "log_close" → "log_return")
            target_log_return_h{n}(t) = log_close(t+n) - log_close(t)

        log_close 모드 (레거시):
            target_log_close_h{n}(t) = log_close(t+n)
        """
        target_cols = []

        if self.target_type == "log_return":
            # "target_log_close" → "target_log_return"
            prefix = self.target_col_name.replace("log_close", "log_return")

            for h in self.horizons:
                col_name = f"{prefix}_h{h}"
                future_log_close = df_run.groupby('ticker')[self.target_col_name].shift(-h)
                # 현재 시점의 log_close는 shift(0) — transform 없이 직접 사용
                df_run[col_name] = future_log_close - df_run[self.target_col_name]
                target_cols.append(col_name)

        else:  # log_close (레거시)
            for h in self.horizons:
                col_name = f"{self.target_col_name}_h{h}"
                df_run[col_name] = df_run.groupby('ticker')[self.target_col_name].shift(-h)
                target_cols.append(col_name)

        return target_cols

    # ──────────────────────────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────────────────────────

    def _evaluate(self, eval_df: pd.DataFrame, target_cols: List[str]) -> Dict[str, float]:
        """Horizon별 RMSE 및 평균 RMSE 계산"""
        if eval_df.empty:
            return {'avg_rmse': np.nan, 'per_horizon': {}, 'samples': 0}

        preds_df = self.model.predict(eval_df[self.feature_cols])
        if isinstance(preds_df, np.ndarray):
            preds_df = pd.DataFrame(preds_df, index=eval_df.index, columns=target_cols)

        per_horizon = {}
        for col in target_cols:
            y_true = eval_df[col].values
            y_pred = preds_df[col].values
            mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
            rmse = np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)) if mask.any() else np.nan
            per_horizon[col] = rmse

        valid_rmses = [v for v in per_horizon.values() if not np.isnan(v)]
        avg_rmse = float(np.mean(valid_rmses)) if valid_rmses else np.nan

        return {
            'avg_rmse'    : avg_rmse,
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
        preds = preds.reset_index(drop=True)

        for col in target_cols:
            result[f'pred_{col}'] = preds[col].values
            result[f'true_{col}'] = eval_df[col].reset_index(drop=True).values

        result['fold'] = fold
        return result
