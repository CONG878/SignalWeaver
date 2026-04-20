"""
Universe Selection Module - Facade Pattern

모델의 예측 결과(정확도, 수익성, 위험도)를 종합하여 최적의 투자 후보군을 선정합니다.
복잡한 평가 흐름을 `select_investment_universe` 단일 인터페이스(Facade)로 제공합니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from tqdm import tqdm
from scipy.stats import spearmanr

from src.utils.trading import find_best_trade_vectorized
from src.utils.risk import (
    calculate_risk_metrics,
    calculate_composite_risk_score,
    normalize_risk_scores,
)
from src.utils.integration import reconstruct_log_close
from src.universe.filters import apply_hard_filters


# ==========================================
# 1. 정확도 평가 (Accuracy Evaluation)
# ==========================================

def evaluate_model_accuracy(
    df_past_predictions: pd.DataFrame,
    model_date: str,
    target_columns: Optional[List[str]] = None,
    log_close_ref: Optional[pd.DataFrame] = None,
    integration_order: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    과거 예측 데이터로부터 모델 정확도를 평가합니다.

    Parameters
    ----------
    df_past_predictions : pd.DataFrame
        과거 예측 결과
    model_date : str
        모델 학습 기준일 (이전 데이터만 평가에 사용)
    target_columns : List[str], optional
        평가할 타겟 컬럼 리스트
    log_close_ref : pd.DataFrame, optional
        log_return_1d 모드 전용. 로그 종가 역산을 위한 앵커 데이터프레임.
        필수 컬럼: ticker, date, target_log_close, target_log_return_1d
        integration_order=2 시 추가 필수: target_log_return_1d_lag1
    integration_order : int, default=1
        log-close 역산 수치 적분 차수.
        0 — 직사각형 / 1 — 사다리꼴 / 2 — Adams-Moulton 2-step
    verbose : bool
        진행 상황 출력 여부
    """
    if target_columns is None:
        target_columns = [f'target_log_close_h{h}' for h in range(1, 6)]

    sorted_cols = sorted(target_columns, key=lambda c: int(c.split('_h')[-1]))

    model_date_dt = pd.to_datetime(model_date)
    df_eval = df_past_predictions[
        df_past_predictions['date'] <= model_date_dt
    ].copy()

    if len(df_eval) == 0:
        raise ValueError(
            f"❌ {model_date} 이전의 예측 데이터가 없습니다.\n"
            f"   predictions.parquet의 날짜 범위를 확인하세요."
        )

    use_conversion = (log_close_ref is not None)
    if use_conversion:
        ref_cols = ['ticker', 'date', 'target_log_close', 'target_log_return_1d']
        if integration_order == 2:
            lag1_col = 'target_log_return_1d_lag1'
            if lag1_col not in log_close_ref.columns:
                raise KeyError(
                    f"integration_order=2에는 log_close_ref에 '{lag1_col}' 컬럼이 필요합니다.\n"
                    "02단계(02_build_dataset.ipynb)를 재실행하고 "
                    "05단계에서 log_close_ref에 해당 컬럼을 포함하세요."
                )
            ref_cols.append(lag1_col)
        ref = log_close_ref[[c for c in ref_cols if c in log_close_ref.columns]].copy()
        ref['date'] = pd.to_datetime(ref['date'])
        df_eval = df_eval.merge(ref, on=['ticker', 'date'], how='left')

    accuracy_metrics = []
    tickers  = df_eval['ticker'].unique()
    iterator = tqdm(tickers, desc="정확도 평가") if verbose else tickers

    for ticker in iterator:
        ticker_eval = df_eval[df_eval['ticker'] == ticker].copy()

        rmse_list, ic_list = [], []

        for idx, col in enumerate(sorted_cols):
            pred_col = f'pred_{col}'
            true_col = f'true_{col}'
            if pred_col not in ticker_eval.columns or true_col not in ticker_eval.columns:
                continue

            if use_conversion and 'target_log_close' in ticker_eval.columns:
                log_close_base = ticker_eval['target_log_close'].values
                delta_y_t      = ticker_eval['target_log_return_1d'].values \
                                 if 'target_log_return_1d' in ticker_eval.columns \
                                 else np.zeros(len(ticker_eval))

                # order=2 앵커
                delta_y_t_minus_1 = (
                    ticker_eval['target_log_return_1d_lag1'].values
                    if integration_order == 2
                    and 'target_log_return_1d_lag1' in ticker_eval.columns
                    else None
                )

                pred_delta_h = ticker_eval[pred_col].values
                true_delta_h = ticker_eval[true_col].values

                cum_pred = sum(
                    ticker_eval[f'pred_{c}'].values for c in sorted_cols[:idx + 1]
                    if f'pred_{c}' in ticker_eval.columns
                )
                cum_true = sum(
                    ticker_eval[f'true_{c}'].values for c in sorted_cols[:idx + 1]
                    if f'true_{c}' in ticker_eval.columns
                )

                # order=2: delta_y_h_minus_1
                if integration_order == 2:
                    pred_col_prev = f'pred_{sorted_cols[idx - 1]}' if idx > 0 else None
                    true_col_prev = f'true_{sorted_cols[idx - 1]}' if idx > 0 else None
                    pred_dhm1 = (
                        ticker_eval[pred_col_prev].values if pred_col_prev
                        else delta_y_t
                    )
                    true_dhm1 = (
                        ticker_eval[true_col_prev].values if true_col_prev
                        else delta_y_t
                    )
                else:
                    pred_dhm1 = None
                    true_dhm1 = None

                pred_vals = reconstruct_log_close(
                    log_close_base, cum_pred, delta_y_t, pred_delta_h,
                    delta_y_t_minus_1=delta_y_t_minus_1,
                    delta_y_h_minus_1=pred_dhm1,
                    order=integration_order,
                )
                true_vals = reconstruct_log_close(
                    log_close_base, cum_true, delta_y_t, true_delta_h,
                    delta_y_t_minus_1=delta_y_t_minus_1,
                    delta_y_h_minus_1=true_dhm1,
                    order=integration_order,
                )

            else:
                pred_vals = ticker_eval[pred_col].values
                true_vals = ticker_eval[true_col].values

            mask = ~np.isnan(pred_vals) & ~np.isnan(true_vals)
            if mask.sum() < 2:
                continue

            pred = pred_vals[mask]
            true = true_vals[mask]
            rmse_list.append(np.sqrt(np.mean((pred - true) ** 2)))

            if mask.sum() >= 3:
                ic, _ = spearmanr(pred, true)
                ic_list.append(ic if not np.isnan(ic) else 0.0)

        if not rmse_list:
            continue

        rmse    = float(np.mean(rmse_list))
        ic_mean = float(np.mean(ic_list)) if ic_list else 0.0

        # ✨ v3.10.0: Directional Accuracy
        dir_matches = []
        for col in sorted_cols:
            pred_col = f'pred_{col}'
            true_col = f'true_{col}'
            if pred_col not in ticker_eval.columns or true_col not in ticker_eval.columns:
                continue
            p = ticker_eval[pred_col].values
            t = ticker_eval[true_col].values
            valid = ~np.isnan(p) & ~np.isnan(t)
            p_v, t_v = p[valid], t[valid]
            if len(p_v) > 1:
                match = np.sign(p_v[1:] - p_v[:-1]) == np.sign(t_v[1:] - t_v[:-1])
                dir_matches.extend(match.tolist())
        directional_accuracy = float(np.mean(dir_matches)) if dir_matches else np.nan

        accuracy_metrics.append({
            'ticker'               : ticker,
            'rmse'                 : rmse,
            'confidence_rmse'      : 1 / (1 + rmse),
            'ic_mean'              : ic_mean,
            'directional_accuracy' : directional_accuracy,
        })

    df_accuracy = pd.DataFrame(accuracy_metrics)
    df_accuracy['accuracy_rank'] = df_accuracy['ic_mean'].rank(ascending=False)

    if verbose:
        print(f"\n✅ 정확도 평가 완료")
        print(f"   - 평가 종목 수: {len(df_accuracy)}")
        print(f"   - RMS RMSE: {(df_accuracy['rmse'].pow(2).mean()) ** 0.5:.4f}")
        print(f"   - 평균 IC: {df_accuracy['ic_mean'].mean():.4f}")
        print(f"   - 평균 방향성 정확도: {df_accuracy['directional_accuracy'].mean():.4f}")

    return df_accuracy


# ==========================================
# 2. 수익성 평가 (Return Evaluation)
# ==========================================

def evaluate_expected_returns(
    df_future_forecasts: pd.DataFrame,
    min_hold_days: int = 5,
    max_daily_return: Optional[float] = 0.16,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    미래 예측 데이터로부터 기대 수익률을 계산합니다.
    (변경 없음)
    """
    max_daily_log_return = np.log1p(max_daily_return) if max_daily_return is not None else None

    if verbose:
        limit_str = f"{max_daily_return:.1%}/일" if max_daily_return is not None else "없음"
        print(f"\n💰 수익성 평가 중 (시간당 로그 수익률 기준, 상한={limit_str})...")

    return_metrics  = []
    failed_tickers  = []

    tickers  = df_future_forecasts['ticker'].unique()
    iterator = tqdm(tickers, desc="최적 수익률 계산") if verbose else tickers

    for ticker in iterator:
        try:
            ticker_data = (
                df_future_forecasts[df_future_forecasts['ticker'] == ticker]
                .sort_values('date')
                .reset_index(drop=True)
            )

            if len(ticker_data) < min_hold_days:
                failed_tickers.append((ticker, f"데이터 부족 ({len(ticker_data)}일)"))
                continue

            log_prices = ticker_data['pred_log_close'].values

            buy_idx, sell_idx, daily_log_return, hold_days = find_best_trade_vectorized(
                log_prices,
                min_hold=min_hold_days,
                max_daily_log_return=max_daily_log_return,
            )

            if np.isnan(daily_log_return) or np.isinf(daily_log_return):
                failed_tickers.append((ticker, "유효한 거래 없음 (상한 초과 포함)"))
                continue

            buy_date   = ticker_data.iloc[buy_idx]['date']
            sell_date  = ticker_data.iloc[sell_idx]['date']
            buy_price  = ticker_data.iloc[buy_idx]['pred_close']
            sell_price = ticker_data.iloc[sell_idx]['pred_close']

            total_log_return  = log_prices[sell_idx] - log_prices[buy_idx]
            total_return_pct  = np.expm1(total_log_return) * 100
            annualized_return = daily_log_return * 244.5

            return_metrics.append({
                'ticker'           : ticker,
                'daily_log_return' : daily_log_return,
                'total_log_return' : total_log_return,
                'total_return_pct' : total_return_pct,
                'annualized_return': annualized_return,
                'hold_days'        : hold_days,
                'buy_date'         : buy_date,
                'sell_date'        : sell_date,
                'buy_price'        : buy_price,
                'sell_price'       : sell_price,
                'price_change_pct' : total_return_pct,
            })

        except Exception as e:
            failed_tickers.append((ticker, f"오류: {str(e)}"))
            continue

    df_return = pd.DataFrame(return_metrics)

    if len(df_return) == 0:
        raise ValueError("❌ 수익성 평가 가능한 종목이 없습니다.")

    df_return['return_rank'] = df_return['daily_log_return'].rank(
        ascending=False, method='min'
    )

    if verbose:
        print(f"\n✅ 수익성 평가 완료")
        print(f"   - 성공 종목 수: {len(df_return):,}개")
        print(f"   - 실패 종목 수: {len(failed_tickers):,}개")
        print(f"   - 평균 총 수익률: {df_return['total_return_pct'].mean():.2f}%")

    return df_return


# ==========================================
# 3. 위험도 평가 (Risk Evaluation)
# ==========================================

def evaluate_risk_metrics(
    df_future_forecasts: pd.DataFrame,
    df_meta: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    예측 시계열로부터 종목 내재 위험도를 평가합니다.
    (변경 없음)
    """
    if verbose:
        print(f"\n⚠️  위험도 평가 중 (5대 표준 지표)...")

    risk_results        = []
    failed_risk_tickers = []

    tickers  = df_future_forecasts['ticker'].unique()
    iterator = tqdm(tickers, desc="위험 지표 계산") if verbose else tickers

    for ticker in iterator:
        try:
            ticker_data = (
                df_future_forecasts[df_future_forecasts['ticker'] == ticker]
                .sort_values('date')
                .reset_index(drop=True)
            )

            if len(ticker_data) < 5:
                failed_risk_tickers.append((ticker, f"데이터 부족 ({len(ticker_data)}일)"))
                continue

            log_prices = ticker_data['pred_log_close'].values
            metrics    = calculate_risk_metrics(log_prices, is_log_prices=True)

            if any(np.isnan(v) for v in metrics.values()):
                failed_risk_tickers.append((ticker, "NaN 발생"))
                continue

            composite_score = calculate_composite_risk_score(metrics)

            risk_results.append({
                'ticker': ticker,
                **metrics,
                'risk_composite_raw': composite_score,
            })

        except Exception as e:
            failed_risk_tickers.append((ticker, f"오류: {str(e)}"))
            continue

    df_risk = pd.DataFrame(risk_results)

    if len(df_risk) == 0:
        raise ValueError("❌ 위험도 평가 가능한 종목이 없습니다.")

    df_risk = normalize_risk_scores(df_risk, score_col='risk_composite_raw')
    df_risk['safety_score'] = 1 - df_risk['risk_score_normalized']

    df_risk = df_risk.merge(
        df_meta[['ticker', 'liquidity_score', 'is_suspended', 'is_delisted']],
        on='ticker',
        how='left',
    )
    df_risk['liquidity_score'] = df_risk['liquidity_score'].fillna(0)
    df_risk['is_suspended']    = df_risk['is_suspended'].fillna(0).astype(int)
    df_risk['is_delisted']     = df_risk['is_delisted'].fillna(0).astype(int)

    if verbose:
        print(f"\n✅ 위험도 평가 완료")
        print(f"   - 성공 종목 수: {len(df_risk):,}개")
        print(f"   - 실패 종목 수: {len(failed_risk_tickers):,}개")
        print(f"   - 평균 변동성: {df_risk['volatility'].mean():.6f}")
        print(f"   - 평균 MDD: {df_risk['max_drawdown'].mean():.2%}")

    return df_risk


# ==========================================
# 4. 통합 Universe 선정 (Facade)
# ==========================================

def select_investment_universe(
    df_past_predictions: pd.DataFrame,
    df_future_forecasts: pd.DataFrame,
    df_meta: pd.DataFrame,
    *,
    model_date: str,
    top_k: int = 200,
    min_hold_days: int = 5,
    max_daily_return: Optional[float] = 0.16,
    target_columns: Optional[List[str]] = None,
    log_close_ref: Optional[pd.DataFrame] = None,
    integration_order: int = 1,
    filter_config: Optional[Dict] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    정확도, 수익성, 위험도를 종합 평가하여 투자 후보군을 선정합니다 (Facade).

    Parameters
    ----------
    df_past_predictions : pd.DataFrame
        과거 예측 결과 (Step 3)
    df_future_forecasts : pd.DataFrame
        미래 예측 결과 (Step 4)
    df_meta : pd.DataFrame
        메타 정보 (최신 날짜만 필터링 된 데이터)
    model_date : str
        모델 학습 기준일 (정확도 평가 기준점)
    top_k : int, default=200
        최종 선정할 후보 종목 수
    min_hold_days : int, default=5
        최소 보유 기간 (일)
    max_daily_return : float, optional, default=0.16
        최적 거래 탐색 시 허용 가능한 일평균 수익률 상한.
    target_columns : List[str], optional
        평가할 타겟 컬럼 리스트
    log_close_ref : pd.DataFrame, optional
        log_return_1d 모드 역산용 앵커 데이터프레임.
        integration_order=2 시 target_log_return_1d_lag1 컬럼 포함 필요.
    integration_order : int, default=1
        log-close 역산 수치 적분 차수.
        0 — 직사각형 / 1 — 사다리꼴 / 2 — Adams-Moulton 2-step
    filter_config : dict, optional
        하드 필터 설정
    verbose : bool
        진행 상황 출력 여부

    Returns
    -------
    dict
        평가 결과 및 최종 후보군(candidates)을 포함하는 딕셔너리
    """
    if verbose:
        print("\n" + "=" * 65)
        print("🚀 Universe 선정 시작")
        print(f"   최소 보유 기간:  {min_hold_days}일")
        if max_daily_return is not None:
            print(f"   수익률 상한:     일평균 {max_daily_return:.1%}")
        else:
            print(f"   수익률 상한:     없음")
        print(f"   integration_order: {integration_order}")
        print("=" * 65)

    df_accuracy = evaluate_model_accuracy(
        df_past_predictions,
        model_date=model_date,
        target_columns=target_columns,
        log_close_ref=log_close_ref,
        integration_order=integration_order,
        verbose=verbose,
    )

    df_return = evaluate_expected_returns(
        df_future_forecasts,
        min_hold_days=min_hold_days,
        max_daily_return=max_daily_return,
        verbose=verbose,
    )

    df_risk = evaluate_risk_metrics(
        df_future_forecasts,
        df_meta,
        verbose=verbose,
    )

    if verbose:
        print(f"\n🔗 평가 지표 통합 중...")

    df_universe = df_accuracy.merge(df_return, on='ticker', how='inner')
    df_universe = df_universe.merge(df_risk,   on='ticker', how='inner')

    if verbose:
        print(f"   - 통합 완료: {len(df_universe):,}개 종목")

    df_filtered, filter_stats = apply_hard_filters(
        df_universe,
        df_future_forecasts,
        df_meta,
        config=filter_config,
        verbose=verbose,
    )

    if verbose:
        print(f"\n📊 수익률 기준 정렬 및 Top-K 선정 중...")

    df_filtered = df_filtered.sort_values(
        'daily_log_return', ascending=False
    ).reset_index(drop=True)

    if len(df_filtered) < top_k:
        if verbose:
            print(f"   ⚠️  필터링 후 종목 수({len(df_filtered)})가 TOP_K({top_k})보다 적습니다.")
        df_candidates = df_filtered.copy()
    else:
        df_candidates = df_filtered.head(top_k).copy()

    df_candidates['return_rank'] = range(1, len(df_candidates) + 1)

    if verbose:
        print(f"\n✅ Universe 선정 완료")
        print(f"   - 최종 후보 종목 수: {len(df_candidates):,}개")
        print(f"   - 평균 일평균 로그 수익률: {df_candidates['daily_log_return'].mean():.6f}")
        print(f"   - 평균 총 수익률: {df_candidates['total_return_pct'].mean():.2f}%")
        print("\n" + "=" * 65)

    return {
        'accuracy'    : df_accuracy,
        'returns'     : df_return,
        'risk'        : df_risk,
        'full'        : df_filtered,
        'candidates'  : df_candidates,
        'filter_stats': filter_stats,
    }
