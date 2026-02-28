"""
Universe Selection Module - Facade Pattern

✨ H3 패치 (2026-02-08):
    - 기존: 미사용 함수들
    - 개선: Step 5 노트북의 200줄 로직을 모듈로 캡슐화
    - 패턴: Facade (복잡한 흐름을 단일 인터페이스로 제공)

✨ v3.7.2 패치 (2026-02-24):
    - evaluate_expected_returns(): max_daily_log_return 파라미터 추가
    - select_investment_universe(): max_daily_log_return 파라미터 추가 및 전파

주요 기능:
1. evaluate_model_accuracy(): 과거 예측 정확도 평가
2. evaluate_expected_returns(): 미래 기대 수익률 계산
3. evaluate_risk_metrics(): 종목 내재 위험 평가
4. select_investment_universe(): 전체 흐름 통합 (Facade)

사용 예시:
    >>> from src.universe.select_universe import select_investment_universe
    >>>
    >>> results = select_investment_universe(
    ...     df_past_predictions,
    ...     df_future_forecasts,
    ...     df_meta,
    ...     model_date='2026-01-20',
    ...     top_k=200
    ... )
    >>>
    >>> df_candidates = results['candidates']  # Top-K 후보
    >>> df_full = results['full']              # 전체 Universe
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from tqdm import tqdm
from scipy.stats import spearmanr

# 범용 계산 유틸리티 (utils에 유지)
from src.utils.trading import find_best_trade_vectorized
from src.utils.risk import (
    calculate_risk_metrics,
    calculate_composite_risk_score,
    normalize_risk_scores,
)

# 한국 시장 특화 필터 (universe로 이전됨)
from src.universe.filters import apply_hard_filters


# ==========================================
# 1. 정확도 평가 (Accuracy Evaluation)
# ==========================================

def evaluate_model_accuracy(
    df_past_predictions: pd.DataFrame,
    model_date: str,
    target_columns: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    과거 예측 데이터로부터 모델 정확도 평가

    노트북의 정확도 평가 루프를 모듈로 추출

    Parameters
    ----------
    df_past_predictions : pd.DataFrame
        과거 예측 결과 (Step 3 출력)
        필수 컬럼: ticker, date, pred_target_log_close_h*, true_target_log_close_h*
    model_date : str
        모델 학습 기준일 (이 날짜까지만 평가)
    target_columns : list, optional
        평가할 Horizon 리스트 (기본: h1~h5)
    verbose : bool
        진행 상황 출력 여부

    Returns
    -------
    pd.DataFrame
        종목별 정확도 지표
        컬럼: ticker, rmse, mae, ic_mean, accuracy_rank
    """
    if target_columns is None:
        target_columns = [f'target_log_close_h{h}' for h in range(1, 6)]

    model_date_dt = pd.to_datetime(model_date)
    df_eval = df_past_predictions[
        df_past_predictions['date'] <= model_date_dt
    ].copy()

    if len(df_eval) == 0:
        raise ValueError(
            f"❌ {model_date} 이전의 예측 데이터가 없습니다.\n"
            f"   predictions.parquet의 날짜 범위를 확인하세요."
        )

    accuracy_metrics = []
    tickers = df_eval['ticker'].unique()
    iterator = tqdm(tickers, desc="정확도 평가") if verbose else tickers

    for ticker in iterator:
        ticker_eval = df_eval[df_eval['ticker'] == ticker]

        rmse_list, ic_list = [], []
        for col in target_columns:
            pred_col = f'pred_{col}'
            true_col = f'true_{col}'
            if pred_col not in ticker_eval.columns or true_col not in ticker_eval.columns:
                continue

            valid = ticker_eval[[pred_col, true_col]].dropna()
            if len(valid) < 2:
                continue

            pred = valid[pred_col].values
            true = valid[true_col].values
            rmse_list.append(np.sqrt(np.mean((pred - true) ** 2)))

            if len(valid) >= 3:
                ic, _ = spearmanr(pred, true)
                ic_list.append(ic if not np.isnan(ic) else 0.0)

        if not rmse_list:
            continue

        rmse    = float(np.mean(rmse_list))
        ic_mean = float(np.mean(ic_list)) if ic_list else 0.0

        accuracy_metrics.append({
            'ticker'         : ticker,
            'rmse'           : rmse,
            'confidence_rmse': 1 / (1 + rmse),
            'ic_mean'        : ic_mean,
        })

    df_accuracy = pd.DataFrame(accuracy_metrics)
    df_accuracy['accuracy_rank'] = df_accuracy['ic_mean'].rank(ascending=False)

    if verbose:
        print(f"\n✅ 정확도 평가 완료")
        print(f"   - 평가 종목 수: {len(df_accuracy)}")
        print(f"   - 평균 RMSE: {df_accuracy['rmse'].mean():.4f}")
        print(f"   - 평균 IC (Information Coefficient): {df_accuracy['ic_mean'].mean():.4f}")

    return df_accuracy


# ==========================================
# 2. 수익성 평가 (Return Evaluation)
# ==========================================

def evaluate_expected_returns(
    df_future_forecasts: pd.DataFrame,
    min_hold_days: int = 5,
    max_daily_return: Optional[float] = 0.16,  # ✨ v3.7.2: config 직관값 (0.16 = 16%)
    verbose: bool = True,
) -> pd.DataFrame:
    """
    미래 예측 데이터로부터 기대 수익률 계산.

    노트북의 수익성 평가 루프를 모듈로 추출.

    Parameters
    ----------
    df_future_forecasts : pd.DataFrame
        미래 예측 결과 (Step 4 출력)
        필수 컬럼: ticker, date, pred_log_close, pred_close
    min_hold_days : int
        최소 보유 기간 (일)
    max_daily_return : float or None, default=0.16
        허용 가능한 일평균 수익률 상한 (직관적 소수 형태). ✨ v3.7.2
        내부에서 ``np.log1p(max_daily_return)`` 으로 변환하여 탐색에 사용.
        이 값을 초과하는 거래는 탐색 대상에서 제외되며,
        상한 이하의 차선 거래가 반환됩니다.
        ``None`` 이면 상한 없이 전체 탐색합니다.
    verbose : bool
        진행 상황 출력 여부

    Returns
    -------
    pd.DataFrame
        종목별 수익성 지표
        컬럼: ticker, daily_log_return, total_log_return, total_return_pct,
              annualized_return, hold_days, buy_date, sell_date,
              buy_price, sell_price, return_rank

    Examples
    --------
    >>> df_return = evaluate_expected_returns(df_future, min_hold_days=5)
    >>> print(df_return.nlargest(10, 'daily_log_return'))

    >>> # 상한 비활성화 (기존 동작)
    >>> df_return = evaluate_expected_returns(df_future, max_daily_return=None)
    """
    # 직관적 수익률 → 로그 수익률 변환
    max_daily_log_return = np.log1p(max_daily_return) if max_daily_return is not None else None

    if verbose:
        if max_daily_return is not None:
            print(f"\n💰 수익성 평가 중 (시간당 로그 수익률 기준, 상한={max_daily_return:.1%}/일)...")
        else:
            print(f"\n💰 수익성 평가 중 (시간당 로그 수익률 기준, 상한=없음)...")

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

            # ✨ v3.7.2: max_daily_log_return 전파
            buy_idx, sell_idx, daily_log_return, hold_days = find_best_trade_vectorized(
                log_prices,
                min_hold=min_hold_days,
                max_daily_log_return=max_daily_log_return,
            )

            if np.isnan(daily_log_return) or np.isinf(daily_log_return):
                failed_tickers.append((ticker, "유효한 거래 없음 (상한 초과 포함)"))
                continue

            buy_date  = ticker_data.iloc[buy_idx]['date']
            sell_date = ticker_data.iloc[sell_idx]['date']
            buy_price  = ticker_data.iloc[buy_idx]['pred_close']
            sell_price = ticker_data.iloc[sell_idx]['pred_close']

            total_log_return = log_prices[sell_idx] - log_prices[buy_idx]
            total_return_pct = np.expm1(total_log_return) * 100
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
        print(f"   - 실패 종목 수: {len(failed_tickers):,}개 (상한 초과로 제외된 종목 포함)")
        print(f"   - 평균 시간당 로그 수익률: {df_return['daily_log_return'].mean():.6f}")
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
    예측 시계열로부터 종목 내재 위험 평가.

    노트북의 위험도 평가 루프를 모듈로 추출.

    Parameters
    ----------
    df_future_forecasts : pd.DataFrame
        미래 예측 결과 (Step 4 출력)
        필수 컬럼: ticker, date, pred_log_close
    df_meta : pd.DataFrame
        메타 정보 (Step 2 출력)
        필수 컬럼: ticker, liquidity_score, is_suspended, is_delisted
    verbose : bool
        진행 상황 출력 여부

    Returns
    -------
    pd.DataFrame
        종목별 위험 지표
        컬럼: ticker, volatility, downside_risk, var_95, cvar_95,
              max_drawdown, skewness, kurtosis, risk_composite_raw,
              risk_score_normalized, safety_score
    """
    if verbose:
        print(f"\n⚠️  위험도 평가 중 (5대 표준 지표)...")

    risk_results       = []
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
    top_k: int = 100,
    min_hold_days: int = 5,                        # ✨ v3.7.2: config 연동
    max_daily_return: Optional[float] = 0.16,      # ✨ v3.7.2: config 직관값 (0.16 = 16%)
    filter_config: Optional[Dict] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Step 5 전체 로직 캡슐화 (Facade Pattern).

    노트북에서는 이 함수 하나만 호출하면 됩니다.

    Parameters
    ----------
    df_past_predictions : pd.DataFrame
        과거 예측 결과 (Step 3)
    df_future_forecasts : pd.DataFrame
        미래 예측 결과 (Step 4)
    df_meta : pd.DataFrame
        메타 정보 (Step 2, 최신 날짜만)
    model_date : str
        모델 학습 기준일 (정확도 평가 기준)
    top_k : int
        최종 선정할 후보 종목 수
    min_hold_days : int, default=5
        최소 보유 기간 (일). config의 ``strategy.min_hold_days`` 에서 읽어 전달. ✨ v3.7.2
    max_daily_return : float or None, default=0.16
        최적 거래 탐색 시 허용 가능한 일평균 수익률 상한 (직관적 소수 형태). ✨ v3.7.2
        config의 ``strategy.max_daily_return`` 에서 읽어 전달.
        내부에서 ``np.log1p()`` 변환 후 탐색에 사용.
        상한을 초과하는 거래는 탐색 제외 → 차선 거래가 후보로 제안됨.
        하드 필터와 달리 종목 자체는 제거되지 않습니다.
        ``None`` 이면 상한 없이 탐색합니다.
    filter_config : dict, optional
        하드 필터 설정 (없으면 기본값 사용)
    verbose : bool
        진행 상황 출력 여부

    Returns
    -------
    dict
        - 'accuracy'     : 정확도 평가 결과 (DataFrame)
        - 'returns'      : 수익성 평가 결과 (DataFrame)
        - 'risk'         : 위험도 평가 결과 (DataFrame)
        - 'full'         : 전체 Universe (필터링 후, DataFrame)
        - 'candidates'   : Top-K 후보 (DataFrame)
        - 'filter_stats' : 필터링 통계 (dict)

    Examples
    --------
    >>> # config 값을 직접 전달하는 권장 방식
    >>> results = select_investment_universe(
    ...     df_past_pred, df_future, df_meta_latest,
    ...     model_date=cfg['universe']['model_date'],
    ...     top_k=200,
    ...     min_hold_days=cfg['strategy']['min_hold_days'],
    ...     max_daily_return=cfg['strategy']['max_daily_return'],
    ... )

    >>> # 상한 비활성화 (기존 동작)
    >>> results = select_investment_universe(
    ...     df_past_pred, df_future, df_meta_latest,
    ...     model_date='2026-01-20',
    ...     max_daily_return=None,
    ... )
    """
    if verbose:
        print("\n" + "=" * 65)
        print("🚀 Universe 선정 시작")
        if max_daily_return is not None:
            print(f"   최소 보유 기간:  {min_hold_days}일")
            print(f"   수익률 상한:     일평균 {max_daily_return:.1%}")
        else:
            print(f"   최소 보유 기간:  {min_hold_days}일")
            print(f"   수익률 상한:     없음")
        print("=" * 65)

    # ── 1. 정확도 평가 ────────────────────────────────────────────────
    df_accuracy = evaluate_model_accuracy(
        df_past_predictions,
        model_date=model_date,
        verbose=verbose,
    )

    # ── 2. 수익성 평가 ✨ v3.7.2: config 파라미터 전파 ────────────────
    df_return = evaluate_expected_returns(
        df_future_forecasts,
        min_hold_days=min_hold_days,
        max_daily_return=max_daily_return,
        verbose=verbose,
    )

    # ── 3. 위험도 평가 ────────────────────────────────────────────────
    df_risk = evaluate_risk_metrics(
        df_future_forecasts,
        df_meta,
        verbose=verbose,
    )

    # ── 4. 3대 지표 통합 ──────────────────────────────────────────────
    if verbose:
        print(f"\n🔗 평가 지표 통합 중...")

    df_universe = df_accuracy.merge(df_return, on='ticker', how='inner')
    df_universe = df_universe.merge(df_risk,   on='ticker', how='inner')

    if verbose:
        print(f"   - 통합 완료: {len(df_universe):,}개 종목")

    # ── 5. 하드 필터링 ────────────────────────────────────────────────
    df_filtered, filter_stats = apply_hard_filters(
        df_universe,
        df_future_forecasts,
        df_meta,
        config=filter_config,
        verbose=verbose,
    )

    # ── 6. 수익률 기준 정렬 및 Top-K 선정 ────────────────────────────
    if verbose:
        print(f"\n📊 수익률 기준 정렬 및 Top-K 선정 중...")

    df_filtered = df_filtered.sort_values(
        'daily_log_return', ascending=False
    ).reset_index(drop=True)

    if len(df_filtered) < top_k:
        if verbose:
            print(f"   ⚠️  필터링 후 종목 수({len(df_filtered)})가 TOP_K({top_k})보다 적습니다.")
            print(f"   → 전체 {len(df_filtered)}개 종목을 후보로 선정합니다.")
        df_candidates = df_filtered.copy()
    else:
        df_candidates = df_filtered.head(top_k).copy()

    df_candidates['return_rank'] = range(1, len(df_candidates) + 1)

    if verbose:
        print(f"\n✅ Universe 선정 완료")
        print(f"   - 최종 후보 종목 수: {len(df_candidates):,}개")
        print(f"   - 평균 일평균 로그 수익률: {df_candidates['daily_log_return'].mean():.6f}")
        print(f"   - 평균 총 수익률: {df_candidates['total_return_pct'].mean():.2f}%")
        print(f"   - 평균 리스크 점수: {df_candidates['risk_composite_raw'].mean():.4f}")
        print("\n" + "=" * 65)

    return {
        'accuracy'    : df_accuracy,
        'returns'     : df_return,
        'risk'        : df_risk,
        'full'        : df_filtered,
        'candidates'  : df_candidates,
        'filter_stats': filter_stats,
    }
