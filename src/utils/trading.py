"""
Trading utility functions

최적 매매 시점 탐색 및 수익률 계산
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple

# ✨ v3.7.2: 신뢰 불가 거래의 기본 상한 (일평균 로그 수익률)
# log1p(0.16) ≈ 0.1484 — 보유 기간 내내 일평균 16% 이상이면 예측 오류로 간주
DEFAULT_MAX_DAILY_LOG_RETURN: float = np.log1p(0.16)


def find_best_trade_vectorized(
    log_prices: np.ndarray,
    min_hold: int = 5,
    max_daily_log_return: Optional[float] = DEFAULT_MAX_DAILY_LOG_RETURN,
) -> Tuple[int, int, float, int]:
    """
    벡터화된 완전 탐색으로 최적 매매 시점 탐색.

    시간당 로그 수익률 = (log(매도가) - log(매수가)) / 보유기간
    을 최대화하는 매수/매도 시점을 탐색하되,
    ``max_daily_log_return`` 을 초과하는 거래는 신뢰 불가 예측으로 간주하여
    탐색 대상에서 제외합니다.

    Parameters
    ----------
    log_prices : np.ndarray
        로그 종가 배열 (이미 로그 변환됨).
    min_hold : int, default=5
        최소 보유 기간 (영업일).
    max_daily_log_return : float or None, default=log1p(0.16)
        허용 가능한 일평균 로그 수익률 상한.
        이 값을 초과하는 (i, j) 쌍은 유효 탐색 범위에서 제외됩니다.
        상한을 초과하는 거래가 있더라도 상한 이하의 차선 거래가 반환됩니다.
        ``None`` 으로 설정하면 상한 없이 전체 탐색합니다.

    Returns
    -------
    buy_idx : int
        매수 시점 인덱스.
    sell_idx : int
        매도 시점 인덱스.
    daily_log_return : float
        일평균 로그 수익률. 유효한 거래가 전혀 없으면 ``-np.inf``.
    hold_days : int
        실제 보유 기간. 유효한 거래가 전혀 없으면 0.

    Notes
    -----
    복잡도: O(N²) — NumPy 벡터화로 빠름.

    상한 초과 거래 제외 동작
        - 상한 이하 거래가 존재하면: 그 중 최고 수익률 거래 반환.
        - 모든 거래가 상한 초과이면: ``(0, 0, -inf, 0)`` 반환.
          호출부에서 ``np.isinf(daily_log_return)`` 으로 감지 가능.

    Examples
    --------
    >>> log_prices = np.array([7.5, 7.6, 7.55, 7.7, 7.65])
    >>> buy, sell, ret, days = find_best_trade_vectorized(log_prices, min_hold=2)
    >>> print(f"매수: {buy}, 매도: {sell}, 일평균 수익률: {ret:.4f}")

    >>> # 상한 비활성화
    >>> buy, sell, ret, days = find_best_trade_vectorized(
    ...     log_prices, min_hold=2, max_daily_log_return=None
    ... )
    """
    n = len(log_prices)

    # 상삼각 행렬 생성 (i < j)
    i_indices = np.arange(n)[:, None]  # (n, 1)
    j_indices = np.arange(n)[None, :]  # (1, n)

    # 보유기간 행렬
    hold_days_matrix = j_indices - i_indices  # (n, n)

    # 로그 수익률 행렬 (매도 - 매수)
    log_returns = log_prices[None, :] - log_prices[:, None]  # (n, n)

    # 일평균 로그 수익률 행렬
    daily_log_returns = log_returns / np.where(hold_days_matrix > 0, hold_days_matrix, 1)

    # ── 유효 조건 1: 최소 보유 기간 ─────────────────────────────────────
    valid_mask = hold_days_matrix >= min_hold

    # ── 유효 조건 2: 수익률 상한 ✨ v3.7.2 ──────────────────────────────
    if max_daily_log_return is not None:
        valid_mask = valid_mask & (daily_log_returns <= max_daily_log_return)

    # 유효하지 않은 셀은 -inf 로 마스킹
    daily_log_returns_masked = np.where(valid_mask, daily_log_returns, -np.inf)

    # 유효한 거래가 하나도 없으면 조기 반환
    if not np.any(valid_mask):
        return 0, 0, float(-np.inf), 0

    # 최적 거래 탐색
    best_flat_idx = np.argmax(daily_log_returns_masked)
    best_i, best_j = np.unravel_index(best_flat_idx, daily_log_returns_masked.shape)

    best_return   = float(daily_log_returns_masked[best_i, best_j])
    actual_hold   = int(hold_days_matrix[best_i, best_j])

    return int(best_i), int(best_j), best_return, actual_hold


def calculate_expected_return_from_forecasts(
    df_forecast: pd.DataFrame,
    min_hold: int = 5,
    max_daily_log_return: Optional[float] = DEFAULT_MAX_DAILY_LOG_RETURN,
) -> pd.DataFrame:
    """
    종목별 미래 예측 데이터로부터 기대 수익률 계산.

    Parameters
    ----------
    df_forecast : pd.DataFrame
        미래 예측 데이터 (columns: ticker, date, pred_log_close, ...).
    min_hold : int, default=5
        최소 보유 기간.
    max_daily_log_return : float or None, default=log1p(0.16)
        ``find_best_trade_vectorized`` 에 전달될 수익률 상한.

    Returns
    -------
    pd.DataFrame
        종목별 기대 수익 지표.
        columns: ticker, expected_daily_log_return, optimal_buy_date,
                 optimal_sell_date, optimal_hold_days, expected_total_return,
                 num_forecast_days.
    """
    results = []

    for ticker in df_forecast['ticker'].unique():
        ticker_data = (
            df_forecast[df_forecast['ticker'] == ticker]
            .sort_values('date')
        )

        if len(ticker_data) < min_hold:
            continue

        log_prices = ticker_data['pred_log_close'].values
        dates      = ticker_data['date'].values

        buy_idx, sell_idx, daily_return, hold_days = find_best_trade_vectorized(
            log_prices,
            min_hold=min_hold,
            max_daily_log_return=max_daily_log_return,
        )

        # 유효한 거래가 없는 종목은 결과에서 제외
        if np.isinf(daily_return):
            continue

        results.append({
            'ticker'                    : ticker,
            'expected_daily_log_return' : daily_return,
            'optimal_buy_date'          : dates[buy_idx],
            'optimal_sell_date'         : dates[sell_idx],
            'optimal_hold_days'         : hold_days,
            'expected_total_return'     : daily_return * hold_days,
            'num_forecast_days'         : len(ticker_data),
        })

    return pd.DataFrame(results)
