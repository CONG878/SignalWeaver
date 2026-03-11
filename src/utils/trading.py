"""
Trading Utility Functions

미래 예측 종가로부터 최적 거래 시점과 기대 수익률을 계산합니다.
벡터화된 O(N²) 알고리즘으로 고성능을 제공하며, v3.7.2부터 비현실적 수익률 필터링을 지원합니다.

## 버전
- v3.7.2: 최대 일평균 수익률 상한 필터링 도입
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple

# v3.7.2: 신뢰 불가 거래의 기본 상한 (일평균 로그 수익률)
# log1p(0.16) ≈ 0.1484 — 보유 기간 내 일평균 16% 이상이면 예측 오류 가능성 높음
DEFAULT_MAX_DAILY_LOG_RETURN: float = np.log1p(0.16)


def find_best_trade_vectorized(
    log_prices: np.ndarray,
    min_hold: int = 5,
    max_daily_log_return: Optional[float] = DEFAULT_MAX_DAILY_LOG_RETURN,
) -> Tuple[int, int, float, int]:
    """
    벡터화된 완전 탐색으로 최적 매매 시점 탐색.

    시간당 로그 수익률 = (log(매도가) - log(매수가)) / 보유기간 을 최대화하는
    매수/매도 시점을 탐색합니다. ``max_daily_log_return`` 을 초과하는 거래는
    신뢰 불가 예측으로 간주하여 탐색 대상에서 제외합니다.

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
    **벡터화 알고리즘** (O(N²) 시간복잡도)

    1. 행렬 생성
       - i_indices: (N,1) 배열 → 매수 시점
       - j_indices: (1,N) 배열 → 매도 시점
       - hold_days_matrix = j_indices - i_indices → (N,N) 보유기간 행렬
       - log_returns = log_prices[j] - log_prices[i] → (N,N) 로그 수익률 행렬

    2. 마스킹
       - 보유기간 >= min_hold 조건
       - 일평균 수익률 <= max_daily_log_return 조건
       - 두 조건을 AND로 결합하여 유효 범위 결정

    3. 최적값 탐색
       - np.argmax()로 유효 범위 내 최고 수익률 찾기
       - np.unravel_index()로 2D 인덱스 복원

    **상한 초과 거래 처리**
    - 상한 이하 거래가 존재: 최고 수익률 거래 반환
    - 모든 거래가 상한 초과: (0, 0, -inf, 0) 반환 (유효 거래 없음)

    Examples
    --------
    >>> log_prices = np.log([100, 110, 105, 120])
    >>> buy, sell, ret, days = find_best_trade_vectorized(log_prices, min_hold=2)
    >>> print(f"Buy: {buy}, Sell: {sell}, Daily Return: {ret:.4f}")
    """
    n = len(log_prices)

    # 행렬 생성: 상삼각 (i < j)
    i_indices = np.arange(n)[:, None]
    j_indices = np.arange(n)[None, :]

    # 보유기간 행렬
    hold_days_matrix = j_indices - i_indices

    # 로그 수익률 행렬
    log_returns = log_prices[None, :] - log_prices[:, None]

    # 일평균 로그 수익률 행렬
    daily_log_returns = log_returns / np.where(hold_days_matrix > 0, hold_days_matrix, 1)

    # 유효 조건 1: 최소 보유 기간
    valid_mask = hold_days_matrix >= min_hold

    # 유효 조건 2: 수익률 상한 (v3.7.2)
    if max_daily_log_return is not None:
        valid_mask = valid_mask & (daily_log_returns <= max_daily_log_return)

    # 유효하지 않은 셀은 -inf로 마스킹
    daily_log_returns_masked = np.where(valid_mask, daily_log_returns, -np.inf)

    # 유효한 거래가 없으면 조기 반환
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
    미래 예측 종가로부터 종목별 기대 수익률 계산.

    각 종목에 대해 find_best_trade_vectorized()를 호출하여
    최적 매수/매도 시점과 수익률을 계산합니다.

    Parameters
    ----------
    df_forecast : pd.DataFrame
        미래 예측 데이터 (필수 컬럼: ticker, date, pred_log_close).
    min_hold : int, default=5
        최소 보유 기간 (영업일).
    max_daily_log_return : float or None, default=log1p(0.16)
        ``find_best_trade_vectorized`` 에 전달될 수익률 상한. v3.7.2 도입.

    Returns
    -------
    pd.DataFrame
        종목별 기대 수익 지표.
        컬럼: ticker, expected_daily_log_return, optimal_buy_date,
             optimal_sell_date, optimal_hold_days, expected_total_return,
             num_forecast_days.

    Notes
    -----
    - 유효한 거래가 없는 종목은 결과에서 제외됨
    - log_prices는 미리 로그 변환되어야 함
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
