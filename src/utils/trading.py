"""
Trading utility functions

최적 매매 시점 탐색 및 수익률 계산
"""

import numpy as np
import pandas as pd
from typing import Tuple


def find_best_trade_vectorized(
    log_prices: np.ndarray,
    min_hold: int = 5
) -> Tuple[int, int, float, int]:
    """
    벡터화된 완전 탐색으로 최적 매매 시점 탐색
    
    시간당 로그 수익률 = (log(매도가) - log(매수가)) / 보유기간
    을 최대화하는 매수/매도 시점 탐색
    
    Parameters
    ----------
    log_prices : np.ndarray
        로그 종가 배열 (이미 로그 변환됨)
    min_hold : int, default=5
        최소 보유 기간 (영업일)
        
    Returns
    -------
    buy_idx : int
        매수 시점 인덱스
    sell_idx : int
        매도 시점 인덱스
    daily_log_return : float
        일평균 로그 수익률 (시간당)
    hold_days : int
        실제 보유 기간
        
    Notes
    -----
    복잡도: O(N²) - NumPy 벡터화로 빠름
    
    Examples
    --------
    >>> log_prices = np.array([7.5, 7.6, 7.55, 7.7, 7.65])
    >>> buy, sell, return_rate, days = find_best_trade_vectorized(log_prices, min_hold=2)
    >>> print(f"매수: {buy}, 매도: {sell}, 일평균 수익률: {return_rate:.4f}")
    """
    n = len(log_prices)
    
    # 상삼각 행렬 생성 (i < j)
    i_indices = np.arange(n)[:, None]  # (n, 1)
    j_indices = np.arange(n)[None, :]  # (1, n)
    
    # 보유기간 계산
    hold_days = j_indices - i_indices
    
    # 로그 수익률 계산 (매도 - 매수)
    log_returns = log_prices[None, :] - log_prices[:, None]  # (n, n)
    
    # 시간당 로그 수익률 = 로그 수익률 / 보유기간
    daily_log_returns = log_returns / np.where(hold_days > 0, hold_days, 1)
    
    # 유효한 거래만 선택 (보유기간 >= min_hold)
    valid_mask = hold_days >= min_hold
    daily_log_returns = np.where(valid_mask, daily_log_returns, -np.inf)
    
    # 최적 거래 찾기
    best_idx = np.argmax(daily_log_returns)
    best_i, best_j = np.unravel_index(best_idx, daily_log_returns.shape)
    best_return = daily_log_returns[best_i, best_j]
    actual_hold = hold_days[best_i, best_j]
    
    return int(best_i), int(best_j), float(best_return), int(actual_hold)


def calculate_expected_return_from_forecasts(
    df_forecast: pd.DataFrame,
    min_hold: int = 5
) -> pd.DataFrame:
    """
    종목별 미래 예측 데이터로부터 기대 수익률 계산
    
    Parameters
    ----------
    df_forecast : pd.DataFrame
        미래 예측 데이터 (columns: ticker, date, pred_log_close, ...)
    min_hold : int, default=5
        최소 보유 기간
        
    Returns
    -------
    pd.DataFrame
        종목별 기대 수익 지표
        columns: ticker, expected_daily_log_return, optimal_buy_date, 
                 optimal_sell_date, optimal_hold_days
    """
    results = []
    
    for ticker in df_forecast['ticker'].unique():
        ticker_data = df_forecast[df_forecast['ticker'] == ticker].sort_values('date')
        
        if len(ticker_data) < min_hold:
            continue
            
        # 로그 가격 (이미 변환됨)
        log_prices = ticker_data['pred_log_close'].values
        dates = ticker_data['date'].values
        
        # 최적 매매 시점 탐색
        buy_idx, sell_idx, daily_return, hold_days = find_best_trade_vectorized(
            log_prices, min_hold
        )
        
        results.append({
            'ticker': ticker,
            'expected_daily_log_return': daily_return,
            'optimal_buy_date': dates[buy_idx],
            'optimal_sell_date': dates[sell_idx],
            'optimal_hold_days': hold_days,
            'expected_total_return': daily_return * hold_days,  # 총 수익률
            'num_forecast_days': len(ticker_data)
        })
    
    return pd.DataFrame(results)
