"""
Purpose:
    기술적 지표(Technical Indicators) 계산 함수 모음
    - 노트북에 분산되어 있던 RSI, MACD, Bollinger 등을 통합
    - 표준화된 인터페이스 제공
    - 재사용 가능한 순수 함수

Design Principles:
    - Pure Functions: 입력 → 출력만, side-effect 없음
    - Pandas Series 기반: 종목별 그룹 연산에 적합
    - Defensive Programming: 0 나누기, NaN 처리 내장

Usage:
    from src.features.technical import calc_rsi, calc_macd, calc_bollinger
    
    df['rsi'] = df.groupby('ticker')['close'].transform(
        lambda x: calc_rsi(x, period=14)
    )
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Moving Averages (이동평균)
# ---------------------------------------------------------------------

def calc_sma(series: pd.Series, window: int) -> pd.Series:
    """
    Simple Moving Average (단순 이동평균)
    
    Parameters
    ----------
    series : pd.Series
        가격 시계열 (예: close)
    window : int
        이동평균 기간
        
    Returns
    -------
    pd.Series
        이동평균 값
    """
    return series.rolling(window=window).mean()


def calc_ema(series: pd.Series, span: int) -> pd.Series:
    """
    Exponential Moving Average (지수 이동평균)
    
    Parameters
    ----------
    series : pd.Series
        가격 시계열
    span : int
        EMA 기간
        
    Returns
    -------
    pd.Series
        지수 이동평균 값
    """
    return series.ewm(span=span, adjust=False).mean()


# ---------------------------------------------------------------------
# RSI (Relative Strength Index)
# ---------------------------------------------------------------------

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI (Relative Strength Index) 계산
    
    상대강도지수: 과매수/과매도 판단 지표
    - 70 이상: 과매수
    - 30 이하: 과매도
    
    Parameters
    ----------
    series : pd.Series
        가격 시계열 (일반적으로 close)
    period : int, default 14
        RSI 계산 기간
        
    Returns
    -------
    pd.Series
        RSI 값 (0~100)
        
    Examples
    --------
    >>> close = pd.Series([100, 102, 101, 103, 105, 104, 106])
    >>> rsi = calc_rsi(close, period=6)
    >>> rsi.iloc[-1]  # 마지막 RSI 값
    """
    delta = series.diff()
    
    # 상승분/하락분 분리
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # 평균 계산
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # RS (Relative Strength)
    rs = avg_gain / (avg_loss + 1e-10)  # 0 나누기 방지
    
    # RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


# ---------------------------------------------------------------------
# MACD (Moving Average Convergence Divergence)
# ---------------------------------------------------------------------

def calc_macd(
    series: pd.Series,
    short: int = 12,
    long: int = 26,
    signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD (Moving Average Convergence Divergence) 계산
    
    추세 전환 및 모멘텀 판단 지표
    
    Parameters
    ----------
    series : pd.Series
        가격 시계열
    short : int, default 12
        단기 EMA 기간
    long : int, default 26
        장기 EMA 기간
    signal : int, default 9
        시그널선 EMA 기간
        
    Returns
    -------
    macd : pd.Series
        MACD 선 (단기 EMA - 장기 EMA)
    macd_signal : pd.Series
        시그널 선 (MACD의 EMA)
    macd_hist : pd.Series
        히스토그램 (MACD - 시그널)
        
    Examples
    --------
    >>> macd, signal, hist = calc_macd(df['close'])
    >>> # 골든크로스: macd > signal (매수 신호)
    >>> # 데드크로스: macd < signal (매도 신호)
    """
    ema_short = calc_ema(series, span=short)
    ema_long = calc_ema(series, span=long)
    
    macd = ema_short - ema_long
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    
    return macd, macd_signal, macd_hist


# ---------------------------------------------------------------------
# Bollinger Bands (볼린저 밴드)
# ---------------------------------------------------------------------

def calc_bollinger(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands (볼린저 밴드) 계산
    
    변동성 기반 추세 및 반전 판단 지표
    
    Parameters
    ----------
    series : pd.Series
        가격 시계열
    window : int, default 20
        이동평균 기간
    num_std : float, default 2.0
        표준편차 배수
        
    Returns
    -------
    upper : pd.Series
        상단 밴드 (중심선 + num_std * 표준편차)
    middle : pd.Series
        중심선 (이동평균)
    lower : pd.Series
        하단 밴드 (중심선 - num_std * 표준편차)
        
    Examples
    --------
    >>> upper, middle, lower = calc_bollinger(df['close'])
    >>> # 상단 밴드 돌파: 과매수 가능성
    >>> # 하단 밴드 터치: 과매도 가능성
    """
    middle = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    
    upper = middle + num_std * std
    lower = middle - num_std * std
    
    return upper, middle, lower


# ---------------------------------------------------------------------
# ATR (Average True Range)
# ---------------------------------------------------------------------

def calc_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    ATR (Average True Range) 계산
    
    변동성 측정 지표
    
    Parameters
    ----------
    high : pd.Series
        고가
    low : pd.Series
        저가
    close : pd.Series
        종가
    period : int, default 14
        ATR 계산 기간
        
    Returns
    -------
    pd.Series
        ATR 값
        
    Note
    ----
    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = True Range의 이동평균
    """
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


# ---------------------------------------------------------------------
# Stochastic Oscillator (스토캐스틱)
# ---------------------------------------------------------------------

def calc_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator (스토캐스틱) 계산
    
    과매수/과매도 판단 지표
    
    Parameters
    ----------
    high : pd.Series
        고가
    low : pd.Series
        저가
    close : pd.Series
        종가
    k_period : int, default 14
        %K 계산 기간
    d_period : int, default 3
        %D (시그널) 계산 기간
        
    Returns
    -------
    k : pd.Series
        %K 선 (Fast Stochastic)
    d : pd.Series
        %D 선 (Slow Stochastic, %K의 이동평균)
        
    Note
    ----
    %K = 100 * (현재가 - 최저가) / (최고가 - 최저가)
    %D = %K의 이동평균
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    # 0 나누기 방지
    denominator = highest_high - lowest_low
    denominator = denominator.replace(0, np.nan)
    
    k = 100 * (close - lowest_low) / denominator
    d = k.rolling(window=d_period).mean()
    
    return k, d


# ---------------------------------------------------------------------
# Volume-based Indicators (거래량 지표)
# ---------------------------------------------------------------------

def calc_volume_ratio(
    volume: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    거래량 비율 계산
    
    현재 거래량 / 평균 거래량
    
    Parameters
    ----------
    volume : pd.Series
        거래량
    window : int, default 20
        평균 계산 기간
        
    Returns
    -------
    pd.Series
        거래량 비율 (1.0 = 평균, 2.0 = 2배)
    """
    avg_volume = volume.rolling(window=window).mean()
    return volume / (avg_volume + 1e-9)


def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    OBV (On-Balance Volume) 계산
    
    가격과 거래량의 관계를 이용한 추세 지표
    
    Parameters
    ----------
    close : pd.Series
        종가
    volume : pd.Series
        거래량
        
    Returns
    -------
    pd.Series
        OBV 값 (누적 거래량)
        
    Note
    ----
    - 종가 상승: +거래량
    - 종가 하락: -거래량
    - 종가 동일: 0
    """
    price_diff = close.diff()
    
    # 방향 결정
    direction = np.where(price_diff > 0, 1, np.where(price_diff < 0, -1, 0))
    
    # OBV = 누적 (방향 * 거래량)
    obv = (direction * volume).cumsum()
    
    return obv


# ---------------------------------------------------------------------
# Usage Example (Documentation)
# ---------------------------------------------------------------------
"""
# 단일 종목 사용
import pandas as pd
from src.features.technical import calc_rsi, calc_macd, calc_bollinger

df = pd.read_csv("stock_data.csv")
df['rsi'] = calc_rsi(df['close'], period=14)

macd, signal, hist = calc_macd(df['close'])
df['macd'] = macd
df['macd_signal'] = signal
df['macd_hist'] = hist

# 다중 종목 그룹 연산
df['rsi'] = df.groupby('ticker')['close'].transform(
    lambda x: calc_rsi(x, 14)
)

# Bollinger Bands
bollinger_results = []
for ticker, group in df.groupby('ticker'):
    upper, middle, lower = calc_bollinger(group['close'])
    temp = pd.DataFrame({
        'ticker': ticker,
        'date': group['date'].values,
        'bb_upper': upper.values,
        'bb_middle': middle.values,
        'bb_lower': lower.values
    })
    bollinger_results.append(temp)

bb_df = pd.concat(bollinger_results)
df = df.merge(bb_df, on=['ticker', 'date'])
"""