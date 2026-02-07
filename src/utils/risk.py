"""
Risk assessment utilities

종목 자체의 내재적 위험 평가 지표
- 모델 예측 오차와 독립적
- 시계열 데이터만으로 계산 가능
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def calculate_risk_metrics(
    prices: np.ndarray,
    returns: Optional[np.ndarray] = None,
    is_log_prices: bool = True
) -> Dict[str, float]:
    """
    종목의 내재적 위험 지표 계산 (5개 표준 지표)
    
    Parameters
    ----------
    prices : np.ndarray
        가격 시계열 (로그 가격 또는 일반 가격)
    returns : np.ndarray, optional
        로그 수익률 (제공되지 않으면 자동 계산)
    is_log_prices : bool, default=True
        prices가 로그 변환되었는지 여부
        
    Returns
    -------
    dict : 위험 지표
        - volatility: 변동성 (로그 수익률 표준편차)
        - downside_risk: 하방 위험 (음수 수익률만)
        - var_95: VaR (5% 분위수)
        - cvar_95: CVaR (최악 5% 평균)
        - max_drawdown: 최대 낙폭
        - skewness: 비대칭도 (음수면 하락 쏠림)
        - kurtosis: 첨도 (높으면 극단값 빈번)
        
    Notes
    -----
    모든 지표는 **예측값 시계열**에 대해 계산되어야 함
    → 미래 위험을 측정하는 것이 목적
    
    Examples
    --------
    >>> log_prices = np.array([7.5, 7.52, 7.48, 7.55, 7.53])
    >>> risk = calculate_risk_metrics(log_prices, is_log_prices=True)
    >>> print(f"변동성: {risk['volatility']:.4f}")
    """
    # 로그 수익률 계산
    if returns is None:
        if is_log_prices:
            # 이미 로그 가격 → 차분
            returns = np.diff(prices)
        else:
            # 일반 가격 → 로그 변환 후 차분
            returns = np.diff(np.log(prices))
    
    if len(returns) < 2:
        # 데이터 부족 시 NaN 반환
        return {
            'volatility': np.nan,
            'downside_risk': np.nan,
            'var_95': np.nan,
            'cvar_95': np.nan,
            'max_drawdown': np.nan,
            'skewness': np.nan,
            'kurtosis': np.nan
        }
    
    # ==========================================
    # 1. 기본 변동성 (Volatility)
    # ==========================================
    volatility = np.std(returns, ddof=1)
    
    # ==========================================
    # 2. 하방 위험 (Downside Risk)
    # ==========================================
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 1:
        downside_risk = np.std(downside_returns, ddof=1)
    else:
        downside_risk = 0.0  # 손실 없음
    
    # ==========================================
    # 3. VaR (Value at Risk, 95% 신뢰수준)
    # ==========================================
    var_95 = np.percentile(returns, 5)
    
    # ==========================================
    # 4. CVaR (Conditional VaR, Expected Shortfall)
    # ==========================================
    # 최악 5% 평균
    worst_returns = returns[returns <= var_95]
    if len(worst_returns) > 0:
        cvar_95 = np.mean(worst_returns)
    else:
        cvar_95 = var_95
    
    # ==========================================
    # 5. 최대 낙폭 (Maximum Drawdown)
    # ==========================================
    cumulative_returns = np.cumsum(returns)
    cumulative_wealth = np.exp(cumulative_returns)  # 로그→일반
    
    # 누적 최고점
    running_max = np.maximum.accumulate(cumulative_wealth)
    
    # Drawdown 계산
    drawdown = (cumulative_wealth - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # ==========================================
    # 6. 비대칭도 (Skewness)
    # ==========================================
    # 음수: 하락 쏠림 (왼쪽 꼬리 길음)
    # 양수: 상승 쏠림 (오른쪽 꼬리 길음)
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    
    if std_return > 0:
        skewness = np.mean(((returns - mean_return) / std_return) ** 3)
    else:
        skewness = 0.0
    
    # ==========================================
    # 7. 첨도 (Kurtosis, Excess Kurtosis)
    # ==========================================
    # 높을수록 극단값(Fat Tail) 빈번
    # 정규분포 기준값 = 3, Excess = Kurtosis - 3
    if std_return > 0:
        kurtosis = np.mean(((returns - mean_return) / std_return) ** 4)
        excess_kurtosis = kurtosis - 3  # Fisher's definition
    else:
        excess_kurtosis = 0.0
    
    return {
        'volatility': float(volatility),
        'downside_risk': float(downside_risk),
        'var_95': float(var_95),
        'cvar_95': float(cvar_95),
        'max_drawdown': float(max_drawdown),
        'skewness': float(skewness),
        'kurtosis': float(excess_kurtosis)  # Excess Kurtosis 저장
    }


def calculate_composite_risk_score(risk_metrics: Dict[str, float]) -> float:
    """
    복합 리스크 스코어 계산 (가중 평균)
    
    높을수록 위험함 (0~1 스케일)
    
    Parameters
    ----------
    risk_metrics : dict
        calculate_risk_metrics의 출력
        
    Returns
    -------
    float
        복합 리스크 점수 [0, 1]
        - 0: 매우 안전
        - 1: 매우 위험
        
    Notes
    -----
    가중치 설계:
    - Volatility (30%): 기본 변동성
    - Downside Risk (25%): 하방 위험 (투자자가 민감)
    - CVaR (20%): 극단 손실
    - MDD (15%): 심리적 타격
    - Kurtosis (10%): Fat Tail 위험
    
    Skewness는 방향성이므로 별도 처리
    """
    # NaN 체크
    if any(np.isnan(v) for v in risk_metrics.values()):
        return np.nan
    
    # 각 지표를 [0, 1] 범위로 정규화 (임시 - 나중에 전체 종목 기준으로 재정규화)
    # 여기서는 가중합만 계산
    
    # 위험 증가 방향으로 통일
    components = {
        'volatility': risk_metrics['volatility'],
        'downside_risk': risk_metrics['downside_risk'],
        'cvar_abs': abs(risk_metrics['cvar_95']),  # 음수이므로 절댓값
        'mdd_abs': abs(risk_metrics['max_drawdown']),  # 음수이므로 절댓값
        'kurtosis_pos': max(risk_metrics['kurtosis'], 0)  # 양수만 (Fat Tail)
    }
    
    # 가중 평균 (정규화 전 원점수)
    weights = {
        'volatility': 0.30,
        'downside_risk': 0.25,
        'cvar_abs': 0.20,
        'mdd_abs': 0.15,
        'kurtosis_pos': 0.10
    }
    
    raw_score = sum(components[k] * weights[k] for k in weights.keys())
    
    return raw_score


def calculate_risk_metrics_for_ticker(
    df_ticker: pd.DataFrame,
    price_col: str = 'pred_close',
    is_log: bool = False
) -> Dict[str, float]:
    """
    단일 종목의 DataFrame에서 위험 지표 계산 (편의 함수)
    
    Parameters
    ----------
    df_ticker : pd.DataFrame
        단일 종목 데이터 (date로 정렬 필요)
    price_col : str
        가격 컬럼명 ('pred_close' 또는 'pred_log_close')
    is_log : bool
        해당 컬럼이 로그 변환되었는지 여부
        
    Returns
    -------
    dict
        위험 지표
    """
    prices = df_ticker[price_col].values
    
    return calculate_risk_metrics(prices, is_log_prices=is_log)


# ==========================================
# Annualization helpers
# ==========================================

def annualize_volatility(daily_vol: float, trading_days: int = 252) -> float:
    """
    일별 변동성 → 연율화
    
    Parameters
    ----------
    daily_vol : float
        일별 로그 수익률 표준편차
    trading_days : int
        연간 영업일 수 (미국: 252, 한국: ~250)
        
    Returns
    -------
    float
        연율화 변동성
        
    Notes
    -----
    σ_annual = σ_daily × √T
    """
    return daily_vol * np.sqrt(trading_days)


def annualize_return(daily_log_return: float, trading_days: int = 252) -> float:
    """
    일평균 로그 수익률 → 연율화
    
    Parameters
    ----------
    daily_log_return : float
        일평균 로그 수익률
    trading_days : int
        연간 영업일 수
        
    Returns
    -------
    float
        연율화 수익률 (로그)
    """
    return daily_log_return * trading_days


# ==========================================
# Risk-Adjusted Return Metrics
# ==========================================

def calculate_sharpe_ratio(
    mean_return: float,
    volatility: float,
    risk_free_rate: float = 0.03
) -> float:
    """
    Sharpe Ratio 계산 (위험 조정 수익률)
    
    Parameters
    ----------
    mean_return : float
        연율화 평균 수익률
    volatility : float
        연율화 변동성
    risk_free_rate : float
        무위험 수익률 (연율, 한국 국고채 3년물 ~3%)
        
    Returns
    -------
    float
        Sharpe Ratio = (수익률 - 무위험) / 변동성
        
    Notes
    -----
    > 1.0: 우수
    > 2.0: 매우 우수
    > 3.0: 탁월
    """
    if volatility == 0:
        return np.nan
    
    return (mean_return - risk_free_rate) / volatility


def calculate_sortino_ratio(
    mean_return: float,
    downside_risk: float,
    risk_free_rate: float = 0.03
) -> float:
    """
    Sortino Ratio 계산 (하방 위험 조정 수익률)
    
    Sharpe보다 실용적 (상승 변동성은 페널티 없음)
    
    Parameters
    ----------
    mean_return : float
        연율화 평균 수익률
    downside_risk : float
        연율화 하방 위험
    risk_free_rate : float
        무위험 수익률
        
    Returns
    -------
    float
        Sortino Ratio = (수익률 - 무위험) / 하방위험
    """
    if downside_risk == 0:
        return np.nan
    
    return (mean_return - risk_free_rate) / downside_risk


# ==========================================
# Normalization & Ranking
# ==========================================

def normalize_risk_scores(
    df: pd.DataFrame,
    score_col: str = 'risk_composite_raw',
    method: str = 'minmax'
) -> pd.DataFrame:
    """
    위험 점수 정규화 및 순위 부여
    
    Parameters
    ----------
    df : pd.DataFrame
        위험 지표가 포함된 DataFrame
    score_col : str
        정규화할 원본 점수 컬럼명
    method : str, default='minmax'
        정규화 방법 ('minmax', 'zscore')
        
    Returns
    -------
    pd.DataFrame
        정규화된 점수 및 순위 컬럼 추가됨
        - risk_score_normalized: [0, 1] 범위, 높을수록 위험
        - risk_rank: 위험 순위 (1 = 가장 안전)
    """
    df = df.copy()
    
    if method == 'minmax':
        # Min-Max Scaling: [0, 1]
        min_val = df[score_col].min()
        max_val = df[score_col].max()
        
        if max_val == min_val:
            df['risk_score_normalized'] = 0.5
        else:
            df['risk_score_normalized'] = (
                (df[score_col] - min_val) / (max_val - min_val)
            )
    
    elif method == 'zscore':
        # Z-Score 정규화 후 Sigmoid
        mean_val = df[score_col].mean()
        std_val = df[score_col].std()
        
        if std_val == 0:
            df['risk_score_normalized'] = 0.5
        else:
            z_scores = (df[score_col] - mean_val) / std_val
            # Sigmoid: (-inf, +inf) → (0, 1)
            df['risk_score_normalized'] = 1 / (1 + np.exp(-z_scores))
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # 순위 부여 (낮은 위험 = 1위)
    df['risk_rank'] = df[score_col].rank(method='min')
    
    return df