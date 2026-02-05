"""
Hard filtering utilities for Korean stock market

한국 시장 특수성 반영:
- 상장폐지/관리종목
- 거래정지
- 작전주/테마주 (급등 + 거래량 폭증)
- 저가주 (동전주) 위험
- 유동성 기준
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


# ==========================================
# 1. 거래 가능성 필터 (Tradability)
# ==========================================

def filter_tradability(
    df: pd.DataFrame,
    meta_df: pd.DataFrame,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    거래 불가능 종목 제거
    
    Parameters
    ----------
    df : pd.DataFrame
        평가 대상 종목 데이터
    meta_df : pd.DataFrame
        메타 정보 (is_suspended, is_delisted 포함)
    verbose : bool
        필터링 로그 출력 여부
        
    Returns
    -------
    df_filtered : pd.DataFrame
        필터링된 데이터
    stats : dict
        필터링 통계
        
    Notes
    -----
    제거 대상:
    - is_suspended = 1 (거래정지)
    - is_delisted = 1 (상장폐지)
    
    Phase 2 설계서: "실전에서 매수 가능 후보에서만 제외"
    """
    initial_count = len(df)
    
    # 메타 정보 병합
    df = df.merge(
        meta_df[['ticker', 'is_suspended', 'is_delisted']],
        on='ticker',
        how='left'
    )
    
    # 결측값 처리 (메타 없으면 안전한 것으로 간주)
    df['is_suspended'] = df['is_suspended'].fillna(0).astype(int)
    df['is_delisted'] = df['is_delisted'].fillna(0).astype(int)
    
    # 필터링
    suspended_count = (df['is_suspended'] == 1).sum()
    delisted_count = (df['is_delisted'] == 1).sum()
    
    df_filtered = df[
        (df['is_suspended'] == 0) & 
        (df['is_delisted'] == 0)
    ].copy()
    
    stats = {
        'initial': initial_count,
        'suspended': suspended_count,
        'delisted': delisted_count,
        'removed': initial_count - len(df_filtered),
        'remaining': len(df_filtered)
    }
    
    if verbose:
        print(f"\n[거래 가능성 필터]")
        print(f"   - 초기 종목: {stats['initial']:,}개")
        print(f"   - 거래정지: {stats['suspended']:,}개 제거")
        print(f"   - 상장폐지: {stats['delisted']:,}개 제거")
        print(f"   ✅ 남은 종목: {stats['remaining']:,}개")
    
    return df_filtered, stats


# ==========================================
# 2. 작전주/테마주 필터 (Manipulation Detection)
# ==========================================

def detect_manipulation_risk(
    df_ticker: pd.DataFrame,
    surge_threshold: float = 1.0,      # 20일 내 100% 급등
    volume_multiplier: float = 5.0,     # 평균 대비 5배 폭증
    price_lookback: int = 20,
    volume_lookback: int = 60
) -> bool:
    """
    단일 종목의 작전주/테마주 혐의 탐지
    
    Parameters
    ----------
    df_ticker : pd.DataFrame
        단일 종목 시계열 (date, close, volume 포함)
    surge_threshold : float
        급등 기준 (1.0 = 100%)
    volume_multiplier : float
        거래량 폭증 기준 (배수)
    price_lookback : int
        급등 판단 기간
    volume_lookback : int
        평균 거래량 계산 기간
        
    Returns
    -------
    bool
        True = 작전주 혐의, False = 정상
        
    Notes
    -----
    Phase 2 설계서 2-1-2 "사후 탐지" 로직:
    - 20일 내 +100% 이상 급등
    - 거래량/거래대금 60일 평균 대비 5배 이상 폭증
    """
    if len(df_ticker) < max(price_lookback, volume_lookback):
        return False  # 데이터 부족
    
    # 최근 가격만 확인 (예측값 기준)
    recent_prices = df_ticker.tail(price_lookback)['pred_close'].values
    
    if len(recent_prices) < 2:
        return False
    
    # 1. 급등 체크 (최근 20일)
    price_change_ratio = recent_prices[-1] / recent_prices[0] - 1
    is_surge = price_change_ratio > surge_threshold
    
    # 2. 거래량 폭증 체크 (메타 데이터가 있다면)
    if 'volume' in df_ticker.columns:
        recent_volume = df_ticker.tail(20)['volume'].mean()
        avg_volume = df_ticker.tail(volume_lookback)['volume'].mean()
        
        if avg_volume > 0:
            volume_ratio = recent_volume / avg_volume
            is_volume_surge = volume_ratio > volume_multiplier
        else:
            is_volume_surge = False
    else:
        is_volume_surge = False  # 거래량 데이터 없으면 판단 불가
    
    # 두 조건 모두 충족 시 작전주 의심
    return is_surge and is_volume_surge


def filter_manipulation(
    df: pd.DataFrame,
    df_future: pd.DataFrame,
    surge_threshold: float = 1.0,
    volume_multiplier: float = 5.0,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    작전주/테마주 필터링
    
    Parameters
    ----------
    df : pd.DataFrame
        평가 대상 종목 데이터
    df_future : pd.DataFrame
        예측 시계열 (종목별 pred_close, volume 포함)
    surge_threshold : float
        급등 기준
    volume_multiplier : float
        거래량 폭증 기준
    verbose : bool
        로그 출력 여부
        
    Returns
    -------
    df_filtered : pd.DataFrame
        필터링된 데이터
    stats : dict
        필터링 통계
    """
    initial_count = len(df)
    manipulation_tickers = []
    
    for ticker in df['ticker'].unique():
        ticker_data = df_future[
            df_future['ticker'] == ticker
        ].sort_values('date').reset_index(drop=True)
        
        if detect_manipulation_risk(
            ticker_data,
            surge_threshold=surge_threshold,
            volume_multiplier=volume_multiplier
        ):
            manipulation_tickers.append(ticker)
    
    # 제거
    df_filtered = df[~df['ticker'].isin(manipulation_tickers)].copy()
    
    stats = {
        'initial': initial_count,
        'manipulation': len(manipulation_tickers),
        'removed': len(manipulation_tickers),
        'remaining': len(df_filtered)
    }
    
    if verbose:
        print(f"\n[작전주/테마주 필터]")
        print(f"   - 초기 종목: {stats['initial']:,}개")
        print(f"   - 작전주 혐의: {stats['manipulation']:,}개 제거")
        print(f"   ✅ 남은 종목: {stats['remaining']:,}개")
        
        if len(manipulation_tickers) > 0 and len(manipulation_tickers) <= 10:
            print(f"   ⚠️  제거된 종목: {manipulation_tickers}")
    
    return df_filtered, stats


# ==========================================
# 3. 저가주 필터 (Penny Stock)
# ==========================================

def filter_penny_stocks(
    df: pd.DataFrame,
    df_future: pd.DataFrame,
    min_price: float = 1000.0,  # 최소 1,000원
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    저가주 (동전주) 필터링
    
    Parameters
    ----------
    df : pd.DataFrame
        평가 대상 종목 데이터
    df_future : pd.DataFrame
        예측 시계열
    min_price : float
        최소 가격 기준 (원)
    verbose : bool
        로그 출력 여부
        
    Returns
    -------
    df_filtered : pd.DataFrame
        필터링된 데이터
    stats : dict
        필터링 통계
        
    Notes
    -----
    한국 시장 특성:
    - 1,000원 미만: 극심한 변동성, 작전 가능성
    - 거래소 규정: 500원 미만 관리종목 지정 가능
    """
    initial_count = len(df)
    
    # 종목별 최근 평균 가격 계산
    ticker_avg_prices = []
    
    for ticker in df['ticker'].unique():
        ticker_data = df_future[df_future['ticker'] == ticker]
        
        # 예측 가격 평균
        avg_price = np.exp(ticker_data['pred_log_close'].mean())
        
        ticker_avg_prices.append({
            'ticker': ticker,
            'avg_pred_price': avg_price
        })
    
    df_prices = pd.DataFrame(ticker_avg_prices)
    
    # 필터링
    valid_tickers = df_prices[
        df_prices['avg_pred_price'] >= min_price
    ]['ticker'].values
    
    df_filtered = df[df['ticker'].isin(valid_tickers)].copy()
    
    stats = {
        'initial': initial_count,
        'penny_stocks': initial_count - len(df_filtered),
        'removed': initial_count - len(df_filtered),
        'remaining': len(df_filtered)
    }
    
    if verbose:
        print(f"\n[저가주 필터]")
        print(f"   - 초기 종목: {stats['initial']:,}개")
        print(f"   - {min_price:,.0f}원 미만: {stats['penny_stocks']:,}개 제거")
        print(f"   ✅ 남은 종목: {stats['remaining']:,}개")
    
    return df_filtered, stats


# ==========================================
# 4. 유동성 필터 (Liquidity)
# ==========================================

def filter_liquidity(
    df: pd.DataFrame,
    meta_df: pd.DataFrame,
    min_liquidity: float = 50_000_000,  # 5천만 원
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    유동성 필터링
    
    Parameters
    ----------
    df : pd.DataFrame
        평가 대상 종목 데이터
    meta_df : pd.DataFrame
        메타 정보 (liquidity_score 포함)
    min_liquidity : float
        최소 유동성 점수 (20일 평균 거래대금)
    verbose : bool
        로그 출력 여부
        
    Returns
    -------
    df_filtered : pd.DataFrame
        필터링된 데이터
    stats : dict
        필터링 통계
        
    Notes
    -----
    Phase 2 설계서 2-1-5:
    - 최소 기준: 5천만 원 (소액 계좌 기준)
    - 계좌 규모별 조정 가능
    """
    initial_count = len(df)
    
    # 메타 정보 병합
    df = df.merge(
        meta_df[['ticker', 'liquidity_score']],
        on='ticker',
        how='left'
    )
    
    # 결측값 처리
    df['liquidity_score'] = df['liquidity_score'].fillna(0)
    
    # 필터링
    df_filtered = df[df['liquidity_score'] >= min_liquidity].copy()
    
    stats = {
        'initial': initial_count,
        'low_liquidity': initial_count - len(df_filtered),
        'removed': initial_count - len(df_filtered),
        'remaining': len(df_filtered),
        'min_liquidity': min_liquidity
    }
    
    if verbose:
        print(f"\n[유동성 필터]")
        print(f"   - 초기 종목: {stats['initial']:,}개")
        print(f"   - {min_liquidity:,.0f}원 미만: {stats['low_liquidity']:,}개 제거")
        print(f"   ✅ 남은 종목: {stats['remaining']:,}개")
    
    return df_filtered, stats


# ==========================================
# 5. 통합 하드 필터 (All-in-One)
# ==========================================

def apply_hard_filters(
    df: pd.DataFrame,
    df_future: pd.DataFrame,
    meta_df: pd.DataFrame,
    config: Dict = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    모든 하드 필터를 순차적으로 적용
    
    Parameters
    ----------
    df : pd.DataFrame
        평가 대상 종목 데이터
    df_future : pd.DataFrame
        예측 시계열
    meta_df : pd.DataFrame
        메타 정보
    config : dict, optional
        필터링 설정 (없으면 기본값 사용)
    verbose : bool
        로그 출력 여부
        
    Returns
    -------
    df_filtered : pd.DataFrame
        최종 필터링된 데이터
    all_stats : dict
        각 필터별 통계
        
    Examples
    --------
    >>> df_final, stats = apply_hard_filters(
    ...     df_candidates,
    ...     df_future,
    ...     df_meta_latest,
    ...     verbose=True
    ... )
    >>> print(f"최종 유니버스: {len(df_final)} 종목")
    """
    if config is None:
        config = {
            'min_liquidity': 50_000_000,      # 5천만 원
            'min_price': 1000.0,              # 1천 원
            'surge_threshold': 1.0,           # 100% 급등
            'volume_multiplier': 5.0          # 5배 폭증
        }
    
    if verbose:
        print("\n" + "=" * 65)
        print("🚧 하드 필터링 시작")
        print("=" * 65)
        print(f"   초기 종목 수: {len(df):,}개")
    
    all_stats = {}
    
    # 1. 거래 가능성 (거래정지, 상장폐지)
    df, stats1 = filter_tradability(df, meta_df, verbose=verbose)
    all_stats['tradability'] = stats1
    
    # 2. 작전주/테마주
    df, stats2 = filter_manipulation(
        df, df_future,
        surge_threshold=config['surge_threshold'],
        volume_multiplier=config['volume_multiplier'],
        verbose=verbose
    )
    all_stats['manipulation'] = stats2
    
    # 3. 저가주
    df, stats3 = filter_penny_stocks(
        df, df_future,
        min_price=config['min_price'],
        verbose=verbose
    )
    all_stats['penny_stocks'] = stats3
    
    # 4. 유동성
    df, stats4 = filter_liquidity(
        df, meta_df,
        min_liquidity=config['min_liquidity'],
        verbose=verbose
    )
    all_stats['liquidity'] = stats4
    
    # 최종 요약
    if verbose:
        print("\n" + "=" * 65)
        print("✅ 하드 필터링 완료")
        print("=" * 65)
        total_removed = all_stats['tradability']['initial'] - len(df)
        removal_rate = total_removed / all_stats['tradability']['initial'] * 100
        
        print(f"   - 초기 종목: {all_stats['tradability']['initial']:,}개")
        print(f"   - 총 제거: {total_removed:,}개 ({removal_rate:.1f}%)")
        print(f"   - 최종 유니버스: {len(df):,}개")
        
        print(f"\n[제거 세부 내역]")
        print(f"   1. 거래정지/상폐: {stats1['removed']:,}개")
        print(f"   2. 작전주/테마주: {stats2['removed']:,}개")
        print(f"   3. 저가주: {stats3['removed']:,}개")
        print(f"   4. 저유동성: {stats4['removed']:,}개")
    
    return df, all_stats


# ==========================================
# 6. 필터링 결과 분석
# ==========================================

def analyze_filter_impact(
    all_stats: Dict[str, Dict],
    save_path: str = None
) -> pd.DataFrame:
    """
    필터링 영향 분석 리포트 생성
    
    Parameters
    ----------
    all_stats : dict
        apply_hard_filters의 출력
    save_path : str, optional
        CSV 저장 경로
        
    Returns
    -------
    pd.DataFrame
        필터별 영향 분석표
    """
    records = []
    
    for filter_name, stats in all_stats.items():
        records.append({
            'filter': filter_name,
            'initial': stats['initial'],
            'removed': stats['removed'],
            'remaining': stats['remaining'],
            'removal_rate_%': stats['removed'] / stats['initial'] * 100
        })
    
    df_report = pd.DataFrame(records)
    
    if save_path:
        df_report.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"✅ 필터링 리포트 저장: {save_path}")
    
    return df_report
