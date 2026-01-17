"""
Purpose:
    - Raw 데이터를 가공하여 학습용 Feature 및 Meta 데이터셋 구축
    - 기술적 지표 계산 및 운영용 메타 정보(유동성, 리스크) 생성
    - 결과를 통합 Parquet 및 디버깅용 개별 CSV로 저장
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.features.technical import (
    calc_rsi, calc_macd, calc_bollinger, calc_sma, calc_volume_ratio
)

def build_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """기술적 지표 Feature 생성 로직"""
    print("\n🔨 Building Features...")
    df = df.copy().sort_values(['ticker', 'date']).reset_index(drop=True)
    params = config['preprocessing']
    
    # 1. 이동평균 (MA)
    for window in params['technical_windows']:
        df[f'feature_ma_{window}'] = df.groupby('ticker')['close'].transform(
            lambda x: calc_sma(x, window)
        )
    
    # 2. 변동성 (20일 표준편차)
    df['feature_volatility_20'] = df.groupby('ticker')['close'].transform(
        lambda x: x.pct_change().rolling(20).std()
    )
    
    # 3. 거래량 비율
    df['feature_volume_ratio'] = df.groupby('ticker')['volume'].transform(
        lambda x: calc_volume_ratio(x, params['volume_window'])
    )
    
    # 4. RSI
    df['feature_rsi_14'] = df.groupby('ticker')['close'].transform(
        lambda x: calc_rsi(x, params['rsi_period'])
    )
    
    # 5. MACD (그룹별 계산 후 병합)
    macd_results = []
    for ticker, group in df.groupby('ticker'):
        macd, signal, hist = calc_macd(group['close'])
        temp = pd.DataFrame({
            'ticker': ticker,
            'date': group['date'].values,
            'feature_macd': macd.values,
            'feature_macd_signal': signal.values,
            'feature_macd_hist': hist.values
        })
        macd_results.append(temp)
    df = df.merge(pd.concat(macd_results), on=['ticker', 'date'], how='left')
    
    # 6. Bollinger Bands
    bb_results = []
    for ticker, group in df.groupby('ticker'):
        upper, mid, lower = calc_bollinger(group['close'])
        temp = pd.DataFrame({
            'ticker': ticker,
            'date': group['date'].values,
            'feature_bb_upper': upper.values,
            'feature_bb_middle': mid.values,
            'feature_bb_lower': lower.values
        })
        bb_results.append(temp)
    df = df.merge(pd.concat(bb_results), on=['ticker', 'date'], how='left')
    
    return df

def build_universe_meta(df: pd.DataFrame) -> pd.DataFrame:
    """운영 판단용 메타 지표 생성"""
    print("🏛️ Building Universe Meta...")
    df = df.copy()
    
    # 유동성 점수 (20일 평균 거래대금 추정)
    df['liquidity_score'] = df['close'] * df['volume']
    df['liquidity_score'] = df.groupby('ticker')['liquidity_score'].transform(
        lambda x: x.rolling(20).mean()
    )
    
    # 리스크 점수 (변동성 기반)
    df['risk_volatility'] = df['feature_volatility_20'].fillna(0)
    df['risk_volume_surge'] = (df['feature_volume_ratio'] > 3.0).astype(int)
    
    # 복합 리스크 (0~1 정규화)
    max_vol = df['risk_volatility'].max()
    df['risk_composite'] = (
        (df['risk_volatility'] / max_vol if max_vol > 0 else 0) * 0.5 +
        df['risk_volume_surge'] * 0.5
    )
    
    # 거래 가능 여부 플래그
    df['is_suspended'] = (df['volume'] == 0).astype(int)
    df['is_delisted'] = 0  # Raw 데이터 기반 판단 필요 (현재 Placeholder)
    
    return df

def save_processed_data(df: pd.DataFrame, config: dict, ticker_name_map: dict = None):
    """결과 데이터를 날짜별 폴더에 저장 (Parquet + CSV)"""
    ref_date = config['project']['reference_date']
    base_dir = Path(config['paths']['processed_dir']) / ref_date
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 통합 Parquet 저장
    if config['preprocessing'].get('save_parquet', True):
        parquet_path = base_dir / "dataset.parquet"
        df.to_parquet(parquet_path, compression='snappy', index=False)
        print(f"✅ Integrated Parquet Saved: {parquet_path}")
        
    # 2. 개별 CSV 저장 (디버깅용)
    if config['preprocessing'].get('save_csv', False):
        csv_dir = base_dir / "csv"
        csv_dir.mkdir(exist_ok=True)
        print(f"📂 Saving Debug CSVs to {csv_dir}...")
        
        for ticker, group in tqdm(df.groupby('ticker'), desc="Saving CSVs"):
            name = ticker_name_map.get(ticker, ticker) if ticker_name_map else ticker
            safe_name = str(name).replace('/', '_').replace('\\', '_')
            group.to_csv(csv_dir / f"{safe_name}.csv", index=False, encoding='utf-8-sig')

def filter_by_history(df: pd.DataFrame, min_history: int, threshold_ratio: float = 1.0) -> pd.DataFrame:
    """
    종목별 이력을 필터링하고 데이터 길이를 표준화합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
    min_history : int
        기술적 지표 계산 등을 위해 제거할 초기 데이터 기간
    threshold_ratio : float (0.0 ~ 1.0)
        최장 기간 대비 유지할 최소 길이 비율. 1.0이면 기존처럼 '최장 길이와 일치'하는 종목만 유지.
    """
    print(f"\n✂️  Filtering history (Min History: {min_history})...")
    
    # 1. 초기 준비 기간(min_history) 제거 (벡터화 연산)
    # 각 종목 내에서 행 번호를 매기고 min_history 이후인 것만 선택
    df = df[df.groupby('ticker').cumcount() >= min_history].copy()
    
    # 2. 종목별 데이터 길이 계산
    counts = df.groupby('ticker')['date'].transform('count')
    max_length = counts.max()
    
    # 3. 길이 기반 필터링 (최장 길이의 특정 비율 이상만 유지)
    # threshold_ratio가 1.0이면 기존 로직과 동일하게 '완벽히 일치'하는 종목만 남음
    required_length = int(max_length * threshold_ratio)
    mask = counts >= required_length
    
    df_filtered = df[mask].copy()
    
    print(f"   - Max length found: {max_length}")
    print(f"   - Required length: {required_length} (Ratio: {threshold_ratio})")
    print(f"   - Tickers: {df['ticker'].nunique()} -> {df_filtered['ticker'].nunique()}")
    print(f"   - Total rows: {len(df):,} -> {len(df_filtered):,}")
    
    return df_filtered