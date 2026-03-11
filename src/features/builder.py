"""
Feature & Meta Builder Module

01단계의 Raw 데이터를 가공하여 모델 학습용 피처와 시스템 운영 지표를 생성합니다.
v3.6.0 원칙에 따라 가격 스케일에 종속되지 않는 무차원(Scale-Invariant) 지표를 주로 생성합니다.

주요 생성 그룹:
    1. 무차원 기술적 지표 (이동평균 이격도, 변동성, %B, RSI 등)
    2. 메타 지표 (유동성 점수, 리스크 팩터, 상장 폐지/거래 정지 플래그)
    3. 스케일 보정 피처 (log_liquidity 등 모델에 직접 투입되는 메타 파생 피처)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.features.technical import (
    calc_rsi, calc_macd, calc_bollinger, calc_sma, calc_volume_ratio
)

def build_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Scale-Invariant 기술적 지표 모음 생성"""
    print("\n🔨 Building Features (Scale-Invariant)...")
    df = df.copy().sort_values(['ticker', 'date']).reset_index(drop=True)
    params = config['preprocessing']
    
    # 1. 이동평균 이격도 (MA Disparity)
    for window in params['technical_windows']:
        ma_series = df.groupby('ticker')['close'].transform(lambda x: calc_sma(x, window))
        df[f'feature_ma_{window}_disparity'] = (df['close'] / ma_series) - 1.0
    
    # 2. 변동성 및 거래량 비율
    df['feature_volatility_20'] = df.groupby('ticker')['close'].transform(
        lambda x: x.pct_change().rolling(20).std()
    )
    df['feature_volume_ratio'] = df.groupby('ticker')['volume'].transform(
        lambda x: calc_volume_ratio(x, params['volume_window'])
    )
    
    # 3. RSI
    df['feature_rsi_14'] = df.groupby('ticker')['close'].transform(
        lambda x: calc_rsi(x, params['rsi_period'])
    )
    
    # 4. MACD
    macd_results = []
    for ticker, group in df.groupby('ticker'):
        macd, signal, hist = calc_macd(group['close'])
        macd_results.append(pd.DataFrame({
            'ticker': ticker,
            'date': group['date'].values,
            'feature_macd': macd.values,
            'feature_macd_signal': signal.values,
            'feature_macd_hist': hist.values
        }))
    df = df.merge(pd.concat(macd_results), on=['ticker', 'date'], how='left')
    
    # 5. 볼린저 밴드 (%B 및 Bandwidth)
    bb_results = []
    for ticker, group in df.groupby('ticker'):
        upper, mid, lower = calc_bollinger(group['close'])
        pct_b = (group['close'] - lower) / (upper - lower + 1e-9)
        width = (upper - lower) / mid
        
        bb_results.append(pd.DataFrame({
            'ticker': ticker,
            'date': group['date'].values,
            'feature_bb_pct_b': pct_b.values,
            'feature_bb_width': width.values
        }))
    df = df.merge(pd.concat(bb_results), on=['ticker', 'date'], how='left')
    
    return df


def build_universe_meta(df: pd.DataFrame) -> pd.DataFrame:
    """운영 판단용 메타 지표 및 관련 피처 생성"""
    print("🏛️ Building Universe Meta & Log Features...")
    df = df.copy()
    
    # 유동성 점수 (운영용) 및 로그 유동성 (학습용)
    df['liquidity_score'] = df['close'] * df['volume']
    df['liquidity_score'] = df.groupby('ticker')['liquidity_score'].transform(
        lambda x: x.rolling(20).mean()
    )
    df['feature_log_liquidity'] = np.log1p(df['liquidity_score'])
    
    # 리스크 메타 지표 (0~1 정규화)
    df['risk_volatility'] = df['feature_volatility_20'].fillna(0)
    df['risk_volume_surge'] = (df['feature_volume_ratio'] > 3.0).astype(int)
    
    max_vol = df['risk_volatility'].max()
    df['risk_composite'] = (
        (df['risk_volatility'] / max_vol if max_vol > 0 else 0) * 0.5 +
        df['risk_volume_surge'] * 0.5
    )
    
    df['is_suspended'] = (df['volume'] == 0).astype(int)
    df['is_delisted'] = 0 
    
    return df


def save_processed_data(df: pd.DataFrame, config: dict, ticker_name_map: dict = None, paths = None):
    """최종 가공된 데이터 파켓(전체) 및 CSV(개별) 저장"""
    if paths is None:
        from src.utils.config import ProjectPaths
        paths = ProjectPaths.from_config(config)
        
    if config['preprocessing'].get('save_parquet', True):
        parquet_path = paths.get_dataset_parquet()
        df.to_parquet(parquet_path, compression='snappy', index=False)
        print(f"✅ Integrated Parquet Saved: {parquet_path}")
        
    if config['preprocessing'].get('save_csv', False):
        csv_dir = paths.get_processed_csv_dir()
        csv_dir.mkdir(parents=True, exist_ok=True)
        print(f"📂 Saving Debug CSVs to {csv_dir}...")
        for ticker, group in tqdm(df.groupby('ticker'), desc="Saving CSVs"):
            name = ticker_name_map.get(ticker, ticker) if ticker_name_map else ticker
            safe_name = str(name).replace('/', '_').replace('\\', '_')
            group.to_csv(csv_dir / f"{safe_name}.csv", index=False, encoding='utf-8-sig')


def filter_by_history(df: pd.DataFrame, min_history: int, threshold_ratio: float = 1.0) -> pd.DataFrame:
    """히스토리가 부족한 신규 상장 종목 필터링"""
    print(f"\n✂️  Filtering history (Min History: {min_history})...")
    df = df[df.groupby('ticker').cumcount() >= min_history].copy()
    counts = df.groupby('ticker')['date'].transform('count')
    
    max_length = counts.max()
    required_length = int(max_length * threshold_ratio)
    mask = counts >= required_length
    df_filtered = df[mask].copy()
    
    print(f"   - Max length found: {max_length}")
    print(f"   - Required length: {required_length} (Ratio: {threshold_ratio})")
    print(f"   - Tickers: {df['ticker'].nunique()} -> {df_filtered['ticker'].nunique()}")
    print(f"   - Total rows: {len(df):,} -> {len(df_filtered):,}")
    
    return df_filtered