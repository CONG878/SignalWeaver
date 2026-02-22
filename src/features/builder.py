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
    print("\n🔨 Building Features (Scale-Invariant)...")
    df = df.copy().sort_values(['ticker', 'date']).reset_index(drop=True)
    params = config['preprocessing']
    
    # 1. 이동평균 이격도 (MA Disparity)
    for window in params['technical_windows']:
        ma_series = df.groupby('ticker')['close'].transform(lambda x: calc_sma(x, window))
        # 단순 이동평균 가격이 아닌, 종가 대비 비율(-1.0 ~ )로 무차원화
        df[f'feature_ma_{window}_disparity'] = (df['close'] / ma_series) - 1.0
    
    # 2. 변동성 (20일 표준편차 - 이미 수익률 기반이므로 무차원)
    df['feature_volatility_20'] = df.groupby('ticker')['close'].transform(
        lambda x: x.pct_change().rolling(20).std()
    )
    
    # 3. 거래량 비율 (이미 비율이므로 무차원)
    df['feature_volume_ratio'] = df.groupby('ticker')['volume'].transform(
        lambda x: calc_volume_ratio(x, params['volume_window'])
    )
    
    # 4. RSI (0~100 스케일 고정)
    df['feature_rsi_14'] = df.groupby('ticker')['close'].transform(
        lambda x: calc_rsi(x, params['rsi_period'])
    )
    
    # 5. MACD (그룹별 계산 후 병합 - MACD 오실레이터는 일단 현행 유지)
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
    
    # 6. 볼린저 밴드 -> %B 및 Bandwidth 로 무차원화 변환
    bb_results = []
    for ticker, group in df.groupby('ticker'):
        upper, mid, lower = calc_bollinger(group['close'])
        
        # %B: 주가가 밴드 내 어느 위치에 있는지 (0=하단, 1=상단)
        pct_b = (group['close'] - lower) / (upper - lower + 1e-9)
        # Bandwidth: 밴드의 폭 (변동성)
        width = (upper - lower) / mid
        
        temp = pd.DataFrame({
            'ticker': ticker,
            'date': group['date'].values,
            'feature_bb_pct_b': pct_b.values,
            'feature_bb_width': width.values
        })
        bb_results.append(temp)
    df = df.merge(pd.concat(bb_results), on=['ticker', 'date'], how='left')
    
    return df

def build_universe_meta(df: pd.DataFrame) -> pd.DataFrame:
    """운영 판단용 메타 지표 생성 및 Feature 병행 생성"""
    print("🏛️ Building Universe Meta & Log Features...")
    df = df.copy()
    
    # 유동성 점수 (20일 평균 거래대금 추정) - 운영 필터용 원본 유지
    df['liquidity_score'] = df['close'] * df['volume']
    df['liquidity_score'] = df.groupby('ticker')['liquidity_score'].transform(
        lambda x: x.rolling(20).mean()
    )
    
    # 🌟 피처용 로그 변환 변수 추가 (스케일 이펙트 완화)
    # feature_ 접두사를 붙여 03단계에서 모델 학습에 자동 포함되도록 함
    df['feature_log_liquidity'] = np.log1p(df['liquidity_score'])
    
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

# [save_processed_data 및 filter_by_history 함수는 기존 코드와 동일하게 유지]
def save_processed_data(df: pd.DataFrame, config: dict, ticker_name_map: dict = None, paths = None):
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