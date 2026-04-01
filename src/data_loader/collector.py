"""
Data Collection Engine

KRX 주가 데이터 및 종목 마스터 정보를 수집합니다.

## 버전
- v4.0.0: change_pct 자동 검증 로직 추가.
          단순 등락률이면 'change_rate', 퍼센트 등락률이면 'change_pct'로 저장.
          산출물 파일명 변경 반영 (ProjectPaths v4.0.0 준수):
            krx_prices_{date}.parquet → prices.parquet
            ticker_master_{date}.csv  → ticker_master.csv
- v3.8.1: Fallback 강화 (3단계 우선순위)
- v3.8.0: ProjectPaths 기반 경로 관리 중앙화
"""

import time
import random
import math
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import FinanceDataReader as fdr
except ImportError:
    raise ImportError(
        "FinanceDataReader가 설치되지 않았습니다. 설치: pip install finance-datareader"
    )


# ==========================================
# 종목 리스트 조회
# ==========================================

def get_ticker_universe(reference_date: str) -> List[Tuple[str, str]]:
    """
    KRX 상장 종목 리스트 조회 (3단계 Fallback).

    우선순위:
    1. FinanceDataReader API
    2. data/01_raw/{reference_date}/stock_list.csv
    3. data/01_raw/{reference_date}/ticker_master.csv

    Returns
    -------
    List[Tuple[str, str]]
        (ticker_6digit, name) 튜플 리스트
    """
    print(f"🔍 KRX 전체 종목 조회 중 (기준일: {reference_date})...")

    # 1. FDR API
    try:
        all_stocks  = fdr.StockListing('KRX')
        ticker_list = [
            (str(code).zfill(6), str(name))
            for code, name in zip(all_stocks['Code'], all_stocks['Name'])
        ]
        print(f"✅ FDR을 통해 {len(ticker_list)}개 종목 조회 완료")
        return ticker_list
    except Exception as e:
        print(f"⚠️ FDR 조회 실패 (오류: {e}).")

    # 2. stock_list.csv
    fallback_path = Path(f"data/01_raw/{reference_date}/stock_list.csv")
    if fallback_path.exists():
        try:
            df_fb = pd.read_csv(fallback_path)
            ticker_list = [
                (str(code).zfill(6), str(name))
                for code, name in zip(df_fb['Code'], df_fb['Name'])
            ]
            print(f"✅ 로컬 파일에서 {len(ticker_list)}개 종목 로드 완료")
            return ticker_list
        except Exception as fe:
            print(f"❌ 로컬 파일 로드 중 오류: {fe}")
    else:
        print(f"❌ 로컬 파일 없음: {fallback_path}")

    raise RuntimeError(
        f"종목 리스트 조회 실패: FDR API 및 로컬 파일({fallback_path}) 모두 사용 불가"
    )


# ==========================================
# change_pct 자동 검증
# ==========================================

def _is_pct_scale(df: pd.DataFrame) -> bool:
    """
    `change_rate` 컬럼이 실제로 퍼센트 등락률인지 판별합니다.

    연속된 두 행의 close 비율로 계산한 단순 등락률과 stored 값을 비교합니다.
    같은 수집 출처라면 모든 종목이 동일한 규칙을 따르므로
    최초 20개 유효 행으로 판별하면 충분합니다.

    Returns
    -------
    False : 단순 등락률 (change_rate 그대로 유지)
    True  : 퍼센트 등락률 (change_pct로 rename 필요)
    """
    sample = df.dropna(subset=['close', 'change_rate']).copy()
    sample = sample.sort_values(['ticker', 'date'])

    errors_rate = []
    errors_pct  = []

    for _, grp in sample.groupby('ticker'):
        if len(grp) < 2:
            continue
        closes = grp['close'].values
        stored = grp['change_rate'].values
        for i in range(1, min(5, len(grp))):
            computed = closes[i] / closes[i - 1] - 1
            if abs(computed) < 1e-9:
                continue
            errors_rate.append(abs(computed       - stored[i]))
            errors_pct.append( abs(computed * 100 - stored[i]))
        if len(errors_rate) >= 20:
            break

    if not errors_rate:
        warnings.warn(
            "등락률 컬럼 타입 판별에 충분한 샘플이 없어 'change_rate'로 기본 처리합니다.",
            UserWarning,
        )
        return False

    return float(np.mean(errors_pct)) < float(np.mean(errors_rate))


def _convert_to_pct_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """
    수집된 데이터의 등락률 컬럼을 검증합니다.

    change_rate가 확정 컬럼명입니다.
    퍼센트 등락률 데이터 소스를 만났을 때만 예외적으로 change_pct로 변환합니다.

      단순 등락률 → 'change_rate' 유지 (정상, 변환 없음)
      퍼센트 등락률 → 'change_pct' (예외적 소스)
    """
    if 'change_rate' not in df.columns:
        return df

    if _is_pct_scale(df):
        print("⚠️  등락률 컬럼 검증: 퍼센트 등락률 감지 → 'change_pct'로 저장")
        return df.rename(columns={'change_rate': 'change_pct'})

    print("✅ 등락률 컬럼 검증: 단순 등락률 확인 → 'change_rate' 유지")
    return df


# ==========================================
# RawPriceCollector
# ==========================================

class RawPriceCollector:
    """
    KRX 원시 데이터 수집 및 다중 포맷 저장.

    v4.0.0 변경:
    - 수집 완료 후 change_pct 자동 검증 및 컬럼명 결정
    - 산출물 파일명: prices.parquet, ticker_master.csv (날짜 중복 제거)
    """

    def __init__(self, config: Dict[str, Any], paths=None):
        self.cfg      = config
        self.ref_date = config['project']['reference_date']

        if paths is None:
            from src.utils.config import ProjectPaths
            self.paths = ProjectPaths.from_config(config)
        else:
            self.paths = paths

        self.base_dir    = self.paths.raw_dir
        self.csv_dir     = self.paths.get_raw_csv_dir()

        # v4.0.0: 날짜 중복 제거된 파일명
        self.parquet_path = self.paths.get_raw_parquet()      # prices.parquet
        self.master_path  = self.paths.get_ticker_master()    # ticker_master.csv

        self.base_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg['data_collection'].get('save_csv', False):
            self.csv_dir.mkdir(parents=True, exist_ok=True)

    def fetch_ohlcv(self, ticker: str) -> pd.DataFrame:
        """단일 종목 OHLCV 데이터 조회 및 표준화."""
        try:
            start     = self.cfg['data_collection']['start_date']
            end       = self.cfg['data_collection']['end_date']
            start_fmt = pd.to_datetime(start).strftime('%Y-%m-%d')
            end_fmt   = pd.to_datetime(end).strftime('%Y-%m-%d')

            df = fdr.DataReader(ticker, start_fmt, end_fmt)

            if df.empty:
                return pd.DataFrame()

            df = df.reset_index()
            df = df.rename(columns={
                'Date'  : 'date',
                'Open'  : 'open',
                'High'  : 'high',
                'Low'   : 'low',
                'Close' : 'close',
                'Volume': 'volume',
                'Change': 'change_rate',   # 확정 컬럼명. 퍼센트 소스이면 collect_all에서 변환.
            })
            df['ticker'] = ticker
            df['date']   = pd.to_datetime(df['date'])

            return df
        except Exception:
            return pd.DataFrame()

    def collect_all(self, ticker_list: List[Tuple[str, str]]) -> Dict[str, int]:
        """전체 종목 수집 및 다중 포맷 저장 실행."""
        # 종목 마스터 저장 (v4.0.0: ticker_master.csv)
        df_master = pd.DataFrame(ticker_list, columns=['ticker', 'name'])
        df_master.to_csv(self.master_path, index=False, encoding='utf-8-sig')
        print(f"✅ 종목 마스터 저장 완료: {self.master_path}")

        stats   = {'success': 0, 'failed': 0, 'empty': 0}
        all_dfs = []

        for ticker, name in tqdm(ticker_list, desc="수집 중"):
            df = self.fetch_ohlcv(ticker)

            if df.empty:
                stats['empty'] += 1
                continue

            if self.cfg['data_collection'].get('save_csv', False):
                safe_name = name.replace('/', '_').replace('\\', '_')
                csv_path  = self.csv_dir / f"{safe_name}.csv"
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')

            all_dfs.append(df)
            stats['success'] += 1

            min_s     = self.cfg['data_collection']['min_sleep']
            max_s     = self.cfg['data_collection']['max_sleep']
            wait_time = 10 ** random.uniform(math.log10(min_s), math.log10(max_s))
            time.sleep(wait_time)

        if all_dfs:
            print(f"📦 데이터 병합 중... (총 {len(all_dfs)}개 종목)")
            df_total = pd.concat(all_dfs, ignore_index=True)
            df_total = df_total.sort_values(['ticker', 'date'])

            # v4.0.0: change_rate가 확정 컬럼명. 퍼센트 소스이면 예외적으로 change_pct로 변환.
            df_total = _convert_to_pct_if_needed(df_total)

            # v4.0.0: prices.parquet (날짜 중복 제거)
            df_total.to_parquet(
                self.parquet_path,
                compression='snappy',
                index=False,
            )
            print(f"✅ 통합 Parquet 저장 완료: {self.parquet_path}")

        return stats