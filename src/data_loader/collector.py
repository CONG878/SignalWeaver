"""
Purpose:
    - FinanceDataReader를 이용한 KRX 주가 데이터 수집
    - 종목 마스터(ticker_master), 통합 Parquet, 개별 CSV 다중 저장 지원
    - 로그 스케일 랜덤 대기를 통한 서버 차단 회피

Design Principles:
    - 통합 저장: 파이프라인 효율을 위해 전종목 데이터를 하나의 Parquet으로 병합
    - 메타 분리: 종목명 등 메타 정보는 별도 마스터 파일로 관리하여 데이터 중복 방지
    - 유연성: 디버깅을 위한 개별 CSV 저장 옵션 제공
"""

import time
import random
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any
import pandas as pd
from tqdm import tqdm

try:
    import FinanceDataReader as fdr
except ImportError:
    raise ImportError(
        "FinanceDataReader가 설치되지 않았습니다. 설치: pip install finance-datareader"
    )


def get_ticker_universe(reference_date: str) -> List[Tuple[str, str]]:
    """
    KRX 전체 종목 리스트 조회
    """
    print(f"🔍 KRX 전체 종목 조회 중 (기준일: {reference_date})...")
    try:
        all_stocks = fdr.StockListing('KRX')
        ticker_list = list(zip(
            all_stocks['Code'].values,
            all_stocks['Name'].values
        ))
        print(f"✅ 총 {len(ticker_list)}개 종목 조회 완료")
        return ticker_list
    except Exception as e:
        raise RuntimeError(f"종목 리스트 조회 실패: {e}")


class RawPriceCollector:
    """
    KRX 원시 데이터 수집 및 다중 포맷 저장 클래스
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Parameters
        ----------
        config : dict
            config.yaml에서 로드된 설정 딕셔너리
        """
        self.cfg = config
        self.ref_date = config['project']['reference_date']
        
        # 경로 설정: data/01_raw/{YYYYMMDD}
        self.base_dir = Path(config['paths']['raw_dir']) / self.ref_date
        self.csv_dir = self.base_dir / "csv"
        
        # 결과 파일 경로
        self.parquet_path = self.base_dir / f"krx_prices_{self.ref_date}.parquet"
        self.master_path = self.base_dir / f"ticker_master_{self.ref_date}.csv"
        
        # 디렉토리 생성
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg['data_collection'].get('save_csv', False):
            self.csv_dir.mkdir(parents=True, exist_ok=True)

    def fetch_ohlcv(self, ticker: str) -> pd.DataFrame:
        """단일 종목 OHLCV 조회 및 표준화"""
        try:            
            start = self.cfg['data_collection']['start_date']
            end = self.cfg['data_collection']['end_date']
            
            # 날짜 포맷 표준화 (FinanceDataReader 대응)
            start_fmt = pd.to_datetime(start).strftime('%Y-%m-%d')
            end_fmt = pd.to_datetime(end).strftime('%Y-%m-%d')
            
            df = fdr.DataReader(ticker, start_fmt, end_fmt)
            
            if df.empty:
                return pd.DataFrame()
            
            # 컬럼명 표준화 (데이터 계약 준수)
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Change': 'change_pct'
            })
            df['ticker'] = ticker
            df['date'] = pd.to_datetime(df['date'])
            
            return df
        except Exception:
            return pd.DataFrame()

    def collect_all(self, ticker_list: List[Tuple[str, str]]) -> Dict[str, int]:
        """
        전체 종목 수집 및 통합 저장 실행
        """
        # 1. 종목 마스터 파일 저장 (ticker-name 매핑)
        df_master = pd.DataFrame(ticker_list, columns=['ticker', 'name'])
        df_master.to_csv(self.master_path, index=False, encoding='utf-8-sig')
        print(f"✅ 종목 마스터 저장 완료: {self.master_path}")

        stats = {'success': 0, 'failed': 0, 'empty': 0}
        all_dfs = []

        # 2. 개별 종목 순회 수집
        for ticker, name in tqdm(ticker_list, desc="수집 중"):
            df = self.fetch_ohlcv(ticker)
            
            if df.empty:
                stats['empty'] += 1
                continue

            # A. 개별 CSV 저장 (선택 사항)
            if self.cfg['data_collection'].get('save_csv', False):
                safe_name = name.replace('/', '_').replace('\\', '_')
                csv_path = self.csv_dir / f"{safe_name}.csv"
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')

            # B. 통합 Parquet을 위한 리스트 추가
            all_dfs.append(df)
            stats['success'] += 1

            # 3. Rate limiting (로그 스케일 랜덤 대기)
            min_s = self.cfg['data_collection']['min_sleep']
            max_s = self.cfg['data_collection']['max_sleep']
            wait_time = 10 ** random.uniform(math.log10(min_s), math.log10(max_s))
            time.sleep(wait_time)

        # 4. 통합 Parquet 파일 저장
        if all_dfs:
            print(f"📦 데이터 병합 중... (총 {len(all_dfs)}개 종목)")
            df_total = pd.concat(all_dfs, ignore_index=True)
            df_total = df_total.sort_values(['ticker', 'date'])
            
            df_total.to_parquet(
                self.parquet_path, 
                compression='snappy', 
                index=False
            )
            print(f"✅ 통합 Parquet 저장 완료: {self.parquet_path}")

        return stats