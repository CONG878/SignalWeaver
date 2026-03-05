"""
Purpose:
    - FinanceDataReader를 이용한 KRX 주가 데이터 수집
    - 종목 마스터(ticker_master), 통합 Parquet, 개별 CSV 다중 저장 지원
    - 로그 스케일 랜덤 대기를 통한 서버 차단 회피

Design Principles:
    - 통합 저장: 파이프라인 효율을 위해 전종목 데이터를 하나의 Parquet으로 병합
    - 메타 분리: 종목명 등 메타 정보는 별도 마스터 파일로 관리하여 데이터 중복 방지
    - 유연성: 디버깅을 위한 개별 CSV 저장 옵션 제공

✨ H1+H2 패치 (2026-02-08):
    - ProjectPaths 클래스 사용으로 경로 관리 중앙화
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
    KRX 전체 종목 리스트 조회 (FDR 실패 시 로컬 파일 Fallback 적용)
    """
    print(f"🔍 KRX 전체 종목 조회 중 (기준일: {reference_date})...")
    
    # 1. 기본 시도: FinanceDataReader
    try:
        all_stocks = fdr.StockListing('KRX')
        # FDR 결과에서도 만약을 대비해 'Code'를 6자리로 정렬하여 반환
        ticker_list = [
            (str(code).zfill(6), str(name)) 
            for code, name in zip(all_stocks['Code'], all_stocks['Name'])
        ]
        print(f"✅ FDR을 통해 {len(ticker_list)}개 종목 조회 완료")
        return ticker_list

    except Exception as e:
        print(f"⚠️ FDR 조회 실패 (오류: {e}).")
        fallback_path = Path(f"data/01_raw/{reference_date}/stock_list.csv")
        
        if fallback_path.exists():
            try:
                # 2. Fallback: 로컬 CSV 파일 (헤더는 이미 영문 'Code', 'Name'으로 맞춰진 상태 가정)
                df_fallback = pd.read_csv(fallback_path)
                
                # 데이터 타입 보존 및 6자리 패딩 (005930 유지)
                # 이미 6자리라면 변화가 없고, 숫자로 읽혀 5자리가 된 경우 앞을 0으로 채움
                ticker_list = [
                    (str(code).zfill(6), str(name))
                    for code, name in zip(df_fallback['Code'], df_fallback['Name'])
                ]
                
                print(f"✅ 로컬 파일에서 {len(ticker_list)}개 종목 로드 완료 (6자리 패딩 적용)")
                return ticker_list
            
            except Exception as fe:
                print(f"❌ 로컬 파일 로드 중 오류 발생: {fe}")
        else:
            print(f"❌ 로컬 파일이 존재하지 않습니다: {fallback_path}")
        
        raise e


class RawPriceCollector:
    """
    KRX 원시 데이터 수집 및 다중 포맷 저장 클래스
    
    ✨ H1+H2 패치: ProjectPaths 기반 경로 관리
    """

    def __init__(self, config: Dict[str, Any], paths=None):
        """
        Parameters
        ----------
        config : dict
            config.yaml에서 로드된 설정 딕셔너리
        paths : ProjectPaths, optional
            경로 객체 (없으면 내부에서 생성)
        """
        self.cfg = config
        self.ref_date = config['project']['reference_date']
        
        # ✨ H2 패치: ProjectPaths 사용
        if paths is None:
            from src.utils.config import ProjectPaths
            self.paths = ProjectPaths.from_config(config)
        else:
            self.paths = paths
        
        # 경로 설정
        self.base_dir = self.paths.raw_dir
        self.csv_dir = self.paths.get_raw_csv_dir()
        
        # 결과 파일 경로
        self.parquet_path = self.paths.get_raw_parquet()
        self.master_path = self.paths.get_ticker_master()
        
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
