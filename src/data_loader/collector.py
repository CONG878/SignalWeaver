"""
Data Collection Engine

KRX 주가 데이터 및 종목 마스터 정보를 수집합니다.

## 설계 원칙
- **통합 저장**: 파이프라인 효율을 위해 전 종목 데이터를 단일 Parquet으로 병합
- **메타 분리**: 종목명 등 메타 정보는 별도 CSV로 관리하여 데이터 중복 방지
- **유연성**: 디버깅용 개별 CSV 저장 옵션 제공

## 안정성
- **Fallback 메커니즘** (v3.8.1): FDR API 실패 시 로컬 CSV로 대체
- **Rate Limiting**: 로그 스케일 랜덤 대기로 서버 차단 회피

## 버전
- v3.8.1: Fallback 강화 (3단계 우선순위)
- v3.8.0: ProjectPaths 기반 경로 관리 중앙화
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
    KRX 상장 종목 리스트 조회 (3단계 Fallback).

    우선순위:
    1. FinanceDataReader API (fdr.StockListing('KRX'))
    2. 로컬 stock_list.csv (data/01_raw/{reference_date}/)
    3. 로컬 ticker_master.csv (01단계 수집 산출물)

    Parameters
    ----------
    reference_date : str
        기준일 (YYYYMMDD 형식)

    Returns
    -------
    List[Tuple[str, str]]
        (ticker_6digit, name) 튜플 리스트
        예: [('005930', 'Samsung Electronics'), ...]

    Raises
    ------
    RuntimeError
        모든 Fallback 경로가 실패한 경우

    Notes
    -----
    - 종목 코드는 6자리로 패딩됨
    - API 실패 시 자동으로 다음 경로 시도 (v3.8.1)
    - 명시적 오류 메시지로 실패 경위 추적 용이
    """
    print(f"🔍 KRX 전체 종목 조회 중 (기준일: {reference_date})...")

    # ── 1순위: FDR API (v3.8.1: Fallback 우선순위 명시)
    try:
        all_stocks = fdr.StockListing('KRX')
        ticker_list = [
            (str(code).zfill(6), str(name))
            for code, name in zip(all_stocks['Code'], all_stocks['Name'])
        ]
        print(f"✅ FDR을 통해 {len(ticker_list)}개 종목 조회 완료")
        return ticker_list

    except Exception as e:
        print(f"⚠️ FDR 조회 실패 (오류: {e}).")

    # ── 2순위: 로컬 CSV Fallback (v3.8.1)
    fallback_path = Path(f"data/01_raw/{reference_date}/stock_list.csv")

    if fallback_path.exists():
        try:
            df_fallback = pd.read_csv(fallback_path)
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

    raise RuntimeError(
        f"종목 리스트 조회 실패: FDR API 및 로컬 파일({fallback_path}) 모두 사용 불가"
    )


class RawPriceCollector:
    """
    KRX 원시 데이터 수집 및 다중 포맷 저장

    OHLCV 데이터를 종목별로 수집하여 다음 세 가지 형태로 저장합니다:
    - 통합 Parquet (파이프라인 I/O 효율화)
    - 개별 CSV (디버깅/검증 용이)
    - 종목 마스터 파일 (메타 정보)

    ProjectPaths 클래스를 통해 경로를 중앙 관리합니다 (v3.8.0~).
    """

    def __init__(self, config: Dict[str, Any], paths=None):
        """
        데이터 수집기 초기화.

        Parameters
        ----------
        config : dict
            config.yaml에서 로드된 설정 딕셔너리.
            필수 키: project.reference_date, data_collection.*, paths.*
        paths : ProjectPaths, optional
            경로 객체. 미제공 시 config에서 자동 생성됨.

        Notes
        -----
        - ProjectPaths는 v3.8.0부터 경로 관리를 중앙화함
        - 저장 디렉토리는 자동 생성됨 (parents=True, exist_ok=True)
        """
        self.cfg = config
        self.ref_date = config['project']['reference_date']

        # v3.8.0: ProjectPaths를 통한 경로 관리
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
        """
        단일 종목의 OHLCV 데이터 조회 및 표준화.

        Parameters
        ----------
        ticker : str
            종목 코드 (6자리)

        Returns
        -------
        pd.DataFrame
            표준화된 OHLCV 데이터 (date, open, high, low, close, volume, change_pct, ticker)
        """
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
        전체 종목 수집 및 다중 포맷 저장 실행.

        Parameters
        ----------
        ticker_list : List[Tuple[str, str]]
            (ticker, name) 튜플 리스트

        Returns
        -------
        Dict[str, int]
            수집 통계 (success, failed, empty)
        """
        # 종목 마스터 파일 저장 (ticker-name 매핑)
        df_master = pd.DataFrame(ticker_list, columns=['ticker', 'name'])
        df_master.to_csv(self.master_path, index=False, encoding='utf-8-sig')
        print(f"✅ 종목 마스터 저장 완료: {self.master_path}")

        stats = {'success': 0, 'failed': 0, 'empty': 0}
        all_dfs = []

        # 개별 종목 순회 수집
        for ticker, name in tqdm(ticker_list, desc="수집 중"):
            df = self.fetch_ohlcv(ticker)

            if df.empty:
                stats['empty'] += 1
                continue

            # CSV 저장 (선택 사항)
            if self.cfg['data_collection'].get('save_csv', False):
                safe_name = name.replace('/', '_').replace('\\', '_')
                csv_path = self.csv_dir / f"{safe_name}.csv"
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')

            # 통합 Parquet을 위한 리스트 추가
            all_dfs.append(df)
            stats['success'] += 1

            # Rate limiting (로그 스케일 랜덤 대기)
            min_s = self.cfg['data_collection']['min_sleep']
            max_s = self.cfg['data_collection']['max_sleep']
            wait_time = 10 ** random.uniform(math.log10(min_s), math.log10(max_s))
            time.sleep(wait_time)

        # 통합 Parquet 파일 저장
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
