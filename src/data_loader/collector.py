"""
Purpose:
    KRX 원시 주가 데이터 수집 전용 모듈
    - 책임: 외부 API → CSV 저장만 수행
    - Feature 계산, 필터링, 변환 등은 일절 수행하지 않음

Design Principles:
    - Single Responsibility: 데이터 "수집"만 담당
    - No Transformation: 값 변경 없이 API 그대로 저장
    - Fail-Safe: 개별 종목 실패가 전체를 중단하지 않음
"""

from __future__ import annotations

import os
import time
import random
import math
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from pykrx import stock
from tqdm import tqdm


# ---------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------

def safe_filename(name: str) -> str:
    """
    파일 시스템에서 허용되지 않는 문자를 제거
    
    Parameters
    ----------
    name : str
        원본 종목명
        
    Returns
    -------
    str
        안전한 파일명 (최대 120자)
    """
    forbidden = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    safe_name = ''.join('_' if c in forbidden else c for c in name)
    return safe_name.strip()[:120]


def ensure_directory(path: Path) -> None:
    """디렉토리가 없으면 생성"""
    path.mkdir(parents=True, exist_ok=True)


def random_sleep(min_seconds: float, max_seconds: float) -> None:
    """
    로그 스케일 랜덤 대기
    
    짧은 시간이 더 자주 나오지만, 가끔 긴 대기로 패턴을 깨뜨림
    
    Parameters
    ----------
    min_seconds : float
        최소 대기 시간 (예: 0.05초)
    max_seconds : float
        최대 대기 시간 (예: 1.0초)
        
    Distribution
    ------------
    로그 균등 분포를 사용하여:
    - min_seconds 근처 값들이 가장 빈번하게 발생
    - max_seconds에 가까울수록 지수적으로 확률 감소
    - 하지만 가끔씩 긴 대기시간으로 패턴 차단 회피
    
    Examples
    --------
    min=0.05, max=1.0 설정 시:
    - 약 60%: 0.05~0.2초
    - 약 30%: 0.2~0.5초  
    - 약 10%: 0.5~1.0초
    """
    log_wait = random.uniform(math.log(min_seconds), math.log(max_seconds))
    wait_time = math.exp(log_wait)
    time.sleep(wait_time)


# ---------------------------------------------------------------------
# Raw Data Collector
# ---------------------------------------------------------------------

class RawPriceCollector:
    """
    KRX 종목별 원시 시세 수집기
    
    책임
    ----
    1. pykrx API를 통해 원시 데이터 조회
    2. API가 제공하는 모든 컬럼 보존
    3. 컬럼명 표준화 (한글 → 영문)
    4. CSV 파일로 저장
    
    비책임
    ------
    - 우리가 계산하는 Feature (이동평균, RSI, MACD 등)
    - Target 계산 (수익률, 로그 변환 등)
    - 데이터 필터링 또는 선택적 제거
    - 결측치 보정
    
    원칙
    ----
    "API가 주는 것은 전부 받는다" (무판단 원칙)
    "우리가 계산하는 것은 절대 하지 않는다" (단일 책임 원칙)
    """
    
    def __init__(
        self,
        output_dir: Path,
        start_date: str,
        end_date: str,
        min_sleep: float = 0.05,
        max_sleep: float = 0.8
    ):
        """
        Parameters
        ----------
        output_dir : Path
            CSV 파일 저장 경로 (예: data/01_raw/krx_prices/)
        start_date : str
            조회 시작일 (YYYYMMDD)
        end_date : str
            조회 종료일 (YYYYMMDD)
        min_sleep : float
            최소 대기 시간 (초). 기본값 0.05초
        max_sleep : float
            최대 대기 시간 (초). 기본값 0.8초
            
        Notes
        -----
        로그 스케일 랜덤 대기를 사용하여:
        - 대부분의 요청은 빠르게 처리 (min_sleep 근처)
        - 가끔씩 긴 대기로 패턴 회피 (max_sleep 까지)
        - 서버 차단 위험 감소
        """
        self.output_dir = Path(output_dir)
        self.start_date = start_date
        self.end_date = end_date
        self.min_sleep = min_sleep
        self.max_sleep = max_sleep
        
        ensure_directory(self.output_dir)
        
    def fetch_single_ticker(self, ticker: str) -> pd.DataFrame:
        """
        단일 종목의 원시 시세 데이터 조회
        
        Parameters
        ----------
        ticker : str
            종목코드 (예: "005930")
            
        Returns
        -------
        DataFrame
            API가 제공하는 모든 원시 데이터 (컬럼명만 정규화)
            실패 시 빈 DataFrame 반환
            
        Note
        ----
        pykrx의 get_market_ohlcv_by_date는 다음을 제공:
        - 기본: 시가, 고가, 저가, 종가, 거래량
        - 추가: 등락률, 거래대금, 시가총액 (종목에 따라 다름)
        
        우리는 API가 주는 모든 컬럼을 그대로 저장합니다.
        """
        try:
            df = stock.get_market_ohlcv_by_date(
                self.start_date,
                self.end_date,
                ticker
            )
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            # 인덱스(날짜)를 컬럼으로 변환
            df = df.reset_index()
            
            # 컬럼명만 정규화 (값은 건드리지 않음)
            df = self._normalize_columns(df)
            
            return df
            
        except Exception as e:
            tqdm.write(f"[WARN] {ticker} fetch failed: {e}")
            return pd.DataFrame()
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        컬럼명을 표준 영문 스키마로 변환
        
        Note
        ----
        - 값 변환이 아닌 컬럼명 변환만 수행
        - API가 제공하는 모든 컬럼을 매핑 (버리지 않음)
        - 매핑되지 않은 컬럼도 그대로 유지
        
        Parameters
        ----------
        df : DataFrame
            pykrx API 응답 원본
            
        Returns
        -------
        DataFrame
            컬럼명만 영문으로 변환된 데이터
        """
        # pykrx가 제공하는 모든 컬럼 매핑
        rename_map = {
            '날짜': 'date',
            '시가': 'open',
            '고가': 'high',
            '저가': 'low',
            '종가': 'close',
            '거래량': 'volume',
            '거래대금': 'amount',
            '등락률': 'change_pct',
            '시가총액': 'market_cap',
            '상장주식수': 'shares_outstanding',
        }
        
        # 매핑된 것만 rename (나머지는 유지)
        df = df.rename(columns=rename_map)
        
        # date 컬럼 타입 통일
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def save_ticker_data(
        self,
        ticker: str,
        name: str,
        df: pd.DataFrame
    ) -> bool:
        """
        종목 데이터를 CSV로 저장
        
        Parameters
        ----------
        ticker : str
            종목코드
        name : str
            종목명 (파일명으로 사용)
        df : DataFrame
            저장할 데이터
            
        Returns
        -------
        bool
            저장 성공 여부
        """
        if df.empty:
            return False
        
        safe_name = safe_filename(name)
        file_path = self.output_dir / f"{safe_name}.csv"
        
        try:
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            return True
        except Exception as e:
            tqdm.write(f"[ERROR] {ticker}({name}) save failed: {e}")
            return False
    
    def collect_all(
        self,
        ticker_list: List[Tuple[str, str]]
    ) -> dict:
        """
        전체 종목 수집 메인 루프
        
        Parameters
        ----------
        ticker_list : List[Tuple[str, str]]
            [(종목코드, 종목명), ...] 형태의 리스트
            
        Returns
        -------
        dict
            수집 결과 통계
            - success: 성공 개수
            - failed: 실패 개수
            - empty: 데이터 없음 개수
        """
        stats = {'success': 0, 'failed': 0, 'empty': 0}
        
        for ticker, name in tqdm(ticker_list, desc="Collecting KRX Prices"):
            # 1. 데이터 조회
            df = self.fetch_single_ticker(ticker)
            
            if df.empty:
                stats['empty'] += 1
                continue
            
            # 2. 저장
            success = self.save_ticker_data(ticker, name, df)
            
            if success:
                stats['success'] += 1
            else:
                stats['failed'] += 1
            
            # 3. 서버 부하 방지 (로그 스케일 랜덤 대기)
            random_sleep(self.min_sleep, self.max_sleep)
        
        return stats


# ---------------------------------------------------------------------
# 종목 리스트 조회
# ---------------------------------------------------------------------

def get_ticker_universe(reference_date: str) -> List[Tuple[str, str]]:
    """
    KRX 전체 종목 리스트 조회 (KOSPI + KOSDAQ + KONEX)
    
    Parameters
    ----------
    reference_date : str
        기준일 (YYYYMMDD)
        
    Returns
    -------
    List[Tuple[str, str]]
        [(종목코드, 종목명), ...] 리스트
    """
    try:
        all_tickers = stock.get_market_ticker_list(reference_date, market="ALL")
        
        ticker_list = []
        for ticker in tqdm(all_tickers, desc="Loading Ticker Names"):
            try:
                name = stock.get_market_ticker_name(ticker)
                ticker_list.append((ticker, name))
            except:
                ticker_list.append((ticker, ticker))  # 실패 시 코드를 이름으로
        
        return ticker_list
        
    except Exception as e:
        raise RuntimeError(f"Failed to fetch ticker universe: {e}")


# ---------------------------------------------------------------------
# Usage Example (Documentation)
# ---------------------------------------------------------------------
"""
from src.data_loader.collector import RawPriceCollector, get_ticker_universe
from pathlib import Path

# 1. 종목 리스트 확보
tickers = get_ticker_universe("20251031")

# 2. Collector 초기화
collector = RawPriceCollector(
    output_dir=Path("data/01_raw/krx_prices"),
    start_date="20200101",
    end_date="20251031",
    min_sleep=0.05,  # 최소 50ms
    max_sleep=0.8    # 최대 800ms (로그 분포)
)

# 3. 수집 실행
stats = collector.collect_all(tickers)
print(f"Success: {stats['success']}, Failed: {stats['failed']}")
"""