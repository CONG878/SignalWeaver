"""
KRX 데이터 수집 (2025년 대응 버전)
- FinanceDataReader 사용
- 설치: pip install finance-datareader
"""

import time
import random
import math
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm

try:
    import FinanceDataReader as fdr
except ImportError:
    raise ImportError(
        "FinanceDataReader가 설치되지 않았습니다.\n"
        "설치: pip install finance-datareader"
    )


# ==========================================
# 종목 리스트 조회
# ==========================================
def get_ticker_universe(reference_date: str) -> List[Tuple[str, str]]:
    """
    FinanceDataReader로 KRX 전체 종목 리스트 조회
    
    Parameters
    ----------
    reference_date : str
        기준일 (YYYYMMDD) - 실제로는 사용 안 됨 (최신 데이터 조회)
        
    Returns
    -------
    List[Tuple[str, str]]
        [(종목코드, 종목명), ...] 리스트
        
    Note
    ----
    FDR은 기준일 지정 불가능, 항상 최신 상장 종목 반환
    """
    print(f"🔍 KRX 전체 종목 조회 중 (FinanceDataReader)...")
    
    try:
        all_stocks = fdr.StockListing('KRX')
        print(f"  - KRX: {len(all_stocks)}개")
        
        # (코드, 이름) 튜플 리스트 생성
        ticker_list = list(zip(
            all_stocks['Code'].values,
            all_stocks['Name'].values
        ))
        
        print(f"✅ 총 {len(ticker_list)}개 종목 조회 완료")
        return ticker_list
        
    except Exception as e:
        raise RuntimeError(f"종목 리스트 조회 실패: {e}")


# ==========================================
# OHLCV 조회
# ==========================================
def fetch_ohlcv(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    FinanceDataReader로 OHLCV 조회
    
    Parameters
    ----------
    ticker : str
        종목코드 (예: "005930")
    start_date : str
        시작일 (YYYYMMDD)
    end_date : str
        종료일 (YYYYMMDD)
        
    Returns
    -------
    DataFrame
        OHLCV 데이터 (실패 시 빈 DataFrame)
    """
    try:
        # 날짜 포맷 변환 (YYYYMMDD → YYYY-MM-DD)
        start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
        end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
        
        df = fdr.DataReader(ticker, start, end)
        
        if df.empty:
            return pd.DataFrame()
        
        # 표준화
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
        
        # ticker 컬럼 추가
        df['ticker'] = ticker
        
        # 날짜 타입 통일
        df['date'] = pd.to_datetime(df['date'])
        
        return df
        
    except Exception:
        return pd.DataFrame()


# ==========================================
# Collector Class
# ==========================================
class RawPriceCollector:
    """
    KRX 원시 데이터 수집기 (FinanceDataReader)
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
            CSV 저장 경로
        start_date : str
            조회 시작일 (YYYYMMDD)
        end_date : str
            조회 종료일 (YYYYMMDD)
        min_sleep : float
            최소 대기 시간 (초)
        max_sleep : float
            최대 대기 시간 (초)
        """
        self.output_dir = Path(output_dir)
        self.start_date = start_date
        self.end_date = end_date
        self.min_sleep = min_sleep
        self.max_sleep = max_sleep
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_single_ticker(self, ticker: str) -> pd.DataFrame:
        """단일 종목 조회"""
        return fetch_ohlcv(ticker, self.start_date, self.end_date)
    
    def save_ticker_data(self, ticker: str, name: str, df: pd.DataFrame) -> bool:
        """CSV 저장"""
        if df.empty:
            return False
        
        # 파일명 안전화
        safe_name = name.replace('/', '_').replace('\\', '_').replace('*', '_')
        file_path = self.output_dir / f"{safe_name}.csv"
        
        try:
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            return True
        except Exception as e:
            tqdm.write(f"[ERROR] {ticker}({name}) 저장 실패: {e}")
            return False
    
    def collect_all(self, ticker_list: List[Tuple[str, str]]) -> dict:
        """
        전체 종목 수집
        
        Returns
        -------
        dict
            {'success': int, 'failed': int, 'empty': int}
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
            
            # 3. Rate limiting (로그 스케일 랜덤 대기)
            wait_time = 10 ** random.uniform(
                math.log10(self.min_sleep),
                math.log10(self.max_sleep)
            )
            time.sleep(wait_time)
        
        return stats


# ==========================================
# Usage Example
# ==========================================
"""
from src.data_loader.collector import RawPriceCollector, get_ticker_universe
from pathlib import Path

# 1. 종목 리스트
tickers = get_ticker_universe("20251226")

# 2. Collector 초기화
collector = RawPriceCollector(
    output_dir=Path("data/01_raw/krx_prices_20251226"),
    start_date="20250101",
    end_date="20251226"
)

# 3. 수집 실행
stats = collector.collect_all(tickers)
print(f"Success: {stats['success']}, Failed: {stats['failed']}")
"""