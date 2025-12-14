"""
Purpose:
    - processed dataset 로딩을 담당하는 최소 어댑터
    - 스크립트(03_train_predict.py)와 데이터 저장 형식 간의 결합을 분리

Design scope (intentional):
    - parquet 단일 파일 로딩만 지원
    - 검증, 캐싱, 로깅, 옵션 없음
    - 향후 DB / feature store / partitioned dataset으로 교체 가능
"""

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def load_processed_dataset(path: Path) -> pd.DataFrame:
    """
    Load processed dataset from storage.

    Parameters
    ----------
    path : Path
        Path to processed dataset (parquet)

    Returns
    -------
    DataFrame
        Loaded dataset
    """
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")

    return pd.read_parquet(path)