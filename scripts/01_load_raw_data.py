"""
Purpose:
    - KRX 원시 주가 데이터를 로드하여 raw parquet으로 고정
    - 이후 모든 파이프라인의 기준 입력(anchor)

Scope:
    - 수집 로직 단순화 (CSV 기준)
    - 전처리/정제/지표 계산 절대 수행하지 않음
"""

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

RAW_INPUT_PATH = Path("data/01_raw/krx_prices.csv")
RAW_OUTPUT_PATH = Path("data/01_raw/krx_prices.parquet")
RAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

REQUIRED_COLS = [
    "date",      # YYYY-MM-DD
    "ticker",    # 종목코드
    "close",     # 종가
    "volume",    # 거래량
]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    df = pd.read_csv(RAW_INPUT_PATH)

    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 최소한의 타입 정리
    df = df[REQUIRED_COLS].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    df.to_parquet(RAW_OUTPUT_PATH, index=False)
    print(f"[DONE] Raw data saved to {RAW_OUTPUT_PATH}")


if __name__ == "__main__":
    main()