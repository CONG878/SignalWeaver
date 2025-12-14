"""
Purpose:
    - 실제 데이터 없이도 아키텍처 판단이 가능하도록
      전체 파이프라인을 관통할 수 있는 synthetic KRX-like 데이터 생성

Design principles:
    - 의미/통계적 타당성은 중요하지 않음
    - 컬럼 스키마, 타입, 흐름 검증이 목적
    - 소규모 (빠른 실행, 디버깅 용이)

Output:
    data/01_raw/krx_prices.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

OUTPUT_PATH = Path("data/01_raw/krx_prices.parquet")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

N_TICKERS = 5
N_DAYS = 260
START_DATE = "2023-01-01"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    tickers = [f"TICKER_{i:02d}" for i in range(1, N_TICKERS + 1)]
    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="B")

    records = []

    for ticker in tickers:
        price = 100 + np.cumsum(np.random.normal(0, 1, size=len(dates)))
        volume = np.random.randint(1_000_000, 5_000_000, size=len(dates))

        for d, p, v in zip(dates, price, volume):
            records.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "close": round(float(p), 2),
                    "volume": int(v),
                }
            )

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"[DONE] Synthetic raw data saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()