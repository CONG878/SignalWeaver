"""
Purpose:
    - raw 주가 데이터 → 모델링용 feature dataset 생성
    - 03_train_predict.py에서 요구하는 컬럼 스키마를 고정

Scope (의도적 제한):
    - 지표는 최소한만 계산
    - 수학적 정교함보다 재현성과 스키마 고정이 목적
"""

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

RAW_DATA_PATH = Path("data/01_raw/krx_prices.parquet")
OUTPUT_PATH = Path("data/03_processed/dataset.parquet")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Feature functions (minimal)
# ---------------------------------------------------------------------

def calc_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    df = pd.read_parquet(RAW_DATA_PATH)

    df = df.sort_values(["ticker", "date"])

    # --- Feature engineering (per ticker)
    df["ma_20"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(20).mean())
    df["volatility_20"] = df.groupby("ticker")["close"].transform(lambda x: x.pct_change().rolling(20).std())
    df["rsi_14"] = df.groupby("ticker")["close"].transform(calc_rsi)

    # --- Target: forward 5-day return
    df["target_lgbm"] = (
        df.groupby("ticker")["close"].shift(-5) / df["close"] - 1
    )

    df = df.dropna().reset_index(drop=True)

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"[DONE] Feature dataset saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()