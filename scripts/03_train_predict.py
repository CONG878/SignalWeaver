"""
Purpose:
    - 데이터 로딩 → Walk-forward 학습 → 예측 → 후보 종목 선정까지
      현재 아키텍처를 end-to-end로 관통하는 실행 스크립트
    - LightGBM 1차 예측 기준

Scope (의도적 제한):
    - 하이퍼파라미터 튜닝 없음
    - GRU 단계 없음 (후속 확장 포인트)
    - 결과 저장은 CSV/Parquet 단순 출력
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.data_loader.loader import load_processed_dataset
from src.models.lightgbm_model import LightGBMModel
from src.modeling.trainer import WalkForwardTrainer
from src.universe.select_universe import (
    build_production_universe,
    select_candidates_from_scores,
)
from src.models.artifact import save_model_artifact


# ---------------------------------------------------------------------
# Config (임시: 추후 config/settings.py로 이전 가능)
# ---------------------------------------------------------------------

DATA_PATH = Path("data/03_processed/dataset.parquet")
OUTPUT_DIR = Path("data/05_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "rsi_14",
    "ma_20",
    "volatility_20",
]

TARGET_COL = "target_lgbm"
DATE_COL = "date"

LGBM_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.01,
    "num_leaves": 64,
    "verbose": -1,
}

TRAIN_WINDOW = 252 * 3
TEST_WINDOW = 5
STEP = 5

TOP_K = 30
MIN_LIQUIDITY = 1e8
MAX_RISK_SCORE = 0.7


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    # 1. 데이터 로딩
    df = load_processed_dataset(DATA_PATH)

    # 2. 모델 초기화
    model = LightGBMModel(
        model_version="v1",
        params=LGBM_PARAMS,
        feature_list=FEATURE_COLS,
    )

    # 3. Walk-forward 학습/예측
    trainer = WalkForwardTrainer(
        model=model,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        date_col=DATE_COL,
    )

    pred_scores = trainer.run(
        df=df,
        train_window=TRAIN_WINDOW,
        test_window=TEST_WINDOW,
        step=STEP,
    )

    # 4. 운영 유니버스 필터
    prod_df = build_production_universe(
        df.merge(pred_scores, on=["date", "ticker"], how="inner"),
        min_liquidity=MIN_LIQUIDITY,
        max_risk_score=MAX_RISK_SCORE,
    )

    # 5. 후보 종목 선정
    candidates = select_candidates_from_scores(
        prod_df,
        score_col="score",
        top_k=TOP_K,
    )

    # 6. 결과 저장
    candidates_path = OUTPUT_DIR / "candidates_lgbm.parquet"
    candidates.to_parquet(candidates_path)

    # 7. 모델 아티팩트 저장
    save_model_artifact(
        model_name="lightgbm",
        model_version="v1",
        model_object=model,
        metadata={
            "feature_list": FEATURE_COLS,
            "training_window": TRAIN_WINDOW,
            "test_window": TEST_WINDOW,
            "hyperparameters": LGBM_PARAMS,
            "data_source": str(DATA_PATH),
        },
    )

    print(f"[DONE] Candidates saved to {candidates_path}")


if __name__ == "__main__":
    main()