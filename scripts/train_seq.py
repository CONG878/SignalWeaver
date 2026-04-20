"""
03c. Seq 모델 학습 스크립트

노트북(03c_train_seq.ipynb) 대신 스크립트로 실행합니다.
Jupyter 커널과 달리 OS가 프로세스 메모리를 독립 관리하므로 장시간 학습에 안정적입니다.

## 사용법

# 기본 실행 (처음부터 학습)
python scripts/train_seq.py

# 체크포인트에서 이어서 학습
python scripts/train_seq.py --resume

# 설정 파일 지정
python scripts/train_seq.py --config config/config.yaml

# 에포크 수 임시 지정 (config 값 무시)
python scripts/train_seq.py --epochs 20

# 저장만 (학습 없이 현재 weights.pt에서 평가 후 저장)
python scripts/train_seq.py --eval-only

## 출력 경로
data/03_seq/{model_date}/{seq_model}/
  weights.pt          추론 전용 최선 가중치
  config.json         모델 메타데이터
  checkpoint.pt       resume 용 (학습 중 자동 갱신)
  test_predictions.parquet  05단계 입력
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

# 프로젝트 루트를 sys.path에 추가 (어느 디렉토리에서 실행해도 동작)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, ProjectPaths
from src.models.gru_model import GRUModel
from src.modeling.seq_trainer import SeqTrainer


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SignalWeaver Seq 모델 학습 (GRU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config",    default="config/config.yaml",
                   help="설정 파일 경로 (기본: config/config.yaml)")
    p.add_argument("--resume",    action="store_true",
                   help="checkpoint.pt에서 이어서 학습")
    p.add_argument("--epochs",    type=int, default=None,
                   help="에포크 수 임시 지정 (미지정 시 config 값 사용)")
    p.add_argument("--eval-only", action="store_true",
                   help="학습 없이 기존 모델 평가 및 저장만 수행")
    p.add_argument("--n-folds",   type=int, default=1, choices=[1, 2],
                   help="Fold 수. 1=1-Fold(기본), 2=2-Fold(앙상블 대비)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 설정 로드 ────────────────────────────────────────────
    cfg_path = PROJECT_ROOT / args.config
    cfg       = load_config(str(cfg_path))
    train_cfg = cfg["training"]
    seq_cfg   = cfg["sequence"]
    gru_cfg   = cfg["gru_params"].copy()

    active_seq = cfg.get("active_seq_model", "gru")
    if active_seq != "gru":
        raise ValueError(f"active_seq_model='{active_seq}'은 현재 미지원. 'gru'만 지원합니다.")

    # CLI --epochs가 있으면 config 값 덮어쓰기
    if args.epochs is not None:
        gru_cfg["epochs"] = args.epochs
        print(f"   --epochs {args.epochs} 으로 config 값 덮어쓰기")

    paths = ProjectPaths.from_config(cfg)
    paths.ensure_dirs()

    print("=" * 65)
    print("🚀 [03c] Seq 모델 학습 시작")
    print("=" * 65)
    print(f"   기준일:          {paths.reference_date}")
    print(f"   모델:            {active_seq}")
    print(f"   seq_len:         {seq_cfg['seq_len']}")
    print(f"   forecast_horizon:{seq_cfg['forecast_horizon']}")
    print(f"   stride:          {seq_cfg.get('stride', 1)}")
    print(f"   epochs:          {gru_cfg['epochs']}")
    print(f"   resume:          {args.resume}")
    print(f"   n_folds:         {args.n_folds}")
    print(f"   입력:  {paths.get_dataset_parquet()}")
    print(f"   출력:  {paths.get_seq_model_dir()}")
    print()

    # ── 데이터 로드 ──────────────────────────────────────────
    dataset_path = paths.get_dataset_parquet()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"dataset.parquet이 없습니다: {dataset_path}\n"
            "02단계(02_build_dataset.ipynb)를 먼저 실행하세요."
        )

    print("📥 데이터 로드 중...")
    df = pd.read_parquet(dataset_path)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    target_col   = (
        "target_log_return_1d"
        if seq_cfg["target_type"] == "log_return_1d"
        else "target_log_close"
    )

    if target_col not in df.columns:
        raise KeyError(
            f"'{target_col}' 컬럼이 없습니다. "
            "02단계에서 해당 타겟 컬럼이 생성됐는지 확인하세요."
        )

    print(f"   행수: {len(df):,}  종목: {df['ticker'].nunique():,}  피처: {len(feature_cols)}")

    # ── eval-only 모드 ───────────────────────────────────────
    if args.eval_only:
        _run_eval_only(paths, gru_cfg, seq_cfg, feature_cols, df, target_col, train_cfg)
        return

    # ── 모델 초기화 ──────────────────────────────────────────
    seq_model_dir = paths.get_seq_model_dir()
    model = GRUModel(
        model_version    = f"v4.0.0_gru_{paths.reference_date}",
        params           = gru_cfg,
        seq_len          = seq_cfg["seq_len"],
        forecast_horizon = seq_cfg["forecast_horizon"],
        feature_list     = feature_cols,
        target_type      = seq_cfg["target_type"],
        checkpoint_dir   = seq_model_dir,
    )

    # ── Trainer 초기화 ───────────────────────────────────────
    trainer = SeqTrainer(
        model            = model,
        feature_cols     = feature_cols,
        target_col       = target_col,
        seq_len          = seq_cfg["seq_len"],
        forecast_horizon = seq_cfg["forecast_horizon"],
        stride           = seq_cfg.get("stride", 1),
        date_col         = "date",
        integration_order=cfg.get('integration_order', 1),   # ← 추가
    )

    # ── 학습 실행 ────────────────────────────────────────────
    results = trainer.run(
        df                = df,
        train_end         = train_cfg["train_end"],
        valid_window_days = train_cfg["valid_window_days"],
        test_window_days  = train_cfg["test_window_days"],
        n_folds           = args.n_folds,
        resume            = args.resume,
        fit_kwargs        = {
            "epochs":   gru_cfg["epochs"],
            "patience": gru_cfg["patience"],
        },
    )

    # ── 결과 저장 ────────────────────────────────────────────
    print("\n💾 저장 중...")

    # 모델 (weights.pt + config.json)
    results["final_model"].save(str(seq_model_dir))
    print(f"   ✅ 모델: {seq_model_dir}")

    # test_predictions.parquet
    test_path = paths.get_seq_test_predictions()
    results["test_predictions"].to_parquet(test_path, engine='fastparquet', index=False)
    print(f"   ✅ test_predictions: {test_path}")

    # val_predictions (n_folds=2인 경우만)
    if results["val_predictions"] is not None:
        val_path = paths.get_seq_val_predictions()
        results["val_predictions"].to_parquet(val_path, engine='fastparquet', index=False)
        print(f"   ✅ val_predictions:  {val_path}")
    else:
        print("   ⏭️  val_predictions 생략 (1-Fold 모드)")

    # ── 성능 요약 ────────────────────────────────────────────
    m = results["test_metrics"]
    print(f"\n📊 성능 요약")
    print(f"   테스트 Avg RMSE: {m['avg_rmse']:.6f}")
    print(f"   테스트 Avg IC:   {m['avg_ic']:.4f}")
    print(f"\n   Horizon별 테스트 (첫 5개):")
    for h, metrics in list(m["per_horizon"].items())[:5]:
        print(f"     h{h}: RMSE={metrics['rmse']:.6f}, IC={metrics['ic_mean']:.4f}")

    print("\n" + "=" * 65)
    print("✅ [03c] 완료")
    print("=" * 65)
    print(f"\n다음 단계: 04_forecast_future.ipynb 실행")
    print(f"  config.yaml의 active_model을 'gru'로 설정하거나")
    print(f"  active_seq_model: '{active_seq}' 확인 후 실행하세요.")


def _run_eval_only(paths, gru_cfg, seq_cfg, feature_cols, df, target_col, train_cfg):
    """학습 없이 기존 모델로 평가 및 저장."""
    seq_model_dir = paths.get_seq_model_dir()
    weights_path  = seq_model_dir / "weights.pt"

    if not weights_path.exists():
        raise FileNotFoundError(
            f"--eval-only이지만 weights.pt가 없습니다: {weights_path}\n"
            "먼저 학습을 실행하세요."
        )

    print("📂 기존 모델 로드 중...")
    model = GRUModel.load(str(seq_model_dir))

    trainer = SeqTrainer(
        model            = model,
        feature_cols     = feature_cols,
        target_col       = target_col,
        seq_len          = seq_cfg["seq_len"],
        forecast_horizon = seq_cfg["forecast_horizon"],
        stride           = seq_cfg.get("stride", 1),
    )

    # 평가만 수행 (epochs=0 불가 → 직접 평가)
    from src.data_loader.seq_builder import split_by_date
    splits = split_by_date(
        df=df, feature_cols=feature_cols, target_col=target_col,
        seq_len=seq_cfg["seq_len"], forecast_horizon=seq_cfg["forecast_horizon"],
        stride=seq_cfg.get("stride", 1),
        train_end=train_cfg["train_end"],
        valid_window_days=train_cfg["valid_window_days"],
        test_window_days=train_cfg["test_window_days"],
        embargo_gap=seq_cfg["forecast_horizon"],
    )

    metrics, predictions = trainer._evaluate_and_build(
        splits["ds_test"], df, fold="test"
    )

    test_path = paths.get_seq_test_predictions()
    predictions.to_parquet(test_path, engine='fastparquet', index=False)

    print(f"   ✅ test_predictions: {test_path}")
    print(f"   테스트 Avg RMSE: {metrics['avg_rmse']:.6f}")
    print(f"   테스트 Avg IC:   {metrics['avg_ic']:.4f}")
    print("\n✅ eval-only 완료")


if __name__ == "__main__":
    main()
