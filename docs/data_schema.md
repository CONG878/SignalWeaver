# 📄 Data Schema — v4.0.0 (Confirmed)

> **문서 유형**: 확정 스키마 (구현 완료 기준)
> **작성 기준일**: 2026-03-25
> **확정 기준일**: 2026-04-01
> **Schema Version**: 4.0.0
> **이전 안정 버전**: v3.10.0

---

## 0. v4.0.0 변경 범위 요약

| 카테고리 | 항목 | 성격 |
|----------|------|------|
| **신규** | SeqModelBase 계층 | 아키텍처 |
| **신규** | GRUModel (On-the-fly DataLoader, 체크포인트) | 모델 |
| **신규** | SeqDataset (인덱스 기반 on-the-fly) | 데이터 |
| **신규** | split_by_date() → SeqDataset 직접 반환 | 데이터 |
| **신규** | SeqTrainer (n_folds, resume) | 파이프라인 |
| **신규** | scripts/train_seq.py (03c 노트북 대체) | 실행 |
| **개명** | 노트북 번호 체계 (00계열 + a/b/c 접미사) | 스키마 |
| **개명** | 03_train_predict → 03a_train_tabular | 스키마 |
| **개명** | 99_meta → 00_prep (시간독립/날짜의존 분리) | 스키마 |
| **개명** | 01단계 산출물 파일명 날짜 중복 제거 | 스키마 |
| **개명** | 모델 파일명 v1 제거 | 스키마 |
| **개명** | change_pct → change_rate (자동 검증) | 스키마 |
| **수정** | config.yaml 대폭 확장 | 설정 |
| **수정** | ProjectPaths 전면 재편 | 인프라 |
| **수정** | get_predictions_parquet → get_test_predictions_parquet | 인프라 |
| **수정** | paths.model_dir 중복 제거 | 설정 |
| **수정** | Parquet 저장 엔진: fastparquet 명시 | 인프라 |
| **보류** | 기존 모델 pickle → 포맷별 마이그레이션 | v4.1.0 |
| **보류** | EnsembleModel Seq 모델 통합 | v4.1.0 |

---

## 1. 노트북 및 스크립트 체계

### 1.1 접미사 규칙

- **접미사 생략**: 해당 번호 내 최상위가 유일한 경우
- **접미사 표기 (a, b, c, ...)**: 최상위가 복수이거나 대등한 관계인 경우

### 1.2 확정 목록

| 파일명 | 역할 | 변경 여부 |
|--------|------|-----------|
| `00a_save_trading_days.ipynb` | 캘린더 생성 | 개명 (구: 99) |
| `00b_save_macro_data.ipynb` | 매크로 수집 | 개명 (구: 98) |
| `00c_forecast_macro.ipynb` | 매크로 미래값 추정 | 개명 (구: 97) |
| `01_collect_data.ipynb` | KRX 수집 | 유지 |
| `02_build_dataset.ipynb` | 피처 엔지니어링 | 유지 |
| `03a_train_tabular.ipynb` | Tabular 모델 학습 | 개명 (구: 03_train_predict) |
| `03b_train_ensemble.ipynb` | 앙상블 가중치 최적화 | 유지 |
| `03c_train_seq.ipynb` | Seq 모델 학습 (대화형) | **신규** |
| `scripts/train_seq.py` | Seq 모델 장기 학습용 CLI | **신규** (03c와 병존) |
| `04_forecast_future.ipynb` | 미래 예측 | 수정 (GRU 분기 추가) |
| `05_universe_selection.ipynb` | 유니버스 선정 | 유지 |

### 1.3 00 계열 구조

```
00a_save_trading_days   ─┐  (독립)
00b_save_macro_data     ─┴─→ 00c_forecast_macro
```

### 1.4 scripts/train_seq.py 사용법

```bash
# 처음부터 학습 (config.yaml의 epochs 사용)
python scripts/train_seq.py

# 에포크 수 임시 지정
python scripts/train_seq.py --epochs 20

# 체크포인트에서 이어서 학습
python scripts/train_seq.py --resume

# 이어서 + 에포크 지정 (예: epoch 11~20)
python scripts/train_seq.py --resume --epochs 10

# 학습 없이 평가만 (weights.pt 필요)
python scripts/train_seq.py --eval-only

# 2-Fold (앙상블 대비, v4.1.0)
python scripts/train_seq.py --n-folds 2
```

---

## 2. 디렉토리 구조

```
SignalWeaver/
├── config/config.yaml
├── data/
│   ├── 00_prep/
│   │   ├── krx_calendar.csv              ← 시간 독립 (긴 달력, 필터링하여 사용)
│   │   └── {ref_date}/
│   │       ├── macro_regime.parquet      ← 날짜 의존
│   │       └── macro_regime_forecast.parquet
│   ├── 01_raw/{ref_date}/
│   │   ├── prices.parquet                ← (변경) 날짜 중복 제거
│   │   ├── ticker_master.csv             ← (변경) 날짜 중복 제거
│   │   └── csv/
│   ├── 02_processed/{ref_date}/
│   │   └── dataset.parquet
│   ├── 03_training/{model_date}/{model_name}/    Tabular 트랙
│   │   ├── {YYYYMMDD}_{param_hash}.pkl           ← (변경) v1 제거
│   │   ├── registry.json
│   │   ├── val_predictions.parquet
│   │   └── test_predictions.parquet
│   ├── 03_seq/{model_date}/gru/                  Seq 트랙
│   │   ├── weights.pt                    추론 전용 최선 가중치
│   │   ├── config.json                   아키텍처 + 메타
│   │   ├── checkpoint.pt                 resume용 (학습 중 자동 갱신, 완료 후 보존)
│   │   ├── val_predictions.parquet       n_folds=2일 때만 생성
│   │   └── test_predictions.parquet
│   ├── 04_forecasts/{ref_date}/{model_name}/
│   │   └── future_forecasts.parquet
│   ├── 05_universe/{ref_date}/{model_name}/
│   │   ├── universe_candidates.parquet
│   │   ├── investment_report.csv / .xlsx
│   │   └── filter_statistics.json
├── docs/
├── scripts/
│   └── train_seq.py                      ← 신규
└── src/
    ├── data_loader/
    │   ├── collector.py                  change_pct 자동 검증
    │   └── seq_builder.py               SeqDataset, split_by_date
    ├── modeling/
    │   ├── trainer.py                   Tabular WFT (유지)
    │   └── seq_trainer.py               SeqTrainer
    ├── models/
    │   ├── base.py
    │   ├── seq_base.py                  SeqModelBase
    │   ├── gru_model.py                 GRUModel
    │   ├── lightgbm_model.py
    │   ├── randomforest_model.py
    │   ├── mlp_model.py
    │   ├── ensemble_model.py
    │   └── artifact.py                  v1 제거
    ├── universe/
    └── utils/
        ├── config.py                    ProjectPaths 전면 재편
        ├── risk.py
        ├── trading.py
        └── trapezoidal.py
```

---

## 3. 파일명 정리

### 3.1 01단계 산출물

| 변경 전 | 변경 후 |
|---------|---------|
| `krx_prices_{YYYYMMDD}.parquet` | `prices.parquet` |
| `ticker_master_{YYYYMMDD}.csv` | `ticker_master.csv` |

### 3.2 모델 파일명

| 변경 전 | 변경 후 |
|---------|---------|
| `{YYYYMMDD}_v1_{param_hash}.pkl` | `{YYYYMMDD}_{param_hash}.pkl` |

### 3.3 Seq 트랙 저장 포맷

pickle 미사용. 디렉토리 단위 저장.

```json
// config.json 구조
{
  "model_name": "gru",
  "model_version": "v4.0.0_gru_{ref_date}",
  "seq_len": 60,
  "forecast_horizon": 20,
  "n_features": 20,
  "hidden_size": 128,
  "num_layers": 2,
  "dropout": 0.2,
  "bidirectional": false,
  "feature_list": ["feature_ma_5_disparity", "..."],
  "target_type": "log_return_1d",
  "target_columns": ["target_log_return_1d_h1", "..."],
  "trained_at": "2026-04-01T00:00:00",
  "val_rmse": null,
  "test_rmse": 0.0
}
```

### 3.4 체크포인트 구조

```python
# checkpoint.pt 내용
{
    "epoch":           int,         # 마지막 학습 에포크
    "net_state":       state_dict,  # 현재 가중치
    "optimizer_state": state_dict,  # Adam 모멘텀 등
    "best_val_loss":   float,       # 지금까지 최선 val_loss
    "best_net_state":  state_dict,  # 지금까지 최선 가중치
    "no_improve":      int,         # 연속 미개선 횟수
}
```

**resume 우선순위**:
1. `checkpoint.pt` 존재 → 전체 컨텍스트 복원
2. `checkpoint.pt` 없고 `weights.pt` 존재 → 가중치만 복원 + 경고
3. 둘 다 없음 → `FileNotFoundError`

---

## 4. 컬럼명 재정비

### 4.1 change_pct 자동 검증

수집 완료 후 `collector._is_pct_scale()`로 자동 판별.

- 단순 등락률 → `change_rate` (FDR 기본)
- 퍼센트 등락률 → `change_pct` (예외적 소스)

### 4.2 접두사 체계 (변경 없음)

| 접두사 | 의미 |
|--------|------|
| `feature_` | 모델 입력 피처 |
| `target_` | 학습 타겟 |
| `pred_` | 모델 출력 예측값 |
| `true_` | 예측과 대응하는 실측값 |
| 없음 | 운영 메타 |

---

## 5. config.yaml 확정

```yaml
project:
  reference_date: "20260323"    # 데이터 수집 기준일

sequence:
  seq_len: 60
  forecast_horizon: 20
  stride: 5                      # 메모리 절감: stride=1 대비 1/5
  target_type: "log_return_1d"  # 항상 log_return_1d 권장

gru_params:
  hidden_size: 128
  num_layers: 2
  dropout: 0.2
  learning_rate: 0.001
  batch_size: 256
  epochs: 10                    # 매 실행당 추가 에포크 수
  patience: 10
  bidirectional: false

active_seq_model: "gru"

calendar:
  start_date:   "2020-01-01"   # 긴 달력 (00a 전용)
  end_date:     "2035-12-31"
  forecast_end: "2026-05-27"   # 예측 상한 (00c, 04단계 공용)

paths:
  raw_dir: "data/01_raw"
  processed_dir: "data/02_processed"
  training_dir: "data/03_training"
  seq_dir: "data/03_seq"
  forecasts_dir: "data/04_forecasts"
  universe_dir: "data/05_universe"
  prep_dir: "data/00_prep"      # 구: meta_dir: "data/99_meta"
```

---

## 6. 모델 아키텍처 계층

```
ModelBase (ABC)
├── LightGBMModel          pickle (→ v4.1.0 마이그레이션)
├── RandomForestMultiModel pickle
├── MLPModel               pickle
├── EnsembleModel          pickle
└── SeqModelBase (ABC)
    └── GRUModel           weights.pt + config.json + checkpoint.pt
```

---

## 7. Seq 트랙 핵심 설계

### 7.1 SeqDataset — On-the-fly

```python
# 메모리 사용량 비교
# 기존: O(N_samples × seq_len × n_features) float32  → 수 GB
# v4.0.0: O(원본 df) + O(N_samples × 2) int          → df 크기 + 수 MB
```

인덱스 목록 `(global_start, global_end)` 쌍만 보관. `__getitem__` 시 원본 DataFrame 슬라이스 즉석 추출.

### 7.2 split_by_date() — 실제 거래일 달력 기준 분할

```python
# stride와 무관하게 실제 거래일 기준으로 창 크기 적용
# df의 전체 날짜를 달력으로 사용 → stride=5에서도 45거래일이 정확히 45일
splits = split_by_date(df, ..., all_trading_dates=pd.DatetimeIndex(...))
# 반환: {ds_train, ds_val, ds_test: SeqDataset, dates: dict}
```

### 7.3 SeqTrainer.run() 파라미터

```python
trainer.run(
    df,
    train_end         = "2025-11-05",
    valid_window_days = 45,   # 실제 거래일 기준
    test_window_days  = 45,   # 실제 거래일 기준
    n_folds           = 1,    # 1: 권장 / 2: 앙상블 대비
    resume            = False,
    fit_kwargs        = {"epochs": 100, "patience": 10},
)
```

### 7.4 파라미터 관계 및 제약

```
[필수 데이터 길이 per ticker]
  min_history + seq_len + forecast_horizon

[충분한 데이터 범위 조건]
  data_end - train_end ≥ embargo + valid_window + test_window
  embargo = forecast_horizon

[seq_len vs forecast_horizon]
  완전히 독립. 배수 관계 불필요.
  seq_len: 모델의 "기억 창문" (과거 참조 길이)
  forecast_horizon: 모델의 "예측 창문" (출력 크기)

[stride 영향]
  샘플 수: N_tickers × (유효 날짜 수 / stride)
  날짜 경계: stride와 무관 (실제 거래일 달력 기준)
```

### 7.5 predictions.parquet 컬럼 규격

Tabular 트랙과 동일하여 05단계가 구분 없이 처리 가능.

```python
# Tabular (forecast_horizon=5)
['date', 'ticker', 'fold',
 'pred_target_log_return_1d_h1', ..., 'h5',
 'true_target_log_return_1d_h1', ..., 'h5']

# Seq (forecast_horizon=20)
['date', 'ticker', 'fold',
 'pred_target_log_return_1d_h1', ..., 'h20',
 'true_target_log_return_1d_h1', ..., 'h20']
```

---

## 8. 인프라

### 8.1 ProjectPaths 변경 요약

```python
# 제거
meta_dir        # → prep_dir / prep_dated_dir으로 분리
paths.model_dir # → training_dir로 통일

# 추가
prep_dir: Path          # data/00_prep/
prep_dated_dir: Path    # data/00_prep/{ref_date}/
seq_dir: Path           # data/03_seq/{model_date}/{seq_model}/

# 추가 메서드
get_calendar()           → prep_dir / "krx_calendar.csv"
get_macro_parquet()      → prep_dated_dir / "macro_regime.parquet"
get_macro_forecast_parquet() → prep_dated_dir / "macro_regime_forecast.parquet"
get_seq_model_dir()      → seq_dir
get_seq_val_predictions() → seq_dir / "val_predictions.parquet"
get_seq_test_predictions() → seq_dir / "test_predictions.parquet"

# 개명
get_predictions_parquet() → get_test_predictions_parquet()
get_raw_parquet()        → raw_dir / "prices.parquet"
get_ticker_master()      → raw_dir / "ticker_master.csv"
```

### 8.2 Parquet 저장 엔진

Seq 트랙 predictions는 `fastparquet` 엔진 필요.

```python
df.to_parquet(path, index=False, engine="fastparquet")
```

Tabular 트랙은 기본 엔진(`pyarrow`) 유지. 혼용 시 읽기 엔진을 통일해야 합니다.

### 8.3 04단계 NaN 처리

Seq 트랙 예측 시 입력 시퀀스 윈도우에 NaN이 있을 수 있음. `ffill → bfill → fillna(0)` 적용.

```python
X_window_df = X_window_df.ffill().bfill().fillna(0)
```

### 8.4 config.py 신규 헬퍼

```python
_SEQ_MODEL_NAMES = frozenset({"gru", "lstm"})

def is_seq_model(active_model_str: str) -> bool:
    """순수 seq 모델 단독 여부. is_ensemble()과 대칭."""
```

---

## 9. 달력 운용 정책

```
krx_calendar.csv: 시간 독립, data/00_prep/ 직접 위치
  → 장기 범위로 한 번 생성, 코드에서 필터링
  → calendar.start_date / end_date: 달력 생성 범위 (00a 전용)
  → calendar.forecast_end: 예측 상한 (00c, 04단계 공용)

macro_regime.parquet: 날짜 의존, data/00_prep/{ref_date}/
macro_regime_forecast.parquet: 날짜 의존, data/00_prep/{ref_date}/
```

---

## 10. 보류 항목 → v4.1.0

| 항목 |
|------|
| EnsembleModel Seq 모델 통합 (Tabular + Seq 혼합 앙상블) |
| 기존 모델 pickle → 포맷별 마이그레이션 (LGBM→.txt, RF→joblib, MLP→state_dict) |
| `train_seq.py`의 원자적 Parquet 쓰기 (tmp → rename) |

---

## 11. 버전 이력

```
v3.10.0  2026-03-16  MINOR   Stable Baseline (Seq 트랙 도입 전 기준점)
v4.0.0   2026-04-01  MAJOR   Seq 모델 트랙 + 스키마 전반 정비 (확정)
v4.1.0   미정        MINOR   EnsembleModel Seq 통합, pickle 마이그레이션
```

---

*Schema Version: 4.0.0*
*Status: ✅ Confirmed*
*Maintained by: SignalWeaver Team*
