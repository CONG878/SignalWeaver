# Schema Changelog (Updated v4.0.0)

All notable changes to the data schema will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [4.0.0] - 2026-04-01

### 🔴 MAJOR Changes — Seq 모델 트랙 신설 + 스키마 전반 정비

변경 범위: **전체 파이프라인, 모델 계층, 설정 파일, 노트북 체계**  
작성 기준일: 2026-03-25
확정 기준일: 2026-04-01
이전 안정 버전: v3.10.0 (Stable Baseline)

---

### 1. Seq 모델 트랙 신설 (핵심)

#### 1.1 신규 파일

| 파일 | 설명 |
|------|------|
| `src/models/seq_base.py` | `SeqModelBase` 추상 클래스. 3D 시퀀스 입력 시그니처. |
| `src/models/gru_model.py` | `GRUModel` 구현. PyTorch GRU, `weights.pt + config.json` 저장. |
| `src/data_loader/seq_builder.py` | 슬라이딩 윈도우 시퀀스 생성. 종목 경계 보호, NaN 자동 제거. |
| `src/modeling/seq_trainer.py` | `SeqTrainer`. WFT와 독립. 동일 평가 파라미터로 기간 일치. |
| `03c_train_seq.ipynb` | Seq 모델 학습 전용 노트북. |
| `scripts/train_seq.py` | Seq 모델 학습용 스크립트. |

#### 1.2 모델 저장 포맷 (GRUModel)

pickle 미사용. 디렉토리 단위 저장.

```
data/03_seq/{model_date}/gru/
├── weights.pt      torch.save(state_dict)
└── config.json     아키텍처 + 학습 설정 + 메타데이터
```

기존 Tabular 모델(pickle)은 v4.1.0에서 마이그레이션 예정.

#### 1.3 val/test_predictions.parquet 컬럼 규격 통일

Seq 트랙 산출물이 Tabular 트랙과 동일한 컬럼 규격을 따릅니다.
05단계 `select_investment_universe()`가 구분 없이 처리 가능합니다.

```python
# Tabular (forecast_horizon=5)
['date', 'ticker', 'fold',
 'pred_target_log_return_1d_h1', ..., 'pred_target_log_return_1d_h5',
 'true_target_log_return_1d_h1', ..., 'true_target_log_return_1d_h5']

# Seq (forecast_horizon=20)
['date', 'ticker', 'fold',
 'pred_target_log_return_1d_h1', ..., 'pred_target_log_return_1d_h20',
 'true_target_log_return_1d_h1', ..., 'true_target_log_return_1d_h20']
```

#### 1.4 04단계 GRU 예측 경로

`is_seq_model(active_model)` 분기로 Tabular / Seq 경로 자동 선택.  
입력 구성 방식만 다르며 역산 로직(`trapezoid_log_close`)은 동일.

#### 1.5 메모리 최적화 (On-the-fly SeqDataset)
초기 `build_sequences()` 방식의 메모리 폭증(OOM) 문제를 해결하기 위해 `SeqDataset`을 전면 재설계했습니다 (v4.0.0 rev2). 
전체 텐서를 미리 메모리에 적재하지 않고, `(ticker, start_idx)` 인덱스 목록만 보관한 뒤 배치 요청 시 원본 DataFrame에서 즉석(On-the-fly)으로 슬라이스를 추출합니다.

#### 1.6 장시간 학습용 CLI 스크립트 (`scripts/train_seq.py`)
Jupyter 커널의 메모리 해제 지연 및 장기 학습 불안정성을 극복하기 위해 독립된 OS 프로세스로 동작하는 학습 스크립트를 추가했습니다.
- `--resume`: `checkpoint.pt`에서 학습 이어서 진행
- `--eval-only`: 학습 없이 기존 가중치(`weights.pt`)로 평가 및 예측 파일만 재생성
- `--n-folds`: 1(기본) / 2(v4.1.0 앙상블 대비용)

#### 1.7 Parquet 저장 엔진 최적화 (`fastparquet`)
Seq 트랙의 3D 텐서 기반 예측 결과를 `pyarrow`로 저장 시 발생하는 메타데이터 충돌(`Repetition level histogram size mismatch`)을 해결하기 위해, Seq 예측 산출물의 I/O 표준 엔진으로 호환성이 뛰어난 `fastparquet`을 정식 채택했습니다.

---

### 2. 노트북 번호 체계 개편

#### 2.1 접미사 규칙

- **접미사 생략**: 해당 번호 내 최상위가 유일한 경우.
- **접미사 표기**: 최상위가 복수이거나 대등한 관계인 경우.

#### 2.2 개명 목록

| 변경 전 | 변경 후 |
|---------|---------|
| `99_save_trading_days.ipynb` | `00a_save_trading_days.ipynb` |
| `98_save_macro_data.ipynb` | `00b_save_macro_data.ipynb` |
| `97_forecast_macro.ipynb` | `00c_forecast_macro.ipynb` |
| `03_train_predict.ipynb` | `03a_train_tabular.ipynb` |
| *(없음)* | `03c_train_seq.ipynb` ← 신규 |

실행 순서가 번호순과 일치합니다.

---

### 3. 파일명 정리

#### 3.1 01단계 산출물 날짜 중복 제거

경로가 이미 날짜를 포함하므로 파일명에서 날짜 제거.

| 변경 전 | 변경 후 |
|---------|---------|
| `krx_prices_{YYYYMMDD}.parquet` | `prices.parquet` |
| `ticker_master_{YYYYMMDD}.csv` | `ticker_master.csv` |

`ProjectPaths.get_raw_parquet()`, `get_ticker_master()` 수정으로 완결.

#### 3.2 모델 파일명 `v1` 제거

`registry.json`이 버전 관리를 담당.

| 변경 전 | 변경 후 |
|---------|---------|
| `{YYYYMMDD}_v1_{param_hash}.pkl` | `{YYYYMMDD}_{param_hash}.pkl` |

---

### 4. 컬럼명 재정비

#### 4.1 change_pct 자동 검증

01단계 수집 완료 후 등락률 컬럼을 자동 검증합니다.

- 단순 등락률 → `change_rate`
- 퍼센트 등락률 → `change_pct` (현행 유지)

검증 로직: `src/data_loader/collector._detect_change_col_type()`

파급 범위: `collector.py`, `02_build_dataset.ipynb`, `builder.py`, `04_forecast_future.ipynb`

---

### 5. config.yaml 수정

#### 5.1 신규 섹션

```yaml
sequence:
  seq_len: 60
  forecast_horizon: 20
  stride: 1
  target_type: "log_return_1d"

gru_params:
  hidden_size: 128
  num_layers: 2
  dropout: 0.2
  learning_rate: 0.001
  batch_size: 256
  epochs: 100
  patience: 10
  bidirectional: false

active_seq_model: "gru"

paths:
  seq_dir: "data/03_seq"    # 신규
```

#### 5.2 제거

```yaml
# 삭제: training_dir과 중복
paths:
  model_dir: "data/03_training"   # ← 제거
```

#### 5.3 주석 보강

`universe.model_date`와 `project.reference_date`의 역할 차이를 config.yaml 주석으로 명시.

---

### 6. src/utils/config.py 수정

```python
# 신규
_SEQ_MODEL_NAMES = frozenset({"gru", "lstm"})

def is_seq_model(active_model_str: str) -> bool:
    """is_ensemble()과 대칭. 순수 seq 모델 단독 여부 판단."""
    ...

# 모델 alias에 seq 모델 추가
_MODEL_ALIAS["gru"]  = ("gru",  "gru")
_MODEL_ALIAS["lstm"] = ("lstm", "lstm")

# ProjectPaths 신규 필드/메서드
seq_dir: Path
get_seq_model_dir() -> Path
get_seq_val_predictions() -> Path
get_seq_test_predictions() -> Path

# 수정 메서드
get_raw_parquet()  → "prices.parquet"       (날짜 중복 제거)
get_ticker_master() → "ticker_master.csv"   (날짜 중복 제거)
get_test_predictions_parquet()              (구: get_predictions_parquet)
```

---

### 7. ⑤ 추가 정비 (스키마)

| 항목 | 처리 |
|------|------|
| `paths.model_dir` 중복 | config.yaml에서 제거. `training_dir`으로 통일. |
| `get_predictions_parquet()` 명칭 모호 | `get_test_predictions_parquet()`으로 개명. |
| `universe.model_date` 역할 | config.yaml 주석으로 명확화. 통합하지 않음. |

---

### 8. 보류 항목 → v4.1.0

| 항목 |
|------|
| EnsembleModel에 Seq 모델 통합 (Option A) |
| 기존 모델 pickle → 포맷별 마이그레이션 (LGBM→.txt, RF→joblib, MLP→state_dict) |

---

### 호환성 노트

**v3.x → v4.0.0 마이그레이션**

재실행이 필요한 항목:
- 01단계: `prices.parquet`, `ticker_master.csv`로 파일명 변경. 기존 파일은 수동 rename 또는 재수집.
- 03a 이후: `get_predictions_parquet()` → `get_test_predictions_parquet()` 참조 코드 수정.
- 노트북 개명: 내부 참조 문구 확인 후 수정.

재실행 불필요:
- `data/02_processed/`, `data/03_training/` 기존 산출물은 스키마 변경 없음.
- Tabular 모델 (.pkl) 기존 파일은 그대로 사용 가능.

---

## [3.10.0] - 2026-03-16

### 🟢 MINOR Changes — 사다리꼴 모듈화 + pred_log_return 스키마 분리 + 앙상블 버그 수정

변경 범위: **02단계, 03단계, 03b단계, 04단계, 05단계, src/features/builder.py, src/modeling/trainer.py, src/universe/select_universe.py**  
신규 파일: **src/utils/trapezoidal.py**  
02단계 스키마 변경 있음 (feature_risk_composite 추가). 전체 재실행 필요.

---

#### 1. `src/utils/trapezoidal.py` 신설 — 사다리꼴 적분 모듈화

**배경:** v3.9.1에서 도입된 사다리꼴 역산 수식이 03단계 Trainer 내부(`_evaluate`), 03단계 노트북 finalize 셀, 04단계 Recursive Extension 루프, 05단계 `evaluate_model_accuracy()` 네 곳에 중복 정의되어 있었습니다. 수식 변경 시 네 곳을 동시에 수정해야 하는 유지보수 리스크가 있었습니다.

**수정 내용:**
- `src/utils/trapezoidal.py` 신설. `trapezoid_log_close(log_close_base, cum_pred, delta_y_t, delta_y_h)` 함수 정의.
- numpy 브로드캐스팅을 활용하므로 scalar, array, Series 모두 동일하게 처리.
- 중복 수식이 있던 네 곳 모두 이 함수 호출로 교체.

#### 2. 03b단계 `log_return_1d` 타겟 미인식 버그 수정

**배경:** v3.9.0에서 `log_return_1d` 타겟이 신설되었으나, `03b_train_ensemble.ipynb`의 `target_prefix` 결정 로직이 `target_type`과 무관하게 `'pred_target_log_close_h'`로 하드코딩되어 있었습니다. `log_return_1d` 모드로 학습한 단일 모델의 `val_predictions.parquet`를 읽으면 `pred_cols`가 빈 리스트가 되고, 이후 최적화가 의미 없는 결과를 반환했습니다.

**수정 내용:**
- `target_prefix`를 `target_type`에 따라 동적으로 결정:
  - `log_return_1d` → `'pred_target_log_return_1d_h'`
  - `log_close`     → `'pred_target_log_close_h'`
- `pred_cols`가 빈 리스트인 경우 명시적 `KeyError` 발생 및 진단 메시지 출력.

#### 3. 03b단계 `base_canonical` 취약점 수정

**배경:** 앙상블 예측 결과를 저장할 때 첫 번째 구성 모델의 DataFrame을 통째로 `copy()` 후 `pred_` 컬럼만 덮어쓰는 방식을 사용했습니다. 구성 모델 순서가 바뀌거나 단일 모델 간 스키마가 달라질 경우, 05단계에서 잘못된 메타값(`fold`, `close`, `target_*` 등)을 조용히 참조할 수 있었습니다.

**수정 내용:**
- `_build_ensemble_df()` 헬퍼 함수를 신설. `date`, `ticker`, `fold`, `true_*` 컬럼만 명시적으로 추출해 새 DataFrame을 구성하고 `pred_` 컬럼을 직접 채움.
- val과 test 두 곳 모두 동일하게 적용.

#### 4. `pred_log_return` 스키마 분리 (`04_forecast_future.ipynb`)

**배경:** `future_forecasts.parquet`에 `log_return_1d` 모드의 raw 예측값인 `pred_log_return`이 "참고용"으로 포함되어 있었습니다. 공식 산출 스키마에 디버깅용 컬럼이 혼재하면 05단계 등 하위 스텝에서 실수로 이 컬럼을 참조할 위험이 있었습니다.

**수정 내용:**
- 04단계 예측 루프의 `row` 딕셔너리 구성에서 `pred_log_return` 키를 기본 제거.
- `config.yaml`에 `debug.save_raw_predictions: true` 플래그를 추가하면 필요 시에만 포함되도록 분리.
- `future_forecasts.parquet` 공식 스키마: `date`, `ticker`, `horizon`, `chunk_idx`, `pred_log_close`, `pred_close`.

#### 5. `risk_composite` 역할 분리 (`src/features/builder.py`, `04_forecast_future.ipynb`)

**배경:** `risk_composite`가 모델 학습 피처와 Universe 선정 운영 메타의 두 역할을 `feature_` 접두사 없이 겸하고 있었습니다. 03단계 `feature_cols` 정의에서 `or c in ['risk_composite']` 예외 처리가 필요했으며, 이는 개발 초기 잔재였습니다.

**수정 내용:**
- `build_universe_meta()`에서 동일한 값으로 두 컬럼을 명시적으로 생성: `feature_risk_composite`(모델 학습 피처, `feature_` 접두사), `risk_composite`(운영 메타, 접두사 없음, Universe 선정 전용).
- `04_forecast_future.ipynb`의 `calculate_features_for_ticker()`도 동일하게 적용.
- `03_train_predict.ipynb`의 `feature_cols` 정의에서 `or c in ['risk_composite']` 예외 제거.

#### 6. Directional Accuracy 추가 (`src/universe/select_universe.py`, `05_universe_selection.ipynb`)

**배경:** 기획서 8.2에 명시된 방향성 예측 정확도가 미구현 상태였습니다.

**수정 내용:**
- `evaluate_model_accuracy()`에서 종목별 `directional_accuracy` 산출. 인접 시점 간 방향 일치율(`sign(pred 변화) == sign(true 변화)`).
- `investment_report.xlsx`의 정확도 섹션에 `방향성정확도` 컬럼 추가.

#### 7. Top-k Precision 추가 (`05_universe_selection.ipynb`)

**배경:** 기획서 8.2에 명시된 상위 추천 종목의 실현 수익 비율 지표가 미구현 상태였습니다.

**수정 내용:**
- 05단계 노트북에 6️⃣번 셀 신설. `test_predictions.parquet`만으로 계산 가능하므로 추가 데이터 수집 불필요.
- K=10, 20, 50, 100, 전체 후보 각각에 대해 Precision과 무작위 선택 기준선을 함께 출력.

---

## [3.9.2] - 2026-03-16

### 🔵 PATCH Changes — 코드·문서 정합성 패치 (스키마 변경 없음)

변경 범위: **02단계, 03단계, 05단계, 99단계, src/utils/risk.py, src/models/artifact.py**  
산출물 스키마 변경 없음. 기존 parquet·모델 파일 재실행 불필요.

---

#### 1. 위험도 지표 정의 통일 (`src/utils/risk.py`)

**배경:** 기획서의 5대 표준 지표(Volatility/Downside Risk/VaR/CVaR/MDD)와 달리
`calculate_composite_risk_score()`에 VaR 대신 Kurtosis가 포함되어 있었습니다.

**수정 내용:**
- `kurtosis_pos` (0.10) 제거 → `var_abs` (0.20) 추가
- 연쇄 조정: `cvar_abs` 0.20→0.15, `mdd_abs` 0.15→0.10 (합계 1.00 유지)

#### 2. 인간 검수 리포트 개편 (`05_universe_selection.ipynb`)

**배경:** `risk_composite_raw` 단일 점수가 5개 지표를 압축하여 정보 손실을 유발했습니다.

**수정 내용:**
- `리스크점수` 컬럼 제거
- `하방위험`, `VaR(95%)`, `CVaR(95%)` 개별 컬럼 추가
- Excel Sheet 3 위험등급 분류 기준: `리스크점수` → `변동성` 분위수(4구간)

#### 3. `filter_statistics.json` 저장 누락 수정 (`05_universe_selection.ipynb`)

**수정 내용:** 저장 셀(5️⃣)에 `json.dump(filter_stats, ...)` 블록 추가

#### 4. `99_save_trading_days` 가드 제거 및 경고 억제

**수정 내용:**
- `if __name__` 블록 제거, `update_market_calendar()` 호출을 독립 셀로 분리
- `get_calendar()` 호출을 `warnings.catch_warnings()` 컨텍스트로 감싸 경고 억제

#### 5. 수동 배치 파일 명시 (`02_build_dataset.ipynb`)

**수정 내용:** Fallback 설명 셀에 수동 준비 안내 및 코드 예시 추가

#### 6. `param_hash` 문서화 (`src/models/artifact.py`, `03_train_predict.ipynb`)

**수정 내용:**
- `save_model_artifact()` docstring에 `Notes` 섹션 추가 (hash 충돌 위험 명시)
- `03_train_predict.ipynb` 호출부에 `"hyperparameters"` 키 명시 추가

---

## [3.9.1] - 2026-03-09

### 🔵 PATCH Changes — log_return_1d 역산 보정 (사다리꼴 적분)

변경 범위: **04단계 (Forecasts), 05단계 (Universe)**

#### 1. log_return_1d 역산 로직 개선

**수정 내용:**
- **역산 공식 변경**: `y(t+k) = y(t) + ΣΔy_i + (Δy(t) - Δy(t+k))/2`
- **앵커 컬럼 추가**: 사다리꼴 보정의 앵커가 되는 `Δy(t)`를 `log_close_ref`에 포함하여 05단계에 전달.

---

## [3.9.0] - 2026-03-09

### 🟢 MINOR Changes — log_return_1d 타겟 도입 및 log_return 폐기

변경 범위: **02단계 (Feature), 03단계 (Training), 04단계 (Forecasts), 05단계 (Universe)**

#### 1. log_return_1d 모드 신설 및 타겟 스키마 변경

**수정 내용:**
- **신규 타겟**: `target_log_return_1d_h{n}` (v3.9.0 신규).
- **정식 폐기**: `target_log_return_h{n}` (누적 로그 수익률) 물리적 삭제.
- **보고 지표 스케일 통일**: raw log return 스케일 그대로 저장, 평가 시 변환.

---

## [3.8.1] - 2026-03-05

### 🔵 PATCH Changes — API Fallback 및 datetime 타입 버그 수정

---

## [3.8.0] - 2026-02-28

### 🟢 MINOR Changes — MLP 모델 도입 및 앙상블 확장성

---

## [3.7.2] - 2026-02-25

### 🔵 PATCH Changes — 비현실적 수익률 거래 필터링

---

## [3.7.1] - 2026-02-24

### 🔵 PATCH Changes — 04단계 버그 수정 및 97단계 신설

---

## [3.7.0] - 2026-02-22

### 🟢 MINOR Changes - log_close 롤백 + Embargo Gap

---

## [3.6.0] - 2026-02-21

### 🟢 MINOR Changes - Scale-Invariant Features & IC Evaluation

---

## [3.5.0] - 2026-02-20

### 🟢 MINOR Changes - Training Pipeline Improvement

---

## [3.4.0] - 2026-02-17

### 🟢 MINOR Changes - Model Diversification

---

## [3.3.0] - 2026-02-09

### 🟢 MINOR Changes - Infrastructure Modernization

---

## [3.2.1] - 2026-02-09

### 🔵 PATCH Changes

---

## [3.2.0] - 2026-02-07

### 🟢 MINOR Changes

---

## [3.1.x] / [3.0.0]

- **3.1.1/3.1.0**: Target 생성 위치 02단계 이동 및 Multi-Horizon Direct Forecasting 도입.
- **3.0.0**: Target 생성 위치 변경 (01 → 02).

---

## [2.0.0] / [1.0.0]

- **2.0.0**: Feature Prefix 통일 (`feature_` 접두어).
- **1.0.0**: Initial Release (수집 → Feature → LightGBM 학습).

---

### 버전별 변경 이력

| Version | Date | Type | 주요 변경 사항 |
|---------|------|------|----------------|
| **4.0.0** | 2026-04-01 | 🔴 MAJOR | Seq 모델(GRU) 트랙 신설, 파이프라인 스키마 개편 및 메모리/I/O 최적화 |
| **3.10.0** | 2026-03-16 | 🟢 MINOR | 사다리꼴 모듈화 + 03b 버그·취약점 수정 + pred_log_return 분리 + risk_composite 역할 분리 + Directional Accuracy + Top-k Precision |
| **3.9.2** | 2026-03-16 | 🔵 PATCH | 위험도 지표 통일 + 리포트 개편 + 필터 통계 저장 + 노트북 구조·문서화 |
| **3.9.1** | 2026-03-09 | 🔵 PATCH | log_return_1d 역산: 사다리꼴 적분 보정 (오차 감소) |
| **3.9.0** | 2026-03-09 | 🟢 MINOR | log_return_1d 타겟 신규 추가 + log_return 정식 폐기 |
| **3.8.1** | 2026-03-05 | 🔵 PATCH | API Fallback 강화 및 datetime 타입 버그 수정 |
| **3.8.0** | 2026-02-28 | 🟢 MINOR | MLP 모델 도입 + 앙상블 조합 동적 지정 + 04단계 건너뜀 처리 |
| **3.7.2** | 2026-02-25 | 🔵 PATCH | 비현실적 수익률 필터링 (max_daily_return) |
| **3.7.1** | 2026-02-24 | 🔵 PATCH | 04단계 피처 스키마 동기화 + 97단계 신설 |
| **3.7.0** | 2026-02-22 | 🟢 MINOR | log_close 롤백 + Embargo Gap |
| **3.6.0** | 2026-02-21 | 🟢 MINOR | Scale-invariant 피처 + IC 평가 + 매크로 통합 |
| **3.5.0** | 2026-02-20 | 🟢 MINOR | 2-Fold 구조 도입 |
| **3.0.0** | 2026-01-18 | 🔴 MAJOR | Target 생성 및 위치 변경 |

---

**Last Updated**: 2026-04-01
**Schema Version**: 4.0.0
**Status**: ✅ Confirmed
**Maintained by**: SignalWeaver Team
