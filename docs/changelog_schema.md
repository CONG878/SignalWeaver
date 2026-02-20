# Schema Changelog (Updated v3.5.0)

All notable changes to the data schema will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.5.0] - 2026-02-20

### 🟢 MINOR Changes - Training Pipeline Improvement

변경 범위: **03단계 (Training), 04단계 (Forecasts)**
02단계 및 05단계 스키마 변경 없음.

---

#### 1. 검증/테스트 분리 — 2-Fold Walk-Forward 구조

**배경:**
기존 `num_valid=3` 구조에서 검증 폴드(Fold 0~1)의 예측값이 버려지고, 앙상블 가중치 최적화가 테스트셋(Fold 2)으로 수행되어 데이터 누수가 발생하고 있었습니다.

**변경 사항:**

`num_valid` 파라미터 제거 → 2-Fold 고정 구조 채택

```
Before (v3.4.0, num_valid=3):
  Fold 0 (Valid-1): train [0,E],   eval [E, E+V]   → 예측값 버림
  Fold 1 (Valid-2): train [V,E+V], eval [E+V,E+2V] → 예측값 버림
  Fold 2 (TEST)   : train [2V,E+2V], eval [E+2V,E+3V]
                    → predictions.parquet (단일 파일)
                    → 앙상블 가중치 최적화 입력 (테스트셋 누수)

After (v3.5.0):
  [검증 폴드]: train [0, E],   valid [E, E+V]
               → val_predictions.parquet (앙상블 가중치용)
               → Early stopping 기준
  [테스트 폴드]: train [V, E+V], test [E+V, E+V+T]
                → test_predictions.parquet (최종 평가 전용)
                → final_model
```

**파일 구조 변경:**

```
# Before (v3.4.0)
data/03_training/{date}/{model_name}/
  └── predictions.parquet          # 단일 예측 파일

# After (v3.5.0)
data/03_training/{date}/{model_name}/
  ├── val_predictions.parquet      # ✨ 검증 폴드 (앙상블 가중치 최적화 입력)
  └── test_predictions.parquet     # ✨ 테스트 폴드 (최종 평가 전용)
```

**예측 파일 컬럼 구성 (log_return 모드):**

```python
# val_predictions.parquet / test_predictions.parquet 공통
columns = [
    'date', 'ticker', 'fold',                    # 메타 (fold='valid'|'test')

    # 모델 원시 예측 (log_return)
    'pred_target_log_return_h1',
    'pred_target_log_return_h2',
    'pred_target_log_return_h3',
    'pred_target_log_return_h4',
    'pred_target_log_return_h5',

    # 정답값
    'true_target_log_return_h1',
    # ... h2~h5 동일

    # 역산값 (target_log_close 조인 후 생성)
    'pred_log_close_target_log_return_h1',
    'pred_close_target_log_return_h1',
    # ... h2~h5 동일
]
```

**03b_train_ensemble.ipynb 변경:**

```python
# Before (v3.4.0): 테스트셋으로 가중치 최적화 (데이터 누수)
df_lgbm = pd.read_parquet(".../predictions.parquet")
weights = minimize(rmse, ...)  # 테스트셋 기반

# After (v3.5.0): 검증셋으로 가중치 최적화 → 테스트셋은 순수 평가
df_lgbm_val = pd.read_parquet(".../val_predictions.parquet")
weights = minimize(rmse, ...)  # 검증셋 기반

df_lgbm_test = pd.read_parquet(".../test_predictions.parquet")
# 테스트셋 성능 확인 (가중치 변경 없음)
```

**설정 변경 (config.yaml):**

```yaml
# Before (v3.4.0)
training:
  num_valid: 3           # ← 제거

# After (v3.5.0)
training:
  valid_window_days: 60  # 검증 윈도우 (거래일)
  test_window_days: 60   # 테스트 윈도우 (거래일)
  # num_valid 제거됨
```

**수정된 파일:**
- 🔧 `src/modeling/trainer.py` — 2-Fold 고정 구조, `val_predictions` 저장 추가, 폴드 기간 출력
- 🔧 `03_train_predict.ipynb` — 저장 셀: `val_predictions.parquet` + `test_predictions.parquet` 분리
- 🔧 `03b_train_ensemble.ipynb` — 가중치 최적화 입력 교체, 테스트셋 평가 셀 신규 추가
- 🔧 `config/config.yaml` — `num_valid` 제거

---

#### 2. log_return 타겟 전환

**배경:**
기존 `target_log_close`는 비정상(non-stationary) 시계열로, 모델이 절대 가격 수준을 기억해야 하는 trivial solution으로 수렴할 위험이 있었습니다. log_return은 0 근방에서 안정적인 정상 시계열이므로 패턴 학습에 유리합니다.

**수학적 정의:**

```
target_log_return_h{n}(t) = log(close(t+n)) - log(close(t))
                           = target_log_close(t+n) - target_log_close(t)
                           = 누적 로그 수익률

역산:
  pred_log_close(t+n) = target_log_close(t) + pred_log_return_h{n}
  pred_close(t+n)     = exp(pred_log_close(t+n))
```

**타겟 컬럼명 변경:**

```
Before (v3.4.0): target_log_close_h{n}    (비정상 시계열)
After  (v3.5.0): target_log_return_h{n}   (정상 시계열)
```

**02단계 변경 없음:**
`target_log_close` 컬럼은 02단계에 그대로 유지됩니다. Trainer가 이 컬럼을 기준값으로 읽어 horizon별 log_return을 동적으로 계산합니다.

**03단계 Trainer 변경:**

```python
# Before (v3.4.0): log_close 방식
col_name = f"target_log_close_h{h}"
df_run[col_name] = df_run.groupby('ticker')['target_log_close'].shift(-h)

# After (v3.5.0): log_return 방식
col_name = f"target_log_return_h{h}"
future_log_close  = df_run.groupby('ticker')['target_log_close'].shift(-h)
df_run[col_name]  = future_log_close - df_run['target_log_close']
```

**04단계 Recursive Extension 변경:**

```python
# Before (v3.4.0)
pred_log_close = model.predict(X, target_name=target_name).iloc[0]
pred_close     = exp(pred_log_close)

# After (v3.5.0)
log_close_base  = log(latest_row['close'])          # chunk 직전 close
pred_log_return = model.predict(X, target_name=...) # 모델 원시 출력
pred_log_close  = log_close_base + pred_log_return  # 역산
pred_close      = exp(pred_log_close)

# ⚠️ target_name 순회: model.target_columns 대신 sorted(model.models.keys()) 사용
```

**future_forecasts.parquet 컬럼 추가:**

```python
# v3.5.0 신규 컬럼 (log_return 모드에서만 생성)
'pred_log_return'   # 모델 원시 출력값 보존

# 기존 컬럼 유지 (하위 호환)
'pred_log_close'    # 역산된 로그 종가
'pred_close'        # 역산된 종가 (원화)
```

**설정 추가 (config.yaml):**

```yaml
training:
  target_col_name: "target_log_close"  # 02단계 기준값 컬럼 (변경 없음)
  target_type: "log_return"            # ✨ 신규: "log_return" | "log_close"
```

**수정된 파일:**
- 🔧 `src/modeling/trainer.py` — `target_type` 파라미터 추가, `_build_target_cols()` 분기
- 🔧 `03_train_predict.ipynb` — Trainer 초기화에 `target_type` 전달, 저장 셀 역산 로직
- 🔧 `03b_train_ensemble.ipynb` — `pred_cols` 필터링: `pred_target_log_return_h` prefix 기준
- 🔧 `04_forecast_future.ipynb` — Recursive Extension 역산 로직, `sorted(model.models.keys())` 순회
- 🔧 `config/config.yaml` — `target_type: "log_return"` 추가

---

### 호환성

**비호환 (재실행 필요):**
- 03단계 모델(.pkl): `models.keys()`가 `target_log_close_h{n}` → v3.5.0 코드에서 ValueError
- 03단계 예측 파일: `predictions.parquet` → `val_predictions.parquet` + `test_predictions.parquet`
- 03b: 가중치 최적화 입력 파일명 변경

**호환 (재실행 불필요):**
- 01단계 Raw 데이터
- 02단계 Feature 데이터셋 (`target_log_close` 컬럼 유지)
- 04단계 출력 컬럼명 (`pred_log_close`, `pred_close` 유지)
- 05단계 입력 스키마

---

## [3.4.0] - 2026-02-17

### 🟢 MINOR Changes - Model Diversification

#### 1. 모델 다양화 (RandomForest + Ensemble)

**변경 사항**:
- RandomForest 멀티아웃풋 모델 추가
- 앙상블 학습 단계(03b_train_ensemble.ipynb) 도입
- 모델별 폴더 계층 추가: `03_training/{date}/{model_name}/`
- config: `active_model` 옵션 추가

**신규 추가 파일**:
- `src/models/randomforest_model.py` — RandomForestMultiModel
- `src/models/ensemble_model.py` — EnsembleModel
- `03b_train_ensemble.ipynb` — OOF 기반 최적 가중치 탐색

**폴더 구조 변경**:

```
# Before (v3.3.0)
data/03_training/{YYYYMMDD}/
  ├── *.pkl
  ├── registry.json
  └── predictions.parquet

# After (v3.4.0)
data/03_training/{YYYYMMDD}/
  ├── lightgbm/   { *.pkl, registry.json, predictions.parquet }
  ├── randomforest/ { ... }
  └── ensemble/   { ... }
```

**설정 변경** (config.yaml):
```yaml
randomforest_params: { n_estimators: 40, max_depth: 8, ... }
active_model: "ensemble"   # 'lightgbm' | 'randomforest' | 'ensemble'
```

**수정된 파일**: `src/models/base.py`, `src/models/artifact.py`, `src/utils/config.py`,
`03_train_predict.ipynb`, `04_forecast_future.ipynb`, `config/config.yaml`

**호환성**:
- ✅ v3.3.0 LightGBM 모델 자동 호환
- ❌ v3.4.0 모델은 v3.3.0 코드에서 로드 불가

---

## [3.3.0] - 2026-02-09

### 🟢 MINOR Changes - Infrastructure Modernization

#### H1 - 데이터 폴더 구조 개선

```
# Before (v3.2.x)
data/03_results/{YYYYMMDD}/
  ├── predictions.parquet
  ├── *.pkl
  ├── forecasts/future_forecasts.parquet
  └── universe/investment_report.xlsx

# After (v3.3.0)
data/
  ├── 03_training/{YYYYMMDD}/
  ├── 04_forecasts/{YYYYMMDD}/
  └── 05_universe/{YYYYMMDD}/
```

#### H2 - ProjectPaths 클래스 도입

`src/utils/config.py`에 `ProjectPaths` 클래스 추가. 모든 노트북의 경로 관리 통일.

#### H3 - select_universe.py 모듈 정리

Facade Pattern 적용, 단일 진입점 함수화.

---

## [3.2.1] - 2026-02-09

### 🔵 PATCH Changes

#### 1. Multi-Horizon Walk-Forward 버그 수정

Horizon별 dropna 후 길이 불일치 → 모든 Horizon의 교집합 인덱스 사용.

#### 2. Recursive Extension Chunk 오염 방지

Chunk 1+ 예측 시 실제 volume 참조 → 최근 20일 평균으로 대체.

---

## [3.2.0] - 2026-02-07

### 🟢 MINOR Changes - Multi-Stage Pipeline

- 04단계 추가: Recursive Extension 미래 예측
- 05단계 추가: 3대 평가 지표 기반 유니버스 선정

---

## [3.1.1] - 2026-01-21

### 🔵 PATCH Changes

Target 생성 위치 재변경: 03단계 → 02단계 (모든 모델이 동일 Target 공유)

---

## [3.1.0] - 2026-01-21

### 🟢 MINOR Changes

Multi-Horizon Direct Forecasting (h1~h5) 도입. Target-Centric Alignment.

---

## [3.0.0] - 2026-01-18

### 🔴 MAJOR Changes

Target 생성 위치 변경: 01_raw → 02_processed. Feature와의 시간 정렬 명확화.

---

## [2.0.0] - 2024-12-28

### 🔴 MAJOR Changes

Feature Prefix 통일: `ma_5` → `feature_ma_5`.

---

## [1.0.0] - 2024-12-01

Initial Release. Step 1~3 (수집 → Feature → LightGBM 학습).

---

**Last Updated**: 2026-02-20
**Maintained by**: SignalWeaver Team
