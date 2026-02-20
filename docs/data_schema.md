# 📄 Data Schema Definition (v3.5.0)

본 스키마는 SignalWeaver 프로젝트의 데이터 계약을 정의합니다.

---

## 📌 Schema Version & Metadata

| 속성 | 값 |
|------|-----|
| **Schema Version** | `3.5.0` |
| **Last Updated** | 2026-02-20 |
| **Latest Changes** | 학습 파이프라인 개선 — 2-Fold 구조 + log_return 타겟 전환 |
| **Compatibility** | v3.4.x 비호환 (타겟 컬럼명 변경, 예측 파일 구조 변경) |

---

## 🔄 최근 변경 이력 요약

### v3.5.0 (2026-02-20) - 🟢 MINOR
- **검증/테스트 분리 (2-Fold 구조)**: `num_valid` 파라미터 제거, 검증 폴드 예측 결과를 별도 파일로 저장
  - `predictions.parquet` → `val_predictions.parquet` + `test_predictions.parquet` 분리
  - 앙상블 가중치 최적화 입력: 테스트셋 → 검증셋으로 변경 (데이터 누수 제거)
- **log_return 타겟 전환**: 학습 타겟을 비정상 시계열(log_close)에서 정상 시계열(log_return)으로 교체
  - 타겟 컬럼명: `target_log_close_h{n}` → `target_log_return_h{n}`
  - 예측 출력에 `pred_log_close_*`, `pred_close_*` 역산 컬럼 추가
  - 04단계 Recursive Extension에서 `log_close_base + pred_log_return`으로 역산

### v3.4.0 (2026-02-17) - 🟢 MINOR
- **모델 다양화**: RandomForest 모델 + 앙상블 학습 도입
  - LightGBM 외 RandomForest 멀티아웃풋 모델 추가
  - 03b_train_ensemble.ipynb: OOF 기반 가중치 최적화
  - `03_training/{date}/` → `03_training/{date}/{model_name}/` 계층 확장
  - config.yaml: `randomforest_params` + `active_model` 옵션 추가

### v3.3.0 (2026-02-09) - 🟢 MINOR
- **H1**: `data/03_results/` 분해 → `03_training/`, `04_forecasts/`, `05_universe/` 독립
- **H2**: `ProjectPaths` 클래스 도입 → 경로 관리 통일
- **H3**: `src/universe/select_universe.py` Facade Pattern 적용

---

## 📌 1. 파일 저장 규칙 / 포맷 (Updated v3.5.0)

### 1.1 기본 포맷

| 단계 | 폴더 | 포맷 | 이유 |
|------|------|------|------|
| **01단계 (Raw)** | `data/01_raw/{date}/` | CSV + 통합 Parquet | API 원본 보존 + 파이프라인 효율성 |
| **02단계 (Processed)** | `data/02_processed/{date}/` | Parquet + 선택적 CSV | 고속 I/O + 디버깅 지원 |
| **03단계 (Training)** | `data/03_training/{date}/{model_name}/` | Parquet + 모델별 폴더 | 모델 분리 + 메타데이터 관리 |
| **04단계 (Forecasts)** | `data/04_forecasts/{date}/` | Parquet + 선택적 CSV | 미래 예측값 저장 |
| **05단계 (Universe)** | `data/05_universe/{date}/` | Parquet + CSV + JSON | 투자 후보 선정 결과 |

### 1.2 파일 네이밍 규칙 (v3.5.0)

```
# 01단계: 원시 데이터
data/01_raw/{YYYYMMDD}/
  ├── krx_prices_{YYYYMMDD}.parquet    # 통합 원시 데이터
  ├── ticker_master_{YYYYMMDD}.csv     # 종목 메타
  └── csv/{종목명}.csv                  # 개별 CSV (옵션)

# 02단계: 전처리 데이터
data/02_processed/{YYYYMMDD}/
  ├── dataset.parquet                   # 통합 Feature 데이터셋
  └── csv/{종목명}.csv                  # 개별 CSV (옵션)

# 03단계: 학습 검증 예측 (✨ v3.5.0: 예측 파일 2개로 분리)
data/03_training/{YYYYMMDD}/
  ├── lightgbm/
  │   ├── v1_lgbm_{YYYYMMDD}_{hash}.pkl  # LightGBM Booster 객체
  │   ├── registry.json                   # 메타데이터
  │   ├── val_predictions.parquet         # ✨ 검증 폴드 예측 (앙상블 가중치용)
  │   └── test_predictions.parquet        # ✨ 테스트 폴드 예측 (최종 평가용)
  │
  ├── randomforest/
  │   ├── v1_rf_{YYYYMMDD}_{hash}.pkl
  │   ├── registry.json
  │   ├── val_predictions.parquet
  │   └── test_predictions.parquet
  │
  └── ensemble/
      ├── v1_ens_{YYYYMMDD}_{hash}.pkl   # EnsembleModel 객체 (가중치 포함)
      ├── registry.json
      ├── val_predictions.parquet
      └── test_predictions.parquet

# 04단계: 미래 예측
data/04_forecasts/{YYYYMMDD}/
  ├── lightgbm/
  │   ├── future_forecasts.parquet
  │   └── csv/{종목명}_forecast.csv       # 종목별 CSV (옵션)
  ├── randomforest/
  │   ├── future_forecasts.parquet
  │   └── csv/{종목명}_forecast.csv
  └── ensemble/
      ├── future_forecasts.parquet
      └── csv/{종목명}_forecast.csv

# 05단계: 유니버스 선정
data/05_universe/{YYYYMMDD}/
  ├── universe_full.parquet             # 전체 평가 완료 종목
  ├── universe_candidates.parquet       # Top-K 후보
  ├── investment_report.csv             # 상세 리포트 (CSV)
  ├── investment_report.xlsx            # Excel 리포트 (선택)
  └── filter_statistics.json            # 필터링 통계

# 메타 데이터
data/99_meta/
  └── krx_calendar.csv                  # 영업일 캘린더
```

---

## 📌 2. 공통 기본 컬럼

모든 단계에서 공통으로 사용되는 필수 컬럼입니다.

| 컬럼 | 타입 | 설명 | 예시 |
|------|------|------|------|
| **date** | datetime64 | 거래일 | 2024-01-15 |
| **ticker** | str | 종목 코드 | 005930 |
| **close** | float64 | 종가 | 70500.0 |

---

## 📌 3. Step 1 (Raw Data) - 입력 스키마

### 3.1 OHLCV 데이터

```python
# 컬럼 정의 (FinanceDataReader 표준)
dtypes = {
    'Date'  : 'datetime64[ns]',
    'Open'  : 'float64',
    'High'  : 'float64',
    'Low'   : 'float64',
    'Close' : 'float64',
    'Volume': 'float64'
}
index.name = 'ticker'  # 종목 코드
```

---

## 📌 4. Step 2 (Processed) - Feature + Target 스키마

### 4.1 Feature 카테고리

#### 가격 관련 (Price)
- `feature_log_return_1d`: log(Close / Close_shift_1)
- `feature_close_shift_{d}`: d일 전 종가

#### 기술적 지표 (Technical)
- `feature_ma_5`, `feature_ma_60`: 5일/60일 이동평균
- `feature_rsi_14`: RSI(14)
- `feature_macd`, `feature_macd_signal`, `feature_macd_hist`: MACD
- `feature_bb_upper`, `feature_bb_middle`, `feature_bb_lower`: Bollinger Bands
- `feature_volatility_20`: 20일 변동성

#### 거래량 지표 (Liquidity)
- `feature_volume_ratio`: 거래량 비율 (volume / volume_ma)
- `liquidity_score`: 20일 평균 거래대금

#### 복합 지표 (Meta)
- `risk_composite`: 변동성 + 거래량 급등 기반 복합 위험 지표

### 4.2 Target 스키마 (✨ v3.5.0 변경)

02단계에서는 학습 타겟의 기준값(`target_log_close`)만 생성합니다. Horizon별 타겟은 03단계 Trainer 내부에서 `target_type`에 따라 동적으로 생성됩니다.

```python
# 02단계 저장 컬럼 (변경 없음)
'target_log_close'   # = log(close), Trainer의 기준값으로 사용

# 03단계 Trainer 내부에서 동적 생성 (v3.5.0: log_return 모드)
# target_log_return_h{n}(t) = target_log_close(t+n) - target_log_close(t)
# = log(close(t+n)) - log(close(t))
# = 누적 로그 수익률 (정상 시계열)
'target_log_return_h1'   # t+1일 누적 로그 수익률
'target_log_return_h2'   # t+2일 누적 로그 수익률
'target_log_return_h3'   # t+3일 누적 로그 수익률
'target_log_return_h4'   # t+4일 누적 로그 수익률
'target_log_return_h5'   # t+5일 누적 로그 수익률

# [레거시] log_close 모드 (target_type="log_close")
# 'target_log_close_h{n}' = log(close(t+n))  — 비정상 시계열, 비권장
```

### 4.3 메타 컬럼 (Meta)

```python
'date'           # 거래일
'ticker'         # 종목코드
'close'          # 종가 (원본)
'volume'         # 거래량 (원본)
'is_suspended'   # 거래 정지 여부 (0/1)
'is_delisted'    # 상장 폐지 여부 (0/1, Placeholder)
```

---

## 📌 5. Step 3 (Training) - 모델 & 예측 스키마 (✨ v3.5.0 변경)

### 5.1 폴더 구조

```
data/03_training/{YYYYMMDD}/{model_name}/
  ├── *.pkl                       # 모델 객체
  ├── registry.json               # 메타데이터
  ├── val_predictions.parquet     # ✨ 검증 폴드 예측 (앙상블 가중치 최적화용)
  └── test_predictions.parquet    # ✨ 테스트 폴드 예측 (최종 성능 평가 전용)
```

### 5.2 2-Fold Walk-Forward 구조 (✨ v3.5.0 신규)

```
전체 데이터 타임라인:
|──────────────────|──────────────|──────────────|
0                  E            E+V            E+V+T

[검증 폴드]
  훈련: [0, E]          → Early stopping 기준
  검증: [E, E+V]        → val_predictions.parquet 저장
                          (앙상블 가중치 최적화 입력)

[테스트 폴드]  (훈련 구간 rolling)
  훈련: [V, E+V]        → 실전 배포 시뮬레이션 (고정 길이 윈도우)
  테스트: [E+V, E+V+T]  → test_predictions.parquet 저장
                          (최종 성능 평가 전용, 가중치 최적화에 미사용)

E = train_end (config.yaml)
V = valid_window_days (거래일)
T = test_window_days (거래일)
```

### 5.3 val_predictions.parquet 스키마 (✨ v3.5.0 신규)

앙상블 가중치 최적화 전용. 테스트셋과 완전히 분리됩니다.

```python
columns = [
    # 메타
    'date', 'ticker', 'fold',           # fold = 'valid'

    # 모델 원시 예측 (log_return 모드)
    'pred_target_log_return_h1',
    'pred_target_log_return_h2',
    'pred_target_log_return_h3',
    'pred_target_log_return_h4',
    'pred_target_log_return_h5',

    # 정답값
    'true_target_log_return_h1',
    'true_target_log_return_h2',
    'true_target_log_return_h3',
    'true_target_log_return_h4',
    'true_target_log_return_h5',

    # 역산값 (target_log_close(t) 조인 후 생성)
    'pred_log_close_target_log_return_h1',  # = target_log_close(t) + pred_h1
    'pred_close_target_log_return_h1',       # = exp(pred_log_close_h1)
    # ... h2~h5 동일
]
```

### 5.4 test_predictions.parquet 스키마 (✨ v3.5.0 신규)

최종 성능 평가 전용. 앙상블 가중치 최적화에 사용 금지.

```python
# val_predictions와 동일한 컬럼 구성
# fold = 'test'
```

### 5.5 Registry 메타데이터 (registry.json)

```json
{
  "model_name": "lightgbm_multi",
  "model_version": "v1_lgbm_20260220_abc123",
  "created_date": "2026-02-20T10:30:00",
  "target_type": "log_return",
  "target_columns": [
    "target_log_return_h1",
    "target_log_return_h2",
    "target_log_return_h3",
    "target_log_return_h4",
    "target_log_return_h5"
  ],
  "training_fold": {
    "val_train_period": ["2022-11-14", "2025-02-14"],
    "val_eval_period":  ["2025-02-17", "2025-05-09"],
    "test_train_period": ["2022-11-14", "2025-05-09"],
    "test_eval_period":  ["2025-05-12", "2025-08-01"]
  },
  "training_metrics": {
    "val_avg_rmse": 0.0187,
    "test_avg_rmse": 0.0201,
    "per_horizon_test_rmse": {
      "target_log_return_h1": 0.0152,
      "target_log_return_h2": 0.0178,
      "target_log_return_h3": 0.0198,
      "target_log_return_h4": 0.0218,
      "target_log_return_h5": 0.0261
    }
  },
  "feature_count": 13
}
```

### 5.6 모델 객체 명세

#### LightGBMModel (.pkl)
```python
class LightGBMModel:
    model_name: str       # "lightgbm_multi"
    model_version: str
    models: Dict[str, lgb.Booster]  # key = target_log_return_h{n}
    feature_list: List[str]
    target_columns: List[str]       # 레거시 호환용 (models.keys() 우선 사용)
    is_fitted: bool = True

# ⚠️ 주의: 04단계에서 target_name 순회 시
# model.target_columns 대신 sorted(model.models.keys()) 사용
```

#### RandomForestMultiModel (.pkl)
```python
class RandomForestMultiModel:
    model_name: str       # "randomforest_multi"
    model_version: str
    model: MultiOutputRegressor
    feature_list: List[str]
    target_columns: List[str]
    is_fitted: bool = True
```

#### EnsembleModel (.pkl)
```python
class EnsembleModel:
    model_name: str       # "ensemble"
    model_version: str
    models: List[ModelBase]   # [LightGBMModel, RandomForestMultiModel]
    weights: List[float]      # 검증셋 기반 최적화된 가중치
    target_columns: List[str]
    feature_list: List[str]
    is_fitted: bool = True
```

---

## 📌 6. Step 3b (Ensemble Training) - 앙상블 가중치 최적화 (✨ v3.5.0 변경)

### 6.1 목표
검증 폴드 예측값으로 앙상블 가중치를 최적화하고, 테스트셋은 순수 평가에만 사용합니다.

### 6.2 실행 흐름

```
Input (✨ 변경):
  ├── .../lightgbm/val_predictions.parquet    ← 가중치 최적화 입력
  └── .../randomforest/val_predictions.parquet

Process:
  [1] val_predictions 로드
  [2] pred_cols 필터링: prefix = "pred_target_log_return_h"
      (역산 컬럼 pred_log_close_*, pred_close_* 제외)
  [3] Scipy.minimize: RMSE 최소화
      목표: minimize(RMSE(w1*pred_lgbm + w2*pred_rf))
      제약: w1 + w2 = 1.0, 0 ≤ w1, w2 ≤ 1.0
  [4] test_predictions으로 최종 성능 확인 (가중치 변경 없음)
  [5] EnsembleModel 생성 (검증셋 기반 최적 가중치 포함)

Output:
  ├── .../ensemble/v1_ens_*.pkl
  ├── .../ensemble/registry.json
  ├── .../ensemble/val_predictions.parquet
  └── .../ensemble/test_predictions.parquet
```

---

## 📌 7. Step 4 (Forecasts) - 미래 예측 결과 스키마 (✨ v3.5.0 변경)

### 7.1 Recursive Extension 역산 로직

```python
# log_return 모드 (v3.5.0 기본값)
log_close_base  = log(latest_close)       # chunk 직전 시점 close
pred_log_return = model.predict(X)        # 모델 원시 출력
pred_log_close  = log_close_base + pred_log_return   # 역산
pred_close      = exp(pred_log_close)

# ⚠️ 주의: target_name 순회 기준
# sorted(model.models.keys()) 사용 (model.target_columns 사용 금지)
```

### 7.2 future_forecasts.parquet 스키마

```python
columns = [
    'date',              # 예측 대상 날짜
    'ticker',            # 종목 코드
    'horizon',           # 예측 시차 (1~5)
    'chunk_idx',         # Recursive Extension chunk 번호
    'pred_log_close',    # 역산된 예측 로그 종가 (하위 호환 유지)
    'pred_close',        # 예측 종가 (원화)
    'pred_log_return',   # ✨ 모델 원시 출력 (log_return 모드에서만 존재)
]
```

---

## 📌 8. Step 5 (Universe) - 최종 선정 결과 스키마

### 8.1 universe_full.parquet
```python
columns = [
    'ticker', 'name',
    'accuracy_score',        # 정확도 지표 (0~1)
    'profitability_score',   # 수익성 지표 (0~1)
    'risk_composite',        # 위험 지표 (0~1)
    'composite_score',       # 종합 점수
    'selected'               # 선정 여부
]
```

### 8.2 universe_candidates.parquet
```python
columns = [
    'ticker', 'name',
    'composite_score',
    'rank',
    'recommendation'         # 'BUY', 'HOLD', 'SELL'
]
```

---

## 📌 9. 전체 파이프라인 데이터 흐름

```
[02 Processed]
  dataset.parquet
  └─ target_log_close (기준값)
  └─ feature_* (피처)
        │
        ▼
[03 Training] ← target_type="log_return" (config.yaml)
  Trainer 내부에서 target_log_return_h{n} 동적 생성
        │
        ├─ [검증 폴드]  → val_predictions.parquet  ──┐
        │                                              │ 앙상블
        └─ [테스트 폴드] → test_predictions.parquet   │ 가중치 최적화
                                   │                  │ (val 전용)
                          최종 성능 평가 전용 ◀────────┘
        │
        ▼
[03b Ensemble] (active_model="ensemble" 시)
  val_predictions 기반 가중치 최적화
  test_predictions으로 최종 성능 확인 (가중치 변경 없음)
        │
        ▼
[04 Forecasts]
  sorted(model.models.keys()) 기반 target_name 순회
  log_close_base + pred_log_return = pred_log_close 역산
  → future_forecasts.parquet (pred_log_close, pred_close, pred_log_return)
        │
        ▼
[05 Universe]
  pred_log_close 기반 수익성/정확도/위험 평가
  → investment_report
```

---

## 📌 10. 설정 파일 스키마 (config.yaml) - v3.5.0

```yaml
training:
  train_end: "2025-02-14"       # 검증 폴드 훈련 종료일
  valid_window_days: 60         # 검증 윈도우 (거래일)
  test_window_days: 60          # 테스트 윈도우 (거래일)
  # ✨ v3.5.0: num_valid 제거 (2-Fold 고정)

  target_col_name: "target_log_close"   # 02단계 기준값 컬럼 (변경 없음)
  target_type: "log_return"             # ✨ v3.5.0 신규: "log_return" | "log_close"

  horizons: [1, 2, 3, 4, 5]
  lgbm_params: { ... }
  randomforest_params: { ... }

active_model: "ensemble"        # "lightgbm" | "randomforest" | "ensemble"
```

---

## 📌 11. 호환성 노트

### v3.4.0 → v3.5.0 마이그레이션

**비호환 항목 (재실행 필요):**
- 03단계 전체 재실행 필수
  - 타겟 컬럼명 변경: `target_log_close_h{n}` → `target_log_return_h{n}`
  - 예측 파일 구조 변경: `predictions.parquet` → `val_predictions.parquet` + `test_predictions.parquet`
- 03b 재실행 필수
  - 가중치 최적화 입력 변경: test → val predictions

**호환 항목 (재실행 불필요):**
- 01단계 (Raw 데이터): 변경 없음
- 02단계 (Feature 데이터): `target_log_close` 컬럼 유지, 변경 없음
- 04단계: `pred_log_close`, `pred_close` 컬럼명 유지 (하위 호환)
- 05단계: 입력 컬럼 변경 없음

**v3.4.0 모델(.pkl) 호환성:**
- ❌ 비호환. v3.4.0 모델의 `models.keys()`는 `target_log_close_h{n}` → 04단계에서 `ValueError` 발생
- v3.5.0으로 03단계부터 전체 재실행 필요

---

## 🔍 스키마 버전 관리 정책

### Semantic Versioning

```
schema_version: "MAJOR.MINOR.PATCH"

MAJOR: 근본 구조 변경 (하위 호환 불가)
MINOR: 기능 추가 / 파이프라인 개선
PATCH: 버그 수정
```

### 버전별 변경 이력

| Version | Date | Type | 주요 변경 사항 |
|---------|------|------|----------------|
| **3.5.0** | 2026-02-20 | 🟢 MINOR | 2-Fold 구조 + log_return 타겟 전환 |
| **3.4.0** | 2026-02-17 | 🟢 MINOR | RF 모델 + 앙상블 학습 |
| **3.3.0** | 2026-02-09 | 🟢 MINOR | 폴더 구조 개선 + 경로 중앙화 |
| **3.2.1** | 2026-02-09 | 🔵 PATCH | Multi-Horizon 버그 + Chunk 오염 방지 |
| **3.2.0** | 2026-02-07 | 🟢 MINOR | 04단계(미래예측) + 05단계(유니버스) |
| 3.1.1 | 2026-01-21 | 🔵 PATCH | Target 생성 위치 재변경 |
| 3.1.0 | 2026-01-21 | 🟢 MINOR | Multi-horizon 예측 |
| 3.0.0 | 2026-01-18 | 🔴 MAJOR | Target 위치 변경 |

---

**Last Updated**: 2026-02-20
**Schema Version**: 3.5.0
**Status**: ✅ Stable
**Maintained by**: SignalWeaver Team
