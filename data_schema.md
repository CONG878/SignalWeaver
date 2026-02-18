# 📄 Data Schema Definition (v3.4.0)

본 스키마는 SignalWeaver 프로젝트의 데이터 계약을 정의합니다.

---

## 📌 Schema Version & Metadata

| 속성 | 값 |
|------|-----|
| **Schema Version** | `3.4.0` |
| **Last Updated** | 2026-02-17 |
| **Latest Changes** | 모델 다양화 - RF 모델 도입 + 앙상블 학습 |
| **Compatibility** | v3.3.x 부분 호환 (모델 폴더 계층 추가) |

---

## 🔄 최근 변경 이력 요약

### v3.4.0 (2026-02-17) - 🟢 MINOR
- **모델 다양화**: RandomForest 모델 + 앙상블 학습 도입
  - LightGBM 외 RandomForest 멀티아웃풋 모델 추가
  - 03b_train_ensemble.ipynb: OOF 기반 가중치 최적화
  - 03_training/{date}/ → 03_training/{date}/{model_name}/ 계층 확장
  - config.yaml: randomforest_params + active_model 옵션 추가

### v3.3.0 (2026-02-09) - 🟢 MINOR
- **H1 (폴더 구조 개선)**: `data/03_results/` 분해 → `03_training/`, `04_forecasts/`, `05_universe/` 독립
- **H2 (경로 중앙화)**: `ProjectPaths` 클래스 도입 → 모든 노트북 경로 관리 통일
- **H3 (모듈 정리)**: `src/universe/select_universe.py` Facade Pattern 적용

### v3.2.1 (2026-02-09) - 🔵 PATCH
- **Critical**: Multi-Horizon Walk-Forward 데이터 누수 버그 수정
- **Critical**: Recursive Extension Chunk 오염 방지

### v3.2.0 (2026-02-07) - 🟢 MINOR
- **04단계 추가**: Recursive Extension을 이용한 미래 주가 예측
- **05단계 추가**: 3대 평가 지표(정확도/수익성/리스크) 기반 유니버스 선정

---

## 📌 파일 저장 규칙 / 포맷 (Updated v3.4.0)

### 1.1 기본 포맷

| 단계 | 폴더 | 포맷 | 이유 |
|------|------|------|------|
| **01단계 (Raw)** | `data/01_raw/{date}/` | CSV + 통합 Parquet | API 원본 보존 + 파이프라인 효율성 |
| **02단계 (Processed)** | `data/02_processed/{date}/` | Parquet + 선택적 CSV | 고속 I/O + 디버깅 지원 |
| **03단계 (Training)** | `data/03_training/{date}/{model_name}/` | 🔄 Updated: Parquet + 모델별 폴더 | 모델 분리 + 메타데이터 관리 |
| **04단계 (Forecasts)** | `data/04_forecasts/{date}/` | Parquet + 선택적 CSV | 미래 예측값 저장 |
| **05단계 (Universe)** | `data/05_universe/{date}/` | Parquet + CSV + JSON | 투자 후보 선정 결과 |

### 1.2 파일 네이밍 규칙 (v3.4.0)

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

# 03단계: 학습 검증 예측 (H4 v3.4.0: 모델별 폴더 분리)
data/03_training/{YYYYMMDD}/
  ├── lightgbm/                         # ✨ LightGBM 전용 폴더
  │   ├── v1_lgbm_20260213_abc123.pkl  # LightGBM Booster 객체
  │   ├── registry.json                # 메타데이터
  │   └── predictions.parquet          # OOF 검증 예측값
  │
  ├── randomforest/                    # ✨ RandomForest 전용 폴더
  │   ├── v1_rf_20260213_def456.pkl    # MultiOutputRegressor 객체
  │   ├── registry.json                # 메타데이터
  │   └── predictions.parquet          # OOF 검증 예측값
  │
  └── ensemble/                        # ✨ 앙상블 모델 폴더
      ├── v1_ens_20260213_ghi789.pkl   # EnsembleModel 객체 (가중치 포함)
      ├── registry.json                # 메타데이터
      └── predictions.parquet          # 블렌딩된 예측값

# 04단계: 미래 예측
data/04_forecasts/{YYYYMMDD}/
  ├── lightgbm/
  │   ├── future_forecasts.parquet         # LightGBM 미래 예측값
  │   └── csv/{종목명}_forecast.csv         # 종목별 CSV (옵션)
  │
  ├── randomforest/
  │   ├── future_forecasts.parquet         # RandomForest 미래 예측값
  │   └── csv/{종목명}_forecast.csv         # 종목별 CSV (옵션)
  │
  └── ensemble/
      ├── future_forecasts.parquet         # 앙상블 미래 예측값
      └── csv/{종목명}_forecast.csv         # 종목별 CSV (옵션)

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
columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

# 타입
dtypes = {
    'Date': 'datetime64[ns]',    # 거래일
    'Open': 'float64',            # 시가
    'High': 'float64',            # 고가
    'Low': 'float64',             # 저가
    'Close': 'float64',           # 종가
    'Volume': 'float64'           # 거래량
}

# 인덱스
index.name = 'ticker'  # 종목 코드
```

---

## 📌 4. Step 2 (Processed) - Feature + Target 스키마

### 4.1 Feature 카테고리

#### 가격 관련 (Price)
- `feature_log_close`: log(Close)
- `feature_log_return_1d`: log(Close / Close_shift_1)
- `feature_close_shift_{d}`: d일 전 종가

#### 기술적 지표 (Technical)
- `feature_ma_5`, `feature_ma_60`: 5일/60일 이동평균
- `feature_rsi_14`: RSI(14) 지표
- `feature_volatility_5`: 5일 변동성

#### 거래량 지표 (Liquidity)
- `feature_volume_ma_5`: 5일 평균 거래량
- `liquidity_score`: 거래량 기반 유동성 점수

#### 복합 지표 (Meta)
- `risk_composite`: 종합 위험 지표
- `trend_score`: 추세 강도

### 4.2 Target 스키마

```python
# Target: 로그 종가의 t일 뒤 값
target_col = f"target_log_close_h{horizon}"

# 예시 (horizon=[1, 2, 3, 4, 5])
target_log_close_h1  # t+1일의 log(Close)
target_log_close_h2  # t+2일의 log(Close)
target_log_close_h3  # t+3일의 log(Close)
target_log_close_h4  # t+4일의 log(Close)
target_log_close_h5  # t+5일의 log(Close)

# Alignment: 03_training에서 사용
# 학습 샘플: X_t → y_t (과거 데이터로 미래 예측)
```

### 4.3 메타 컬럼 (Meta)

```python
# 추적용 컬럼
'date'              # 거래일
'ticker'            # 종목코드
'price_current'     # 종가 (원본)
'volume'            # 거래량 (원본)
```

---

## 📌 5. Step 3 (Training) - 모델 & OOF 예측 스키마 (v3.4.0)

### 5.1 폴더 구조

```
data/03_training/{YYYYMMDD}/
├── lightgbm/
│   ├── v1_lgbm_20260213_abc123.pkl   # LightGBMModel 객체
│   ├── registry.json                 # 메타데이터
│   └── predictions.parquet           # OOF 검증 결과
│
├── randomforest/                     # ✨ NEW
│   ├── v1_rf_20260213_def456.pkl     # RandomForestMultiModel 객체
│   ├── registry.json
│   └── predictions.parquet
│
└── ensemble/                         # ✨ NEW
    ├── v1_ens_20260213_ghi789.pkl    # EnsembleModel 객체
    ├── registry.json
    └── predictions.parquet           # 블렌딩 결과
```

### 5.2 OOF 예측 결과 (predictions.parquet)

```python
# 컬럼 구성
columns = [
    'date', 'ticker',                      # 메타 (인덱싱용)
    
    # 예측값 (5 horizons)
    'pred_target_log_close_h1',
    'pred_target_log_close_h2',
    'pred_target_log_close_h3',
    'pred_target_log_close_h4',
    'pred_target_log_close_h5',
    
    # 정답값 (검증용)
    'true_target_log_close_h1',
    'true_target_log_close_h2',
    'true_target_log_close_h3',
    'true_target_log_close_h4',
    'true_target_log_close_h5',
    
    # 추가 메타
    'train_date',
    'valid_fold',
]

# 크기: (n_samples, 13) → walk-forward 검증 결과
# NaN 처리: dropna() 후 저장
```

### 5.3 Registry 메타데이터 (registry.json)

```json
{
  "model_name": "lightgbm_multi",
  "model_version": "v1_lgbm_20260213_abc123",
  "created_date": "2026-02-13T10:30:00",
  "hyperparameters": {
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 100
  },
  "training_metrics": {
    "rmse_h1": 0.0234,
    "rmse_h2": 0.0312,
    "mae_all": 0.0189
  },
  "feature_count": 48,
  "target_columns": ["pred_target_log_close_h1", ..., "pred_target_log_close_h5"]
}
```

### 5.4 모델 객체 명세

#### LightGBMModel (.pkl)
```python
# Pickle된 객체 구조
class LightGBMModel:
    model_name: str = "lightgbm_multi"
    model_version: str
    model: lgb.Booster           # LightGBM Booster
    feature_list: List[str]      # 학습에 사용된 Feature 이름
    target_columns: List[str]    # ['pred_target_log_close_h1', ..., 'h5']
    is_fitted: bool = True
```

#### RandomForestMultiModel (.pkl) - ✨ NEW
```python
# Pickle된 객체 구조
class RandomForestMultiModel:
    model_name: str = "randomforest_multi"
    model_version: str
    model: MultiOutputRegressor  # sklearn.ensemble 래퍼
    feature_list: List[str]      # 학습에 사용된 Feature 이름
    target_columns: List[str]    # ['pred_target_log_close_h1', ..., 'h5']
    is_fitted: bool = True
```

#### EnsembleModel (.pkl) - ✨ NEW
```python
# Pickle된 객체 구조
class EnsembleModel:
    model_name: str = "ensemble"
    model_version: str
    models: List[ModelBase]      # [LightGBMModel, RandomForestMultiModel, ...]
    weights: List[float]         # [0.3, 0.7, ...]
    target_columns: List[str]    # 상속받은 메타데이터
    feature_list: List[str]      # 상속받은 메타데이터
    is_fitted: bool = True
```

---

## 📌 6. Step 3b (Ensemble Training) - 새로운 노트북 (v3.4.0)

### 6.1 목표
- 개별 모델(LGBM, RF)의 OOF 예측값 결합
- Scipy.optimize로 최적 가중치 탐색
- 블렌딩된 앙상블 모델 저장

### 6.2 실행 흐름

```
Input: 
  ├── data/03_training/{date}/lightgbm/predictions.parquet
  └── data/03_training/{date}/randomforest/predictions.parquet

Process:
  [1] OOF 로드 → numpy 배열 변환
  [2] Scipy.minimize: RMSE 최소화
      - 목표: minimize(RMSE(w1*pred_lgbm + w2*pred_rf))
      - 제약: w1 + w2 = 1.0, 0 ≤ w1, w2 ≤ 1.0
  [3] EnsembleModel 생성 (가중치 포함)
  [4] 블렌딩 예측값 생성

Output:
  ├── data/03_training/{date}/ensemble/v1_ens_*.pkl
  ├── data/03_training/{date}/ensemble/registry.json
  └── data/03_training/{date}/ensemble/predictions.parquet
```

---

## 📌 7. Step 4 (Forecasts) - 미래 예측 결과 스키마

```
data/04_forecasts/{YYYYMMDD}/
├── lightgbm/
│   ├── future_forecasts.parquet
│   │   columns: [
│   │       'date', 'ticker',                  # 메타
│   │       'forecast_date',                   # 예측 시점
│   │       'target_horizon_days',             # 예측 일수 (1~60)
│   │       'pred_target_log_close',           # 예측값
│   │       'confidence_interval_lower',       # CI 하한
│   │       'confidence_interval_upper'        # CI 상한
│   │   ]
│   │
│   └── csv/{종목명}_forecast.csv              # 종목별 CSV (선택)
│
├── randomforest/                     # ✨ NEW
│   ├── future_forecasts.parquet
│   │   columns: [
│   │       ...
│   │   ]
│   │
│   └── csv/{종목명}_forecast.csv
│
└── ensemble/                         # ✨ NEW
    ├── future_forecasts.parquet
    │   columns: [
    │       ...
    │   ]
    │
    └── csv/{종목명}_forecast.csv
```

---

## 📌 8. Step 5 (Universe) - 최종 선정 결과 스키마

### 8.1 universe_full.parquet
```python
# 모든 종목의 종합 평가 결과
columns = [
    'ticker', 'name',              # 메타
    'accuracy_score',              # 정확도 지표 (0~1)
    'profitability_score',         # 수익성 지표 (0~1)
    'risk_composite',              # 위험 지표 (0~1)
    'composite_score',             # 종합 점수
    'selected'                     # 선정 여부
]
```

### 8.2 universe_candidates.parquet
```python
# Top-K 후보
columns = [
    'ticker', 'name',
    'composite_score',
    'rank',
    'recommendation'               # 'BUY', 'HOLD', 'SELL'
]
```

---

## 📌 9. Step 2~5 데이터 흐름

### 9.1 전체 파이프라인

```
[02 Processed]
├─ Feature dataset
│
└─→ [03 Training]
    ├─ 03_train_predict.ipynb
    │  └─ active_model = "lightgbm" | "randomforest"
    │     └─ Individual model OOF
    │
    └─ 03b_train_ensemble.ipynb (Optional)
       └─ Blend OOF → Ensemble model
       
└─→ [04 Forecasts]
    └─ 04_forecast_future.ipynb
       └─ Load model (active_model) → Recursive Extension
       
└─→ [05 Universe]
    └─ 05_universe_selection.ipynb
       └─ Evaluate + Filter + Rank
```

---

## 📌 10. 모델 선택 메커니즘 (v3.4.0)

### 10.1 config.yaml 설정

```yaml
# 모델 선택
active_model: "ensemble"  # 'lightgbm', 'randomforest', 'ensemble'

training:
  # LightGBM 파라미터
  lgbm_params:
    num_leaves: 31
    learning_rate: 0.05
    # ...
  
  # RandomForest 파라미터 (NEW)
  randomforest_params:
    n_estimators: 40
    max_depth: 8
    min_samples_split: 10
    # ...
```

### 10.2 노트북별 동작

| 노트북 | 03_train_predict | 03b_train_ensemble | 04_forecast_future |
|--------|-----------------|------------------|-----------------|
| **lightgbm** | LGBM 학습 | ⏭️ Skip | LGBM 로드 후 추론 |
| **randomforest** | RF 학습 | ⏭️ Skip | RF 로드 후 추론 |
| **ensemble** | LGBM + RF 학습 | 가중치 최적화 | Ensemble 로드 후 추론 |

---

## 📌 11. 호환성 노트

### v3.3.0 → v3.4.0 마이그레이션

**기존 모델 호환성**:
- v3.3.0에서 저장한 LightGBM 모델은 v3.4.0에서 자동 호환
- 폴더 구조: 기존 flat 구조도 인식 가능 (backward compatibility)

**신규 기능**:
- RandomForest 모델 추가
- 앙상블 학습 (03b_train_ensemble.ipynb) 선택사항

---

## 🔍 스키마 버전 관리 정책

### Semantic Versioning

```
schema_version: "MAJOR.MINOR.PATCH"

예: "3.4.0"
    │  │  └─ PATCH: 버그 수정
    │  └──── MINOR: 구조 개선 (3.4.0)
    └─────── MAJOR: 근본 구조 변경
```

### 버전별 변경 이력

| Version | Date | Type | 주요 변경 사항 |
|---------|------|------|----------------|
| **3.4.0** | 2026-02-17 | 🟢 MINOR | RF 모델 + 앙상블 학습 |
| **3.3.0** | 2026-02-09 | 🟢 MINOR | 폴더 구조 개선 + 경로 중앙화 |
| **3.2.1** | 2026-02-09 | 🔵 PATCH | Multi-Horizon 버그 + Chunk 오염 방지 |
| **3.2.0** | 2026-02-07 | 🟢 MINOR | 04단계(미래예측) + 05단계(유니버스) |
| 3.1.1 | 2026-01-21 | 🔵 PATCH | Target 생성 위치 재변경 |
| 3.1.0 | 2026-01-21 | 🟢 MINOR | Multi-horizon 예측 |
| 3.0.0 | 2026-01-18 | 🔴 MAJOR | Target 위치 변경 |

---

**Last Updated**: 2026-02-17  
**Schema Version**: 3.4.0  
**Status**: ✅ Stable  
**Maintained by**: SignalWeaver Team
