# 📄 Data Schema Definition (v3.7.1)

본 스키마는 SignalWeaver 프로젝트의 데이터 계약을 정의합니다.

---

## 📌 Schema Version & Metadata

| 속성 | 값 |
|------|-----|
| **Schema Version** | `3.7.1` |
| **Last Updated** | 2026-02-24 |
| **Latest Changes** | 04단계 버그 수정 — v3.6.0 피처 스키마 동기화 + 매크로 미래값 반영 |
| **Compatibility** | v3.7.0 완전 호환 (재실행 불필요) |

---

## 🔄 최근 변경 이력 요약

### v3.7.1 (2026-02-24) - 🔵 PATCH
- **04단계 피처 스키마 동기화**: `calculate_features_for_ticker` 함수가 v3.6.0 이전 스키마(`feature_ma_*`, `feature_bb_upper/lower`, `liquidity_score`)를 사용하던 누락 업데이트 수정. builder.py v3.6.0과 일치하도록 재작성.
- **97단계 신설**: `97_forecast_macro.ipynb` — 매크로 지표 미래값 추정 파이프라인. Damped Holt(kospi, usd_krw, vix) 및 SES(us_return_1d) 적용.
- **매크로 미래값 반영**: 04단계 Recursive Extension 루프에서 미래 날짜의 매크로 피처가 NaN으로 채워지던 버그 수정. `macro_regime_forecast.parquet` 로드 및 조인 로직 추가.
- **정적/캘린더 피처 반영**: `new_row` 구성 시 `feature_is_kospi`, `feature_is_monday`, `feature_is_friday` 누락 수정.
- **모델 로드 방식 수정**: `pickle.load()` 직접 호출 → `Model.load()` 클래스메서드 사용으로 수정.

### v3.7.0 (2026-02-22) - 🟢 MINOR
- **log_return deprecated → log_close 롤백**: 피처 확장 이후 성능 열위 확인. `target_type="log_return"` 사용 시 `DeprecationWarning` 발생. 타겟 컬럼명 `target_log_return_h{n}` → `target_log_close_h{n}` 복원.
- **Embargo Gap 도입**: 훈련/검증 경계 look-ahead bias 방지. `G = max(horizons)` 자동 계산. 검증 윈도우 크기는 유지하고 훈련 샘플 끝을 G일 앞당김.

### v3.6.0 (2026-02-21) - 🟢 MINOR
- **Scale-invariant 피처 리팩토링**: MA → Disparity, BB → %B/Bandwidth, liquidity_score → log_liquidity
- **매크로/레짐 피처 통합**: KOSPI 수익률, 환율, VIX, 레짐 → `data/99_meta/macro_regime.parquet`에서 조인
- **98단계 신설**: `98_save_macro_data.ipynb` — 전역 메타 데이터 수집 파이프라인
- **IC/ICIR 평가 지표 도입**: Trainer에 Cross-Sectional IC 추가, 앙상블 가중치 최적화 목표 → `-IC`
- **`feature_` 접두어 규칙 확립**: 모든 학습 피처에 의무 적용

### v3.5.0 (2026-02-20) - 🟢 MINOR
- **2-Fold Walk-Forward 구조**: `val_predictions.parquet` + `test_predictions.parquet` 분리, 앙상블 가중치 누수 제거
- **log_return 타겟 도입** (→ v3.7.0에서 deprecated)

### v3.4.0 (2026-02-17) - 🟢 MINOR
- **RandomForest + 앙상블**: `03_training/{date}/{model_name}/` 계층 확장, `03b_train_ensemble.ipynb` 신설

---

## 📌 1. 파일 저장 규칙 / 포맷

### 1.1 기본 포맷

| 단계 | 폴더 | 포맷 |
|------|------|------|
| **01단계 (Raw)** | `data/01_raw/{date}/` | CSV + 통합 Parquet |
| **02단계 (Processed)** | `data/02_processed/{date}/` | Parquet + 선택적 CSV |
| **03단계 (Training)** | `data/03_training/{date}/{model_name}/` | Parquet + 모델별 폴더 |
| **04단계 (Forecasts)** | `data/04_forecasts/{date}/` | Parquet + 선택적 CSV |
| **05단계 (Universe)** | `data/05_universe/{date}/` | Parquet + CSV + JSON |
| **99_meta** | `data/99_meta/` | Parquet + CSV |

### 1.2 파일 네이밍 규칙 (v3.7.0)

```
# 01단계: 원시 데이터
data/01_raw/{YYYYMMDD}/
  ├── krx_prices_{YYYYMMDD}.parquet
  ├── ticker_master_{YYYYMMDD}.csv
  └── csv/{종목명}.csv

# 02단계: 전처리 데이터
data/02_processed/{YYYYMMDD}/
  ├── dataset.parquet
  └── csv/{종목명}.csv

# 03단계: 학습/검증/테스트 예측
data/03_training/{YYYYMMDD}/
  ├── lightgbm/
  │   ├── v1_lgbm_{YYYYMMDD}_{hash}.pkl
  │   ├── registry.json
  │   ├── val_predictions.parquet      # 검증 폴드 예측 (앙상블 가중치용)
  │   └── test_predictions.parquet     # 테스트 폴드 예측 (최종 평가용)
  ├── randomforest/  { 동일 구조 }
  └── ensemble/      { 동일 구조 }

# 04단계: 미래 예측
data/04_forecasts/{YYYYMMDD}/
  ├── lightgbm/future_forecasts.parquet
  ├── randomforest/future_forecasts.parquet
  └── ensemble/future_forecasts.parquet

# 05단계: 유니버스 선정
data/05_universe/{YYYYMMDD}/
  ├── universe_full.parquet
  ├── universe_candidates.parquet
  ├── investment_report.csv
  ├── investment_report.xlsx
  └── filter_statistics.json

# 전역 메타 데이터 (✨ v3.6.0 신설)
data/99_meta/
  ├── krx_calendar.csv                  # 영업일 캘린더
  ├── macro_regime.parquet              # 매크로/레짐 데이터 (98단계 출력, 과거 실측)
  └── macro_regime_forecast.parquet     # 매크로/레짐 미래 추정값 (97단계 출력) ✨ v3.7.1
```

---

## 📌 2. 공통 기본 컬럼

| 컬럼 | 타입 | 설명 | 예시 |
|------|------|------|------|
| **date** | datetime64 | 거래일 | 2024-01-15 |
| **ticker** | str | 종목 코드 | 005930 |
| **close** | float64 | 종가 | 70500.0 |

---

## 📌 3. Step 1 (Raw Data) - 입력 스키마

### 3.1 OHLCV 데이터

```python
dtypes = {
    'Date'  : 'datetime64[ns]',
    'Open'  : 'float64',
    'High'  : 'float64',
    'Low'   : 'float64',
    'Close' : 'float64',
    'Volume': 'float64'
}
index.name = 'ticker'
```

---

## 📌 4. Step 2 (Processed) - Feature + Target 스키마

### 4.1 Feature 카테고리 (✨ v3.6.0 변경)

모든 학습 피처는 `feature_` 접두어를 가집니다.
`feature_cols = [c for c in df.columns if c.startswith('feature_')]`로 자동 인식.

#### 기술적 지표 (Technical) — ✨ v3.6.0 변경
- `feature_disparity_5`, `feature_disparity_60`: 이격도 `close/ma_n - 1` *(구: `feature_ma_5/60` 절대가격)*
- `feature_rsi_14`: RSI(14)
- `feature_macd`, `feature_macd_signal`, `feature_macd_hist`: MACD
- `feature_bb_pct_b`: Bollinger %B `(close - lower) / (upper - lower)` *(구: `feature_bb_upper/lower` 절대가격)*
- `feature_bb_bandwidth`: Bollinger Bandwidth `(upper - lower) / middle` *(신규)*
- `feature_volatility_20`: 20일 변동성

#### 거래량 지표 (Liquidity) — ✨ v3.6.0 변경
- `feature_volume_ratio`: `volume / volume_ma`
- `feature_log_liquidity`: `log(20일 평균 거래대금)` *(구: `liquidity_score` 원화 절대값)*

#### 매크로/레짐 피처 (Macro) — ✨ v3.6.0 신규
`data/99_meta/macro_regime.parquet`에서 날짜 기준 left join 후 `feature_` 접두어 부여.

- `feature_kospi`: KOSPI 지수
- `feature_usd_krw`: USD/KRW 환율
- `feature_vix`: VIX 지수
- `feature_us_return_1d`: 직전 거래일 미국 시장 수익률
- `feature_market_regime`: 시장 레짐 (-1=Bear, 0=Neutral, 1=Bull)

#### 기업/캘린더 피처 — ✨ v3.6.0 신규
- `feature_is_kospi`: KOSPI 상장 여부 (KOSPI=1, KOSDAQ=0)
- `feature_is_monday`: 월요일 여부 (0/1)
- `feature_is_friday`: 금요일 여부 (0/1)

#### 제거된 피처 (v3.6.0~)
- ~~`sector`~~: 고차원 범주형 변수 → 피처 인플레이션 방지 목적으로 제외
- ~~`feature_ma_5`, `feature_ma_60`~~: `feature_disparity_*`로 대체
- ~~`feature_bb_upper`, `feature_bb_lower`~~: `feature_bb_pct_b/bandwidth`로 대체
- ~~`liquidity_score`~~: `feature_log_liquidity`로 대체

### 4.2 Target 스키마 (✨ v3.7.0 변경)

02단계에서는 `target_log_close` 기준값만 저장합니다.
Horizon별 타겟은 03단계 Trainer 내부에서 동적으로 생성됩니다.

```python
# 02단계 저장 컬럼 (v3.7.0 현재, 변경 없음)
'target_log_close'          # = log(close). Trainer 기준값으로 사용.

# 03단계 Trainer 내부에서 동적 생성 (v3.7.0 기본: log_close 모드)
'target_log_close_h1'       # = log(close(t+1))
'target_log_close_h2'       # = log(close(t+2))
'target_log_close_h3'       # = log(close(t+3))
'target_log_close_h4'       # = log(close(t+4))
'target_log_close_h5'       # = log(close(t+5))

# [DEPRECATED v3.7.0] log_return 모드 — 사용 금지
# target_type="log_return" 설정 시 DeprecationWarning 발생
# 'target_log_return_h{n}' = log(close(t+n)) - log(close(t))
```

### 4.3 메타 컬럼

```python
'date'           # 거래일
'ticker'         # 종목코드
'close'          # 종가 (원본)
'volume'         # 거래량 (원본)
'is_suspended'   # 거래 정지 여부 (0/1)
'is_delisted'    # 상장 폐지 여부 (0/1, Placeholder)
```

---

## 📌 5. Step 3 (Training) - 모델 & 예측 스키마

### 5.1 폴더 구조

```
data/03_training/{YYYYMMDD}/{model_name}/
  ├── *.pkl                       # 모델 객체
  ├── registry.json               # 메타데이터
  ├── val_predictions.parquet     # 검증 폴드 예측 (앙상블 가중치 최적화용)
  └── test_predictions.parquet    # 테스트 폴드 예측 (최종 성능 평가 전용)
```

### 5.2 2-Fold Walk-Forward 구조 (✨ v3.7.0: Embargo Gap 추가)

```
전체 데이터 타임라인 (G = max(horizons) = 5):

|────────────|░░░░░|──────────────|──────────────|
0           E-G    E            E+V            E+V+T

[검증 폴드]
  실제 훈련: [0, E-G]      ← embargo gap(G일) 제거됨
  embargo:  [E-G, E]       ← look-ahead 오염 구간 (미사용)
  검증:     [E, E+V]       → val_predictions.parquet
                             (앙상블 가중치 최적화 입력)

[테스트 폴드]  (훈련 구간 rolling)
  실제 훈련: [V, E+V-G]    ← embargo gap(G일) 제거됨
  embargo:  [E+V-G, E+V]
  테스트:   [E+V, E+V+T]   → test_predictions.parquet
                              (최종 성능 평가 전용)

E = train_end (config.yaml)
V = valid_window_days (거래일)
T = test_window_days (거래일)
G = max(horizons) — 자동 계산, 별도 설정 불필요
```

### 5.3 val_predictions.parquet 스키마 (✨ v3.7.0: 컬럼명 복원)

앙상블 가중치 최적화 전용. 테스트셋과 완전히 분리됩니다.

```python
columns = [
    # 메타
    'date', 'ticker', 'fold',           # fold = 'valid'

    # 모델 원시 예측 (log_close 모드, v3.7.0 기본)
    'pred_target_log_close_h1',
    'pred_target_log_close_h2',
    'pred_target_log_close_h3',
    'pred_target_log_close_h4',
    'pred_target_log_close_h5',

    # 정답값
    'true_target_log_close_h1',
    'true_target_log_close_h2',
    'true_target_log_close_h3',
    'true_target_log_close_h4',
    'true_target_log_close_h5',
]
```

### 5.4 test_predictions.parquet 스키마

val_predictions와 동일한 컬럼 구성. `fold = 'test'`.

### 5.5 평가 지표 (✨ v3.6.0 추가)

```python
metrics = {
    'avg_rmse'    : float,   # Horizon 평균 RMSE
    'avg_ic'      : float,   # Horizon 평균 IC (Spearman)
    'per_horizon' : {
        'target_log_close_h1': {
            'rmse'   : float,  # RMSE
            'ic_mean': float,  # Daily Cross-Sectional IC 평균
            'icir'   : float,  # IC / IC_std (안정성 지표)
        },
        # ... h2~h5 동일
    },
    'samples'     : int,     # 유효 샘플 수
}
```

---

## 📌 6. Step 3b (Ensemble) - 앙상블 가중치 최적화

```python
# 가중치 최적화 입력: val_predictions (검증 폴드 전용)
# 최적화 목표 (v3.6.0~): -IC 최소화 = IC 최대화
weights = minimize(lambda w: -ic(ensemble_pred(w), true), ...)

# 테스트 폴드 평가: test_predictions으로 최종 성능 확인 (가중치 변경 없음)
```

---

## 📌 7. Step 4 (Forecasts) - 미래 예측 결과 스키마

### 7.1 Recursive Extension 역산 로직 (v3.7.0: log_close 기본값)

```python
# log_close 모드 (v3.7.0 기본값)
pred_log_close = model.predict(X)    # 모델 원시 출력 = log(close)
pred_close     = exp(pred_log_close)

# [DEPRECATED] log_return 모드 역산 (v3.5.0 ~ v3.6.0, 현재 비권장)
# log_close_base  = log(latest_close)
# pred_log_return = model.predict(X)
# pred_log_close  = log_close_base + pred_log_return
```

### 7.2 future_forecasts.parquet 스키마

```python
columns = [
    'date',           # 예측 대상 날짜
    'ticker',         # 종목 코드
    'horizon',        # 예측 시차 (1~5)
    'chunk_idx',      # Recursive Extension chunk 번호
    'pred_log_close', # 예측 로그 종가
    'pred_close',     # 예측 종가 (원화)
    # 'pred_log_return' — log_return 모드에서만 존재 (v3.5.0~v3.6.0, 현재 deprecated)
]
```

---

## 📌 8. Step 5 (Universe) - 최종 선정 결과 스키마

### 8.1 universe_full.parquet

```python
columns = [
    'ticker', 'name',
    'accuracy_score',        # IC 기반 정확도 지표 (✨ v3.6.0: directional_accuracy → IC)
    'profitability_score',   # 수익성 지표
    'risk_composite',        # 위험 지표
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

## 📌 9. Step 99_meta - 전역 메타 데이터 스키마 (✨ v3.6.0 신설)

### 9.1 macro_regime.parquet (98단계 출력 — 과거 실측값)

```python
columns = [
    'date',              # 거래일
    'kospi',             # KOSPI 지수
    'usd_krw',           # USD/KRW 환율
    'vix',               # VIX 지수
    'us_return_1d',      # 직전 거래일 미국 시장 수익률
    'market_regime',     # 시장 레짐 (-1=Bear, 0=Neutral, 1=Bull)
]
# 02단계에서 조인 시 컬럼명에 feature_ 접두어 자동 부여
```

### 9.2 macro_regime_forecast.parquet (97단계 출력 — 미래 추정값) ✨ v3.7.1

```python
# macro_regime.parquet와 동일한 스키마. 미래 영업일 행만 포함.
# 04단계에서 macro_regime.parquet와 concat 후 left join.
columns = [
    'date',              # 미래 거래일 (krx_calendar.csv 기준)
    'kospi',             # Damped Holt 추정값 (φ=0.90)
    'usd_krw',           # Damped Holt 추정값 (φ=0.85)
    'vix',               # Damped Holt 추정값 (φ=0.85)
    'us_return_1d',      # SES 추정값 (zero 수렴)
    'market_regime',     # kospi 추정값으로 재계산 (98단계 동일 로직)
]
```

### 9.3 krx_calendar.csv

```python
columns = ['date']   # 영업일 날짜 목록 (datetime)
```

---

## 📌 10. 전체 파이프라인 데이터 흐름 (v3.7.0)

```
[98 Meta] (선행 실행)
  98_save_macro_data.ipynb
  → data/99_meta/macro_regime.parquet  (과거 실측값)

[97 Macro Forecast] (✨ v3.7.1 신설, 04단계 전 선행 실행)
  97_forecast_macro.ipynb
  → data/99_meta/macro_regime_forecast.parquet  (미래 추정값)
        │
        ▼ (두 파일 concat 후 date 기준 left join)
[02 Processed]
  dataset.parquet
  └─ target_log_close (기준값)
  └─ feature_disparity_*, feature_bb_pct_b, feature_log_liquidity (scale-invariant)
  └─ feature_kospi, feature_usd_krw, feature_vix, feature_us_return_1d, feature_market_regime (매크로)
  └─ feature_is_kospi, feature_is_monday/friday (기업/캘린더)
        │
        ▼
[03 Training] ← target_type="log_close" (v3.7.0 기본값)
  Trainer 내부: G = max(horizons) embargo gap 적용
  실제 훈련 종료 = train_end - G
  타겟 동적 생성: target_log_close_h{n}
        │
        ├─ [검증 폴드] → val_predictions.parquet  ──┐
        │                                             │ -IC 최소화
        └─ [테스트 폴드] → test_predictions.parquet  │ (앙상블 가중치)
                                  │                  │
                         최종 성능 평가 전용 ◀────────┘
        │
        ▼
[03b Ensemble] (active_model="ensemble" 시)
  val_predictions 기반 -IC 최소화 가중치 최적화
  test_predictions 최종 성능 확인
        │
        ▼
[04 Forecasts] (✨ v3.7.1: 매크로/정적/캘린더 피처 완전 반영)
  macro_regime.parquet + macro_regime_forecast.parquet → concat → date 기준 조회
  log_close 모드: pred_log_close = model.predict(X)
  → future_forecasts.parquet (pred_log_close, pred_close)
        │
        ▼
[05 Universe]
  IC 기반 accuracy_score 산출
  → investment_report
```

---

## 📌 11. 설정 파일 스키마 (config.yaml) - v3.7.0

```yaml
training:
  train_end: "2025-08-13"       # 검증 폴드 훈련 종료일

  valid_window_days: 60         # 검증 윈도우 (거래일)
  test_window_days: 60          # 테스트 윈도우 (거래일)

  # Embargo gap은 max(horizons)로 자동 계산 — 별도 설정 없음
  # horizons 변경 시 자동 연동됨

  target_col_name: "target_log_close"   # 02단계 기준값 컬럼
  target_type: "log_close"              # ✨ v3.7.0: "log_close" 권장
                                        # "log_return" → DeprecationWarning 발생

  horizons: [1, 2, 3, 4, 5]
  lgbm_params:      { ... }
  randomforest_params: { ... }

active_model: "lightgbm"        # "lightgbm" | "randomforest" | "ensemble"
```

---

## 📌 12. 호환성 노트

### v3.6.0 → v3.7.0 마이그레이션

**비호환 (재실행 필요):**
- 03단계 전체 재실행 필수
  - 타겟 컬럼명: `target_log_return_h{n}` → `target_log_close_h{n}`
  - 예측 파일 컬럼: `pred_target_log_return_h{n}` → `pred_target_log_close_h{n}`
- 03b 재실행 필수: `pred_cols` 필터 prefix 복원

**호환 (재실행 불필요):**
- 01, 02단계: 변경 없음
- 04단계: `pred_log_close`, `pred_close` 컬럼명 유지
- 05단계: 입력 스키마 변경 없음

### v3.5.0 → v3.6.0 마이그레이션

**비호환 (재실행 필요):**
- 02단계: 피처 컬럼명 전면 변경 (disparity, %B, log_liquidity, 매크로 피처 추가)
- 03단계: 피처 변경에 따른 모델 재학습
- 98단계 선행 실행 필요 (`macro_regime.parquet` 생성)

---

## 🔍 스키마 버전 관리 정책

### Semantic Versioning

```
MAJOR: 근본 구조 변경 (하위 호환 불가)
MINOR: 기능 추가 / 파이프라인 개선
PATCH: 버그 수정
```

### 버전별 변경 이력

| Version | Date | Type | 주요 변경 사항 |
|---------|------|------|----------------|
| **3.7.1** | 2026-02-24 | 🔵 PATCH | 04단계 피처 스키마 동기화 + 97단계 신설 |
| **3.7.0** | 2026-02-22 | 🟢 MINOR | log_close 롤백 + Embargo Gap |
| **3.6.0** | 2026-02-21 | 🟢 MINOR | Scale-invariant 피처 + IC 평가 + 매크로 통합 |
| **3.5.0** | 2026-02-20 | 🟢 MINOR | 2-Fold 구조 + log_return 타겟 (현재 deprecated) |
| **3.4.0** | 2026-02-17 | 🟢 MINOR | RF 모델 + 앙상블 학습 |
| **3.3.0** | 2026-02-09 | 🟢 MINOR | 폴더 구조 개선 + 경로 중앙화 |
| **3.2.1** | 2026-02-09 | 🔵 PATCH | Multi-Horizon 버그 + Chunk 오염 방지 |
| **3.2.0** | 2026-02-07 | 🟢 MINOR | 04단계(미래예측) + 05단계(유니버스) |
| 3.1.1 | 2026-01-21 | 🔵 PATCH | Target 생성 위치 재변경 |
| 3.1.0 | 2026-01-21 | 🟢 MINOR | Multi-horizon 예측 |
| 3.0.0 | 2026-01-18 | 🔴 MAJOR | Target 위치 변경 |

---

**Last Updated**: 2026-02-24
**Schema Version**: 3.7.1
**Status**: ✅ Stable
**Maintained by**: SignalWeaver Team
