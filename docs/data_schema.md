# 📄 Data Schema Definition (v3.8.0)

본 스키마는 SignalWeaver 프로젝트의 데이터 계약을 정의합니다.

---

## 📌 Schema Version & Metadata

| 속성 | 값 |
|------|-----|
| **Schema Version** | `3.8.0` |
| **Last Updated** | 2026-02-28 |
| **Latest Changes** | MLP 모델 도입 + 앙상블 조합 동적 지정 + 04단계 건너뜀 처리 |
| **Compatibility** | 단일 모델 운용 시 v3.7.x 완전 호환. 앙상블 사용 시 `active_model` 값 변경 필요 |

---

## 🔄 최근 변경 이력 요약

### v3.8.0 (2026-02-28) - 🟢 MINOR
- **MLP Multi-output 모델 신설**: `src/models/mlp_model.py`. PyTorch 기반 단일 네트워크로 h1~h5 동시 출력. `StandardScaler` 내부 캡슐화, Early Stopping 지원.
- **앙상블 조합 동적 지정**: `active_model`에 `+` 구분자로 조합 지정. `"lgbm+rf"`, `"lgbm+rf+mlp"` 등. 폴더명은 short 약칭 조합 사용.
- **모델명 정칭/약칭 허용**: `lightgbm`/`lgbm`, `randomforest`/`rf`, `mlp`. 단일 모델 폴더는 canonical 정칭 유지(하위 호환).
- **config.py 헬퍼 신설**: `resolve_model_name()`, `parse_active_model()`, `is_ensemble()`, `get_folder_name()`.
- **04단계 건너뜀 처리**: 종목별 예측 루프 `try/except` 감싸기. 실패 종목 `skipped_tickers`에 수집 후 루프 완료 시 보고.

### v3.7.2 (2026-02-25) - 🔵 PATCH
- **비현실적 수익률 필터링**: `config.yaml`에 `strategy.max_daily_return` 추가. `utils/trading.py`, `universe/select_universe.py`에서 일평균 로그 수익률 상한 초과 거래를 건너뛰고 차선 거래 제안.

### v3.7.1 (2026-02-24) - 🔵 PATCH
- **04단계 피처 스키마 동기화**: `calculate_features_for_ticker` → builder.py v3.6.0 일치.
- **97단계 신설**: `97_forecast_macro.ipynb` — 매크로 지표 미래값 추정(Damped Holt / SES).
- **매크로/정적/캘린더 피처 미래값 반영**: `new_row` 구성 시 NaN 누적 버그 수정.
- **모델 로드 방식 수정**: `pickle.load()` → `Model.load()` 클래스메서드.

### v3.7.0 (2026-02-22) - 🟢 MINOR
- **log_return deprecated → log_close 롤백**: 타겟 컬럼명 `target_log_close_h{n}` 복원.
- **Embargo Gap 도입**: `G = max(horizons)` 자동 계산. 훈련 샘플 끝을 G일 앞당김.

### v3.6.0 (2026-02-21) - 🟢 MINOR
- **Scale-invariant 피처**: MA → Disparity, BB → %B/Width, liquidity_score → log_liquidity.
- **매크로/레짐 피처 통합**: `data/99_meta/macro_regime.parquet`.
- **98단계 신설**, **IC/ICIR 평가 도입**, **`feature_` 접두어 규칙 확립**.

### v3.5.0 (2026-02-20) - 🟢 MINOR
- **2-Fold Walk-Forward 구조**: `val_predictions.parquet` + `test_predictions.parquet` 분리.

### v3.4.0 (2026-02-17) - 🟢 MINOR
- **RandomForest + 앙상블**: `03_training/{date}/{model_name}/` 계층 확장.

---

## 📌 1. 파일 저장 규칙 / 포맷

### 1.1 기본 포맷

| 단계 | 폴더 | 포맷 |
|------|------|------|
| **01단계 (Raw)** | `data/01_raw/{date}/` | CSV + 통합 Parquet |
| **02단계 (Processed)** | `data/02_processed/{date}/` | Parquet + 선택적 CSV |
| **03단계 (Training)** | `data/03_training/{date}/{folder_name}/` | Parquet + 모델별 폴더 |
| **04단계 (Forecasts)** | `data/04_forecasts/{date}/{folder_name}/` | Parquet + 선택적 CSV |
| **05단계 (Universe)** | `data/05_universe/{date}/{folder_name}/` | Parquet + CSV + JSON |
| **99_meta** | `data/99_meta/` | Parquet + CSV |

### 1.2 파일 네이밍 규칙 (✨ v3.8.0)

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

# 03단계: 학습/검증/테스트 예측 (✨ v3.8.0: folder_name 동적 결정)
data/03_training/{YYYYMMDD}/
  ├── lightgbm/                          # 단일 모델: canonical 정칭
  │   ├── v1_lgbm_{YYYYMMDD}_{hash}.pkl
  │   ├── registry.json
  │   ├── val_predictions.parquet
  │   └── test_predictions.parquet
  ├── randomforest/  { 동일 구조 }
  ├── mlp/           { 동일 구조 }       # ✨ v3.8.0 신설
  └── lgbm+rf/       { 동일 구조 }       # ✨ v3.8.0: 앙상블은 short 약칭 조합
      # lgbm+mlp/, lgbm+rf+mlp/ 등 조합에 따라 생성

# 04단계: 미래 예측 (folder_name 동일 규칙 적용)
data/04_forecasts/{YYYYMMDD}/
  ├── lightgbm/future_forecasts.parquet
  ├── randomforest/future_forecasts.parquet
  ├── mlp/future_forecasts.parquet
  └── lgbm+rf/future_forecasts.parquet

# 05단계: 유니버스 선정 (folder_name 동일 규칙 적용)
data/05_universe/{YYYYMMDD}/
  └── {folder_name}/
      ├── universe_full.parquet
      ├── universe_candidates.parquet
      ├── investment_report.csv
      ├── investment_report.xlsx
      └── filter_statistics.json

# 전역 메타 데이터
data/99_meta/
  ├── krx_calendar.csv
  ├── macro_regime.parquet               # 98단계 출력 (과거 실측)
  └── macro_regime_forecast.parquet      # 97단계 출력 (미래 추정) ✨ v3.7.1
```

### 1.3 folder_name 결정 규칙 (✨ v3.8.0)

`active_model` 값에 따라 `get_folder_name()`이 폴더명을 결정합니다.

| active_model 값 | folder_name | 비고 |
|---|---|---|
| `"lightgbm"` 또는 `"lgbm"` | `lightgbm` | 단일 모델: canonical 정칭 |
| `"randomforest"` 또는 `"rf"` | `randomforest` | 단일 모델: canonical 정칭 |
| `"mlp"` | `mlp` | 단일 모델 |
| `"lgbm+rf"` | `lgbm+rf` | 앙상블: short 약칭 조합 |
| `"lgbm+rf+mlp"` | `lgbm+rf+mlp` | 앙상블: short 약칭 조합 |
| `"lightgbm+randomforest"` | `lgbm+rf` | 정칭 입력도 short로 정규화 |

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
    'Volume': 'float64',
}
index.name = 'ticker'
```

---

## 📌 4. Step 2 (Processed) - Feature + Target 스키마

### 4.1 Feature 카테고리 (v3.6.0~)

모든 학습 피처는 `feature_` 접두어를 가집니다.
`feature_cols = [c for c in df.columns if c.startswith('feature_')]`로 자동 인식.

| 카테고리 | 컬럼 예시 | 설명 |
|---|---|---|
| 기술적 지표 (이격도) | `feature_ma_5_disparity`, `feature_ma_60_disparity` | (close/MA) - 1 |
| 기술적 지표 (볼린저) | `feature_bb_pct_b`, `feature_bb_width` | 무차원 |
| 기술적 지표 (기타) | `feature_volatility_20`, `feature_volume_ratio`, `feature_rsi_14` | |
| 유동성 | `feature_log_liquidity` | log1p(close × volume 20일 평균) |
| 매크로 | `feature_kospi`, `feature_usd_krw`, `feature_vix`, `feature_us_return_1d`, `feature_market_regime` | 98단계 출처 |
| 정적 | `feature_is_kospi` | ticker별 고정값 |
| 캘린더 | `feature_is_monday`, `feature_is_friday` | 요일 플래그 |

### 4.2 Target 컬럼 (v3.7.0~)

```python
# 02단계에서 생성되는 기준값 (Trainer 내부에서 horizon별 shift)
'target_log_close'        # log(close), 기준 컬럼

# Trainer 내부에서 동적 생성 (horizon별)
'target_log_close_h1'     # log(close(t+1))
'target_log_close_h2'     # log(close(t+2))
...
'target_log_close_h5'     # log(close(t+5))

# [DEPRECATED v3.7.0]
# 'target_log_return_h{n}'  — log_close 대비 성능 열위, 사용 금지
```

---

## 📌 5. Step 3 (Training) - 모델 & 예측 스키마

### 5.1 폴더 구조

```
data/03_training/{YYYYMMDD}/{folder_name}/
  ├── *.pkl                       # 모델 객체
  ├── registry.json               # 메타데이터
  ├── val_predictions.parquet     # 검증 폴드 예측 (앙상블 가중치 최적화용)
  └── test_predictions.parquet    # 테스트 폴드 예측 (최종 성능 평가 전용)
```

### 5.2 2-Fold Walk-Forward 구조 (v3.7.0: Embargo Gap 추가)

```
전체 데이터 타임라인 (G = max(horizons) = 5):

|────────────|░░░░░|──────────────|──────────────|
0           E-G    E            E+V            E+V+T

[검증 폴드]
  실제 훈련: [0, E-G]      ← embargo gap(G일) 제거됨
  embargo:  [E-G, E]       ← look-ahead 오염 구간 (미사용)
  검증:     [E, E+V]       → val_predictions.parquet

[테스트 폴드]  (훈련 구간 rolling)
  실제 훈련: [V, E+V-G]    ← embargo gap(G일) 제거됨
  embargo:  [E+V-G, E+V]
  테스트:   [E+V, E+V+T]   → test_predictions.parquet

E = train_end (config.yaml)
V = valid_window_days (거래일)
T = test_window_days (거래일)
G = max(horizons) — 자동 계산, 별도 설정 불필요
```

### 5.3 val_predictions.parquet 스키마

앙상블 가중치 최적화 전용. 테스트셋과 완전히 분리됩니다.

```python
columns = [
    'date', 'ticker', 'fold',            # fold = 'valid'
    'pred_target_log_close_h1',
    'pred_target_log_close_h2',
    'pred_target_log_close_h3',
    'pred_target_log_close_h4',
    'pred_target_log_close_h5',
    'true_target_log_close_h1',
    'true_target_log_close_h2',
    'true_target_log_close_h3',
    'true_target_log_close_h4',
    'true_target_log_close_h5',
]
```

### 5.4 test_predictions.parquet 스키마

val_predictions와 동일한 컬럼 구성. `fold = 'test'`.

### 5.5 평가 지표 (v3.6.0~)

```python
metrics = {
    'avg_rmse'    : float,
    'avg_ic'      : float,
    'per_horizon' : {
        'target_log_close_h1': {
            'rmse'   : float,
            'ic_mean': float,
            'icir'   : float,
        },
        # h2~h5 동일
    },
    'samples' : int,
}
```

### 5.6 지원 모델 클래스 (✨ v3.8.0)

| 모델 | 클래스 | 파일 | 특징 |
|---|---|---|---|
| LightGBM | `LightGBMModel` | `src/models/lightgbm_model.py` | Horizon별 독립 Booster |
| RandomForest | `RandomForestMultiModel` | `src/models/randomforest_model.py` | MultiOutputRegressor 래핑 |
| MLP | `MLPModel` | `src/models/mlp_model.py` | 단일 네트워크, 공유 잠재 표현 |
| Ensemble | `EnsembleModel` | `src/models/ensemble_model.py` | 가중 평균 래퍼 |

**MLP 아키텍처:**
```
Input(feature_dim)
  → [Linear → BatchNorm1d → ReLU → Dropout(rate)] × len(hidden_dims)
  → Linear(output_dim=5)
```

---

## 📌 6. Step 3b (Ensemble) - 앙상블 가중치 최적화 (✨ v3.8.0)

```python
# 가중치 최적화 입력: val_predictions (검증 폴드 전용)
# 최적화 목표: -IC(Spearman) 최소화

# n-model (2개 이상): SLSQP (sum=1 등식 제약)
result = minimize(neg_ic, x0=[1/n]*n, bounds=[(0,1)]*n,
                  method='SLSQP',
                  constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

# 테스트 폴드 평가: test_predictions으로 최종 성능 확인 (가중치 변경 없음)
```

**입력 순서 주의:** `"rf+mlp"`와 `"mlp+rf"`는 별개 조합으로 처리됩니다.

---

## 📌 7. Step 4 (Forecasts) - 미래 예측 결과 스키마

### 7.1 Recursive Extension 역산 로직 (v3.7.0: log_close 기본값)

```python
# log_close 모드 (v3.7.0~ 기본값, 권장)
pred_log_close = model.predict(X)
pred_close     = exp(pred_log_close)

# [DEPRECATED] log_return 모드 (v3.5.0~v3.6.0, 비권장)
# pred_log_close = log_close_base + model.predict(X)
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
    # 'pred_log_return' — log_return 모드에서만 존재 (현재 deprecated)
]
```

### 7.3 예측 실패 종목 처리 (✨ v3.8.0)

종목별 루프를 `try/except`로 감싸 예측 실패 시 해당 종목을 건너뜁니다.
실패 종목은 `skipped_tickers: List[(ticker, ErrorType, message)]`에 수집되며
루프 완료 후 출력됩니다. 부분 예측(청크 일부만 성공)은 결과에서 제외됩니다.

---

## 📌 8. Step 5 (Universe) - 최종 선정 결과 스키마

### 8.1 universe_full.parquet

```python
columns = [
    'ticker', 'name',
    'accuracy_score',        # IC 기반 정확도 (v3.6.0~)
    'profitability_score',
    'risk_composite',
    'composite_score',
    'selected',
]
```

### 8.2 universe_candidates.parquet

```python
columns = [
    'ticker', 'name',
    'composite_score',
    'rank',
    'recommendation',        # 'BUY', 'HOLD', 'SELL'
]
```

### 8.3 수익률 상한 필터 (✨ v3.7.2)

최적 거래 탐색 시 일평균 로그 수익률이 `strategy.max_daily_return`을 초과하는
거래는 건너뛰고 차선 거래를 제안합니다.

```python
max_log_return = np.log1p(cfg['strategy']['max_daily_return'])
# daily_log_return > max_log_return → skip, 차선 거래로 이동
```

---

## 📌 9. Step 99_meta - 전역 메타 데이터 스키마 (v3.6.0~)

### 9.1 macro_regime.parquet (98단계 출력 — 과거 실측값)

```python
columns = [
    'date',           # 거래일
    'kospi',          # KOSPI 지수
    'usd_krw',        # USD/KRW 환율
    'vix',            # VIX 지수
    'us_return_1d',   # 직전 거래일 미국 시장 수익률
    'market_regime',  # 시장 레짐 (-1=Bear, 0=Neutral, 1=Bull)
]
# 02단계 조인 시 feature_ 접두어 자동 부여
```

### 9.2 macro_regime_forecast.parquet (97단계 출력 — 미래 추정값, v3.7.1~)

```python
# macro_regime.parquet와 동일한 스키마. 미래 영업일 행만 포함.
# 04단계에서 macro_regime.parquet와 concat 후 date 기준 left join.
columns = [
    'date',           # 미래 거래일 (krx_calendar.csv 기준)
    'kospi',          # Damped Holt 추정값 (φ=0.90)
    'usd_krw',        # Damped Holt 추정값 (φ=0.85)
    'vix',            # Damped Holt 추정값 (φ=0.85)
    'us_return_1d',   # SES 추정값 (zero 수렴)
    'market_regime',  # kospi 추정값으로 재계산
]
```

### 9.3 krx_calendar.csv

```python
columns = ['date']   # 영업일 날짜 목록 (datetime)
```

---

## 📌 10. 전체 파이프라인 데이터 흐름 (v3.8.0)

```
[98 Meta] (선행 실행)
  98_save_macro_data.ipynb
  → data/99_meta/macro_regime.parquet

[97 Macro Forecast] (04단계 전 선행 실행, v3.7.1~)
  97_forecast_macro.ipynb
  → data/99_meta/macro_regime_forecast.parquet
        │
        ▼
[02 Processed]
  dataset.parquet
  └─ target_log_close
  └─ feature_ma_{w}_disparity, feature_bb_pct_b/width, feature_log_liquidity
  └─ feature_kospi, feature_usd_krw, feature_vix, feature_us_return_1d, feature_market_regime
  └─ feature_is_kospi, feature_is_monday/friday
        │
        ▼
[03 Training] ← target_type="log_close", Embargo Gap G=max(horizons)
  단일 모델: lightgbm/ | randomforest/ | mlp/
        │
        ├─ [검증 폴드] → val_predictions.parquet  ──┐
        │                                             │ -IC 최소화
        └─ [테스트 폴드] → test_predictions.parquet  │
                                  │                  │
                         최종 성능 평가 전용          │
        │                                            │
        ▼                                            │
[03b Ensemble] (active_model에 '+' 포함 시)          │
  parse_active_model() → 구성 모델 동적 로드          │
  2-model: L-BFGS-B / n-model: SLSQP ◀─────────────┘
  → {folder_name}/ (예: lgbm+rf/, lgbm+rf+mlp/)
        │
        ▼
[04 Forecasts]
  is_ensemble() / resolve_model_name() 기반 모델 동적 로드
  종목별 try/except — 실패 종목 skipped_tickers 수집
  → {folder_name}/future_forecasts.parquet
        │
        ▼
[05 Universe]
  max_daily_return 수익률 상한 필터 (v3.7.2~)
  IC 기반 accuracy_score 산출
  → {folder_name}/investment_report
```

---

## 📌 11. 설정 파일 스키마 (config.yaml) - v3.8.0

```yaml
training:
  train_end: "2025-08-14"
  valid_window_days: 60
  test_window_days: 60
  target_col_name: "target_log_close"
  target_type: "log_close"        # "log_return" → DeprecationWarning
  horizons: [1, 2, 3, 4, 5]

  lgbm_params:
    objective: "regression"
    num_leaves: 31
    learning_rate: 0.05
    # ...

  randomforest_params:
    n_estimators: 40
    max_depth: 15
    # ...

  mlp_params:                     # ✨ v3.8.0 신설
    hidden_dims:   [128, 64, 32, 32, 16]
    dropout_rates: [0.2, 0.2, 0.1, 0.0, 0.0]
    learning_rate: 0.001
    batch_size:    2048
    epochs:        200
    patience:      15
    weight_decay:  0.0001

# ✨ v3.8.0: '+' 구분자로 앙상블 조합 지정 가능
# 단일: "lightgbm" | "randomforest" | "mlp"
# 약칭: "lgbm" | "rf"
# 앙상블: "lgbm+rf" | "lgbm+mlp" | "lgbm+rf+mlp" 등
active_model: "lightgbm"

strategy:
  min_hold_days: 5
  max_daily_return: 0.16          # ✨ v3.7.2: 일평균 수익률 상한
```

---

## 📌 12. 호환성 노트

### v3.7.x → v3.8.0 마이그레이션

**비호환 (조치 필요):**
- `active_model: "ensemble"` 사용 시 → `"lgbm+rf"` 등 명시적 조합으로 변경 필요
- 기존 `ensemble/` 폴더 → `lgbm+rf/`로 이동 또는 03b 재실행

**호환 (재실행 불필요):**
- 단일 모델(`lightgbm`, `randomforest`) 운용: 변경 없음
- 01, 02, 03, 05단계 산출물: 변경 없음

### v3.7.1 → v3.7.2 마이그레이션

**호환 (재실행 불필요):**
- 01~04단계: 변경 없음
- 05단계: 입력 스키마 변경 없음, 필터링 결과만 달라짐

### v3.6.0 → v3.7.0 마이그레이션

**비호환 (재실행 필요):**
- 03단계 전체 재실행 필수
  - 타겟 컬럼명: `target_log_return_h{n}` → `target_log_close_h{n}`
  - 예측 파일 컬럼: `pred_target_log_return_h{n}` → `pred_target_log_close_h{n}`

**호환 (재실행 불필요):**
- 01, 02단계, 04단계 컬럼명 유지, 05단계 입력 스키마 변경 없음

### v3.5.0 → v3.6.0 마이그레이션

**비호환 (재실행 필요):**
- 02단계: 피처 컬럼명 전면 변경
- 03단계: 피처 변경에 따른 모델 재학습
- 98단계 선행 실행 필요

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
| **3.8.0** | 2026-02-28 | 🟢 MINOR | MLP 모델 + 앙상블 동적 조합 + 04단계 건너뜀 처리 |
| **3.7.2** | 2026-02-25 | 🔵 PATCH | 비현실적 수익률 필터링 (max_daily_return) |
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

**Last Updated**: 2026-02-28
**Schema Version**: 3.8.0
**Status**: ✅ Stable
**Maintained by**: SignalWeaver Team
