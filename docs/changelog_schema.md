# Schema Changelog (Updated v3.7.0)

All notable changes to the data schema will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.7.0] - 2026-02-22

### 🟢 MINOR Changes - Training Integrity

변경 범위: **03단계 (Training)**
01, 02, 04, 05단계 스키마 변경 없음.

---

#### 1. log_return 타겟 Deprecated → log_close 롤백

**배경:**
v3.5.0에서 도입한 `log_return` 타겟이 v3.6.0 피처 확장(Scale-invariant features, Macro 통합) 이후
`log_close` 대비 성능 지표(RMSE, IC)가 지속적으로 열위로 나타났습니다.
이론적으로 정상(stationary) 시계열인 `log_return`이 유리하나,
피처 구성과의 상호작용 혹은 스케일 문제 등 현재 원인이 불명확합니다.
원인 분석 및 개선 완료 전까지 `log_close`로 롤백하고 `log_return`을 deprecated 처리합니다.

**변경 사항:**

```
Before (v3.5.0 ~ v3.6.0): target_type = "log_return"  (기본값)
After  (v3.7.0):           target_type = "log_close"   (기본값, 권장)
                           target_type = "log_return"   → DeprecationWarning 자동 발생
```

`"log_return"` 설정 시 발생하는 경고:

```python
DeprecationWarning: [DEPRECATED v3.7.0] target_type='log_return'은 deprecated 상태입니다.
피처 확장 이후 log_close 대비 성능이 열위로 확인되어 롤백되었습니다.
원인 분석 및 개선 완료 전까지 target_type='log_close'를 사용하세요.
재도입 여부는 향후 버전에서 결정됩니다.
```

**타겟 컬럼명 복원:**

```
Before (v3.5.0 ~ v3.6.0): target_log_return_h{n}   (정상 시계열, 현재 deprecated)
After  (v3.7.0):           target_log_close_h{n}    (v3.4.0과 동일, 현재 권장)
```

**예측 파일 컬럼 복원:**

```python
# val_predictions.parquet / test_predictions.parquet 공통 (v3.7.0~)
columns = [
    'date', 'ticker', 'fold',

    # 모델 원시 예측 (log_close 모드, v3.7.0~)
    'pred_target_log_close_h1', ..., 'pred_target_log_close_h5',

    # 정답값
    'true_target_log_close_h1', ..., 'true_target_log_close_h5',
]
```

**03b_train_ensemble.ipynb pred_cols 필터 복원:**

```python
# Before (v3.5.0 ~ v3.6.0)
pred_cols = [c for c in df.columns if c.startswith('pred_target_log_return_h')]

# After (v3.7.0)
pred_cols = [c for c in df.columns if c.startswith('pred_target_log_close_h')]
```

**config.yaml 변경:**

```yaml
# Before (v3.5.0 ~ v3.6.0)
training:
  target_type: "log_return"

# After (v3.7.0)
training:
  target_type: "log_close"   # 롤백. "log_return" 설정 시 DeprecationWarning 발생
```

**수정된 파일:**
- 🔧 `src/modeling/trainer.py` — 기본값 `"log_close"` 복원, `"log_return"` 시 `DeprecationWarning`

---

#### 2. Embargo Gap 도입 — 훈련/검증 경계 누수 방지

**배경:**
타겟이 h5(5일 후)일 때, 검증 구간 직전 5 거래일의 훈련 샘플은 검증 타겟의 미래 날짜와 겹칩니다.
이 구간을 제거하지 않으면 미묘한 look-ahead bias가 발생합니다.

**설계 원칙 — 검증 윈도우 크기 보존:**

embargo gap은 검증 시작을 미루는 것이 아니라, **훈련 샘플의 끝을 앞당기는 방식**으로 적용합니다.
검증/테스트 윈도우 크기는 변하지 않습니다.

```
❌ 잘못된 방식 (검증 윈도우 축소):
   실제 훈련: [0, E]       검증: [E+G, E+G+V]   ← 검증이 G만큼 밀림

✅ 올바른 방식 (훈련 샘플 손실, 윈도우 보존):
   실제 훈련: [0, E-G]     embargo: [E-G, E]
   검증:      [E, E+V]                           ← 윈도우 크기 변화 없음
```

**Embargo gap 자동 계산 (`max(horizons)`):**

`embargo_gap_days`는 별도 파라미터나 config 항목이 없습니다.
`run()` 내부에서 `G = max(self.horizons)`로 자동 계산되므로,
`horizons`를 변경하면 gap이 자동으로 연동됩니다.

```
horizons = [1, 2, 3, 4, 5]  →  G = 5 (자동)
horizons = [1, 2, 3]        →  G = 3 (자동)
```

**2-Fold 구조 변경 (G = max(horizons)):***

```
Before (v3.5.0 ~ v3.6.0):
  [검증 폴드] 훈련: [0, E]       검증: [E, E+V]
  [테스트 폴드] 훈련: [V, E+V]   테스트: [E+V, E+V+T]

After (v3.7.0):
  [검증 폴드] 실제 훈련: [0, E-G]    embargo: [E-G, E]    검증: [E, E+V]
  [테스트 폴드] 실제 훈련: [V, E+V-G] embargo: [E+V-G, E+V] 테스트: [E+V, E+V+T]

E = train_end, V = valid_window_days, T = test_window_days, G = max(horizons)
```

훈련 샘플 손실(G일)은 01단계 수집 시작일을 앞당겨 보완합니다.

**trainer.run() 시그니처 — 변경 없음:**

```python
# v3.6.0과 시그니처 동일. 내부에서 G = max(self.horizons) 자동 계산.
results = trainer.run(
    df=df,
    train_end=train_cfg['train_end'],
    valid_window_days=train_cfg['valid_window_days'],
    test_window_days=train_cfg['test_window_days'],
    fit_kwargs=fit_kwargs
)

# 반환 dict에 신규 키 추가
results['embargo_gap_days']   # ✨ 적용된 embargo gap (= max(horizons))
```

**수정된 파일:**
- 🔧 `src/modeling/trainer.py` — `G = max(self.horizons)` 자동 계산, 날짜 인덱스 슬라이싱 수정

---

### 호환성 (v3.6.0 → v3.7.0)

**비호환 (재실행 필요):**
- 03단계 전체 재실행 필수
  - 타겟 컬럼명: `target_log_return_h{n}` → `target_log_close_h{n}`
  - 예측 파일 컬럼: `pred_target_log_return_h{n}` → `pred_target_log_close_h{n}`
  - v3.6.0 모델(.pkl) 키가 `target_log_return_h{n}` → 04단계에서 컬럼 불일치
- 03b 재실행 필수
  - `pred_cols` 필터 prefix 복원: `pred_target_log_return_h` → `pred_target_log_close_h`

**호환 (재실행 불필요):**
- 01단계 Raw 데이터: 변경 없음
- 02단계 Feature 데이터셋: `target_log_close` 기준 컬럼 유지, 변경 없음
- 04단계: `pred_log_close`, `pred_close` 컬럼명 유지 (하위 호환)
- 05단계: 입력 스키마 변경 없음

---

## [3.6.0] - 2026-02-21

### 🟢 MINOR Changes - Scale-Invariant Features & IC Evaluation

변경 범위: **02단계 (Feature), 03단계 (Training), 05단계 (Universe), 98단계 (신설)**

---

#### 1. 피처 엔지니어링 개선 — Scale-Invariance

**배경:**
트리 모델은 절대 가격 스케일에 무관하나, 피처 자체가 가격 수준에 종속되면
종목 간 비교 시 bias가 발생할 수 있습니다. 주요 피처를 무차원(scale-invariant) 형태로 변환합니다.

**변경 피처 (src/features/builder.py):**

| 이전 (v3.5.0) | 이후 (v3.6.0) | 변환 방식 |
|---|---|---|
| `feature_ma_5`, `feature_ma_60` (절대 가격) | `feature_disparity_5`, `feature_disparity_60` | `close / ma - 1` (이격도) |
| `feature_bb_upper`, `feature_bb_lower` (절대 가격) | `feature_bb_pct_b`, `feature_bb_bandwidth` | %B, Bandwidth (무차원) |
| `liquidity_score` (원화 거래대금) | `feature_log_liquidity` | `log(거래대금)` |

**신규 피처:**

```python
# 매크로 피처 (99_meta/macro_regime.parquet에서 조인, feature_ 접두어 자동 부여)
'feature_kospi_return'      # KOSPI 일간 로그 수익률
'feature_usdkrw_return'     # USD/KRW 환율 변화율
'feature_vix'               # VIX 지수 (공포 지수)
'feature_regime'            # 시장 레짐 (-1=Bear, 0=Neutral, 1=Bull, 인코딩)

# 기업 정보 피처
'feature_is_kospi'          # KOSPI=1 / KOSDAQ=0 (FDR StockListing 기반)

# 캘린더 피처
'feature_is_monday'         # 월요일 여부 (0/1)
'feature_is_friday'         # 금요일 여부 (0/1)
```

**제거 피처:**
- `sector` (범주형 고차원 변수 → 피처 인플레이션 방지 목적으로 제외)

**접두어 규칙 확립:**
모든 학습 피처에 `feature_` 접두어를 의무화합니다.
`feature_cols = [c for c in df.columns if c.startswith('feature_')]` 자동 인식.

**수정된 파일:**
- 🔧 `src/features/builder.py` — Disparity, %B/Bandwidth, log_liquidity 변환, sector 제거
- 🔧 `02_build_dataset.ipynb` — 매크로 병합, is_kospi 병합, 캘린더 피처 추가

---

#### 2. 평가 지표 고도화 — IC / ICIR 도입

**배경:**
RMSE는 예측값의 절대 오차를 측정하나, 실제 투자에서 중요한 것은 종목 간 상대적 순위 예측력입니다.
Cross-Sectional IC(Spearman 상관계수)를 주요 지표로 추가합니다.

**지표 정의:**

```
IC (Information Coefficient):
  날짜별 전 종목에 걸쳐 예측값과 실제값 간 Spearman 상관계수 계산 후 평균.
  IC > 0: 예측이 실제 수익률 순위를 올바르게 맞춤.
  IC > 0.05: 실용적 수준의 예측력.

ICIR (IC Information Ratio):
  IC_mean / IC_std. 예측 안정성 지표.
  ICIR > 0.5: 안정적인 알파 신호.
```

**trainer.py 변경:**

```python
# Before (v3.5.0): RMSE만 계산
return {'avg_rmse': ..., 'per_horizon': {'rmse': ...}}

# After (v3.6.0): RMSE + IC + ICIR
return {
    'avg_rmse': ...,
    'avg_ic'  : ...,       # ✨ 신규
    'per_horizon': {
        'rmse'   : ...,
        'ic_mean': ...,    # ✨ 신규
        'icir'   : ...,    # ✨ 신규
    }
}
```

**03b_train_ensemble.ipynb 앙상블 가중치 최적화 목표 변경:**

```python
# Before (v3.5.0): RMSE 최소화
weights = minimize(lambda w: rmse(ensemble_pred(w), true), ...)

# After (v3.6.0): -IC 최소화 (= IC 최대화)
weights = minimize(lambda w: -ic(ensemble_pred(w), true), ...)
```

**05단계 평가 지표 변경 (select_universe.py):**

```python
# Before (v3.5.0): directional_accuracy (방향 정확도)
accuracy_score = (pred_direction == true_direction).mean()

# After (v3.6.0): Time-series IC (Spearman 상관계수)
accuracy_score = spearmanr(pred_log_close, true_log_close).correlation
```

**수정된 파일:**
- 🔧 `src/modeling/trainer.py` — `_evaluate()` 메서드에 IC/ICIR 계산 추가
- 🔧 `src/universe/select_universe.py` — `directional_accuracy` → IC 기반 평가
- 🔧 `03b_train_ensemble.ipynb` — 가중치 최적화 목표 `-IC`로 변경

---

#### 3. 전역 메타 데이터 파이프라인 신설 — 98단계

**배경:**
매크로 경제 지표(KOSPI, 환율, VIX)와 시장 레짐 정보를 전역 메타 데이터로 중앙화합니다.
외부 데이터 수집을 모두 FinanceDataReader로 전환하여 pykrx 로그인 이슈를 우회합니다.

**신설 파일:**
- 📄 `98_save_macro_data.ipynb` — 매크로/레짐 데이터 수집 및 저장
- 📁 `data/99_meta/macro_regime.parquet` — 수집 결과 저장

**매크로 데이터 저장 구조:**

```python
# data/99_meta/macro_regime.parquet
columns = [
    'date',              # 거래일
    'kospi_return',      # KOSPI 일간 로그 수익률
    'usdkrw_return',     # USD/KRW 환율 변화율
    'vix',               # VIX 지수
    'regime',            # 시장 레짐 (0=Bear, 1=Bull)
]
# → 02_build_dataset.ipynb에서 조인 후 feature_ 접두어 부여
```

**폴더 구조 추가:**

```
data/99_meta/               # ✨ v3.6.0 신설
  ├── krx_calendar.csv      # 영업일 캘린더 (기존)
  └── macro_regime.parquet  # ✨ 매크로/레짐 데이터 (신규)
```

---

### 호환성 (v3.5.0 → v3.6.0)

**비호환 (재실행 필요):**
- 02단계 재실행 필수
  - 피처 컬럼명 변경: `feature_ma_{n}` → `feature_disparity_{n}`, `feature_bb_*` → `feature_bb_pct_b/bandwidth`
  - 신규 컬럼: `feature_log_liquidity`, `feature_is_kospi`, `feature_is_monday/friday`, `feature_kospi_return` 등
  - 제거 컬럼: `sector`, `feature_ma_5/60`, `feature_bb_upper/lower`, `liquidity_score`
- 03단계 재실행 필수 (피처 컬럼 변경에 따른 모델 재학습)
- 98단계 선행 실행 필요 (`macro_regime.parquet` 미존재 시 02단계 경고 출력 후 매크로 피처 제외)

**호환 (재실행 불필요):**
- 01단계 Raw 데이터: 변경 없음
- 04단계: 예측 파일 구조 변경 없음
- 05단계: `pred_log_close` 컬럼 유지

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
  └── predictions.parquet

# After (v3.5.0)
data/03_training/{date}/{model_name}/
  ├── val_predictions.parquet      # 검증 폴드 (앙상블 가중치 최적화 입력)
  └── test_predictions.parquet     # 테스트 폴드 (최종 평가 전용)
```

**설정 변경 (config.yaml):**

```yaml
# Before (v3.4.0)
training:
  num_valid: 3

# After (v3.5.0)
training:
  valid_window_days: 60
  test_window_days: 60
```

**수정된 파일:**
- 🔧 `src/modeling/trainer.py` — 2-Fold 고정 구조, `val_predictions` 저장 추가
- 🔧 `03_train_predict.ipynb` — 저장 셀: 파일 2개 분리
- 🔧 `03b_train_ensemble.ipynb` — 가중치 최적화 입력 교체
- 🔧 `config/config.yaml` — `num_valid` 제거

---

#### 2. log_return 타겟 전환 (→ v3.7.0에서 deprecated 처리)

**배경:**
`target_log_close`가 비정상(non-stationary) 시계열이라는 이론적 우려로 `log_return`으로 전환.
이후 v3.6.0 피처 확장 후 성능 열위가 확인되어 v3.7.0에서 롤백됨.

**타겟 컬럼명:**

```
Before (v3.4.0): target_log_close_h{n}
After  (v3.5.0): target_log_return_h{n}   ← v3.7.0에서 deprecated
```

---

### 호환성 (v3.4.0 → v3.5.0)

**비호환:** 03단계 모델/예측 파일 전체 재실행 필요.
**호환:** 01, 02, 04, 05단계 변경 없음.

---

## [3.4.0] - 2026-02-17

### 🟢 MINOR Changes - Model Diversification

모델 다양화 (RandomForest + Ensemble)

**변경 사항:**
- RandomForest 멀티아웃풋 모델 추가
- 앙상블 학습 단계(03b_train_ensemble.ipynb) 도입
- 모델별 폴더 계층 추가: `03_training/{date}/{model_name}/`
- config: `active_model` 옵션 추가

**신규 추가 파일:**
- `src/models/randomforest_model.py` — RandomForestMultiModel
- `src/models/ensemble_model.py` — EnsembleModel
- `03b_train_ensemble.ipynb` — OOF 기반 최적 가중치 탐색

**폴더 구조 변경:**

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

**설정 변경 (config.yaml):**

```yaml
randomforest_params: { n_estimators: 40, max_depth: 8, ... }
active_model: "ensemble"   # 'lightgbm' | 'randomforest' | 'ensemble'
```

**수정된 파일:** `src/models/base.py`, `src/models/artifact.py`, `src/utils/config.py`,
`03_train_predict.ipynb`, `04_forecast_future.ipynb`, `config/config.yaml`

**호환성:**
- ✅ v3.3.0 LightGBM 모델 자동 호환
- ❌ v3.4.0 모델은 v3.3.0 코드에서 로드 불가

---

## [3.3.0] - 2026-02-09

### 🟢 MINOR Changes - Infrastructure Modernization

**H1 - 데이터 폴더 구조 개선:**

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

**H2 - ProjectPaths 클래스 도입:** `src/utils/config.py`에 추가. 모든 노트북 경로 관리 통일.

**H3 - select_universe.py 모듈 정리:** Facade Pattern 적용, 단일 진입점 함수화.

---

## [3.2.1] - 2026-02-09

### 🔵 PATCH Changes

**1. Multi-Horizon Walk-Forward 버그 수정:** Horizon별 dropna 후 길이 불일치 → 교집합 인덱스 사용.

**2. Recursive Extension Chunk 오염 방지:** Chunk 1+ 예측 시 volume → 최근 20일 평균으로 대체.

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

**Last Updated**: 2026-02-22
**Schema Version**: 3.7.0
**Status**: ✅ Stable
**Maintained by**: SignalWeaver Team
