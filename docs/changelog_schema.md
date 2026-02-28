# Schema Changelog (Updated v3.8.0)

All notable changes to the data schema will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.8.0] - 2026-02-28

### 🟢 MINOR Changes — MLP 모델 도입 및 앙상블 확장성

변경 범위: **03단계 (Training), 03b단계 (Ensemble), 04단계 (Forecasts), src/utils, src/models**
01, 02, 05단계 스키마 변경 없음.

---

#### 1. MLP Multi-output 모델 신설

**배경:**
LightGBM(타겟별 독립 Booster)·RandomForest(MultiOutputRegressor)와 달리,
단일 네트워크가 h1~h5를 동시에 출력하는 공유 잠재 표현(shared representation) 구조의
세 번째 모델을 도입합니다. MultiOutputRegressor 없이 진정한 단일 모델로 다중 출력을 구현합니다.

**신규 파일:**
- 📄 `src/models/mlp_model.py` — `MLPModel` 클래스

**아키텍처:**

```
Input(feature_dim)
  → [Linear → BatchNorm1d → ReLU → Dropout(rate)] × len(hidden_dims)
  → Linear(output_dim)      # output_dim = len(horizons) = 5
```

**설계 결정:**
- PyTorch 기반 구현 (CPU 환경)
- `StandardScaler` 내부 캡슐화 — fit/predict/save/load 일관성 보장
- `eval_set` 기반 Early Stopping 지원 (Trainer 계약 준수)
- `drop_last=True` (훈련 DataLoader) — BatchNorm1d 배치 크기 1 오류 방지
- 손실 함수: MSELoss

**config.yaml 추가:**

```yaml
training:
  mlp_params:
    hidden_dims:    [128, 64, 32, 32, 16]
    dropout_rates:  [0.2, 0.2, 0.1, 0.0, 0.0]   # 레이어별 지정, 0.0이면 Dropout 미추가
    learning_rate:  0.001
    batch_size:     2048
    epochs:         200
    patience:       15
    weight_decay:   0.0001
```

**Trainer 호환성:**
`WalkForwardTrainer`의 `model.fit(X, y, eval_set=[...], **fit_kwargs)` 시그니처를
그대로 준수합니다. `epochs`, `patience`는 `fit_kwargs`로 전달됩니다.
Trainer 코드 수정 없음.

**수정된 파일:**
- ✨ `src/models/mlp_model.py` — 신규 생성
- 🔧 `config/config.yaml` — `mlp_params` 섹션 추가
- 🔧 `03_train_predict.ipynb` — `mlp` 분기 추가

---

#### 2. 앙상블 확장성 — 모델 조합 동적 지정

**배경:**
기존 `active_model: 'ensemble'`은 고정된 LGBM+RF 조합만 지원했습니다.
세 번째 모델(MLP) 추가와 함께 임의의 모델 조합을 지정할 수 있도록 확장합니다.

**`active_model` 필드 변경:**

```yaml
# Before (v3.7.x): 고정 조합
active_model: "ensemble"   # 항상 LGBM+RF

# After (v3.8.0): '+' 구분자로 조합 지정
active_model: "lgbm+rf"        # 2-model
active_model: "lgbm+mlp"       # 2-model
active_model: "lgbm+rf+mlp"    # 3-model
active_model: "lightgbm"       # 단일 모델 (하위 호환)
```

**모델 이름 정칭/약칭 대응표:**

| 허용 입력 | canonical (단일 모델 폴더) | short (앙상블 폴더 구성) |
|---|---|---|
| `lightgbm`, `lgbm` | `lightgbm` | `lgbm` |
| `randomforest`, `rf` | `randomforest` | `rf` |
| `mlp` | `mlp` | `mlp` |

**폴더명 결정 규칙:**

```
단일 모델 → canonical 정칭 사용 (하위 호환)
앙상블    → short 약칭을 '+' 로 연결

active_model: "lightgbm"              → data/.../lightgbm/
active_model: "lgbm+rf"              → data/.../lgbm+rf/
active_model: "lightgbm+randomforest+mlp" → data/.../lgbm+rf+mlp/
```

**가중치 최적화:** SLSQP(sum=1 등식 제약) 방법 적용.
입력 순서(`"rf+mlp"` vs `"mlp+rf"`)와 무관하게 동일한 최적 IC를 산출하며,
경계값 수렴(`[1.0, 0.0]`) 문제가 완화됩니다.

**신규 헬퍼 함수 (`src/utils/config.py`):**

```python
resolve_model_name(name)       → (canonical, short)
parse_active_model(str)        → List[(canonical, short)]
is_ensemble(str)               → bool
get_folder_name(str)           → str   # 폴더명 반환
```

**`ProjectPaths` 변경:**
- `folder_name` 필드 추가 — `get_folder_name()` 기반으로 자동 결정
- `training_dir` / `forecasts_dir` / `universe_dir`가 `folder_name` 기반으로 동적 결정
- `get_member_model_dir(alias)` 추가 — 앙상블 구성 시 개별 모델 폴더 참조용

**수정된 파일:**
- 🔧 `src/utils/config.py` — 헬퍼 함수 4개 신설, `ProjectPaths` 확장
- 🔧 `config/config.yaml` — `active_model` 주석 업데이트
- 🔧 `03_train_predict.ipynb` — `is_ensemble()` 조기 차단, 약칭 허용
- 🔧 `03b_train_ensemble.ipynb` — 전면 재작성 (동적 로드)
- 🔧 `04_forecast_future.ipynb` — `load_model` 셀: `is_ensemble()` / `resolve_model_name()` 기반 동적 로드

---

#### 3. 04단계 예측 실패 종목 건너뜀 및 보고

**배경:**
Recursive Extension 중 특정 종목에서 피처에 `inf`가 발생할 경우
(`StandardScaler.transform()` 오류 등) 전체 파이프라인이 중단되는 문제가 있었습니다.

**수정 내용:**
종목별 루프를 `try/except`로 감싸 예측 실패 시 해당 종목을 건너뛰고,
`skipped_tickers` 목록에 `(ticker, ErrorType, message)`를 수집합니다.
루프 완료 후 제외 종목 수 및 상세 목록을 출력합니다.

```python
# 루프 완료 후 출력 예시
✅ Recursive Extension 완료
   성공: 847개 / 전체: 850개

⚠️  제외된 종목: 3개
   Ticker       ErrorType                 Message
   005930       ValueError                Input X contains infinity...
```

**수정된 파일:**
- 🔧 `04_forecast_future.ipynb` — `recursive_predict` 셀 패치

---

### 호환성 (v3.7.x → v3.8.0)

**비호환 (재실행 필요):**
- `active_model: "ensemble"`을 사용하던 경우 `"lgbm+rf"`로 변경 필요
  (기존 `ensemble/` 폴더는 `lgbm+rf/`로 이동 또는 재실행)

**호환 (재실행 불필요):**
- 단일 모델(`lightgbm`, `randomforest`) 운용: 변경 없음
- 01, 02, 03, 05단계 산출물: 변경 없음

---

## [3.7.2] - 2026-02-25

### 🔵 PATCH Changes — 비현실적 수익률 거래 필터링

변경 범위: **05단계 (Universe), `src/utils/trading.py`, `src/universe/select_universe.py`**
01~04단계 스키마 변경 없음.

---

#### 1. 일평균 수익률 상한 필터링

**배경:**
Recursive Extension 예측 결과에서 비현실적으로 급등하는 종목이 투자 후보로 선정되는
사례가 발생했습니다. 최적 거래(매수·매도 시점) 탐색 시 일평균 로그 수익률이
설정 상한을 초과하는 거래는 건너뛰고 차선 거래를 제안합니다.

**설정 추가 (`config.yaml`):**

```yaml
strategy:
  max_daily_return: 0.16   # 일평균 수익률 상한 (비율)
                           # 내부적으로 np.log1p(0.16)으로 변환하여 처리
```

**처리 방식:**

```python
# utils/trading.py, universe/select_universe.py 내부
max_log_return = np.log1p(cfg['strategy']['max_daily_return'])

# 최적 거래 탐색 루프에서
daily_log_return = total_log_return / hold_days
if daily_log_return > max_log_return:
    continue   # 건너뜀 → 차선 거래로 이동
```

**수정된 파일:**
- 🔧 `src/utils/trading.py` — 일평균 수익률 검사 로직 추가
- 🔧 `src/universe/select_universe.py` — `max_daily_return` 파라미터 연동
- 🔧 `config/config.yaml` — `strategy.max_daily_return` 추가

---

### 호환성 (v3.7.1 → v3.7.2)

**호환 (재실행 불필요):**
- 01~04단계: 변경 없음
- 05단계: 입력 스키마 변경 없음, 출력 필터링 결과만 달라짐
- `config.yaml`에 `strategy.max_daily_return`이 없을 경우 기존 동작 유지
  (`trading.py`에서 키 부재 시 필터 미적용으로 처리 권장)

---

## [3.7.1] - 2026-02-24

### 🔵 PATCH Changes — 04단계 버그 수정 및 97단계 신설

변경 범위: **04단계 (Forecasts), 97단계 (신설)**
01, 02, 03, 05단계 스키마 변경 없음. v3.7.0 산출물 재실행 불필요.

---

#### 1. 04단계 `calculate_features_for_ticker` 스키마 동기화

**배경:**
v3.6.0에서 `src/features/builder.py`의 피처 스키마가 변경되었으나,
`04_forecast_future.ipynb` 내부의 `calculate_features_for_ticker` 함수가
이전 버전(v3.5.0) 스키마를 그대로 사용하고 있었습니다.
이로 인해 04단계 Recursive Extension 루프에서 모델이 학습 때와 다른 피처를 입력받는 버그가 발생했습니다.

**수정 내용 — 피처 스키마 v3.6.0 일치:**

| 이전 (누락 업데이트) | 이후 (v3.6.0 일치) |
|---|---|
| `feature_ma_5`, `feature_ma_60` (절대 가격) | `feature_ma_5_disparity`, `feature_ma_60_disparity` (이격도) |
| `feature_bb_upper`, `feature_bb_lower` (절대 가격) | `feature_bb_pct_b`, `feature_bb_width` (무차원) |
| `liquidity_score` (원화 거래대금) | `feature_log_liquidity` (로그 변환) |

**수정된 파일:**
- 🔧 `04_forecast_future.ipynb` — `calculate_features_for_ticker` 재작성

---

#### 2. 04단계 매크로/정적/캘린더 피처 미래값 반영

**배경:**
Recursive Extension 루프에서 `new_row`(미래 예측 행)를 `df_ticker`에 추가할 때
매크로 피처(`feature_kospi` 등), 정적 피처(`feature_is_kospi`),
캘린더 피처(`feature_is_monday/friday`)가 채워지지 않아 NaN이 누적되는 버그가 있었습니다.

**수정 내용:**

```python
new_row = {
    **get_macro_row(pred_date),
    'feature_is_kospi'  : is_kospi_val,
    'feature_is_monday' : int(pred_date.weekday() == 0),
    'feature_is_friday' : int(pred_date.weekday() == 4),
}
```

**수정된 파일:**
- 🔧 `04_forecast_future.ipynb` — 매크로 로드 셀 신설, `new_row` 구성 확장

---

#### 3. 04단계 모델 로드 방식 수정

**수정 내용:**

```python
# Before: pickle.load() 직접 호출 → dict 반환, AttributeError 발생
# After:  클래스메서드 사용
if active_model == 'lightgbm':
    model = LightGBMModel.load(str(model_path))
elif active_model == 'randomforest':
    model = RandomForestMultiModel.load(str(model_path))
elif active_model == 'ensemble':
    model = EnsembleModel.load(str(model_path))
```

**수정된 파일:**
- 🔧 `04_forecast_future.ipynb` — 모델 로드 셀 수정

---

#### 4. 97단계 신설 — 매크로 지표 미래값 추정

**신설 파일:**
- 📄 `97_forecast_macro.ipynb`
- 📁 `data/99_meta/macro_regime_forecast.parquet`

```python
# data/99_meta/macro_regime_forecast.parquet
columns = [
    'date',          # 미래 거래일
    'kospi',         # Damped Holt 추정값 (φ=0.90)
    'usd_krw',       # Damped Holt 추정값 (φ=0.85)
    'vix',           # Damped Holt 추정값 (φ=0.85)
    'us_return_1d',  # SES 추정값 (zero 수렴)
    'market_regime', # kospi 추정값 기반 재계산
]
```

**수정된 파일:**
- ✨ `97_forecast_macro.ipynb` — 신규 생성

---

### 호환성 (v3.7.0 → v3.7.1)

**호환 (재실행 불필요):** 01~03단계, 05단계 변경 없음.
**04단계만 재실행 필요** (피처 스키마 동기화 효과 적용).
97단계 선행 실행 후 04단계 실행 권장 (매크로 미래값 반영).

---

## [3.7.0] - 2026-02-22

### 🟢 MINOR Changes - log_close 롤백 + Embargo Gap

변경 범위: **03단계 (Training)**

---

#### 1. log_return Deprecated → log_close 롤백

`target_type="log_return"` 사용 시 `DeprecationWarning` 발생.
기본값: `"log_close"` 복원.

타겟 컬럼명: `target_log_return_h{n}` → `target_log_close_h{n}` (복원)

---

#### 2. Embargo Gap 도입

```
훈련 샘플 끝을 G일 앞당김 (G = max(horizons), 자동 계산)
검증/테스트 윈도우 크기는 변화 없음

[검증 폴드] 실제 훈련: [0, E-G]    embargo: [E-G, E]    검증: [E, E+V]
[테스트 폴드] 실제 훈련: [V, E+V-G] embargo: [E+V-G, E+V] 테스트: [E+V, E+V+T]
```

**수정된 파일:**
- 🔧 `src/modeling/trainer.py`
- 🔧 `config/config.yaml` — `target_type: "log_close"` 복원

---

### 호환성 (v3.6.0 → v3.7.0)

**비호환:** 03단계 전체 재실행 필수 (타겟 컬럼명 변경).
**호환:** 01, 02단계, 04단계 컬럼명 유지, 05단계 입력 스키마 변경 없음.

---

## [3.6.0] - 2026-02-21

### 🟢 MINOR Changes - Scale-Invariant Features & IC Evaluation

변경 범위: **02단계 (Feature), 03단계 (Training), 05단계 (Universe), 98단계 (신설)**

---

#### 1. 피처 엔지니어링 개선 — Scale-Invariance

| 이전 | 이후 |
|---|---|
| `feature_ma_5`, `feature_ma_60` | `feature_ma_5_disparity`, `feature_ma_60_disparity` |
| `feature_bb_upper`, `feature_bb_lower` | `feature_bb_pct_b`, `feature_bb_width` |
| `liquidity_score` | `feature_log_liquidity` |

#### 2. 매크로/레짐 피처 통합

신규 피처: `feature_kospi`, `feature_usd_krw`, `feature_vix`, `feature_us_return_1d`, `feature_market_regime`
출처: `data/99_meta/macro_regime.parquet`

#### 3. IC/ICIR 평가 지표 도입

앙상블 가중치 최적화 목표: MSE → `-IC(Spearman)` 최소화

#### 4. 98단계 신설

- 📄 `98_save_macro_data.ipynb`
- 📁 `data/99_meta/macro_regime.parquet`

---

### 호환성 (v3.5.0 → v3.6.0)

**비호환:** 02, 03단계 전체 재실행 필수. 98단계 선행 실행 필요.
**호환:** 01단계, 04단계 예측 파일 구조, 05단계 입력 스키마 변경 없음.

---

## [3.5.0] - 2026-02-20

### 🟢 MINOR Changes - Training Pipeline Improvement

변경 범위: **03단계 (Training)**

2-Fold Walk-Forward 구조 도입. `val_predictions.parquet` + `test_predictions.parquet` 분리.
앙상블 가중치 최적화를 테스트셋이 아닌 검증셋 기반으로 수행하여 데이터 누수 제거.

**수정된 파일:**
- 🔧 `src/modeling/trainer.py`, `03_train_predict.ipynb`, `03b_train_ensemble.ipynb`, `config/config.yaml`

---

### 호환성 (v3.4.0 → v3.5.0)

**비호환:** 03단계 모델/예측 파일 전체 재실행 필요.
**호환:** 01, 02, 04, 05단계 변경 없음.

---

## [3.4.0] - 2026-02-17

### 🟢 MINOR Changes - Model Diversification

RandomForest 멀티아웃풋 모델 추가. 앙상블 학습 단계(`03b_train_ensemble.ipynb`) 도입.
모델별 폴더 계층 추가: `03_training/{date}/{model_name}/`.

**신규 파일:** `src/models/randomforest_model.py`, `src/models/ensemble_model.py`, `03b_train_ensemble.ipynb`

---

## [3.3.0] - 2026-02-09

### 🟢 MINOR Changes - Infrastructure Modernization

H1: 단계별 독립 폴더 구조 (`03_training/`, `04_forecasts/`, `05_universe/`).
H2: `ProjectPaths` 클래스 도입 — 경로 관리 중앙화.
H3: `select_universe.py` Facade Pattern 적용.

---

## [3.2.1] - 2026-02-09

### 🔵 PATCH Changes

Multi-Horizon Walk-Forward 버그 수정. Chunk 오염 방지 (volume → 최근 20일 평균).

---

## [3.2.0] - 2026-02-07

### 🟢 MINOR Changes

04단계 (Recursive Extension 미래 예측), 05단계 (유니버스 선정) 추가.

---

## [3.1.1] - 2026-01-21 / [3.1.0] - 2026-01-21

Target 생성 위치 재변경 (03 → 02). Multi-Horizon Direct Forecasting (h1~h5) 도입.

---

## [3.0.0] - 2026-01-18

Target 생성 위치 변경 (01_raw → 02_processed).

---

## [2.0.0] - 2024-12-28

Feature Prefix 통일: `ma_5` → `feature_ma_5`.

---

## [1.0.0] - 2024-12-01

Initial Release. Step 1~3 (수집 → Feature → LightGBM 학습).

---

### 버전별 변경 이력

| Version | Date | Type | 주요 변경 사항 |
|---------|------|------|----------------|
| **3.8.0** | 2026-02-28 | 🟢 MINOR | MLP 모델 도입 + 앙상블 조합 동적 지정 + 04단계 건너뜀 처리 |
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
