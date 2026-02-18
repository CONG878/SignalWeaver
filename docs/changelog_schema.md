# Schema Changelog (Updated v3.4.0)

All notable changes to the data schema will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.4.0] - 2026-02-17

### 🟢 MINOR Changes - Model Diversification

#### 1. 모델 다양화 (RandomForest + Ensemble)

**변경 사항**:
- RandomForest 멀티아웃풋 모델 추가
- 앙상블 학습 단계(03b_train_ensemble.ipynb) 도입
- 모델별 폴더 계층 추가 (03_training/{date}/{model_name}/)
- 설정 옵션: active_model 추가

**신규 추가 파일**:
```
✅ src/models/randomforest_model.py
   - RandomForestMultiModel 클래스
   - Scikit-learn MultiOutputRegressor 래퍼

✅ src/models/ensemble_model.py
   - EnsembleModel 클래스
   - 여러 모델의 가중치 블렌딩

✅ 03b_train_ensemble.ipynb
   - OOF 데이터 기반 최적 가중치 탐색
   - Scipy.optimize.minimize 사용
```

**폴더 구조 변경**:

Before (v3.3.0):
```
data/03_training/{YYYYMMDD}/
├── *.pkl
├── registry.json
└── predictions.parquet
```

After (v3.4.0):
```
data/03_training/{YYYYMMDD}/
├── lightgbm/
│   ├── *.pkl
│   ├── registry.json
│   └── predictions.parquet
├── randomforest/
│   ├── *.pkl
│   ├── registry.json
│   └── predictions.parquet
└── ensemble/
    ├── *.pkl
    ├── registry.json
    └── predictions.parquet
```

**설정 변경** (config.yaml):

```yaml
# NEW: RandomForest 파라미터
randomforest_params:
  n_estimators: 40
  max_depth: 8
  min_samples_split: 10
  min_samples_leaf: 5
  n_jobs: -1
  random_state: 42
  max_samples: 0.6

# NEW: 모델 선택 옵션
active_model: "ensemble"  # 'lightgbm' | 'randomforest' | 'ensemble'
```

**수정된 파일**:
- 🔧 `src/models/base.py` - ModelBase 호환성 확장 (3개 모델)
- 🔧 `src/models/artifact.py` - 모델별 폴더 지원
- 🔧 `src/utils/config.py` - ProjectPaths 메서드 추가
- 🔧 `03_train_predict.ipynb` - 모델 선택 로직 추가
- 🔧 `04_forecast_future.ipynb` - 모델 로딩 확장
- 🔧 `config/config.yaml` - 설정 옵션 추가

**이점**:
- ✅ 여러 모델 알고리즘 비교 가능
- ✅ 앙상블로 예측 정확도 향상
- ✅ 모델별 성능 추적 가능
- ✅ 유연한 모델 선택 (config에서 한 줄)

**노트북 실행 순서** (Updated):
1. 01_collect_data.ipynb
2. 02_build_dataset.ipynb
3. 03_train_predict.ipynb
   - active_model = "lightgbm" → 실행
   - active_model = "randomforest" → 재실행 (양쪽 모두 필요)
4. **03b_train_ensemble.ipynb** (NEW - 선택사항)
   - 앙상블 사용 시에만 실행
5. 04_forecast_future.ipynb
6. 05_universe_selection.ipynb

**마이그레이션**:
- 기존 LightGBM 모델: 자동 호환 (폴더 구조만 자동 생성)
- 새로운 모델 추가: config.yaml에서 active_model 변경 후 03_train_predict 재실행

**호환성**:
- ✅ 후방 호환 (backward compatible): v3.3.0 모델 그대로 사용 가능
- ❌ 전방 호환 불가: v3.4.0 모델은 v3.3.0 코드에서 로드 불가

---

## [3.3.0] - 2026-02-09

### 🟢 MINOR Changes - Infrastructure Modernization

#### 1. H1 - 데이터 폴더 구조 개선

**변경 사항**:
- `data/03_results/` 구조 분해 → 단계별 독립 폴더
- 새로운 폴더 구조로 계층 명확화

**Before (v3.2.x)**:
```
data/03_results/{YYYYMMDD}/
├── predictions.parquet        ← Step 3
├── *.pkl
├── forecasts/
│   └── future_forecasts.parquet ← Step 4
└── universe/
    └── investment_report.xlsx  ← Step 5
```

**After (v3.3.0)**:
```
data/
├── 03_training/{YYYYMMDD}/      ← 학습 검증 예측
│   ├── *.pkl
│   ├── registry.json
│   ├── predictions.parquet
│   └── csv/
│
├── 04_forecasts/{YYYYMMDD}/     ← 미래 예측
│   ├── future_forecasts.parquet
│   └── csv/
│
└── 05_universe/{YYYYMMDD}/      ← 최종 선정
    ├── universe_full.parquet
    ├── universe_candidates.parquet
    ├── investment_report.csv
    ├── investment_report.xlsx
    └── filter_statistics.json
```

**이점**:
- ✅ Step별로 폴더 계층이 일관성 있음
- ✅ 각 단계의 산출물이 명확히 분리됨
- ✅ 파이프라인의 선형 구조 반영

**마이그레이션**:
- 기존 파일은 그대로 유효 (포맷 변경 없음)
- 폴더 이동만 필요

#### 2. H2 - ProjectPaths 클래스 도입 (경로 중앙화)

**변경 사항**:
- `src/utils/config.py`에 `ProjectPaths` 클래스 추가
- 모든 노트북의 경로 관리 통일

**Before (v3.2.x)**:
```python
# 모든 노트북에서 반복되는 코드
from pathlib import Path
ref_date = cfg['project']['reference_date']
raw_dir = Path(cfg['paths']['raw_dir']) / ref_date
processed_dir = Path(cfg['paths']['processed_dir']) / ref_date
result_dir = Path("data/03_results") / ref_date  # 하드코딩!

raw_parquet = raw_dir / f"krx_prices_{ref_date}.parquet"
dataset = processed_dir / "dataset.parquet"
predictions = result_dir / "predictions.parquet"
```

**After (v3.3.0)**:
```python
# 통일된 인터페이스
from src.utils.config import load_config, ProjectPaths

cfg = load_config()
paths = ProjectPaths.from_config(cfg)

raw_parquet = paths.get_raw_parquet()
dataset = paths.get_dataset_parquet()
predictions = paths.get_predictions_parquet()  # 03_training 자동 매핑
forecasts = paths.get_forecasts_parquet()      # 04_forecasts 자동 매핑
universe = paths.get_universe_candidates()     # 05_universe 자동 매핑
```

**제공 메서드**:
- `get_raw_parquet()`: 01단계
- `get_dataset_parquet()`: 02단계
- `get_predictions_parquet()`: 03단계 (training)
- `get_forecasts_parquet()`: 04단계 (forecasts)
- `get_universe_candidates()`: 05단계 (universe)
- `ensure_dirs()`: 모든 출력 폴더 자동 생성

**이점**:
- ✅ 폴더 구조 변경 시 `config.py`만 수정
- ✅ 모든 노트북에서 일관된 경로 사용
- ✅ 타입 안전성 (IDE 자동완성)
- ✅ ~30줄의 경로 조립 코드 → 2줄로 단축

**영향받는 파일**:
- `01_collect_data.ipynb`: ProjectPaths 사용
- `02_build_dataset.ipynb`: ProjectPaths 사용
- `03_train_predict.ipynb`: ProjectPaths 사용
- `04_forecast_future.ipynb`: ProjectPaths 사용
- `05_universe_selection.ipynb`: ProjectPaths 사용

#### 3. H3 - 모듈 정리 (Facade Pattern)

**변경 사항**:
- `src/universe/select_universe.py`에 Facade Pattern 적용
- Step 5 노트북의 복잡한 로직 캡슐화

**Before (v3.2.x)**:
```python
# 05_universe_selection.ipynb에서 200줄의 직접 로직
for ticker in tickers:
    accuracy = calculate_accuracy(...)
    profitability = calculate_profitability(...)
    risk = calculate_risk(...)
    # ... 복잡한 필터링 로직
```

**After (v3.3.0)**:
```python
# 캡슐화된 인터페이스
from src.universe.select_universe import select_universe_candidates

candidates = select_universe_candidates(
    predictions_df=predictions,
    forecasts_df=forecasts,
    config=cfg
)
```

**이점**:
- ✅ 복잡한 비즈니스 로직 투명화
- ✅ 노트북 가독성 향상
- ✅ 로직 재사용성 증대

---

## [3.2.1] - 2026-02-09

### 🔵 PATCH Changes - Bug Fixes

#### 1. Multi-Horizon Walk-Forward 버그 수정

**문제**:
- 각 Horizon별로 shift + dropna 후 길이가 불일치
- 예: h1에서 100개, h2에서 99개 → 예측 오차 누적

**해결**:
```python
# Before (v3.2.0)
for h in horizons:
    df_h = df.loc[:, [f'feature_*', f'target_*_h{h}']].dropna()
    # → df_h의 길이가 h마다 다름

# After (v3.2.1)
# 모든 Horizon의 교집합 인덱스만 사용
valid_idx = df.dropna().index
df_train = df.loc[valid_idx]
```

**영향**:
- ✅ 정확도 평가 신뢰성 향상
- ✅ 예측 결과 일관성 보장

#### 2. Recursive Extension Chunk 오염 방지

**문제**:
- Chunk 1+ 예측 시 실제 과거 volume 참조
- 재귀 진행에 따른 오차 누적

**해결**:
```python
# Before (v3.2.0)
forecast_features[:, 'volume_col'] = actual_volume  # ❌ 실제값 사용

# After (v3.2.1)
forecast_features[:, 'volume_col'] = volume_ma_20   # ✅ 평균값 사용
```

**영향**:
- ✅ Chunk 진행에 따른 오차 감소
- ✅ 장기 예측(t+30~t+60) 신뢰성 향상

---

## [3.2.0] - 2026-02-07

### 🟢 MINOR Changes - Multi-Stage Pipeline

#### 1. Step 4 추가 (Recursive Extension)

**변경 사항**:
- 04_forecast_future.ipynb 추가
- 미래(t+1 ~ t+60) 주가 예측

**프로세스**:
```
Chunk 0: t-5~t-1 Feature → t+0~t+4 예측
Chunk 1: Chunk0 + Feature → t+5~t+9 예측
...
Chunk 12: Chunk11 + Feature → t+55~t+59 예측
```

**데이터 구조**:
```
04_forecasts/{date}/future_forecasts.parquet
└─ (n_tickers × 60_days, 5 columns)
   - date, ticker, forecast_date, pred_target_log_close, ...
```

#### 2. Step 5 추가 (Universe Selection)

**변경 사항**:
- 05_universe_selection.ipynb 추가
- 3대 평가 지표 기반 투자 후보 선정

**평가 지표**:
- **정확도**: 과거 예측 오차 (RMSE, MAE)
- **수익성**: 예측값 기반 수익률
- **위험**: 예측값 변동성

**필터링**:
- Hard Filter (강제 제외): 유동성 부족, 거래 중단 등
- Score-based Ranking: 종합 점수로 순위 결정

**데이터 구조**:
```
05_universe/{date}/
├── universe_full.parquet          # 전체 평가 결과
├── universe_candidates.parquet    # Top-K 후보
├── investment_report.csv          # 상세 리포트
└── filter_statistics.json         # 필터링 통계
```

---

## [3.1.1] - 2026-01-21

### 🔵 PATCH Changes

#### Target 생성 위치 재변경 (03 → 02)

**Before (v3.1.0)**:
- Target을 03_train_predict.ipynb에서 생성

**After (v3.1.1)**:
- Target을 02_build_dataset.ipynb에서 생성
- 이유: 모든 모델이 동일한 Target을 공유 (Feature와 Target의 일관성)

---

## [3.1.0] - 2026-01-21

### 🟢 MINOR Changes

#### Multi-Horizon Direct Forecasting

**변경**:
- 단일 예측(h=1)에서 다중 예측(h1~h5)으로 확장
- Target-Centric Alignment: $X_{t-h} \rightarrow y_t$

**데이터 정렬**:
```python
# Feature를 h일 과거로 Shift
X_h = X.shift(h)          # t-h 기준
y = y_original            # t 기준

# 결과: X_h와 y의 index가 일치 → 직관적
```

---

## [3.0.0] - 2026-01-18

### 🔴 MAJOR Changes

#### Target 위치 변경

**Before (v2.x)**:
- Target을 01_raw에서 정의

**After (v3.0.0)**:
- Target을 02_processed에서 생성
- 이유: Feature와의 시간 정렬 명확화

---

## [2.0.0] - 2024-12-28

### 🔴 MAJOR Changes

#### Feature Prefix 통일

**변경**:
- 모든 Feature에 `feature_` prefix 추가
- 예: `ma_5` → `feature_ma_5`

---

## [1.0.0] - 2024-12-01

### Initial Release

**초기 스키마**:
- Step 1: 데이터 수집 (Raw OHLCV)
- Step 2: Feature 엔지니어링
- Step 3: 모델 학습 (LightGBM)

---

**Last Updated**: 2026-02-17  
**Maintained by**: SignalWeaver Team
