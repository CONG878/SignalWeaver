# Schema Changelog (Updated v3.3.0)

All notable changes to the data schema will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

#### 3. H3 - 모듈 구조 정리 (Facade Pattern)

**변경 사항**:
- `src/universe/select_universe.py` Facade Pattern 적용
- Step 5의 복잡한 200줄 로직을 `select_investment_universe()` 함수로 캡슐화

**Before (v3.2.x, 노트북에 혼재)**:
```python
# 정확도 평가 루프 (50줄)
df_accuracy = []
for ticker in df_past['ticker'].unique():
    # rmse, directional_accuracy 계산...

# 수익성 평가 루프 (50줄)
df_return = []
for ticker in df_future['ticker'].unique():
    # find_best_trade...

# 위험도 평가 루프 (50줄)
df_risk = []
for ticker in df_future['ticker'].unique():
    # calculate_risk_metrics...

# 통합 및 필터링 (50줄)
df_universe = df_accuracy.merge(...).merge(...)
df_filtered, stats = apply_hard_filters(...)
```

**After (v3.3.0, 함수화)**:
```python
# 모든 로직 캡슐화
results = select_investment_universe(
    df_past_predictions,
    df_future_forecasts,
    df_meta,
    model_date='2026-01-20',
    top_k=100
)

df_candidates = results['candidates']
df_full = results['full']
filter_stats = results['filter_stats']
```

**구현된 함수들**:
- `evaluate_model_accuracy()`: 과거 예측 정확도 평가
- `evaluate_expected_returns()`: 미래 기대 수익률 계산
- `evaluate_risk_metrics()`: 종목 내재 위험 평가
- `select_investment_universe()`: 전체 흐름 Facade

**이점**:
- ✅ 복잡한 로직을 간단한 인터페이스로 제공
- ✅ Step 5 노트북 200줄 → 50줄로 단축
- ✅ 비즈니스 로직과 프리젠테이션 분리
- ✅ 테스트 가능한 구조

**영향받는 파일**:
- `src/universe/select_universe.py`: 전면 재구성 (v3.2.0 함수 활용)
- `src/universe/filters.py`: 기존 필터 함수 유지

### 호환성

| 측면 | 호환성 | 비고 |
|------|--------|------|
| 데이터 포맷 | ✅ 완전 | .pkl, .parquet 포맷 변경 없음 |
| 모델 | ✅ 완전 | 기존 .pkl 파일 로드 가능 |
| 폴더 구조 | ⚠️ 부분 | 폴더 이동 필요 |
| 코드 | ⚠️ 부분 | ProjectPaths 사용으로 수정 필요 |

---

## [3.2.1] - 2026-02-09

### 🔴 Critical Bug Fixes

#### 1. Multi-Horizon Walk-Forward 데이터 누수 버그 수정

**파일**: `src/modeling/trainer.py` - `_evaluate()`, `_predict_with_metadata()`

**문제**:
```python
# 버그 코드
for h in horizons:
    target_id = f"{self.target_col_name}_h{h}"
    temp_df = base_df.copy()
    for f in self.feature_cols:
        temp_df[f] = temp_df.groupby('ticker')[f].shift(h)
    
    # ❌ 문제: h마다 다른 NaN 제거로 인덱스 불일치
    eval_df = temp_df[temp_df[self.date_col].isin(eval_dates)].dropna()
    
    # 길이가 다르면 연산 오류!
    y_true = eval_df[self.target_col_name].values
    y_pred = self.model.predict(...).values  # 길이 불일치
```

**영향**:
- ❌ h1, h2, ... h5가 서로 다른 행 수를 가짐
- ❌ RMSE 평균 계산 시 IndexError 또는 부정확한 결과
- ❌ Walk-Forward Fold별 성능 평가 신뢰도 하락

**해결**:
```python
# 수정 코드: 모든 horizon의 교집합 인덱스 사용
valid_indices = None

for h in horizons:
    temp_df = base_df.copy()
    for f in self.feature_cols:
        temp_df[f] = temp_df.groupby('ticker')[f].shift(h)
    
    eval_slice = temp_df[temp_df[self.date_col].isin(eval_dates)]
    h_valid_indices = eval_slice.dropna().index
    
    # ✅ 교집합 계산
    if valid_indices is None:
        valid_indices = h_valid_indices
    else:
        valid_indices = valid_indices.intersection(h_valid_indices)

# ✅ 모든 horizon에서 같은 인덱스 사용
for h in horizons:
    temp_df = base_df.copy()
    for f in self.feature_cols:
        temp_df[f] = temp_df.groupby('ticker')[f].shift(h)
    
    eval_df = temp_df.loc[valid_indices]  # 공통 인덱스
    y_true = eval_df[self.target_col_name].values
    y_pred = self.model.predict(...).values  # 길이 일치!
```

**영향**:
- ✅ 정확도 평가 정확성 향상
- ✅ Walk-Forward Fold 성능 신뢰도 증가
- ✅ 모델 검증 결과 재현성 보장

#### 2. Recursive Extension 데이터 오염 방지

**파일**: `04_forecast_future.ipynb` - Recursive Extension 루프

**문제**:
```python
# 버그 코드
for chunk_idx in range(NUM_CHUNKS):
    df_ticker = calculate_features_for_ticker(df_ticker, cfg)
    
    for h_idx, target_name in enumerate(model.target_columns, 1):
        pred_log_close = model.predict(..., target_name=target_name).iloc[0]
        
        # ❌ 문제: 실제 과거 데이터를 그대로 참조
        new_row = {
            'close': np.exp(pred_log_close),
            'volume': latest_row.get('volume', 0),  # 실제 과거 거래량!
            'open': np.exp(pred_log_close),
            'high': np.exp(pred_log_close),
            'low': np.exp(pred_log_close),
        }
        df_ticker = pd.concat([df_ticker, pd.DataFrame([new_row])], ...)
```

**영향**:
- ❌ Chunk 1 이후: 예측값 + 실제 과거 volume 혼합
- ❌ Feature 재계산 시 부정확한 `feature_volume_ratio` 생성
- ❌ Chunk가 진행될수록 예측 정확도 급격히 하락
- ❌ Chunk 2+는 신뢰도 매우 낮음

**해결** (v1.1 패치):
```python
# 수정 코드
CHUNK_SIZE = 5
for chunk_idx in range(NUM_CHUNKS):
    # ✅ 최근 거래량 평균 사전 계산 (데이터 오염 방지)
    recent_volume_mean = df_ticker['volume'].tail(20).mean()
    
    df_ticker = calculate_features_for_ticker(df_ticker, cfg)
    
    for h_idx, target_name in enumerate(model.target_columns, 1):
        pred_log_close = model.predict(..., target_name=target_name).iloc[0]
        
        # ✅ 평균값 사용 (예측 순수성 유지)
        new_row = {
            'close': np.exp(pred_log_close),
            'volume': recent_volume_mean,  # 최근 평균 사용!
            'open': np.exp(pred_log_close),
            'high': np.exp(pred_log_close) * 1.02,
            'low': np.exp(pred_log_close) * 0.98,
        }
        df_ticker = pd.concat([df_ticker, pd.DataFrame([new_row])], ...)
```

**영향**:
- ✅ Chunk 1+ 예측 품질 향상
- ✅ 거래량 기반 Feature 정확도 개선
- ✅ 예측값 체인의 순수성 유지
- ✅ Chunk 진행에 따른 오차 누적 개선

---

## [3.2.0] - 2026-02-07

### 🟢 MINOR Changes

#### 1. 04단계 추가: 미래 주가 예측 (Recursive Extension)

**변경 사항**:
- 학습된 모델을 이용한 미래 주가 예측 파이프라인 추가
- Recursive Extension: 5일 Chunk 단위 반복 예측으로 60일 이상 확장 가능
- 예측 오차 누적 고려 (뒤로 갈수록 신뢰도 하락)

**새로운 파일 구조**:
```
data/03_results/{YYYYMMDD}/forecasts/
  ├── future_forecasts.parquet         # 통합 미래 예측
  └── csv/{종목명}_forecast.csv        # 개별 CSV (옵션)
```

**새로운 컬럼**:
| 컬럼 | 설명 |
|------|------|
| `date` | 예측 대상 날짜 |
| `ticker` | 종목 코드 |
| `horizon` | 예측 시차 (1~5) |
| `chunk_idx` | Recursive 단계 (0, 1, 2, ...) |
| `pred_log_close` | 예측 로그 종가 |
| `pred_close` | 예측 종가 (원화) |

**영향**:
- ✅ 모델 구조 변경 없음 (기존 03단계 모델 재사용)
- ✅ 02~03단계 출력 변경 없음 (상위 호환)
- ⚠️ Chunk 0: 신뢰도 높음, Chunk 2+: 신뢰도 낮음 (오차 누적)
- ❌ v3.2.1: Chunk 오염 버그 → 수정됨

#### 2. 05단계 추가: 유니버스 선정 (Universe Selection)

**변경 사항**:
- 3대 평가 지표 기반 투자 종목 선정 파이프라인
- Hard Constraints 필터링

**새로운 파일 구조**:
```
data/03_results/{YYYYMMDD}/universe/
  ├── universe_full.parquet            # 전체 평가 종목
  ├── universe_candidates.parquet      # Top-K 후보
  ├── investment_report.csv            # CSV 리포트
  ├── investment_report.xlsx           # Excel 멀티시트
  └── filter_statistics.json           # 필터링 통계
```

**새로운 지표 (정확도 평가)**:
```python
# 과거 예측 오차 기반 (model_train_date 기준)
- rmse: Root Mean Square Error
- mae: Mean Absolute Error
- directional_accuracy: 방향성 정확도 (0~1)
- confidence_rmse: RMSE 역수 기반 신뢰도
- accuracy_rank: 정확도 순위
```

**새로운 지표 (수익성 평가)**:
```python
# 미래 예측 수익률 기반 (forecast_date 기준)
- daily_log_return: 시간당 로그 수익률 (복리)
- total_log_return: 총 로그 수익률
- total_return_pct: 총 수익률 (%)
- hold_days: 최적 보유 기간
- buy_date, sell_date: 최적 매매 시점
- buy_price, sell_price: 예상 매매가
- return_rank: 수익률 순위
```

**새로운 지표 (위험도 평가)** ⭐:
```python
# 예측값 시계열 기반 내재 위험 (미래 위험)
- volatility: 변동성 (로그 수익률 표준편차)
- downside_risk: 하방 위험 (음수 수익률만)
- var_95: VaR (5% 분위수)
- cvar_95: CVaR (최악 5% 평균)
- max_drawdown: 최대 낙폭
- skewness: 비대칭도 (음수=하락쏠림)
- kurtosis: 초과 첨도 (Fat Tail 지표)
- risk_composite_raw: 복합 리스크 (원점수)
- risk_score_normalized: 복합 리스크 (0~1 정규화)
- risk_rank: 위험 순위
```

**Hard Constraints** (필수 조건):
| 조건 | 제거 기준 |
|------|---------|
| 거래정지/상폐 | `is_suspended=1` 또는 `is_delisted=1` |
| 저유동성 | 20일 평균 거래대금 < 5천만 원 |
| 고위험 | `risk_composite_raw` > 0.8 |
| 저정확도 | `accuracy_rank` > 1000 |

**이중 날짜 기준** (새로운 개념):
```
model_train_date (2026-01-20)
  ↓ [정확도 평가: 과거 예측 오차]
  
forecast_date (2026-02-07+)
  ↓ [수익성 평가: 미래 예측 수익률]
```

**영향**:
- ✅ 02~04단계 출력 변경 없음
- ✅ 기존 모델 재사용 (새로운 학습 불필요)
- ⚠️ 05단계는 new output 전용 (기존 파이프라인 영향 없음)

#### 3. 위험도 평가 체계 신규 도입

**5대 표준 지표** (src/utils/risk.py):
1. **Volatility**: 기본 변동성 (표준편차)
2. **Downside Risk**: 하방 위험만 (손실 변동성)
3. **VaR/CVaR**: 극단 리스크 (극단값 기반)
4. **MDD**: 최대 낙폭 (심리적 영향)
5. **Skew/Kurt**: 분포 형태 (비대칭성, Fat Tail)

**특징**:
- 모델 예측 오차와 독립 (내재 위험)
- 예측값 시계열 기반 (미래 위험도)
- 복합 점수로 통합 (가중 평균)

**위험 vs 정확도 차이**:
```
정확도 (Accuracy):
  - "모델이 얼마나 잘 맞추는가?"
  - 과거 데이터로 측정
  - Epistemic 불확정성
  
위험도 (Risk):
  - "이 종목이 얼마나 불안정한가?"
  - 미래 데이터로 예상
  - Aleatoric 불확정성
```

#### 4. 필터링 유틸리티 추가

**새로운 모듈** (src/universe/filters.py):
- `filter_tradability()`: 거래정지/상장폐지 제거
- `detect_manipulation_risk()`: 작전주/테마주 탐지
- `filter_manipulation()`: 작전주 제거
- `filter_penny_stocks()`: 저가주 제거
- `filter_liquidity()`: 저유동성 제거
- `apply_hard_filters()`: 통합 필터 적용

**Phase 2 설계서 반영**:
> "학습은 전부, 투자 후보는 엄선해서"
> "작전주 필터는 2단계 구조: 사후 탐지 + 사전 경고"

---

## [3.1.1] - 2026-01-21

### 🔵 PATCH Changes

#### Target 생성 위치 재변경 (03 → 02)

**변경 사항**:
- **v3.0.0**: 03단계(학습 직전)에서 Target 생성
- **v3.1.0**: 03단계 유지
- **v3.1.1**: **02단계(전처리)로 복귀** ← 현재

**복귀 이유**:
1. **재현성 보장**: 02단계 산출물만으로 학습 환경 완전 재현 가능
2. **책임 분리 명확화**: 
   - 02단계: 데이터 준비 완료 (Feature + Target)
   - 03단계: 순수 모델 학습 및 예측
3. **파이프라인 안정성**: 실험 변경 시 02단계만 재실행하면 일관된 데이터셋 확보
4. **운영 편의성**: Target이 없는 데이터셋은 불완전하다는 직관에 부합

**구현**:
```python
# 02_build_dataset.ipynb (최종 단계)
df_final['target_log_close'] = np.where(
    df_final['close'] > 1, 
    np.log(df_final['close']), 
    0
)
```

**영향**:
- ✅ **02단계 출력**: `target_log_close` 컬럼 포함
- ✅ **03단계 입력**: Target이 이미 준비된 상태로 시작
- ⚠️ **하위 호환**: v3.0.x 사용자는 02단계 재실행 권장

---

## [3.1.0] - 2026-01-21

### 🟢 MINOR Changes

#### 1. Multi-Horizon 예측 구조 도입

**변경 사항**:
- **v3.0.x**: 단일 시점(t+1) 예측
- **v3.1.0**: **5일치(h1~h5) 동시 예측**

**동기**:
- 실전 운용에서는 원거리 예측(먼 미래) 필요
- 단일 모델로 여러 시차를 학습하여 효율성 향상
- Recursive Extension의 기반 제공 (5일 Chunk → 60일 확장)

**구현 개념**:
```
t일의 가격을 맞추기 위해:
- h=1: t-1일 피처 사용
- h=2: t-2일 피처 사용
- h=3: t-3일 피처 사용
- h=4: t-4일 피처 사용
- h=5: t-5일 피처 사용

→ 5개의 독립된 모델이 동일한 정답(t일 가격) 예측
```

**Target 컬럼** (식별자, 데이터셋에는 없음):
```python
model.target_columns = [
    'target_log_close_h1',
    'target_log_close_h2',
    'target_log_close_h3',
    'target_log_close_h4',
    'target_log_close_h5'
]
```

**예측 결과 컬럼**:
| 컬럼명 | 설명 |
|--------|------|
| `pred_target_log_close_h1` | h=1 예측값 (1일 전 정보로 예측) |
| `pred_target_log_close_h2` | h=2 예측값 (2일 전 정보로 예측) |
| ... | ... |
| `true_target_log_close_h1` | 정답 (참조용, 모든 h에 대해 동일) |

**영향**:
- ⚠️ **모델 클래스**: `LightGBMModel`이 Multi-output 지원
- ⚠️ **Trainer 로직**: `WalkForwardTrainer`에 Target-Centric 정렬 구현
- ✅ **하위 호환**: 기존 단일 예측도 `horizons=[1]`로 지원

#### 2. Target-Centric Alignment 패턴

**변경 사항**:
- 예측 결과의 `date` 컬럼이 **실제 예측 대상일**과 일치하도록 데이터 정렬

**Before (Feature-Centric)**:
```
date       | feature_ma_5 | target
2026-01-15 | 100          | 102  (다음날 가격)
2026-01-16 | 101          | 103
```
→ "이 날짜의 피처로 다음 날을 예측"

**After (Target-Centric)**:
```
date       | feature_ma_5 (shifted) | target
2026-01-16 | 100 (전날값)           | 102  (이 날의 정답)
2026-01-17 | 101 (전날값)           | 103  (이 날의 정답)
```
→ "이 날짜의 가격을 예측 대상으로 삼음"

**장점**:
- ✅ 예측 결과 해석 직관성 향상
- ✅ 백테스트 로직 단순화
- ✅ 운영 환경과 동일한 시간 개념

**영향**:
- ⚠️ **Trainer 로직**: `src/modeling/trainer.py` 전면 개편
- ✅ **데이터셋 구조**: 변경 없음
- ✅ **예측 결과**: `date` 컬럼 의미 변경

#### 3. Shift-then-Slice 패턴

**변경 사항**:
- 기존: 슬라이싱 후 시프트 → 경계면 데이터 손실
- 개선: **시프트 후 슬라이싱** → 데이터 손실 방지

**Before (Slice-then-Shift)**:
```python
train_df = df[df['date'].isin(train_dates)]
train_df['feature'] = train_df['feature'].shift(1)  # 첫 행 NaN
```
→ 학습 구간의 첫 날짜 데이터 손실

**After (Shift-then-Slice)**:
```python
df_shifted = df.copy()
df_shifted['feature'] = df_shifted.groupby('ticker')['feature'].shift(1)
train_df = df_shifted[df_shifted['date'].isin(train_dates)].dropna()
```
→ 시프트된 값이 이미 준비된 상태로 슬라이싱

**장점**:
- ✅ Walk-Forward 각 Fold의 유효 데이터 최대화
- ✅ 경계 구간 예측 품질 향상

---

## [3.0.0] - 2026-01-18

### 🔴 Breaking Changes

#### 1. Target 생성 위치 변경 (v2.x에서)

**변경 사항**:
- **v2.x**: 02단계에서 `target_return`, `target_log_return` 생성
- **v3.0.0**: **03단계**에서 `target_log_close` 생성

**이유**:
- 예측 목표(horizon)가 실험마다 다를 수 있음
- 02단계는 "Feature 준비"만 담당하는 단일 책임 원칙
- 03단계에서 학습 직전에 Target 정의하여 유연성 확보

#### 2. Feature Shift 도입

**변경 사항**:
- t일 행에 t-1일의 Feature를 배치
- 의도: "어제 정보로 오늘 종가 예측"

**구현**:
```python
# 03단계에서 필수 적용
for col in feature_cols:
    df[col] = df.groupby('ticker')[col].shift(1)
df = df.dropna(subset=feature_cols)
```

#### 3. Ticker Feature 제외

**변경 사항**:
- **v2.x**: Ticker를 Categorical Feature로 사용 가능
- **v3.0.0**: **Ticker를 Feature로 사용 안 함**

**이유**:
- 신규 상장 종목 예측 불가 문제
- 차원 폭발 (2,900개 종목 → 2,900차원)
- 일반화 성능 저하

---

## [2.0.0] - 2024-12-28

### 🔴 Breaking Changes

#### Feature 명명 규칙 통일
- 모든 Feature 컬럼에 `feature_` prefix 추가

---

## [1.0.0] - 2024-12-01

### Initial Release
- 기본 OHLCV 컬럼
- 기술적 지표: `ma_5`, `rsi_14`, `macd` 등

---

## Version History Summary

| Version | Date | Type | Key Changes |
|---------|------|------|-------------|
| **3.3.0** | 2026-02-09 | 🟢 MINOR | H1(폴더구조) + H2(경로중앙화) + H3(모듈정리) |
| **3.2.1** | 2026-02-09 | 🔵 PATCH | Multi-Horizon 버그 + Chunk 오염 방지 |
| **3.2.0** | 2026-02-07 | 🟢 MINOR | 04단계(미래예측) + 05단계(유니버스) + 위험도 |
| **3.1.1** | 2026-01-21 | 🔵 PATCH | Target 위치 재변경 (03→02) |
| **3.1.0** | 2026-01-21 | 🟢 MINOR | Multi-horizon, Target-Centric Alignment |
| 3.0.0 | 2026-01-18 | 🔴 MAJOR | Target 위치 변경, Feature Shift, Ticker 제외 |
| 2.0.0 | 2024-12-28 | 🔴 MAJOR | Feature prefix 통일 |
| 1.0.0 | 2024-12-01 | - | Initial release |

---

## Breaking Changes Impact Matrix (v3.3.0)

| Change | 01 | 02 | 03 | 04 | 05 | 모델 | 하위호환 |
|--------|----|----|----|----|----|----|---------|
| **v3.3.0: 폴더구조** | ✅ | ✅ | ⚠️ | ⚠️ | ⚠️ | ✅ | 🟡 부분 |
| **v3.3.0: 경로중앙화** | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ✅ | 🟡 부분 |
| **v3.3.0: 모듈정리** | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ |
| **v3.2.1: 버그수정** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟢 높음 |

**범례**:
- ✅ 영향 없음
- ⚠️ 코드 수정 필요 (주로 경로 참조)
- ❌ 재학습 필요
- 🟢 높음: 자동 또는 최소 작업
- 🟡 부분: 단계별 마이그레이션 필요

---

## Breaking Changes Impact Matrix (v3.2.1)

| Change | 01 | 02 | 03 | 04 | 05 | 모델 | 하위호환 |
|--------|----|----|----|----|----|----|---------|
| **v3.2.1: 다중호라이즌버그** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟢 높음 |
| **v3.2.1: 청크오염방지** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟢 높음 |

**결론**: v3.2.1은 버그 수정만이므로 완전 하위 호환

---

**Last Updated**: 2026-02-09  
**Current Version**: 3.3.0 (with 3.2.1 critical patches)  
**Status**: ✅ Stable  
**Maintained by**: SignalWeaver Team