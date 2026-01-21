# Schema Changelog

All notable changes to the data schema will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.1.1] - 2026-01-21

### 🔵 PATCH Changes

#### Target 생성 위치 재변경 (03단계 → 02단계)

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
- ⚠️ **하위 호환**: v3.0.x 사용자는 02단계 재실행 권장 (마이그레이션 불필요)

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

→ 5개의 독립된 모델이 동일한 정답(t일 가격)을 예측
```

**Target 컬럼 (식별자)**:
```python
# 모델 내부적으로만 사용 (데이터셋에는 없음)
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

---

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
- ✅ 예측 결과 해석 직관성 향상 ("2026-01-17의 예측값")
- ✅ 백테스트 로직 단순화 (날짜 매칭 혼란 제거)
- ✅ 운영 환경과 동일한 시간 개념

**구현 (Trainer 내부)**:
```python
# Shift-then-Slice 패턴
for h in horizons:
    # 1. 전체 데이터를 먼저 시프트
    for col in feature_cols:
        temp_df[col] = temp_df.groupby('ticker')[col].shift(h)
    
    # 2. 시프트 후 날짜로 슬라이싱
    train_df = temp_df[temp_df['date'].isin(train_dates)].dropna()
    
    # 3. Target은 시프트하지 않음 (오늘의 정답)
    model.fit(train_df[feature_cols], train_df['target_log_close'])
```

**영향**:
- ⚠️ **Trainer 로직**: `src/modeling/trainer.py` 전면 개편
- ✅ **데이터셋 구조**: 변경 없음 (Trainer가 런타임에 처리)
- ✅ **예측 결과**: `date` 컬럼 의미 변경 (타깃 시점)

---

#### 3. Shift-then-Slice 패턴

**변경 사항**:
- 기존: 슬라이싱 후 시프트 → 경계면 데이터 손실
- 개선: **시프트 후 슬라이싱** → 데이터 손실 방지

**Before (Slice-then-Shift)**:
```python
train_df = df[df['date'].isin(train_dates)]  # 먼저 자름
train_df['feature'] = train_df['feature'].shift(1)  # 첫 행 NaN 발생
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

### 📚 Documentation

#### Multi-Horizon 예측 가이드

**설정 (config.yaml)**:
```yaml
training:
  horizons: [1, 2, 3, 4, 5]  # 5일치 동시 예측
  target_col_name: "target_log_close"
```

**사용 예시**:
```python
# 모델 초기화
model = LightGBMModel(
    model_version="v3.1",
    params=lgbm_params,
    feature_list=feature_cols
)

# Trainer 실행 (자동으로 Multi-horizon 학습)
trainer = WalkForwardTrainer(
    model=model,
    horizons=[1, 2, 3, 4, 5],
    target_col_name='target_log_close'
)

results = trainer.run(df, ...)

# 예측 결과
predictions = results['test_predictions']
# 컬럼: date, ticker, pred_target_log_close_h1, ..., pred_target_log_close_h5
```

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

**마이그레이션 가이드**:
```python
# v2.x 데이터셋 (02단계 출력)
df = pd.read_parquet("data/02_processed/dataset.parquet")

# Target 컬럼 제거 (있다면)
target_cols = [c for c in df.columns if c.startswith('target_')]
if target_cols:
    df = df.drop(columns=target_cols)

# v3.0.0 형식으로 저장
df.to_parquet("dataset_v3.parquet", index=False)
```

```python
# v3.0.0 사용법 (03단계)
df = pd.read_parquet("data/02_processed/dataset.parquet")

# Target 생성 (학습 직전)
df['target_log_close'] = np.log(df['close'])
```

**영향**:
- ⚠️ **02단계 출력**: Target 컬럼 제거 필요
- ⚠️ **03단계 학습 코드**: Target 생성 로직 추가 필요
- ✅ **하위 호환**: 02단계 Feature는 그대로 사용 가능

---

#### 2. Feature Shift 도입
**변경 사항**:
- t일 행에 t-1일의 Feature를 배치
- 의도: "어제 정보로 오늘 종가 예측"

**구현**:
```python
# 03단계에서 필수 적용
for col in feature_cols:
    df[col] = df.groupby('ticker')[col].shift(1)

# Shift로 발생한 NaN 제거
df = df.dropna(subset=feature_cols)
```

**이유**:
- Look-ahead Bias 방지
- 실전 운영 환경과 동일한 조건

**영향**:
- ⚠️ 각 종목의 첫 행(NaN) 자동 제거됨
- ⚠️ 학습 가능 데이터 길이 1일 감소

---

#### 3. Ticker Feature 제외
**변경 사항**:
- **v2.x**: Ticker를 Categorical Feature로 사용 가능
- **v3.0.0**: **Ticker를 Feature로 사용 안 함**

**이유**:
- 신규 상장 종목 예측 불가 문제
- 차원 폭발 (2,900개 종목 → 2,900차원)
- 일반화 성능 저하

**대체 수단**:
```python
# Meta Features로 종목 특성 표현
feature_cols = [
    'feature_ma_5', 'feature_rsi_14', ...  # 기술지표
    'liquidity_score',  # 유동성 (종목별 차이 반영)
    'risk_composite'    # 리스크 (종목별 차이 반영)
]

# ❌ 사용 금지
# categorical_features = ['ticker']
```

**영향**:
- ✅ 신규 종목 즉시 예측 가능
- ✅ 모델 크기 감소
- ⚠️ 종목 고유 패턴 학습 불가 (Trade-off)

---

### ✨ New Features

#### 1. 데이터 길이 표준화 함수
**추가된 함수**: `filter_by_history()`

**위치**: `src/features/builder.py`

**기능**:
```python
def filter_by_history(
    df: pd.DataFrame, 
    min_history: int = 60,
    threshold_ratio: float = 1.0
) -> pd.DataFrame:
    """
    종목별 데이터 길이 표준화
    - min_history: 초기 제거 기간 (warmup)
    - threshold_ratio: 최장 길이 대비 유지 비율
    """
```

**이유**:
- Batch 학습 시 길이 불일치 문제 해결
- Feature 준비 기간(60일) 일관성 있게 제거

**사용 예**:
```python
# 02단계에서 자동 적용
df_final = filter_by_history(
    df_meta, 
    min_history=60,
    threshold_ratio=1.0  # 최장 길이와 일치하는 종목만 유지
)
```

---

#### 2. 통합 모델 지향 설계
**변경 사항**:
- 종목별 개별 모델 → **전체 종목 통합 모델**

**특징**:
```python
# 단일 모델로 전체 종목 처리
model = LightGBMModel(
    feature_list=['feature_ma_5', 'liquidity_score', ...],
    categorical_features=[]  # Ticker 사용 안 함
)

# 전체 데이터로 학습
model.fit(X_all, y_all)

# 신규 종목도 즉시 예측
new_stock_pred = model.predict(new_stock_features)
```

**장점**:
- ✅ 종목 간 공통 패턴 학습
- ✅ 신규 상장 종목 즉시 예측
- ✅ 모델 관리 간소화 (1개 vs 2,900개)

---

### 🔧 Changed

#### 1. 파일 저장 구조 개선
**변경 전**: 날짜별 분산 저장
```
data/01_raw/csv/삼성전자.csv
data/02_processed/csv/삼성전자.csv
```

**변경 후**: 날짜별 폴더 + 통합 Parquet
```
data/01_raw/{YYYYMMDD}/
  ├── krx_prices_{YYYYMMDD}.parquet  # 통합 (기계용)
  ├── ticker_master_{YYYYMMDD}.csv   # 종목 마스터
  └── csv/{종목명}.csv                # 개별 (사람용, 옵션)

data/02_processed/{YYYYMMDD}/
  ├── dataset.parquet                 # 통합 (기계용)
  └── csv/{종목명}.csv                # 개별 (사람용, 옵션)
```

**이유**:
- 날짜별 버전 관리 용이
- 파이프라인 효율성 (통합 Parquet)
- 디버깅 편의성 (개별 CSV)

**하위 호환**: ✅ (파일 위치만 변경, 스키마 동일)

---

#### 2. Feature 계산 모듈화
**변경 사항**:
- 기술지표 계산 로직 `src/features/technical.py`로 통합
- `calc_rsi()`, `calc_macd()`, `calc_bollinger()` 등 재사용 가능 함수

**Before**:
```python
# 노트북에 분산된 계산 로직
df['RSI'] = ...  # RSI 계산
df['MACD'] = ...  # MACD 계산
```

**After**:
```python
# 모듈 임포트 & 재사용
from src.features.technical import calc_rsi, calc_macd

df['feature_rsi_14'] = df.groupby('ticker')['close'].transform(
    lambda x: calc_rsi(x, period=14)
)
```

---

### 📚 Documentation

#### 1. 단계별 책임 명확화
각 단계의 책임을 명확히 정의:

| 단계 | 책임 | Target 포함 |
|------|------|------------|
| **01** | API 원시 데이터 수집 | ❌ |
| **02** | Feature 계산, Meta 생성 | ❌ |
| **03** | Target 생성, 학습, 예측 | ✅ |

#### 2. Feature Shift 주의사항 문서화
```python
# ⚠️ 주의: Feature Shift는 03단계에서만 적용
# 02단계 출력에는 Shift 적용되지 않음

# ✅ 올바른 사용 (03단계)
for col in feature_cols:
    df[col] = df.groupby('ticker')[col].shift(1)
df = df.dropna(subset=feature_cols)

# ❌ 잘못된 사용 (02단계)
# Shift를 02단계에서 적용하면 안 됨
```

---

## [2.0.0] - 2024-12-28

### 🔴 Breaking Changes

#### 1. Feature 명명 규칙 통일
**변경 사항**:
- 모든 Feature 컬럼에 `feature_` prefix 추가

**마이그레이션**:
```python
# v1.x → v2.0
rename_map = {
    'ma_5': 'feature_ma_5',
    'rsi_14': 'feature_rsi_14',
    'macd': 'feature_macd',
    ...
}
df = df.rename(columns=rename_map)
```

---

### ✨ New Features

#### 1. Universe Meta 컬럼 추가
- `liquidity_score`, `risk_composite`
- `is_suspended`, `is_delisted`

---

### 🔧 Changed

#### 1. 파일 포맷 변경
- 01단계: CSV
- 02단계 이후: Parquet

---

## [1.0.0] - 2024-12-01

### Initial Release
- 기본 OHLCV 컬럼
- 기술적 지표: `ma_5`, `rsi_14`, `macd` 등

---

## Migration Guides

### v3.1.0 → v3.1.1 (PATCH)

#### Target 위치 변경 처리

**자동 마이그레이션**: 02단계 재실행하면 자동으로 Target 포함됨

```bash
# 02단계 재실행
jupyter nbconvert --execute 02_build_dataset.ipynb

# 결과: data/02_processed/{date}/dataset.parquet에 target_log_close 포함
```

**수동 마이그레이션** (v3.1.0 데이터셋이 있는 경우):
```python
import pandas as pd
import numpy as np

# v3.1.0 데이터셋 로드 (Target 없음)
df = pd.read_parquet("data/02_processed/{date}/dataset.parquet")

# Target 추가
df['target_log_close'] = np.where(df['close'] > 1, np.log(df['close']), 0)

# 저장
df.to_parquet("dataset_v3.1.1.parquet", index=False)
```

---

### v3.0.x → v3.1.0 (MINOR)

#### Step 1: 코드 업데이트

**config.yaml**:
```yaml
training:
  horizons: [1, 2, 3, 4, 5]  # 추가
  target_col_name: "target_log_close"  # 추가
```

**03_train_predict.ipynb**:
```python
# Before (v3.0.x)
model = LightGBMModel(...)
model.fit(X_train, y_train)

# After (v3.1.0)
model = LightGBMModel(...)
trainer = WalkForwardTrainer(
    model=model,
    horizons=[1, 2, 3, 4, 5],  # Multi-horizon 설정
    target_col_name='target_log_close'
)
results = trainer.run(df, ...)
```

#### Step 2: 데이터셋 확인

v3.0.x 데이터셋은 v3.1.0과 호환됩니다. 추가 작업 불필요.

---

### v2.x → v3.0.0 (MAJOR)

#### Step 1: 02단계 데이터 정리
```python
import pandas as pd

# v2.x 데이터 로드
df = pd.read_parquet("data/02_processed/dataset_v2.parquet")

# Target 컬럼 제거 (v3.0.0에서는 03단계에서 생성)
target_cols = [c for c in df.columns if c.startswith('target_')]
if target_cols:
    print(f"Removing target columns: {target_cols}")
    df = df.drop(columns=target_cols)

# v3.0.0 형식으로 저장
df.to_parquet("dataset_v3.parquet", index=False)
print("✅ Migration complete: v2.x → v3.0.0")
```

#### Step 2: 03단계 학습 코드 업데이트
```python
# v3.0.0 학습 템플릿
df = pd.read_parquet("data/02_processed/dataset.parquet")

# 1. Target 생성 (v3.0.0 필수)
df['target_log_close'] = np.log(df['close'])

# 2. Feature Shift (v3.0.0 필수)
for col in feature_cols:
    df[col] = df.groupby('ticker')[col].shift(1)

# 3. NaN 제거
df = df.dropna(subset=feature_cols + ['target_log_close'])

# 4. 학습
model.fit(df[feature_cols], df['target_log_close'])
```

#### Step 3: Ticker Feature 제거
```python
# ❌ v2.x (사용 금지)
model = LightGBMModel(
    feature_list=['ticker', 'feature_ma_5', ...],
    categorical_features=['ticker']  # ← 제거 필요
)

# ✅ v3.0.0 (권장)
model = LightGBMModel(
    feature_list=['feature_ma_5', 'liquidity_score', ...],
    categorical_features=[]  # Ticker 없음
)
```

---

### v1.x → v2.0.0 (MAJOR)

#### Step 1: 컬럼명 변경
```python
feature_renames = {
    'ma_5': 'feature_ma_5',
    'ma_20': 'feature_ma_20',
    'rsi_14': 'feature_rsi_14',
    # ... 모든 Feature 컬럼
}
df = df.rename(columns=feature_renames)
```

---

## Version History Summary

| Version | Date | Type | Key Changes |
|---------|------|------|-------------|
| **3.1.1** | 2026-01-21 | 🔵 PATCH | Target 생성 위치 재변경 (03→02), 재현성 강화 |
| **3.1.0** | 2026-01-21 | 🟢 MINOR | Multi-horizon 예측, Target-Centric Alignment |
| 3.0.0 | 2026-01-18 | 🔴 MAJOR | Target 생성 위치 변경, Feature Shift, Ticker 제외 |
| 2.0.0 | 2024-12-28 | 🔴 MAJOR | Feature prefix 통일, Universe Meta 추가 |
| 1.0.0 | 2024-12-01 | - | Initial release |

---

## Breaking Changes Impact Matrix

| Change | 01단계 | 02단계 | 03단계 | 모델 | 하위 호환 |
|--------|--------|--------|--------|------|-----------|
| **v3.1.1: Target 위치 복귀** | ✅ | ⚠️ | ✅ | ✅ | 🟢 높음 |
| **v3.1.0: Multi-horizon** | ✅ | ✅ | ⚠️ | ❌ | 🟢 높음 |
| v3.0.0: Target 위치 변경 | ✅ | ⚠️ | ⚠️ | ❌ | 🟡 보통 |
| v3.0.0: Feature Shift | ✅ | ✅ | ⚠️ | ❌ | 🟡 보통 |
| v3.0.0: Ticker 제외 | ✅ | ✅ | ⚠️ | ❌ | 🔴 낮음 |

**범례**:
- ✅ 영향 없음
- ⚠️ 코드 수정 필요
- ❌ 재학습 필요
- 🟢 높음: 자동 또는 최소 작업
- 🟡 보통: 단계별 마이그레이션 필요
- 🔴 낮음: 전면 재작업 필요

---

## Notes

- **MAJOR 업데이트 주기**: 분기당 1회 이내
- **MINOR 업데이트**: 기능 추가 시 즉시 릴리스
- **PATCH 업데이트**: 버그 수정 및 문서 개선
- **마이그레이션 지원**: 모든 Breaking Change에 스크립트 제공
- **테스트 커버리지**: 스키마 변경 시 단위 테스트 필수
- **문서 우선**: 코드 변경 전 스키마 문서 업데이트

---

**Last Updated**: 2026-01-21  
**Current Version**: 3.1.1  
**Maintained by**: SignalWeaver Team