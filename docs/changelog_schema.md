# Schema Changelog

All notable changes to the data schema will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.0.0] - 2026-01-18

### 🔴 Breaking Changes

#### 1. Target 생성 위치 변경
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
| **3.0.0** | 2026-01-18 | 🔴 MAJOR | Target 생성 위치 변경, Feature Shift, Ticker 제외 |
| 2.0.0 | 2024-12-28 | 🔴 MAJOR | Feature prefix 통일, Universe Meta 추가 |
| 1.0.0 | 2024-12-01 | - | Initial release |

---

## Breaking Changes Impact Matrix

| Change | 01단계 | 02단계 | 03단계 | 모델 |
|--------|--------|--------|--------|------|
| Target 생성 위치 변경 | ✅ | ⚠️ | ⚠️ | ❌ |
| Feature Shift | ✅ | ✅ | ⚠️ | ❌ |
| Ticker Feature 제외 | ✅ | ✅ | ⚠️ | ❌ |

**범례**:
- ✅ 영향 없음
- ⚠️ 코드 수정 필요
- ❌ 재학습 필요

---

## Notes

- **MAJOR 업데이트 주기**: 분기당 1회 이내
- **마이그레이션 지원**: 모든 Breaking Change에 스크립트 제공
- **테스트 커버리지**: 스키마 변경 시 단위 테스트 필수
- **문서 우선**: 코드 변경 전 스키마 문서 업데이트

---

**Last Updated**: 2026-01-18  
**Current Version**: 3.0.0  
**Maintained by**: SignalWeaver Team