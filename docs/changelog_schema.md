# Schema Changelog (Updated v3.2.0)

All notable changes to the data schema will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

**새로운 모듈** (src/utils/filters.py):
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
| **3.2.0** | 2026-02-07 | 🟢 MINOR | 04단계(미래예측) + 05단계(유니버스) + 위험도 |
| **3.1.1** | 2026-01-21 | 🔵 PATCH | Target 위치 재변경 (03→02) |
| **3.1.0** | 2026-01-21 | 🟢 MINOR | Multi-horizon, Target-Centric Alignment |
| 3.0.0 | 2026-01-18 | 🔴 MAJOR | Target 위치 변경, Feature Shift, Ticker 제외 |
| 2.0.0 | 2024-12-28 | 🔴 MAJOR | Feature prefix 통일 |
| 1.0.0 | 2024-12-01 | - | Initial release |

---

## Breaking Changes Impact Matrix (v3.2.0)

| Change | 01단계 | 02단계 | 03단계 | 04단계 | 05단계 | 모델 | 하위 호환 |
|--------|--------|--------|--------|--------|--------|------|-----------|
| **v3.2.0: 04+05단계 추가** | ✅ | ✅ | ✅ | 🆕 | 🆕 | ✅ | 🟢 높음 |

**범례**:
- ✅ 영향 없음
- 🆕 신규 추가 (하위 호환)
- ⚠️ 코드 수정 필요
- ❌ 재학습 필요
- 🟢 높음: 자동 또는 최소 작업
- 🟡 보통: 단계별 마이그레이션 필요

---

**Last Updated**: 2026-02-07  
**Current Version**: 3.2.0  
**Maintained by**: SignalWeaver Team
