# 📄 data_schema.md (Updated for Feature Prefix Convention)

본 스키마는 다음 요구사항을 충족하도록 구성되었다:
- LightGBM 1차 예측
- 후보 종목 선택 (유동성·리스크 필터)
- GRU 2차 정밀 예측
- 운영/학습 데이터 분리
- 향후 백테스트 자동화, 모델 재현성

---

# 📌 1. 파일 저장 규칙 / 포맷

### 1.1 기본 포맷
- **01단계 (원시 데이터)**: CSV (종목별 개별 파일)
- **02단계 이후 (통합 데이터)**: Parquet (Snappy 압축)

### 1.2 파일 네이밍

```
# 01단계: 원시 데이터 (pykrx)
data/01_raw/krx_prices_{YYYYMMDD}/{종목명}.csv

# 02단계: 통합 데이터셋 (Feature + Meta)
data/03_processed/dataset_{YYYYMMDD}.parquet

# 03단계: 모델 예측 결과
data/05_results/predictions_{YYYYMMDD}.parquet
```

### 1.3 스키마 버전 관리
- `schema_version`: 스키마 변경 시 버전 증가
- `as_of_date`: 데이터 생성 기준일

---

# 📌 2. 공통 기본 컬럼 (모든 단계에서 사용)

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `date` | date | 거래일 |
| `ticker` | string | 종목 코드 |
| `open` | float | 시가 |
| `high` | float | 고가 |
| `low` | float | 저가 |
| `close` | float | 종가 |
| `volume` | int | 거래량 |
| `amount` | float | 거래대금 (옵션, pykrx 제공 시) |
| `change_pct` | float | 등락률 (%, pykrx 제공 시) |
| `as_of_date` | date | 데이터 생성 기준일 |
| `schema_version` | int | 스키마 버전 |

---

# 📌 3. Feature 스키마 (feature_ prefix)

## ⚠️ 명명 규칙
모든 Feature는 **`feature_` prefix**를 사용합니다.

### 3.1 가격 기반 기본 지표

| 컬럼명 | 설명 |
|--------|------|
| `feature_ma_5` | 5일 단순 이동평균 |
| `feature_ma_20` | 20일 단순 이동평균 |
| `feature_ma_60` | 60일 단순 이동평균 |
| `feature_volatility_20` | 20일 수익률 표준편차 |

### 3.2 기술적 지표 (Technical Indicators)

| 컬럼명 | 설명 |
|--------|------|
| `feature_rsi_14` | RSI (14일) |
| `feature_macd` | MACD 값 (12-26 EMA 차이) |
| `feature_macd_signal` | MACD 시그널 (9일 EMA) |
| `feature_macd_hist` | MACD 히스토그램 |
| `feature_bollinger_upper` | 볼린저 상단 (20일, 2σ) |
| `feature_bollinger_middle` | 볼린저 중심선 (20일 MA) |
| `feature_bollinger_lower` | 볼린저 하단 (20일, 2σ) |

### 3.3 거래량 지표

| 컬럼명 | 설명 |
|--------|------|
| `feature_volume_ratio` | 거래량 비율 (현재/20일 평균) |

---

# 📌 4. Universe Meta (운영 판단용 지표)

02단계에서 생성되며, 03단계 (학습/운영)에서 활용됩니다.

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `liquidity_score` | float | 유동성 점수 (20일 평균 거래대금) |
| `risk_composite` | float | 복합 리스크 점수 (0~1) |
| `risk_volatility` | float | 변동성 리스크 성분 |
| `risk_volume_surge` | int | 거래량 급증 플래그 (0/1) |
| `is_suspended` | int | 거래정지 여부 (0: 정상, 1: 정지) |
| `is_delisted` | int | 상장폐지 여부 (0: 정상, 1: 폐지) |

### 사용 목적
- **학습 시**: 리스크 플래그를 Feature로 활용 가능
- **운영 시**: Universe 필터링 기준으로 활용

---

# 📌 5. Label (타깃) 스키마

## ⚠️ 생성 시점
**Target은 03단계에서 생성**합니다 (02단계에서는 생성하지 않음).

### 5.1 LightGBM용 타깃

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `target_return` | float | n일 후 수익률 ((future_close/close) - 1) |
| `target_log_return` | float | n일 후 로그 수익률 |
| `target_direction` | int | 방향성 라벨 (1: 상승, 0: 하락) |

### 5.2 GRU용 타깃 (예정)

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `target_gru` | float | GRU 예측 목표 |
| `mask` | int | 시퀀스 패딩 여부 (0/1) |

---

# 📌 6. 모델 예측 결과 스키마

### LightGBM 예측 결과

| 컬럼명 | 설명 |
|--------|------|
| `date` | 예측 기준일 |
| `ticker` | 종목 코드 |
| `score_lgbm` | LightGBM 예측 점수 |
| `rank_lgbm` | 종목 순위 |
| `selected_lgbm` | 선택 여부 (True/False) |

---

# 📌 7. 백테스트/실매매용 시그널 스키마

| 컬럼명 | 설명 |
|--------|------|
| `signal_buy` | 매수 신호 (1/0) |
| `signal_sell` | 매도 신호 (1/0) |
| `weight` | 포트폴리오 비중 |
| `execution_price` | 체결가 |
| `position` | 포지션 상태 |
| `pnl_daily` | 일일 손익 |
| `pnl_cum` | 누적 손익 |

---

# 📌 8. 운영 단계 추가 컬럼

| 컬럼명 | 설명 |
|--------|------|
| `data_version` | 데이터셋 버전 (날짜·해시) |
| `model_version_lgbm` | LightGBM 모델 버전 |
| `model_version_gru` | GRU 모델 버전 |
| `latency_ms` | 추론 시간 (밀리초) |
| `trace_id` | 파이프라인 추적 ID |

---

# 📌 9. 스키마 버전 관리 정책

## 버전 표기법: Semantic Versioning

```
schema_version: "MAJOR.MINOR.PATCH"

예: "2.0.0"
    │  │  └─ PATCH: 문서/주석 변경, 메타데이터 추가 (하위 호환)
    │  └──── MINOR: 컬럼 추가, Feature 확장 (하위 호환)
    └─────── MAJOR: 컬럼 삭제, 타입 변경, 명명 규칙 변경 (하위 호환 X)
```

## 변경 수준별 정의

### MAJOR 변경 (하위 호환 불가)
**트리거**:
- 기존 컬럼 삭제
- 컬럼명 변경 (예: `ma_5` → `feature_ma_5`)
- 타입 변경 (예: `int` → `float`)
- 필수 컬럼 추가 (모든 레코드에 값 필요)

**버전 증가**: `2.0.0` → `3.0.0`

**조치**:
- 기존 파이프라인 코드 수정 필요
- `changelog_schema.md`에 **Breaking Changes** 명시
- 데이터 마이그레이션 스크립트 제공

### MINOR 변경 (하위 호환 유지)
**트리거**:
- 신규 Feature 컬럼 추가
- 옵션 메타 컬럼 추가 (NULL 허용)
- 신규 인덱스/파티션 추가

**버전 증가**: `2.0.0` → `2.1.0`

**조치**:
- 기존 코드는 그대로 작동
- `changelog_schema.md`에 **New Features** 명시

### PATCH 변경 (영향 없음)
**트리거**:
- 문서/주석 업데이트
- 스키마 설명 보강
- 예시 코드 수정

**버전 증가**: `2.0.0` → `2.0.1`

**조치**:
- 코드 변경 불필요
- Git commit만으로 충분

---

## 데이터 파일에 버전 기록하는 방법

### Parquet 메타데이터에 포함
```python
import pandas as pd

df.to_parquet(
    "dataset.parquet",
    compression='snappy',
    index=False,
    # Parquet 메타데이터에 버전 기록
    metadata={
        'schema_version': '2.0.0',
        'created_at': '2024-12-28T10:30:00Z',
        'pipeline_version': 'v1.2.3',
        'data_source': 'pykrx',
    }
)
```

### CSV의 경우 별도 메타 파일
```
dataset_20241228.csv
dataset_20241228.meta.json  ← 버전 정보
```

**meta.json 예시**:
```json
{
  "schema_version": "2.0.0",
  "created_at": "2024-12-28T10:30:00Z",
  "row_count": 681286,
  "column_count": 26,
  "md5_checksum": "a3f2c9..."
}
```

---

# ✔ 주요 변경 사항

## v2.0.0 (2024-12-28) - MAJOR RELEASE ⚠️

**Breaking Changes**:

### 1. Feature 명명 규칙 통일
- **변경 전**: `ma_5`, `rsi_14`, `macd` 등
- **변경 후**: `feature_ma_5`, `feature_rsi_14`, `feature_macd` 등
- **이유**: 03단계에서 Feature 자동 탐지 및 일괄 처리 용이

### 2. Universe Meta 컬럼 추가
- `liquidity_score`: 유동성 점수
- `risk_composite`: 복합 리스크
- `is_suspended`, `is_delisted`: 거래 가능 여부
- **이유**: 03단계가 도메인 판단을 하지 않도록 사전 계산

### 3. Target 생성 시점 변경
- **변경 전**: 02단계에서 Target 생성
- **변경 후**: 03단계에서 필요 시 생성
- **이유**: 예측 기간(horizon)이 실험마다 다를 수 있음

### 4. 파일 포맷 정책 변경
- **01단계**: CSV (pykrx API 직접 저장)
- **02단계 이후**: Parquet (I/O 속도 향상)
- **이유**: 수천 개 종목 처리 시 성능 개선

---

# ✔ 결론

- 현재 스키마는 **LightGBM 학습 → Universe 선정 → 예측 → 백테스트**까지 전 과정을 지원
- `feature_` prefix 통일로 **자동화 파이프라인 구축** 용이
- **02단계 출력 = 03단계 입력**으로 명확한 계약 관계 확립
- 향후 GRU 2차 예측 추가 시에도 **스키마 변경 없이 확장 가능**