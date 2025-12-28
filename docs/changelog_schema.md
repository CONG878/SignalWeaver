# Schema Changelog

All notable changes to the data schema will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2024-12-28

### 🔴 Breaking Changes

#### 1. Feature 명명 규칙 통일
**변경 사항**:
- 모든 Feature 컬럼에 `feature_` prefix 추가
- 이유: 03단계에서 Feature 자동 탐지 및 일괄 처리 용이

**마이그레이션 가이드**:
```python
# 기존 코드 (v1.x)
df['ma_5']
df[['rsi_14', 'macd', 'bollinger_upper']]

# 새 코드 (v2.0)
df['feature_ma_5']
df[['feature_rsi_14', 'feature_macd', 'feature_bollinger_upper']]

# 자동 마이그레이션
v1_cols = ['ma_5', 'ma_20', 'rsi_14', ...]
v2_cols = ['feature_' + c for c in v1_cols]
df = df.rename(columns=dict(zip(v1_cols, v2_cols)))
```

**영향**:
- ✅ 02단계 출력: 새 컬럼명 적용
- ⚠️ 03단계 학습 코드: Feature 리스트 수정 필요
- ⚠️ 기존 모델: v1.x 데이터로 학습된 모델은 v2.0 데이터 사용 불가

---

### ✨ New Features

#### 1. Universe Meta 컬럼 추가
**추가된 컬럼**:
- `liquidity_score` (float): 20일 평균 거래대금 기반 유동성 점수
- `risk_composite` (float): 복합 리스크 점수 (0~1)
- `risk_volatility` (float): 변동성 리스크 성분
- `risk_volume_surge` (int): 거래량 급증 플래그
- `is_suspended` (int): 거래정지 여부
- `is_delisted` (int): 상장폐지 여부

**용도**:
- 03단계 학습 시 Feature로 활용 가능
- 운영 시 Universe 필터링 기준

**하위 호환**: ✅ (NULL 허용 컬럼이므로 기존 코드 영향 없음)

---

### 🔧 Changed

#### 1. Target 생성 시점 변경
**변경 전**: 02단계에서 `target_return` 등 생성
**변경 후**: 03단계에서 필요 시 생성

**이유**: 
- 예측 기간(horizon)이 실험마다 다를 수 있음
- 02단계는 "Feature 준비"만 담당

**영향**: ⚠️ 기존 02단계 출력에 의존하던 코드 수정 필요

#### 2. 파일 포맷 정책 변경
**변경 전**: 전체 CSV
**변경 후**: 
- 01단계: CSV (pykrx 원시 데이터)
- 02단계 이후: Parquet (통합 데이터셋)

**이유**: I/O 성능 향상 (수천 개 종목 처리 시)

**하위 호환**: ✅ (파일 포맷만 변경, 스키마는 동일)

---

### 📚 Documentation

#### 1. Feature 준비 기간 명시
- 02단계에서 종목별 최초 60일 데이터 제거
- 이유: MA_60 등 장기 지표 계산을 위한 warmup 기간

---

## [1.0.0] - 2024-12-01

### Initial Release

#### Features
- 기본 OHLCV 컬럼
- 기술적 지표: `ma_5`, `ma_20`, `ma_60`, `rsi_14`, `macd` 등
- Target: `target_return`, `target_log_return`

---

## Migration Guide: v1.x → v2.0

### Step 1: 컬럼명 변경 스크립트 실행
```python
# migrate_v1_to_v2.py
import pandas as pd

def migrate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """v1.x DataFrame을 v2.0 스키마로 변환"""
    
    # Feature 컬럼 매핑
    feature_renames = {
        # 이동평균
        'ma_5': 'feature_ma_5',
        'ma_20': 'feature_ma_20',
        'ma_60': 'feature_ma_60',
        
        # 기술지표
        'rsi_14': 'feature_rsi_14',
        'macd': 'feature_macd',
        'macd_signal': 'feature_macd_signal',
        'macd_hist': 'feature_macd_hist',
        
        # 볼린저
        'bollinger_upper': 'feature_bollinger_upper',
        'bollinger_middle': 'feature_bollinger_middle',
        'bollinger_lower': 'feature_bollinger_lower',
        
        # 거래량
        'volatility_20': 'feature_volatility_20',
        'volume_ratio': 'feature_volume_ratio',
    }
    
    df = df.rename(columns=feature_renames)
    
    # 메타데이터 업데이트
    df.attrs['schema_version'] = '2.0.0'
    df.attrs['migrated_from'] = 'v1.0.0'
    
    return df

# 실행
df_v1 = pd.read_parquet('dataset_v1.parquet')
df_v2 = migrate_schema(df_v1)
df_v2.to_parquet('dataset_v2.parquet')
```

### Step 2: 모델 재학습
```bash
# v1.x 모델은 v2.0 데이터와 호환 불가
# 모든 모델을 재학습해야 함
python scripts/03_train_predict.py --schema-version 2.0.0
```

### Step 3: 코드베이스 업데이트
```python
# 모든 Feature 참조를 업데이트
# grep -r "df\['ma_5'\]" src/
# → df['feature_ma_5']로 변경
```

---

## Version History Summary

| Version | Date | Type | Description |
|---------|------|------|-------------|
| 2.0.0 | 2024-12-28 | 🔴 MAJOR | Feature 명명 규칙 통일, Universe Meta 추가 |
| 1.0.0 | 2024-12-01 | - | Initial release |

---

## Notes

- **Breaking Changes는 최소화**: MAJOR 업데이트는 분기당 1회 이내로 제한
- **마이그레이션 스크립트 제공**: 모든 Breaking Change에는 자동 변환 스크립트 포함
- **테스트 커버리지**: 스키마 변경 시 반드시 단위 테스트 추가
