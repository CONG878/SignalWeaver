# Schema Update v3.4.0 - RandomForest & Ensemble Models

## 📋 변경 개요

| 항목 | 내용 |
|------|------|
| **버전** | v3.3.0 → v3.4.0 |
| **타입** | 🟢 MINOR (새 모델 추가) |
| **핵심 변경** | RF 모델 도입 + 앙상블 학습 단계 추가 |
| **영향 범위** | 03_training 폴더 구조, config.yaml, 모델 로딩 로직 |
| **하위 호환성** | 부분 호환 (기존 LightGBM 모델 유효) |

---

## 🔄 Step별 변경사항

### Step 3: 모델 학습 & 검증 (Updated)

#### 3.1 폴더 구조 변경

**Before (v3.3.0)**:
```
data/03_training/{YYYYMMDD}/
├── *.pkl                    # 통일된 모델 파일
├── registry.json
└── predictions.parquet      # 검증 예측값
```

**After (v3.4.0)** - 모델별 분리:
```
data/03_training/{YYYYMMDD}/
├── lightgbm/               # ✨ NEW: LightGBM 전용 폴더
│   ├── v1_lgbm_20260213_abc123.pkl
│   ├── registry.json
│   └── predictions.parquet  # OOF 검증 예측값
│
├── randomforest/           # ✨ NEW: RandomForest 전용 폴더
│   ├── v1_rf_20260213_def456.pkl
│   ├── registry.json
│   └── predictions.parquet  # OOF 검증 예측값
│
└── ensemble/               # ✨ NEW: 앙상블 모델 폴더
    ├── v1_ens_20260213_ghi789.pkl
    ├── registry.json
    └── predictions.parquet  # 블렌딩된 OOF 예측값
```

**이점**:
- ✅ 모델별로 아티팩트를 명확히 분리 (매니지먼트 용이)
- ✅ 개별 모델 성능 추적 가능 (registry 개별 관리)
- ✅ 04단계에서 active_model에 따라 자동으로 올바른 모델 로드

#### 3.2 새로운 노트북: 03b_train_ensemble.ipynb

```
03_train_predict.ipynb (기존)
        ↓
    LightGBM 학습 + OOF 저장
    RandomForest 학습 + OOF 저장
        ↓
03b_train_ensemble.ipynb (✨ NEW)
        ↓
    [1] 개별 모델의 OOF 예측값 로드
    [2] Scipy.optimize로 최적 가중치 탐색
    [3] EnsembleModel 인스턴스화 + 저장
```

**목표**:
- LGBM OOF 예측 + RF OOF 예측 → 가중치 최적화
- 블렌딩 예측값 저장: `ensemble/predictions.parquet`

#### 3.3 모델 선택 메커니즘

config.yaml에서 `active_model` 지정:
```yaml
active_model: "ensemble"  # 'lightgbm', 'randomforest', 'ensemble'
```

모드별 동작:
```python
# 03_train_predict.ipynb
if model_type == "lightgbm":
    model = LightGBMModel(...)
elif model_type == "randomforest":
    model = RandomForestMultiModel(...)

# 03b_train_ensemble.ipynb (새 노트북)
# → 개별 모델 OOF 로드 → 최적 가중치 탐색 → EnsembleModel 저장

# 04_forecast_future.ipynb
if active_model == "lightgbm":
    model = LightGBMModel.load(...)
elif active_model == "randomforest":
    model = RandomForestMultiModel.load(...)
elif active_model == "ensemble":
    model = EnsembleModel.load(...)
```

---

## 📐 파일 저장 규칙 (Updated v3.4.0)

### 3단계 모델 아티팩트

| 파일 | 위치 | 형식 | 설명 |
|------|------|------|------|
| **LightGBM 모델** | `03_training/{date}/lightgbm/v1_lgbm_*.pkl` | .pkl | Booster 객체 |
| **RandomForest 모델** | `03_training/{date}/randomforest/v1_rf_*.pkl` | .pkl | MultiOutputRegressor 객체 |
| **Ensemble 모델** | `03_training/{date}/ensemble/v1_ens_*.pkl` | .pkl | EnsembleModel 객체 (가중치 포함) |
| **Registry** | `03_training/{date}/{model_name}/registry.json` | .json | 모델 메타데이터 |
| **OOF 검증 예측** | `03_training/{date}/{model_name}/predictions.parquet` | .parquet | 검증용 과거 예측값 |

---

## ⚙️ 설정 변경 (config.yaml)

### Before (v3.3.0)
```yaml
training:
  lgbm_params:
    # LightGBM 파라미터만 존재
```

### After (v3.4.0)
```yaml
training:
  horizons: [1, 2, 3, 4, 5]
  
  lgbm_params:
    n_estimators: 100
    num_leaves: 31
    learning_rate: 0.05
    # ...
  
  randomforest_params:  # ✨ NEW
    n_estimators: 40
    max_depth: 8
    min_samples_split: 10
    min_samples_leaf: 5
    n_jobs: -1
    random_state: 42
    max_samples: 0.6

# ✨ NEW: 모델 선택 옵션
active_model: "ensemble"  # 'lightgbm' | 'randomforest' | 'ensemble'
```

---

## 🔌 모델 인터페이스 확장

### ModelBase (상속 구조)

```
ModelBase (추상 클래스)
├── LightGBMModel ✓ (기존)
├── RandomForestMultiModel ✓ (NEW)
└── EnsembleModel ✓ (NEW)
```

### 각 모델의 메서드

| 메서드 | LightGBM | RandomForest | Ensemble |
|--------|----------|--------------|----------|
| `fit()` | ✅ | ✅ | ❌ (사전 학습 필수) |
| `predict()` | ✅ | ✅ | ✅ |
| `save()` | ✅ | ✅ | ✅ |
| `load()` | ✅ | ✅ | ✅ |

### 새로운 클래스들

#### RandomForestMultiModel
```python
class RandomForestMultiModel(ModelBase):
    """
    Scikit-learn RandomForest + MultiOutputRegressor
    - fit() 시 여러 타깃 동시 학습
    - predict() 시 모든 호라이즌 예측
    """
    def __init__(self, model_version, params, feature_list):
        # params에서 n_estimators, max_depth 등 추출
        self.model = MultiOutputRegressor(RandomForestRegressor(**params))
```

#### EnsembleModel
```python
class EnsembleModel(ModelBase):
    """
    여러 모델을 가중치로 블렌딩
    - 03b에서 사전 학습된 모델 조립용
    - fit() 불가 (사전 학습 모델 필수)
    - predict(): (w1*pred_lgbm + w2*pred_rf + ...)
    """
    def __init__(self, model_version, models, weights):
        self.models = models        # [LightGBMModel, RandomForestMultiModel, ...]
        self.weights = weights      # [0.3, 0.7, ...]
```

---

## 📊 Step 3 데이터 흐름 (Updated)

### 순서 1: 개별 모델 학습 (03_train_predict.ipynb)

```
[02_processed 데이터]
├─ Feature + Target
│
└─→ 03_train_predict.ipynb
    ├─ active_model = "lightgbm"
    │   └─→ LightGBMModel.fit()
    │       └─→ save: data/03_training/{date}/lightgbm/*.pkl
    │           + predictions.parquet (OOF)
    │
    └─ active_model = "randomforest"
        └─→ RandomForestMultiModel.fit()
            └─→ save: data/03_training/{date}/randomforest/*.pkl
                + predictions.parquet (OOF)
```

### 순서 2: 앙상블 최적화 (03b_train_ensemble.ipynb - NEW)

```
[개별 모델 OOF]
├─ data/03_training/{date}/lightgbm/predictions.parquet
├─ data/03_training/{date}/randomforest/predictions.parquet
│
└─→ 03b_train_ensemble.ipynb
    ├─ [1] OOF 로드
    │
    ├─ [2] Scipy.optimize.minimize로 가중치 탐색
    │   └─ 목표: minimize(RMSE(w1*pred_lgbm + w2*pred_rf))
    │      제약: w1 + w2 = 1.0
    │
    ├─ [3] EnsembleModel 생성 및 저장
    │   └─ models = [lgbm_loaded, rf_loaded]
    │   └─ weights = [best_w1, best_w2]
    │
    └─→ save: data/03_training/{date}/ensemble/*.pkl
        + predictions.parquet (블렌딩 예측)
```

### 순서 3: 모델 선택 (04_forecast_future.ipynb)

```
config.yaml에서 active_model 읽기
│
├─ "lightgbm"    → LightGBMModel.load()
├─ "randomforest" → RandomForestMultiModel.load()
└─ "ensemble"     → EnsembleModel.load()
  
→ 04단계에서 미래 예측 수행 (모델 선택에 무관)
```

---

## 📝 ProjectPaths 업데이트 (v3.4.0)

### 추가 메서드

```python
@dataclass
class ProjectPaths:
    # 기존 메서드들...
    
    # ✨ NEW: 모델별 경로 조회
    def get_model_dir(self, model_name: str) -> Path:
        """
        모델별 전용 디렉토리 반환
        
        Examples:
        - get_model_dir("lightgbm") → data/03_training/{date}/lightgbm/
        - get_model_dir("randomforest") → data/03_training/{date}/randomforest/
        - get_model_dir("ensemble") → data/03_training/{date}/ensemble/
        """
        return self.training_dir / model_name
    
    def get_model_path(self, model_name: str) -> Path:
        """
        모델 폴더 내 최신 .pkl 파일 반환
        """
        model_dir = self.get_model_dir(model_name)
        pkl_files = list(model_dir.glob("*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl found in {model_dir}")
        return max(pkl_files, key=lambda p: p.stat().st_mtime)
    
    def get_predictions_parquet(self, model_name: str = None) -> Path:
        """
        OOF 검증 예측값 (모델별)
        
        Args:
            model_name: 'lightgbm', 'randomforest', 'ensemble'
                       (None이면 active_model 사용)
        """
        if model_name is None:
            model_name = self.active_model
        return self.get_model_dir(model_name) / "predictions.parquet"
```

---

## 🔄 노트북 실행 순서 (Updated)

1. **01_collect_data.ipynb** → `data/01_raw/`
2. **02_build_dataset.ipynb** → `data/02_processed/`
3. **03_train_predict.ipynb** → `data/03_training/{model_name}/` (LightGBM 또는 RF)
   - ⚠️ **중요**: 양쪽 모델 모두 학습해야 앙상블 가능
     ```python
     # 방법 1: 순차 실행
     # config.yaml: active_model = "lightgbm" → 03_train_predict.ipynb 실행
     # config.yaml: active_model = "randomforest" → 03_train_predict.ipynb 다시 실행
     
     # 방법 2: 병렬 실행 (수동)
     # 터미널에서 2개의 노트북 프로세스 동시 실행
     ```
4. **03b_train_ensemble.ipynb** (NEW) → `data/03_training/ensemble/` (선택사항)
   - active_model을 "ensemble"로 설정하려면 반드시 실행
5. **04_forecast_future.ipynb** → `data/04_forecasts/`
6. **05_universe_selection.ipynb** → `data/05_universe/`

---

## 🔀 사용 시나리오

### 시나리오 A: LightGBM만 사용
```yaml
# config.yaml
active_model: "lightgbm"
```
- 03_train_predict: LightGBM만 학습
- 03b_train_ensemble: ⏭️ 스킵
- 04,05단계: LightGBM 예측값 사용

### 시나리오 B: RandomForest만 사용
```yaml
# config.yaml
active_model: "randomforest"
```
- 03_train_predict: RandomForest만 학습
- 03b_train_ensemble: ⏭️ 스킵
- 04,05단계: RandomForest 예측값 사용

### 시나리오 C: 앙상블 사용 (권장)
```yaml
# config.yaml
active_model: "ensemble"
```
- 03_train_predict: LightGBM + RandomForest 양쪽 학습
- 03b_train_ensemble: 앙상블 가중치 최적화
- 04,05단계: 블렌딩된 예측값 사용

---

## 📋 변경 파일 목록

### 새로 추가된 파일
- ✅ `src/models/randomforest_model.py` - RandomForestMultiModel 클래스
- ✅ `src/models/ensemble_model.py` - EnsembleModel 클래스
- ✅ `03b_train_ensemble.ipynb` - 앙상블 최적화 노트북

### 수정된 파일
- 🔧 `src/models/base.py` - ModelBase 확장 (3개 모델 호환성)
- 🔧 `src/modeling/trainer.py` - WalkForwardTrainer 호환성 확인
- 🔧 `src/models/artifact.py` - 모델별 폴더 구조 지원
- 🔧 `src/utils/config.py` - ProjectPaths 메서드 추가
- 🔧 `config/config.yaml` - randomforest_params + active_model 추가
- 🔧 `03_train_predict.ipynb` - 모델 선택 로직 추가
- 🔧 `04_forecast_future.ipynb` - 모델 로딩 로직 확장

---

## ✔️ 마이그레이션 가이드 (v3.3.0 → v3.4.0)

### 필수 작업 없음
- 기존 LightGBM 모델은 그대로 유효
- 폴더 구조는 자동으로 생성됨

### 선택 작업: 앙상블 사용

1. **양쪽 모델 학습**:
   ```bash
   # 터미널
   jupyter notebook 03_train_predict.ipynb &
   jupyter notebook 03_train_predict.ipynb &  # 다시 실행
   ```
   
   또는 `config.yaml`에서 순차 실행:
   ```yaml
   active_model: "lightgbm"
   # 03_train_predict 실행
   
   active_model: "randomforest"
   # 03_train_predict 다시 실행
   ```

2. **앙상블 학습**:
   ```bash
   jupyter notebook 03b_train_ensemble.ipynb
   ```

3. **설정 업데이트**:
   ```yaml
   config/config.yaml:
   active_model: "ensemble"
   ```

4. **이후 단계 실행**:
   ```bash
   jupyter notebook 04_forecast_future.ipynb
   jupyter notebook 05_universe_selection.ipynb
   ```

---

## 🔍 스키마 호환성

| 버전 | 모델 지원 | 폴더 구조 | 호환성 |
|------|---------|---------|-------|
| **v3.3.0** | LightGBM only | Flat | ✅ 기존 |
| **v3.4.0** | LGBM + RF + Ensemble | Hierarchical | ✅ 상위 호환 |

**주의**: 
- v3.4.0에서 생성한 모델은 v3.3.0 코드에서 로드 불가능
- 기존 v3.3.0 모델은 v3.4.0에서 자동 호환됨

---

## 📅 버전 정보

| 항목 | 값 |
|------|-----|
| **이전 버전** | v3.3.0 (2026-02-09) |
| **신규 버전** | v3.4.0 (2026-02-17) |
| **변경 타입** | 🟢 MINOR |
| **파일 포맷** | 변경 없음 (.pkl, .parquet 유효) |
| **폴더 구조** | 변경 있음 (모델별 계층 추가) |
