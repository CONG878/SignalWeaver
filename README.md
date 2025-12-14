# SignalWeaver

**SignalWeaver**는 시계열 데이터(Time-series)로부터 유의미한 투자 신호(Signal)를 직조(Weave)해내는 **MLOps 기반의 정량 투자 파이프라인**입니다.

이 프로젝트는 단순한 주가 예측을 넘어, **데이터 수집 → 피처 엔지니어링 → Walk-Forward 검증 → 모델 아티팩트 관리 → 리스크 필터링**으로 이어지는 전체 라이프사이클을 엔지니어링 관점에서 구조화하는 데 초점을 맞췄습니다.

## 🏗 Architectural Blueprint

### 1. 핵심 설계 원칙 (Design Principles)

이 파이프라인은 모듈 간의 결합도를 낮추고 확장성을 높이기 위해 다음과 같은 원칙으로 구성되어 있습니다.

1.  **계약 기반 분리 (Decoupling by Contract)**: `ModelBase` 인터페이스를 통해 모든 모델 객체의 **`fit`**, **`predict`**, **`save`** 메서드 호출 규약을 통일했습니다.
2.  **검증 전략 고정 (Fixed Validation Strategy)**: `WalkForwardTrainer` 클래스가 Walk-Forward 검증 로직을 전담하여, 모든 실험의 시계열 검증 방법론을 재현 가능하게 고정합니다.
3.  **데이터 스키마 고정**: 모든 단계의 <a href="./docs/data_schema.md" target="_blank">데이터 입/출력 스키마</a>(`data_schema.md` 참조)를 고정하여 모듈 간의 데이터 오염을 방지합니다.

### 2. 파이프라인 <a href="./docs/dependency_diagram.svg" target="_blank">의존성 다이어그램</a>

`docs/dependency_diagram.svg` 참조

## 💻 Core Implementation (Implemented Contracts)

| 모듈 | 파일 | 담당 역할 (계약) |
| :--- | :--- | :--- |
| **Model Interface** | `src/models/base.py` | 모든 예측 모델의 **fit/predict/save** 규격 정의 |
| **Validation Engine** | `src/modeling/trainer.py` | 시계열 **Walk-Forward** 학습 로직 전담 |
| **Model Artifact** | `src/models/artifact.py` | 모델 메타데이터 및 버전 **Registry** 관리 |
| **Data Adapter** | `src/data_loader/loader.py` | 데이터 저장소(Parquet)와의 **I/O 책임 분리** |
| **First-Stage Model** | `src/models/lightgbm_model.py` | `ModelBase`를 구현한 LightGBM 예측기 |
| **Selection Logic** | `src/universe/select_universe.py` | 학습용/운영용 **유니버스 선정 로직** 정의 |

## 📂 Directory Structure

```bash
SignalWeaver/
├── config/                   # 설정 및 하이퍼파라미터
├── data/                     # 데이터 저장소
│   ├── 01_raw                    # 원천 데이터
│   ├── 03_processed              # 학습용 피처 매트릭스
│   ├── 04_models                 # 모델 아티팩트 & Registry
│   └── 05_results                # 최종 산출물
├── docs/                     # 아키텍처 문서, 스키마, 변경 이력
│   └── data_schema.md            # 데이터 모듈 간의 계약 문서
│   └── dependency_diagram.svg    # 의존성 다이어그램
├── scripts/                  # 실행 가능한 파이프라인 스크립트
├── src/                      # 소스 코드 모듈
│   ├── data_loader/              # 데이터 로딩 및 저장 어댑터
│   ├── features/                 # 기술적 지표 및 전처리
│   ├── modeling/                 # 학습 루프 및 평가 로직
│   ├── models/                   # 모델 구현체 (Base, LightGBM, etc.)
│   └── universe/                 # 종목 선정 및 리스크 관리
└── README.md
```

## 🚀 Getting Started

### 1\. 환경 설정

```bash
pip install -r requirements.txt
```

### 2\. 파이프라인 실행

전체 파이프라인은 `scripts/` 내의 스크립트를 순차적으로 실행하여 재현할 수 있습니다.

0. **모의 데이터 생성**: 실제 데이터 대신 파이프라인 재현용 데이터(mock data) 생성. **데이터 수집 및 로드**를 대체 가능.
    ```bash
    python scripts/00_generate_synthetic_data.py
    ```
1.  **데이터 수집 및 로드**: KRX/YF 데이터를 수집하여 표준 포맷(Parquet)으로 저장합니다.
    ```bash
    python scripts/01_load_raw_data.py
    ```
2.  **피처 엔지니어링**: 기술적 지표 생성 및 타겟 라벨링을 수행합니다.
    ```bash
    python scripts/02_build_features.py
    ```
3.  **학습 및 예측**: Walk-Forward 방식으로 모델을 학습하고 후보 종목을 선정합니다.
    ```bash
    python scripts/03_train_predict.py
    ```

## 🛠 Tech Stack

  - **Language**: Python 3.9+
  - **Data Processing**: Pandas, NumPy
  - **Machine Learning**: LightGBM
  - **Format**: Parquet (Snappy compression)

## 📝 License

This project is for portfolio purposes.
