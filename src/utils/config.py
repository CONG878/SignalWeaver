"""
Configuration management module

✨ H2 패치: 경로 관리 중앙화
- ProjectPaths 클래스 추가
- 모든 노트북에서 일관된 경로 사용
- 단계별 독립 경로 지원 (H1 패치 반영)

✨ v3.8.0 패치: 앙상블 확장성
- 모델명 해석 헬퍼 함수 추가 (parse_active_model, get_folder_name 등)
- active_model에 '+' 구분자로 앙상블 조합 지정 지원
- ProjectPaths.training_dir이 get_folder_name() 기반으로 동적 결정됨
"""

import yaml
from pathlib import Path
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass


# ==========================================
# 기본 설정 로드
# ==========================================

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """YAML 설정 파일을 로드합니다."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_path(base_path: str, reference_date: str) -> Path:
    """기준일이 포함된 경로를 생성합니다."""
    return Path(base_path) / f"dataset_{reference_date}.parquet"


# ==========================================
# ✨ v3.8.0: 모델명 해석 헬퍼
# ==========================================

# 허용 입력값 → (canonical, short) 매핑 테이블
# canonical : 단일 모델 폴더명, 하위 호환 정칭
# short      : 앙상블 폴더명 구성용 약칭
_MODEL_ALIAS: Dict[str, Tuple[str, str]] = {
    "lightgbm":    ("lightgbm",    "lgbm"),
    "lgbm":        ("lightgbm",    "lgbm"),
    "randomforest": ("randomforest", "rf"),
    "rf":          ("randomforest", "rf"),
    "mlp":         ("mlp",          "mlp"),
}


def resolve_model_name(name: str) -> Tuple[str, str]:
    """
    단일 모델 이름(정칭 또는 약칭)을 (canonical, short) 튜플로 정규화.

    Parameters
    ----------
    name : str
        예: "lightgbm", "lgbm", "randomforest", "rf", "mlp"

    Returns
    -------
    (canonical, short) : Tuple[str, str]
        canonical — 단일 모델 폴더명 / 하위 호환 정칭
        short     — 앙상블 폴더명 구성에 사용할 약칭

    Raises
    ------
    ValueError
        알 수 없는 모델명인 경우
    """
    key = name.strip().lower()
    if key not in _MODEL_ALIAS:
        allowed = sorted(set(_MODEL_ALIAS.keys()))
        raise ValueError(
            f"알 수 없는 모델명: '{name}'\n"
            f"허용값: {allowed}"
        )
    return _MODEL_ALIAS[key]


def parse_active_model(active_model_str: str) -> List[Tuple[str, str]]:
    """
    active_model 문자열을 파싱하여 (canonical, short) 튜플 리스트 반환.

    단일 모델과 앙상블('+' 구분) 모두 처리.

    Parameters
    ----------
    active_model_str : str
        예: "lightgbm", "lgbm+rf", "lgbm+rf+mlp"

    Returns
    -------
    List[Tuple[str, str]]
        예: [("lightgbm", "lgbm"), ("randomforest", "rf")]

    Examples
    --------
    >>> parse_active_model("lightgbm")
    [("lightgbm", "lgbm")]

    >>> parse_active_model("lgbm+rf")
    [("lightgbm", "lgbm"), ("randomforest", "rf")]

    >>> parse_active_model("lgbm+rf+mlp")
    [("lightgbm", "lgbm"), ("randomforest", "rf"), ("mlp", "mlp")]
    """
    parts = [p.strip() for p in active_model_str.split("+")]
    return [resolve_model_name(p) for p in parts]


def is_ensemble(active_model_str: str) -> bool:
    """
    active_model 문자열이 앙상블(2개 이상 모델 조합)인지 판단.

    Parameters
    ----------
    active_model_str : str
        예: "lightgbm", "lgbm+rf"

    Returns
    -------
    bool
        True if ensemble (2+), False if single model
    """
    return "+" in active_model_str


def get_folder_name(active_model_str: str) -> str:
    """
    active_model 문자열로부터 폴더명을 결정.

    - 단일 모델: canonical 정칭 사용 (하위 호환)
    - 앙상블:    short 약칭을 '+' 로 연결

    Parameters
    ----------
    active_model_str : str
        예: "lightgbm", "lgbm+rf", "lightgbm+randomforest+mlp"

    Returns
    -------
    str
        폴더명 문자열
        예: "lightgbm", "lgbm+rf", "lgbm+rf+mlp"

    Examples
    --------
    >>> get_folder_name("lightgbm")
    "lightgbm"

    >>> get_folder_name("lgbm+rf")
    "lgbm+rf"

    >>> get_folder_name("lightgbm+randomforest+mlp")
    "lgbm+rf+mlp"
    """
    models = parse_active_model(active_model_str)
    if len(models) == 1:
        canonical, _ = models[0]
        return canonical          # 단일 모델: 정칭 (하위 호환)
    else:
        return "+".join(short for _, short in models)  # 앙상블: 약칭 조합


# ==========================================
# ✨ H2 패치: ProjectPaths 클래스
# ==========================================

@dataclass
class ProjectPaths:
    """
    프로젝트 전체 경로를 관리하는 중앙화 클래스

    사용 예:
    >>> paths = ProjectPaths.from_config(cfg)
    >>> model_path = paths.get_model_path("lightgbm_multi")
    >>> forecast_output = paths.forecasts_dir

    특징:
    - H1 패치 반영: 단계별 독립 경로
    - 하위 호환성 유지: 기존 단일 모델 코드도 동작
    - 타입 안전: IDE 자동완성 지원
    - ✨ v3.8.0: training_dir이 get_folder_name() 기반으로 동적 결정됨
      예) active_model="lgbm+rf" → training_dir = .../lgbm+rf/
    """

    reference_date: str   # 기준일
    active_model: str     # 현재 활성화된 모델 (원본 문자열, 예: "lgbm+rf")
    folder_name: str      # 폴더명 (예: "lightgbm", "lgbm+rf")

    # 기본 경로
    raw_dir: Path
    processed_dir: Path
    meta_dir: Path

    # 단계별 경로 (H1 패치)
    training_dir: Path      # Step 3
    forecasts_dir: Path     # Step 4
    universe_dir: Path      # Step 5

    @classmethod
    def from_config(cls, config: Dict[str, Any], reference_date: str = None) -> "ProjectPaths":
        """
        config.yaml에서 ProjectPaths 객체 생성.

        Parameters
        ----------
        config : dict
            load_config()의 반환값
        reference_date : str, optional
            기준일 (없으면 config에서 추출)

        Returns
        -------
        ProjectPaths
            초기화된 경로 객체
        """
        ref_date = reference_date or config['project']['reference_date']
        active_model_str = config.get('active_model', 'lightgbm')
        folder = get_folder_name(active_model_str)
        paths_cfg = config['paths']
        model_date = config['universe']['model_date']

        return cls(
            reference_date=ref_date,
            active_model=active_model_str,
            folder_name=folder,

            # 기본 경로
            raw_dir=Path(paths_cfg['raw_dir']) / ref_date,
            processed_dir=Path(paths_cfg['processed_dir']) / ref_date,
            meta_dir=Path(paths_cfg['meta_dir']),

            # 단계별 경로
            # ✨ v3.8.0: folder_name 기반 동적 결정
            training_dir=Path(paths_cfg['training_dir']) / model_date / folder,
            forecasts_dir=Path(paths_cfg['forecasts_dir']) / ref_date / folder,
            universe_dir=Path(paths_cfg['universe_dir']) / ref_date / folder,
        )

    # ==========================================
    # Step 1: Raw Data
    # ==========================================

    def get_raw_parquet(self) -> Path:
        """통합 Raw Parquet 경로"""
        return self.raw_dir / f"krx_prices_{self.reference_date}.parquet"

    def get_ticker_master(self) -> Path:
        """종목 마스터 경로"""
        return self.raw_dir / f"ticker_master_{self.reference_date}.csv"

    def get_raw_csv_dir(self) -> Path:
        """개별 CSV 디렉토리"""
        return self.raw_dir / "csv"

    # ==========================================
    # Step 2: Processed Data
    # ==========================================

    def get_dataset_parquet(self) -> Path:
        """통합 Feature 데이터셋 경로"""
        return self.processed_dir / "dataset.parquet"

    def get_processed_csv_dir(self) -> Path:
        """개별 CSV 디렉토리"""
        return self.processed_dir / "csv"

    # ==========================================
    # Step 3: Training (학습 검증 예측)
    # ==========================================

    def get_predictions_parquet(self) -> Path:
        """학습 검증용 예측 결과 (과거 데이터)"""
        return self.training_dir / "predictions.parquet"

    def get_predictions_csv_dir(self) -> Path:
        """개별 예측 CSV 디렉토리"""
        return self.training_dir / "csv"

    def get_model_dir(self) -> Path:
        """모델 저장 디렉토리"""
        return self.training_dir

    def get_model_path(self, pattern: str = "*.pkl") -> Path:
        """특정 모델 파일 경로 검색"""
        model_dir = self.get_model_dir()
        files = list(model_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(
                f"모델 파일을 찾을 수 없습니다: {model_dir}/{pattern}"
            )

        return files[0]

    def get_member_model_dir(self, model_name_or_alias: str) -> Path:
        """
        앙상블 구성 모델 각각의 폴더 경로 반환.

        앙상블 운용 시 개별 모델의 val/test predictions를 읽을 때 사용.

        Parameters
        ----------
        model_name_or_alias : str
            예: "lightgbm", "lgbm", "randomforest", "rf", "mlp"

        Returns
        -------
        Path
            예: data/03_training/{model_date}/lightgbm/
        """
        canonical, _ = resolve_model_name(model_name_or_alias)
        # training_dir 상위(model_date 레벨)에서 canonical 폴더를 참조
        return self.training_dir.parent / canonical

    # ==========================================
    # Step 4: Forecasts (미래 예측)
    # ==========================================

    def get_forecasts_parquet(self) -> Path:
        """미래 예측 결과"""
        return self.forecasts_dir / "future_forecasts.parquet"

    def get_forecasts_csv_dir(self) -> Path:
        """개별 예측 CSV 디렉토리"""
        return self.forecasts_dir / "csv"

    # ==========================================
    # Step 5: Universe (최종 선정)
    # ==========================================

    def get_universe_full(self) -> Path:
        """전체 평가 완료 Universe"""
        return self.universe_dir / "universe_full.parquet"

    def get_universe_candidates(self) -> Path:
        """Top-K 후보"""
        return self.universe_dir / "universe_candidates.parquet"

    def get_investment_report_csv(self) -> Path:
        """투자 리포트 (CSV)"""
        return self.universe_dir / "investment_report.csv"

    def get_investment_report_excel(self) -> Path:
        """투자 리포트 (Excel)"""
        return self.universe_dir / "investment_report.xlsx"

    def get_filter_statistics(self) -> Path:
        """필터링 통계 (JSON)"""
        return self.universe_dir / "filter_statistics.json"

    # ==========================================
    # Meta & Utilities
    # ==========================================

    def get_calendar(self) -> Path:
        """영업일 캘린더"""
        return self.meta_dir / "krx_calendar.csv"

    def ensure_dirs(self):
        """
        모든 출력 디렉토리 생성.
        노트북 시작 시 호출하여 디렉토리 자동 생성.
        """
        dirs_to_create = [
            self.raw_dir,
            self.processed_dir,
            self.training_dir,
            self.forecasts_dir,
            self.universe_dir,
            self.meta_dir,
        ]

        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        """디버깅용 출력"""
        return (
            f"ProjectPaths(\n"
            f"  reference_date='{self.reference_date}'\n"
            f"  active_model='{self.active_model}'  →  folder='{self.folder_name}'\n"
            f"  raw={self.raw_dir}\n"
            f"  processed={self.processed_dir}\n"
            f"  training={self.training_dir}\n"
            f"  forecasts={self.forecasts_dir}\n"
            f"  universe={self.universe_dir}\n"
            f")"
        )
