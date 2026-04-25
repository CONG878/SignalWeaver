"""
Configuration & Path Management

YAML 설정 파일 로드, 경로 중앙 관리, 모델명 정규화를 담당합니다.

## 버전
- v4.1.0: get_seq_csv_dir() 추가 (Seq 트랙 종목별 CSV 저장 경로).
- v4.0.0: Seq 트랙 경로 추가, is_seq_model() 신설,
          paths.model_dir 중복 제거, get_test_predictions_parquet() 명칭 명확화,
          파일명에서 날짜 중복 제거 (prices.parquet, ticker_master.csv),
          99_meta → 00_prep 재편:
            prep_dir       (data/00_prep/)            시간 독립 파일
            prep_dated_dir (data/00_prep/{ref_date}/) 날짜 의존 파일
- v3.8.0: ProjectPaths 클래스 도입으로 경로 관리 중앙화
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


# ==========================================
# 모델명 해석 헬퍼 함수
# ==========================================

_MODEL_ALIAS: Dict[str, Tuple[str, str]] = {
    "lightgbm":     ("lightgbm",     "lgbm"),
    "lgbm":         ("lightgbm",     "lgbm"),
    "randomforest": ("randomforest", "rf"),
    "rf":           ("randomforest", "rf"),
    "mlp":          ("mlp",          "mlp"),
    # v4.0.0: seq 모델
    "gru":          ("gru",          "gru"),
    "lstm":         ("lstm",         "lstm"),
}

_SEQ_MODEL_NAMES: frozenset = frozenset({"gru", "lstm"})


def resolve_model_name(name: str) -> Tuple[str, str]:
    """단일 모델 이름을 (canonical, short) 튜플로 정규화."""
    key = name.strip().lower()
    if key not in _MODEL_ALIAS:
        allowed = sorted(set(_MODEL_ALIAS.keys()))
        raise ValueError(f"알 수 없는 모델명: '{name}'\n허용값: {allowed}")
    return _MODEL_ALIAS[key]


def parse_active_model(active_model_str: str) -> List[Tuple[str, str]]:
    """active_model 문자열을 (canonical, short) 튜플 리스트로 파싱."""
    parts = [p.strip() for p in active_model_str.split("+")]
    return [resolve_model_name(p) for p in parts]


def is_ensemble(active_model_str: str) -> bool:
    """active_model이 앙상블(2개 이상)인지 판단."""
    return "+" in active_model_str


def is_seq_model(active_model_str: str) -> bool:
    """
    active_model이 순수 seq 모델 단독인지 판단. is_ensemble()과 대칭.

    Examples
    --------
    >>> is_seq_model("gru")      → True
    >>> is_seq_model("lgbm")     → False
    >>> is_seq_model("lgbm+gru") → False  (혼합 앙상블, v4.1.0에서 처리)
    """
    parts = [p.strip().lower() for p in active_model_str.split("+")]
    return all(p in _SEQ_MODEL_NAMES for p in parts)


def get_folder_name(active_model_str: str) -> str:
    """active_model 문자열로부터 폴더명 결정."""
    models = parse_active_model(active_model_str)
    if len(models) == 1:
        canonical, _ = models[0]
        return canonical
    return "+".join(short for _, short in models)


# ==========================================
# v4.0.0: ProjectPaths 클래스
# ==========================================

@dataclass
class ProjectPaths:
    """
    프로젝트 경로 관리 (중앙화)

    v4.1.0 변경 사항:
    - get_seq_csv_dir() 추가 (Seq 트랙 종목별 CSV 저장 경로)

    v4.0.0 변경 사항:
    - seq_dir / get_seq_*() 추가 (Seq 트랙 전용)
    - model_dir 제거 (training_dir으로 통일)
    - get_raw_parquet(): prices.parquet (날짜 중복 제거)
    - get_ticker_master(): ticker_master.csv (날짜 중복 제거)
    - get_test_predictions_parquet(): 명칭 명확화
    - 99_meta → 00_prep 재편
        prep_dir       data/00_prep/            시간 독립 파일 (krx_calendar.csv)
        prep_dated_dir data/00_prep/{ref_date}/ 날짜 의존 파일 (macro_regime 등)
    """

    reference_date: str
    active_model:   str
    folder_name:    str

    # 기본 경로
    raw_dir:       Path
    processed_dir: Path

    # 00_prep — v4.0.0
    prep_dir:       Path   # data/00_prep/
    prep_dated_dir: Path   # data/00_prep/{ref_date}/

    # 단계별 경로
    training_dir:  Path    # Tabular 트랙 (Step 3a)
    seq_dir:       Path    # Seq 트랙    (Step 3c)
    forecasts_dir: Path    # Step 4
    universe_dir:  Path    # Step 5

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], reference_date: str = None
    ) -> "ProjectPaths":
        """config.yaml에서 ProjectPaths 객체 생성."""
        ref_date         = reference_date or config['project']['reference_date']
        active_model_str = config.get('active_model', 'lightgbm')
        folder           = get_folder_name(active_model_str)
        paths_cfg        = config['paths']
        model_date       = config['universe']['model_date']

        seq_model_name = (
            folder if is_seq_model(active_model_str)
            else config.get('active_seq_model', 'gru')
        )

        prep_base = Path(paths_cfg.get('prep_dir', 'data/00_prep'))

        return cls(
            reference_date = ref_date,
            active_model   = active_model_str,
            folder_name    = folder,

            raw_dir        = Path(paths_cfg['raw_dir'])       / ref_date,
            processed_dir  = Path(paths_cfg['processed_dir']) / ref_date,

            prep_dir       = prep_base,
            prep_dated_dir = prep_base / ref_date,

            training_dir   = Path(paths_cfg['training_dir']) / model_date / folder,
            seq_dir        = Path(paths_cfg.get('seq_dir', 'data/03_seq')) / model_date / seq_model_name,
            forecasts_dir  = Path(paths_cfg['forecasts_dir']) / ref_date / folder,
            universe_dir   = Path(paths_cfg['universe_dir'])  / ref_date / folder,
        )

    # ──────────────────────────────────────────
    # Step 1: Raw Data
    # ──────────────────────────────────────────

    def get_raw_parquet(self) -> Path:
        """통합 Raw Parquet. v4.0.0: 날짜 중복 제거."""
        return self.raw_dir / "prices.parquet"

    def get_ticker_master(self) -> Path:
        """종목 마스터. v4.0.0: 날짜 중복 제거."""
        return self.raw_dir / "ticker_master.csv"

    def get_raw_csv_dir(self) -> Path:
        return self.raw_dir / "csv"

    # ──────────────────────────────────────────
    # Step 2: Processed Data
    # ──────────────────────────────────────────

    def get_dataset_parquet(self) -> Path:
        return self.processed_dir / "dataset.parquet"

    def get_processed_csv_dir(self) -> Path:
        return self.processed_dir / "csv"

    # ──────────────────────────────────────────
    # Step 3a: Tabular Training
    # ──────────────────────────────────────────

    def get_val_predictions_parquet(self) -> Path:
        return self.training_dir / "val_predictions.parquet"

    def get_test_predictions_parquet(self) -> Path:
        """v4.0.0: get_predictions_parquet() → get_test_predictions_parquet()."""
        return self.training_dir / "test_predictions.parquet"

    def get_predictions_csv_dir(self) -> Path:
        return self.training_dir / "csv"

    def get_model_dir(self) -> Path:
        return self.training_dir

    def get_model_path(self, pattern: str = "*.pkl") -> Path:
        files = list(self.training_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"모델 파일을 찾을 수 없습니다: {self.training_dir}/{pattern}"
            )
        return files[0]

    def get_member_model_dir(self, model_name_or_alias: str) -> Path:
        canonical, _ = resolve_model_name(model_name_or_alias)
        return self.training_dir.parent / canonical

    # ──────────────────────────────────────────
    # Step 3c: Seq Training
    # ──────────────────────────────────────────

    def get_seq_model_dir(self) -> Path:
        """Seq 모델 디렉토리 (weights.pt, config.json)."""
        return self.seq_dir

    def get_seq_val_predictions(self) -> Path:
        return self.seq_dir / "val_predictions.parquet"

    def get_seq_test_predictions(self) -> Path:
        return self.seq_dir / "test_predictions.parquet"

    def get_seq_csv_dir(self) -> Path:
        """v4.1.0: Seq 트랙 종목별 CSV 저장 디렉토리."""
        return self.seq_dir / "csv"

    # ──────────────────────────────────────────
    # Step 4: Forecasts
    # ──────────────────────────────────────────

    def get_forecasts_parquet(self) -> Path:
        return self.forecasts_dir / "future_forecasts.parquet"

    def get_forecasts_csv_dir(self) -> Path:
        return self.forecasts_dir / "csv"

    # ──────────────────────────────────────────
    # Step 5: Universe
    # ──────────────────────────────────────────

    def get_universe_full(self) -> Path:
        return self.universe_dir / "universe_full.parquet"

    def get_universe_candidates(self) -> Path:
        return self.universe_dir / "universe_candidates.parquet"

    def get_investment_report_csv(self) -> Path:
        return self.universe_dir / "investment_report.csv"

    def get_investment_report_excel(self) -> Path:
        return self.universe_dir / "investment_report.xlsx"

    def get_filter_statistics(self) -> Path:
        return self.universe_dir / "filter_statistics.json"

    # ──────────────────────────────────────────
    # Step 00: Prep Data — v4.0.0
    # ──────────────────────────────────────────

    def get_calendar(self) -> Path:
        """
        KRX 영업일 달력. 시간 독립 파일 → prep_dir 직접 위치.
        긴 기간으로 한 번 생성하고 코드에서 필요한 범위를 필터링하여 사용합니다.
        """
        return self.prep_dir / "krx_calendar.csv"

    def get_macro_parquet(self) -> Path:
        """매크로 실측 데이터. 날짜 의존 → prep_dated_dir/{ref_date}/ 하위."""
        return self.prep_dated_dir / "macro_regime.parquet"

    def get_macro_csv(self) -> Path:
        """매크로 실측 데이터. CSV 양식."""
        return self.prep_dated_dir / "macro_regime.csv"

    def get_macro_forecast_parquet(self) -> Path:
        """매크로 미래 추정값. 날짜 의존 → prep_dated_dir/{ref_date}/ 하위."""
        return self.prep_dated_dir / "macro_regime_forecast.parquet"

    def get_macro_forecast_csv(self) -> Path:
        """매크로 미래 추정값. CSV 양식."""
        return self.prep_dated_dir / "macro_regime_forecast.csv"

    # ──────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────

    def ensure_dirs(self):
        """모든 출력 디렉토리 생성."""
        for dir_path in [
            self.raw_dir,
            self.processed_dir,
            self.prep_dir,
            self.prep_dated_dir,
            self.training_dir,
            self.seq_dir,
            self.forecasts_dir,
            self.universe_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"ProjectPaths(\n"
            f"  reference_date='{self.reference_date}'\n"
            f"  active_model='{self.active_model}'  →  folder='{self.folder_name}'\n"
            f"  raw={self.raw_dir}\n"
            f"  processed={self.processed_dir}\n"
            f"  prep={self.prep_dir}\n"
            f"  prep_dated={self.prep_dated_dir}\n"
            f"  training={self.training_dir}\n"
            f"  seq={self.seq_dir}\n"
            f"  forecasts={self.forecasts_dir}\n"
            f"  universe={self.universe_dir}\n"
            f")"
        )
