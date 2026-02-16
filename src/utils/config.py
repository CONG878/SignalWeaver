"""
Configuration management module

✨ H2 패치: 경로 관리 중앙화
- ProjectPaths 클래스 추가
- 모든 노트북에서 일관된 경로 사용
- 단계별 독립 경로 지원 (H1 패치 반영)
"""

import yaml
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """YAML 설정 파일을 로드합니다."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_path(base_path: str, reference_date: str) -> Path:
    """기준일이 포함된 경로를 생성합니다."""
    return Path(base_path) / f"dataset_{reference_date}.parquet"


# ==========================================
# ✨ H2 패치: ProjectPaths 클래스
# ==========================================

@dataclass
class ProjectPaths:
    """
    프로젝트 전체 경로를 관리하는 중앙화 클래스
    
    사용 예:
    >>> paths = ProjectPaths.from_config(cfg, ref_date)
    >>> model_path = paths.get_model_path("lightgbm_multi")
    >>> forecast_output = paths.forecasts.output_dir
    
    특징:
    - H1 패치 반영: 단계별 독립 경로
    - 하위 호환성 유지: 기존 코드도 동작
    - 타입 안전: IDE 자동완성 지원
    """
    
    reference_date: str # 기준일
    active_model: str # 현재 활성화된 모델 
    
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
        config.yaml에서 ProjectPaths 객체 생성
        
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
        active_model = config.get('active_model', 'lightgbm') # 기본값 lgbm
        paths_cfg = config['paths']
        
        return cls(
            reference_date=ref_date,
            active_model=active_model,
            
            # 기본 경로
            raw_dir=Path(paths_cfg['raw_dir']) / ref_date,
            processed_dir=Path(paths_cfg['processed_dir']) / ref_date,
            meta_dir=Path(paths_cfg['meta_dir']),
            
            # 단계별 경로
            training_dir=Path(paths_cfg['training_dir']) / config['universe']['model_date'] / active_model,
            forecasts_dir=Path(paths_cfg['forecasts_dir']) / ref_date / active_model,
            universe_dir=Path(paths_cfg['universe_dir']) / ref_date / active_model,
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
        """모델 저장 디렉토리 (모델 인자 제거)"""
        return self.training_dir
    
    def get_model_path(self, pattern: str = "*.pkl") -> Path:
        """특정 모델 파일 경로 검색 (인자 제거됨)"""
        model_dir = self.get_model_dir()
        files = list(model_dir.glob(pattern))
        
        if not files:
            raise FileNotFoundError(
                f"모델 파일을 찾을 수 없습니다: {model_dir}/{pattern}"
            )
        
        return files[0]
    
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
        모든 출력 디렉토리 생성
        
        노트북 시작 시 호출하여 디렉토리 자동 생성
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
            f"  raw={self.raw_dir}\n"
            f"  processed={self.processed_dir}\n"
            f"  training={self.training_dir}\n"
            f"  forecasts={self.forecasts_dir}\n"
            f"  universe={self.universe_dir}\n"
            f")"
        )


# ==========================================
# 사용 예시
# ==========================================

"""
# 기본 사용
from src.utils.config import load_config, ProjectPaths

cfg = load_config()
paths = ProjectPaths.from_config(cfg)

# Step 1
raw_data = pd.read_parquet(paths.get_raw_parquet())
ticker_master = pd.read_csv(paths.get_ticker_master())

# Step 3
model = LightGBMModel.load(str(paths.get_model_path("lightgbm_multi")))
predictions = pd.read_parquet(paths.get_predictions_parquet())

# Step 4
forecasts = pd.read_parquet(paths.get_forecasts_parquet())

# Step 5
candidates = pd.read_parquet(paths.get_universe_candidates())

# 디렉토리 자동 생성
paths.ensure_dirs()
"""
