"""
CSV 저장 유틸리티 — 종목별 개별 파일 저장 공통 모듈

## 설계 원칙
- 파이프라인 01~04단계의 종목별 CSV 저장을 단일 인터페이스로 통일
- config.yaml의 output.save_csv.stage_XX 플래그로 단계별 활성화 제어
- 05단계(Universe 선정)는 전 종목 대상이 아니므로 이 모듈 적용 대상 제외

## 단계별 저장 경로
    01  data/01_raw/{ref_date}/csv/
    02  data/02_processed/{ref_date}/csv/
    03  data/03_training/{model_date}/{model_name}/csv/   (Tabular)
        data/03_seq/{model_date}/{seq_model}/csv/         (Seq)
    04  data/04_forecasts/{ref_date}/{model_name}/csv/
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from tqdm import tqdm


def save_ticker_csv(
    df: pd.DataFrame,
    output_dir: Path,
    ticker_name_map: Optional[Dict[str, str]] = None,
    ticker_col: str = "ticker",
    encoding: str = "utf-8-sig",
    desc: str = "Saving CSVs",
) -> None:
    """
    DataFrame을 종목별로 분할하여 개별 CSV 파일로 저장합니다.

    Parameters
    ----------
    df : pd.DataFrame
        저장할 데이터. ticker_col 컬럼이 반드시 포함되어야 합니다.
    output_dir : Path
        저장 디렉토리. 존재하지 않으면 자동 생성합니다.
    ticker_name_map : dict, optional
        {ticker: name} 형태의 종목코드-종목명 매핑.
        지정하지 않으면 ticker 코드를 그대로 파일명으로 사용합니다.
    ticker_col : str
        종목 구분 컬럼명. 기본값 'ticker'.
    encoding : str
        CSV 인코딩. 기본값 'utf-8-sig' (Excel 한글 호환).
    desc : str
        tqdm 진행바에 표시할 설명 문자열.

    Examples
    --------
    >>> from src.utils.export import save_ticker_csv
    >>> save_ticker_csv(df_predictions, paths.get_predictions_csv_dir(),
    ...                 ticker_name_map, desc="Saving prediction CSVs")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ticker, group in tqdm(df.groupby(ticker_col), desc=desc):
        name = (
            ticker_name_map.get(str(ticker), str(ticker))
            if ticker_name_map
            else str(ticker)
        )
        safe_name = str(name).replace("/", "_").replace("\\", "_")
        group.to_csv(output_dir / f"{safe_name}.csv", index=False, encoding=encoding)


def should_save_csv(config: dict, stage: int) -> bool:
    """
    config.yaml의 output.save_csv.stage_XX 값을 읽어 CSV 저장 여부를 반환합니다.

    Parameters
    ----------
    config : dict
        load_config()로 로드한 설정 딕셔너리.
    stage : int
        파이프라인 단계 번호 (1~4).

    Returns
    -------
    bool
        True이면 해당 단계에서 종목별 CSV를 저장합니다.

    Examples
    --------
    >>> should_save_csv(cfg, 1)   # output.save_csv.stage_01 → True
    True
    >>> should_save_csv(cfg, 3)   # output.save_csv.stage_03 → False
    False
    """
    return bool(
        config.get("output", {})
              .get("save_csv", {})
              .get(f"stage_{stage:02d}", False)
    )


def load_ticker_name_map(paths) -> Dict[str, str]:
    """
    ticker_master.csv에서 {ticker: name} 매핑을 로드합니다.
    파일이 없거나 로드 실패 시 빈 dict를 반환합니다.

    Parameters
    ----------
    paths : ProjectPaths
        ProjectPaths 인스턴스.

    Returns
    -------
    dict
        {ticker_str: name_str} 매핑.
    """
    try:
        import pandas as pd
        df_master = pd.read_csv(paths.get_ticker_master())
        return dict(zip(df_master["ticker"].astype(str), df_master["name"]))
    except Exception:
        return {}
