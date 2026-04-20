"""
.. deprecated:: v4.1.0
    이 모듈은 ``src/utils/integration.py`` 로 대체되었습니다.
    하위 호환성을 위해 유지되며, 향후 버전에서 제거될 예정입니다.
"""
from src.utils.integration import trapezoid_log_close  # noqa: F401

__all__ = ["trapezoid_log_close"]