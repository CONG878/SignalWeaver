"""
GRUModel — PyTorch 기반 GRU 시계열 예측 모델

## v4.0.0 rev2 변경
- On-the-fly DataLoader 지원: fit()이 Dataset을 직접 받음 (텐서 전달 병행 지원)
- 체크포인트 저장/재개:
    - 에포크마다 val_loss가 개선되면 checkpoint.pt 저장
    - fit(resume=True)이면 checkpoint.pt에서 이어서 학습
    - 학습 완료 후 checkpoint.pt 보존 (resume 시 옵티마이저 상태 복원용)
    - weights.pt는 save()에서 별도 생성 (추론 전용, best_net_state만)
    - resume=True 우선순위: checkpoint.pt → weights.pt(경고) → FileNotFoundError
- 저장 포맷: weights.pt + config.json (pickle 미사용)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.models.seq_base import SeqModelBase


# ──────────────────────────────────────────────────────────────
# 내부 네트워크
# ──────────────────────────────────────────────────────────────

class _GRUNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        forecast_horizon: int,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.directions = 2 if bidirectional else 1
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_size * self.directions, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


# ──────────────────────────────────────────────────────────────
# GRUModel
# ──────────────────────────────────────────────────────────────

class GRUModel(SeqModelBase):
    """
    GRU 시계열 예측 모델.

    Parameters
    ----------
    model_version : str
    params : Dict[str, Any]
        config.yaml의 gru_params 섹션.
    seq_len : int
    forecast_horizon : int
    feature_list : List[str]
    target_type : str
    checkpoint_dir : str or Path, optional
        체크포인트를 저장할 디렉토리. None이면 체크포인트 저장 안 함.
        03c 노트북에서 paths.get_seq_model_dir()을 전달하면 됩니다.
    """

    CHECKPOINT_NAME = "checkpoint.pt"

    def __init__(
        self,
        model_version: str,
        params: Dict[str, Any],
        seq_len: int,
        forecast_horizon: int,
        feature_list: List[str],
        target_type: str = "log_return_1d",
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__(
            model_name="gru",
            model_version=model_version,
            seq_len=seq_len,
            forecast_horizon=forecast_horizon,
            feature_list=feature_list,
            target_type=target_type,
        )
        self.params          = params
        self.hidden_size     = params.get("hidden_size",    128)
        self.num_layers      = params.get("num_layers",       2)
        self.dropout         = params.get("dropout",         0.2)
        self.learning_rate   = params.get("learning_rate", 0.001)
        self.batch_size      = params.get("batch_size",     256)
        self.weight_decay    = params.get("weight_decay",  1e-4)
        self.bidirectional   = params.get("bidirectional", False)
        self.checkpoint_dir  = Path(checkpoint_dir) if checkpoint_dir else None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net: Optional[_GRUNet] = None
        self._val_rmse:   Optional[float] = None
        self._test_rmse:  Optional[float] = None
        self._trained_at: Optional[str]   = None

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------

    def fit(
        self,
        train_dataset: Union[Dataset, np.ndarray],
        y_train: Optional[np.ndarray] = None,
        eval_set: Optional[Any] = None,
        resume: bool = False,
        **kwargs,
    ) -> None:
        """
        GRU 학습.

        Parameters
        ----------
        train_dataset : SeqDataset or np.ndarray
            SeqDataset을 권장 (on-the-fly, 메모리 효율적).
            np.ndarray (N, seq_len, n_features)도 지원 (하위 호환).
        y_train : np.ndarray, optional
            train_dataset이 np.ndarray일 때만 필요.
        eval_set : SeqDataset or (X_val, y_val), optional
            Early Stopping 기준 검증 데이터.
        resume : bool, default=False
            True: checkpoint.pt에서 이어서 학습.
            False: 처음부터 학습 (checkpoint가 있으면 에러).
        **kwargs
            epochs, patience
        """
        epochs   = kwargs.pop("epochs",   self.params.get("epochs",   100))
        patience = kwargs.pop("patience", self.params.get("patience",  10))

        # ── DataLoader 구성 ─────────────────────────────────────
        train_loader = self._make_loader(train_dataset, y_train, shuffle=True)
        val_loader   = self._make_loader_from_eval(eval_set)

        n_features = self._infer_n_features(train_dataset, y_train)

        # ── 네트워크 / 옵티마이저 초기화 ─────────────────────────
        if self.net is None:
            self.net = _GRUNet(
                n_features=n_features,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                forecast_horizon=self.forecast_horizon,
                bidirectional=self.bidirectional,
            ).to(self.device)

        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()

        # ── 체크포인트 처리 ──────────────────────────────────────
        start_epoch    = 1
        best_val_loss  = float("inf")
        best_state     = None
        no_improve     = 0

        ckpt_path = self._checkpoint_path()

        if resume:
            weights_path = (self.checkpoint_dir / "weights.pt") if self.checkpoint_dir else None

            if ckpt_path is not None and ckpt_path.exists():
                # 우선순위 1: checkpoint.pt — 전체 컨텍스트 복원
                ckpt = torch.load(ckpt_path, map_location=self.device)
                self.net.load_state_dict(ckpt["net_state"])
                optimizer.load_state_dict(ckpt["optimizer_state"])
                start_epoch   = ckpt["epoch"] + 1
                best_val_loss = ckpt["best_val_loss"]
                no_improve    = ckpt.get("no_improve", 0)
                best_state    = ckpt["best_net_state"]
                print(f"   ✅ 체크포인트 재개: epoch {start_epoch}부터 / best_val_loss={best_val_loss:.6f}")
            elif weights_path is not None and weights_path.exists():
                # 우선순위 2: weights.pt — 가중치만 복원 (옵티마이저 상태 없음)
                import warnings
                warnings.warn(
                    "checkpoint.pt가 없어 weights.pt에서 가중치만 복원합니다. "
                    "옵티마이저 상태(모멘텀 등)가 리셋되어 초반 학습률이 불안정할 수 있습니다.",
                    UserWarning,
                )
                self.net.load_state_dict(torch.load(weights_path, map_location=self.device))
                print("   ⚠️  weights.pt에서 가중치만 복원 (옵티마이저 상태 없음)")
            else:
                raise FileNotFoundError(
                    f"resume=True로 설정했지만 복원할 파일이 없습니다.\n"
                    f"  checkpoint.pt: {ckpt_path}\n"
                    f"  weights.pt:    {weights_path}\n"
                    "처음부터 학습하려면 resume=False로 설정하세요."
                )
        else:
            if ckpt_path is not None and ckpt_path.exists():
                raise FileNotFoundError(
                    f"resume=False이지만 체크포인트가 이미 존재합니다: {ckpt_path}\n"
                    "이어서 학습하려면 resume=True로, "
                    "처음부터 다시 하려면 체크포인트를 직접 삭제 후 실행하세요."
                )


        # ── 학습 루프 ────────────────────────────────────────────
        for epoch in range(start_epoch, start_epoch + epochs):
            self.net.train()
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                criterion(self.net(X_b), y_b).backward()
                optimizer.step()

            # 검증
            if val_loader is not None:
                val_loss = self._eval_loss(val_loader, criterion)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state    = {k: v.clone() for k, v in self.net.state_dict().items()}
                    no_improve    = 0

                    # 개선될 때마다 체크포인트 저장
                    self._save_checkpoint(
                        ckpt_path, optimizer, epoch,
                        best_val_loss, best_state, no_improve
                    )
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(
                            f"   [GRU] Early stopping — epoch {epoch} | "
                            f"best val_loss: {best_val_loss:.6f}"
                        )
                        break
            else:
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}

        # ── 최선 가중치 복원 및 체크포인트 승격 ──────────────────
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        self.is_fitted   = True
        self._trained_at = datetime.now().isoformat()

        # checkpoint.pt 보존: 다음 resume 시 옵티마이저 상태 복원에 필요
        # weights.pt는 save()에서 별도 생성 (추론 전용, best_net_state만)

    # ------------------------------------------------------------------
    # 예측
    # ------------------------------------------------------------------

    def predict(self, X_seq: np.ndarray, **kwargs) -> np.ndarray:
        self.validate_fitted()
        self.validate_input_shape(X_seq)
        self.net.eval()

        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.net(X_t).cpu().numpy()

    # ------------------------------------------------------------------
    # 저장 / 불러오기
    # ------------------------------------------------------------------

    def save(self, dir_path: str) -> None:
        """weights.pt + config.json 저장."""
        self.validate_fitted()
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.net.state_dict(), path / "weights.pt")

        config = {
            "model_name":       self.model_name,
            "model_version":    self.model_version,
            "seq_len":          self.seq_len,
            "forecast_horizon": self.forecast_horizon,
            "n_features":       len(self.feature_list),
            "feature_list":     self.feature_list,
            "target_type":      self.target_type,
            "target_columns":   self.target_columns,
            "hidden_size":      self.hidden_size,
            "num_layers":       self.num_layers,
            "dropout":          self.dropout,
            "bidirectional":    self.bidirectional,
            "trained_at":       self._trained_at,
            "val_rmse":         self._val_rmse,
            "test_rmse":        self._test_rmse,
        }
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, dir_path: str) -> "GRUModel":
        """디렉토리에서 모델 복원."""
        path = Path(dir_path)

        weights_path = path / "weights.pt"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"weights.pt를 찾을 수 없습니다: {weights_path}\n"
                "학습이 완료되지 않았거나 경로가 잘못되었습니다."
            )

        with open(path / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        inst = cls(
            model_version    = cfg["model_version"],
            params           = {
                "hidden_size":   cfg["hidden_size"],
                "num_layers":    cfg["num_layers"],
                "dropout":       cfg["dropout"],
                "bidirectional": cfg["bidirectional"],
            },
            seq_len          = cfg["seq_len"],
            forecast_horizon = cfg["forecast_horizon"],
            feature_list     = cfg["feature_list"],
            target_type      = cfg["target_type"],
        )
        inst.net = _GRUNet(
            n_features       = cfg["n_features"],
            hidden_size      = cfg["hidden_size"],
            num_layers       = cfg["num_layers"],
            dropout          = cfg["dropout"],
            forecast_horizon = cfg["forecast_horizon"],
            bidirectional    = cfg["bidirectional"],
        ).to(inst.device)

        inst.net.load_state_dict(
            torch.load(weights_path, map_location=inst.device)
        )
        inst.net.eval()
        inst.target_columns = cfg.get("target_columns", inst.target_columns)
        inst._trained_at    = cfg.get("trained_at")
        inst._val_rmse      = cfg.get("val_rmse")
        inst._test_rmse     = cfg.get("test_rmse")
        inst.is_fitted      = True
        return inst

    # ------------------------------------------------------------------
    # 메타
    # ------------------------------------------------------------------

    def get_meta(self) -> Dict[str, Any]:
        return {
            "model_name":       self.model_name,
            "model_version":    self.model_version,
            "seq_len":          self.seq_len,
            "forecast_horizon": self.forecast_horizon,
            "feature_list":     self.feature_list,
            "target_type":      self.target_type,
            "target_columns":   self.target_columns,
            "hidden_size":      self.hidden_size,
            "num_layers":       self.num_layers,
            "dropout":          self.dropout,
            "bidirectional":    self.bidirectional,
            "trained_at":       self._trained_at,
            "val_rmse":         self._val_rmse,
            "test_rmse":        self._test_rmse,
            "is_fitted":        self.is_fitted,
        }

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _make_loader(
        self,
        dataset: Union[Dataset, np.ndarray],
        y: Optional[np.ndarray],
        shuffle: bool,
    ) -> DataLoader:
        if isinstance(dataset, Dataset):
            ds = dataset
        else:
            # np.ndarray fallback
            mask = ~(np.isnan(dataset).any(axis=(1, 2)) | np.isnan(y).any(axis=1))
            X_t = torch.tensor(dataset[mask], dtype=torch.float32)
            y_t = torch.tensor(y[mask],       dtype=torch.float32)
            ds  = TensorDataset(X_t, y_t)

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=len(ds) >= self.batch_size,
        )

    def _make_loader_from_eval(self, eval_set) -> Optional[DataLoader]:
        if eval_set is None:
            return None
        if isinstance(eval_set, Dataset):
            return DataLoader(eval_set, batch_size=self.batch_size, shuffle=False)
        # tuple (X_val, y_val)
        X_val, y_val = eval_set
        mask = ~(np.isnan(X_val).any(axis=(1, 2)) | np.isnan(y_val).any(axis=1))
        return DataLoader(
            TensorDataset(
                torch.tensor(X_val[mask], dtype=torch.float32),
                torch.tensor(y_val[mask], dtype=torch.float32),
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )

    def _infer_n_features(self, dataset, y) -> int:
        if isinstance(dataset, Dataset):
            x0, _ = dataset[0]
            return x0.shape[-1]
        return dataset.shape[2]

    def _checkpoint_path(self) -> Optional[Path]:
        if self.checkpoint_dir is None:
            return None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self.checkpoint_dir / self.CHECKPOINT_NAME

    def _save_checkpoint(self, path, optimizer, epoch,
                         best_val_loss, best_state, no_improve):
        if path is None:
            return
        torch.save({
            "epoch":           epoch,
            "net_state":       self.net.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss":   best_val_loss,
            "best_net_state":  best_state,
            "no_improve":      no_improve,
        }, path)

    def _eval_loss(self, loader: DataLoader, criterion: nn.Module) -> float:
        self.net.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                loss = criterion(self.net(X_b), y_b)
                total += loss.item() * len(X_b)
                n     += len(X_b)
        return total / n if n > 0 else float("inf")
