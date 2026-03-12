"""TabuLa-based synthetic data generator."""

from __future__ import annotations
 
import os
import subprocess
import sys
from pathlib import Path
 
import pandas as pd
 
from src.generators.base import BaseGenerator
 
_TABULA_REPO = "https://github.com/zhao-zilong/Tabula"
_TABULA_DIR  = Path(__file__).resolve().parent.parent.joinpath("models").joinpath("tabula")
 
 
def _ensure_tabula():
    if not _TABULA_DIR.exists():
        subprocess.run(
            ["git", "clone", _TABULA_REPO, str(_TABULA_DIR)],
            check=True,
        )
    if str(_TABULA_DIR) not in sys.path:
        sys.path.insert(0, str(_TABULA_DIR))
 
 
class TabulaGenerator(BaseGenerator):
 
    def fit(self, df: pd.DataFrame, model_dir: Path) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = self.params.get("cuda_device", "0")
        _ensure_tabula()
        from tabula.tabula import Tabula  # noqa: PLC0415
 
        model_dir.mkdir(parents=True, exist_ok=True)
        model = Tabula(
            llm=self.params["llm"],
            batch_size=self.params["batch_size"],
            epochs=self.params["epochs"],
            max_length=self.params["max_length"],
        )
        model.fit(df)
        model.save(str(model_dir.joinpath("tabula_model")))
 
    def sample(self, n_samples: int, model_dir: Path) -> pd.DataFrame:
        os.environ["CUDA_VISIBLE_DEVICES"] = self.params.get("cuda_device", "0")
        _ensure_tabula()
        from tabula.tabula import Tabula  # noqa: PLC0415
 
        model = Tabula.load(str(model_dir.joinpath("tabula_model")))
        return model.sample(n_samples)