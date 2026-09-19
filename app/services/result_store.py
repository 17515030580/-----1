# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional

import pandas as pd

from app.utils.json_utils import json_safe
from app.utils.prediction_id import normalize_prediction_id


class ResultStore:
    def __init__(self, root: Path, enabled: bool = True) -> None:
        self.root = Path(root).resolve()
        self.enabled = enabled
        self.root.mkdir(parents=True, exist_ok=True)
        # When persistence is off, 64 recent live tasks remain available in RAM.
        self._recent: OrderedDict = OrderedDict()
        self._lock = RLock()

    def _result_dir(self, prediction_id: Any) -> Path:
        task_id = normalize_prediction_id(prediction_id)
        if not task_id or task_id in {".", ".."} or any(c in task_id for c in '/\\\\:\x00'):
            raise ValueError("prediction_id 不是有效的任务编号。")
        result_dir = (self.root / task_id).resolve()
        if result_dir.parent != self.root:
            raise ValueError("prediction_id 不允许指向结果目录之外。")
        return result_dir

    def load(self, prediction_id: Any) -> Optional[Dict[str, Any]]:
        """Use the SAME store root as /predict, not the process cwd."""
        result_dir = self._result_dir(prediction_id)
        task_id = result_dir.name
        with self._lock:
            if task_id in self._recent:
                return deepcopy(self._recent[task_id])
        path = result_dir / "prediction_result.json"
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("预测结果 JSON 必须是对象。")
        stored_id = normalize_prediction_id(payload.get("prediction_id"))
        if stored_id is not None and stored_id != task_id:
            raise ValueError("预测结果文件中的 prediction_id 与请求不一致。")
        payload["prediction_id"] = task_id
        return payload

    def save(self, prediction_id: str, payload: Dict[str, Any]) -> Dict[str, str]:
        result_dir = self._result_dir(prediction_id)
        prediction_id = result_dir.name
        safe_payload = json_safe(payload)
        safe_payload["prediction_id"] = prediction_id
        if not self.enabled:
            with self._lock:
                self._recent[prediction_id] = deepcopy(safe_payload)
                self._recent.move_to_end(prediction_id)
                while len(self._recent) > 64:
                    self._recent.popitem(last=False)
            return {}
        result_dir.mkdir(parents=True, exist_ok=True)

        with (result_dir / "prediction_result.json").open("w", encoding="utf-8") as handle:
            json.dump(safe_payload, handle, ensure_ascii=False, indent=2)

        qc = safe_payload.get("quality_control", {})
        with (result_dir / "quality_control.json").open("w", encoding="utf-8") as handle:
            json.dump(qc, handle, ensure_ascii=False, indent=2)

        drug = safe_payload.get("drug_response", {})
        all_drugs = drug.get("all_drugs", []) if isinstance(drug, dict) else []
        top10 = drug.get("top10", []) if isinstance(drug, dict) else []
        if all_drugs:
            pd.DataFrame(all_drugs).to_csv(
                result_dir / "all_drug_predictions.csv", index=False
            )
        if top10:
            pd.DataFrame(top10).to_csv(
                result_dir / "top10_drug_predictions.csv", index=False
            )

        subtype = safe_payload.get("subtype", {})
        probabilities = subtype.get("probabilities", []) if isinstance(subtype, dict) else []
        if probabilities:
            pd.DataFrame(probabilities).to_csv(
                result_dir / "subtype_probabilities.csv", index=False
            )

        manifest = {
            "prediction_id": prediction_id,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "files": sorted(path.name for path in result_dir.iterdir() if path.is_file()),
        }
        with (result_dir / "download_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        return {name: "/results/{}/download/{}".format(prediction_id, name) for name in manifest["files"]}

    def result_path(self, prediction_id: str, filename: str) -> Path:
        safe_name = Path(filename).name
        expected_root = self._result_dir(prediction_id)
        path = (expected_root / safe_name).resolve()
        if expected_root not in path.parents:
            raise FileNotFoundError(filename)
        return path
