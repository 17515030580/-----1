# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import uuid
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.core.exceptions import PredictionError
from app.schemas.domain import RawOmicsSample
from app.services.model_registry import ModelRegistry
from app.services.result_store import ResultStore
from app.utils.json_utils import json_safe


class PredictionService:
    def __init__(
        self,
        settings: Settings,
        registry: ModelRegistry,
        result_store: ResultStore,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.result_store = result_store
        # ---- 新增：加载参考统计量 ----
        self._ref_stats = self._load_reference_stats()

    def _load_reference_stats(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        加载 reference 目录下的统计量，用于计算 z-score。
        返回: {modality: {gene_name: {"mean": float, "std": float}}}
        """
        stats = {}
        artifact_dir = Path(self.settings.drug_artifact_dir)
        ref_dir = artifact_dir / "reference"
        if not ref_dir.exists():
            return {}  # 没有参考文件时，后面将使用绝对值排序
        for modality in ["expression", "mutation", "methylation"]:
            csv_file = ref_dir / f"{modality}_feature_reference.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                # 假设CSV有 feature_name, reference_mean, reference_std 三列
                stats[modality] = {
                    row["feature_name"]: {"mean": row["reference_mean"], "std": row["reference_std"]}
                    for _, row in df.iterrows()
                }
        return stats

    def _select_top_genes(
        self,
        raw: RawOmicsSample,
        n_per_omics: int = 8
    ) -> Dict[str, Any]:
        """
        从三组学原始数据中选出 top n 个基因，返回用于3D热图的数据结构。
        """
        def _top_from_series(series: pd.Series, modality: str, n: int):
            if series.empty:
                return []
            # 获取该组学的参考统计
            ref_dict = self._ref_stats.get(modality, {})
            # 计算 z-score（如果有参考均值和标准差）
            z_scores = pd.Series(index=series.index, dtype=float)
            for gene, val in series.items():
                if gene in ref_dict:
                    mean = ref_dict[gene]["mean"]
                    std = ref_dict[gene]["std"]
                    z = (val - mean) / std if std != 0 else 0.0
                else:
                    # 没有参考统计，用原始值（或0）作为 z 的替代
                    z = float(val)  # 注意：这里如果原始值很大，可能会影响排序，但作为fallback
                z_scores[gene] = z
            # 按 |z| 降序取 top n
            top_genes = z_scores.abs().sort_values(ascending=False).head(n).index
            results = []
            for gene in top_genes:
                results.append({
                    "gene": str(gene),
                    "value": float(series[gene]),
                    "z_score": float(z_scores[gene])
                })
            return results

        # 提取三组学原始 Series
        expr_series = raw.by_modality().get("expression").values if raw.by_modality().get("expression") else pd.Series(dtype=float)
        mut_series = raw.by_modality().get("mutation").values if raw.by_modality().get("mutation") else pd.Series(dtype=float)
        meth_series = raw.by_modality().get("methylation").values if raw.by_modality().get("methylation") else pd.Series(dtype=float)

        expr_top = _top_from_series(expr_series, "expression", n_per_omics)
        mut_top = _top_from_series(mut_series, "mutation", n_per_omics)
        meth_top = _top_from_series(meth_series, "methylation", n_per_omics)

        # 合并去重（用于3D体素坐标轴）
        union_genes = list(set([item["gene"] for item in expr_top + mut_top + meth_top]))

        return {
            "expression": expr_top,
            "mutation": mut_top,
            "methylation": meth_top,
            "union": union_genes
        }

    async def _run_parallel(self, raw: RawOmicsSample):
        subtype_future = run_in_threadpool(self.registry.subtype.predict, raw)
        drug_future = run_in_threadpool(self.registry.drug.predict, raw)
        return await asyncio.gather(subtype_future, drug_future, return_exceptions=True)

    async def _run_serial(self, raw: RawOmicsSample):
        subtype = await run_in_threadpool(self.registry.subtype.predict, raw)
        drug = await run_in_threadpool(self.registry.drug.predict, raw)
        return subtype, drug

    async def predict(self, raw: RawOmicsSample, request_id: str) -> Dict[str, Any]:
        self.registry.ensure_predictable()
        if self.settings.parallel_inference:
            subtype_result, drug_result = await self._run_parallel(raw)
        else:
            subtype_result, drug_result = await self._run_serial(raw)

        errors = {}
        if isinstance(subtype_result, Exception):
            errors["subtype"] = str(subtype_result)
            subtype_result = {"status": "error", "message": str(subtype_result)}
        if isinstance(drug_result, Exception):
            errors["drug_response"] = str(drug_result)
            drug_result = {"status": "error", "message": str(drug_result)}

        if self.settings.require_both_models and errors:
            raise PredictionError("聚合预测失败。", {"model_errors": errors})
        if drug_result.get("status") != "success":
            raise PredictionError("药敏分支预测失败。", {"model_errors": errors})

        # ---- 新增：挑选 top genes ----
        top_genes = self._select_top_genes(raw, n_per_omics=8)

        prediction_id = "PRED_{}_{}".format(
            datetime.now().strftime("%Y%m%d%H%M%S"), uuid.uuid4().hex[:8]
        )
        quality_control = {
            "input": {
                "patient_id": raw.patient_id,
                "cancer_type": raw.cancer_type,
                "three_omics_complete": True,
                "files": {
                    modality: {
                        "filename": parsed.source_filename,
                        "detected_orientation": parsed.orientation,
                        "uploaded_feature_count": int(len(parsed.values)),
                    }
                    for modality, parsed in raw.by_modality().items()
                },
            },
            "subtype_branch": subtype_result.get("quality_control"),
            "drug_response_branch": drug_result.get("quality_control"),
        }
        payload = json_safe(
            {
                "success": True,
                "request_id": request_id,
                "prediction_id": prediction_id,
                "patient_id": raw.patient_id,
                "cancer_type": raw.cancer_type,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "subtype": subtype_result,
                "drug_response": drug_result,
                "quality_control": quality_control,
                "model_status": self.registry.status(),
                "warnings": [
                    "药物排序依据模型预测IC50，不等同于临床处方建议。"
                ],
                # ---- 新增：加入 top_genes ----
                "top_genes": top_genes,
            }
        )
        downloads = self.result_store.save(prediction_id, payload)
        payload["downloads"] = downloads
        # Save again so the JSON itself also contains download links.
        if downloads:
            self.result_store.save(prediction_id, payload)
        return payload
