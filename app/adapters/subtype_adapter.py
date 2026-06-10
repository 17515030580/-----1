# -*- coding: utf-8 -*-
from __future__ import annotations

import gc
import inspect
import json
import threading
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from app.adapters.base import ModelAdapter
from app.core.exceptions import ArtifactError, InputValidationError, PredictionError
from app.schemas.domain import RawOmicsSample
from app.services.omics_preprocessor import PreprocessedOmics
from app.utils.imports import import_symbol
from app.utils.json_utils import json_safe


SUPPORTED_CANCER_TYPES = {
    "BRCA": "乳腺癌",
    "ESCA": "食管癌",
    "KIDNEY": "肾癌",
    "LUNG": "肺癌",
    "UCEC": "子宫内膜癌",
}

REQUIRED_SUBTYPE_ARTIFACTS = (
    "model/model_state.pt",
    "preprocessing/expression_features.json",
    "preprocessing/mutation_features.json",
    "preprocessing/methylation_features.json",
    "preprocessing/expression_kpca.pkl",
    "preprocessing/mutation_kpca.pkl",
    "preprocessing/methylation_kpca.pkl",
    "metadata/class_id_to_subtype.json",
    "metadata/model_config.json",
)


def normalize_cancer_type(value: str) -> str:
    normalized = str(value or "").strip().upper()
    if normalized not in SUPPORTED_CANCER_TYPES:
        raise InputValidationError(
            "不支持的癌种代码。",
            {
                "cancer_type": value,
                "supported_cancer_types": list(SUPPORTED_CANCER_TYPES),
            },
        )
    return normalized


class SubtypeOmicsPreprocessor:
    """Preprocessing contract used by the exported cancer-subtype artifacts."""

    def __init__(
        self,
        artifact_dir: Path,
        strict_feature_match: bool = True,
        reject_non_finite: bool = True,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.strict_feature_match = strict_feature_match
        self.reject_non_finite = reject_non_finite
        self.feature_lists: Dict[str, List[str]] = {}
        self.transformers: Dict[str, Any] = {}
        self.methylation_medians: Optional[np.ndarray] = None
        self.preprocessing_version = None
        self._load()

    @staticmethod
    def _read_json(path: Path):
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @classmethod
    def _load_feature_list(cls, path: Path, modality: str) -> List[str]:
        payload = cls._read_json(path)
        entries = payload.get("features") if isinstance(payload, dict) else payload
        feature_key = payload.get("feature_key") if isinstance(payload, dict) else None
        if not isinstance(entries, list) or not entries:
            raise ArtifactError(
                "亚型特征清单格式错误。",
                {"modality": modality, "path": str(path)},
            )

        if isinstance(entries[0], dict):
            if not feature_key:
                raise ArtifactError(
                    "对象形式的亚型特征清单缺少feature_key。",
                    {"modality": modality, "path": str(path)},
                )
            try:
                features = [str(item[feature_key]).strip() for item in entries]
            except (KeyError, TypeError) as exc:
                raise ArtifactError(
                    "亚型特征对象与feature_key不匹配。",
                    {"modality": modality, "feature_key": feature_key},
                ) from exc
        else:
            features = [str(item).strip() for item in entries]

        if any(not item for item in features) or len(features) != len(set(features)):
            raise ArtifactError(
                "亚型特征清单包含空值或重复项。", {"modality": modality}
            )
        expected_count = payload.get("feature_count") if isinstance(payload, dict) else None
        if expected_count is not None and int(expected_count) != len(features):
            raise ArtifactError(
                "亚型特征数量与清单声明不一致。",
                {
                    "modality": modality,
                    "declared": int(expected_count),
                    "actual": len(features),
                },
            )
        return features

    def _load(self) -> None:
        preprocessing_dir = self.artifact_dir / "preprocessing"
        missing = [
            str(self.artifact_dir / relative)
            for relative in REQUIRED_SUBTYPE_ARTIFACTS
            if relative.startswith("preprocessing/")
            and not (self.artifact_dir / relative).is_file()
        ]
        if missing:
            raise ArtifactError(
                "癌症亚型预处理产物不完整。", {"missing_files": missing}
            )

        for modality in ("expression", "mutation", "methylation"):
            feature_file = preprocessing_dir / (modality + "_features.json")
            transformer_file = preprocessing_dir / (modality + "_kpca.pkl")
            self.feature_lists[modality] = self._load_feature_list(
                feature_file, modality
            )
            # These files are trusted outputs from the supplied training pipeline.
            self.transformers[modality] = joblib.load(transformer_file)

        medians_path = preprocessing_dir / "methylation_cpg_medians.npy"
        if medians_path.is_file():
            medians = np.asarray(np.load(medians_path), dtype=np.float64).reshape(-1)
            if len(medians) == len(self.feature_lists["methylation"]):
                self.methylation_medians = medians

        manifest_path = self.artifact_dir / "metadata" / "preprocessing_manifest.json"
        if manifest_path.is_file():
            manifest = self._read_json(manifest_path)
            self.preprocessing_version = manifest.get(
                "artifact_version", manifest.get("preprocessing_version")
            )

    def transform(self, raw: RawOmicsSample) -> PreprocessedOmics:
        arrays: Dict[str, np.ndarray] = {}
        qc_modalities: Dict[str, Any] = {}

        for modality, parsed in raw.by_modality().items():
            required = self.feature_lists[modality]
            required_set = set(required)
            uploaded = [str(item).strip() for item in parsed.values.index]
            uploaded_set = set(uploaded)
            missing = [item for item in required if item not in uploaded_set]
            extra = [item for item in uploaded if item not in required_set]

            if missing and self.strict_feature_match:
                raise InputValidationError(
                    "上传文件缺少所选癌种亚型模型要求的特征。",
                    {
                        "cancer_type": raw.cancer_type,
                        "modality": modality,
                        "missing_feature_count": len(missing),
                        "missing_feature_examples": missing[:20],
                    },
                )

            aligned = parsed.values.reindex(required).to_numpy(dtype=np.float64)
            missing_mask = np.isnan(aligned)
            if missing_mask.any():
                if modality == "methylation" and self.methylation_medians is not None:
                    aligned[missing_mask] = self.methylation_medians[missing_mask]
                    fill_method = "training_median"
                else:
                    aligned[missing_mask] = 0.0
                    fill_method = "zero"
            else:
                fill_method = None
            aligned = aligned.reshape(1, -1)

            if self.reject_non_finite and not np.isfinite(aligned).all():
                raise InputValidationError(
                    "亚型特征对齐后仍包含NaN或无穷值。",
                    {"cancer_type": raw.cancer_type, "modality": modality},
                )

            try:
                transformed = self.transformers[modality].transform(aligned)
            except Exception as exc:
                raise InputValidationError(
                    "组学数据无法通过所选癌种训练阶段保存的KPCA转换器。",
                    {
                        "cancer_type": raw.cancer_type,
                        "modality": modality,
                        "error": str(exc),
                    },
                ) from exc

            transformed = np.asarray(transformed, dtype=np.float32).reshape(1, -1)
            arrays[modality] = transformed
            qc_modalities[modality] = {
                "source_filename": parsed.source_filename,
                "detected_orientation": parsed.orientation,
                "uploaded_feature_count": len(uploaded),
                "required_feature_count": len(required),
                "matched_feature_count": len(required) - len(missing),
                "missing_feature_count": len(missing),
                "extra_feature_count": len(extra),
                "missing_feature_examples": missing[:20],
                "extra_feature_examples": extra[:20],
                "missing_fill_method": fill_method,
                "final_embedding_dimension": int(transformed.shape[1]),
                "status": "passed" if not missing else "filled_missing",
            }

        return PreprocessedOmics(
            patient_id=raw.patient_id,
            expression=arrays["expression"],
            mutation=arrays["mutation"],
            methylation=arrays["methylation"],
            quality_control={
                "cancer_type": raw.cancer_type,
                "required_modalities": ["expression", "mutation", "methylation"],
                "all_three_modalities_received": True,
                "strict_feature_match": self.strict_feature_match,
                "preprocessing_version": self.preprocessing_version,
                "modalities": qc_modalities,
            },
        )


class SubtypeAdapter(ModelAdapter):
    """One real MTEGDRP cancer-subtype model for a single cancer cohort."""

    name = "subtype"

    def __init__(
        self,
        cohort: str,
        artifact_dir: Path,
        class_path: str,
        device_name: str,
        forward_mode: str,
        output_mode: str,
        strict_feature_match: bool,
        reject_non_finite: bool,
    ) -> None:
        self.cohort = normalize_cancer_type(cohort)
        self.artifact_dir = Path(artifact_dir)
        self.class_path = class_path
        self.device_name = device_name
        self.forward_mode = forward_mode
        self.output_mode = output_mode
        self.strict_feature_match = strict_feature_match
        self.reject_non_finite = reject_non_finite
        self.loaded = False
        self.load_error = None
        self.model = None
        self.device = None
        self.preprocessor: Optional[SubtypeOmicsPreprocessor] = None
        self.class_names: List[str] = []
        self.manifest: Dict[str, Any] = {}
        self.model_config: Dict[str, Any] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _read_json(path: Path):
        if not path.is_file():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _resolve_device(torch, requested: str):
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested)

    @staticmethod
    def _extract_state_dict(checkpoint):
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    checkpoint = value
                    break
        if not isinstance(checkpoint, dict):
            raise ArtifactError("亚型模型权重不是state_dict格式。")
        return {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in checkpoint.items()
        }

    @staticmethod
    def _load_class_names(artifact_dir: Path) -> List[str]:
        id_map = artifact_dir / "metadata" / "class_id_to_subtype.json"
        if id_map.is_file():
            payload = SubtypeAdapter._read_json(id_map)
            try:
                return [str(payload[str(index)]) for index in range(len(payload))]
            except (KeyError, TypeError) as exc:
                raise ArtifactError("class_id_to_subtype.json格式错误。") from exc

        label_map = artifact_dir / "metadata" / "subtype_label_map.json"
        if label_map.is_file():
            payload = SubtypeAdapter._read_json(label_map)
            if isinstance(payload, list):
                return [str(item) for item in payload]
            if isinstance(payload, dict) and isinstance(payload.get("classes"), list):
                return [str(item) for item in payload["classes"]]
            if isinstance(payload, dict):
                try:
                    ordered = sorted(payload.items(), key=lambda item: int(item[1]))
                    return [str(label) for label, _ in ordered]
                except (TypeError, ValueError) as exc:
                    raise ArtifactError("subtype_label_map.json格式错误。") from exc
        raise ArtifactError("未找到亚型标签映射文件。")

    def load(self) -> None:
        try:
            import torch

            missing = [
                str(self.artifact_dir / relative)
                for relative in REQUIRED_SUBTYPE_ARTIFACTS
                if not (self.artifact_dir / relative).is_file()
            ]
            if missing:
                raise ArtifactError(
                    "癌症亚型模型产物不完整。", {"missing_files": missing}
                )

            self.model_config = self._read_json(
                self.artifact_dir / "metadata" / "model_config.json"
            )
            self.manifest = self._read_json(
                self.artifact_dir / "metadata" / "model_manifest.json"
            )
            configured_cohort = str(
                self.model_config.get("cohort", self.manifest.get("cohort", ""))
            ).upper()
            if configured_cohort and configured_cohort != self.cohort:
                raise ArtifactError(
                    "亚型产物癌种与请求路由不一致。",
                    {"requested": self.cohort, "artifact": configured_cohort},
                )

            self.preprocessor = SubtypeOmicsPreprocessor(
                self.artifact_dir,
                strict_feature_match=self.strict_feature_match,
                reject_non_finite=self.reject_non_finite,
            )
            class_path = self.manifest.get("class_path", self.class_path)
            init_kwargs = self.model_config.get(
                "init_kwargs", self.manifest.get("model_init_kwargs", {})
            )
            self.forward_mode = self.manifest.get(
                "forward_mode", self.forward_mode
            )
            self.output_mode = self.manifest.get("output_mode", self.output_mode)
            if self.model_config.get("output") == "classification_logits":
                self.output_mode = "logits"

            model_class = import_symbol(class_path)
            self.device = self._resolve_device(torch, self.device_name)
            self.model = model_class(**init_kwargs).to(self.device)
            checkpoint = torch.load(
                self.artifact_dir / "model" / "model_state.pt",
                map_location=self.device,
            )
            self.model.load_state_dict(
                self._extract_state_dict(checkpoint), strict=True
            )
            self.model.eval()
            self.class_names = self._load_class_names(self.artifact_dir)
            expected_classes = int(
                self.model_config.get("num_classes", len(self.class_names))
            )
            if expected_classes != len(self.class_names):
                raise ArtifactError(
                    "亚型模型类别数与标签映射不一致。",
                    {
                        "model_class_count": expected_classes,
                        "label_count": len(self.class_names),
                    },
                )
            self.loaded = True
            self.load_error = None
        except Exception as exc:
            self.loaded = False
            self.load_error = str(exc)
            raise

    @staticmethod
    def _extract_tensor(output):
        import torch

        if torch.is_tensor(output):
            return output, None
        if isinstance(output, dict):
            for key in ("probabilities", "probs", "logits", "output", "prediction"):
                value = output.get(key)
                if torch.is_tensor(value):
                    return value, key
        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item):
                    return item, None
        raise PredictionError("无法从亚型模型输出中识别张量。")

    def _call_model(self, expression, mutation, methylation):
        mode = self.forward_mode.lower()
        if mode == "three_args":
            return self.model(expression, mutation, methylation)
        if mode == "keyword_args":
            return self.model(
                expression=expression, mutation=mutation, methylation=methylation
            )
        if mode == "concat":
            import torch

            return self.model(torch.cat([expression, mutation, methylation], dim=1))
        if mode == "dict":
            return self.model(
                {
                    "expression": expression,
                    "mutation": mutation,
                    "methylation": methylation,
                }
            )
        if mode == "data_object":
            return self.model(
                SimpleNamespace(
                    target_ge=expression,
                    target_mut=mutation,
                    target_meth=methylation,
                )
            )
        if mode != "auto":
            raise ArtifactError("未知的SUBTYPE_FORWARD_MODE。", {"mode": mode})

        parameters = [
            item
            for item in inspect.signature(self.model.forward).parameters.values()
            if item.name != "self"
        ]
        if len(parameters) >= 3:
            return self.model(expression, mutation, methylation)
        return self.model(
            SimpleNamespace(
                target_ge=expression,
                target_mut=mutation,
                target_meth=methylation,
            )
        )

    def predict(self, raw: RawOmicsSample) -> Dict[str, Any]:
        if raw.cancer_type != self.cohort:
            raise PredictionError(
                "已加载亚型模型与请求癌种不一致。",
                {"requested": raw.cancer_type, "loaded": self.cohort},
            )
        if not self.loaded or self.model is None or self.preprocessor is None:
            raise PredictionError("亚型模型尚未加载。", {"load_error": self.load_error})
        try:
            import torch

            with self._lock:
                omics = self.preprocessor.transform(raw)
                expression = torch.tensor(
                    omics.expression, dtype=torch.float32, device=self.device
                )
                mutation = torch.tensor(
                    omics.mutation, dtype=torch.float32, device=self.device
                )
                methylation = torch.tensor(
                    omics.methylation, dtype=torch.float32, device=self.device
                )
                with torch.inference_mode():
                    raw_output = self._call_model(
                        expression, mutation, methylation
                    )
                    tensor, detected_key = self._extract_tensor(raw_output)
                    tensor = tensor.detach().float().reshape(1, -1)
                    mode = self.output_mode.lower()
                    if mode == "probabilities" or detected_key in (
                        "probabilities",
                        "probs",
                    ):
                        probabilities = tensor
                    elif mode == "logits":
                        probabilities = torch.softmax(tensor, dim=1)
                    elif mode == "auto":
                        row = tensor[0]
                        one = torch.tensor(1.0, device=row.device)
                        looks_like_probability = bool(
                            torch.all(row >= 0)
                            and torch.all(row <= 1)
                            and torch.isclose(row.sum(), one, atol=1e-3)
                        )
                        probabilities = (
                            tensor
                            if looks_like_probability
                            else torch.softmax(tensor, dim=1)
                        )
                    else:
                        raise ArtifactError(
                            "未知的SUBTYPE_OUTPUT_MODE。", {"mode": mode}
                        )
                    probs = probabilities.cpu().numpy()[0]

            if len(probs) != len(self.class_names):
                raise PredictionError(
                    "亚型输出类别数与标签映射不一致。",
                    {
                        "output_count": len(probs),
                        "label_count": len(self.class_names),
                    },
                )
            order = np.argsort(-probs)
            top1 = int(order[0])
            top2 = int(order[1]) if len(order) > 1 else top1
            entropy = float(
                -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)))
            )
            class_rows = [
                {
                    "subtype": self.class_names[index],
                    "probability": float(probs[index]),
                }
                for index in order
            ]
            return json_safe(
                {
                    "status": "success",
                    "cancer_type": self.cohort,
                    "cancer_name": SUPPORTED_CANCER_TYPES[self.cohort],
                    "model_name": self.manifest.get("model_name", "MTEGDRP"),
                    "model_version": self.manifest.get(
                        "artifact_version", self.manifest.get("model_version")
                    ),
                    "preprocessing_version": self.preprocessor.preprocessing_version,
                    "device": str(self.device),
                    "predicted_subtype": self.class_names[top1],
                    "top1_probability": float(probs[top1]),
                    "top2_subtype": self.class_names[top2],
                    "top2_probability": float(probs[top2]),
                    "probability_margin": float(probs[top1] - probs[top2]),
                    "prediction_entropy": entropy,
                    "probabilities": class_rows,
                    "quality_control": omics.quality_control,
                }
            )
        except (InputValidationError, PredictionError):
            raise
        except Exception as exc:
            raise PredictionError(
                "癌症亚型预测失败。",
                {"cancer_type": self.cohort, "error": str(exc)},
            ) from exc

    def close(self) -> None:
        self.model = None
        self.preprocessor = None
        self.loaded = False
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cancer_type": self.cohort,
            "cancer_name": SUPPORTED_CANCER_TYPES[self.cohort],
            "loaded": self.loaded,
            "artifact_dir": str(self.artifact_dir),
            "device": str(self.device) if self.device is not None else self.device_name,
            "model_version": self.manifest.get(
                "artifact_version", self.manifest.get("model_version")
            ),
            "class_count": len(self.class_names),
            "classes": self.class_names,
            "forward_mode": self.forward_mode,
            "output_mode": self.output_mode,
            "error": self.load_error,
        }


class SubtypeModelRouter(ModelAdapter):
    """Validate five cohort artifacts and lazy-load the selected subtype model."""

    name = "subtype"

    def __init__(
        self,
        artifact_root: Path,
        artifact_version: str,
        cohorts: List[str],
        cache_size: int,
        class_path: str,
        device_name: str,
        forward_mode: str,
        output_mode: str,
        strict_feature_match: bool,
        reject_non_finite: bool,
        enabled: bool,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self.artifact_version = artifact_version
        self.cohorts = [normalize_cancer_type(item) for item in cohorts]
        self.cache_size = max(1, int(cache_size))
        self.class_path = class_path
        self.device_name = device_name
        self.forward_mode = forward_mode
        self.output_mode = output_mode
        self.strict_feature_match = strict_feature_match
        self.reject_non_finite = reject_non_finite
        self.enabled = enabled
        self.ready = False
        self.load_error = None
        self._availability: Dict[str, Dict[str, Any]] = {}
        self._cache: "OrderedDict[str, SubtypeAdapter]" = OrderedDict()
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        with self._lock:
            return bool(self._cache)

    def _artifact_dir(self, cohort: str) -> Path:
        return self.artifact_root / cohort / self.artifact_version

    def load(self) -> None:
        if not self.enabled:
            self.ready = False
            self.load_error = "SUBTYPE_ENABLED=false。"
            return

        # Validate the implementation class without loading any large cohort artifact.
        import_symbol(self.class_path)
        availability: Dict[str, Dict[str, Any]] = {}
        for cohort in self.cohorts:
            artifact_dir = self._artifact_dir(cohort)
            missing = [
                relative
                for relative in REQUIRED_SUBTYPE_ARTIFACTS
                if not (artifact_dir / relative).is_file()
            ]
            classes: List[str] = []
            label_error = None
            if not missing:
                try:
                    classes = SubtypeAdapter._load_class_names(artifact_dir)
                except Exception as exc:
                    label_error = str(exc)
            availability[cohort] = {
                "cancer_name": SUPPORTED_CANCER_TYPES[cohort],
                "available": not missing and label_error is None,
                "loaded": False,
                "artifact_dir": str(artifact_dir),
                "class_count": len(classes),
                "classes": classes,
                "missing_files": missing,
                "error": label_error,
            }

        self._availability = availability
        available = [
            cohort for cohort, item in availability.items() if item["available"]
        ]
        self.ready = bool(available)
        self.load_error = None if self.ready else "没有完整可用的癌症亚型模型产物。"
        if not self.ready:
            raise ArtifactError(
                self.load_error,
                {"artifact_root": str(self.artifact_root), "cohorts": availability},
            )

    def _get_adapter_locked(self, cohort: str) -> SubtypeAdapter:
        if cohort in self._cache:
            adapter = self._cache.pop(cohort)
            self._cache[cohort] = adapter
            return adapter

        while len(self._cache) >= self.cache_size:
            evicted_cohort, evicted = self._cache.popitem(last=False)
            evicted.close()
            if evicted_cohort in self._availability:
                self._availability[evicted_cohort]["loaded"] = False

        adapter = SubtypeAdapter(
            cohort=cohort,
            artifact_dir=self._artifact_dir(cohort),
            class_path=self.class_path,
            device_name=self.device_name,
            forward_mode=self.forward_mode,
            output_mode=self.output_mode,
            strict_feature_match=self.strict_feature_match,
            reject_non_finite=self.reject_non_finite,
        )
        adapter.load()
        self._cache[cohort] = adapter
        self._availability[cohort].update(adapter.status())
        self._availability[cohort]["available"] = True
        return adapter

    def predict(self, raw: RawOmicsSample) -> Dict[str, Any]:
        cohort = normalize_cancer_type(raw.cancer_type or "")
        if not self.enabled:
            return {
                "status": "pending_model",
                "cancer_type": cohort,
                "message": "癌症亚型模型当前未启用。",
            }
        if not self.ready:
            raise PredictionError(
                "癌症亚型模型不可用。", {"load_error": self.load_error}
            )
        cohort_status = self._availability.get(cohort) or {}
        if not cohort_status.get("available"):
            raise ArtifactError(
                "所选癌种的亚型模型产物不完整。",
                {"cancer_type": cohort, "status": cohort_status},
            )
        # The lock also prevents an adapter from being evicted during inference.
        with self._lock:
            adapter = self._get_adapter_locked(cohort)
            return adapter.predict(raw)

    def clear(self) -> None:
        with self._lock:
            for adapter in self._cache.values():
                adapter.close()
            self._cache.clear()
            for item in self._availability.values():
                item["loaded"] = False

    def status(self) -> Dict[str, Any]:
        with self._lock:
            cohorts = {
                cohort: dict(item) for cohort, item in self._availability.items()
            }
            for cohort, adapter in self._cache.items():
                cohorts[cohort].update(adapter.status())
                cohorts[cohort]["available"] = True
            loaded_cohorts = list(self._cache)
        return {
            "name": self.name,
            "enabled": self.enabled,
            "loaded": bool(loaded_cohorts),
            "ready": self.ready,
            "artifact_dir": str(self.artifact_root),
            "artifact_version": self.artifact_version,
            "device": self.device_name,
            "supported_cancer_types": self.cohorts,
            "available_cancer_types": [
                cohort
                for cohort, item in cohorts.items()
                if item.get("available")
            ],
            "loaded_cancer_types": loaded_cohorts,
            "cache_size": self.cache_size,
            "cohorts": cohorts,
            "error": self.load_error,
        }
