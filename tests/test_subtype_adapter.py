# -*- coding: utf-8 -*-
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from app.adapters.subtype_adapter import (
    SubtypeAdapter,
    SubtypeOmicsPreprocessor,
    normalize_cancer_type,
)
from app.core.exceptions import InputValidationError
from app.schemas.domain import ParsedOmics, RawOmicsSample


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_preprocessing_artifacts(root):
    preprocessing = root / "preprocessing"
    metadata = root / "metadata"
    preprocessing.mkdir(parents=True)
    metadata.mkdir(parents=True)
    expression_features = {
        "feature_key": "gene_id",
        "feature_count": 2,
        "features": [
            {"feature_index": 0, "gene_id": "ENSG1.1"},
            {"feature_index": 1, "gene_id": "ENSG2.1"},
        ],
    }
    _write_json(preprocessing / "expression_features.json", expression_features)
    _write_json(
        preprocessing / "mutation_features.json",
        {"feature_count": 2, "features": ["TP53", "EGFR"]},
    )
    _write_json(
        preprocessing / "methylation_features.json",
        {"feature_count": 2, "features": ["cg1", "cg2"]},
    )
    fit_data = np.asarray([[0.0, 0.0], [1.0, 0.5], [0.2, 1.0]])
    for modality in ("expression", "mutation", "methylation"):
        joblib.dump(
            PCA(n_components=1).fit(fit_data),
            preprocessing / (modality + "_kpca.pkl"),
        )
    _write_json(
        metadata / "preprocessing_manifest.json",
        {"artifact_version": "v1.0.0"},
    )
    _write_json(
        metadata / "class_id_to_subtype.json", {"0": "IDC", "1": "ILC"}
    )


def test_subtype_preprocessor_reads_exported_feature_and_joblib_formats(tmp_path):
    _build_preprocessing_artifacts(tmp_path)
    preprocessor = SubtypeOmicsPreprocessor(tmp_path)
    expression = ParsedOmics(
        "expression",
        "P1",
        pd.Series({"ENSG2.1": 0.3, "ENSG1.1": 0.8}),
        "expression.csv",
        "wide",
    )
    mutation = ParsedOmics(
        "mutation",
        "P1",
        pd.Series({"TP53": 1.0, "EGFR": 0.0}),
        "mutation.csv",
        "wide",
    )
    methylation = ParsedOmics(
        "methylation",
        "P1",
        pd.Series({"cg1": 0.7, "cg2": 0.2}),
        "methylation.csv",
        "wide",
    )
    raw = RawOmicsSample(
        "P1", expression, mutation, methylation, cancer_type="BRCA"
    )
    transformed = preprocessor.transform(raw)
    assert transformed.expression.shape == (1, 1)
    assert transformed.mutation.shape == (1, 1)
    assert transformed.methylation.shape == (1, 1)
    assert preprocessor.feature_lists["expression"] == ["ENSG1.1", "ENSG2.1"]


def test_real_label_map_direction_is_preserved(tmp_path):
    _build_preprocessing_artifacts(tmp_path)
    assert SubtypeAdapter._load_class_names(tmp_path) == ["IDC", "ILC"]


def test_cancer_type_validation():
    assert normalize_cancer_type(" lung ") == "LUNG"
    with pytest.raises(InputValidationError):
        normalize_cancer_type("UNKNOWN")
