"""Canonical prediction IDs shared by API validation and result storage."""
from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Optional


def normalize_prediction_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("prediction_id 不能是布尔值。")
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real) and math.isfinite(value) and float(value).is_integer():
        return str(int(value))
    raise ValueError("prediction_id 必须是字符串或整数。")
