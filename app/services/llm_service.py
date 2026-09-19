# -*- coding: utf-8 -*-
"""Read the prediction store and send only the approved summary to DeepSeek."""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from typing import Any, Dict, Optional

from app.config import get_settings
from app.core.exceptions import AppError, InputValidationError
from app.services.result_store import ResultStore
from app.utils.prediction_id import normalize_prediction_id


SYSTEM_KNOWLEDGE = """
OncoFusion 是一个多组学肿瘤辅助分析系统，支持五类癌症：BRCA（乳腺癌）、ESCA（食管癌）、KIDNEY（肾癌）、LUNG（肺癌）、UCEC（子宫内膜癌）。
用户上传基因表达、基因突变、DNA甲基化三种数据后，系统调用亚型分类模型和药物敏感性模型，给出亚型概率、候选药物推荐及可靠性评级，并提供多组学可视化和报告导出。
回答时只依据提供的字段解释。未提供的 IC50、可靠性、原始组学或其他患者信息不能编造；模型推荐不等同于临床处方，应由医生结合临床情况判断。
"""


def _text(value: Any) -> Optional[str]:
    # Never stringify nested objects: they could contain unapproved fields.
    return value if isinstance(value, str) else None


def _probability(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        probability = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return probability if math.isfinite(probability) and 0 <= probability <= 1 else None


def _approved_summary(data: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """Explicit allowlist; no patient_id, raw omics, QC or full JSON."""
    subtype = data.get("subtype")
    subtype = subtype if isinstance(subtype, dict) else {}
    drug = data.get("drug_response")
    drug = drug if isinstance(drug, dict) else {}
    top10 = drug.get("top10")
    top10 = top10 if isinstance(top10, list) else []

    # 基础白名单：亚型 + 概率 + Top5 药物名称。
    summary: Dict[str, Any] = {
        "prediction_id": task_id,
        "cancer_type": _text(data.get("cancer_type")),
        "predicted_subtype": _text(subtype.get("predicted_subtype")),
        "top1_probability": _probability(subtype.get("top1_probability")),
        "top2_subtype": _text(subtype.get("top2_subtype")),
        "top2_probability": _probability(subtype.get("top2_probability")),
        "top5_drug_names": [
            _text(row.get("drug_name")) for row in top10[:5]
            if isinstance(row, dict) and _text(row.get("drug_name")) is not None
        ],
    }

    # ===== 可选扩展：需要团队授权后再启用 =====
    # 取消下面注释，可让 AI 解释 IC50 数值、可靠性等级和历史依据。
    # 注意：这些字段仍不含患者 ID、原始组学数据和完整 JSON。
    #
    # summary["top5_drug_details"] = [
    #     {
    #         "drug_name": _text(row.get("drug_name")),
    #         "predicted_ic50": _probability(row.get("predicted_ic50_original"))
    #             if isinstance(row.get("predicted_ic50_original"), (int, float)) else None,
    #         "reliability_level": _text(row.get("reliability_level")),
    #         "predicted_percentile": row.get("predicted_percentile")
    #             if isinstance(row.get("predicted_percentile"), (int, float)) else None,
    #         "training_sample_count": row.get("training_sample_count")
    #             if isinstance(row.get("training_sample_count"), int) else None,
    #     }
    #     for row in top10[:5] if isinstance(row, dict)
    # ]

    return summary


def build_prompt(
    question: str,
    prediction_id: Any = None,
    result_store: Optional[ResultStore] = None,
) -> str:
    try:
        task_id = normalize_prediction_id(prediction_id)
    except ValueError as exc:
        raise InputValidationError(str(exc)) from exc

    context = SYSTEM_KNOWLEDGE
    if task_id is not None:
        # /ask supplies the same live store used by /predict. The fallback
        # preserves standalone callers using configured disk storage.
        if result_store is None:
            settings = get_settings()
            result_store = ResultStore(settings.result_root, enabled=settings.store_results)
        try:
            data = result_store.load(task_id)
        except ValueError as exc:
            raise InputValidationError(
                "任务编号无效，或预测结果文件格式/编号不一致。",
                {"prediction_id": task_id},
            ) from exc
        except OSError as exc:
            raise AppError(
                "PREDICTION_READ_ERROR", "无法读取预测结果，请检查结果存储目录和权限。",
                500, {"prediction_id": task_id},
            ) from exc
        if data is None:
            # Fail locally BEFORE constructing a client or making an LLM request.
            raise AppError(
                "PREDICTION_NOT_FOUND",
                "未找到 prediction_id 为 {} 的预测结果。请检查后端 RESULT_ROOT，或重新运行预测。".format(task_id),
                404, {"prediction_id": task_id},
            )
        context += "\n当前任务预测摘要（仅已授权字段；null 表示未提供）：\n"
        context += json.dumps(_approved_summary(data, task_id), ensure_ascii=False, allow_nan=False)
    else:
        context += "\n未绑定预测任务，仅回答系统功能及通用知识，不推断具体患者结果。"
    return "{}\n\n用户问题：{}".format(context, question)


@lru_cache(maxsize=1)
def _get_llm_client():
    # Missing LLM configuration must not prevent the prediction API starting.
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise AppError("LLM_NOT_CONFIGURED", "未配置 DEEPSEEK_API_KEY，AI 助手暂不可用。", 503)
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise AppError(
            "LLM_DEPENDENCY_MISSING", "AI 助手依赖缺失，请安装 requirements-llm.txt。", 503,
        ) from exc
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=90.0, max_retries=0)


def ask_llm(
    question: str,
    prediction_id: Any = None,
    result_store: Optional[ResultStore] = None,
) -> str:
    prompt = build_prompt(question, prediction_id, result_store)
    client = _get_llm_client()
    try:
        response = client.chat.completions.create(
            # Preserve the existing default; allow deployment configuration.
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            messages=[
                {"role": "system", "content": "你是 OncoFusion 系统的智能助手，请根据提供的背景信息回答用户问题。回答要专业、简洁、易懂。预测不是临床处方建议。"},
                {"role": "user", "content": prompt},
            ],
            stream=False, temperature=0.7,
            # 原为 800，导致模型思考过程耗尽 token，最终回答为空。
            max_tokens=4096,
        )
        content = response.choices[0].message.content
        if not content:
            # 不再抛 502；返回用户能理解的提示。
            return (
                "当前授权范围仅包含亚型、概率和 Top5 药物名称，"
                "无法回答该问题。如需更多信息，请联系管理员开放字段授权。"
            )
        return content
    except AppError:
        raise
    except Exception as exc:
        # SDK errors can expose request bodies or credentials; do not echo them.
        raise AppError("LLM_REQUEST_FAILED", "AI 服务调用失败，请检查模型配置、网络或额度。", 502) from exc
