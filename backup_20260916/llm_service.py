import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

SYSTEM_KNOWLEDGE = """
OncoFusion 是一个多组学肿瘤辅助分析系统，支持五类癌症：BRCA（乳腺癌）、ESCA（食管癌）、KIDNEY（肾癌）、LUNG（肺癌）、UCEC（子宫内膜癌）。
用户上传基因表达、基因突变、DNA甲基化三份CSV文件后，系统会：
1. 调用亚型分类模型，预测癌症亚型（如 BRCA 的 IDC/ILC）。
2. 调用药物敏感性模型，预测231种药物的IC50，给出Top10推荐及可靠性评级。
3. 返回 top_genes 用于3D体素热图。
回答用户问题时，请基于以上系统功能和提供的预测结果进行解释。
"""

def build_prompt(question: str, prediction_id: str = None) -> str:
    """构建发给大模型的 prompt，可选的 prediction_id 用于提供预测上下文"""
    context = SYSTEM_KNOWLEDGE

    if prediction_id:
        path = f"runtime_results/{prediction_id}/prediction_result.json"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            subtype = data.get("subtype", {})
            drug = data.get("drug_response", {})
            top5 = drug.get("top10", [])[:5]
            context += f"""
            当前患者预测结果（ID: {prediction_id}）：
            - 癌种：{data.get('cancer_type')}
            - 预测亚型：{subtype.get('predicted_subtype')}（概率 {subtype.get('top1_probability', 0):.2f}）
            - 第二可能亚型：{subtype.get('top2_subtype')}（概率 {subtype.get('top2_probability', 0):.2f}）
            - Top5 推荐药物：{', '.join([d['drug_name'] for d in top5])}
            """
        else:
            context += f"\n注意：未找到 prediction_id 为 {prediction_id} 的预测结果。"

    return f"{context}\n\n用户问题：{question}"

def ask_llm(question: str, prediction_id: str = None) -> str:
    """调用 DeepSeek 返回回答"""
    try:
        prompt = build_prompt(question, prediction_id)
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": "你是 OncoFusion 系统的智能助手，请根据提供的背景信息回答用户问题。回答要专业、简洁、易懂。"},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.7,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"大模型调用失败: {str(e)}"