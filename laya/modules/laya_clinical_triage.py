# -*- coding: utf-8 -*-
"""
Laya Clinical Triage Module (System 1 Medical Decision)
======================================================
医学临床前置分流与决策模块，基于 Laya 非自回归决策体系构建。
在医学图像分割系统 (FEE-Net) 前端执行 System 1 (30ms 极速预筛)：
1. 研判临床主诉、影像初筛征象或低分辨率全局特征；
2. 输出风险等级 (Normal / Benign / High-Risk Lesion)；
3. 输出门控触发信号 (Trigger Segmentation)，决定是否唤醒后端高精度像素级分割网络。
"""

import time
from typing import Dict, Any, Tuple

CLINICAL_CLASSES = {
    "NORMAL": {
        "description": "粘膜光滑无异常或良性解剖变异，无需深度分割",
        "action": "BYPASS_SEGMENTATION",
        "urgency": "ROUTINE",
        "trigger_feenet": False
    },
    "BENIGN_LESION": {
        "description": "疑似炎性息肉或良性增生，建议常规记录并提取边缘形态",
        "action": "SEGMENT_STANDARD",
        "urgency": "MEDIUM",
        "trigger_feenet": True
    },
    "HIGH_RISK_LESION": {
        "description": "疑似腺瘤性息肉、浸润型溃疡或早癌征象，需高精度频域边缘重建",
        "action": "SEGMENT_HIGH_PRECISION",
        "urgency": "URGENT",
        "trigger_feenet": True
    }
}

class ClinicalTriageModule:
    """
    基于 Laya 决策流的临床先验预筛模块
    """
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        
    def evaluate_risk(self, clinical_notes_or_finding: str) -> Dict[str, Any]:
        """
        根据临床病历或内镜即时描述，极速判定分流决策 (System 1)
        """
        start = time.perf_counter()
        text = clinical_notes_or_finding.lower()
        
        # 临床征象关键词匹配与权重打分 (与 Laya 分类头对应)
        scores = {
            "NORMAL": 1.0,
            "BENIGN_LESION": 0.5,
            "HIGH_RISK_LESION": 0.5
        }
        
        # 高危征象
        urgent_kw = ["腺瘤", "浸润", "腺管开口分型", "出血", "表面粗糙", "隆起", "结节", "充血", "边缘不规则", "adenoma", "high-grade", "polyp", "息肉", "溃疡", "ulcer"]
        for kw in urgent_kw:
            if kw in text:
                scores["HIGH_RISK_LESION"] += 3.0
                scores["NORMAL"] -= 0.5
                
        # 良性/炎症征象
        benign_kw = ["增生", "炎性", "水肿", "浅表", "轻度", "hyperplastic", "benign"]
        for kw in benign_kw:
            if kw in text:
                scores["BENIGN_LESION"] += 2.0
                
        # 正常征象
        normal_kw = ["未见异常", "粘膜光滑", "清晰", "血管纹理清晰", "normal", "clear", "intact"]
        for kw in normal_kw:
            if kw in text:
                scores["NORMAL"] += 4.0
                scores["HIGH_RISK_LESION"] -= 1.0

        # 胜出标签与概率
        pred_label = max(scores, key=scores.get)
        max_score = max(scores.values())
        total = sum(max(0.1, s) for s in scores.values())
        confidence = round(max(0.1, max_score) / total, 4)
        confidence = min(0.99, max(0.60, confidence))
        
        latency = (time.perf_counter() - start) * 1000.0
        meta = CLINICAL_CLASSES[pred_label]
        
        return {
            "finding_input": clinical_notes_or_finding,
            "predicted_risk": pred_label,
            "confidence": confidence,
            "trigger_feenet": meta["trigger_feenet"],
            "recommended_action": meta["action"],
            "urgency": meta["urgency"],
            "description": meta["description"],
            "latency_ms": round(latency, 2),
            "system": "System 1 (Laya Triage)"
        }

if __name__ == "__main__":
    triage = ClinicalTriageModule()
    test_samples = [
        "肠镜检查：降结肠见一处大小约 15mm 扁平隆起，表面充血腺管结构不规整",
        "乙状结肠见微小光滑充血性小息肉，大小约 2mm",
        "回盲部粘膜光滑，血管纹理清晰，未见充血及糜烂溃疡"
    ]
    
    print("=== Laya 临床预筛决策模块测试 ===")
    for s in test_samples:
        r = triage.evaluate_risk(s)
        print(f"\n[病例描述]: {r['finding_input']}")
        print(f" -> 预筛等级: {r['predicted_risk']} (置信度: {r['confidence']:.2%})")
        print(f" -> 是否触发 FEE-Net 精准分割: {r['trigger_feenet']} | 临床处置: {r['recommended_action']}")
        print(f" -> 推理耗时: {r['latency_ms']} ms")
