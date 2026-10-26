# -*- coding: utf-8 -*-
"""
Laya Task Router (System 1 Decision & Dispatcher)
=================================================
基于 Convai Innovations 开源的 Laya 决策模型 (convaiinnovations/laya-multilingual)
实现极速 (30-40ms) 非自回归任务分流，负责将复杂任务路由至 Pro 深度思考模型，
将轻量/工程任务路由至 Flash / Flash-Lite 高速模型，支持多 Agent 并行调度。
"""

import os
import sys
import time
import math
from typing import Dict, Any, Tuple, Optional

# 目标任务类型定义与对应推荐的 Agent / 模型级别
ROUTING_TARGETS = {
    "ALGO_ARCH_DEEP": {
        "description": "复杂网络架构设计、数学推导、频域小波/边缘注意力模块、创新损失函数",
        "recommended_model": "pro",
        "agent_role": "Core Algorithm Architect",
        "priority": "HIGH"
    },
    "BASELINE_DATA_PIPELINE": {
        "description": "基线模型训练、数据加载管道、批量实验执行、训练参数微调",
        "recommended_model": "flash",
        "agent_role": "Baseline & Data Engineer",
        "priority": "MEDIUM"
    },
    "METRICS_VISUAL_OBSIDIAN": {
        "description": "评测指标统计、高清对比图与收敛曲线绘制、Obsidian 实验知识库留档更新",
        "recommended_model": "flash_lite",
        "agent_role": "Metrics & Visualization Engineer",
        "priority": "LOW"
    },
    "QUALITY_GATE_CHECK": {
        "description": "单元测试、梯度回传检验、张量形状断言、代码质量与实验复现性门禁",
        "recommended_model": "local_verify",
        "agent_role": "Quality Gate Inspector",
        "priority": "IMMEDIATE"
    }
}

class LayaTaskRouter:
    """
    Laya 决策分流路由器
    具备 Hugging Face 权重直接推理与本地轻量级离线热备双模能力
    """
    def __init__(self, model_id: str = "convaiinnovations/laya-multilingual", temperature: float = 1.2):
        self.model_id = model_id
        self.temperature = temperature
        self.pipeline = None
        self.is_hf_ready = False
        
        # 尝试初始化 Hugging Face Pipeline
        self._init_hf_model()

    def _init_hf_model(self):
        # 预设国内镜像加速
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        try:
            import transformers
            from transformers import pipeline
            # 优先以 local_files_only 模式尝试，若已在本地则极速秒开
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_id,
                device=-1,
                model_kwargs={"local_files_only": True}
            )
            self.is_hf_ready = True
        except Exception:
            # 本地尚无缓存或未联网时，秒级降级至 System 1 嵌入判别器，绝不阻塞主流程
            self.is_hf_ready = False

    def download_weights(self):
        """显式下载并缓存 Laya 模型权重"""
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        from transformers import pipeline
        print(f"正在从镜像源下载并缓存 {self.model_id} ...")
        self.pipeline = pipeline("text-classification", model=self.model_id, device=-1)
        self.is_hf_ready = True
        print("Laya 权重缓存成功！")

    def _rule_fallback_predict(self, task_prompt: str) -> Tuple[str, float]:
        """
        轻量级 System 1 离线回退决策算法 (在未联网/无缓存时提供毫秒级确定性响应)
        """
        text = task_prompt.lower()
        
        # 关键字得分矩阵
        scores = {
            "ALGO_ARCH_DEEP": 0.0,
            "BASELINE_DATA_PIPELINE": 0.0,
            "METRICS_VISUAL_OBSIDIAN": 0.0,
            "QUALITY_GATE_CHECK": 0.0
        }
        
        # 深度算法特征
        deep_keywords = ["架构", "小波", "dwt", "egm", "损失", "loss", "数学", "推导", "fee-net", "attention", "mamba", "formula", "derivative", "理论"]
        for kw in deep_keywords:
            if kw in text:
                scores["ALGO_ARCH_DEEP"] += 2.5
                
        # 基线与训练特征
        pipeline_keywords = ["train", "训练", "epoch", "batch", "baseline", "unet", "dataloader", "dataset", "调参", "learning_rate", "lr", "实验", "跑"]
        for kw in pipeline_keywords:
            if kw in text:
                scores["BASELINE_DATA_PIPELINE"] += 2.0
                
        # 指标与可视化留档
        visual_keywords = ["obsidian", "留档", "画图", "plot", "curve", "dice", "iou", "hd95", "可视化", "visual", "表格", "文档", "markdown", "exp_"]
        for kw in visual_keywords:
            if kw in text:
                scores["METRICS_VISUAL_OBSIDIAN"] += 2.2
                
        # 质量与测试特征
        quality_keywords = ["test", "测试", "验证", "check", "assert", "检查", "smoke", "自检", "门禁", "pytest", "unit"]
        for kw in quality_keywords:
            if kw in text:
                scores["QUALITY_GATE_CHECK"] += 2.8

        # 平滑处理与温度校准
        max_target = max(scores, key=scores.get)
        max_val = scores[max_target]
        if max_val == 0:
            max_target = "BASELINE_DATA_PIPELINE"
            confidence = 0.55
        else:
            # 温度 Softmax 仿真
            exp_sum = sum(math.exp(v / self.temperature) for v in scores.values())
            confidence = math.exp(max_val / self.temperature) / exp_sum
            confidence = min(0.98, max(0.65, confidence))
            
        return max_target, confidence

    def route_task(self, task_prompt: str) -> Dict[str, Any]:
        """
        对输入的任务描述进行 System 1 毫秒级分流
        """
        start_time = time.perf_counter()
        
        target = "BASELINE_DATA_PIPELINE"
        confidence = 0.85
        mode = "local_fallback"
        
        if self.is_hf_ready and self.pipeline is not None:
            try:
                res = self.pipeline(task_prompt[:256]) # 限制 token 预算
                # 解析输出
                if isinstance(res, list) and len(res) > 0:
                    raw_label = res[0].get("label", "")
                    raw_score = res[0].get("score", 0.8)
                    # 映射至系统预设路由标签
                    target = raw_label if raw_label in ROUTING_TARGETS else self._rule_fallback_predict(task_prompt)[0]
                    confidence = raw_score
                    mode = "laya_hf_pipeline"
            except Exception:
                target, confidence = self._rule_fallback_predict(task_prompt)
                mode = "local_fallback"
        else:
            target, confidence = self._rule_fallback_predict(task_prompt)
            mode = "local_fallback"
            
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        meta = ROUTING_TARGETS.get(target, ROUTING_TARGETS["BASELINE_DATA_PIPELINE"])
        
        return {
            "task_prompt": task_prompt,
            "target_route": target,
            "confidence": round(confidence, 4),
            "recommended_model": meta["recommended_model"],
            "agent_role": meta["agent_role"],
            "priority": meta["priority"],
            "description": meta["description"],
            "latency_ms": round(latency_ms, 2),
            "inference_mode": mode
        }

if __name__ == "__main__":
    router = LayaTaskRouter()
    print("=== Laya Task Router 测试 ===")
    test_cases = [
        "需要设计带有小波高频融合的编码器架构并推导梯度",
        "启动经典 U-Net 和 Attention U-Net 的对比训练 20 个 Epoch",
        "在 Obsidian 04_Experiment_Logs 中更新指标对比曲线并导出 300DPI 图像",
        "运行单元测试检查张量维度和模型前向反向传播"
    ]
    
    for case in test_cases:
        res = router.route_task(case)
        print(f"\n[任务]: {res['task_prompt']}")
        print(f" -> 决策路由: {res['target_route']} (置信度: {res['confidence']:.2%})")
        print(f" -> 分配模型: {res['recommended_model']} | 对应 Agent: {res['agent_role']}")
        print(f" -> 响应延迟: {res['latency_ms']} ms [{res['inference_mode']}]")
