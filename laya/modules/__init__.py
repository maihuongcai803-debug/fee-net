# -*- coding: utf-8 -*-
"""Laya 视觉分流（System 1）组件。

- LayaVisualTriageNet / ClinicalVisualTriageNet : 超轻量二分类筛查头
- ClinicalTriageModule                          : 临床分层判定封装
- LayaCascadeRouter / ClinicalCascadeRouter     : 三级动态级联路由
"""
from .laya_visual_triage import (
    ClinicalVisualTriageNet,
    LayaVisualTriageNet,
    CLASS_NORMAL,
    CLASS_LESION,
    CLASS_NAMES,
)
from .laya_clinical_triage import ClinicalTriageModule
from .laya_cascade_router import ClinicalCascadeRouter, LayaCascadeRouter

__all__ = [
    "ClinicalVisualTriageNet",
    "LayaVisualTriageNet",
    "CLASS_NORMAL",
    "CLASS_LESION",
    "CLASS_NAMES",
    "ClinicalTriageModule",
    "ClinicalCascadeRouter",
    "LayaCascadeRouter",
]
