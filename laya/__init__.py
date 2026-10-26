# -*- coding: utf-8 -*-
"""Laya 分流子系统（System 1）。

与分割主线（models/ = FEE-Net, System 2）解耦的独立子系统：
极速视觉筛查 → 三级动态级联 → 仅高风险样本进入全量分割。

入口：
    python laya/train_laya_triage.py     # 训练 + 临床安全校准
    python laya/eval_cascade.py          # 三级级联基准评测
"""
