# -*- coding: utf-8 -*-
"""EI conference front matter: English block, classification codes, footnotes,
and the English halves of the bilingual table/figure captions.
"""

# ---------------------------------------------------------------- English block
TITLE_EN = ("FEE-Net: An Audited Frequency-Guided Edge-Aware Network for Colonoscopic "
            "Polyp Segmentation — Component-Level Negative Results and a Conformally "
            "Certified Triage Cascade")

AUTHORS_EN = "Wenzhe Han"

AFFIL_EN = ["Beijing Institute of Technology"]

ABSTRACT_EN = (
    "Colorectal cancer is the third most frequently diagnosed malignancy and the "
    "second leading cause of cancer death worldwide, yet 17%–28% of polyps are still "
    "missed at colonoscopy even by experts. Deep polyp segmenters report in-domain Dice "
    "of 0.85–0.95 but fall to roughly 0.6–0.7 on unseen centres, and reported efficiency "
    "gains rarely carry any statistical guarantee. We argue that the binding constraint "
    "on translation is assurance rather than accuracy. We audit a frequency-guided "
    "edge-aware network (FEE-Net, 8.539 M parameters, 12.59 GFLOPs) against three "
    "identically trained baselines on a 150-case held-out partition of Kvasir-SEG that "
    "was isolated before training. First, the composite architecture does not beat a "
    "plain U-Net on Dice (79.47% versus 84.14%) despite 10.0% more parameters. Second, "
    "component attribution reverses the design rationale and locates the real value of the "
    "design: removing the edge-gated module raises Dice to 83.08%, only 1.06 points below "
    "U-Net, while lowering the 95% Hausdorff distance to 31.71 px — the best of the study, "
    "12.7% below U-Net (36.33 px) and 5.0% below the strongest baseline UNet+⁠+ (33.38 px). "
    "The gain is therefore in boundary localisation rather than region overlap, which is the "
    "clinically relevant quantity for resection margins. The three auxiliary losses are each beneficial "
    "(−2.86 without deep supervision, −1.03 without boundary IoU, −0.63 without edge BCE, "
    "with a non-additive joint effect of −2.95). Third, a triage cascade whose head is "
    "fine-tuned only on a distribution-matched, leakage-free cohort attains a calibrated "
    "threshold τ* = 0.59 with an exact Clopper-Pearson bound of FNR ≤ 3.92% at 95% "
    "confidence, a measured false-negative rate of 0/75 on a disjoint certification "
    "split, 99.25% bypass of lesion-free mucosa, and a measured 1.91× latency reduction "
    "(11.30 ms to 5.93 ms). Two silent failure modes are documented: an all-positive gate "
    "cohort collapses the head to a constant, and BatchNorm statistics corrupt a frozen "
    "backbone (Dice 81.42% to 14.06%)."
)

KEYWORDS_EN = ("colonoscopic polyp segmentation; discrete wavelet transform; component "
               "ablation; conformal risk control; early-exit triage; leakage-free "
               "evaluation; reproducibility")

# ---------------------------------------------------------------- Chinese metadata
CLC = "TP391.41"          # 中图分类号：计算机图像处理
DOC_CODE = "A"            # 文献标识码
ARTICLE_NO = ""           # 文章编号（由编辑部填写）

FUND = "基金项目：×××基金（编号：×××）；×××项目（编号：×××）"
BIO = ("作者简介：×××（19××—），男，博士，教授，主要研究方向为医学图像分析与计算机辅助诊断。"
       "E-mail：×××@×××.edu.cn")

# ---------------------------------------------------------------- bilingual captions
# 键为中文图/表题开头（"表1"/…/"图1"/…），值为对应的英文题名
CAPTION_EN = {
    "表1": ("Partitions used in this study. The test partition is isolated before training; "
            "the mixed screening cohort is derived only from held-out test frames and is split "
            "into disjoint calibration and certification halves."),
    "表2": ("Segmentation performance on the 150-frame independent test partition. All rows "
            "share the split, augmentation, 60-epoch budget and loss family; values are "
            "single-run point estimates under a fixed seed."),
    "表3": ("Component attribution on the validation fold (170 frames, 60 epochs, identical "
            "seed and schedule). Deltas are relative to the full configuration."),
    "表4": ("Certified triage operating point. Calibration and certification halves are "
            "disjoint at the frame level; latency is measured with CUDA events on one device."),
    "表5": ("Reported polyp segmentation accuracy in the frequency-guided and boundary-aware "
            "literature with the evaluation setting stated. Values are not directly "
            "comparable across rows."),
    "图1": ("Qualitative behaviour on held-out test frames. Rows are the 10th, 50th and 90th "
            "percentile of lesion area; column headers carry the mean test Dice of each "
            "configuration and the value under each panel is the per-case Dice."),
    "图2": ("Accuracy-computation plane for the seven configurations of Table 2, plotted "
            "directly from the benchmark CSV. Marker area scales inversely with HD95."),
}
