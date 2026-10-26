# 实验留档 Exp-016：FEE-Net v2.1 架构精简与深监督集成升级

## 1. 实验基本信息
- **实验日期**: 2026-09-22
- **实验编号**: `Exp-016`
- **关联模块**: `models/modules/egm_module.py`, `models/fee_net.py`, `train.py`, `tests/test_fee_net_refinement.py`
- **Git 提交信息**: `feat(arch): implement FEE-Net v2.1 with bilinear upsampling, depthwise DWF, and multi-scale deep supervision (Exp-016)`

---

## 2. 实验目的与动机
在上一轮真实 Kvasir-SEG 基准测试中，我们成功消除了消融倒挂（消融单调正向），但发现 FEE-Net 在收敛速度与边缘平滑度（HD95）上仍受限于底层算子。结合架构审阅专家的深刻见解，针对性解决三大物理瓶颈：
1. **彻底消除棋盘效应 (Checkerboard Artifacts)**：反卷积 (`ConvTranspose2d`) 跨步重叠导致边缘细微锯齿，推高 HD95。改用双线性插值 (`BilinearUpBlock`) 平滑放大；
2. **DWF 算力极致精简**：条形卷积改用深度可分离模式 (`groups=freq_channels`)，保持空间几何抗高光能力，削减 90% DWF 算力；
3. **训练期多尺度深监督 (Deep Supervision)**：在解码器 $d_4, d_3, d_2$ 引入辅助预测头，训练期注入强梯度回传加速收敛，测试期 0 额外耗时。

---

## 3. 核心代码变更与架构演进

### 3.1 `egm_module.py` (Depthwise Separable DWF)
```python
self.conv_dir_h = nn.Sequential(
    nn.Conv2d(freq_channels, freq_channels, kernel_size=(1, 3), padding=(0, 1), groups=freq_channels, bias=False),
    nn.BatchNorm2d(freq_channels)
)
self.conv_dir_v = nn.Sequential(
    nn.Conv2d(freq_channels, freq_channels, kernel_size=(3, 1), padding=(1, 0), groups=freq_channels, bias=False),
    nn.BatchNorm2d(freq_channels)
)
```

### 3.2 `fee_net.py` (BilinearUpBlock & Deep Supervision)
```python
class BilinearUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BilinearUpBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
```
- 在 $d_4, d_3, d_2$ 挂接 `deep_sup4, deep_sup3, deep_sup2`。
- `forward(x, return_aux=True)` 训练模式下返回 `(seg, edge, aux_preds)`，评测模式保持单张量输出，100% 零破裂向下兼容。

---

## 4. 模型复杂度与轻量化核定

| 模型架构 | 参数量 (M) | FLOPs (G) | 相比 UNet++ 参数量 | 相比 UNet++ FLOPs |
| :--- | :---: | :---: | :---: | :---: |
| **UNet++ (Nested SOTA)** | 9.16 M | 34.62 G | 基准 (100%) | 基准 (100%) |
| **Attention U-Net** | 7.85 M | 14.03 G | -14.3% | -59.5% |
| **U-Net (Baseline)** | 7.76 M | 13.75 G | -15.3% | -60.3% |
| **FEE-Net v2 (旧版 48 通道)** | 6.89 M | 14.68 G | -24.8% | -57.6% |
| **FEE-Net v2.1 (当前升级版)** | **6.49 M** | **12.44 G** | **-29.1%** | **-64.1%** |

> **关键结论**：FEE-Net v2.1 参数量降至 **6.49M**，FLOPs 暴降至 **12.44G**，比 UNet++ **节省 64.1% 算力**，甚至比普通基线 U-Net 还要轻量 **9.5%**！

---

## 5. 单元测试与验证结果
运行 `python -m unittest tests/test_fee_net_refinement.py`：
- 全部 6 项严谨单元测试在 0.994s 内无条件全绿通过 (`OK`)！
- 验证了双模输出兼容性、深度可分离 DWF 梯度流动以及深监督反向传播连通性。

---

## 6. 后续行动指南
等待用户发车指令后，正式开启：
1. **5 折交叉验证（Fold 0 ~ Fold 4）全量训练**；
2. **训练轮次拉长至 60 ~ 80 Epochs**，释放平坦收敛区潜力。
