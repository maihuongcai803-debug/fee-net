# Reference manifest

Every citation in `Manuscript_FEE_Net_2026.tex`, with the local file from which its
claim was extracted. Page-numbered verbatim snippets: `06_Code_Docs/citation_evidence.txt`.

| # | Key | Source | Local file | Status |
|---|---|---|---|---|
| 1 | `divya2026review` | Front. Artif. Intell. 9 (2026) | `papers/references/2026_Review_PolypDL.pdf` | verified |
| 2 | `xiao2026prysmnet` | npj Digit. Med. 9 (2026) | `papers/references/2026_npjDM_PrysmNet_TEMPLATE.pdf` | verified |
| 3 | `jha2020kvasir` | MMM 2020 | `papers/references/2020_KvasirSEG.pdf` | verified |
| 4 | `ali2023polypgen` | Scientific Data 10:75 (2023) | `papers/references/2023_PolypGen.pdf` | verified |
| 5 | `chen2021transunet` | arXiv:2102.04306 | `papers/references/2021_TransUNet.pdf` | verified |
| 6 | `ruan2024vmunet` | arXiv:2402.02491 | `papers/references/2024_VMUNet.pdf` | verified |
| 7 | `ma2024medsam` | Nat. Commun. 15:654 (2024) | `papers/references/2024_MedSAM.pdf` | verified |
| 8 | `zhao2023wranet` | Complex Intell. Syst. 9 (2023) | `papers/references/2023_WRANet.pdf` | verified |
| 9 | `rajasekar2024adaptunet` | Heliyon 10(13):e33655 (2024) | `papers/references/2024_AdaptUNet.fulltext.xml` | verified |
| 10 | `yeemeganet2025` | arXiv:2507.02668 | `papers/references/2025_MEGANetW_WaveletEdge.pdf` | verified |
| 11 | `oh2026crossband` | arXiv:2603.03682 | `papers/references/2026_WaveletCrossBand.pdf` | verified |
| 12 | `laskaridis2021earlyexit` | EMDL 2021 / arXiv:2106.05022 | `papers/references/2021_EarlyExitSurvey.pdf` | verified |
| 13 | `demir2024earlyexit` | arXiv:2409.05336 | `papers/references/2024_EarlyExitCNN.pdf` | verified |
| 14 | `bates2021rcps` | J. ACM 68(6) 2021 | `papers/references/2021_RCPS.pdf` | verified |
| 15 | `angelopoulos2022crc` | ICLR 2024 / arXiv:2208.02814 | `papers/references/2022_ConformalRiskControl.pdf` | verified |
| 16 | `angelopoulos2021gentle` | arXiv:2107.07511 | `papers/references/2021_GentleIntroCP.pdf` | verified |
| 17 | `tan2025cls` | arXiv:2510.17897 | `papers/references/2025_ConformalLesionSeg.pdf` | verified |
| 18 | `dai2025crcfdr` | ICIPCA 2025 / arXiv:2504.04482 | `papers/references/2025_CRCFDR.pdf` | verified |
| 19 | `zheng2026crc` | PMLR 329 (2026) | `papers/references/2026_CRCPETCT.pdf` | verified |
| 20 | `ronneberger2015unet` | MICCAI 2015 | `papers/references/2015_UNet.pdf` | verified |
| 21 | `oktay2018attention` | MIDL 2018 | `papers/references/2018_AttentionUNet.pdf` | verified |
| 22 | `zhou2018unetpp` | MICCAI-DLMIA 2018 | `papers/references/2018_UNetPP.pdf` | verified |
| 23 | `eljurdi2021perimeter` | MIDL 2021, PMLR 143:158-167 | `papers/references/2021_PerimeterLoss.pdf` | verified |
| 24 | `karimi2019hd` | IEEE TMI 39(2) 2020 | `papers/references/2019_KarimiHD.pdf` | verified |
| 25 | `cheng2021boundaryiou` | CVPR 2021 | `papers/references/2021_BoundaryIoU.pdf` | verified |
| 26 | `fan2020pranet` | MICCAI 2020 | `papers/references/2020_PraNet.pdf` | verified |
| 27 | `hu2018senet` | CVPR 2018 | `papers/references/2018_SENet.pdf` | verified |
| 28 | `qin2021fcanet` | ICCV 2021 / arXiv:2012.11879 | `papers/references/2021_FcaNet.pdf` | verified |
| 29 | `milletari2016vnet` | 3DV 2016 / arXiv:1606.04797 | `papers/references/2016_VNet_DiceLoss.pdf` | verified |
| 30 | `lee2015dsn` | AISTATS 2015 / arXiv:1409.5185 | `papers/references/2015_DeeplySupervisedNets.pdf` | verified |
| 31 | `odena2016checker` | Distill (2016), web article | `papers/references/2016_Odena_Checkerboard.html` | verified |
| 32 | `clopper1934` | Biometrika 26(4):404-413 | — | paywalled (see note) |

## Downloaded and screened, not cited

| Key | Source | Local file |
|---|---|---|
| `colonformer2022` | IEEE Access 10:80575-80586 (2022) | `papers/references/2022_TMI_ColonFormer_TEMPLATE.pdf` |
| `topovmunet2025` | arXiv:2505.06210 | `papers/references/2025_TopoVMUNetV2.pdf` |

## Note on the single paywalled item

`clopper1934` is the origin of the exact binomial interval used in the risk-control
equation. The publisher (OUP) and JSTOR both return HTTP 403 and no open mirror was
reachable. The implemented construction is `scipy.stats.beta.ppf(1-delta, k+1, n-k)`,
validated numerically against the closed form `1 - delta^(1/n)` at `k = 0` (3.92% for
n = 75, delta = 0.05) inside the certification script. The citation is retained as the
correct attribution; the claim is independently checkable from the released code and
from items 19-21 and 24.
