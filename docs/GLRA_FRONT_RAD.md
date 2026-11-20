# GLRA 前端 + NSVCG RAD（2026-11-20）

> ⚠️ **2026-11-20 23:40 更正：本文以下结果全部作废。**  
> 原因：`GLRAFrontRAD.forward` 在 eval 时仍把 `labels` 转成 one-hot 传给 RAD；
> `Transformer.forward` 用 `gt_target is not None` 分支，**test 标签被用于前向**。
> 已修：eval 时 `gt=None`、不传 labels；`78.19 / 75.75` 等旧数字**不可引用**，
> 等待修复后重跑。

> 方向：保留 GLRA 的 **BERT + ViT + RA** 前端，把 GLRA 原来的 `fc` 分类头换成
> NSVCG 的 **RAD diffusion/denoising** 后端。  
> 代码：`~/mre/MRE-MI/glra_rad_20261120/`  
> 数据/图像管线：GLRA `MREProcessor` + `MNRE_MI_Dataset`（4 全局 + 8 局部 crop，ViT-large-384）。  
> 训练：8 epoch、bs16、lr5e-5、warmup 0.01、seed42、ViT 冻结。

## 1. 结构

```
GLRA BERT + ViT + RA
  → final_representation (B,6,768)
      [global_text, local_text(head,tail), global_mm, local_mm]
  → RAD (DenoiseNetwork/Transformer, steps=80, scale=0.8)
  → 26 类 logits
辅助分类头 fc(global_text)
loss = CE(diff) + 2·CE(cls)
```

和 H1（RACG + GLRA 后半）正好相反：

| 实验 | 前端 | 后端 |
|---|---|---|
| B0 | NSVCG RACG | RAD |
| E1–E4 | CHF | RAD |
| H1 | NSVCG RACG | GLRA RA+fc |
| **FR** | **GLRA BERT+ViT+RA** | **RAD** |

## 2. 结果（MRE-MI seed42，8 epoch）

| 口径 | raw F1 | VS F1 | VS macro | None% |
|---|---:|---:|---:|---:|
| all test | 67.38 | **78.19** | 55.32 | 15.2% |
| **clean test（去掉与 train 重叠的 1,018 条）** | **62.76** | **75.75** | 53.09 | — |

对照：

| 模型 | all test VS | clean test VS |
|---|---:|---:|
| B0（RACG+RAD，seed42） | 70.11 | 67.06 |
| H1（RACG+GLRA 后半，seed42） | 71.41 | 68.89 |
| GLRA 原版（BERT+ViT+RA+fc，README 记录） | 71.25 | — |
| **FR（GLRA 前端+RAD，seed42）** | **78.19** | **75.75** |

- **all test**：FR 比 B0 seed42 高 **+8.08**，比 H1 seed42 高 **+6.78**，比 GLRA 原版 VS 高 **+6.94**。
- **clean test**：FR 75.75，比 B0 seed42 的 67.06 高 **+8.69**，比 H1 seed42 的 68.89 高 **+6.85**。
- clean 上 raw 只有 62.76，VS 把排名优势放大到 75.75；说明 FR 的 raw 头校准偏，但关系排序很强。
- 与 GLRA 原版 + VS 的 71.25 比，FR 的 78.19 是 **+6.94**；与 GLRA + stacking 的 74.55 比，FR 仍高 **+3.64**（all test）。

## 3. 怎么读

1. **GLRA 前端本身比 RACG 强很多**：clean raw 62.76 vs B0 61.26/61.74，all raw 67.38；ViT 图像特征 + RA 的作用明显。
2. **RAD 和 GLRA 前端是互补的**：同一个 GLRA 前端，换成 RAD 后 all VS 78.19 vs GLRA 原版 71.25（+6.94）；clean VS 75.75。
3. 这和 H1 的结论并不矛盾：
   - RACG 前端下，GLRA 后半略好（H1 71.11 vs B0 69.86）；
   - GLRA 前端下，RAD 明显更好（FR 78.19 vs GLRA 原版 71.25）。
   说明后端优劣依赖前端特征空间：**强前端 + RAD** 是目前最好的组合。
4. 但是 **单 seed、8 epoch**，而且 FR 的训练/评测 raw 有 1 pp 左右的不稳定（训练 log 66.22，eval-only 67.38）；5 seed 前不能当最终结论。

## 4. 边界

- **单 seed42**；B0/H1 是 5-seed 均值（clean VS 66.68/67.89），FR 只有 seed42，比较是“单点 vs 均值”，不是配对。
- FR 是 **8 epoch / bs16 / lr5e-5**（GLRA 配方），B0/H1 是 16 epoch / bs32 / lr3e-5；训练预算不同，不能把 +8 全部归因于架构。
- 数据管线不同：FR 用 GLRA 的 ViT 图像和 `<h>/<t>` 文本，B0/H1 用 NSVCG 的 BLIP/H5 和实体位置；**前端和特征都变了**，不是纯后端消融。
- test 中 1,018/4,500（22.6%）与 train 有句子+关系重叠；clean 子集已单独报告（75.75），但 clean 也只是一种切片。
- VS 是 val 拟合、test 评估；FR 的 VS 对正则化不敏感（lam 0→0.1：78.06→76.74；半 val 拟合 77.79/77.98），不是单点过拟合。
- 结果不能直接和 GLRA 论文 65.54 比：训练/评测配置不同；这里只报本实验的数字。

## 5. 下一步

1. **FR 的 seed43–46**（5 seed），确认 78.19 / clean 75.75 是否稳定。
2. 训练预算对齐：FR 也跑 16 epoch / bs32 / lr3e-5，排除“8 epoch 配方”解释。
3. 做 **GLRA 前端 + fc** 的同管线 control（同 8 epoch/同数据），把“RAD 的增量”从“前端/配方”中分离出来。
4. 如果 5 seed 仍 >75，值得做逐类/错误分析，看增益来自哪些关系、是否只是 VS 的类先验。
