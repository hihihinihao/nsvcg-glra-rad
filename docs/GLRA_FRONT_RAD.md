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

## 2. 修正后结果（2026-11-21，seed42，8 epoch）

> 旧 `78.19 / 75.75` 是 label leak 的结果，已作废；下面是修复后重跑。

| 口径 | raw F1 | VS F1 | VS macro | None% |
|---|---:|---:|---:|---:|
| all test | 58.45 | **67.06** | 39.87 | 33.3% |
| clean test（去掉 1,018 条与 train 重叠） | 52.36 | **63.08** | 36.60 | — |

对照（seed42）：

| 模型 | all test VS | clean test VS |
|---|---:|---:|
| **B0：RACG + RAD** | **70.11** | **67.06** |
| **H1：RACG + GLRA 后半** | **71.41** | **68.89** |
| GLRA 原版：GLRA 前端 + fc | 71.25 | — |
| **FR：GLRA 前端 + RAD** | **67.06** | **63.08** |

- **FR 比 B0 低 3.05（all）/ 3.98（clean）**；比 H1 低 4.35 / 5.81。
- 修正后 raw 只有 58.45（旧泄漏版 67.38），VS 67.06（旧 78.19）。
- overlap 子集：raw **85.13**，clean 子集 raw **52.36**；all-test 被重叠样本抬高很多。
- **结论：GLRA 前端 + RAD 在这个 8 epoch 设置下明显差于 RACG + RAD；之前的“强前端 + RAD”结论不成立。**

## 3. 怎么读

1. **旧 +8 分是 label leak**：eval 时把 test label 的 one-hot 传给 RAD，RAD 走了训练分支；修复后 raw 从 67.38 掉到 58.45，VS 从 78.19 掉到 67.06。
2. **GLRA 前端本身没有救回来后端**：clean raw 52.36，比 B0 的 61.26/61.74 低很多；ViT 特征 + RA 在这个训练预算下没有体现出优势。
3. **不能再说“RAD 与 GLRA 前端互补”**：修正后 FR 比 H1 低 5.8（clean），比 B0 低 4.0（clean）；互补性没有证据。
4. **当前唯一稳定的正结果仍是 H1**（RACG + GLRA 后半，MRE-MI VS +1.25，5 seed）。

## 4. 边界

- 单 seed42；FR 没有再跑 5 seed（本次只验证修正后的 seed42）。
- FR 是 **8 epoch / bs16 / lr5e-5**（GLRA 配方），B0/H1 是 16 epoch / bs32 / lr3e-5；训练预算不同，不能完全排除“FR 只是欠训练”。
- 数据管线不同：FR 用 GLRA 的 ViT 图像和 `<h>/<t>` 文本，B0/H1 用 NSVCG 的 BLIP/H5 和实体位置；前端和特征都变了，不是纯后端消融。
- test 中 1,018/4,500（22.6%）与 train 有句子+关系重叠；clean 子集已单独报告（63.08）。
- 结果不能直接和 GLRA 论文 65.54 比：训练/评测配置不同。

## 5. 下一步

1. **不建议直接 5 seed FR**：修正后单 seed 已低于 B0/H1 4–6 分，先解决训练预算/前端适配问题。
2. 训练预算对齐：FR 也跑 **16 epoch / bs32 / lr3e-5**，排除“8 epoch 欠训练”。
3. 做 **GLRA 前端 + fc** 的同管线 control（同 8 epoch/同数据），把“RAD 的增量”从“前端/配方”中分离出来。
4. 如果 16 epoch 仍低于 B0，判定“GLRA 前端 + RAD”这条线不成立，回到 H1（RACG + GLRA 后半）。
