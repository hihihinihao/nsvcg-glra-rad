# GLRA 前端 + NSVCG RAD

这个包是 **GLRA 的 BERT+ViT+RA 前端** 与 **NSVCG 的 RAD diffusion 后端** 的组合实现，用于 MRE-MI / MNRE 的关系抽取。

- **前端**：GLRA `BERT + ViT + RA`，输出 6 个 token：
  `[global_text, head, tail, global_mm, local_mm]`
- **后端**：NSVCG `DenoiseNetwork + Transformer`（RAD），把 6 个 token 作为 condition，做 80 步 diffusion / DDIM 采样。
- **辅助头**：`fc(global_text)`，训练 loss 为
  `CE(diff) + 2 * CE(cls)`。

> 这不是 GLRA 原版复现：GLRA 原版的 `fc` 分类头被 RAD 替换；ViT 冻结，BERT/RA/RAD 训练。

> ⚠️ **2026-11-20 更正：** 早期 FR 结果（all 78.19 / clean 75.75）**已作废**——当时 `forward` 在 eval 时仍把 `labels` 转成 one-hot 传给 RAD，`Transformer.forward` 用 `gt_target is not None` 分支，导致 **test 标签泄露**。当前代码已改为：**eval 时 `gt=None`、不传 labels**；请只用修复后重跑的数字。

---

## 1. 目录

```
glra_front_rad_package/
├─ README.md
├─ train_glra_front_rad.py      # 训练/评测主脚本
├─ glra_front.py                # GLRA BERT+ViT+RA 前端
├─ modules/                     # GLRA 数据管线（dataset.py）
├─ opennre/                     # NSVCG RAD（device-safe softmax_nn）
├─ smoke.py                     # 单 batch 冒烟
├─ vs_logits.py                 # val 拟合 VS，test 报 raw/VS
├─ run_mnre.sbatch              # MNRE：1 全局 + 2 局部，20 epoch
└─ run_mremi.sbatch             # MRE-MI：4 全局 + 8 局部，8 epoch
```

---

## 2. 数据格式

使用 **GLRA 格式**的数据目录，至少包含：

```
<DATA>/
├─ train.txt
├─ val.txt
├─ test.txt
├─ ours_rel2id.json
├─ images/             # 主图（每实例 1–4 张）
├─ images_vg/          # 辅助 crop 图
└─ images_vg.pth       # image name -> aux crop 文件名列表
```

每行 `train.txt` / `val.txt` / `test.txt` 是 JSON：

```json
{
  "id": "20104",
  "tokens": ["RT", "@user", ":", "..."],
  "h": {"name": "HIV / AIDS", "pos": [10, 13]},
  "t": {"name": "South Africa", "pos": [14, 16]},
  "images": ["20102_0.jpg"],
  "relation": "None"
}
```

### MNRE

MNRE 的 GLRA 格式数据默认在：

```
~/mre/data/MNRE_GLRA/
```

- 12,247 / 1,624 / 1,614，23 类（含 None）；
- 单图：**1 个全局 + 2 个局部 crop**，因此必须设置：

```bash
export GLRA_N_MAIN=1
export GLRA_N_AUX=2
```

如果还没有这份数据，用 `mre-mi-glra/code/convert_mnre_to_glra.py` 从 MNRE 原始包转换；转换后检查 `train.txt/val.txt/test.txt` 行数和 `images_vg.pth` 是否完整。

---

## 3. 环境

- Python 3.10；
- PyTorch 2.5 + CUDA；
- `transformers`、`scikit-learn`、`scipy`、`Pillow`、`numpy`；
- BERT-base-uncased（本地 HF cache 或离线路径）；
- ViT-large-patch32-384 权重：
  `~/mre/pretrained_models/ViTL-32-384`。

如果使用离线环境：

```bash
export HF_HOME=$HOME/mre/hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

`run_mnre.sbatch` 默认用：

```bash
PYTHON=$HOME/mre/venv/bin/python
DATA=$HOME/mre/data/MNRE_GLRA
```

如果路径不同，提交前改这两个变量，或：

```bash
PYTHON=/path/to/python DATA=/path/to/MNRE_GLRA sbatch run_mnre.sbatch 42
```

---

## 4. 冒烟（先跑这个）

在包目录下：

```bash
cd glra_front_rad_package

# MNRE
GLRA_N_MAIN=1 GLRA_N_AUX=2 NSVCG_BENCH=mnre \
python smoke.py --data_path ~/mre/data/MNRE_GLRA --device cuda
```

期望输出：

```
shapes [(1, 23), (1, 23)]
loss ... OK
```

如果是 CPU 冒烟，把 `--device cuda` 去掉；会慢一些，但只需一个 batch。

---

## 5. 训练 MNRE

```bash
cd glra_front_rad_package
sbatch run_mnre.sbatch 42
```

或直接跑：

```bash
GLRA_N_MAIN=1 GLRA_N_AUX=2 NSVCG_BENCH=mnre \
python train_glra_front_rad.py \
  --data_path ~/mre/data/MNRE_GLRA \
  --image_path ~/mre/data/MNRE_GLRA/images \
  --aux_image_path ~/mre/data/MNRE_GLRA/images_vg \
  --aux_image_dict ~/mre/data/MNRE_GLRA/images_vg.pth \
  --text_model_name bert-base-uncased \
  --image_model_name ~/mre/pretrained_models/ViTL-32-384 \
  --batch_size 16 --epochs 20 --lr 5e-5 \
  --seed 42 --max_seq_length 128 --num_workers 4
```

说明：

- MNRE 建议 **20 epoch**（GLRA 原配方）；
- MRE-MI 用 **8 epoch**（`run_mremi.sbatch`），两者不要混。
- 输入 `GLRA_N_MAIN=1`、`GLRA_N_AUX=2` 必须与数据槽位一致，否则 `RA` 会读到错误的 global/local 切分。

---

## 6. 输出

训练完成后：

```
runs/mnre_glra_front_rad_seed42/
├─ config.json
├─ epochs.jsonl
├─ best.pt
├─ metrics.json        # val/test 的 micro/macro/accuracy/None/FN
├─ val_logits.npz      # raw/joint/gold
└─ test_logits.npz
```

`metrics.json` 的 `test` 是 **all-test** 口径；`val_logits.npz` / `test_logits.npz` 可直接用于 VS。

---

## 7. 计算 raw / VS

```bash
python vs_logits.py --run runs/mnre_glra_front_rad_seed42
```

输出：

```
raw { micro_f1, macro_f1_relations, accuracy, none_prediction_ratio }
VS  { ... }
```

并写 `runs/.../vs.json`。

- **MNRE 主口径用 raw**（VS 在 MNRE 上通常掉点）；
- MRE-MI 主口径用 VS；
- 报数时写清 `raw` / `VS`，不要混。

---

## 8. MRE-MI（可选）

```bash
cd glra_front_rad_package
sbatch run_mremi.sbatch 42
```

- 4 全局 + 8 局部，8 epoch；
- 数据 `~/mre/data/MRE-MI`；
- 输出 `runs/glra_front_rad_seed42/`。

MRE-MI 的 all-test 有 **1,018/4,500 条与 train 精确重叠**（句子+关系），所以：

- 报 all-test 时注明重叠；
- 如果要做 clean 口径，去掉 test 中“tokens+relation”在 train 出现的行，再算指标；
- 早期 FR 的 all 78.19 / clean 75.75 **已因 eval label leak 作废**；修复后等待重跑，不要引用旧数字。

MNRE 的精确重叠是 **0 行**（按 tokens+relation），不受这个影响。

---

## 9. 常见问题

**`ModuleNotFoundError: modules`**  
必须在包根目录运行；`train_glra_front_rad.py` / `smoke.py` 会把包根加入 `sys.path`。不要在 `modules/` 里面运行。

**`images_vg.pth` KeyError**  
`images_vg.pth` 的 key 必须与 `train.txt` 的 `images` 文件名一致；检查转换脚本输出。

**OOM**  
把 `--batch_size` 降到 8，或减少 `GLRA_N_AUX`（必须与数据槽位一致）。ViT 冻结但仍占显存。

**ViT pooler warning**  
`Some weights of ViTModel were not initialized ... pooler` 是 GLRA 发布版已知问题；ViT 冻结，pooler 随机初始化，不影响运行。

**HF 联网错误**  
设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，并确认 BERT/ViT 权重在本地缓存/路径。

---

## 10. 边界

- 单 seed 先跑通；5 seed 再下结论。
- 前端是 GLRA 的 BERT+ViT+RA，但 **不是 GLRA 原版**：分类头被 RAD 替换，训练预算不同。
- ViT 冻结；BERT、RA、RAD 可训练。
- MNRE 20 epoch / MRE-MI 8 epoch，不能直接横向比较。
- 结果依赖数据转换质量（`images_vg.pth`、crop 图、实体位置）。

---

## 11. 来源

- GLRA：`JinFish/MRE-MI` @ `7ffd3c1`（`models/BertModels.py`、`modules/dataset.py`）。
- RAD：NSVCG 的 `DenoiseNetwork` / `Transformer`（`opennre/model/softmax_nn.py`，已做 device-safe 小改）。
- 组合与实验记录：`projects/nsvcg-mnre/reports/GLRA_FRONT_RAD.md`。
