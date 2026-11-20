"""GLRA front-end (BERT+ViT+RA) + NSVCG RAD backend.

Data pipeline and GLRA front-end come from JinFish/MRE-MI; the RAD diffusion
backend is imported from the NSVCG work tree.  Condition = GLRA's 6-token
final representation; auxiliary classifier reads the global-text token.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent
GLRA_ROOT = ROOT.parent  # ~/mre/MRE-MI in the original checkout
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GLRA_ROOT))
LOCAL_RAD = ROOT / "opennre"
NSVCG_WORK = Path.home() / "mre/nsvcg/nsvcg_glra_hybrid_20261120/work"
sys.path.insert(0, str(ROOT if LOCAL_RAD.exists() else NSVCG_WORK))

from modules.dataset import MREProcessor, MNRE_MI_Dataset  # noqa: E402
from opennre.model.softmax_nn import DenoiseNetwork, Transformer  # noqa: E402
from glra_front import GLRAFront  # noqa: E402


class GLRAFrontRAD(nn.Module):
    def __init__(self, front, num_class):
        super().__init__()
        self.front = front
        self.denoise_network = DenoiseNetwork(
            module_dim=768, module_id=0, layer_num=3,
            num_class=num_class, dropout=0.1)
        self.transformer = Transformer(
            d_model=768, dropout=0.5, num_queries=1, nhead=8,
            dim_feedforward=768, num_encoder_layers=1, num_decoder_layers=1,
            normalize_before=False, return_intermediate_dec=True,
            activation="prelu", num_class=num_class, scale=0.8,
            denoise_network=self.denoise_network, steps=80)
        self.fc = nn.Linear(768, num_class)

    def forward(self, input_ids, token_type_ids, attention_mask,
                main_images, aux_images, labels=None):
        cond = self.front.encode(input_ids, attention_mask, token_type_ids,
                                 main_images, aux_images)
        gt = None
        if self.training and labels is not None:
            gt = F.one_hot(labels, self.fc.out_features)
        diff = self.transformer(cond, gt, None)
        cls = self.fc(cond[:, 0, :])
        return diff, cls


def to_device(batch):
    return [x.cuda() if torch.is_tensor(x) else x for x in batch]


def metrics(y, z):
    p = z.argmax(1)
    positive = y > 0
    return {
        "micro_f1": 100 * f1_score(y, p, labels=range(1, z.shape[1]), average="micro", zero_division=0),
        "macro_f1_relations": 100 * f1_score(y, p, labels=range(1, z.shape[1]), average="macro", zero_division=0),
        "accuracy": 100 * float((p == y).mean()),
        "none_prediction_ratio": 100 * float((p == 0).mean()),
        "fn_none": int((positive & (p == 0)).sum()),
        "fn_all_positive_errors": int((positive & (p != y)).sum()),
    }


def evaluate(model, loader):
    model.eval()
    logits, gold = [], []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch)
            diff, _ = model(*batch[:5])
            logits.append(diff.cpu().numpy())
            gold.append(batch[5].cpu().numpy())
    z = np.concatenate(logits)
    y = np.concatenate(gold)
    return metrics(y, z), z, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=str(Path.home() / "mre/data/MRE-MI"))
    parser.add_argument("--image_path", default=str(Path.home() / "mre/data/MRE-MI/images"))
    parser.add_argument("--aux_image_path", default=str(Path.home() / "mre/data/MRE-MI/images_vg"))
    parser.add_argument("--aux_image_dict", default=str(Path.home() / "mre/data/MRE-MI/images_vg.pth"))
    parser.add_argument("--text_model_name", default="bert-base-uncased")
    parser.add_argument("--image_model_name", default=str(Path.home() / "mre/pretrained_models/ViTL-32-384"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    args.device = "cuda"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    bench = os.environ.get("NSVCG_BENCH", "mremi")
    prefix = "" if bench == "mremi" else bench + "_"
    out = ROOT / "runs" / (prefix + "glra_front_rad_seed" + str(args.seed))
    out.mkdir(parents=True, exist_ok=False)

    processor = MREProcessor(args)
    train_ds = MNRE_MI_Dataset(processor, args, "train.txt")
    val_ds = MNRE_MI_Dataset(processor, args, "val.txt")
    test_ds = MNRE_MI_Dataset(processor, args, "test.txt")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    rel2id = processor.get_relation_dict()
    front = GLRAFront(processor.tokenizer, args)
    model = GLRAFrontRAD(front, len(rel2id)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(steps * args.warmup_ratio),
        num_training_steps=steps)

    config = dict(vars(args), bench=bench, backend="rad", frontend="glra_bert_vit_ra")
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({"start": config}, ensure_ascii=False), flush=True)

    best_f1, best_epoch = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        total, count = 0.0, 0
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            batch = to_device(batch)
            input_ids, token_type_ids, attention_mask, main_images, aux_images, labels = batch
            optimizer.zero_grad(set_to_none=True)
            diff, cls = model(input_ids, token_type_ids, attention_mask,
                              main_images, aux_images, labels)
            loss = F.cross_entropy(diff, labels) + 2 * F.cross_entropy(cls, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += float(loss.detach()) * len(labels)
            count += len(labels)
            if step % 50 == 0:
                print(json.dumps({"epoch": epoch + 1, "step": step,
                                  "loss": float(loss.detach())}), flush=True)
        val, _, _ = evaluate(model, val_loader)
        score = val["micro_f1"]
        with (out / "epochs.jsonl").open("a") as fh:
            fh.write(json.dumps({"epoch": epoch + 1, "seconds": time.time() - t0,
                                 "train_loss": total / max(1, count), "val": val}) + "\n")
        if score > best_f1:
            best_f1, best_epoch = score, epoch + 1
            torch.save({"state_dict": model.state_dict(), "epoch": best_epoch,
                        "val_micro_f1": score}, out / "best.pt")
        print(json.dumps({"epoch": epoch + 1, "val_micro": score,
                          "best": best_f1}), flush=True)

    ckpt = torch.load(out / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.cuda()
    val, zv, yv = evaluate(model, val_loader)
    test, z, y = evaluate(model, test_loader)
    np.savez_compressed(out / "val_logits.npz", raw=zv, joint=zv, gold=yv)
    np.savez_compressed(out / "test_logits.npz", raw=z, joint=z, gold=y)
    (out / "metrics.json").write_text(json.dumps(
        {"config": config, "best_epoch": best_epoch, "val": val, "test": test},
        indent=2), encoding="utf-8")
    print("COMPLETE " + str(out), flush=True)


if __name__ == "__main__":
    main()
