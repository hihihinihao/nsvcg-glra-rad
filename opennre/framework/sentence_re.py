"""Rewritten NSVCG trainer.

Differences from the released ``opennre/framework/sentence_re.py``:
  * loader signature follows the rewritten data loader;
  * ``transformers.AdamW`` (removed in modern transformers) -> ``torch.optim.AdamW``
    with the same parameter grouping and no weight decay on bias / LayerNorm;
  * ``seqeval`` import made optional (it is unused);
  * the released variable names are kept verbatim: the model returns
    ``(diffusion, diffusion, diffusion, classifier, None)``, the training loss is
    ``2*CE(classifier) + CE(diffusion)`` and inference takes the *diffusion*
    branch (``logits = rep_old`` in the release).

The optimisation objective, the branch used for prediction and the metric
definitions are unchanged.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .data_loader import SentenceRELoader
from .utils import AverageMeter


class focal_loss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, num_classes=23, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.alpha = torch.zeros(num_classes)
        self.alpha[0] += (1 - alpha)
        self.alpha[1:] += alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=-1)
        preds_logsoft = F.log_softmax(preds, dim=-1)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        loss = torch.mul(alpha, loss.t())
        return loss.mean() if self.size_average else loss.sum()


class SentenceRE(nn.Module):
    def __init__(self, model, train_path, val_path, test_path, ckpt,
                 batch_size=64, max_epoch=100, lr=0.1, weight_decay=1e-3,
                 warmup_step=300, opt="adamw", sample_ratio=1.0,
                 pretrain_path=None, num_workers=8, seed=42,
                 cond_dropout="", cond_tokens=6, cond_drop_p=0.3):
        super().__init__()
        self.max_epoch = max_epoch
        self.testing = False
        self.ckpt = ckpt
        # ---- shortcut suppression (attention key dropout on the condition) --
        # "uniform"   : drop 1-2 of the five independent tokens, keep C0.
        # "phrase"    : drop C0 and the phrase token together, prob cond_drop_p.
        # "curriculum": same pair, p=1.0 for the first half of training, then
        #               cond_drop_p for the rest (learn the fallback first).
        # A dedicated CPU generator keeps the *global* RNG stream identical to
        # the no-dropout arm, so per-epoch data order stays paired with D0.
        self.cond_dropout = (cond_dropout or "").strip().lower()
        self.cond_drop_p = float(cond_drop_p)
        self.cond_tokens = int(cond_tokens)
        self.seed = int(seed)
        self._mask_gen = torch.Generator()
        self._mask_gen.manual_seed(self.seed * 7919 + 17)
        self.cond_stats = {"batches": 0, "dropped_samples": 0, "rows": 0}

        common = dict(batch_size=batch_size, sample_ratio=sample_ratio,
                      num_workers=num_workers, pretrain_path=pretrain_path)
        self.train_loader = SentenceRELoader(
            train_path, model.rel2id, model.sentence_encoder.tokenize,
            shuffle=True, split="train", **common)
        self.val_loader = SentenceRELoader(
            val_path, model.rel2id, model.sentence_encoder.tokenize,
            shuffle=False, split="val", **common)
        self.test_loader = SentenceRELoader(
            test_path, model.rel2id, model.sentence_encoder.tokenize,
            shuffle=False, split="test", **common)

        self.model = model
        self.parallel_model = nn.DataParallel(self.model)
        self.criterion = nn.CrossEntropyLoss()
        self.focal_loss = focal_loss()

        params = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        grouped_params = [
            {
                "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
                "lr": lr,
                "ori_lr": lr,
            },
            {
                "params": [p for n, p in params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
                "ori_lr": lr,
            },
        ]
        if opt != "adamw":
            raise ValueError("this reconstruction only supports adamw")
        self.optimizer = optim.AdamW(grouped_params)
        self.lr = lr

        training_steps = len(self.train_loader.dataset) // batch_size * self.max_epoch
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_step,
            num_training_steps=max(training_steps, 1))

        if torch.cuda.is_available():
            self.cuda()

    def _to_device(self, data):
        if torch.cuda.is_available():
            for i in range(len(data)):
                try:
                    data[i] = data[i].cuda()
                except Exception:
                    pass
        return data

    def _cond_mask(self, batch, device, epoch=0):
        """Attention key mask for the condition sequence, or None.

        Token order (M2/M2n): [C0, head, tail, phrase, global, object].
          uniform   : keep C0, drop 1 or 2 of the five independent tokens.
          phrase    : with prob cond_drop_p drop C0 and phrase together (0, 3).
          curriculum: same pair; p=1.0 while epoch < max_epoch//2, else
                      cond_drop_p.
        The mask is drawn on CPU from a dedicated generator, so the global RNG
        stream (and therefore the per-epoch data order) is unchanged.
        """
        if not self.cond_dropout:
            return None
        n = self.cond_tokens
        if n not in (5, 6):
            raise RuntimeError("shortcut suppression assumes 5 or 6 condition tokens")
        g = self._mask_gen
        mask = torch.ones(batch, n)
        if self.cond_dropout == "uniform":
            k = torch.where(torch.rand(batch, generator=g) < 0.5, 1, 2)  # 1 or 2
            scores = torch.rand(batch, n - 1, generator=g)
            order = scores.argsort(dim=1)                    # tokens 1..5
            rank = torch.empty_like(order)
            rank.scatter_(1, order, torch.arange(n - 1).expand(batch, -1))
            drop = rank < k.unsqueeze(1)
            mask[:, 1:] = (~drop).float()
        elif self.cond_dropout in ("phrase", "curriculum"):
            if self.cond_dropout == "curriculum" and epoch < self.max_epoch // 2:
                p = 1.0
            else:
                p = self.cond_drop_p
            hit = torch.rand(batch, generator=g) < p
            if n == 6:
                # M2/M2n order: [C0, head, tail, phrase, global, object]
                mask[hit, 0] = 0.0
                mask[hit, 3] = 0.0
            else:
                # M1 order: [head, tail, phrase, global, object] -- no C0 to drop
                mask[hit, 2] = 0.0
        else:
            raise ValueError("unknown NSVCG_COND_DROPOUT: %r" % self.cond_dropout)
        self.cond_stats["batches"] += 1
        self.cond_stats["rows"] += int(batch)
        self.cond_stats["dropped_samples"] += int((mask.sum(1) < n).sum())
        return mask.to(device).view(batch, 1, n)

    def train_model(self, metric="acc"):
        best_metric = 0
        for epoch in range(self.max_epoch):
            self.train()
            logging.info("=== Epoch %d train ===", epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            t = tqdm(self.train_loader, ncols=110)
            for data in t:
                data = self._to_device(data)
                label = data[0]
                args = data[3:]
                cond_mask = self._cond_mask(label.size(0), label.device, epoch)
                _, _, diff_logits, cls_logits, aux = self.parallel_model(
                    *args, label, cond_mask=cond_mask)

                loss = 2 * self.criterion(cls_logits, label) + self.criterion(
                    diff_logits, label)
                if aux is not None:
                    # token-redundancy penalty (NSVCG_COND_ORTH); no-op otherwise
                    loss = loss + aux.sum()
                pred = diff_logits.max(-1)[1]
                acc = float((pred == label).long().sum()) / label.size(0)

                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)

                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()

            logging.info("=== Epoch %d val ===", epoch)
            result = self.eval_model(self.val_loader)[0]
            logging.info("Metric %s current / best: %s / %s",
                         metric, result[metric], best_metric)
            if result[metric] > best_metric:
                best_metric = result[metric]
                folder = "/".join(self.ckpt.split("/")[:-1])
                if folder and not __import__("os").path.exists(folder):
                    __import__("os").makedirs(folder, exist_ok=True)
                torch.save({"state_dict": self.model.state_dict()}, self.ckpt)
                logging.info("Best ckpt and saved.")
        logging.info("Best %s on val set: %f", metric, best_metric)
        if self.cond_dropout:
            logging.info(
                "condition dropout (%s): batches=%d rows=%d dropped_rows=%d",
                self.cond_dropout, self.cond_stats["batches"],
                self.cond_stats["rows"], self.cond_stats["dropped_samples"])
        return best_metric

    def eval_model(self, eval_loader):
        self.eval()
        avg_acc = AverageMeter()
        avg_loss = AverageMeter()
        pred_result = []
        id_list = []
        feature_list = []
        with torch.no_grad():
            t = tqdm(eval_loader, ncols=110)
            for data in t:
                data = self._to_device(data)
                label = data[0]
                img_id = data[1]
                args = data[3:]
                _, _, diff_logits, cls_logits, _ = self.parallel_model(*args, label)
                loss = self.criterion(diff_logits, label) + 2 * self.criterion(
                    cls_logits, label)
                score, pred = diff_logits.max(-1)
                id_list.append(img_id)
                feature_list.append(diff_logits)
                for i in range(pred.size(0)):
                    pred_result.append(pred[i].item())
                acc = float((pred == label).long().sum()) / label.size(0)
                avg_acc.update(acc, pred.size(0))
                avg_loss.update(loss.item(), pred.size(0))
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)
        self.last_pred_result = pred_result
        result, correct_category, org_category, n_category, dpt, dpf = \
            eval_loader.dataset.eval(pred_result)
        return (result, correct_category, org_category, n_category, dpt, dpf,
                id_list, feature_list)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
