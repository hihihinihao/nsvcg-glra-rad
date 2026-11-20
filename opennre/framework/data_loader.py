"""Rewritten NSVCG data loader.

The released ``opennre/framework/data_loader.py`` hard-codes the authors' home
directory, loads six input families that the encoder never reads (ViT h5 features,
gpt-4o entity captions, test-time grounding features), and performs an O(N^2)
entity-caption scan per item.  This rewrite keeps every tensor that the encoder
actually consumes byte-for-byte identical in shape and order, drops the dead
branches, and makes all paths configurable through environment variables.

Consumed tensors (in the order ``TMR_RE.forward`` expects them):
    token, att_mask, pos1, pos2, token_phrase, att_mask_phrase,
    image_ori*, image_ori_objects*, weights,
    head_entity_* (x3)*, tail_entity_* (x3)*,
    blip_feat, blip_obj_feat, ground*, ground_label*
(*) carried for signature compatibility; the released encoder ignores them.
"""
import json
import logging
import os
import random

import h5py
import numpy as np
import sklearn.metrics
import torch
import torch.utils.data as data
from torchvision import transforms
from tqdm import tqdm
from transformers import BertTokenizer

DATA_ROOT = os.environ.get("NSVCG_DATA", "data")
FEAT_ROOT = os.environ.get("NSVCG_FEAT", os.path.join(DATA_ROOT, "h5"))
TEXT_ROOT = os.environ.get(
    "NSVCG_TEXT", os.path.join(DATA_ROOT, "txt")
).replace("\\", "/")

IMG_TOKENS = 32          # InstructBLIP Q-Former query tokens
TEXT_TOKENS = 68         # 32 + 68 = 100, the stored visual-feature length
OBJ_SLOTS = 12           # obj_features rows expected by the release

_H5_CACHE = {}


def _open_h5(path):
    """Open one h5 file per worker process (h5py handles are not fork-safe)."""
    pid = os.getpid()
    key = (pid, path)
    if key not in _H5_CACHE:
        _H5_CACHE[key] = h5py.File(path, "r")
    return _H5_CACHE[key]


def _feat_path(split, kind):
    return os.path.join(FEAT_ROOT, "{}_{}.h5".format(kind, split))


class SentenceREDataset(data.Dataset):
    def __init__(self, text_path, rel2id, tokenizer, sample_ratio, pretrain_path,
                 split, kwargs):
        super().__init__()
        self.split = split
        self.mode = split
        self.text_path = text_path

        self.data = []
        with open(text_path, encoding="UTF-8") as f:
            for line in f:
                line = line.rstrip()
                if len(line) > 0:
                    self.data.append(eval(line))

        self.rel2id = rel2id
        logging.info(
            "Loaded sentence RE dataset %s with %d lines and %d relations.",
            text_path, len(self.data), len(self.rel2id),
        )
        logging.info(
            "Ablation switches: zero_img=%s zero_obj=%s zero_w=%s w_const=%s "
            "text_only=%s",
            os.environ.get("NSVCG_ZERO_IMG", "0"),
            os.environ.get("NSVCG_ZERO_OBJ", "0"),
            os.environ.get("NSVCG_ZERO_W", "0"),
            os.environ.get("NSVCG_W_CONST", "0"),
            os.environ.get("NSVCG_TEXT_ONLY", "0"),
        )

        base = os.path.basename(text_path)
        prefix = base.replace("ours_{}.txt".format(split), "")
        if os.environ.get("NSVCG_TEXT_ONLY") == "1":
            # Text-only model: the image->crop dict and the correlation-weight
            # files are image-derived inputs, so do not even read them.
            self.state_dict_ori = {}
            self.weak_ori, self.strong_ori = {}, {}
        else:
            self.state_dict_ori = torch.load(
                os.path.join(TEXT_ROOT, "mre_{}_dict.pth".format(split)),
                map_location="cpu",
            )
            self.weak_ori, self.strong_ori = self._load_weights(split)
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        self.tokenizer_cap = BertTokenizer.from_pretrained(pretrain_path)

        with open(os.path.join(TEXT_ROOT, "phrase_text_{}.json".format(split)),
                  "r", encoding="utf-8") as f_grounding:
            self.phrase_data = json.load(f_grounding)

        self.blip_path = _feat_path(split, "new_features")
        self.obj_path = _feat_path(split, "obj_features")
        self.blip_file = None
        self.obj_file = None

        self.sample_ratio = sample_ratio
        sample_indexes = random.choices(
            list(range(len(self.data))), k=int(len(self.data) * sample_ratio)
        )
        if sample_ratio != 1.0:
            new_state_dict2 = {}
            new_grounding_text = {}
            for num, idx in enumerate(sample_indexes):
                new_state_dict2[num] = self.state_dict_ori[idx]
                new_grounding_text[str(num)] = self.phrase_data[str(idx)]
            self.state_dict_ori = new_state_dict2
            self.phrase_data = new_grounding_text
            self.data = [self.data[idx] for idx in sample_indexes]

    @staticmethod
    def _load_weights(split):
        def read(name):
            out = {}
            with open(os.path.join(TEXT_ROOT, name), "r", encoding="utf-8") as f:
                for line in f:
                    key, score = line.rstrip("\n").split("\t")
                    out[key] = float(score)
            return out

        return read("{}_weight_weak.txt".format(split)), read(
            "{}_weight_strong.txt".format(split)
        )

    def __len__(self):
        return len(self.data)

    def _blip_feature(self, key):
        # Ablation switch (P1 of the MRE-MI plan): zero the global image branch.
        if os.environ.get("NSVCG_ZERO_IMG") == "1":
            return torch.zeros(1, 100, 4096)
        if self.blip_file is None:
            self.blip_file = _open_h5(self.blip_path)
        feat = torch.tensor(self.blip_file[key][:]).float().squeeze(0)
        if feat.size(0) > 100:
            feat = feat[:100, :]
        elif feat.size(0) < 100:
            feat = torch.cat([feat, torch.zeros(100 - feat.size(0), feat.size(1))], 0)
        return feat.unsqueeze(0)

    def _obj_feature(self, img_id):
        # Ablation switch (P1): zero the local-region branch.
        if os.environ.get("NSVCG_ZERO_OBJ") == "1":
            return torch.zeros(1, OBJ_SLOTS, 4096)
        if self.obj_file is None:
            self.obj_file = _open_h5(self.obj_path)
        feat = torch.tensor(self.obj_file[img_id][:]).float().squeeze(1)
        if feat.size(0) > OBJ_SLOTS:
            feat = feat[:OBJ_SLOTS, :]
        elif feat.size(0) < OBJ_SLOTS:
            feat = torch.cat([feat, torch.zeros(OBJ_SLOTS - feat.size(0), feat.size(1))], 0)
        slot = os.environ.get("NSVCG_OBJ_SLOT", "")
        if slot != "":
            feat[int(slot)] = 0
        return feat.unsqueeze(0)

    def __getitem__(self, index):
        item = self.data[index]
        item["grounding"] = self.phrase_data[str(index)]

        if os.environ.get("NSVCG_TEXT_ONLY") == "1":
            # No image tensors at all: [label, img_id, index] + the six text
            # tensors.  collate_fn passes data[3:] straight to the encoder.
            seq = list(self.tokenizer(item, **self.kwargs))
            return [self.rel2id[item["relation"]], item["img_id"], index] + seq

        seq = list(self.tokenizer(item, **self.kwargs))
        img_id = item["img_id"]
        h = item["h"]
        t = item["t"]

        blip_feat = self._blip_feature((h["name"] + t["name"] + img_id).replace(" ", ""))
        blip_obj_feat = self._obj_feature(img_id)

        weight = [self.weak_ori[img_id], self.strong_ori[img_id]]
        # P1-b: strict text-only.  ``weights`` are the last image-derived input:
        # the encoder multiplies both fused visual terms by them before ``tanh``
        # (bert_encoder.py, ``pic_ori_old``/``pic_ori_new``), so a zero weight
        # makes those terms exactly ``tanh(0) = 0`` and leaves no gradient path
        # into any visual module.  Default ("0") keeps the released behaviour.
        if os.environ.get("NSVCG_ZERO_W") == "1":
            weight = [0.0, 0.0]
        elif os.environ.get("NSVCG_W_CONST") == "1":
            # Control arm: keep the visual term but remove ALL per-sample weight
            # information (constant 1.0, the MRE-MI convention).  Separates
            # "weights carry image information" from "any non-zero visual term
            # helps optimisation".
            weight = [1.0, 1.0]

        # Dead branches in the release (kept as correctly shaped placeholders so
        # the positional *args contract of TMR_RE.forward is unchanged).
        pic_ori = torch.zeros((1, 1, 197, 768))
        pic_ori_objects = torch.zeros((1, 3, 197, 768))
        zeros70 = torch.zeros((1, 70), dtype=torch.long)
        final_label = torch.tensor([0])
        final_ground_obj = torch.tensor([0])

        res = (
            [self.rel2id[item["relation"]]] + [img_id] + [index] + seq
            + [pic_ori] + [pic_ori_objects]
            + [torch.tensor(weight).reshape(1, 2)]
            + [zeros70, zeros70, zeros70]
            + [zeros70, zeros70, zeros70]
            + [blip_feat] + [blip_obj_feat]
            + [final_ground_obj] + [final_label]
        )
        return res

    @staticmethod
    def collate_fn(batch):
        data_ = list(zip(*batch))
        labels = data_[0]
        img_id = data_[1]
        index = data_[2]
        seqs = data_[3:]
        batch_labels = torch.tensor(labels).long()
        batch_seqs = [torch.cat(seq, 0) for seq in seqs]
        return [batch_labels] + [img_id] + [index] + batch_seqs

    def eval(self, pred_result, use_name=False):
        correct = 0
        total = len(self.data)
        correct_positive = 0
        pred_positive = 0
        gold_positive = 0
        n_cls = len(self.rel2id)
        correct_category = np.zeros([n_cls, 1])
        org_category = np.zeros([n_cls, 1])
        n_category = np.zeros([n_cls, 1])
        data_with_pred_T = []
        data_with_pred_F = []
        neg = -1
        for name in ["NA", "na", "no_relation", "Other", "Others", "none", "None"]:
            if name in self.rel2id:
                neg = name if use_name else self.rel2id[name]
                break
        y_pred = []
        y_gt = []
        for i in range(total):
            y_pred.append(pred_result[i])
            y_gt.append(self.rel2id[self.data[i]["relation"]])
            if use_name:
                golden = self.data[i]["relation"]
            else:
                golden = self.rel2id[self.data[i]["relation"]]
                n_category[golden] += 1
            data_with_pred = str(self.data[i]) + str(pred_result[i])
            if golden == pred_result[i]:
                correct += 1
                data_with_pred_T.append(data_with_pred)
                if golden != neg:
                    correct_positive += 1
                    correct_category[golden] += 1
                else:
                    correct_category[0] += 1
            else:
                data_with_pred_F.append(data_with_pred)
            if golden != neg:
                gold_positive += 1
                org_category[golden] += 1
            else:
                org_category[0] += 1
            if pred_result[i] != neg:
                pred_positive += 1
        acc = float(correct) / float(total)
        micro_p = float(correct_positive) / float(pred_positive) if pred_positive else 0
        micro_r = float(correct_positive) / float(gold_positive) if gold_positive else 0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0
        result = {"acc": acc, "micro_p": micro_p, "micro_r": micro_r, "micro_f1": micro_f1}
        logging.info("Evaluation result: %s", result)
        return (result, correct_category, org_category, n_category,
                data_with_pred_T, data_with_pred_F)


def SentenceRELoader(text_path, rel2id, tokenizer, batch_size, shuffle,
                     split, sample_ratio=1.0, num_workers=8,
                     pretrain_path=None, collate_fn=SentenceREDataset.collate_fn,
                     **kwargs):
    dataset = SentenceREDataset(
        text_path=text_path, rel2id=rel2id, tokenizer=tokenizer,
        sample_ratio=sample_ratio, pretrain_path=pretrain_path, split=split,
        kwargs=kwargs,
    )
    return data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                           pin_memory=True, num_workers=num_workers,
                           collate_fn=collate_fn)
