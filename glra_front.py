"""GLRA front-end (BERT + ViT + RA), returning the 6-token representation.

This is the upstream GLRA `BertModels.py` model with the final classifier
removed; `encode()` stops at `final_representation` (B, 6, 768):

  [global_text, local_text(head,tail), global_mm, local_mm]
"""
import math
import os

import torch
from torch import nn
from transformers import BertModel, ViTModel


class RA(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.text_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.image_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_2 = nn.Linear(self.hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads,
                                     self.attention_head_size)
        return x.reshape(*new_shape).permute(0, 2, 1, 3)

    def forward(self, text, image):
        bsz, text_len = text.shape[0], text.shape[1]
        image_len = image.shape[1]
        text_emb = self.text_projection(text).unsqueeze(2)
        image_emb = self.image_projection(image).unsqueeze(1)
        relevance = torch.mul(text_emb, image_emb)
        relevance = self.relu(self.w_1(relevance))
        relevance = self.sigmoid(self.w_2(relevance)).reshape(
            bsz, 1, text_len, image_len)

        query = self.transpose_for_scores(self.query(text))
        key = self.transpose_for_scores(self.key(image))
        value = self.transpose_for_scores(self.value(image))
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / math.sqrt(self.attention_head_size)
        probs = torch.softmax(scores, dim=-1) * relevance
        context = torch.matmul(probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        return context.reshape(bsz, text_len, self.hidden_size)


class GLRAFront(nn.Module):
    def __init__(self, tokenizer, args):
        super().__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.text_model_name)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.vit = ViTModel.from_pretrained(args.image_model_name)
        self.vit.eval()
        self.vit_config = self.vit.config
        self.fc1 = nn.Linear(self.vit_config.hidden_size, 768)
        self.ra = RA()
        self.head_start = tokenizer.convert_tokens_to_ids("<h>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<t>")
        self.tokenizer = tokenizer

    def encode(self, input_ids, attention_mask, token_type_ids,
               main_images, aux_images):
        pixel_values = torch.cat([main_images, aux_images], dim=1)
        bsz, img_len, c, h, w = pixel_values.shape
        pixel_values = pixel_values.reshape(bsz * img_len, c, h, w)
        with torch.no_grad():
            img_features = self.vit(pixel_values=pixel_values)[1]
        img_features = self.fc1(img_features.reshape(bsz, img_len, -1))

        n_main = int(os.environ.get("GLRA_N_MAIN", "4"))
        n_aux = int(os.environ.get("GLRA_N_AUX", str(img_len - 4)))
        global_img = img_features[:, :n_main, :]
        local_img = img_features[:, n_main:n_main + n_aux, :]

        bert_out = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        seq = bert_out[0]
        head_idx = torch.eq(input_ids, self.head_start).nonzero(as_tuple=True)
        tail_idx = torch.eq(input_ids, self.tail_start).nonzero(as_tuple=True)
        global_text = seq[:, 0:1, :]
        head = seq[head_idx[0], head_idx[1], :]
        tail = seq[tail_idx[0], tail_idx[1], :]
        local_text = torch.stack([head, tail], dim=1)

        global_mm = self.ra(global_text, global_img)
        local_mm = self.ra(local_text, local_img)
        return torch.cat([global_text, local_text, global_mm, local_mm], dim=1)
