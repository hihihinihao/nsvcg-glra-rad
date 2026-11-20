"""Text-only encoder: the image branch is deleted, not zeroed.

This module is opt-in (``NSVCG_TEXT_ONLY=1``); the default path never imports it.

Construction reuses the *already initialised* text modules of a full ``TMR_RE``
(``bert`` / ``bert2`` / ``linear_phrases`` / ``dropout_linear``) and rebuilds the
fusion layer as ``Linear(768*3, 768)`` with
``ll.weight[:, :2304]`` and the original bias copied over.  That way every shared
parameter is bit-identical to the full model initialised with the same seed, and
the only difference is the removed visual branch — the same canonical-init
control used by the 2026-10-29 audit (condition D).

Inputs (one item is ``[label, img_id, index, token, att_mask, pos1, pos2,
token_phrase, att_mask_phrase]`` from the text-only loader branch); the trainer
appends ``label``, so the signature is the six text tensors plus ``label``.
"""
import torch
import torch.nn as nn

from .bert_encoder import TMR_RE


class TMR_RE_TextOnly(TMR_RE):
    def __init__(self, full_encoder):
        # deliberately skip TMR_RE.__init__: it would build the visual modules
        nn.Module.__init__(self)
        self.max_length = full_encoder.max_length
        self.blank_padding = full_encoder.blank_padding
        self.mask_entity = full_encoder.mask_entity
        self.hidden_size = full_encoder.hidden_size
        self.nclass = getattr(full_encoder, "nclass", 300)
        self.text_only = True

        # reuse the exact same (already initialised) text-side modules
        self.bert = full_encoder.bert
        self.bert2 = full_encoder.bert2
        self.tokenizer = full_encoder.tokenizer
        self.linear_phrases = full_encoder.linear_phrases
        self.dropout_linear = full_encoder.dropout_linear

        # 1536 (head+tail) + 768 (phrase) = 2304; the full model's 3072-wide
        # fusion gets its last 768 columns dropped (they only ever saw the
        # visual term G+L).  Creating the new Linear would consume the global
        # RNG; restore it so the data order matches the other arms.
        rng = torch.get_rng_state()
        self.ll = nn.Linear(768 * 3, 768)
        torch.set_rng_state(rng)
        with torch.no_grad():
            self.ll.weight.copy_(full_encoder.ll.weight[:, :768 * 3])
            self.ll.bias.copy_(full_encoder.ll.bias)

    def forward(self, token, att_mask, pos1, pos2, token_phrase, att_mask_phrase,
                label):
        output_text = self.bert(token, attention_mask=att_mask)
        hidden_text = output_text[0]

        output_phrases = self.bert(token_phrase, attention_mask=att_mask_phrase)
        hidden_phrases = output_phrases[0]

        hidden_phrases = torch.sum(hidden_phrases, dim=1)
        hidden_phrases = self.linear_phrases(hidden_phrases)

        onehot_head = torch.zeros(hidden_text.size()[:2]).float().to(
            hidden_text.device)
        onehot_tail = torch.zeros(hidden_text.size()[:2]).float().to(
            hidden_text.device)
        onehot_head = onehot_head.scatter_(1, pos1, 1)
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)
        head_hidden = (onehot_head.unsqueeze(2) * hidden_text).sum(1)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden_text).sum(1)
        x = torch.cat([head_hidden, tail_hidden], dim=-1)

        xx = torch.cat([x, hidden_phrases], dim=1)          # 2304
        xx = self.ll(self.dropout_linear(xx)).unsqueeze(1)  # b, 1, 768
        rep = xx.mean(1)
        return rep, xx, label, None, None, None, rep, rep
