"""AgentJev decision model.

Architecture: Qwen3 backbone (no LM head) + permutation-equivariant
candidate head.

STATE -> QUESTION -> CANDIDATE hierarchy. PathEncoder encodes every
(state, question, candidate) triple as an independent causal sequence
``[state][question][candidate]`` and reads the hidden state at the
candidate's last valid token. This is numerically correct but recomputes
the shared prefix per candidate. TreeEncoder merges identical token
prefixes within each question and uses ancestor-only attention, sharing
prefix computation during both training and inference.

The backbone encoding step is isolated in :class:`PathEncoder` behind the
:func:`AgentJevModel.encode_candidates` seam, so the tree encoder can be
swapped in (``encoder_impl="tree"``) without touching the candidate head,
losses, or training loop.

Candidate head (per question):
  candidate vectors [C, 1024]
    -> Linear 1024->256
    -> CandidateSetEncoder: 2-layer transformer, hidden 256, no positional
       embeddings (permutation-equivariant over the candidate set)
    -> Linear 256->1024, residual with the backbone vector
    -> ScalarScorer: RMSNorm -> Linear 1024->256 -> SiLU -> Linear 256->1
  -> one scalar logit per candidate; per-question softmax is applied in
     the losses.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Qwen3Model

from .encoders import PathEncoder, TreeEncoder
from .data import DEFAULT_MAX_MASK_BYTES


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class CandidateSetEncoder(nn.Module):
    """Transformer over the candidate set of one question.

    No positional embeddings, so the mapping is permutation-equivariant:
    permuting the candidates permutes the outputs identically.
    """

    def __init__(self, d_model: int = 256, nhead: int = 4, num_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
        # x: [Bq, Cmax, d]; cand_mask: [Bq, Cmax] bool, True = valid.
        # src_key_padding_mask: True = position to ignore.
        return self.encoder(x, src_key_padding_mask=~cand_mask)


class ScalarScorer(nn.Module):
    """RMSNorm -> Linear 1024->256 -> SiLU -> Linear 256->1."""

    def __init__(self, in_dim: int = 1024, hidden: int = 256):
        super().__init__()
        self.norm = RMSNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(x)))).squeeze(-1)


class AgentJevModel(nn.Module):
    def __init__(
        self,
        backbone_path: str,
        set_dim: int = 256,
        set_layers: int = 2,
        set_heads: int = 4,
        encoder_impl: str = "path",
        dtype: torch.dtype = torch.float32,
        tree_max_mask_bytes: int = DEFAULT_MAX_MASK_BYTES,
        tree_attention_impl: str = "auto",
    ):
        super().__init__()
        backbone = Qwen3Model.from_pretrained(backbone_path, torch_dtype=dtype)
        hidden = backbone.config.hidden_size
        self.hidden_size = hidden

        if encoder_impl == "path":
            self.path_encoder = PathEncoder(backbone)
        elif encoder_impl == "tree":
            self.path_encoder = TreeEncoder(backbone, max_mask_bytes=tree_max_mask_bytes,
                                            attention_impl=tree_attention_impl)
        else:
            raise ValueError(f"unknown encoder_impl: {encoder_impl}")

        self.proj_in = nn.Linear(hidden, set_dim)
        self.set_encoder = CandidateSetEncoder(set_dim, set_heads, set_layers)
        self.proj_out = nn.Linear(set_dim, hidden)
        self.scorer = ScalarScorer(hidden, set_dim)

    # LLRD grouping hooks -------------------------------------------------
    def backbone_named_parameters(self):
        return self.path_encoder.backbone.named_parameters()

    def head_named_parameters(self):
        for module in (self.proj_in, self.set_encoder, self.proj_out, self.scorer):
            yield from module.named_parameters()

    # ---------------------------------------------------------------------
    def encode_candidates(self, batch: dict) -> torch.Tensor:
        return self.path_encoder(batch)

    def _score(self, cand_vecs: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(cand_vecs)
        x = x * cand_mask.unsqueeze(-1).to(x.dtype)
        enc = self.set_encoder(x, cand_mask)
        rep = cand_vecs + self.proj_out(enc)
        logits = self.scorer(rep)  # [Bq, Cmax]
        return logits.masked_fill(~cand_mask, -1e4)

    def forward(self, batch: dict, perm_reg: bool = False) -> dict:
        cand_mask = batch["cand_mask"]
        cand_vecs = self.encode_candidates(batch)
        logits = self._score(cand_vecs, cand_mask)

        out = {"logits": logits, "cand_mask": cand_mask}

        if perm_reg:
            # Permutation-invariance regularizer: re-run the head on a
            # randomly permuted candidate order per question and align the
            # logits back. With an exactly equivariant encoder this KL is
            # ~0; it guards the property under bf16 numerics.
            Bq, Cmax = cand_mask.shape
            device = cand_vecs.device
            perm = torch.arange(Cmax, device=device).unsqueeze(0).repeat(Bq, 1)
            for q in range(Bq):
                c = int(cand_mask[q].sum())
                if c > 1:
                    perm[q, :c] = torch.randperm(c, device=device)
            idx_h = perm.unsqueeze(-1).expand(-1, -1, cand_vecs.size(-1))
            vecs_p = cand_vecs.gather(1, idx_h)
            mask_p = cand_mask.gather(1, perm)
            # Reuse the same scoring path as the main branch so the two
            # branches cannot diverge in dtype or masking.
            logits_p = self._score(vecs_p, mask_p)
            inv_perm = torch.argsort(perm, dim=1)
            out["logits_perm"] = logits_p.gather(1, inv_perm)
            out["cand_mask"] = cand_mask

        return out
