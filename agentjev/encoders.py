"""Candidate encoders and segmented tree attention, independent of the scoring head.

PathEncoder is the reference causal implementation. TreeEncoder reuses
that path for small/budget-limited batches and selects dense or segmented
attention for prepacked tries. Parameter names remain backbone.*.
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3Model
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from .data import (DEFAULT_MAX_MASK_BYTES, MIN_TREE_PATH_TOKENS,
                   batch_to_device, build_tree_meta)


class PathEncoder(nn.Module):
    """Independent causal-path encoder (v1).

    Consumes packed per-path token sequences from the collate and returns
    one vector per candidate, scattered into [Bq, Cmax, H].
    """

    def __init__(self, backbone: Qwen3Model):
        super().__init__()
        self.backbone = backbone

    def forward(self, batch: dict) -> torch.Tensor:
        out = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        hs = out.last_hidden_state  # [P, L, H]
        P = hs.size(0)
        pos = batch["cand_end_pos"]
        h = hs[torch.arange(P, device=hs.device), pos]  # [P, H]
        Bq, Cmax = batch["cand_mask"].shape
        cand_vecs = hs.new_zeros(Bq, Cmax, hs.size(-1))
        cand_vecs[batch["q_index"], batch["cand_index"]] = h
        return cand_vecs


class TreeEncoder(PathEncoder):
    """Encode per-question token tries with shared, differentiable prefixes.

    Accepts the same right-padded paths as PathEncoder. Each unique prefix
    is evaluated once; a node attends only to itself and its ancestors.
    RoPE positions use path depth, not the node's packed index. Gathering
    candidate endpoints accumulates gradients into shared prefixes,
    including when paths are identical or one ends inside another.

    Requires full attention with the eager or SDPA backend. With dropout
    disabled, outputs and gradients match independent paths up to floating
    point rounding. Training dropout is shared at merged nodes instead
    of sampled independently per path.

    Collate can precompute topology so forward never reads CUDA tensors on
    the host. Automatic routing uses native segmented causal SDPA for wide
    trees, a dense ancestor mask for fragmented trees, and independent
    paths for small batches or excessive padding. Skewed batches can use a
    flat forest instead of padding every tree to the largest question.
    Questions remain isolated. Speed depends on the workload and device;
    tree_max_mask_bytes limits the dense mask, not total activation memory.
    """

    def __init__(self, backbone: Qwen3Model,
                 max_mask_bytes: int = DEFAULT_MAX_MASK_BYTES,
                 attention_impl: str = "auto"):
        super().__init__(backbone)
        if max_mask_bytes < 0:
            raise ValueError("max_mask_bytes must be non-negative")
        self.max_mask_bytes = max_mask_bytes
        if attention_impl not in ("auto", "dense", "segmented"):
            raise ValueError("tree attention_impl must be auto, dense or segmented")
        self.attention_impl = attention_impl
        self._warned_fallback = False
        self._warned_dropout = False

    def _select_backend(self, meta: dict) -> str:
        if meta["mode"] == "path" or self.max_mask_bytes == 0:
            return "path"
        if self.attention_impl == "segmented":
            return "segmented"
        if self.attention_impl == "auto" and meta["segment_suitable"]:
            if (meta["mask_bytes"] > self.max_mask_bytes
                    or (len(meta["segments"]["groups"]) <= 2
                        and self.backbone.config.hidden_size >= 128
                        and meta["path_tokens"] >= MIN_TREE_PATH_TOKENS)):
                return "segmented"
        if meta["mask_bytes"] > self.max_mask_bytes:
            return "path"
        # On small batches the launch/packing cost usually dominates.
        if self.attention_impl == "auto" and meta["path_tokens"] < MIN_TREE_PATH_TOKENS:
            return "path"
        return "dense"

    def forward(self, batch: dict) -> torch.Tensor:
        config = self.backbone.config
        if config._attn_implementation not in ("eager", "sdpa"):
            raise ValueError("TreeEncoder requires eager or sdpa attention")
        if (getattr(config, "use_sliding_window", False)
                or "sliding_attention" in (getattr(config, "layer_types", None) or [])):
            raise ValueError("TreeEncoder requires full attention; use encoder_impl='path' "
                             "for sliding-window models")

        meta = batch.get("tree_meta")
        if meta is None:
            if (self.attention_impl == "auto"
                    and batch["input_ids"].numel() < MIN_TREE_PATH_TOKENS):
                return super().forward(batch)
            # Compatibility with callers supplying ordinary path batches.
            # Training precomputes this on CPU in collate instead.
            meta = build_tree_meta(batch, max_mask_bytes=self.max_mask_bytes)
        backend = self._select_backend(meta)
        if backend == "path":
            if (not self._warned_fallback and meta.get("fallback_reason") != "small_batch"
                    and (meta["mode"] == "path" or meta["mask_bytes"] > self.max_mask_bytes)):
                warnings.warn("TreeEncoder falling back to path encoding: tree padding "
                              "or attention mask exceeds its budget", RuntimeWarning)
                self._warned_fallback = True
            return super().forward(batch)
        if self.training and config.attention_dropout and not self._warned_dropout:
            warnings.warn("TreeEncoder shares dropout randomness at merged prefixes; "
                          "use attention_dropout=0 for independent-path parity", RuntimeWarning)
            self._warned_dropout = True

        device = batch["input_ids"].device
        meta = batch_to_device(meta, device)
        if backend == "segmented":
            hs = encode_segmented(self.backbone, meta)
        else:
            entry, stop = meta["entry"], meta["exit"]
            # Ancestor test uses O(N) metadata; quadratic work stays on device.
            allowed = ((entry[:, None, :] <= entry[:, :, None])
                       & (entry[:, :, None] <= stop[:, None, :]))
            dtype = self.backbone.get_input_embeddings().weight.dtype
            attention = torch.zeros_like(allowed, dtype=dtype).masked_fill_(
                ~allowed, torch.finfo(dtype).min).unsqueeze(1)
            hs = self.backbone(
                input_ids=meta["input_ids"], attention_mask=attention,
                position_ids=meta["position_ids"], use_cache=False,
            ).last_hidden_state
        h = hs[meta["endpoint_rows"], meta["endpoints"]]
        Bq, Cmax = batch["cand_mask"].shape
        cand_vecs = hs.new_zeros(Bq, Cmax, hs.size(-1))
        cand_vecs[batch["q_index"], batch["cand_index"]] = h
        return cand_vecs


def _segment_layer(layer, hidden_states, position_embeddings, segments):
    residual = hidden_states
    normalized = layer.input_layernorm(hidden_states)
    attention = layer.self_attn
    bq, nmax, _ = normalized.shape
    shape = (bq, nmax, -1, attention.head_dim)
    query = attention.q_norm(attention.q_proj(normalized).view(shape)).transpose(1, 2)
    key = attention.k_norm(attention.k_proj(normalized).view(shape)).transpose(1, 2)
    value = attention.v_proj(normalized).view(shape)
    query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
    # With FP32 weights under autocast, RoPE/qk RMSNorm may leave Q/K in
    # FP32 while V is BF16. Ordinary SDPA casts at dispatcher entry, but
    # CausalBias validates dtypes before reaching that dispatcher.
    query, key = query.to(value.dtype), key.to(value.dtype)
    query = query.transpose(1, 2).reshape(bq * nmax, -1, attention.head_dim)
    key = key.transpose(1, 2).reshape(bq * nmax, -1, attention.head_dim)
    value = value.reshape(bq * nmax, -1, attention.head_dim)

    outputs = []
    dropout = attention.attention_dropout if layer.training else 0.0
    use_gqa = attention.num_key_value_groups > 1
    for group in segments["groups"]:
        qi, ki = group["query_index"], group["key_index"]
        q = query[qi].transpose(1, 2)
        k = key[ki].transpose(1, 2)
        v = value[ki].transpose(1, 2)
        bias = group["bias"].mask
        result = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, is_causal=bias is None,
            dropout_p=dropout, scale=attention.scaling, enable_gqa=use_gqa,
        )
        outputs.append(result.transpose(1, 2).reshape(-1, query.size(1) * attention.head_dim))
    packed = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
    attended = packed[segments["restore_index"]].view(bq, nmax, -1)
    hidden_states = residual + attention.o_proj(attended)
    return hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))


def encode_segmented(backbone, meta: dict) -> torch.Tensor:
    """Encode prepacked trie metadata while preserving backbone parameters.

    ``meta`` supplies input_ids, position_ids, and build_segments() output
    under ``segments``.  No modules or parameters are wrapped or replaced,
    keeping the original Qwen3 state_dict and parameter grouping intact.
    """
    hidden_states = backbone.embed_tokens(meta["input_ids"])
    positions = backbone.rotary_emb(hidden_states, meta["position_ids"])
    segments = meta["segments"]
    checkpoint_layers = backbone.training and backbone.gradient_checkpointing
    for layer in backbone.layers[:backbone.config.num_hidden_layers]:
        if checkpoint_layers:
            # Bind layer in each closure: backward may recompute it after the
            # loop has advanced to a later layer.
            def run_layer(states, current_layer=layer):
                return _segment_layer(current_layer, states, positions, segments)

            hidden_states = checkpoint(run_layer, hidden_states, use_reentrant=False)
        else:
            hidden_states = _segment_layer(layer, hidden_states, positions, segments)
    return backbone.norm(hidden_states)
