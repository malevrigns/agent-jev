"""JSONL dataset + collate for AgentJev.

Schema: one JSON object per state (see synth.py docstring). Each state
carries a list of questions; each question carries candidates and a gold
supervision (distribution or counts), supervision type, weight, and an
ordinal flag.

Batching is in whole states: a batch is a list of state dicts, and all
questions/candidates of a state stay in the same batch. The collate
packs every (state, question, candidate) triple into one token sequence
``[state][question][candidate]`` for the PathEncoder, records the
candidate end position, and builds the question-level candidate mask
and targets.

Truncation contract (v2):
  - The state+question prefix budget is decided ONCE per question, so
    every candidate of the same question is encoded on the identical
    context.
  - The state keeps its HEAD (task/goal fields live at the start), not
    its tail.
  - Candidates are capped at ``max_len // 8`` tokens (keeping the tail,
    where the scored answer text ends).
  - Truncation counts are reported per batch (n_trunc_states /
    n_trunc_questions / n_trunc_cands).

Gold schema validation (v2): questions with non-finite or negative
values, empty candidate sets, zero-sum distributions, zero-trial counts,
or length mismatches are dropped and recorded in ``dropped`` with the
sample id and reason.

Serialization: the ``[STATE]`` marker is added exactly once (generators
already include it); training and inference must share this convention.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps

import torch
from torch.nn.attention.bias import causal_lower_right
from torch.utils.data import Dataset

from .losses import SUP_CODES

def _map_tensors(value, fn):
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, dict):
        return {k: _map_tensors(v, fn) for k, v in value.items()}
    if isinstance(value, list):
        return [_map_tensors(v, fn) for v in value]
    if isinstance(value, tuple):
        return tuple(_map_tensors(v, fn) for v in value)
    return value


def batch_to_device(batch: dict, device, non_blocking: bool = False) -> dict:
    return _map_tensors(batch, lambda t: t.to(device, non_blocking=non_blocking))


def pin_batch_memory(batch: dict) -> dict:
    return _map_tensors(batch, lambda t: t.pin_memory() if t.device.type == "cpu" else t)


@dataclass(frozen=True)
class _SegmentBias:
    """Keep a device-independent CausalBias out of recursive Tensor.to calls.

    Constructing CausalBias inside forward breaks full-graph torch.compile.
    It describes sizes, contains no tensor storage, and materializes a mask
    on the query device only when a fused attention kernel is unavailable.
    """

    mask: object


def build_segments(
    nodes: list[list[int]],
    depths: list[list[int]],
    children: list[list[list[int]]],
    roots: list[list[int]],
) -> dict:
    """Pack maximal single-child chains, grouped by ancestor-prefix length.

    All input arrays are per question; node indices within children/roots
    are local to that question.  Returned indices address the flattened
    ``[Bq, Nmax]`` tree tensors.  Segments of different lengths in one group
    are right-padded: valid queries cannot attend to that padding.
    """
    nmax = max(map(len, nodes))
    grouped = defaultdict(list)
    pending = [(qi, root, []) for qi, row in enumerate(roots) for root in row]
    while pending:
        qi, node, ancestors = pending.pop()
        start_depth = depths[qi][node]
        chain = [node]
        while len(children[qi][node]) == 1:
            node = children[qi][node][0]
            chain.append(node)
        grouped[start_depth].append((qi, ancestors, chain))
        next_ancestors = ancestors + chain
        pending.extend((qi, child, next_ancestors) for child in children[qi][node])

    groups = []
    restore = [0] * (len(nodes) * nmax)
    output_offset = 0
    packed_queries = 0
    attention_pairs = 0
    for prefix_length, segments in sorted(grouped.items()):
        query_length = max(len(chain) for _, _, chain in segments)
        query_rows, key_rows = [], []
        for row, (qi, ancestors, chain) in enumerate(segments):
            base = qi * nmax
            # Any real index is safe here: valid queries never read padding.
            padding = [base + chain[0]] * (query_length - len(chain))
            query_rows.append([base + node for node in chain] + padding)
            key_rows.append([base + node for node in ancestors + chain] + padding)
            for col, node in enumerate(chain):
                restore[base + node] = output_offset + row * query_length + col
        group_queries = len(segments) * query_length
        groups.append({
            "query_index": torch.tensor(query_rows, dtype=torch.long),
            "key_index": torch.tensor(key_rows, dtype=torch.long),
            "bias": _SegmentBias(causal_lower_right(query_length, prefix_length + query_length))
            if prefix_length else _SegmentBias(None),
        })
        packed_queries += group_queries
        attention_pairs += len(segments) * (
            prefix_length * query_length + query_length * (query_length + 1) // 2
        )
        output_offset += group_queries
    return {
        "groups": groups,
        "restore_index": torch.tensor(restore, dtype=torch.long),
        "packed_queries": packed_queries,
        "attention_pairs": attention_pairs,
    }


DEFAULT_MAX_MASK_BYTES = 64 * 1024 * 1024
MIN_TREE_PATH_TOKENS = 1024


def build_tree_meta(batch: dict, max_mask_bytes: int = DEFAULT_MAX_MASK_BYTES,
                    packing: str = "auto") -> dict:
    """Pack right-padded paths into per-question tries and causal segments.

    DFS metadata is linear in unique nodes; segment indices also contain
    each segment's ancestor prefix. Fall back when padding increases work,
    or a fragmented tree exceeds the dense mask budget. Eligible segments
    do not allocate that dense mask. ``input_tokens`` counts packed tree
    tokens; the encoder can still choose paths for small batches.
    """
    if max_mask_bytes < 0:
        raise ValueError("max_mask_bytes must be non-negative")
    if packing not in ("auto", "padded"):
        raise ValueError("packing must be auto or padded")
    paths = batch["input_ids"].tolist()
    masks = batch["attention_mask"].tolist()
    ends = batch["cand_end_pos"].tolist()
    questions = batch["q_index"].tolist()
    Bq = batch["cand_mask"].size(0)
    if not paths:
        raise ValueError("TreeEncoder requires at least one non-empty path")
    if not (len(paths) == len(masks) == len(ends) == len(questions)):
        raise ValueError("path metadata lengths must match input_ids")
    nodes = [[] for _ in range(Bq)]
    depths = [[] for _ in range(Bq)]
    children = [[] for _ in range(Bq)]
    roots = [[] for _ in range(Bq)]
    edges = [{} for _ in range(Bq)]
    endpoints = []
    for path, mask, end, qi in zip(paths, masks, ends, questions):
        if (end < 0 or end >= len(path)
                or mask != [1] * (end + 1) + [0] * (len(path) - end - 1)):
            raise ValueError("TreeEncoder expects right-padded paths with "
                             "cand_end_pos at the last valid token")
        if not 0 <= qi < Bq:
            raise ValueError("q_index outside cand_mask")
        parent = -1
        for depth, token in enumerate(path[:end + 1]):
            key = (parent, token)
            node = edges[qi].get(key)
            if node is None:
                node = len(nodes[qi])
                edges[qi][key] = node
                nodes[qi].append(token)
                depths[qi].append(depth)
                children[qi].append([])
                (roots[qi] if parent == -1 else children[qi][parent]).append(node)
            parent = node
        endpoints.append(parent)

    Nmax = max(map(len, nodes))
    unique_tokens = sum(map(len, nodes))
    endpoint_rows = questions
    # For skewed batches a flat forest reduces BOTH padding and mask area.
    # Independent DFS root intervals still isolate different questions.
    layout = "padded"
    if packing == "auto" and unique_tokens ** 2 < Bq * Nmax ** 2:
        offsets, offset = [], 0
        for tokens in nodes:
            offsets.append(offset)
            offset += len(tokens)
        endpoints = [offsets[qi] + node for qi, node in zip(questions, endpoints)]
        endpoint_rows = [0] * len(questions)
        nodes = [[token for row in nodes for token in row]]
        depths = [[depth for row in depths for depth in row]]
        children = [[[offsets[qi] + child for child in row]
                     for qi, question_children in enumerate(children) for row in question_children]]
        roots = [[offsets[qi] + root for qi, row in enumerate(roots) for root in row]]
        Bq, Nmax = 1, unique_tokens
        layout = "flat"
    path_tokens = batch["input_ids"].numel()
    mask_bytes = Bq * Nmax * Nmax * 4
    meta = dict(mode="tree", input_tokens=Bq * Nmax,
                unique_tokens=unique_tokens, path_tokens=path_tokens,
                mask_bytes=mask_bytes, layout=layout)
    if max_mask_bytes == 0 or Bq * Nmax >= path_tokens:
        meta.update(mode="path", input_tokens=path_tokens,
                    fallback_reason="mask_budget" if max_mask_bytes == 0 else "padding_overhead")
        return meta

    segments = build_segments(nodes, depths, children, roots)
    # Highly fragmented trees would need too many small SDPA launches. Keep
    # them on the dense route, subject to its explicit memory budget.
    segment_suitable = (len(segments["groups"]) <= 4
                        and segments["packed_queries"] <= path_tokens)
    if mask_bytes > max_mask_bytes and not segment_suitable:
        meta.update(mode="path", input_tokens=path_tokens, fallback_reason="mask_budget")
        return meta

    ids = torch.zeros(Bq, Nmax, dtype=torch.long)
    positions = torch.zeros_like(ids)
    # Padding gets disjoint singleton intervals, so it attends only to itself.
    entry = torch.arange(Nmax).expand(Bq, Nmax).clone()
    exit_time = entry.clone()
    for qi, tokens in enumerate(nodes):
        n = len(tokens)
        ids[qi, :n] = torch.tensor(tokens, dtype=torch.long)
        positions[qi, :n] = torch.tensor(depths[qi], dtype=torch.long)
        start, stop = [0] * n, [0] * n
        clock = 0
        stack = [(node, False) for node in reversed(roots[qi])]
        while stack:
            node, closing = stack.pop()
            if closing:
                stop[node] = clock - 1
            else:
                start[node] = clock
                clock += 1
                stack.append((node, True))
                stack.extend((child, False) for child in reversed(children[qi][node]))
        entry[qi, :n] = torch.tensor(start, dtype=torch.long)
        exit_time[qi, :n] = torch.tensor(stop, dtype=torch.long)
    meta.update(input_ids=ids, position_ids=positions, entry=entry, exit=exit_time,
                endpoints=torch.tensor(endpoints, dtype=torch.long), segments=segments,
                endpoint_rows=torch.tensor(endpoint_rows, dtype=torch.long),
                segment_suitable=segment_suitable)
    return meta


def pack_tree_batch(batch: dict, attention_impl: str = "auto", **kwargs) -> dict:
    """Return a batch with precomputed tree metadata; source tensors are unchanged."""
    if attention_impl == "auto" and batch["input_ids"].numel() < MIN_TREE_PATH_TOKENS:
        tokens = batch["input_ids"].numel()
        return {**batch, "tree_meta": dict(mode="path", input_tokens=tokens,
                                           path_tokens=tokens, mask_bytes=0,
                                           fallback_reason="small_batch")}
    return {**batch, "tree_meta": build_tree_meta(batch, **kwargs)}


def with_tree_collate(collate, **options):
    """Add CPU tree packing without changing the underlying path collator."""
    @wraps(collate)
    def packed(states):
        return pack_tree_batch(collate(states), **options)

    return packed


STATE_PREFIX = "[STATE] "
QUESTION_PREFIX = "\n[QUESTION] "
CANDIDATE_PREFIX = "\n[CANDIDATE] "
MIN_STATE_TOKENS = 8

# Supervision aliases used by non-synth producers (factory / external).
SUP_ALIASES = {
    "teacher_distribution": "teacher",
    "empirical_multiclass": "empirical",
    "empirical_binary": "empirical",
}


def normalize_sample(s: dict) -> dict:
    """Normalize schema variants across producers to the synth schema:
    question text under ``text``, supervision in SUP_CODES, gold without
    extra metadata keys, ordinal/weight defaults present."""
    s.setdefault("id", s.get("id") or s.get("source", "?") + "-?")
    for q in s.get("questions", []):
        if "text" not in q and "question" in q:
            q["text"] = q["question"]
        sup = q.get("supervision")
        if sup in SUP_ALIASES:
            q["supervision"] = SUP_ALIASES[sup]
        if isinstance(q.get("gold"), dict):
            q["gold"] = {k: v for k, v in q["gold"].items()
                         if k in ("distribution", "counts")}
        q.setdefault("ordinal", False)
        q.setdefault("weight", 0.3)
    return s


class AgentJevDataset(Dataset):
    def __init__(self, path: str, where: dict | None = None):
        self.samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                s = json.loads(line)
                if where and any(s.get(k) != v for k, v in where.items()):
                    continue
                self.samples.append(normalize_sample(s))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


class SourceMixer:
    """Stratified multi-source sampler for mixed-corpus training.

    Sources are NOT merged on disk. Each batch draws one source with
    probability proportional to its weight, then takes ``batch_states``
    whole states from that source's shuffled cycle. Small real corpora
    are oversampled (cycles repeat); the large synthetic corpus is
    undersampled, both controlled by the weights.
    """

    def __init__(self, specs: list[dict], batch_states: int, seed: int = 0):
        self.names: list[str] = []
        self.datasets: list[AgentJevDataset] = []
        self.weights: list[float] = []
        for spec in specs:
            ds = AgentJevDataset(spec["path"], where=spec.get("where"))
            if len(ds) == 0:
                raise ValueError(f"empty source: {spec}")
            self.names.append(spec["name"])
            self.datasets.append(ds)
            self.weights.append(float(spec["weight"]))
        self.batch_states = batch_states
        self.rng = random.Random(seed)
        self.orders = []
        for ds in self.datasets:
            order = list(range(len(ds)))
            self.rng.shuffle(order)
            self.orders.append(order)
        self.pos = [0] * len(self.datasets)
        self.epochs = [0] * len(self.datasets)
        self.drawn = [0] * len(self.datasets)

    def next_batch(self) -> tuple[str, list[dict]]:
        i = self.rng.choices(range(len(self.datasets)),
                             weights=self.weights)[0]
        ds = self.datasets[i]
        order = self.orders[i]
        out = []
        while len(out) < self.batch_states:
            if self.pos[i] >= len(order):
                self.rng.shuffle(order)
                self.pos[i] = 0
                self.epochs[i] += 1
            take = min(self.batch_states - len(out), len(order) - self.pos[i])
            out.extend(ds[j] for j in order[self.pos[i]:self.pos[i] + take])
            self.pos[i] += take
        self.drawn[i] += len(out)
        return self.names[i], out

    def stats(self) -> dict:
        return {
            "names": list(self.names),
            "sizes": [len(ds) for ds in self.datasets],
            "weights": list(self.weights),
            "drawn": list(self.drawn),
            "epochs": list(self.epochs),
        }


class EpochMixer:
    """Deterministic multi-source sampler with a real epoch boundary.

    One epoch is exactly one shuffled pass over every source: a source's
    shuffled index order is consumed once and then the next epoch begins.
    ``batches_per_epoch`` is therefore a fixed integer known at
    construction, and a config's ``epochs`` is a true pass count rather
    than a resample count. ``SourceMixer`` cannot provide this -- it draws
    a source per batch by weight from a cycling order, so the amount of
    data in a "step budget" depends on the draw.

    A batch still comes from ONE source, and the last batch of a pass may
    be SHORT (``drop_last=False`` semantics, matching the DataLoader path)
    so a pass is an exact partition with no duplicated states.

    Weights order the stream; they no longer size it. Source ``i`` always
    contributes ``repeat_i * ceil(n_i / batch_states)`` batches per epoch,
    and the deficit scheduler gives it ~``w_i / sum(w)`` of every PREFIX
    of the epoch. To oversample a small corpus, raise its ``repeat``.
    """

    RATE = 1_000_000  # integer Bresenham scale: no float drift

    def __init__(self, specs: list[dict], batch_states: int, seed: int = 0):
        self.names: list[str] = []
        self.datasets: list[AgentJevDataset] = []
        self.weights: list[float] = []
        self.repeats: list[int] = []
        for spec in specs:
            ds = AgentJevDataset(spec["path"], where=spec.get("where"))
            if len(ds) == 0:
                raise ValueError(f"empty source: {spec}")
            self.names.append(spec["name"])
            self.datasets.append(ds)
            self.weights.append(float(spec.get("weight", 1.0)))
            self.repeats.append(int(spec.get("repeat", 1)))
        if any(w < 0 for w in self.weights) or sum(self.weights) <= 0:
            raise ValueError("source weights must be >= 0 and not all zero")
        if any(r < 1 for r in self.repeats):
            raise ValueError("source repeat must be >= 1")
        self.batch_states = batch_states
        self.rng = random.Random(seed)
        self.sizes = [len(ds) for ds in self.datasets]
        self.batches_per_source = [
            r * -(-n // batch_states)  # ceil: the final pass batch may be short
            for n, r in zip(self.sizes, self.repeats)
        ]
        self.batches_per_epoch = sum(self.batches_per_source)
        total_w = sum(self.weights)
        self._rates = [max(1, int(round(w / total_w * self.RATE)))
                       for w in self.weights]
        self.drawn = [0] * len(self.names)  # batches, since start
        self.epoch = 0                      # whole-epoch passes started
        self._start_epoch()

    def _passes(self, i: int) -> list[list[dict]]:
        """``repeat_i`` shuffled passes over source i, each sliced into batches."""
        ds, n, bs = self.datasets[i], self.sizes[i], self.batch_states
        out: list[list[dict]] = []
        for _ in range(self.repeats[i]):
            order = list(range(n))
            self.rng.shuffle(order)
            out.extend([ds[j] for j in order[k:k + bs]]
                       for k in range(0, n, bs))
        return out

    def _start_epoch(self) -> None:
        self._queues = [self._passes(i) for i in range(len(self.names))]
        self._cursor = [0] * len(self.names)
        self._deficit = [0] * len(self.names)
        self.epoch += 1

    def next_batch(self) -> tuple[str, list[dict]]:
        if all(self._cursor[i] >= len(self._queues[i])
               for i in range(len(self.names))):
            self._start_epoch()
        active = [i for i in range(len(self.names))
                  if self._cursor[i] < len(self._queues[i])]
        # Deficit (Bresenham) scheduling over the ACTIVE sources: rates are
        # re-normalized each slot, so a drained small source cannot starve
        # before it is drawn and integer arithmetic cannot drift.
        total = sum(self._rates[i] for i in active)
        for i in active:
            self._deficit[i] += self._rates[i]
        i = max(active, key=lambda j: (self._deficit[j], -j))  # ties: lowest index
        self._deficit[i] -= total
        out = self._queues[i][self._cursor[i]]
        self._cursor[i] += 1
        self.drawn[i] += 1
        return self.names[i], out

    def stats(self) -> dict:
        return {
            "names": list(self.names),
            "sizes": list(self.sizes),
            "weights": list(self.weights),
            "repeats": list(self.repeats),
            "batches_per_source": list(self.batches_per_source),
            "batches_per_epoch": self.batches_per_epoch,
            "drawn_batches": list(self.drawn),
            "epoch": self.epoch,
        }


def _validate_gold(q: dict) -> str | None:
    """Return a rejection reason, or None if the question gold is valid."""
    cands = q.get("candidates") or []
    if not cands:
        return "empty candidate set"
    gold = q.get("gold") or {}
    sup = q.get("supervision")
    if sup not in SUP_CODES:
        return f"unknown supervision {sup!r}"
    if "counts" in gold:
        cv = gold["counts"]
        if len(cv) != len(cands):
            return "counts length != n_candidates"
        if any((not isinstance(x, (int, float))) or (not math.isfinite(x)) or x < 0
               for x in cv):
            return "non-finite or negative counts"
        if sum(cv) <= 0:
            return "zero total trials"
        if sup == "binomial_counts" and len(cands) != 2:
            return "binomial_counts requires exactly 2 candidates"
    elif "distribution" in gold:
        dv = gold["distribution"]
        if len(dv) != len(cands):
            return "distribution length != n_candidates"
        if any((not isinstance(x, (int, float))) or (not math.isfinite(x)) or x < 0
               for x in dv):
            return "non-finite or negative distribution"
        if sum(dv) <= 0:
            return "zero-sum distribution"
    else:
        return "gold has neither distribution nor counts"
    w = q.get("weight")
    if not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0:
        return "non-finite or negative weight"
    return None


def make_collate(tokenizer, max_len: int = 512, max_state_tokens: int = 256,
                 encoder_impl: str = "path",
                 tree_max_mask_bytes: int = DEFAULT_MAX_MASK_BYTES,
                 tree_attention_impl: str = "auto"):
    """Collate a list of state dicts into a padded path batch.

    Returns a dict of tensors:
      input_ids [P, L], attention_mask [P, L], cand_end_pos [P]
      q_index [P], cand_index [P]         (path -> question/candidate slot)
      cand_mask [Bq, Cmax] bool
      target_dist [Bq, Cmax] float        (normalized; from counts if given)
      counts [Bq, Cmax] float, trials [Bq]
      sup_type [Bq] long, weight [Bq] float, ordinal [Bq] bool
      n_states, n_paths (ints, for logging)
      n_trunc_states / n_trunc_questions / n_trunc_cands (truncation stats)
      dropped (list of {id, reason} for rejected questions)
    """
    max_cand_tokens = max(8, max_len // 8)

    def collate(states: list[dict]) -> dict:
        input_ids = []
        cand_end_pos = []
        q_index = []
        cand_index = []
        q_rows = []
        # Question-level metadata for row i of the returned tensors, so a
        # caller never has to re-derive which question a row belongs to.
        # Re-derivation is unsound: the acceptance order depends on a
        # tokenizer-dependent drop ("no token budget for state") that no
        # gold-level check can predict. None marks the empty-batch
        # placeholder row.
        q_meta = []
        sources: dict[str, int] = {}
        dropped: list[dict] = []
        n_trunc_states = 0
        n_trunc_questions = 0
        n_trunc_cands = 0

        for si, s in enumerate(states):
            env = s.get("env") or s.get("source", "unknown")
            sid = s.get("id", "?")
            state_text = s["state"]
            if not state_text.startswith("[STATE]"):
                state_text = STATE_PREFIX + state_text
            s_ids = tokenizer(state_text, add_special_tokens=False)["input_ids"]
            if len(s_ids) > max_state_tokens:
                s_ids = s_ids[:max_state_tokens]  # keep the head (task/goal fields)
                n_trunc_states += 1
            for qi_in_state, q in enumerate(s["questions"]):
                reason = _validate_gold(q)
                if reason is not None:
                    dropped.append({"id": sid, "reason": reason})
                    continue
                sources[env] = sources.get(env, 0) + 1
                qi = len(q_rows)
                q_ids = tokenizer(QUESTION_PREFIX + q["text"],
                                  add_special_tokens=False)["input_ids"]
                cands = q["candidates"]
                cand_ids = []
                for c in cands:
                    c_ids = tokenizer(CANDIDATE_PREFIX + str(c),
                                      add_special_tokens=False)["input_ids"]
                    if len(c_ids) > max_cand_tokens:
                        c_ids = c_ids[-max_cand_tokens:]
                        n_trunc_cands += 1
                    cand_ids.append(c_ids)
                # Shared prefix budget for the whole question: every
                # candidate is encoded on the identical state+question
                # context.
                max_c = max(len(c) for c in cand_ids)
                budget_sq = max_len - max_c
                if len(q_ids) > budget_sq - MIN_STATE_TOKENS:
                    q_ids = q_ids[:max(1, budget_sq - MIN_STATE_TOKENS)]
                    n_trunc_questions += 1
                state_budget = min(len(s_ids), budget_sq - len(q_ids))
                if state_budget < len(s_ids):
                    n_trunc_states += 1
                if state_budget <= 0:
                    dropped.append({"id": sid, "reason": "no token budget for state"})
                    continue
                prefix = s_ids[:state_budget] + q_ids
                row = {
                    "n_cands": len(cands),
                    "gold": q["gold"],
                    "sup_type": SUP_CODES[q["supervision"]],
                    "weight": float(q["weight"]),
                    "ordinal": bool(q.get("ordinal", False)),
                }
                q_rows.append(row)
                q_meta.append({
                    "id": sid,
                    "workflow": env,
                    "qtype": q.get("qtype") or ("score" if q.get("ordinal") else "other"),
                    "ordinal": bool(q.get("ordinal", False)),
                    "state_index": si,
                    "question_index": qi_in_state,
                    "gold": q["gold"],
                })
                for ci, c_ids in enumerate(cand_ids):
                    ids = prefix + c_ids
                    input_ids.append(ids)
                    cand_end_pos.append(len(ids) - 1)
                    q_index.append(qi)
                    cand_index.append(ci)

        if not input_ids:  # everything dropped; keep the batch well-formed
            input_ids = [[tokenizer.eos_token_id or 0]]
            cand_end_pos = [0]
            q_index = [0]
            cand_index = [0]
            q_rows = [{
                "n_cands": 1,
                "gold": {"distribution": [1.0]},
                "sup_type": SUP_CODES["empirical"],
                "weight": 0.0,
                "ordinal": False,
            }]
            q_meta = [None]  # placeholder: not backed by any real question

        # pad paths
        P = len(input_ids)
        L = max(len(x) for x in input_ids)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        ids_t = torch.full((P, L), pad_id, dtype=torch.long)
        mask_t = torch.zeros(P, L, dtype=torch.long)
        for i, ids in enumerate(input_ids):
            ids_t[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            mask_t[i, :len(ids)] = 1

        # question-level tensors
        Bq = len(q_rows)
        Cmax = max(r["n_cands"] for r in q_rows)
        cand_mask = torch.zeros(Bq, Cmax, dtype=torch.bool)
        target_dist = torch.zeros(Bq, Cmax, dtype=torch.float)
        counts = torch.zeros(Bq, Cmax, dtype=torch.float)
        trials = torch.zeros(Bq, dtype=torch.float)
        sup_type = torch.zeros(Bq, dtype=torch.long)
        weight = torch.zeros(Bq, dtype=torch.float)
        ordinal = torch.zeros(Bq, dtype=torch.bool)

        for qi, r in enumerate(q_rows):
            c = r["n_cands"]
            cand_mask[qi, :c] = True
            sup_type[qi] = r["sup_type"]
            weight[qi] = r["weight"]
            ordinal[qi] = r["ordinal"]
            gold = r["gold"]
            if "counts" in gold:
                cv = torch.tensor(gold["counts"], dtype=torch.float)
                counts[qi, :c] = cv
                trials[qi] = cv.sum()
                if trials[qi] > 0:
                    target_dist[qi, :c] = cv / trials[qi]
            else:
                dv = torch.tensor(gold["distribution"], dtype=torch.float)
                target_dist[qi, :c] = dv / dv.sum().clamp_min(1e-8)

        return {
            "input_ids": ids_t,
            "attention_mask": mask_t,
            "cand_end_pos": torch.tensor(cand_end_pos, dtype=torch.long),
            "q_index": torch.tensor(q_index, dtype=torch.long),
            "cand_index": torch.tensor(cand_index, dtype=torch.long),
            "cand_mask": cand_mask,
            "target_dist": target_dist,
            "counts": counts,
            "trials": trials,
            "sup_type": sup_type,
            "weight": weight,
            "ordinal": ordinal,
            # Question metadata parallel to row 0..Bq-1 (None = placeholder).
            # Non-tensor, so batch_to_device/pin_batch_memory pass it through.
            "q_meta": q_meta,
            "n_states": len(states),
            "n_paths": P,
            "n_valid_tokens": sum(map(len, input_ids)),
            "n_questions": Bq,
            "n_trunc_states": n_trunc_states,
            "n_trunc_questions": n_trunc_questions,
            "n_trunc_cands": n_trunc_cands,
            "n_dropped": len(dropped),
            "dropped": dropped,
            "sources": sources,
        }
    if encoder_impl == "tree":
        return with_tree_collate(collate, attention_impl=tree_attention_impl,
                                 max_mask_bytes=tree_max_mask_bytes)
    return collate
