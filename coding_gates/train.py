"""Fine-tune AgentJev v1 on coding-gate data with Typed Decisions replay, through the serving encoder.

The bottom layers and embeddings stay frozen (their Linear weights in BF16 exactly as served);
the top --train-layers decoder layers, the final norm and the candidate head train in FP32 under
BF16 autocast. Forward passes use jev_service.prefix.encode, the state-level trie the server runs.
Loss: soft cross-entropy + 0.1 * Brier per question (the v1 protocol's loss and learning rates).
The selection rule is written to protocol.json before step 1 and applied after the run.

Needs coding_gates/data (build_dataset.py) and typed_decisions/prepared (typed_decisions/prepare_data.py).
usage: python coding_gates/train.py [--model-dir models/agentjev-v1] [--out coding_gates/runs/gates_v1]
The candidate: --steps 900 --vary-wording --dev-paraphrase. That run peaked at 4,164 MiB of GPU memory
(4,350 MiB on the largest possible batch) and took about 30 minutes on an RTX 5060, evaluations included.
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from agentjev.model import AgentJevModel  # noqa: E402
from jev_service.contract import encode_paths, prepare  # noqa: E402
from jev_service.prefix import encode  # noqa: E402
from coding_gates.evaluate import PARAPHRASES  # noqa: E402
from coding_gates.wordings import vary  # noqa: E402

HERE = Path(__file__).resolve().parent
TYPED = ROOT / "typed_decisions" / "prepared"


def load_split(req_path, q_path, typed=False):
    reqs = [json.loads(l) for l in open(req_path, encoding="utf-8")]
    gold = {}
    for g in map(json.loads, open(q_path, encoding="utf-8")):
        gold[g["id"]] = g["target"]
    return [(r, [gold[f"{r['id']}:{q['id']}"] for q in r["questions"]]) for r in reqs]


def with_paraphrases(request, targets):
    """A dev request whose gate is asked again in each evaluate.PARAPHRASES wording."""
    gate = targets[[q["id"] for q in request["questions"]].index("passed")]
    extra = [dict({k: v for k, v in p.items() if k != "inverted"}, id="para_" + p["id"]) for p in PARAPHRASES]
    return (dict(request, questions=request["questions"] + extra),
            targets + [gate[::-1] if p.get("inverted") else gate for p in PARAPHRASES])


def forward(model, engine, tok, items):
    """Logits [Q, Cmax] and targets for a list of (request, targets) through the serving trie."""
    prepared = prepare({"requests": [dict(r, id=str(i)) for i, (r, _) in enumerate(items)]})
    paths, locations, rows = encode_paths(prepared, tok, 2048)
    states = [si for si, req in enumerate(prepared) for _ in req["questions"]]
    vectors, mask, usage = encode(engine, paths, locations, rows, "auto", states)
    # Every state must go through the dense trie: the KV-cache fallback cannot carry gradients.
    assert usage["shared_prefix_states"] == len(prepared), usage
    logits = model._score(vectors, mask).float().masked_fill(~mask, float("-inf"))
    target = torch.zeros_like(logits)
    q = 0
    for _, targets in items:
        for t in targets:
            target[q, :len(t)] = torch.tensor(t, device=logits.device)
            q += 1
    return logits, mask, target, usage


def loss_fn(logits, mask, target):
    logp = torch.log_softmax(logits, -1).masked_fill(~mask, 0)
    p = logp.exp() * mask
    ce = -(target * logp).sum(-1)
    brier = ((p - target) ** 2).sum(-1)
    return (ce + .1 * brier).mean(), ce.mean()


@torch.no_grad()
def dev_soft_ce(model, engine, tok, data, batch=8):
    total, n, correct = 0.0, 0, 0
    model.eval()
    for i in range(0, len(data), batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, mask, target, _ = forward(model, engine, tok, data[i:i + batch])
        logp = torch.log_softmax(logits, -1).masked_fill(~mask, 0)
        total += float(-(target * logp).sum()); n += len(target)
        correct += int((logits.argmax(-1) == target.argmax(-1)).sum())
    model.train()
    return total / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "agentjev-v1"),
                    help="the v1 release: model.safetensors, config and tokenizer")
    ap.add_argument("--out", type=Path, default=HERE / "runs" / "gates_v1")
    ap.add_argument("--train-layers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--gate-per-step", type=int, default=4)
    ap.add_argument("--typed-per-step", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--dev-limit", type=int, default=0, help="use only the first N dev requests (smoke runs)")
    ap.add_argument("--largest-batch", action="store_true", help="smoke runs: train on the largest requests to find peak memory")
    ap.add_argument("--vary-wording", action="store_true", help="reword gate training requests (coding_gates/wordings.py)")
    ap.add_argument("--dev-paraphrase", action="store_true", help="select on the gate dev questions plus evaluate.PARAPHRASES")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); rng = random.Random(args.seed)

    gate_train = load_split(HERE / "data" / "train_requests.jsonl", HERE / "data" / "train_questions.jsonl")
    gate_dev = load_split(HERE / "data" / "dev_requests.jsonl", HERE / "data" / "dev_questions.jsonl")
    typed_train = load_split(TYPED / "train_requests.jsonl", TYPED / "train_questions.jsonl")
    typed_dev = load_split(TYPED / "dev_requests.jsonl", TYPED / "dev_questions.jsonl")
    if args.dev_limit:
        gate_dev, typed_dev = gate_dev[:args.dev_limit], typed_dev[:args.dev_limit]
    if args.dev_paraphrase:
        gate_dev = [with_paraphrases(r, t) for r, t in gate_dev]

    protocol = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    protocol.update(
        init=f"{args.model_dir}/model.safetensors sha256 " + hashlib.sha256(
            (Path(args.model_dir) / "model.safetensors").read_bytes()).hexdigest(),
        trainable=f"decoder layers {28 - args.train_layers}-27, final norm, candidate head",
        loss="soft cross-entropy + 0.1 * Brier, mean over questions; Typed Decisions replay in every step",
        wording=("each sampled gate request reworded by coding_gates/wordings.vary (each question keeps its "
                 "original wording with probability 0.25)") if args.vary_wording else "fixed build_dataset wording",
        selection=("Evaluate step 0 and every eval_every steps on gate dev and Typed Decisions dev. A step is "
                   "eligible if its Typed Decisions dev soft CE <= step 0's + 0.01. Choose the eligible step "
                   "with the lowest gate dev soft CE"
                   + (" (mean over every gate dev question, including each request's gate asked again in the "
                      "evaluate.PARAPHRASES wordings)" if args.dev_paraphrase else "")
                   + "; ties go to the earlier step. Test splits and FINAL_PARAPHRASES are not read."),
        data={"gate_train": len(gate_train), "gate_dev": len(gate_dev), "typed_train": len(typed_train),
              "typed_dev": len(typed_dev)})
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=1), encoding="utf-8")

    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AgentJevModel(args.model_dir, pretrained=False)
    model.load_state_dict(load_file(str(Path(args.model_dir) / "model.safetensors")), strict=True)
    backbone = model.path_encoder.backbone
    first_trainable = 28 - args.train_layers
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for i, layer in enumerate(backbone.layers):
        if i >= first_trainable:
            trainable += list(layer.parameters())
        else:  # frozen: store the BF16 cast autocast would make, as the server does
            for m in layer.modules():
                if isinstance(m, torch.nn.Linear):
                    m.to(torch.bfloat16)
    trainable += list(backbone.norm.parameters())
    head = [p for n, p in model.named_parameters() if not n.startswith("path_encoder.")]
    for p in trainable + head:
        p.requires_grad_(True)
    # Recompute each trainable layer in backward instead of storing its activations (8 GB GPU).
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.cuda().train()
    trained_names = {n for n, p in model.named_parameters() if p.requires_grad}
    engine = types.SimpleNamespace(torch=torch, device="cuda:0", path_batch=16, model=model,
                                   tokenizer=types.SimpleNamespace(pad_token_id=tok.pad_token_id))
    opt = torch.optim.AdamW([{"params": trainable, "lr": args.lr}, {"params": head, "lr": args.head_lr}], weight_decay=.01)
    warmup = 20
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1., (s + 1) / warmup) * .5 * (
        1 + math.cos(math.pi * max(0, s - warmup) / max(1, args.steps - warmup))))
    n_train = sum(p.numel() for p in trainable + head)
    print(f"trainable params {n_train/1e6:.1f}M | gate train {len(gate_train)} | typed train {len(typed_train)}", flush=True)

    events = open(args.out / "events.jsonl", "w", buffering=1)
    evals = []

    def evaluate(step):
        g_ce, g_acc = dev_soft_ce(model, engine, tok, gate_dev)
        t_ce, t_acc = dev_soft_ce(model, engine, tok, typed_dev)
        rec = {"step": step, "gate_dev_ce": g_ce, "gate_dev_acc": g_acc, "typed_dev_ce": t_ce, "typed_dev_acc": t_acc}
        evals.append(rec); events.write(json.dumps(rec) + "\n")
        # Only the trained tensors, in BF16 like the release; the rest is the v1 file unchanged.
        state = {k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in model.state_dict().items() if k in trained_names}
        save_file(state, str(args.out / f"step_{step:04d}.trained.safetensors"))
        print(json.dumps(rec), flush=True)

    evaluate(0)
    largest = (sorted(gate_train, key=lambda x: -len(json.dumps(x[0])))[:args.gate_per_step]
               + sorted(typed_train, key=lambda x: -len(json.dumps(x[0])))[:args.typed_per_step])
    start = time.monotonic()
    for step in range(1, args.steps + 1):
        if args.largest_batch:
            items = largest
        else:
            gate_items = rng.sample(gate_train, args.gate_per_step)
            if args.vary_wording:
                gate_items = [vary(r, t, rng) for r, t in gate_items]
            items = gate_items + rng.sample(typed_train, args.typed_per_step)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, mask, target, usage = forward(model, engine, tok, items)
        loss, ce = loss_fn(logits, mask, target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable + head, 1.0)
        opt.step(); sched.step()
        if step % 10 == 0 or step == args.steps:
            events.write(json.dumps({"step": step, "loss": float(loss), "ce": float(ce), "grad_norm": float(norm),
                                     "tokens": usage["backbone_input_tokens"], "seconds": round(time.monotonic() - start, 1),
                                     "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20)}) + "\n")
        if step % 50 == 0:
            print(f"step {step} loss {float(loss):.4f} ce {float(ce):.4f} {time.monotonic() - start:.0f}s "
                  f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB", flush=True)
        if step % args.eval_every == 0:
            evaluate(step)

    base = evals[0]["typed_dev_ce"]
    eligible = [e for e in evals if e["typed_dev_ce"] <= base + .01]
    chosen = min(eligible, key=lambda e: (e["gate_dev_ce"], e["step"]))
    (args.out / "selection.json").write_text(json.dumps({"chosen": chosen, "evals": evals}, indent=1), encoding="utf-8")
    full = load_file(str(Path(args.model_dir) / "model.safetensors"))
    full.update(load_file(str(args.out / f"step_{chosen['step']:04d}.trained.safetensors")))
    save_file(full, str(args.out / "selected.safetensors"))
    print("SELECTED", json.dumps(chosen), flush=True)


if __name__ == "__main__":
    main()
