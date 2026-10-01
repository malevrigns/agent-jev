"""Phase-1 synthetic data generator for AgentJev (v2).

Known-distribution probability environments: biased coin, loaded dice,
urn draws, card draws, small Markov chains, Bayesian update scenarios.
Every sample is generated programmatically with an exact gold
distribution or exact counts, covering the agent question families
(completion / evidence / progress / retry / risk / stop) as
mathematized decision questions.

v2 fixes (see REVIEW.md):
  - beta_prob_gt_half returned P(p <= 0.5); now returns P(p > 0.5).
  - coin uses one unified generative model: draw prior Beta(a0, b0),
    draw p ~ Beta(a0, b0), then draw all observed/audit counts from p.
  - multi-step coin predictions use the Beta-binomial posterior
    predictive (future flips share one unknown p), not Binomial(R, E[p]).
  - dice serializes the bias weight, so identical inputs always have
    identical gold.
  - Markov transition rows and Bayes prior/likelihoods are quantized
    and renormalized BEFORE serialization; gold is computed from the
    exact values the model can see. The Bayes stop threshold (0.8) and
    the risk-level bin edges are serialized into the state text.
  - completion explicitly tracks achieved_goal and covers both success
    and failure termination (no longer constantly FALSE for
    coin/urn/cards).
  - stop/retry/progress questions are redefined as explicit events:
      stop     = "continuing for the remaining R steps reaches the goal"
      retry    = "the next single action succeeds" (predictive event)
      progress = "current progress level" (deterministic, achieved/goal)
    Bayes retry has no exact event without second-test likelihoods, so
    it is downgraded to supervision="heuristic" (weight 0.2).

Output JSONL schema (one line per state):
{
  "id": "synth-000001",
  "env": "coin",
  "state": "[STATE] ...",
  "questions": [
    {"family": "stop", "qtype": "boolean",
     "text": "...", "candidates": ["TRUE", "FALSE"],
     "gold": {"distribution": [0.93, 0.07]},
     "supervision": "known_distribution", "weight": 1.0, "ordinal": false},
    {"family": "evidence", "qtype": "choice",
     "text": "...", "candidates": ["1", "2", "3"],
     "gold": {"counts": [3, 11, 6]},
     "supervision": "multiclass_counts", "weight": 0.4, "ordinal": false}
  ]
}

gold: exactly one of
  "distribution": list[float] summing to 1 (per candidate)
  "counts":       list[int] per candidate (binomial: [k, n-k])
Supervision: known_distribution | deterministic (weight 1.0),
  binomial_counts | multiclass_counts (weight by trials),
  heuristic (weight 0.2; soft policy label, not an exact event),
  teacher reserved for later phases (weight 0.1-0.3).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

# --------------------------------------------------------------------------
# exact-math helpers
# --------------------------------------------------------------------------

def binom_pmf(k: int, n: int, p: float) -> float:
    if k < 0 or k > n:
        return 0.0
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def binom_tail_at_least(k: int, n: int, p: float) -> float:
    return sum(binom_pmf(i, n, p) for i in range(k, n + 1))


def beta_prob_gt_half(a: int, b: int) -> float:
    """P(p > 0.5) for Beta(a, b) with integer a, b (exact identity).

    P(p > 1/2) = 2^{-(a+b-1)} * sum_{j=0}^{a-1} C(a+b-1, j).
    e.g. Beta(2, 1) -> 0.75.
    """
    n = a + b - 1
    return sum(math.comb(n, j) for j in range(0, a)) / 2.0 ** n


def beta_binom_pmf(h: int, R: int, a: float, b: float) -> float:
    """P(H = h) for h successes in R future trials with shared p and a
    Beta(a, b) posterior on p (Beta-binomial posterior predictive)."""
    if h < 0 or h > R:
        return 0.0
    lg = (math.lgamma(a + h) + math.lgamma(b + R - h) - math.lgamma(a + b + R)
          - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)))
    return math.comb(R, h) * math.exp(lg)


def beta_binom_tail_at_least(k: int, R: int, a: float, b: float) -> float:
    """P(at least k successes in R future trials | Beta(a, b) posterior)."""
    if k <= 0:
        return 1.0
    if R <= 0:
        return 0.0
    return sum(beta_binom_pmf(h, R, a, b) for h in range(k, R + 1))


def hypergeom_prob_at_least_one(hit: int, miss: int, draws: int) -> float:
    """P(at least one hit) drawing `draws` without replacement."""
    if draws <= 0:
        return 0.0
    total = hit + miss
    if draws > miss:
        return 1.0
    return 1.0 - math.comb(miss, draws) / math.comb(total, draws)


def mat_pow(T: list[list[float]], h: int) -> list[list[float]]:
    n = len(T)
    R = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    B = [row[:] for row in T]
    while h:
        if h & 1:
            R = [[sum(R[i][k] * B[k][j] for k in range(n)) for j in range(n)]
                 for i in range(n)]
        B = [[sum(B[i][k] * B[k][j] for k in range(n)) for j in range(n)]
             for i in range(n)]
        h >>= 1
    return R


def _round_dist(d: list[float], nd: int = 4) -> list[float]:
    s = sum(d)
    return [round(x / s, nd) for x in d]


def _quantize_stochastic(row: list[float], nd: int = 2) -> list[float]:
    """Normalize, then round a probability row to `nd` decimals with the
    largest-remainder method so the result is non-negative and sums to
    exactly 1. The returned values are what gets serialized, so gold
    computed from them matches what the model can see."""
    s = sum(row)
    if s <= 0 or not row:
        raise ValueError("cannot quantize a non-positive-sum row")
    scale = 10 ** nd
    norm = [x / s for x in row]
    base = [math.floor(x * scale) for x in norm]
    rem = int(round(scale - sum(base)))
    order = sorted(range(len(row)),
                   key=lambda j: norm[j] * scale - base[j], reverse=True)
    for j in order[:rem]:
        base[j] += 1
    return [b / scale for b in base]


def _counts_weight(trials: int) -> float:
    return round(min(1.0, trials / 50.0), 2)


def _q(family, qtype, text, candidates, gold, supervision, weight, ordinal=False):
    return {
        "family": family,
        "qtype": qtype,
        "text": text,
        "candidates": candidates,
        "gold": gold,
        "supervision": supervision,
        "weight": weight,
        "ordinal": ordinal,
    }


BOOL = ["TRUE", "FALSE"]
PROGRESS_LEVELS = ["0-24%", "25-49%", "50-74%", "75-99%", "100%"]
RISK_LEVELS = ["low", "medium", "high"]
RISK_BINS_TEXT = "risk_bins: low if fail_prob<1/3, medium if <2/3, else high"


def _level_of(frac: float) -> int:
    if frac >= 1.0:
        return 4
    return min(3, int(frac * 4))


def _level_onehot(frac: float) -> list[float]:
    d = [0.0] * 5
    d[_level_of(frac)] = 1.0
    return d


def _risk_gold(fail_prob: float) -> list[float]:
    lvl = 0 if fail_prob < 1 / 3 else (1 if fail_prob < 2 / 3 else 2)
    gold = [0.0] * 3
    gold[lvl] = 1.0
    return gold


# --------------------------------------------------------------------------
# environments
# --------------------------------------------------------------------------

def gen_coin(rng: random.Random) -> dict:
    # Unified world model: prior -> p -> all counts.
    a0, b0 = rng.randint(1, 4), rng.randint(1, 4)
    p = rng.betavariate(a0, b0)
    n = rng.randint(0, 20)
    k = sum(1 for _ in range(n) if rng.random() < p)
    K = rng.randint(1, k + 6)          # target total heads (may be <= k)
    achieved = k >= K                  # explicit success termination
    R = rng.randint(0, 12)             # remaining flips allowed (0 = failed if not achieved)
    post_a, post_b = a0 + k, b0 + (n - k)
    p_next = post_a / (post_a + post_b)
    if achieved:
        p_goal = 1.0
    else:
        # Beta-binomial posterior predictive: future flips share one p.
        p_goal = beta_binom_tail_at_least(K - k, R, post_a, post_b)

    state = (f"[STATE] env=biased_coin; prior=Beta({a0},{b0}); observed_flips={n}; "
             f"heads={k}; tails={n-k}; goal=collect {K} heads in total; "
             f"flips_left={R}; last_action=flip; {RISK_BINS_TEXT}")
    qs = []

    # completion (deterministic; TRUE/FALSE both occur)
    qs.append(_q("completion", "boolean",
                 "Has the goal been completed?",
                 BOOL, {"distribution": [1.0, 0.0] if achieved else [0.0, 1.0]},
                 "deterministic", 1.0))

    # retry (predictive event: next single flip lands heads)
    qs.append(_q("retry", "boolean",
                 "If we flip the coin once more, will it land heads?",
                 BOOL, {"distribution": _round_dist([p_next, 1 - p_next])},
                 "known_distribution", 1.0))

    # stop (explicit event: goal reached within the remaining R flips)
    qs.append(_q("stop", "boolean",
                 f"If we keep flipping for the remaining {R} flip(s), will the goal be reached?",
                 BOOL, {"distribution": _round_dist([p_goal, 1 - p_goal])},
                 "known_distribution", 1.0))

    # evidence (deterministic: posterior confidence about bias direction)
    conf = max(beta_prob_gt_half(post_a, post_b), 1 - beta_prob_gt_half(post_a, post_b))
    need = conf < 0.9
    qs.append(_q("evidence", "boolean",
                 "Is more evidence needed to decide whether the coin favors heads?",
                 BOOL, {"distribution": [1.0, 0.0] if need else [0.0, 1.0]},
                 "deterministic", 1.0))

    # progress (deterministic current level)
    qs.append(_q("progress", "score",
                 "What is the current progress toward the goal as a percentage level?",
                 PROGRESS_LEVELS, {"distribution": _level_onehot(min(1.0, k / K))},
                 "deterministic", 1.0, ordinal=True))

    # risk (deterministic level of failure probability)
    qs.append(_q("risk", "score",
                 "What is the risk level of failing the goal?",
                 RISK_LEVELS, {"distribution": _risk_gold(1.0 - p_goal)},
                 "deterministic", 1.0, ordinal=True))

    # counts supervision: audit flips drawn from the same p
    m = rng.choice([10, 20, 30, 50, 80])
    h = sum(1 for _ in range(m) if rng.random() < p)
    qs.append(_q("retry", "boolean",
                 f"An audit ran {m} test flips. Will the next flip land heads?",
                 BOOL, {"counts": [h, m - h]},
                 "binomial_counts", _counts_weight(m)))

    return {"env": "coin", "state": state, "questions": qs}


def gen_dice(rng: random.Random) -> dict:
    faces = list(range(1, 7))
    if rng.random() < 0.5:
        probs = [1 / 6] * 6
        desc = "fair die (all faces weight 1)"
    else:
        hi = rng.choice(faces)
        w_hi = rng.choice([2.0, 3.0, 4.0])
        w = [1.0] * 6
        w[hi - 1] = w_hi
        s = sum(w)
        probs = [x / s for x in w]
        # bias strength is serialized: identical input => identical gold
        desc = f"loaded die favoring face {hi} with weight {w_hi:g} (other faces weight 1)"
    target_face = rng.choice(faces)
    p_t = probs[target_face - 1]
    G = rng.randint(1, 4)               # goal: total hits of target face
    hits = rng.randint(0, G + 1)        # hits so far (may already achieve)
    achieved = hits >= G
    R = rng.randint(0, 10)              # rolls left (0 = failed if not achieved)
    if achieved:
        p_goal = 1.0
    else:
        p_goal = binom_tail_at_least(G - hits, R, p_t) if R > 0 else 0.0

    state = (f"[STATE] env=dice; die={desc}; target=roll face {target_face} "
             f"{G} time(s) in total (have {hits}); rolls_left={R}; "
             f"last_action=roll; {RISK_BINS_TEXT}")
    qs = []

    qs.append(_q("completion", "boolean",
                 "Has the goal been completed?",
                 BOOL, {"distribution": [1.0, 0.0] if achieved else [0.0, 1.0]},
                 "deterministic", 1.0))
    qs.append(_q("retry", "choice",
                 "Which face will the next roll show?",
                 [str(f) for f in faces], {"distribution": _round_dist(probs)},
                 "known_distribution", 1.0))
    qs.append(_q("stop", "boolean",
                 f"If we keep rolling for the remaining {R} roll(s), will the target be reached?",
                 BOOL, {"distribution": _round_dist([p_goal, 1 - p_goal])},
                 "known_distribution", 1.0))
    qs.append(_q("progress", "score",
                 "What is the current progress toward the target as a percentage level?",
                 PROGRESS_LEVELS, {"distribution": _level_onehot(min(1.0, hits / G))},
                 "deterministic", 1.0, ordinal=True))
    qs.append(_q("risk", "score",
                 "What is the risk level of missing the target?",
                 RISK_LEVELS, {"distribution": _risk_gold(1.0 - p_goal)},
                 "deterministic", 1.0, ordinal=True))

    m = rng.choice([12, 24, 36, 60])
    counts = [0] * 6
    for _ in range(m):
        u, acc = rng.random(), 0.0
        for i, pr in enumerate(probs):
            acc += pr
            if u <= acc:
                counts[i] += 1
                break
    qs.append(_q("evidence", "choice",
                 f"An audit rolled the die {m} times. Which face will the next roll show?",
                 [str(f) for f in faces], {"counts": counts},
                 "multiclass_counts", _counts_weight(m)))

    return {"env": "dice", "state": state, "questions": qs}


def gen_urn(rng: random.Random) -> dict:
    colors = ["red", "blue", "green"]
    comp = [rng.randint(1, 8) for _ in colors]
    total0 = sum(comp)
    drawn = [0, 0, 0]
    d = rng.randint(0, min(5, total0 - 1))
    for _ in range(d):
        pool = [c for c in range(3) for _ in range(comp[c] - drawn[c])]
        if pool:
            drawn[rng.choice(pool)] += 1
    rem = [comp[c] - drawn[c] for c in range(3)]
    tot = sum(rem)
    target_c = rng.randrange(3)
    G = rng.randint(1, 4)                       # goal: total target balls
    achieved = drawn[target_c] >= G             # explicit success termination
    need = max(0, G - drawn[target_c])
    if achieved:
        draws_left = rng.randint(0, min(tot, 6))
        p_goal = 1.0
    else:
        draws_left = rng.randint(0, min(tot, need + 6))  # 0 = failed
        # P(at least `need` of target color in draws_left without replacement)
        hit, miss = rem[target_c], tot - rem[target_c]
        p_goal = 0.0
        for kk in range(need, min(hit, draws_left) + 1):
            if draws_left - kk <= miss:
                p_goal += (math.comb(hit, kk) * math.comb(miss, draws_left - kk)
                           / math.comb(tot, draws_left))
    p_next = rem[target_c] / tot

    state = (f"[STATE] env=urn; contents: red={comp[0]} blue={comp[1]} green={comp[2]}; "
             f"drawn_without_replacement: red={drawn[0]} blue={drawn[1]} green={drawn[2]}; "
             f"goal=collect {G} {colors[target_c]} ball(s) in total "
             f"(have {drawn[target_c]}); draws_left={draws_left}; "
             f"last_action=draw; {RISK_BINS_TEXT}")
    qs = []

    qs.append(_q("completion", "boolean",
                 "Has the goal been completed?",
                 BOOL, {"distribution": [1.0, 0.0] if achieved else [0.0, 1.0]},
                 "deterministic", 1.0))
    qs.append(_q("retry", "boolean",
                 f"If we draw once more, will it be a {colors[target_c]} ball?",
                 BOOL, {"distribution": _round_dist([p_next, 1 - p_next])},
                 "known_distribution", 1.0))
    qs.append(_q("stop", "boolean",
                 f"If we keep drawing for the remaining {draws_left} draw(s), will the goal be reached?",
                 BOOL, {"distribution": _round_dist([p_goal, 1 - p_goal])},
                 "known_distribution", 1.0))
    qs.append(_q("progress", "score",
                 "What is the current progress toward the goal as a percentage level?",
                 PROGRESS_LEVELS,
                 {"distribution": _level_onehot(min(1.0, drawn[target_c] / G))},
                 "deterministic", 1.0, ordinal=True))
    qs.append(_q("risk", "score",
                 "What is the risk level of failing the goal?",
                 RISK_LEVELS, {"distribution": _risk_gold(1.0 - p_goal)},
                 "deterministic", 1.0, ordinal=True))
    return {"env": "urn", "state": state, "questions": qs}


def gen_cards(rng: random.Random) -> dict:
    suits = ["hearts", "diamonds", "clubs", "spades"]
    target = rng.choice(suits)
    drawn_t = rng.randint(0, 2)    # target-suit cards we already drew
    drawn_o = rng.randint(0, 12)   # other cards we already drew
    hit = 13 - drawn_t
    miss = 39 - drawn_o
    tot = hit + miss
    achieved = drawn_t >= 1        # explicit success termination
    draws_left = rng.randint(0, min(10, tot))  # 0 = failed if not achieved
    if achieved:
        p_one = 1.0
    else:
        p_one = hypergeom_prob_at_least_one(hit, miss, draws_left)
    p_next = hit / tot

    state = (f"[STATE] env=cards; deck=standard 52; already drawn: {drawn_t} {target} "
             f"and {drawn_o} other card(s); remaining={tot}; "
             f"goal=draw at least one {target}; draws_left={draws_left}; "
             f"last_action=draw; {RISK_BINS_TEXT}")
    qs = []

    qs.append(_q("completion", "boolean",
                 "Has the goal been completed?",
                 BOOL, {"distribution": [1.0, 0.0] if achieved else [0.0, 1.0]},
                 "deterministic", 1.0))
    qs.append(_q("retry", "boolean",
                 f"If we draw once more, will it be a {target}?",
                 BOOL, {"distribution": _round_dist([p_next, 1 - p_next])},
                 "known_distribution", 1.0))
    qs.append(_q("stop", "boolean",
                 f"If we keep drawing for the remaining {draws_left} draw(s), will we get a {target}?",
                 BOOL, {"distribution": _round_dist([p_one, 1 - p_one])},
                 "known_distribution", 1.0))
    qs.append(_q("risk", "score",
                 "What is the risk level of failing to draw the target suit?",
                 RISK_LEVELS, {"distribution": _risk_gold(1.0 - p_one)},
                 "deterministic", 1.0, ordinal=True))
    return {"env": "cards", "state": state, "questions": qs}


def gen_markov(rng: random.Random) -> dict:
    names = ["idle", "working", "blocked"]
    n = 3
    # random rows with a clear diagonal tendency; quantize + renormalize
    # FIRST so gold is computed from exactly what the model can see.
    T = []
    for i in range(n):
        w = [rng.uniform(0.05, 0.3) for _ in range(n)]
        w[i] += rng.uniform(0.4, 0.8)
        s = sum(w)
        T.append(_quantize_stochastic([x / s for x in w], nd=2))
    cur = rng.randrange(n)
    h = rng.randint(1, 5)
    Th = mat_pow(T, h)
    end_dist = Th[cur]

    # P(ever reach blocked within h steps) via subchain on {idle, working}
    sub = [row[:2] for row in T[:2]]
    vec = [1.0 if i == cur else 0.0 for i in range(2)] if cur < 2 else [0.0, 0.0]
    if cur == 2:
        p_block = 1.0
    else:
        for _ in range(h):
            vec = [sum(vec[k] * sub[k][j] for k in range(2)) for j in range(2)]
        survive = sum(vec)
        p_block = 1.0 - survive

    state = (f"[STATE] env=markov; states=idle/working/blocked; "
             f"transition_matrix=[[{T[0][0]:.2f},{T[0][1]:.2f},{T[0][2]:.2f}],"
             f"[{T[1][0]:.2f},{T[1][1]:.2f},{T[1][2]:.2f}],"
             f"[{T[2][0]:.2f},{T[2][1]:.2f},{T[2][2]:.2f}]]; "
             f"current_state={names[cur]}; steps_left={h}; goal=end in working; "
             f"last_action=step; {RISK_BINS_TEXT}")
    qs = []

    qs.append(_q("evidence", "choice",
                 f"What will the environment state be after {h} step(s)?",
                 names, {"distribution": _round_dist(end_dist)},
                 "known_distribution", 1.0))
    p_goal = end_dist[1]
    qs.append(_q("stop", "boolean",
                 f"If we let the process run for the remaining {h} step(s), "
                 "will it end in the working state?",
                 BOOL, {"distribution": _round_dist([p_goal, 1 - p_goal])},
                 "known_distribution", 1.0))
    qs.append(_q("retry", "boolean",
                 "If we take one more step, will it land in the working state?",
                 BOOL, {"distribution": _round_dist([T[cur][1], 1 - T[cur][1]])},
                 "known_distribution", 1.0))
    qs.append(_q("risk", "score",
                 "What is the risk level of hitting the blocked state?",
                 RISK_LEVELS, {"distribution": _risk_gold(p_block)},
                 "deterministic", 1.0, ordinal=True))

    m = rng.choice([10, 20, 40, 60])
    counts = [0] * 3
    for _ in range(m):
        s = cur
        for _ in range(h):
            u, acc = rng.random(), 0.0
            for j, pr in enumerate(T[s]):
                acc += pr
                if u <= acc:
                    s = j
                    break
        counts[s] += 1
    qs.append(_q("progress", "choice",
                 f"A simulation ran {m} rollouts of {h} step(s). Where do rollouts end?",
                 names, {"counts": counts},
                 "multiclass_counts", _counts_weight(m)))
    return {"env": "markov", "state": state, "questions": qs}


def gen_bayes(rng: random.Random) -> dict:
    hyps = ["sensor_fault", "network_fault", "software_bug"]
    # quantize + renormalize the visible parameters FIRST; gold is
    # computed from exactly these values.
    prior = _quantize_stochastic([rng.uniform(0.1, 1.0) for _ in hyps], nd=2)
    like = [round(rng.uniform(0.05, 0.95), 2) for _ in hyps]
    post = [prior[i] * like[i] for i in range(3)]
    s = sum(post)
    post = [x / s for x in post]
    max_post = max(post)
    stop_threshold = 0.8
    enough = max_post >= stop_threshold

    state = (f"[STATE] env=bayes_diagnosis; hypotheses=sensor_fault/network_fault/software_bug; "
             f"prior=[{prior[0]:.2f},{prior[1]:.2f},{prior[2]:.2f}]; "
             f"evidence=alarm_triggered; likelihoods=[{like[0]:.2f},{like[1]:.2f},{like[2]:.2f}]; "
             f"stop_rule: act when max posterior >= {stop_threshold:.2f}; "
             f"tests_left={rng.randint(1, 4)}; last_action=run_diagnostic; {RISK_BINS_TEXT}")
    qs = []

    qs.append(_q("evidence", "choice",
                 "Which root cause is most consistent with the evidence?",
                 hyps, {"distribution": _round_dist(post)},
                 "known_distribution", 1.0))
    qs.append(_q("stop", "boolean",
                 "Do we have enough evidence to stop investigating and act on the diagnosis?",
                 BOOL, {"distribution": [1.0, 0.0] if enough else [0.0, 1.0]},
                 "deterministic", 1.0))
    # No exact value-of-information event exists without second-test
    # likelihoods; keep the soft heuristic label, downgraded.
    qs.append(_q("retry", "boolean",
                 "Should we run another diagnostic test?",
                 BOOL, {"distribution": _round_dist([1 - max_post, max_post])},
                 "heuristic", 0.2))
    qs.append(_q("completion", "boolean",
                 "Is the diagnosis task complete?",
                 BOOL, {"distribution": [1.0, 0.0] if enough else [0.0, 1.0]},
                 "deterministic", 1.0))
    qs.append(_q("risk", "score",
                 "What is the risk level that the current diagnosis is wrong?",
                 RISK_LEVELS, {"distribution": _risk_gold(1.0 - max_post)},
                 "deterministic", 1.0, ordinal=True))
    return {"env": "bayes", "state": state, "questions": qs}


GENERATORS = [gen_coin, gen_dice, gen_urn, gen_cards, gen_markov, gen_bayes]


def generate(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    samples = []
    for i in range(n):
        gen = GENERATORS[i % len(GENERATORS)]
        s = gen(rng)
        s["id"] = f"synth-{seed}-{i:06d}"
        # keep state first in the dict for readability
        samples.append({
            "id": s["id"],
            "env": s["env"],
            "state": s["state"],
            "questions": s["questions"],
        })
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    samples = generate(args.n, args.seed)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    nq = sum(len(s["questions"]) for s in samples)
    sup = {}
    fam = {}
    for s in samples:
        for q in s["questions"]:
            sup[q["supervision"]] = sup.get(q["supervision"], 0) + 1
            fam[q["family"]] = fam.get(q["family"], 0) + 1
    print(f"wrote {len(samples)} states / {nq} questions to {args.out}")
    print(f"supervision mix: {sup}")
    print(f"family mix: {fam}")


if __name__ == "__main__":
    main()
