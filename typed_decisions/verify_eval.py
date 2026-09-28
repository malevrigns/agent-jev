import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def calc_metrics(preds):
    correct = []
    ces = []
    briers = []
    confidences = []
    score_maes = []
    for r in preds:
        p = np.array(r['probs'], dtype=np.float64)
        y = np.array(r['target'], dtype=np.float64)
        pred_idx = int(np.argmax(p))
        gold_idx = int(np.argmax(y))
        is_corr = float(pred_idx == gold_idx)
        correct.append(is_corr)
        ce = float(-np.sum(y * np.log(np.clip(p, 1e-12, 1.0))))
        ces.append(ce)
        brier = float(np.sum((p - y) ** 2))
        briers.append(brier)
        confidences.append(float(p.max()))
        if r['type'] == 'score':
            idx = np.arange(len(p))
            score_maes.append(abs(float(np.sum(idx * p) - np.sum(idx * y))))

    ece = 0.0
    n = len(preds)
    for lo in np.arange(0, 1, 0.1):
        hi = lo + 0.1 if lo < 0.9 else 1.0000001
        ix = [i for i, c in enumerate(confidences) if lo <= c < hi]
        if ix:
            bin_acc = float(np.mean([correct[i] for i in ix]))
            bin_conf = float(np.mean([confidences[i] for i in ix]))
            ece += (len(ix) / n) * abs(bin_acc - bin_conf)

    return {
        'count': n,
        'accuracy': float(np.mean(correct)),
        'correct_count': int(np.sum(correct)),
        'cross_entropy': float(np.mean(ces)),
        'brier': float(np.mean(briers)),
        'ece': float(ece),
        'score_mae': float(np.mean(score_maes)) if score_maes else None
    }

files = [
    ('Phase 4 Baseline', 'phase4_test_predictions.json'),
    ('AgentJev (Uncalibrated)', 'test_predictions.json'),
    ('AgentJev (Calibrated)', 'test_calibrated_predictions.json'),
    ('Laya Specialist', 'laya_test_predictions.json')
]

for name, fname in files:
    p = ROOT / fname
    if not p.exists():
        p = ROOT / 'agentjev_v1' / fname
    if p.exists():
        preds = json.loads(p.read_text(encoding='utf-8'))
        m = calc_metrics(preds)
        print("=== " + name + " ===")
        print("  Accuracy:      {:.2f}% ({}/{})".format(m['accuracy']*100, m['correct_count'], m['count']))
        print("  Cross-Entropy: {:.4f}".format(m['cross_entropy']))
        print("  Brier Score:   {:.4f}".format(m['brier']))
        print("  ECE:           {:.4f}".format(m['ece']))
        print("  Score MAE:     {:.4f}".format(m['score_mae']))
    else:
        print("Not found: " + str(p))
