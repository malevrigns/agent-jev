import json
import time
import urllib.request
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def load_data():
    req_path = ROOT / 'prepared/test_requests.jsonl'
    gold_path = ROOT / 'prepared/test_questions.jsonl'
    requests = [json.loads(s) for s in req_path.read_text(encoding='utf-8').splitlines()]
    gold_items = [json.loads(s) for s in gold_path.read_text(encoding='utf-8').splitlines()]
    gold_map = {item['id']: item for item in gold_items}
    return requests, gold_map

def run_benchmark(port=8149, name="AgentJev v1 (Port 8149)", batch_size=20):
    requests, gold_map = load_data()
    total_cases = len(requests)
    print("=" * 65)
    print("  🚀 开始实时全量 Benchmark: " + name)
    print("  📊 评测数据集: LocalLLaMA/typed-decisions (官方测试集)")
    print("  📋 测试规模: {} 个案例 / {} 道决策题".format(total_cases, len(gold_map)))
    print("  🌐 服务地址: http://127.0.0.1:{}/api/evaluate".format(port))
    print("=" * 65)

    all_predictions = []
    latencies = []
    start_total = time.time()

    for i in range(0, total_cases, batch_size):
        chunk = requests[i:i+batch_size]
        t0 = time.time()
        payload = json.dumps({'requests': chunk}).encode('utf-8')
        req = urllib.request.Request(
            "http://127.0.0.1:{}/api/evaluate".format(port),
            data=payload,
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            res_data = json.loads(resp.read().decode('utf-8'))
        dt = time.time() - t0
        latencies.append(dt)

        # Parse responses
        for case_req, case_res in zip(chunk, res_data['results']):
            case_id = case_req['id']
            for q_req, ans in zip(case_req['questions'], case_res['answers']):
                qid = case_id + ":" + q_req['id']
                gold = gold_map[qid]
                probs = list(ans['distribution'].values())
                all_predictions.append({
                    'id': qid,
                    'case_id': case_id,
                    'workflow': gold['workflow'],
                    'type': gold['question']['type'],
                    'target': gold['target'],
                    'probs': probs
                })

        done = min(i + batch_size, total_cases)
        pct = (done / total_cases) * 100
        avg_ms = (dt / len(chunk)) * 1000
        print("  [{:3d}/{:3d}] {:5.1f}% 完成 | 批次耗时: {:.2f}s | 单案例平均时延: {:.1f}ms".format(
            done, total_cases, pct, dt, avg_ms
        ))

    total_time = time.time() - start_total
    print("\n" + "=" * 65)
    print("  🏁 Benchmark 运行完毕！总用时: {:.2f} 秒，处理题目: {} 道".format(total_time, len(all_predictions)))
    print("=" * 65)

    # Compute metrics
    correct = []
    ces = []
    briers = []
    by_type = {}
    by_wf = {}

    for r in all_predictions:
        p = np.array(r['probs'], dtype=np.float64)
        y = np.array(r['target'], dtype=np.float64)
        is_corr = int(np.argmax(p) == np.argmax(y))
        ce = float(-np.sum(y * np.log(np.clip(p, 1e-12, 1.0))))
        br = float(np.sum((p - y) ** 2))

        correct.append(is_corr)
        ces.append(ce)
        briers.append(br)

        q_type = r['type']
        wf = r['workflow']
        if q_type not in by_type: by_type[q_type] = {'corr': [], 'n': 0}
        by_type[q_type]['corr'].append(is_corr)
        by_type[q_type]['n'] += 1

        if wf not in by_wf: by_wf[wf] = {'corr': [], 'n': 0}
        by_wf[wf]['corr'].append(is_corr)
        by_wf[wf]['n'] += 1

    overall_acc = np.mean(correct) * 100
    print("\n【总体核心指标】")
    print("  • 综合准确率 (Top-1 Acc):  {:.2f}% ({}/{})".format(overall_acc, sum(correct), len(correct)))
    print("  • 软交叉熵损失 (Cross-Entropy): {:.4f}".format(np.mean(ces)))
    print("  • Brier 概率分布误差:        {:.4f}".format(np.mean(briers)))
    print("  • 平均单案例耗时:             {:.1f} ms".format((total_time / total_cases) * 1000))

    print("\n【按任务类型细分准确率】")
    for t in sorted(by_type.keys()):
        acc = np.mean(by_type[t]['corr']) * 100
        n = by_type[t]['n']
        print("  - {:10s} (n={:4d}): {:6.2f}%".format(t, n, acc))

    print("\n【按业务领域细分准确率】")
    for w in sorted(by_wf.keys()):
        acc = np.mean(by_wf[w]['corr']) * 100
        n = by_wf[w]['n']
        print("  - {:25s} (n={:4d}): {:6.2f}%".format(w, n, acc))
    print("=" * 65)

if __name__ == '__main__':
    run_benchmark(port=8149, name="AgentJev v1 (新训练版)")
