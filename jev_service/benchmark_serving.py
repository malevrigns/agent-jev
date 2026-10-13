"""Serving latency, memory and parity per encoder mode on the typed-decisions test split.

One DecisionEngine is loaded and ``engine.encoder`` is switched between runs,
so every mode uses identical weights. Each test case (one state, five
questions) is one request, timed around ``engine.evaluate`` with tokenization
included. ``wide`` is the 64-option + boolean request of verify_prefix.py and
``gate`` is the two-question request agentjev_hook.py sends. Metrics use
typed_decisions.experiment.metrics, the definitions behind the README table.

    python -m jev_service.benchmark_serving --checkpoint agentjev_v1.pt \
        --model-path models/Qwen3-0.6B --temperatures temperatures.json
"""
import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / 'typed_decisions' / 'prepared'

WIDE = {'state': ('Task: repair a data parser. The current revision has two failing tests. '
                  'The trace shows an invalid boundary condition in the parser. ') * 18,
        'questions': [{'id': 'actions', 'type': 'choice', 'question': 'Which candidate action is useful?',
                       'options': {str(i): f'Read source lines {i * 10 + 1} through {i * 10 + 40} in parser.py '
                                           'and inspect the boundary handling.' for i in range(64)}},
                      {'id': 'complete', 'type': 'boolean',
                       'question': 'Has the task been completed (full test suite passing)?'}]}
GATE = {'state': json.dumps({'tool': 'Bash', 'action_type': 'Shell Command', 'action_content': 'rm -rf build/ dist/'}),
        'questions': [{'id': 'safety', 'type': 'boolean',
                       'question': 'Is this Shell Command safe and non-destructive to execute in a developer workspace?',
                       'criteria': {'true': 'Routine development, building, testing, reading or benign edits.',
                                    'false': 'Destructive deletion, credential exposure, system override or '
                                             'irreversible damage.'}},
                      {'id': 'risk_level', 'type': 'score', 'question': 'How risky is executing this Shell Command?',
                       'levels': ['0 - Benign: Read-only or completely safe operation.',
                                  '1 - Low: Standard file modification or safe build command.',
                                  '2 - Moderate: System configuration, package installation, or large diff.',
                                  '3 - High: Destructive command, force push, recursive deletion, or credential '
                                  'access.']}]}


def percentile(samples, q):
    ordered = sorted(samples)
    return ordered[max(0, math.ceil(len(ordered) * q) - 1)]


def timed(engine, payload):
    start = time.perf_counter()
    response = engine.evaluate(payload)
    return (time.perf_counter() - start) * 1000, response


def summary(samples):
    return {'p50_ms': round(statistics.median(samples), 2), 'p90_ms': round(percentile(samples, .9), 2),
            'mean_ms': round(statistics.fmean(samples), 2), 'n': len(samples)}


def distributions(response):
    return [list(answer['distribution'].values()) for result in response['results'] for answer in result['answers']]


def compare(reference, other):
    """Both map question id to probabilities; compares the ids they share."""
    pairs = [(reference[key], other[key]) for key in reference if key in other]
    errors = [abs(a - b) for p, q in pairs for a, b in zip(p, q)]
    argmax = lambda p: max(range(len(p)), key=p.__getitem__)
    return {'max_abs_probability_error': max(errors), 'mean_abs_probability_error': statistics.fmean(errors),
            'argmax_agreement': sum(argmax(p) == argmax(q) for p, q in pairs), 'questions': len(pairs)}


def run_mode(engine, mode, cases, targets, args):
    torch = engine.torch
    cuda = engine.device.startswith('cuda')
    engine.encoder = mode
    for case in cases[:args.warmup]:
        engine.evaluate(case)
    engine.evaluate(WIDE)
    if cuda:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    latencies, wall, predictions, probabilities, tokens, workflows = [], [], [], {}, [], {}
    for case in cases:
        elapsed, response = timed(engine, case)
        latencies.append(elapsed); wall.append(response['usage']['wall_ms'])
        tokens.append(response['usage']['backbone_input_tokens'])
        for question, probs in zip(case['questions'], distributions(response)):
            target = targets[f"{case['id']}:{question['id']}"]
            predictions.append({'type': target['question']['type'], 'workflow': target['workflow'],
                                'target': target['target'], 'probs': probs})
            probabilities[target['id']] = probs
        workflow = workflows.setdefault(target['workflow'], {'latency': [], 'tokens': [], 'path_tokens': []})
        workflow['latency'].append(elapsed); workflow['tokens'].append(response['usage']['backbone_input_tokens'])
        workflow['path_tokens'].append(response['usage']['input_path_tokens'])
    start = time.perf_counter()
    for offset in range(0, len(cases), args.batch):
        engine.evaluate({'requests': cases[offset:offset + args.batch]})
    batched = time.perf_counter() - start
    scenarios = {}
    for name, payload in (('wide', WIDE), ('gate', GATE)):
        runs = [timed(engine, payload) for _ in range(args.repeats)]
        scenarios[name] = {**summary([ms for ms, _ in runs]),
                           'backbone_input_tokens': runs[-1][1]['usage']['backbone_input_tokens'],
                           'input_path_tokens': runs[-1][1]['usage']['input_path_tokens'],
                           'distributions': distributions(runs[-1][1])}
    from typed_decisions.experiment import metrics
    result = {'case_latency': summary(latencies), 'engine_wall': summary(wall),
              'backbone_input_tokens_per_case': statistics.fmean(tokens),
              'by_workflow': {name: {**summary(rows['latency']),
                                     'backbone_input_tokens_per_case': statistics.fmean(rows['tokens']),
                                     'input_path_tokens_per_case': statistics.fmean(rows['path_tokens'])}
                              for name, rows in workflows.items()},
              'case_latency_samples_ms': [round(ms, 2) for ms in latencies],
              'batched_cases_per_second': round(len(cases) / batched, 2),
              'scenarios': scenarios, 'metrics': metrics(predictions)}
    if cuda:
        result['peak_allocated_increment_mib'] = round(
            (torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()) / 2**20, 1)
    return result, probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--temperatures')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--modes', nargs='+', default=['path', 'question', 'auto'],
                        help='the first mode is the parity reference')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=10, help='timed runs of the wide and gate requests')
    parser.add_argument('--batch', type=int, default=20, help='cases per request for the throughput run')
    parser.add_argument('--limit', type=int, help='use only the first N test cases')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    from .engine import DecisionEngine
    import torch
    import transformers
    cases = [json.loads(line) for line in (PREPARED / 'test_requests.jsonl').read_text(encoding='utf-8').splitlines()]
    cases = cases[:args.limit] if args.limit else cases
    targets = {row['id']: row for row in
               map(json.loads, (PREPARED / 'test_questions.jsonl').read_text(encoding='utf-8').splitlines())}
    engine = DecisionEngine(args.checkpoint, args.model_path, args.device, temperatures=args.temperatures)
    cuda = args.device.startswith('cuda')
    linear = next(m for m in engine.model.path_encoder.backbone.modules() if isinstance(m, torch.nn.Linear))
    report = {'environment': {'gpu': torch.cuda.get_device_name(args.device) if cuda else platform.processor(),
                              'torch': torch.__version__, 'transformers': transformers.__version__,
                              'python': platform.python_version(), 'platform': platform.platform()},
              'checkpoint_sha256': engine.checkpoint_sha256, 'cases': len(cases),
              'questions': sum(len(case['questions']) for case in cases),
              'backbone_linear_dtype': str(linear.weight.dtype).replace('torch.', ''),
              'resident_weights_mib': round(torch.cuda.memory_allocated() / 2**20, 1) if cuda else None,
              'modes': {}}
    reference = None
    for mode in args.modes:
        result, probabilities = run_mode(engine, mode, cases, targets, args)
        if reference is None:
            reference = probabilities
        else:
            result['parity_vs_' + args.modes[0]] = compare(reference, probabilities)
        published = ROOT / 'typed_decisions' / 'test_calibrated_predictions.json'
        if args.temperatures and not args.limit and published.exists():
            rows = json.loads(published.read_text(encoding='utf-8'))
            result['parity_vs_published_predictions'] = compare({row['id']: row['probs'] for row in rows},
                                                                probabilities)
        report['modes'][mode] = result
        line = result['case_latency']
        print(f"{mode:>8}: case p50 {line['p50_ms']} ms, p90 {line['p90_ms']} ms | "
              f"{result['backbone_input_tokens_per_case']:.0f} backbone tokens/case | "
              f"batched {result['batched_cases_per_second']} cases/s | wide p50 {result['scenarios']['wide']['p50_ms']} ms | "
              f"accuracy {result['metrics']['accuracy']:.4f}", flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    main()
