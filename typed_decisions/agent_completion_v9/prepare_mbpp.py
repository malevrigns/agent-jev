"""MBPP reference/mutation pairs with official task-ID splits preserved."""
import ast
import concurrent.futures
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from jev_service.contract import prepare
from typed_decisions.agent_completion_v9.canonical import canonicalize
from typed_decisions.agent_completion_v9.audit_data import signature


def mutants(code,seed):
    tree=ast.parse(code);reference=ast.unparse(tree)
    edits=[]
    swaps={ast.Add:ast.Sub,ast.Sub:ast.Add,ast.Mult:ast.Add,ast.Div:ast.FloorDiv,
           ast.Lt:ast.LtE,ast.LtE:ast.Lt,ast.Gt:ast.GtE,ast.GtE:ast.Gt,ast.Eq:ast.NotEq,
           ast.NotEq:ast.Eq,ast.And:ast.Or,ast.Or:ast.And}
    for index,node in enumerate(ast.walk(tree)):
        if isinstance(node,(ast.BinOp,ast.BoolOp)) and type(node.op) in swaps:edits.append((index,'op',None))
        if isinstance(node,ast.Compare):
            for position,op in enumerate(node.ops):
                if type(op) in swaps:edits.append((index,'comparison',position))
        if isinstance(node,ast.Constant) and type(node.value)==int and abs(node.value)<1000:
            for delta in [-1,1]:edits.append((index,'constant',delta))
    random.Random(seed).shuffle(edits)
    result=[]
    for index,kind,argument in edits[:32]:
        new=copy.deepcopy(tree);node=list(ast.walk(new))[index]
        if kind=='op':node.op=swaps[type(node.op)]()
        elif kind=='comparison':node.ops[argument]=swaps[type(node.ops[argument])]()
        else:node.value+=argument
        text=ast.unparse(ast.fix_missing_locations(new))
        if text!=reference and text not in result:result.append(text)
    return reference,result


def execute(payload):
    worker=(ROOT/'fixture_worker.py').read_text()
    result=subprocess.run([sys.executable,'-I','-c',worker],input=json.dumps(payload),text=True,
                          capture_output=True,timeout=12,cwd='/tmp',user=65534,group=65534,extra_groups=[])
    if result.returncode:raise RuntimeError(result.stderr[-500:])
    return json.loads(result.stdout)


def build(task):
    tid=task['task_id']
    split='train' if tid>=601 else 'dev' if tid>=511 else 'test' if tid>=11 else 'prompt'
    if split=='prompt':return [],{'task_id':tid,'skip':'prompt split'}
    try:
        reference,alternatives=mutants(task['code'],tid)
        tests=task['test_list']+task.get('challenge_test_list',[])
        if len(tests)<2:return [],{'task_id':tid,'skip':'too few checks'}
        outcomes=execute({'sources':[reference]+alternatives,'tests':tests,'setup':task.get('test_setup_code','')})
        if any(x!='pass' for x in outcomes[0]):return [],{'task_id':tid,'skip':'reference failed or unsupported','outcomes':outcomes[0]}
        rows=[]
        for mutation,(source,result) in enumerate(zip(alternatives,outcomes[1:])):
            if 'pass' not in result or all(x=='pass' for x in result) or 'TimeoutError' in result:continue
            public_index=result.index('pass')
            for positive,implementation in [(True,reference),(False,source)]:
                state={'requirements':task['text'],'implementation':task.get('test_setup_code','')+'\n'+implementation,
                       'observed_public_tests':{'passed':True},'public_test_examples':tests[public_index]}
                q={'id':'complete','type':'boolean','question':'Does the current implementation satisfy every stated requirement, including edge cases? Passing the limited public test alone is not sufficient.'}
                normalized=prepare({'state':state,'questions':[q]})[0]
                rows.append({'id':f'mbpp:{tid}:{mutation}:{positive}','case_id':f'mbpp:{tid}',
                             'pair_id':f'mbpp:{tid}:{mutation}','workflow':'agent_completion','task_id':tid,
                             'state':canonicalize(normalized['state']),'question':normalized['questions'][0],
                             'target':[float(positive),float(not positive)],'split':split,'source':'MBPP reference and executed AST mutation',
                             'label_semantics':'reference passes supplied tests; negative fails hidden supplied test after passing visible test'})
            if len(rows)>=6:break
        return rows,{'task_id':tid,'split':split,'rows':len(rows),'reference_verified':True}
    except Exception as exc:return [],{'task_id':tid,'skip':type(exc).__name__,'detail':str(exc)[:500]}


def main():
    if (ROOT/'frozen_run').exists():raise RuntimeError('Frozen run')
    probe=execute({'sandbox_probe':True})
    if not all(probe.values()):raise RuntimeError('Sandbox isolation probe failed')
    (ROOT/'sandbox_probe.json').write_text(json.dumps(probe,indent=2))
    raw=ROOT/'raw/mbpp.jsonl'
    tasks=[json.loads(line) for line in raw.read_text().splitlines()]
    grouped={k:[] for k in ['train','dev','test']};audit=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for rows,record in pool.map(build,tasks):
            audit.append(record)
            if rows:grouped[rows[0]['split']].extend(rows)
            if len(audit)%50==0:
                print('processed',len(audit),flush=True)
                tmp=ROOT/'status.tmp'
                tmp.write_text(json.dumps({'status':'preparing_data','checked_tasks':len(audit),'total_tasks':len(tasks)}))
                tmp.replace(ROOT/'status.json')
    # Remove complete training tasks with shapes overlapping held-out tasks.
    holdout={signature(r) for s in ['dev','test'] for r in grouped[s]}
    excluded={r['task_id'] for r in grouped['train'] if signature(r) in holdout}
    grouped['train']=[r for r in grouped['train'] if r['task_id'] not in excluded]
    out=ROOT/'prepared';out.mkdir(exist_ok=True)
    for split,rows in grouped.items():
        (out/f'{split}_questions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out/'pairs_train.jsonl').write_bytes((out/'train_questions.jsonl').read_bytes())
    manifest={'source':'https://github.com/google-research/google-research/tree/master/mbpp',
              'dataset_card':'https://huggingface.co/datasets/google-research-datasets/mbpp','license':'CC-BY-4.0',
              'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'official_split_preserved':True,
              'excluded_training_tasks_for_structure':sorted(excluded),'sandbox_probe':probe,
              'counts':{s:{'rows':len(rows),'tasks':len({r['task_id'] for r in rows})} for s,rows in grouped.items()},
              'limitations':['Reference pass is finite-test evidence, not proof of full correctness',
                             'Public dataset; base-model pretraining contamination unknown',
                             'Mutation failures and retained tasks are a filtered subset; not an MBPP code-generation score']}
    (ROOT/'fixture_audit.json').write_text(json.dumps(audit,indent=2))
    (ROOT/'data_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
