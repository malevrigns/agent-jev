"""Isolated v8 candidate; production serving configuration is unchanged."""
import argparse
import copy
from http.server import ThreadingHTTPServer
import json

from typed_decisions.agent_completion_v9.canonical import canonicalize
from .contract import prepare
from .engine import DecisionEngine
from .server import make_handler


def normalize_request(payload):
    # Validate the original contract first: normalization must not turn null or
    # numeric invalid states into valid-looking string literals.
    prepare(payload)
    value=copy.deepcopy(payload)
    for request in value.get('requests',[value]):
        request['state']=canonicalize(request['state'])
    return value


class CandidateEngine(DecisionEngine):
    def evaluate(self,payload):
        return super().evaluate(normalize_request(payload))

    def info(self):
        return {**super().info(),'normalization':'coding-state-v1','release_status':'candidate'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--port',type=int,default=18768)
    args=parser.parse_args()
    engine=CandidateEngine(args.checkpoint,'/root/agentjev/models/Qwen3-0.6B-Base',
                           max_tokens=4096,encoder='path',path_batch=16)
    print(json.dumps(engine.info()),flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(engine)).serve_forever()


if __name__=='__main__':
    main()
