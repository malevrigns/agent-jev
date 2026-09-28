"""Loopback HTTP service and decision playground; no teacher API calls."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path


def make_handler(engine, page=None):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, value):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status); self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path in ('/health', '/api/info'): self.send_json(200, {'status': 'ready', **engine.info()})
            elif self.path == '/':
                body = (Path(page) if page else Path(__file__).parent/'web'/'index.html').read_bytes()
                self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
            else: self.send_json(404, {'error': 'not found'})

        def do_POST(self):
            if self.path != '/api/evaluate': self.send_json(404, {'error': 'not found'}); return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1_000_000: self.send_json(413, {'error': 'request must be 1..1000000 bytes'}); return
                payload = json.loads(self.rfile.read(length).decode('utf-8'),
                                     parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
                self.send_json(200, engine.evaluate(payload))
            except (ValueError, TypeError, KeyError) as exc: self.send_json(400, {'error': str(exc)})
            except Exception as exc:
                print(f'inference error: {type(exc).__name__}: {exc}', flush=True)
                self.send_json(500, {'error': 'model inference failed; consult service log'})
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='/root/agentjev/runs/phase4/final.pt')
    parser.add_argument('--model-path', default='/root/agentjev/models/Qwen3-0.6B-Base')
    parser.add_argument('--port', type=int, default=18765)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--temperatures', help='Optional primitive temperatures fitted on independent calibration cases')
    parser.add_argument('--page', help='Optional task-specific workbench HTML')
    args = parser.parse_args()
    from .engine import DecisionEngine
    engine = DecisionEngine(args.checkpoint, args.model_path, args.device,
                            args.max_tokens, temperatures=args.temperatures)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(engine,args.page))
    print(json.dumps({'event': 'ready', 'port': args.port, **engine.info()}), flush=True)
    server.serve_forever()


if __name__ == '__main__': main()
