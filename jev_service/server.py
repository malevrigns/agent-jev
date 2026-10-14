"""Loopback HTTP service and decision playground; no teacher API calls."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sys
from pathlib import Path
import time

def opt_out_of_power_throttling():
    """Keep Windows from slowing this process down whenever its window is not in front.

    Windows 11 applies EcoQoS to background processes; for this service that
    doubled request latency (about 35 ms to 75 ms on one RTX 5060 machine).
    Priority is left unchanged. Returns whether the opt-out took effect.
    """
    if sys.platform != 'win32':
        return False
    import ctypes
    from ctypes import wintypes

    class PowerThrottlingState(ctypes.Structure):
        _fields_ = [('Version', wintypes.ULONG), ('ControlMask', wintypes.ULONG), ('StateMask', wintypes.ULONG)]

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    # ProcessPowerThrottling = 4; control EXECUTION_SPEED (0x1) with state 0 = never throttle.
    state = PowerThrottlingState(1, 0x1, 0)
    return bool(kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), 4,
                                               ctypes.byref(state), ctypes.sizeof(state)))


WARMUP = {'state': 'Tests run: 3, Failures: 0', 'questions': [
    {'type': 'boolean', 'question': 'Have all tests passed?'},
    {'type': 'choice', 'question': 'What next?', 'options': ['Commit.', 'Read the failure.']},
    {'type': 'score', 'question': 'How risky?', 'levels': ['Low.', 'High.']}]}


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
    parser.add_argument('--port', type=int, default=8149)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--encoder', default='auto', choices=('auto', 'question', 'shared', 'path'),
                        help='auto shares each state across its questions; question shares only within a question')
    parser.add_argument('--temperatures', help='Optional primitive temperatures fitted on independent calibration cases')
    parser.add_argument('--page', help='Optional task-specific workbench HTML')
    args = parser.parse_args()
    unthrottled = opt_out_of_power_throttling()
    from .engine import DecisionEngine
    engine = DecisionEngine(args.checkpoint, args.model_path, args.device,
                            args.max_tokens, encoder=args.encoder, temperatures=args.temperatures)
    # The first forward pays for CUDA context and kernel setup (about a second);
    # spend it here so the first real request does not.
    warmup = time.perf_counter()
    engine.evaluate(WARMUP)
    warmup_ms = round((time.perf_counter() - warmup) * 1000, 1)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(engine,args.page))
    print(json.dumps({'event': 'ready', 'port': args.port, 'warmup_ms': warmup_ms,
                      'power_throttling_disabled': unthrottled, **engine.info()}), flush=True)
    server.serve_forever()


if __name__ == '__main__': main()
