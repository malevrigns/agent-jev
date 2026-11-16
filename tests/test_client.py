"""agentjev_client against a local stub server. Standard library only."""
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agentjev_client import AgentJev, AgentJevError

ANSWER = {'id': 'done', 'type': 'boolean', 'distribution': {'true': 0.25, 'false': 0.75},
          'probability': 0.25, 'value': False}


class Stub(BaseHTTPRequestHandler):
    received = []

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])).decode('utf-8'))
        Stub.received.append(payload)
        if payload.get('state') == 'reject me':
            status, body = 400, {'error': 'choice requires 2..255 candidates'}
        else:
            states = payload.get('requests', [payload])
            status, body = 200, {'results': [{'id': str(i), 'answers': [ANSWER]} for i in range(len(states))],
                                 'usage': {'generated_tokens': 0}}
        data = json.dumps(body).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class ClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Stub)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.jev = AgentJev(f'http://127.0.0.1:{cls.server.server_address[1]}', timeout=5)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        Stub.received.clear()

    def test_object_state_is_sent_unserialized(self):
        # The server canonicalizes objects; a pre-serialized, indented string
        # would be a different model input than the benchmark used.
        state = {'test_output': 'Tests run: 14, Failures: 1', 'task': 'Fix verifyToken'}
        self.jev.decide_boolean(state, 'Have all tests passed?')
        self.assertEqual(Stub.received[0]['state'], state)

    def test_text_state_is_sent_as_text(self):
        self.jev.evaluate('23 tests passed, 1 failed.', [{'type': 'boolean', 'question': 'Passing?'}])
        self.assertEqual(Stub.received[0]['state'], '23 tests passed, 1 failed.')

    def test_server_error_message_reaches_the_caller(self):
        with self.assertRaises(AgentJevError) as caught:
            self.jev.evaluate('reject me', [{'type': 'boolean', 'question': 'Passing?'}])
        self.assertEqual(caught.exception.status, 400)
        self.assertIn('2..255 candidates', str(caught.exception))

    def test_evaluate_many_sends_one_batched_request(self):
        items = [{'id': 'a', 'state': {'x': 1}, 'questions': [{'type': 'boolean', 'question': 'Q?'}]},
                 {'id': 'b', 'state': 'text', 'questions': [{'type': 'boolean', 'question': 'Q?'}]}]
        results = self.jev.evaluate_many(items)
        self.assertEqual(len(Stub.received), 1)
        self.assertEqual([r['state'] for r in Stub.received[0]['requests']], [{'x': 1}, 'text'])
        self.assertEqual(len(results), 2)

    def test_loopback_ignores_a_system_proxy(self):
        saved = {k: os.environ.get(k) for k in ('HTTP_PROXY', 'http_proxy', 'NO_PROXY', 'no_proxy')}
        os.environ.update(HTTP_PROXY='http://127.0.0.1:9', http_proxy='http://127.0.0.1:9', NO_PROXY='', no_proxy='')
        try:
            jev = AgentJev(f'http://127.0.0.1:{self.server.server_address[1]}', timeout=5)
            self.assertFalse(jev.decide_boolean('x', 'Passing?')['decision'])
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_high_level_helpers_keep_their_fields(self):
        gate = self.jev.decide_boolean('x', 'Passing?')
        self.assertEqual((gate['decision'], gate['prob_true'], gate['prob_false']), (False, 0.25, 0.75))


if __name__ == '__main__':
    unittest.main()
