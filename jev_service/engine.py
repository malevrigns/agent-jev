"""Trained decision-head inference with bounded candidate microbatches."""
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from .contract import API_VERSION, prepare, encode_paths, answer


class DecisionEngine:
    def __init__(self, checkpoint, model_path, device='cuda:0', max_tokens=2048, path_batch=16, encoder='auto', temperatures=None):
        import torch
        from transformers import AutoTokenizer
        from agentjev.model import AgentJevModel
        self.torch = torch; self.device = device; self.max_tokens = max_tokens; self.path_batch = path_batch
        self.lock = threading.Lock(); self.encoder=encoder; self.temperatures={}
        if temperatures:
            calibration=json.loads(Path(temperatures).read_text())
            for kind,value in calibration.items():
                t=float(value['temperature'])
                if kind not in ('boolean','choice','score') or not math.isfinite(t) or not .05<=t<=20:
                    raise ValueError('Invalid calibration temperature')
                self.temperatures[kind]=t
        h = hashlib.sha256()
        with open(checkpoint, 'rb') as stream:
            for block in iter(lambda: stream.read(4*1024*1024), b''): h.update(block)
        self.checkpoint_sha256 = h.hexdigest()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        bundle = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if bundle.get('encoder_schema', '').startswith('routing_'):
            raise ValueError('Use a general decision checkpoint, not a routing-only fine-tune')
        self.trained_config = bundle.get('config')
        if self.trained_config:
            from agentjev.data import make_collate
            cfg = self.trained_config
            self.inference_encoder = cfg.get('encoder_impl', 'path')
            self.collate = make_collate(
                self.tokenizer, max_len=cfg.get('max_len', 512),
                max_state_tokens=cfg.get('max_state_tokens', 256),
                encoder_impl=self.inference_encoder,
                tree_max_mask_bytes=cfg.get('tree_max_mask_bytes', 64 * 1024 * 1024),
                tree_attention_impl=cfg.get('tree_attention_impl', 'auto'))
            self.model = AgentJevModel(
                model_path, set_dim=cfg.get('set_dim', 256),
                set_layers=cfg.get('set_layers', 2),
                set_heads=cfg.get('set_heads', 4),
                encoder_impl=self.inference_encoder,
                dtype=getattr(torch, cfg.get('dtype', 'float32')),
                tree_max_mask_bytes=cfg.get('tree_max_mask_bytes', 64 * 1024 * 1024),
                tree_attention_impl=cfg.get('tree_attention_impl', 'auto'))
        else:
            self.inference_encoder = self.encoder
            self.collate = None
            self.model = AgentJevModel(model_path)
        self.model.load_state_dict(bundle['state_dict'], strict=True)
        self.checkpoint_name = Path(checkpoint).parent.name
        del bundle
        self.model.eval().to(device)

    def info(self):
        trained_encoder = self.inference_encoder
        return {'api_version': API_VERSION, 'model': 'AgentJev-0.6B', 'checkpoint': self.checkpoint_name,
                'checkpoint_sha256': self.checkpoint_sha256, 'types': ['boolean', 'choice', 'score'],
                'max_path_tokens': (self.trained_config.get('max_len', 512)
                                    if self.trained_config else self.max_tokens),
                'max_choice_candidates': 255,
                'output_token_decoding': False,
                'shared_prefix_compute': trained_encoder != 'path', 'encoder': trained_encoder,
                'temperatures':self.temperatures,
                'probability_semantics': 'model distribution; domain calibration is not guaranteed'}

    def evaluate(self, payload):
        prepared = prepare(payload)
        if self.trained_config:
            return self._evaluate_trained(prepared)
        paths, locations, rows = encode_paths(prepared, self.tokenizer, self.max_tokens)
        torch = self.torch; start = time.perf_counter()
        with self.lock, torch.inference_mode(), torch.autocast(
                'cuda' if self.device.startswith('cuda') else 'cpu',
                dtype=torch.bfloat16, enabled=self.device.startswith('cuda')):
            from .prefix import encode
            vectors,mask,encoding_usage=encode(self,paths,locations,rows,self.encoder)
            logits = self.model._score(vectors, mask).float().masked_fill(~mask, float('-inf'))
            if self.temperatures:
                scaling=torch.tensor([self.temperatures.get(q['type'],1.) for q in rows],device=self.device)
                logits=logits/scaling[:,None]
            probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
        index = 0; results = []
        for request in prepared:
            answers = []
            for question in request['questions']:
                answers.append(answer(question, probabilities[index][:len(question['candidates'])])); index += 1
            results.append({'id': request['id'], 'answers': answers})
        return {'api_version': API_VERSION, 'model': self.checkpoint_name, 'results': results,
                'usage': {'questions': len(rows), 'candidate_paths': len(paths), 'input_path_tokens': sum(map(len, paths)),
                          'generated_tokens': 0, 'truncated_inputs': 0, **encoding_usage,
                          'wall_ms': round((time.perf_counter()-start)*1000, 2)}}

    def _evaluate_trained(self, prepared):
        """Use the same collator and encoder settings as agentjev.train."""
        from agentjev.data import batch_to_device

        torch = self.torch
        start = time.perf_counter()
        batch_states = int(self.trained_config.get('eval_batch_states',
                                                   self.trained_config.get('batch_states', 1)))
        if batch_states < 1:
            raise ValueError('eval_batch_states must be positive')
        all_probabilities = []
        path_count = path_tokens = backbone_tokens = truncated = shared_questions = 0
        actual_modes = {}
        with self.lock:
            for offset in range(0, len(prepared), batch_states):
                chunk = prepared[offset:offset + batch_states]
                states = []
                questions = []
                for request in chunk:
                    rows = []
                    for question in request['questions']:
                        n = len(question['candidates'])
                        rows.append({
                            'text': question['text'],
                            'candidates': question['candidates'],
                            # Required by the training collator, but ignored by the model.
                            'gold': {'distribution': [1.0 / n] * n},
                            'supervision': 'teacher', 'weight': 1.0,
                            'ordinal': question['type'] == 'score',
                            'qtype': question['type'],
                        })
                        questions.append(question)
                    states.append({'id': request['id'], 'state': request['state'],
                                   'questions': rows})
                # Tree packing creates CPU tensors and must stay outside inference_mode.
                batch = self.collate(states)
                if batch['n_dropped']:
                    raise ValueError(f"questions could not be encoded: {batch['dropped']}")
                path_count += batch['n_paths']
                path_tokens += batch['n_valid_tokens']
                truncated += (batch['n_trunc_states'] + batch['n_trunc_questions']
                              + batch['n_trunc_cands'])
                meta = batch.get('tree_meta')
                mode = meta['mode'] if meta else 'path'
                actual_modes[mode] = actual_modes.get(mode, 0) + 1
                backbone_tokens += meta['input_tokens'] if meta else batch['n_valid_tokens']
                if mode == 'tree':
                    shared_questions += batch['n_questions']
                batch = batch_to_device(batch, torch.device(self.device))
                with torch.inference_mode(), torch.autocast(
                        'cuda' if self.device.startswith('cuda') else 'cpu',
                        dtype=torch.bfloat16, enabled=self.device.startswith('cuda')):
                    logits = self.model(batch)['logits'].float()
                    logits = logits.masked_fill(~batch['cand_mask'], float('-inf'))
                    if self.temperatures:
                        scaling = torch.tensor(
                            [self.temperatures.get(q['type'], 1.0) for q in questions],
                            device=self.device, dtype=logits.dtype)
                        logits = logits / scaling[:, None]
                    probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
                all_probabilities.extend(probabilities)
        index = 0
        results = []
        for request in prepared:
            answers = []
            for question in request['questions']:
                answers.append(answer(question,
                                      all_probabilities[index][:len(question['candidates'])]))
                index += 1
            results.append({'id': request['id'], 'answers': answers})
        return {'api_version': API_VERSION, 'model': self.checkpoint_name,
                'results': results,
                'usage': {'questions': index, 'candidate_paths': path_count,
                          'input_path_tokens': path_tokens, 'generated_tokens': 0,
                          'truncated_inputs': truncated,
                          'backbone_input_tokens': backbone_tokens,
                          'shared_prefix_questions': shared_questions,
                          'actual_batch_modes': actual_modes,
                          'wall_ms': round((time.perf_counter() - start) * 1000, 2)}}
