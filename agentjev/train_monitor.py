"""Optional wandb monitoring; failures never interrupt training."""
from __future__ import annotations

import os
from datetime import datetime


def init_wandb(cfg, *, config_path: str, out_dir: str, effective: dict,
               total_steps: int):
    """Start monitoring before model load; fall back to offline, then disabled."""
    block = cfg.get("wandb") or {}
    if isinstance(block, bool):        # shorthand: `wandb: true`
        block = {"enabled": block}
    if not block.get("enabled", False):
        return None
    try:
        import wandb
    except ImportError as e:
        print(f"[warn] wandb is enabled but not installed ({e}); "
              f"continuing WITHOUT monitoring. `pip install wandb` to enable.")
        return None

    run_name = block.get("run_name") or (
        f"{os.path.splitext(os.path.basename(config_path))[0]}-"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    # The DERIVED budget rides along with the config: total_steps is not in
    # the yaml, yet it is the number every curve gets read against.
    run_config = {k: v for k, v in cfg.items() if k != "wandb"}
    run_config["effective"] = effective
    run_config["total_steps"] = total_steps
    kwargs = {
        "project": block.get("project", "agentjev"),
        "entity": block.get("entity") or None,
        "name": run_name,
        "tags": list(block.get("tags") or []),
        "config": run_config,
        "dir": out_dir,
    }
    if block.get("mode"):
        kwargs["mode"] = block["mode"]
    try:
        run = wandb.init(**kwargs)
    except Exception as e:
        # The interesting failures are all environment problems -- no API
        # key, no network, a proxy in the way. None of them are reasons to
        # lose a training run, so retry once against local mode.
        print(f"[warn] wandb.init failed ({e}); falling back to LOCAL mode. "
              f"Records still go to {out_dir}/wandb/ and can be uploaded "
              f"later with `wandb sync`. To upload live, run `wandb login`.")
        kwargs["mode"] = "offline"
        try:
            run = wandb.init(**kwargs)
        except Exception as e2:
            print(f"[warn] local wandb.init also failed ({e2}); continuing "
                  f"WITHOUT monitoring. Training and events.jsonl are "
                  f"unaffected.")
            return None
    url = getattr(run, "url", None)
    print(f"wandb: {run_name} -> {url or '(local only; `wandb sync` to upload)'}")
    return run


def _flatten(value: dict, prefix: str = "") -> dict:
    """Flatten nested metrics to dotted keys for consistent wandb storage."""
    out: dict = {}
    for k, v in value.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        out.update(_flatten(v, key) if isinstance(v, dict) else {key: v})
    return out


class WandbLogger:
    """Warn and disable monitoring after a log failure; training continues."""

    def __init__(self, run=None):
        self.run = run
        self.disabled = run is None

    def log(self, event: dict, step: int) -> None:
        if self.disabled:
            return
        try:
            self.run.log(event, step=step)
        except Exception as e:
            self.disabled = True
            print(f"[warn] wandb.log failed at step {step} ({e}); monitoring "
                  f"disabled for the rest of this run. Training continues "
                  f"and events.jsonl is unaffected.")

    def finish(self, summary: dict | None = None) -> None:
        if self.disabled:
            return
        try:
            if summary:
                self.run.summary.update(summary)
            self.run.finish()
            print("wandb: run finished")
        except Exception as e:
            print(f"[warn] wandb.finish failed: {e}")
