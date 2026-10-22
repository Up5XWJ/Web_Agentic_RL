"""Optional contract check against a local checkout of the pinned upstream tag.

OPENAPPS_VERL_SOURCE=/path/to/verl-v0.6.1 runs this without installing veRL,
initializing CUDA, importing its workers or downloading any model.
"""

import ast
import importlib.metadata
import os
from pathlib import Path
import sys
import types

import pytest


@pytest.mark.skipif(not os.environ.get("OPENAPPS_VERL_SOURCE"), reason="optional upstream veRL source checkout")
def test_official_config_composes_and_matches_worker_dataclass_fields(monkeypatch):
    from omegaconf import OmegaConf
    from open_apps.agentic_rl.train import load_config
    from open_apps.agentic_rl.verl_backend import compose_verl_config

    source = Path(os.environ["OPENAPPS_VERL_SOURCE"]) / "verl"
    assert (source / "version/version").read_text().strip() == "0.6.1"
    # Only the package locator is stubbed: Hydra reads the real upstream YAML,
    # and schema fields below come from real upstream Python source via AST.
    package = types.ModuleType("verl")
    package.__file__ = str(source / "__init__.py")
    package.__path__ = [str(source)]
    monkeypatch.setitem(sys.modules, "verl", package)
    version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.6.1" if name == "verl" else version(name))
    repo = Path(__file__).resolve().parents[2]
    merged = compose_verl_config(load_config(repo / "config/agentic_rl/verl.yaml"))
    concrete = OmegaConf.to_container(merged, resolve=True)

    classes = {}
    for filename in source.rglob("*.py"):
        if "config" not in str(filename):
            continue
        for node in ast.walk(ast.parse(filename.read_text())):
            if isinstance(node, ast.ClassDef):
                own = {entry.target.id for entry in node.body
                       if isinstance(entry, ast.AnnAssign) and isinstance(entry.target, ast.Name)}
                bases = [base.id for base in node.bases if isinstance(base, ast.Name)]
                classes[node.name] = (own, bases)

    def fields(name):
        own, bases = classes.get(name, (set(), []))
        return own.union(*(fields(base) for base in bases))

    def check(value, path):
        if isinstance(value, dict):
            if "_target_" in value:
                name = value["_target_"].split(".")[-1]
                assert name in classes, f"Unresolved upstream dataclass {name} at {path}"
                extra = set(value) - fields(name) - {"_target_"}
                assert not extra, f"Unknown {name} arguments at {path}: {sorted(extra)}"
            for key, child in value.items():
                check(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                check(child, f"{path}[{index}]")

    check(concrete["actor_rollout_ref"], "actor_rollout_ref")
    assert merged.actor_rollout_ref.rollout.n == 1
    assert merged.actor_rollout_ref.rollout.multi_turn.enable is False
    assert merged.actor_rollout_ref.actor.optim.total_training_steps == 20
