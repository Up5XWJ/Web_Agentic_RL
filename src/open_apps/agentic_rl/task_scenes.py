"""Offline scene composition for dependency tasks; no servers or browsers.

The source content YAMLs use OpenApps' +list/+dict convention. Materialize that
convention here, then export ordinary Hydra overrides understood by AppServer.
One variant per app is selected; a scene may combine variants of several apps.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.override_parser.types import Quote, QuotedString
from omegaconf import OmegaConf


VARIANTS = ("german", "mandarin", "long_descriptions", "misleading_descriptions",
            "adversarial_descriptions")
TIERS = ("default", "single_language", "single_distraction", "two_variants", "multi_variants")
CORE_APPS = ("todo", "calendar", "messenger", "maps")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def hydra_literal(value: Any) -> str:
    """JSON's escaping rules differ from Hydra for newlines and backslashes."""
    if isinstance(value, str):
        return QuotedString(value, Quote.double).with_quotes()
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return json.dumps(value, allow_nan=False)
    if isinstance(value, list):
        return "[" + ",".join(hydra_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9-]*", key) for key in value):
            raise ValueError("Unsupported Hydra dictionary key")
        return "{" + ",".join(key + ":" + hydra_literal(item) for key, item in value.items()) + "}"
    raise TypeError(f"Cannot encode Hydra value: {type(value).__name__}")


def merge_content(base: dict, patch: dict) -> dict:
    """Apply ordinary replacements and OpenApps' additive keys exactly once."""
    ordinary = {key: value for key, value in patch.items()
                if key != "defaults" and not key.startswith("+")}
    result = OmegaConf.to_container(OmegaConf.merge(base, ordinary), resolve=False)
    for key, value in patch.items():
        if not key.startswith("+"):
            continue
        name = key[1:]
        prior = result.get(name)
        if isinstance(prior, list) and isinstance(value, list):
            result[name] = deepcopy(prior + value)
        elif isinstance(prior, dict) and isinstance(value, dict):
            for child, extra in value.items():
                if isinstance(prior.get(child), list) and isinstance(extra, list):
                    prior[child] = deepcopy(prior[child] + extra)
                else:
                    prior[child] = deepcopy(extra)
        else:
            raise ValueError(f"Unsupported additive content key: {key}")
    return result


def _sample(rows: list, limit: int, rng: random.Random, preferred: list[int] = ()) -> list:
    """Retain some additive content even when its source list is very large."""
    if len(rows) <= limit:
        return deepcopy(rows)
    preferred = list(preferred)
    selected = set(rng.sample(preferred, min(len(preferred), max(1, limit // 2))))
    remaining = [index for index in range(len(rows)) if index not in selected]
    selected.update(rng.sample(remaining, limit - len(selected)))
    return [deepcopy(rows[index]) for index in sorted(selected)]


class SceneBuilder:
    def __init__(self, root: Path):
        self.root = root
        self._content_patches: dict[str, dict] = {}
        with initialize_config_dir(config_dir=str(root / "config"), version_base=None):
            base = compose(config_name="config", overrides=[
                "apps.onlineshop.enable=False", "seed=42",
                "logs_dir=/tmp/openapps_offline_scene",
                "databases_dir=/tmp/openapps_offline_scene/databases",
            ])
        self.base = OmegaConf.to_container(base, resolve=False)

    def variants(self, tier: int, relevant: set[str], rng: random.Random) -> list[dict]:
        if tier == 0:
            return []
        core = sorted(relevant & set(CORE_APPS))
        first = rng.choice(core)
        count = 1 if tier in (1, 2) else 2 if tier == 3 else rng.choice((3, 4))
        remaining = sorted((relevant | {"start_page", "pop_ups"}) - {first})
        apps = [first, *rng.sample(remaining, min(count - 1, len(remaining)))]
        kinds = list(VARIANTS)
        rng.shuffle(kinds)
        if tier == 1:
            kinds = [rng.choice(VARIANTS[:2])]
        elif tier == 2:
            kinds = [rng.choice(VARIANTS[2:])]
        result = []
        for app, kind in zip(apps, kinds):
            folder = self.root / "config/apps" / app
            if app != "pop_ups":
                folder /= "content"
            path = folder / f"{kind}.yaml"
            result.append({"app": app, "variant": kind, "path": str(path.relative_to(self.root)),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        return result

    def background(self, variants: list[dict], tier: int, rng: random.Random) -> tuple[dict, dict]:
        cfg = deepcopy(self.base)
        preferred: dict[str, list[int]] = {}
        for source in variants:
            app = source["app"]
            path = source["path"]
            if path not in self._content_patches:
                self._content_patches[path] = OmegaConf.to_container(
                    OmegaConf.load(self.root / path), resolve=False) or {}
            patch = self._content_patches[path]
            for field in ("init_todos", "events", "saved_places"):
                if "+" + field in patch:
                    size = len(cfg["apps"][app][field])
                    preferred[field] = list(range(size, size + len(patch["+" + field])))
            cfg["apps"][app] = merge_content(cfg["apps"].get(app) or {}, patch)
        # Resolve cross-app title and adversarial-message interpolations while
        # the complete root configuration is available.
        apps = OmegaConf.to_container(OmegaConf.create(cfg).apps, resolve=True)
        before = self.counts(apps)
        limits = {"todo": 14 + tier * 4, "calendar": 10 + tier * 4,
                  "maps": 10 + tier * 6, "contacts": 4 + tier * 2,
                  "messages_per_contact": 4 + tier}
        apps["todo"]["init_todos"] = _sample(apps["todo"]["init_todos"], limits["todo"], rng,
                                                    preferred.get("init_todos", []))
        # Calendar's initializer de-duplicates (title,date); map names are SQL
        # primary keys. Remove source collisions before taking the sample.
        events = list({(event["title"], str(event["date"])): event
                       for event in apps["calendar"]["events"]}.values())
        apps["calendar"]["events"] = _sample(events, limits["calendar"], rng,
            [i for i in preferred.get("events", []) if i < len(events)])
        places = list({place["name"]: place for place in apps["maps"]["saved_places"]}.values())
        apps["maps"]["saved_places"] = _sample(places, limits["maps"], rng,
            [i for i in preferred.get("saved_places", []) if i < len(places)])
        histories = apps["messenger"]["chat_history"]
        essential = [name for name in ("Alice", "Bob", "Charlie", "Fantastic4GroupChat") if name in histories]
        extra = [name for name in histories if name not in essential]
        contacts = essential + rng.sample(extra, min(len(extra), limits["contacts"] - len(essential)))
        apps["messenger"]["chat_history"] = {
            name: _sample(histories[name], limits["messages_per_contact"], rng,
                          list(range(min(5, len(histories[name])), len(histories[name]))))
            for name in contacts
        }
        apps["messenger"]["users"] = [name for name in contacts if name != "Fantastic4GroupChat"]
        # SQLite/calendar forms represent invitees as comma-separated text.
        for event in apps["calendar"]["events"]:
            event["date"] = str(event["date"])
            if isinstance(event.get("invitees"), list):
                event["invitees"] = ", ".join(event["invitees"])
        return apps, {"limits": limits, "before_sampling": before,
                      "after_sampling": self.counts(apps),
                      "policy": "Deterministic bounded sampling; additive YAML rows receive sampling priority."}

    @staticmethod
    def counts(apps: dict) -> dict:
        return {"todo": len(apps["todo"]["init_todos"]), "calendar": len(apps["calendar"]["events"]),
                "maps": len(apps["maps"]["saved_places"]),
                "messenger_contacts": len(apps["messenger"]["chat_history"]),
                "messenger_messages": sum(map(len, apps["messenger"]["chat_history"].values()))}

    @staticmethod
    def overrides(apps: dict) -> list[str]:
        # Materialized values are standalone: runtime need not implement '+'
        # merging or load generator metadata. Database paths remain RL-owned.
        fields = {"todo": ("init_todos", "title"),
                  "calendar": ("events", "style"),
                  "messenger": ("chat_history", "users", "title"),
                  "maps": ("saved_places", "init_location", "title"),
                  "start_page": ("headline", "sub_header", "apps")}
        result = [f"++apps.{app}.{field}={hydra_literal(apps[app][field])}"
                  for app, names in fields.items() for field in names if field in apps[app]]
        if apps.get("pop_ups"):
            result.append("++apps.pop_ups=" + hydra_literal(apps["pop_ups"]))
        return result


def business_state(apps: dict) -> dict:
    """Project seed config to the business endpoint shapes without running apps."""
    return {
        "todo": [{"title": title, "done": done} for title, done in apps["todo"]["init_todos"]],
        "calendar": [{"title": item["title"], "date": str(item["date"]),
                      "description": item.get("description") or "", "location": item.get("location") or "",
                      "url": item.get("url") or "", "invitees": item.get("invitees") or "",
                      "recurring": item.get("recurring")} for item in apps["calendar"]["events"]],
        "messenger": [{"user": name, "messages": [[m[0], m[2], m[3]] for m in messages]}
                      for name, messages in apps["messenger"]["chat_history"].items()],
        "map": [{"name": p["name"], "coords": [p["lat"], p["lng"]]} for p in apps["maps"]["saved_places"]],
        "online_shop": [], "codeeditor": {},
    }
