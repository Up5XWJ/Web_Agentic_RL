"""Deterministic, offline cross-app information-dependency tasks.

Private read graphs resolve answers from the actual seeded records. Public goals
contain entry clues and output formats only. Existing task classes verify final
state; the graph is dataset metadata, not a required browser action sequence.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, timedelta
import io
import random
import re
from typing import Any

from hydra.utils import instantiate

from .task_scenes import SceneBuilder, TIERS, business_state, digest
from .tasks import derive_predicates, family_id, progress_fraction, validate_task
from .types import TaskSpec


TEMPLATES = (
    "calendar_to_todo", "calendar_to_messenger", "messenger_to_calendar", "messenger_to_maps",
    "messenger_calendar_to_todo", "todo_messenger_to_calendar", "maps_calendar_to_messenger",
    "calendar_messenger_to_maps", "messenger_calendar_maps_to_todo", "todo_messenger_calendar_to_maps",
    "calendar_messenger_to_todo", "messenger_maps_to_calendar",
)
PLACES = (
    ("Central Park South Entrance", 40.7681, -73.9819),
    ("Brooklyn Bridge Manhattan Entrance", 40.7120, -74.0048),
    ("Times Square Visitor Meeting Point", 40.7580, -73.9855),
    ("Empire State Building Main Entrance", 40.748817, -73.985428),
    ("Battery Park Group Meeting Point", 40.7033, -74.0170),
)


@dataclass
class DependencyCase:
    task: TaskSpec
    metadata: dict
    apps: dict


def _note(**fields: str) -> str:
    return "; ".join(f"{key.replace('_', ' ')}: {value}" for key, value in fields.items())


def _field(record: dict, path: str) -> Any:
    if path == "month":
        return str(record.get("date", ""))[:7]
    root, _, label = path.partition(".")
    value = record.get(root)
    if not label:
        return value
    if not isinstance(value, str):
        return None
    matches = re.findall(r"(?:^|;\s*)" + re.escape(label) + r":\s*([^;\n]+)", value)
    if len(matches) > 1:
        raise ValueError(f"Ambiguous source field: {path}")
    return matches[0].strip() if matches else None


def _records(apps: dict, app: str):
    if app == "messenger":
        for user, messages in apps[app]["chat_history"].items():
            for index, row in enumerate(messages):
                yield {"user": user, "text": row[0], "sender": row[2]}, (app, "chat_history", user, index)
    elif app == "todo":
        for index, row in enumerate(apps[app]["init_todos"]):
            yield {"title": row[0], "done": row[1]}, (app, "init_todos", index)
    else:
        key = "events" if app == "calendar" else "saved_places"
        for index, row in enumerate(apps[app][key]):
            yield row, (app, key, index)


def _expression(value: Any, resolved: dict) -> Any:
    if isinstance(value, str) and value.startswith("@"):
        node, field = value[1:].split(".", 1)
        if node not in resolved or field not in resolved[node]:
            raise ValueError(f"Unresolved dependency: {value}")
        return resolved[node][field]
    if isinstance(value, dict):
        if "join" in value:
            return value.get("separator", " | ").join(str(_expression(v, resolved)) for v in value["join"])
        return {key: _expression(item, resolved) for key, item in value.items()}
    if isinstance(value, list):
        return [_expression(item, resolved) for item in value]
    return value


def resolve_graph(apps: dict, graph: dict) -> tuple[dict, dict, dict]:
    values, locations = {}, {}
    for node in graph["reads"]:
        if node["id"] in values:
            raise ValueError("Duplicate dependency node")
        selector = _expression(node["selector"], values)
        candidates = []
        for record, location in _records(apps, node["app"]):
            matches = True
            for field, expected in selector.items():
                if field.endswith("__prefix"):
                    matches &= str(_field(record, field[:-8]) or "").startswith(expected)
                else:
                    matches &= _field(record, field) == expected
            if matches:
                candidates.append((record, location))
        if len(candidates) != 1:
            raise ValueError(f"Expected one source for {node['id']}, got {len(candidates)}")
        record, location = candidates[0]
        exports = {name: _field(record, field) for name, field in node["exports"].items()}
        if any(value is None or value == "" for value in exports.values()):
            raise ValueError(f"Missing required source value: {node['id']}")
        values[node["id"]], locations[node["id"]] = exports, location
    return _expression(graph["operation"], values), values, locations


def _references(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value[1:]] if value.startswith("@") else []
    if isinstance(value, dict):
        return [ref for item in value.values() for ref in _references(item)]
    if isinstance(value, list):
        return [ref for item in value for ref in _references(item)]
    return []


def compile_operation(operation: dict, goal: str) -> dict:
    common = {"goal": goal}
    if operation["type"] == "add_todo":
        return common | {"_target_": "open_apps.tasks.tasks.AddToDoTask",
                         "todo_name": operation["title"], "is_done": False}
    if operation["type"] == "add_event":
        return common | {"_target_": "open_apps.tasks.tasks.AddEventTask",
                         "title": operation["title"], "date": operation["date"],
                         "location": operation["location"], "description": None,
                         "url": None, "invitees": []}
    if operation["type"] == "send_message":
        return common | {"_target_": "open_apps.tasks.tasks.SendMessageTask",
                         "to": operation["to"], "message": operation["message"]}
    if operation["type"] == "remove_place":
        return common | {"_target_": "open_apps.tasks.tasks.RemoveLandmarkTask", "name": operation["name"]}
    raise ValueError(f"Unsupported terminal operation: {operation['type']}")


def apply_operation(state: dict, operation: dict) -> dict:
    """Independent endpoint-shaped positive fixture for the existing verifier."""
    result = deepcopy(state)
    kind = operation["type"]
    if kind == "add_todo":
        result["todo"].append({"title": operation["title"], "done": False})
    elif kind == "add_event":
        result["calendar"].append({"title": operation["title"], "date": operation["date"],
                                   "location": operation["location"], "description": "", "url": "", "invitees": []})
    elif kind == "send_message":
        contact = next(row for row in result["messenger"] if row["user"] == operation["to"])
        contact["messages"].append([operation["message"], "you", "fixed offline timestamp"])
    elif kind == "remove_place":
        if sum(p["name"] == operation["name"] for p in result["map"]) != 1:
            raise ValueError("Deletion must identify exactly one existing saved place")
        result["map"] = [p for p in result["map"] if p["name"] != operation["name"]]
    else:
        raise ValueError(kind)
    return result


def build_case(builder: SceneBuilder, template: str, index: int, *, generation_seed: int = 20261010,
               environment_seed: int = 42, content: dict | None = None,
               catalog: Any = None) -> DependencyCase:
    if template not in TEMPLATES:
        raise ValueError(template)
    token = digest([generation_seed, template, index])[:10]
    rng = random.Random(int(token, 16))
    project, request, intake = f"PX-{token}", f"REQ-{rng.getrandbits(32):08x}", f"INT-{rng.getrandbits(32):08x}"
    event_title = rng.choice(("Design review", "Volunteer briefing", "Supplier meeting", "Workshop planning")) + f" EV-{rng.getrandbits(32):08x}"
    contact, receiver = rng.sample(["Alice", "Bob", "Charlie"], 2)
    when = date(2026, 10, 1) + timedelta(days=rng.randrange(84))
    # Explicit month clues avoid relying on the host's current calendar view.
    month, event_date = when.strftime("%Y-%m"), when.isoformat()
    venue_code = f"LOC-{rng.getrandbits(32):08x}"
    place, lat, lng = rng.choice(PLACES)
    full_place = f"{venue_code} | {place}"
    material = rng.choice(("printed agenda and name badges", "prototype and feedback forms",
                           "sample kit and safety checklist", "attendance sheet and handouts"))
    if content is not None:
        if catalog is None:
            raise ValueError("Source-backed content requires its contact catalog")
        project = f"{content['topic']} [{token}]"
        event_title = f"{content['event_title']} EV-{rng.getrandbits(32):08x}"
        contact, receiver = content["contact"], content["receiver"]
        when = date.fromisoformat(content["date"])
        month, event_date = when.strftime("%Y-%m"), when.isoformat()
        place, lat, lng = (content["place"][k] for k in ("name", "lat", "lng"))
        full_place = f"{venue_code} | {place}"
        material = content["materials"]
    reads: list[dict] = []
    additions: list[tuple[str, Any]] = []

    def notice(exports: dict, selector_contact=contact, record_id=request, **fields):
        body = _note(Record=record_id, Status="confirmed", **fields)
        if content is not None:
            body += "; " + _note(Context=content["topic"])
        additions.append(("message", (contact, [body, False, contact, f"{event_date} 09:30"])))
        draft = dict(fields)
        for key in draft:
            draft[key] = {"Title": event_title + " - draft", "Date": (when - timedelta(days=1)).isoformat(),
                          "Venue": "Draft meeting room", "Calendar_event": event_title + " - draft",
                          "Month": (when + timedelta(days=1)).strftime("%Y-%m"),
                          "Remove_place": full_place.replace(venue_code, venue_code + "-OLD"),
                          "Venue_code": venue_code + "-OLD", "Materials": "superseded draft checklist"}[key]
        additions.append(("message", (contact, [_note(Record=record_id, Status="draft", **draft),
                                                False, contact, f"{event_date} 08:00"])))
        reads.append({"id": "notice", "app": "messenger", "selector": {
            "user": selector_contact, "text.Record": record_id,
            "text.Status": "confirmed", "sender": selector_contact},
            "exports": {key: "text." + value for key, value in exports.items()}})

    def calendar(exports: dict, title=event_title, selected_month=month, location=place, **details):
        additions.append(("event", {"title": event_title, "date": event_date, "location": location,
                                    "description": _note(**details) if details else f"Confirmed arrangements for {project}.",
                                    "url": "", "invitees": "", "recurring": None}))
        reads.append({"id": "event", "app": "calendar", "selector": {"title": title, "month": selected_month},
                      "exports": exports})

    def todo():
        additions.append(("todo", [_note(Record=intake, Status="active", Contact=contact, Request=request), False]))
        reads.append({"id": "intake", "app": "todo", "selector": {"title.Record": intake, "title.Status": "active", "done": False},
                      "exports": {"contact": "title.Contact", "request": "title.Request"}})

    def saved(name=full_place):
        additions.append(("place", {"name": name, "lat": lat, "lng": lng}))

    def map_lookup(code: str):
        saved()
        reads.append({"id": "place", "app": "maps", "selector": {
            "name__prefix": {"join": [code, ""], "separator": " | "}}, "exports": {"name": "name"}})

    msg_clue = f"the confirmed message with Record '{request}' in {contact}'s Messenger conversation"
    cal_clue = f"Calendar event '{event_title}' in {month}"
    todo_clue = f"the incomplete Todo entry with Record '{intake}' and Status 'active'"
    calendar_fields = " Leave description, URL and invitees empty, and use no recurrence."
    if template == "calendar_to_todo":
        calendar({"date": "date", "location": "location"})
        operation = {"type": "add_todo", "title": {"join": [f"Prepare {project}", "@event.date", "@event.location"]}}
        goal = f"Read {cal_clue}. Add an incomplete Todo titled 'Prepare {project} | <event date YYYY-MM-DD> | <event location>'."
    elif template == "calendar_to_messenger":
        calendar({"date": "date", "location": "location", "recipient": "description.Notify"}, Notify=receiver)
        operation = {"type": "send_message", "to": "@event.recipient",
                     "message": {"join": [project, "@event.date", "@event.location"]}}
        goal = f"Read {cal_clue}. Send the contact named by Notify in its description exactly '{project} | <event date YYYY-MM-DD> | <event location>' in Messenger."
    elif template == "messenger_to_calendar":
        notice({"title": "Title", "date": "Date", "venue": "Venue"}, Title=event_title, Date=event_date, Venue=place)
        operation = {"type": "add_event", "title": "@notice.title", "date": "@notice.date", "location": "@notice.venue"}
        goal = f"Read {msg_clue}. Create a Calendar event using its Title, Date and Venue fields." + calendar_fields
    elif template == "messenger_to_maps":
        saved()
        notice({"place": "Remove place"}, Remove_place=full_place)
        operation = {"type": "remove_place", "name": "@notice.place"}
        goal = f"Read {msg_clue}. Remove the Maps favorite named by its Remove place field."
    elif template == "messenger_calendar_to_todo":
        notice({"event": "Calendar event", "month": "Month"}, Calendar_event=event_title, Month=month)
        calendar({"date": "date", "location": "location"}, "@notice.event", "@notice.month")
        operation = {"type": "add_todo", "title": {"join": [f"Prepare {project}", "@event.date", "@event.location"]}}
        goal = f"Read {msg_clue}, then find the Calendar event and month it specifies. Add an incomplete Todo titled 'Prepare {project} | <event date YYYY-MM-DD> | <event location>'."
    elif template == "todo_messenger_to_calendar":
        todo()
        notice({"title": "Title", "date": "Date", "venue": "Venue"}, "@intake.contact", request,
               Title=event_title, Date=event_date, Venue=place)
        reads[-1]["selector"]["text.Record"] = "@intake.request"
        operation = {"type": "add_event", "title": "@notice.title", "date": "@notice.date", "location": "@notice.venue"}
        goal = f"Read {todo_clue}. In its Contact's Messenger conversation, find the confirmed message whose Record matches its Request. Create a Calendar event from that message's Title, Date and Venue." + calendar_fields
    elif template == "maps_calendar_to_messenger":
        saved(_note(Place=place, Code=venue_code, Calendar_event=event_title, Month=month))
        reads.append({"id": "place", "app": "maps", "selector": {"name.Code": venue_code},
                      "exports": {"event": "name.Calendar event", "month": "name.Month"}})
        calendar({"date": "date", "location": "location", "recipient": "description.Notify"},
                 "@place.event", "@place.month", Notify=receiver)
        operation = {"type": "send_message", "to": "@event.recipient",
                     "message": {"join": [project, "@event.date", "@event.location"]}}
        goal = f"Find the Maps favorite whose name contains Code '{venue_code}'. Its name records a Calendar event and Month. Read that event and send its Notify contact exactly '{project} | <event date YYYY-MM-DD> | <event location>' in Messenger."
    elif template == "calendar_messenger_to_maps":
        calendar({"contact": "description.Contact", "request": "description.Request"}, Contact=contact, Request=request)
        saved()
        notice({"place": "Remove place"}, "@event.contact", request, Remove_place=full_place)
        reads[-1]["selector"]["text.Record"] = "@event.request"
        operation = {"type": "remove_place", "name": "@notice.place"}
        goal = f"Read {cal_clue}. Its description gives a Contact and Request. Find that contact's confirmed Messenger message whose Record matches Request, then delete the Maps favorite named by Remove place."
    elif template == "messenger_calendar_maps_to_todo":
        notice({"event": "Calendar event", "month": "Month"}, Calendar_event=event_title, Month=month)
        calendar({"date": "date", "code": "location"}, "@notice.event", "@notice.month", location=venue_code)
        map_lookup("@event.code")
        operation = {"type": "add_todo", "title": {"join": [f"Prepare {project}", "@event.date", "@place.name"]}}
        goal = f"Read {msg_clue}, then find the Calendar event and month it specifies. The event's location is a venue code. Find the Maps favorite whose name starts with that code followed by ' | '. Add an incomplete Todo titled 'Prepare {project} | <event date YYYY-MM-DD> | <complete favorite name>'."
    elif template == "todo_messenger_calendar_to_maps":
        todo()
        notice({"event": "Calendar event", "month": "Month"}, "@intake.contact", request,
               Calendar_event=event_title, Month=month)
        reads[-1]["selector"]["text.Record"] = "@intake.request"
        calendar({"place": "location"}, "@notice.event", "@notice.month", location=full_place,
                 Status="cancelled", Reason="This old venue reservation was cancelled; remove its saved meeting point")
        saved()
        operation = {"type": "remove_place", "name": "@event.place"}
        goal = f"Read {todo_clue}. Follow its Contact and Request to the corresponding confirmed Messenger message. Open the Calendar event and month named there. Remove the Maps favorite whose full name equals that event's location; it is the obsolete meeting point."
    elif template == "calendar_messenger_to_todo":
        calendar({"date": "date", "location": "location"})
        notice({"materials": "Materials"}, Materials=material)
        operation = {"type": "add_todo", "title": {"join": [f"Prepare {project}", "@event.date", "@event.location", "@notice.materials"]}}
        goal = f"Combine {cal_clue} with {msg_clue}. Add an incomplete Todo titled 'Prepare {project} | <event date YYYY-MM-DD> | <event location> | <message Materials>'."
    else:  # messenger_maps_to_calendar: join message scheduling data with a place lookup
        notice({"title": "Title", "date": "Date", "code": "Venue code"}, Title=event_title, Date=event_date, Venue_code=venue_code)
        map_lookup("@notice.code")
        operation = {"type": "add_event", "title": "@notice.title", "date": "@notice.date", "location": "@place.name"}
        goal = f"Read {msg_clue}. Resolve its Venue code to the Maps favorite whose name starts with that code followed by ' | '. Create a Calendar event using the message's Title and Date, and the complete favorite name as location." + calendar_fields
    goal += " Preserve other records and submit when finished."
    tier = index // 10
    if tier >= len(TIERS):
        raise ValueError("Each template currently defines exactly 50 independently seeded scenes")
    terminal_app = {"add_todo": "todo", "add_event": "calendar", "send_message": "messenger", "remove_place": "maps"}[operation["type"]]
    relevant = {node["app"] for node in reads} | {terminal_app}
    variants = (catalog.scene_variants(builder, tier, relevant, rng) if content is not None
                else builder.variants(tier, relevant, rng))
    apps, sampling = builder.background(variants, tier, rng)
    if content is not None:
        catalog.ensure_contacts(apps, [contact, receiver])
    # Add disjoint near-miss sources, then the designated sources. IDs and
    # status distinguish authoritative records from drafts and similar cases.
    for kind, row in additions:
        if kind == "message":
            user, message = row
            apps["messenger"]["chat_history"][user].append(deepcopy(message))
        elif kind == "event":
            decoy = deepcopy(row)
            decoy.update(title=row["title"] + " - draft", date=(when + timedelta(days=1)).isoformat(),
                         location="Draft meeting room", description="Unapproved earlier proposal; no confirmed booking.")
            apps["calendar"]["events"].extend([decoy, deepcopy(row)])
        elif kind == "todo":
            archived = row[0].replace("Status: active", "Status: archived").replace(request, request + "-OLD")
            apps["todo"]["init_todos"].extend([[archived, True], deepcopy(row)])
        else:
            decoy = deepcopy(row)
            decoy["name"] = row["name"].replace(venue_code, venue_code + "-OLD")
            apps["maps"]["saved_places"].extend([decoy, deepcopy(row)])
    # Interleave instead of always placing the answer at one fixed position.
    for rows in (apps["todo"]["init_todos"], apps["calendar"]["events"], apps["maps"]["saved_places"]):
        rng.shuffle(rows)
    for history in apps["messenger"]["chat_history"].values():
        # Preserve chronological order within generated conversations: a draft
        # precedes confirmation even when both are injected into old histories.
        generated = [m for m in history if str(m[3]).startswith(event_date)]
        history[:] = [m for m in history if m not in generated] + sorted(generated, key=lambda m: m[3])
    graph = {"reads": reads, "operation": operation}
    expected, resolved, locations = resolve_graph(apps, graph)
    cfg = compile_operation(expected, goal)
    fid = family_id(cfg)
    budget = 30 + 15 * len(reads) + 5 * tier
    task = TaskSpec(f"web_{fid}", goal, cfg, family_id=fid, seed=environment_seed,
                    max_steps=budget, overrides=builder.overrides(apps), progress_predicates=derive_predicates(cfg))
    validate_task(task)
    edges = [{"from": ref.split(".")[0], "to": node["id"], "field": ref}
             for node in reads for ref in _references(node["selector"])]
    edges += [{"from": ref.split(".")[0], "to": "result", "field": ref} for ref in _references(operation)]
    metadata = {
        "task_id": task.task_id, "family_id": fid, "origin": "generated_dependency",
        "template_id": template, "case_index": index, "parent_case_id": f"dependency_{token}",
        "scene_id": digest(business_state(apps)), "environment_seed": environment_seed,
        "complexity_tier": TIERS[tier], "variant_count": len(variants), "variants": variants,
        "background_sampling": sampling, "initial_record_counts": builder.counts(apps),
        "dependency_graph": graph | {"edges": edges}, "resolved_sources": resolved,
        "source_locations": {key: list(value) for key, value in locations.items()},
        "expected_operation": expected, "read_count": len(reads), "apps": sorted(relevant),
        "max_steps": budget, "verification_scope": "Final state; reading order is not graded.",
    }
    if content is not None:
        metadata["content"] = deepcopy(content)
    return DependencyCase(task, metadata, apps)


def _at(tree: Any, path: tuple | list) -> Any:
    for key in path:
        tree = tree[key]
    return tree


def _mutate_field(apps: dict, node: dict, location: tuple, source: str, value: str) -> None:
    row = _at(apps, location)
    root, _, label = source.partition(".")
    if node["app"] == "messenger":
        slot = 0 if root == "text" else 2
    elif node["app"] == "todo":
        slot = 0 if root == "title" else 1
    else:
        slot = root
    if label:
        row[slot] = re.sub(r"((?:^|;\s*)" + re.escape(label) + r":\s*)[^;\n]+",
                           lambda match: match[1] + value, row[slot])
    else:
        row[slot] = value


def validate_case(case: DependencyCase) -> dict:
    """Validate actual fixtures, dependencies and outcomes entirely offline."""
    spec, apps, metadata = case.task, case.apps, case.metadata
    validate_task(spec)
    graph = metadata["dependency_graph"]
    expected, values, locations = resolve_graph(apps, graph)
    if expected != metadata["expected_operation"] or compile_operation(expected, spec.goal) != spec.task_config:
        raise ValueError("Private answer does not match the initial source facts")
    initial = business_state(apps)
    for app in ("todo", "calendar", "messenger", "map"):
        if not initial[app]:
            raise ValueError(f"Required app is empty: {app}")
    calendar_keys = [(row["title"], row["date"]) for row in initial["calendar"]]
    map_names = [row["name"] for row in initial["map"]]
    if len(set(calendar_keys)) != len(calendar_keys) or len(set(map_names)) != len(map_names):
        raise ValueError("Initial records collide with database keys")
    for event in initial["calendar"]:
        date.fromisoformat(event["date"])
    for messages in apps["messenger"]["chat_history"].values():
        if any(len(m) != 4 or not isinstance(m[1], bool) or not all(isinstance(m[i], str) for i in (0, 2, 3)) for m in messages):
            raise ValueError("Invalid initial message shape")
    # An exact terminal answer must not be present in the public instruction.
    answer_key = {"add_todo": "title", "add_event": "title", "send_message": "message", "remove_place": "name"}[expected["type"]]
    if expected[answer_key] in spec.goal:
        raise ValueError("Public goal leaks the resolved terminal answer")
    needed = {ref for node in graph["reads"] for ref in _references(node["selector"])} | set(_references(graph["operation"]))
    counters = {"source_removal_checks": 0, "counterfactual_answer_changes": 0, "broken_reference_checks": 0}
    for node in graph["reads"]:
        location = locations[node["id"]]
        missing = deepcopy(apps)
        _at(missing, location[:-1]).pop(location[-1])
        try:
            resolve_graph(missing, graph)
        except ValueError:
            counters["source_removal_checks"] += 1
        else:
            raise ValueError("A declared source is unnecessary")
        for name, source in node["exports"].items():
            if f"{node['id']}.{name}" not in needed:
                raise ValueError("Declared dependency field is unused")
            value = values[node["id"]][name]
            replacement = ((date.fromisoformat(value) + timedelta(days=1)).isoformat()
                           if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value)) else str(value) + " revised")
            changed = deepcopy(apps)
            _mutate_field(changed, node, location, source, replacement)
            try:
                result, _, _ = resolve_graph(changed, graph)
            except ValueError:
                counters["broken_reference_checks"] += 1
            else:
                if result == expected:
                    raise ValueError("A source field does not affect its claimed result")
                counters["counterfactual_answer_changes"] += 1
    distractor = deepcopy(apps)
    distractor["todo"]["init_todos"].append(["Unrelated background note for offline dependency validation", False])
    if resolve_graph(distractor, graph)[0] != expected:
        raise ValueError("Unrelated background changes the answer")
    verifier = instantiate(spec.task_config)
    target = apply_operation(initial, expected)
    with redirect_stdout(io.StringIO()):
        if verifier.check_if_task_is_complete(initial, initial) or not verifier.check_if_task_is_complete(initial, target):
            raise ValueError("Initial/positive final-state verification failed")
        wrong = deepcopy(target)
        if expected["type"] == "remove_place":
            wrong = deepcopy(initial)
            other = next(row for row in wrong["map"] if row["name"] != expected["name"])
            wrong["map"].remove(other)
        elif expected["type"] == "add_todo":
            wrong["todo"][-1]["title"] += " incorrect"
        elif expected["type"] == "add_event":
            wrong["calendar"][-1]["date"] = (date.fromisoformat(expected["date"]) + timedelta(days=1)).isoformat()
        else:
            next(row for row in wrong["messenger"] if row["user"] == expected["to"])["messages"][-1][0] += " incorrect"
        if verifier.check_if_task_is_complete(initial, wrong):
            raise ValueError("Incorrect final answer passes verification")
        extra = deepcopy(target)
        extra["todo"].append({"title": "Unrequested side effect", "done": False})
        if verifier.check_if_task_is_complete(initial, extra):
            raise ValueError("Unrequested modification passes verification")
    if progress_fraction(spec.progress_predicates, initial, initial) != 0 or progress_fraction(spec.progress_predicates, initial, target) != 1:
        raise ValueError("Progress predicates do not match the actual final operation")
    return counters | {"unique_source_resolution": True, "irrelevant_data_invariance": True,
                       "initial_rejected": True, "correct_final_state_accepted": True,
                       "wrong_final_state_rejected": True, "extra_change_rejected": True}
