"""Export everything the static control room needs into docs/data/.

    python -m chowkpilot.eval.export_site

Writes scenarios.json (catalogue + available replays), network/<id>.json.gz
(road geometry, signals, phase labels) and benchmark.json (summary tables).
"""

from __future__ import annotations

import gzip
import json
import os

from ..scenarios import registry
from .make_replays import CONDITIONS, DEMO
from .replay import REPLAY_DIR, SITE_DATA, network_geometry


def main():
    os.makedirs(os.path.join(SITE_DATA, "network"), exist_ok=True)
    files = set(os.listdir(REPLAY_DIR)) if os.path.isdir(REPLAY_DIR) else set()
    by_id = {s.id: s for s in registry.load_all()}
    catalogue = []
    for sid in DEMO:
        s = by_id[sid]
        geom = network_geometry(sid)
        with gzip.open(os.path.join(SITE_DATA, "network", f"{sid}.json.gz"), "wt", encoding="utf-8") as f:
            json.dump(geom, f, separators=(",", ":"))
        peak = sorted({f.split("__")[1].split(".")[0] for f in files
                       if f.startswith(sid + "__") and f.count("__") == 1})
        conds = {}
        for key, (label, _, _) in CONDITIONS.items():
            ctrls = sorted({f.split("__")[1] for f in files if f.startswith(sid + "__") and f.endswith(f"__{key}.json.gz")})
            if ctrls:
                conds[key] = {"label": label, "controllers": ctrls}
        catalogue.append({"id": sid, "title": s.title, "split": s.split, "kind": s.kind,
                          "attribution": s.description, "replays": peak, "conditions": conds,
                          "junctions": len(geom["tls"])})
    with open(os.path.join(SITE_DATA, "scenarios.json"), "w", encoding="utf-8") as f:
        json.dump(catalogue, f, indent=1)
    bench = os.path.join(registry.ROOT, "results", "benchmark.json")
    if os.path.isfile(bench):
        with open(bench, encoding="utf-8") as f:
            b = json.load(f)
        b.pop("rows", None)
        with open(os.path.join(SITE_DATA, "benchmark.json"), "w", encoding="utf-8") as f:
            json.dump(b, f, separators=(",", ":"))
    print(f"exported {len(catalogue)} scenarios, {len(files)} replays")


if __name__ == "__main__":
    main()
