"""ChowkPilot control room: FastAPI backend.

    uvicorn app.server:app --port 7860

Serves the static front end, network geometry, pre-recorded benchmark replays and
live simulations. Every simulation runs in a worker process (libsumo allows one
simulation per process) and returns real SUMO output.
"""

from __future__ import annotations

import asyncio
import gzip
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import orjson
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from chowkpilot.eval import replay as rp  # noqa: E402
from chowkpilot.scenarios import registry  # noqa: E402

STATIC = os.path.join(ROOT, "app", "static")
RESULTS = os.path.join(ROOT, "results")
MAX_LIVE = int(os.environ.get("CHOWK_MAX_LIVE", "2"))
DEMO_SCENARIOS = ["hyd_secunderabad", "hyd_hitech", "hyd_ameerpet", "syn_arterial3", "syn_grid3x3"]
BASELINES = {"fixed", "webster", "actuated", "maxpressure"}

app = FastAPI(title="ChowkPilot", version="1.0.0")
_pool: ProcessPoolExecutor | None = None
_live = asyncio.Semaphore(MAX_LIVE)
_geometry: dict[str, bytes] = {}


def pool() -> ProcessPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=MAX_LIVE * 2)
    return _pool


def gz_json(obj) -> Response:
    body = gzip.compress(orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY), compresslevel=5)
    return Response(body, media_type="application/json", headers={"Content-Encoding": "gzip"})


@app.get("/api/scenarios")
def scenarios():
    by_id = {s.id: s for s in registry.load_all()}
    out = []
    for sid in DEMO_SCENARIOS:
        s = by_id.get(sid)
        if s is None:
            continue
        replays = sorted(f.split("__")[1].split(".")[0] for f in os.listdir(os.path.join(RESULTS, "replays"))
                         if f.startswith(sid + "__")) if os.path.isdir(os.path.join(RESULTS, "replays")) else []
        out.append({"id": s.id, "title": s.title, "split": s.split, "kind": s.kind, "replays": replays,
                    "attribution": s.description})
    return out


@app.get("/api/network/{sid}")
def network(sid: str):
    if sid not in _geometry:
        try:
            _geometry[sid] = orjson.dumps(rp.network_geometry(sid))
        except KeyError:
            raise HTTPException(404, "unknown scenario")
    return Response(gzip.compress(_geometry[sid]), media_type="application/json",
                    headers={"Content-Encoding": "gzip", "Cache-Control": "public, max-age=3600"})


@app.get("/api/replay/{sid}/{controller}")
def replay(sid: str, controller: str):
    path = os.path.join(RESULTS, "replays", f"{sid}__{controller}.json.gz")
    if not os.path.isfile(path) or not controller.replace("_", "").isalnum():
        raise HTTPException(404, "replay not found")
    with open(path, "rb") as f:
        return Response(f.read(), media_type="application/json",
                        headers={"Content-Encoding": "gzip", "Cache-Control": "public, max-age=3600"})


@app.get("/api/benchmark")
def benchmark():
    path = os.path.join(RESULTS, "benchmark.json")
    if not os.path.isfile(path):
        raise HTTPException(404, "benchmark not computed yet")
    return FileResponse(path, media_type="application/json")


class SimRequest(BaseModel):
    scenario: str = "hyd_secunderabad"
    baseline: str = "fixed"
    scale: float = Field(1.0, ge=0.3, le=1.6)
    profile: str = "peak"
    ambulances: float = Field(4.0, ge=0, le=12)
    noise: float = Field(0.0, ge=0, le=0.4)
    ev_priority: bool = True
    duration: int = Field(600, ge=300, le=900)
    seed: int = Field(7, ge=0, le=10_000)


@app.post("/api/simulate")
async def simulate(req: SimRequest):
    if req.scenario not in DEMO_SCENARIOS or req.baseline not in BASELINES:
        raise HTTPException(400, "unsupported scenario or baseline")
    if req.profile not in ("flat", "peak", "ramp", "surge", "wave"):
        raise HTTPException(400, "unsupported profile")
    if _live.locked():
        raise HTTPException(429, "The simulator is busy with other visitors; please retry in a few seconds.")
    demand = {"scale": req.scale, "profile": req.profile, "ambulances_per_hour": req.ambulances}
    noise = {"noise_sigma": req.noise, "miss_prob": req.noise / 2} if req.noise > 0 else {}
    loop = asyncio.get_running_loop()
    async with _live:
        jobs = [
            loop.run_in_executor(pool(), rp.record, req.scenario, req.baseline, req.seed, req.duration,
                                 demand, {**noise}),
            loop.run_in_executor(pool(), rp.record, req.scenario, "chowkpilot", req.seed, req.duration,
                                 demand, {**noise, "ev_priority": req.ev_priority}),
        ]
        try:
            base, ours = await asyncio.wait_for(asyncio.gather(*jobs), timeout=240)
        except asyncio.TimeoutError:
            raise HTTPException(504, "simulation took too long")
    return gz_json({"request": req.model_dump(), "baseline": base, "chowkpilot": ours})


@app.get("/healthz")
def healthz():
    return {"ok": True}


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
