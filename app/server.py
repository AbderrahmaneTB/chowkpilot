"""ChowkPilot control room: optional live-simulation backend.

    uvicorn app.server:app --port 7860      ->  http://localhost:7860

The control room itself is a static site in docs/ (also published with GitHub
Pages) that plays pre-recorded SUMO runs from docs/data/. This server serves that
same site and adds one endpoint, POST /api/simulate, which runs two fresh SUMO
simulations (a baseline and ChowkPilot, identical vehicles) in worker processes.
"""

from __future__ import annotations

import asyncio
import gzip
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import orjson
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from chowkpilot.eval import replay as rp  # noqa: E402
from chowkpilot.eval.make_replays import DEMO  # noqa: E402

STATIC = os.path.join(ROOT, "docs")  # the static site, also published with GitHub Pages
MAX_LIVE = int(os.environ.get("CHOWK_MAX_LIVE", "2"))
BASELINES = {"fixed", "webster", "actuated", "maxpressure"}

app = FastAPI(title="ChowkPilot", version="1.0.0")
_pool: ProcessPoolExecutor | None = None
_live = asyncio.Semaphore(MAX_LIVE)


def pool() -> ProcessPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=MAX_LIVE * 2)
    return _pool


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


@app.get("/api/healthz")
def healthz():
    return {"ok": True, "live": True}


@app.post("/api/simulate")
async def simulate(req: SimRequest):
    if req.scenario not in DEMO or req.baseline not in BASELINES:
        raise HTTPException(400, "unsupported scenario or baseline")
    if req.profile not in ("flat", "peak", "ramp", "surge", "wave"):
        raise HTTPException(400, "unsupported profile")
    if _live.locked():
        raise HTTPException(429, "The simulator is busy; please retry in a few seconds.")
    demand = {"scale": req.scale, "profile": req.profile, "ambulances_per_hour": req.ambulances}
    noise = {"noise_sigma": req.noise, "miss_prob": req.noise / 2} if req.noise > 0 else {}
    loop = asyncio.get_running_loop()
    async with _live:
        jobs = [
            loop.run_in_executor(pool(), rp.record, req.scenario, req.baseline, req.seed, req.duration,
                                 demand, dict(noise)),
            loop.run_in_executor(pool(), rp.record, req.scenario, "chowkpilot", req.seed, req.duration,
                                 demand, {**noise, "ev_priority": req.ev_priority}),
        ]
        try:
            base, ours = await asyncio.wait_for(asyncio.gather(*jobs), timeout=300)
        except asyncio.TimeoutError:
            raise HTTPException(504, "simulation took too long")
    body = gzip.compress(orjson.dumps({"request": req.model_dump(), "baseline": base, "chowkpilot": ours},
                                      option=orjson.OPT_SERIALIZE_NUMPY), compresslevel=5)
    return Response(body, media_type="application/json", headers={"Content-Encoding": "gzip"})


app.mount("/", StaticFiles(directory=STATIC, html=True), name="site")
