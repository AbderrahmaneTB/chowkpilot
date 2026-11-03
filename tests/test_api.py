"""Live backend: health check and one short paired simulation through the HTTP API.

    python -m pytest -q
"""

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from app.server import app  # noqa: E402


def test_health():
    with TestClient(app) as client:
        assert client.get("/api/healthz").json() == {"ok": True, "live": True}


def test_simulate_runs_two_real_simulations():
    req = {"scenario": "syn_arterial3", "baseline": "fixed", "duration": 300, "seed": 3, "ambulances": 6}
    with TestClient(app) as client:
        r = client.post("/api/simulate", json=req)
        assert r.status_code == 200, r.text
        body = r.json()
        for side in ("baseline", "chowkpilot"):
            run = body[side]
            assert run["trips"] > 0 and run["mean_delay"] >= 0
            assert len(run["frames"]) >= 300  # 1 Hz recording of the requested window
        assert body["chowkpilot"]["explain"], "ChowkPilot runs carry per-decision explanations"
        # same seeded demand on both sides: identical vehicles, so a fair comparison
        assert body["baseline"]["demand"] == body["chowkpilot"]["demand"]
        assert client.post("/api/simulate", json={**req, "scenario": "nowhere"}).status_code == 400
