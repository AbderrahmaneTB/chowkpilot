/* ChowkPilot control room: renders recorded SUMO simulations side by side. */
"use strict";

const TYPE_COLORS = ["#fbbf24", "#60a5fa", "#34d399", "#f472b6", "#a78bfa", "#ffffff"];
const TYPE_NAMES = ["Two-wheeler", "Car", "Auto-rickshaw", "Bus", "Light truck", "Ambulance"];
const CTRL_NAMES = {
  fixed: "Fixed-time (standard plan)", webster: "Fixed-time (Webster-optimised)",
  actuated: "Vehicle-actuated", maxpressure: "Max-pressure", chowkpilot: "ChowkPilot (ours)",
};
// Comparison colors validated with the dataviz CVD checker on the #0f172b surface
// (blue vs green: deutan dE 24.3, normal dE 26.4, both >= 3:1 contrast).
const SERIES = { base: "#3987e5", ours: "#16a34a", muted: "#6b7aa6" };
const $ = (id) => document.getElementById(id);
const fmtTime = (t) => `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
const pct = (a, b) => (b ? ((a - b) / b) * 100 : 0);

const state = {
  geom: null, base: null, ours: null, t: 0, playing: false, speed: 5, tmax: 900,
  cam: { cx: 0, cy: 0, scale: 1 }, scenario: null, baseline: "fixed", custom: false,
};

/* ------------------------------------------------------------------ data */
async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = `${r.status}`;
    try { msg = (await r.json()).detail || msg; } catch (_) { /* ignore */ }
    throw new Error(msg);
  }
  return r.json();
}

// Static data: .json or gzipped .json.gz. Some hosts decompress transparently, so sniff the magic bytes.
async function getData(path) {
  const r = await fetch(`data/${path}`);
  if (!r.ok) throw new Error(`${r.status} for ${path}`);
  const buf = new Uint8Array(await r.arrayBuffer());
  if (buf[0] === 0x1f && buf[1] === 0x8b) {
    const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
    return JSON.parse(await new Response(stream).text());
  }
  return JSON.parse(new TextDecoder().decode(buf));
}

function indexFrames(rec) {
  // per-frame id -> offset maps, built lazily
  rec._maps = new Array(rec.frames.length);
  rec._t0 = rec.frames.length ? rec.frames[0].t : 0;
  return rec;
}

function frameMap(rec, k) {
  if (!rec._maps[k]) {
    const v = rec.frames[k].v, m = new Map();
    for (let i = 0; i < v.length; i += 5) m.set(v[i], i);
    rec._maps[k] = m;
  }
  return rec._maps[k];
}

/* ------------------------------------------------------------------ views */
class View {
  constructor(canvas) {
    this.cv = canvas;
    this.ctx = canvas.getContext("2d");
    this.bg = document.createElement("canvas");
    this.dirty = true;
    new ResizeObserver(() => { this.resize(); render(); }).observe(canvas);
    this.resize();
  }
  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = this.cv.getBoundingClientRect();
    this.w = Math.max(10, r.width); this.h = Math.max(10, r.height);
    this.cv.width = this.w * dpr; this.cv.height = this.h * dpr;
    this.bg.width = this.cv.width; this.bg.height = this.cv.height;
    this.dpr = dpr; this.dirty = true;
  }
  toScreen(x, y) {
    const c = state.cam;
    return [(x - c.cx) * c.scale + this.w / 2, this.h / 2 - (y - c.cy) * c.scale];
  }
  drawStatic() {
    const g = state.geom; if (!g) return;
    const ctx = this.bg.getContext("2d");
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.fillStyle = "#050913"; ctx.fillRect(0, 0, this.w, this.h);
    const s = state.cam.scale;
    ctx.fillStyle = "#1a2440";
    for (const poly of g.junctions) {
      ctx.beginPath();
      poly.forEach(([x, y], i) => { const [px, py] = this.toScreen(x, y); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
      ctx.closePath(); ctx.fill();
    }
    ctx.lineCap = "butt"; ctx.lineJoin = "round";
    for (const lane of g.lanes) {
      ctx.strokeStyle = "#1a2440"; ctx.lineWidth = Math.max(1.2, lane.w * s);
      ctx.beginPath();
      lane.s.forEach(([x, y], i) => { const [px, py] = this.toScreen(x, y); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
      ctx.stroke();
    }
    if (s > 0.9) { // lane markings when zoomed in
      ctx.setLineDash([3 * s, 6 * s]); ctx.strokeStyle = "rgba(160,175,220,0.18)"; ctx.lineWidth = Math.max(0.6, 0.12 * s);
      for (const lane of g.lanes) {
        ctx.beginPath();
        lane.s.forEach(([x, y], i) => { const [px, py] = this.toScreen(x, y); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }
    ctx.font = "500 11px Inter, sans-serif"; ctx.fillStyle = "rgba(141,154,196,0.75)"; ctx.textAlign = "center";
    for (const n of g.names) {
      const [px, py] = this.toScreen(n.x, n.y);
      if (px < -50 || py < -20 || px > this.w + 50 || py > this.h + 20) continue;
      let a = -n.a; if (a > Math.PI / 2) a -= Math.PI; if (a < -Math.PI / 2) a += Math.PI;
      ctx.save(); ctx.translate(px, py); ctx.rotate(a); ctx.fillText(n.n, 0, -8); ctx.restore();
    }
    this.dirty = false;
  }
  draw(rec, t) {
    if (this.dirty) this.drawStatic();
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(this.bg, 0, 0);
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    if (!rec || !rec.frames.length) return;
    const n = rec.frames.length;
    const f = Math.max(0, Math.min(n - 1, t - rec._t0));
    const k = Math.floor(f), k2 = Math.min(n - 1, k + 1), u = f - k;
    const A = rec.frames[k], B = rec.frames[k2], mB = frameMap(rec, k2);
    const s = state.cam.scale, g = state.geom;
    // signals
    for (const sig of g.signals) {
      const st = A.s[sig.j]; if (!st) continue;
      let col = "#ef4444";
      for (const i of sig.links) { const c = st[i]; if (c === "G" || c === "g") { col = "#22c55e"; break; } if (c === "y" || c === "Y") col = "#f59e0b"; }
      const [px, py] = this.toScreen(sig.x, sig.y);
      const half = Math.max(2.5, sig.w * s * 0.5);
      ctx.save(); ctx.translate(px, py); ctx.rotate(-sig.a);
      ctx.fillStyle = col; ctx.shadowColor = col; ctx.shadowBlur = 6;
      ctx.fillRect(-Math.max(1.5, 0.6 * s), -half, Math.max(2.5, 1.1 * s), half * 2);
      ctx.restore();
    }
    // vehicles
    const va = A.v, vb = B.v;
    const blink = Math.floor(performance.now() / 250) % 2;
    for (let i = 0; i < va.length; i += 5) {
      const id = va[i]; let x = va[i + 1] / 10, y = va[i + 2] / 10, ang = va[i + 3];
      const j = mB.get(id);
      if (j !== undefined && u > 0) {
        const xb = vb[j + 1] / 10, yb = vb[j + 2] / 10;
        if (Math.abs(xb - x) + Math.abs(yb - y) < 60) {
          x += (xb - x) * u; y += (yb - y) * u;
          let d = vb[j + 3] - ang; if (d > 180) d -= 360; if (d < -180) d += 360; ang += d * u;
        }
      }
      const code = va[i + 4], ty = code % 10, halted = code >= 10;
      const T = g.types[ty] || g.types[1];
      const [px, py] = this.toScreen(x, y);
      if (px < -20 || py < -20 || px > this.w + 20 || py > this.h + 20) continue;
      const L = Math.max(2.4, T.length * s), W = Math.max(1.4, T.width * s);
      ctx.save(); ctx.translate(px, py); ctx.rotate((ang * Math.PI) / 180);
      ctx.fillStyle = TYPE_COLORS[ty] || "#ccc";
      if (ty === 5) ctx.fillStyle = blink ? "#ffffff" : "#ff4d4d";
      ctx.fillRect(-W / 2, 0, W, L); // SUMO position = front bumper, angle 0 = north
      if (halted && ty !== 5) { ctx.fillStyle = "rgba(239,68,68,0.55)"; ctx.fillRect(-W / 2, 0, W, L); }
      ctx.restore();
    }
  }
}

function fitCamera() {
  const g = state.geom; if (!g) return;
  const v = views.base;
  // frame the signalised area (stop lines) with a margin, not the whole OSM extract
  const pts = g.signals.length ? g.signals : [{ x: (g.bounds[0] + g.bounds[2]) / 2, y: (g.bounds[1] + g.bounds[3]) / 2 }];
  let x0 = Math.min(...pts.map((p) => p.x)), x1 = Math.max(...pts.map((p) => p.x));
  let y0 = Math.min(...pts.map((p) => p.y)), y1 = Math.max(...pts.map((p) => p.y));
  const m = Math.max(90, 0.18 * Math.max(x1 - x0, y1 - y0));
  x0 -= m; x1 += m; y0 -= m; y1 += m;
  state.cam = { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, scale: Math.min(v.w / (x1 - x0), v.h / (y1 - y0)) };
  Object.values(views).forEach((vv) => (vv.dirty = true));
}

function attachPanZoom(view) {
  const cv = view.cv; let drag = null;
  cv.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = cv.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top, c = state.cam;
    const wx = c.cx + (mx - view.w / 2) / c.scale, wy = c.cy - (my - view.h / 2) / c.scale;
    const f = Math.exp(-e.deltaY * 0.0015);
    c.scale = Math.min(40, Math.max(0.05, c.scale * f));
    c.cx = wx - (mx - view.w / 2) / c.scale; c.cy = wy + (my - view.h / 2) / c.scale;
    Object.values(views).forEach((v) => (v.dirty = true)); render();
  }, { passive: false });
  cv.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY }; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const c = state.cam; c.cx -= (e.clientX - drag.x) / c.scale; c.cy += (e.clientY - drag.y) / c.scale;
    drag = { x: e.clientX, y: e.clientY };
    Object.values(views).forEach((v) => (v.dirty = true)); render();
  });
  cv.addEventListener("pointerup", () => (drag = null));
}

/* ------------------------------------------------------------------ HUD, series, explanation */
function stoppedSoFar(rec, t) {
  const s = rec.series; let acc = 0;
  for (let i = 0; i < s.t.length && s.t[i] <= t; i++) acc += s.halting[i] * 10;
  return acc / 60; // vehicle-minutes
}

function frameCounts(rec, t) {
  const k = Math.max(0, Math.min(rec.frames.length - 1, Math.floor(t - rec._t0)));
  const v = rec.frames[k].v; let halted = 0;
  for (let i = 4; i < v.length; i += 5) if (v[i] >= 10) halted++;
  return { n: v.length / 5, halted };
}

function hud(el, rec, t, other) {
  if (!rec) { el.innerHTML = ""; return; }
  const c = frameCounts(rec, t), lost = stoppedSoFar(rec, t);
  let cls = "";
  if (other) { const lo = stoppedSoFar(other, t); cls = lost < lo ? "good" : ""; }
  el.innerHTML = `<div><div class="n">${c.halted}</div><div class="l">stopped now</div></div>
    <div><div class="n">${c.n}</div><div class="l">vehicles in network</div></div>
    <div><div class="n ${cls}">${Math.round(lost).toLocaleString()}</div><div class="l">vehicle-minutes stopped</div></div>`;
}

function drawSeries() {
  const cv = $("cvSeries"), dpr = Math.min(window.devicePixelRatio || 1, 2);
  const r = cv.getBoundingClientRect(); cv.width = r.width * dpr; cv.height = r.height * dpr;
  const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = r.width, H = r.height, pad = { l: 38, r: 10, t: 10, b: 22 };
  ctx.clearRect(0, 0, W, H);
  const recs = [[state.base, SERIES.base], [state.ours, SERIES.ours]].filter((x) => x[0]);
  if (!recs.length) return;
  const tmax = state.tmax; let ymax = 10;
  for (const [rec] of recs) rec.series.t.forEach((tt, i) => { if (tt <= tmax) ymax = Math.max(ymax, rec.series.halting[i]); });
  ymax = Math.ceil(ymax / 10) * 10;
  const X = (t) => pad.l + (t / tmax) * (W - pad.l - pad.r), Y = (v) => H - pad.b - (v / ymax) * (H - pad.t - pad.b);
  ctx.strokeStyle = "#1f2b4b"; ctx.fillStyle = "#5c6a96"; ctx.font = "11px JetBrains Mono, monospace"; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) { const v = (ymax * i) / 4, y = Y(v); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke(); ctx.fillText(String(Math.round(v)), 4, y + 4); }
  for (let m = 0; m <= tmax / 60; m += 5) ctx.fillText(`${m}m`, X(m * 60) - 8, H - 6);
  for (const [rec, col] of recs) {
    ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath();
    let first = true;
    rec.series.t.forEach((tt, i) => { if (tt > tmax) return; const x = X(tt), y = Y(rec.series.halting[i]); first ? ctx.moveTo(x, y) : ctx.lineTo(x, y); first = false; });
    ctx.stroke();
  }
  ctx.strokeStyle = "rgba(232,237,255,.5)"; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(X(state.t), pad.t); ctx.lineTo(X(state.t), H - pad.b); ctx.stroke(); ctx.setLineDash([]);
  // legend: colored line-key beside text in a text token (text never wears the series color)
  ctx.font = "600 11px Inter, sans-serif"; ctx.lineWidth = 2;
  [[CTRL_NAMES[state.baseline] || "Baseline", SERIES.base], ["ChowkPilot", SERIES.ours]].forEach(([label, col], i) => {
    const y = pad.t + 10 + i * 15;
    ctx.strokeStyle = col; ctx.beginPath(); ctx.moveTo(pad.l + 8, y - 4); ctx.lineTo(pad.l + 22, y - 4); ctx.stroke();
    ctx.fillStyle = "#c7d0ee"; ctx.fillText(label, pad.l + 28, y);
  });
}

function explain() {
  const bars = $("bars"), why = $("why");
  const rec = state.ours, g = state.geom;
  if (!rec || !rec.explain || !rec.explain.length || !g) { bars.innerHTML = ""; why.textContent = ""; return; }
  const j = Number($("selJunction").value || 0);
  let e = null;
  for (const x of rec.explain) { if (x.j !== j) continue; if (x.t > state.t) break; e = x; }
  if (!e) { bars.innerHTML = `<div class="sub">Waiting for the first decision at this junction…</div>`; why.textContent = ""; return; }
  const labels = (g.tls[j] && g.tls[j].phase_labels) || [];
  const name = (p) => labels[p] || `Phase ${p + 1}`;
  const q = e.q, lo = Math.min(...q), hi = Math.max(...q), span = Math.max(1e-6, hi - lo);
  const evid = (p) => (e.stop ? `${e.stop[p]} stopped · wait ${e.wait[p]} s · MP ${e.mp[p]}` : "");
  bars.innerHTML = q.map((v, p) => {
    const w = 8 + (92 * (v - lo)) / span;
    return `<div class="bar ${p === e.a ? "chosen" : ""}" title="${name(p)}: score ${v.toFixed(2)}${e.stop ? " · " + evid(p) : ""}">
      <div class="track"><div class="fill" style="width:${w}%"></div>
      <div class="lbl">${p === e.cur ? "● " : ""}${name(p)}${e.stop ? ` <span class="ev">${evid(p)}</span>` : ""}</div></div>
      <div class="val">${v.toFixed(2)}</div></div>`;
  }).join("");
  const keep = e.a === e.cur;
  const detail = (p) => (e.stop ? ` (${e.stop[p]} stopped vehicles, mean wait up to ${e.wait[p]} s)` : "");
  why.innerHTML = `At <b>${fmtTime(e.t)}</b>: ${keep ? "kept" : "switched to"} <b>${name(e.a)}</b>${detail(e.a)}` +
    (keep ? ", still the highest-value phase." : `, over <b>${name(e.cur)}</b>${detail(e.cur)}.`) +
    ` <span style="color:var(--faint)">● = phase that was green · MP = max-pressure score the network receives as a prior</span>`;
}

/* ------------------------------------------------------------------ main loop */
const views = {};
let lastTs = null;
function tick(ts) {
  if (state.playing && lastTs !== null) {
    state.t = Math.min(state.tmax, state.t + ((ts - lastTs) / 1000) * state.speed);
    if (state.t >= state.tmax) { state.playing = false; $("btnPlay").textContent = "▶ Play"; }
  }
  lastTs = ts;
  render();
  requestAnimationFrame(tick);
}

let lastPanel = 0;
function render() {
  if (!views.base) return;
  views.base.draw(state.base, state.t);
  views.ours.draw(state.ours, state.t);
  $("clock").textContent = fmtTime(state.t);
  $("timeSlider").value = state.t;
  const now = performance.now();
  if (now - lastPanel > 200) {
    hud($("hudBase"), state.base, state.t);
    hud($("hudOurs"), state.ours, state.t, state.base);
    drawSeries(); explain(); lastPanel = now;
  }
}

async function loadScenario(sid, baseline) {
  state.scenario = sid; state.baseline = baseline; state.custom = false;
  $("baseName").textContent = CTRL_NAMES[baseline];
  ["loadBase", "loadOurs"].forEach((id) => $(id).classList.remove("hidden"));
  try {
    const [geom, base, ours] = await Promise.all([
      getData(`network/${sid}.json.gz`), getData(`replays/${sid}__${baseline}.json.gz`), getData(`replays/${sid}__chowkpilot.json.gz`)]);
    setData(geom, base, ours);
  } catch (err) {
    $("loadBase").textContent = $("loadOurs").textContent = `Could not load replay (${err.message}).`;
  }
}

function setData(geom, base, ours) {
  const geomChanged = !state.geom || state.geom.id !== geom.id;
  state.geom = geom; state.base = indexFrames(base); state.ours = indexFrames(ours);
  state.tmax = Math.min(base.frames.length, ours.frames.length) - 1 + base._t0;
  $("timeSlider").max = state.tmax;
  if (geomChanged) {
    fitCamera();
    $("selJunction").innerHTML = geom.tls.map((t, i) =>
      `<option value="${i}">${t.streets.length ? t.streets.slice(0, 2).join(" × ") : "Junction " + (i + 1)}</option>`).join("");
  }
  state.t = Math.min(state.t, state.tmax);
  Object.values(views).forEach((v) => (v.dirty = true));
  ["loadBase", "loadOurs"].forEach((id) => $(id).classList.add("hidden"));
  runSummary(base, ours);
  render();
}

function runSummary(b, o) {
  const item = (label, bv, ov, unit, dp) => {
    const ch = pct(ov, bv), cls = ch < 0 ? "good" : "bad";
    return `<span>${label}: <b>${bv.toFixed(dp)}${unit}</b> → <b>${ov.toFixed(dp)}${unit}</b> <span class="${cls}">(${ch > 0 ? "+" : "−"}${Math.abs(ch).toFixed(0)}%)</span></span>`;
  };
  $("runSummary").innerHTML = `<span>This run, whole network after it empties (${CTRL_NAMES[state.baseline] || "baseline"} → ChowkPilot):</span>` +
    item("mean delay", b.mean_delay, o.mean_delay, " s", 0) + item("stops per vehicle", b.mean_stops, o.mean_stops, "", 2) +
    item("CO₂", b.co2_kg, o.co2_kg, " kg", 0);
}

/* ------------------------------------------------------------------ benchmark */
function hbarRows(rows, fmt) {
  // rows: [{name, value, ci, ours, tip}] -> horizontal bars sharing one zero baseline
  const vals = rows.map((r) => r.value).filter((v) => v !== null && !Number.isNaN(v));
  const lo = Math.min(0, ...rows.map((r) => (r.value ?? 0) - (r.ci || 0)));
  const hi = Math.max(0, ...rows.map((r) => (r.value ?? 0) + (r.ci || 0)));
  const span = Math.max(1e-9, hi - lo), X = (v) => ((v - lo) / span) * 100, z = X(0);
  if (!vals.length) return "";
  return rows.map((r) => {
    if (r.value === null || Number.isNaN(r.value)) return "";
    const a = X(Math.min(0, r.value)), b = X(Math.max(0, r.value));
    const ci = r.ci ? `<div class="ci" style="left:${X(r.value - r.ci)}%;width:${X(r.value + r.ci) - X(r.value - r.ci)}%"></div>` : "";
    return `<div class="hb ${r.ours ? "ours" : ""}" title="${r.tip || ""}"><div class="name">${r.name}</div>
      <div class="trk"><div class="zero" style="left:${z}%"></div><div class="fill ${r.value < 0 ? "neg" : ""}" style="left:${a}%;width:${Math.max(0.6, b - a)}%"></div>${ci}</div>
      <div class="val">${fmt(r.value)}</div></div>`;
  }).join("");
}

const SHORT = { fixed: "Fixed-time", webster: "Webster", actuated: "Actuated", maxpressure: "Max-pressure", chowkpilot: "ChowkPilot" };

async function loadBenchmark() {
  let b;
  try { b = await getData("benchmark.json"); } catch (_) {
    $("kpis").innerHTML = `<div class="kpi"><div class="v neutral">-</div><div class="k">Benchmark not available yet</div></div>`;
    return;
  }
  const s = b.summary;
  const sg = (x) => (x > 0 ? "+" : "−") + Math.abs(x).toFixed(0) + "%";
  const k = (x, label, sub) => `<div class="kpi"><div class="v ${x > 0 ? "neutral" : ""}">${sg(x)}</div><div class="k">${label}</div><div class="s">${sub}</div></div>`;
  $("kpis").innerHTML =
    k(s.delay_vs_fixed, "average delay vs today's fixed-time signals", "unseen Hyderabad junctions, peak demand") +
    k(s.delay_vs_actuated, "average delay vs vehicle-actuated control", "the strongest classical baseline") +
    k(s.co2_vs_fixed, "CO₂ from traffic in the network", "HBEFA4 emission model, vs fixed-time") +
    k(s.ambulance_vs_fixed, "ambulance delay with green corridor", "vs fixed-time, same ambulances");
  $("benchIntro").textContent = `Mean delay per vehicle, ${b.seeds} paired seeds per junction, ${b.horizon / 60}-minute peak with the network fully drained. ` +
    "Whiskers are 95% confidence intervals. These real Hyderabad areas were never used for training or model selection.";
  $("benchDelay").innerHTML = b.test_scenarios.map((sid) => {
    const cells = b.cells[sid];
    const rows = b.controllers.map((c) => ({ name: SHORT[c], value: cells[c].mean_delay.mean, ci: cells[c].mean_delay.ci, ours: c === "chowkpilot",
      tip: `${CTRL_NAMES[c]}: ${cells[c].mean_delay.mean.toFixed(1)} s mean delay (95% CI ±${cells[c].mean_delay.ci.toFixed(1)}, n=${cells[c].mean_delay.n})` }));
    const pa = (b.paired[sid] || {}).actuated, pf = (b.paired[sid] || {}).fixed;
    const delta = pf && pa ? `<div class="delta">ChowkPilot vs fixed-time: <b>${sg(pf.delay_change_pct)}</b> (wins ${pf.wins}/${pf.n} seeds) · vs actuated: <b>${sg(pa.delay_change_pct)}</b> (wins ${pa.wins}/${pa.n}, p=${pa.p_value < 0.001 ? "<0.001" : pa.p_value.toFixed(3)})</div>` : "";
    return `<div class="sm"><div class="sm-title">${b.titles[sid]}</div><div class="sm-sub">mean delay per vehicle, seconds</div>${hbarRows(rows, (v) => v.toFixed(1) + " s")}${delta}</div>`;
  }).join("");
  if (b.robustness) {
    $("benchRobust").innerHTML = b.robustness.conditions.map((cond) => {
      const rows = b.robustness.controllers.map((c) => ({ name: SHORT[c], value: cond.values[c] === null ? null : -cond.values[c], ours: c === "chowkpilot",
        tip: `${CTRL_NAMES[c]}: ${cond.values[c] > 0 ? "+" : ""}${cond.values[c]}% delay vs fixed-time (mean delay ${cond.mean_delay[c]} s; fixed-time ${cond.mean_delay.fixed} s)` }));
      return `<div class="sm"><div class="sm-title">${cond.label}</div><div class="sm-sub">fixed-time: ${cond.mean_delay.fixed} s</div>${hbarRows(rows, (v) => (v >= 0 ? "−" : "+") + Math.abs(v).toFixed(0) + "%")}</div>`;
    }).join("");
  }
  const ctrls = b.controllers;
  const rows = [["mean_delay", "Mean delay (s)", 1], ["p95_delay", "95th-percentile delay (s)", 1], ["mean_stops", "Stops per vehicle", 2],
    ["throughput_horizon", "Vehicles served in peak window", 0], ["co2_kg", "CO₂ (kg)", 1], ["ambulance_delay", "Ambulance delay (s)", 1]];
  let html = `<tr><th>Junction / metric</th>${ctrls.map((c) => `<th>${CTRL_NAMES[c]}</th>`).join("")}</tr>`;
  for (const x of b.test_scenarios) {
    html += `<tr><td colspan="${ctrls.length + 1}" style="color:var(--text);font-weight:700;padding-top:14px">${b.titles[x]}</td></tr>`;
    for (const [key, label, dp] of rows) {
      const vals = ctrls.map((c) => (b.cells[x][c][key] ? b.cells[x][c][key].mean : null));
      const valid = vals.filter((v) => v !== null && !Number.isNaN(v));
      if (!valid.length) continue;
      const best = key === "throughput_horizon" ? Math.max(...valid) : Math.min(...valid);
      html += `<tr><td>${label}</td>${vals.map((v) => `<td class="${v === best ? "best" : ""}">${v === null || Number.isNaN(v) ? "-" : v.toFixed(dp)}</td>`).join("")}</tr>`;
    }
  }
  $("benchTable").innerHTML = html;
  $("benchNote").textContent = b.note || "";
}

/* ------------------------------------------------------------------ lab */
function showResult(b, o, label) {
  const ab = b.by_type.ambulance, ao = o.by_type.ambulance;
  const card = (v, l, good) => `<div class="kpi"><div class="v ${good ? "" : "neutral"}">${v}</div><div class="k">${l}</div></div>`;
  const sgn = (x) => `${x > 0 ? "+" : "−"}${Math.abs(x).toFixed(0)}%`;
  const d = pct(o.mean_delay, b.mean_delay), c = pct(o.co2_kg, b.co2_kg);
  $("labResult").innerHTML =
    card(sgn(d), `mean delay vs ${label}: ${b.mean_delay.toFixed(0)} s → ${o.mean_delay.toFixed(0)} s`, d < 0) +
    card(o.mean_stops.toFixed(2), `stops per vehicle (${label}: ${b.mean_stops.toFixed(2)})`, o.mean_stops < b.mean_stops) +
    card(sgn(c), `CO₂: ${b.co2_kg.toFixed(0)} → ${o.co2_kg.toFixed(0)} kg`, c < 0) +
    card(ab && ao ? `${ao.mean_delay.toFixed(0)} s` : "-", ab && ao ? `ambulance delay (${label}: ${ab.mean_delay.toFixed(0)} s)` : "no ambulance in this run",
      ab && ao && ao.mean_delay < ab.mean_delay);
}

function playInTwin(geom, base, ours, baseline, suffix) {
  state.baseline = baseline; $("baseName").textContent = CTRL_NAMES[baseline] + suffix;
  setData(geom, base, ours);
  state.t = 0; state.playing = true; $("btnPlay").textContent = "❚❚ Pause";
  $("twin").scrollIntoView({ behavior: "smooth" });
}

async function loadCondition() {
  const sid = $("labScenario").value, cond = $("labCond").value, st = $("labStatus");
  st.textContent = "Loading recorded runs…";
  try {
    const [geom, base, ours] = await Promise.all([getData(`network/${sid}.json.gz`),
      getData(`replays/${sid}__fixed__${cond}.json.gz`), getData(`replays/${sid}__chowkpilot__${cond}.json.gz`)]);
    $("selScenario").value = sid;
    const label = $("labCond").selectedOptions[0].textContent;
    playInTwin(geom, base, ours, "fixed", ` · ${label}`);
    showResult(base, ours, "fixed-time");
    st.textContent = `Playing: ${label}.`;
  } catch (err) { st.textContent = `Could not load this condition (${err.message}).`; }
}

async function runLab() {
  const btn = $("btnRun"), st = $("labStatus");
  const req = { scenario: $("labScenario").value, baseline: $("labBaseline").value, scale: Number($("labScale").value),
    profile: $("labProfile").value, ambulances: Number($("labAmb").value), noise: Number($("labNoise").value),
    ev_priority: $("labEV").checked, duration: 600, seed: Math.floor(Math.random() * 9000) + 100 };
  btn.disabled = true; st.innerHTML = `<span style="display:inline-flex;gap:8px;align-items:center"><span class="spinner"></span> Simulating 10 minutes of traffic twice (seed ${req.seed})…</span>`;
  const t0 = performance.now();
  try {
    const res = await getJSON("api/simulate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(req) });
    const geom = await getData(`network/${req.scenario}.json.gz`);
    $("selScenario").value = req.scenario;
    playInTwin(geom, res.baseline, res.chowkpilot, req.baseline, " · your run");
    showResult(res.baseline, res.chowkpilot, CTRL_NAMES[req.baseline].toLowerCase());
    st.textContent = `Done in ${((performance.now() - t0) / 1000).toFixed(0)} s. Now playing in the digital twin above.`;
  } catch (err) {
    st.textContent = `Simulation failed: ${err.message}`;
  } finally { btn.disabled = false; }
}

/* ------------------------------------------------------------------ init */
async function init() {
  views.base = new View($("cvBase"));
  views.ours = new View($("cvOurs"));
  attachPanZoom(views.base); attachPanZoom(views.ours);
  $("legend").innerHTML = TYPE_NAMES.map((n, i) => `<span><i style="background:${TYPE_COLORS[i]}"></i>${n}</span>`).join("") +
    `<span><i style="background:rgba(239,68,68,.75)"></i>stopped (tinted red)</span><span><i style="background:#22c55e;width:4px;height:12px"></i>signal state at stop line</span>`;
  $("btnPlay").onclick = () => { state.playing = !state.playing; if (state.t >= state.tmax) state.t = 0; $("btnPlay").textContent = state.playing ? "❚❚ Pause" : "▶ Play"; };
  $("selSpeed").onchange = (e) => (state.speed = Number(e.target.value));
  $("timeSlider").oninput = (e) => { state.t = Number(e.target.value); render(); };
  $("selJunction").onchange = explain;
  $("labScale").oninput = (e) => ($("oScale").textContent = Number(e.target.value).toFixed(2));
  $("labAmb").oninput = (e) => ($("oAmb").textContent = e.target.value);
  $("labNoise").oninput = (e) => ($("oNoise").textContent = `${Math.round(Number(e.target.value) * 100)}%`);
  $("btnRun").onclick = runLab;
  const list = await getData("scenarios.json");
  $("selScenario").innerHTML = list.filter((s) => s.replays.length).map((s) =>
    `<option value="${s.id}">${s.title}${s.split === "test" ? " · unseen" : " · training layout"}</option>`).join("");
  const withConds = list.filter((s) => Object.keys(s.conditions || {}).length);
  $("labScenario").innerHTML = withConds.map((s) => `<option value="${s.id}">${s.title}</option>`).join("");
  const fillConds = () => {
    const s = withConds.find((x) => x.id === $("labScenario").value);
    $("labCond").innerHTML = s ? Object.entries(s.conditions).map(([k, v]) => `<option value="${k}">${v.label}</option>`).join("") : "";
  };
  $("labScenario").onchange = fillConds; fillConds();
  $("btnLoadCond").onclick = loadCondition;
  const reload = () => loadScenario($("selScenario").value, $("selBaseline").value);
  $("selScenario").onchange = reload; $("selBaseline").onchange = reload;
  if (list.length) await loadScenario($("selScenario").value, "fixed");
  fetch("api/healthz").then((r) => (r.ok ? r.json() : null)).then((h) => { if (h && h.live) $("liveBox").classList.remove("hidden"); }).catch(() => {});
  requestAnimationFrame(tick);
  loadBenchmark();
}

init();
