// AdapFuse-UAV live page: play the video, stream frames to /ws/{id}, draw results.
"use strict";

const $ = (id) => document.getElementById(id);
const video = $("video"), overlay = $("overlay"), stage = $("stage");
const ctx = overlay.getContext("2d");
const grab = document.createElement("canvas");
const gctx = grab.getContext("2d");

const VEHICLES = ["bicycle", "car", "motorcycle", "bus", "truck", "boat"];
const COLORS = { fire: "#ff5a3c", smoke: "#b8bec7", person: "#3ecf6e",
  ...Object.fromEntries(VEHICLES.map(v => [v, "#36c5f0"])) };
const MOD_COLORS = { rgb: "#4f8cff", thermal: "#ff9f40", audio: "#c678dd" };
const SCENE_COLORS = { "normal": "#3ecf6e", "fire / smoke": "#ff5a3c",
  "collapse / flood": "#ffb020", "other disaster": "#e056fd" };
const EVENT_COLORS = { "normal": "#3ecf6e", "fire": "#ff5a3c", "smoke": "#b8bec7", "explosion": "#ff8a3c",
  "collapsed building": "#ffb020", "flood": "#3aa0ff", "landslide": "#c08a5a", "traffic accident": "#e056fd" };
const ZONE_COLORS = { "collapsed building": "#ffb020", "flood water": "#3aa0ff", "landslide": "#c08a5a",
  "accident": "#e056fd" };
const EVENT_NAMES = Object.keys(EVENT_COLORS);
const HAZARD_SOUND = "#ffb020";
const SEND_WIDTH = 640;       // frames are downscaled before sending
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let meta = null, ws = null, inflight = false, seq = 0, latest = null;
let lastScene = null, lastVictim = null, lastTag = null, lastAgree = null, lastZones = new Set();
const doneTimes = [];

// ── status ────────────────────────────────────────────────────────────────
function setStatus(text, state) {
  $("statusText").textContent = text;
  $("statusDot").className = "dot" + (state ? " " + state : "");
}

fetch("/api/health").then(r => r.json()).then(h => {
  setStatus(h.ok ? "models ready · upload a video" : "loading models…", h.ok ? "ok" : "");
  $("device").textContent = h.device || "—";
}).catch(() => setStatus("server not reachable", "err"));

// ── samples ───────────────────────────────────────────────────────────────
fetch("/api/samples").then(r => r.json()).then(list => {
  for (const name of list) {
    const b = document.createElement("button");
    b.className = "btn"; b.textContent = name.replace(/\.mp4$/, "");
    b.onclick = () => openVideo(fetch(`/api/samples/${encodeURIComponent(name)}`, { method: "POST" }));
    $("samples").appendChild(b);
  }
});

// ── upload ────────────────────────────────────────────────────────────────
$("file").addEventListener("change", (e) => e.target.files[0] && upload(e.target.files[0]));
stage.addEventListener("dragover", (e) => { e.preventDefault(); stage.classList.add("drag"); });
stage.addEventListener("dragleave", () => stage.classList.remove("drag"));
stage.addEventListener("drop", (e) => {
  e.preventDefault(); stage.classList.remove("drag");
  if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]);
});

function upload(file) {
  const fd = new FormData(); fd.append("file", file);
  setStatus(`uploading ${file.name}…`);
  openVideo(fetch("/api/upload", { method: "POST", body: fd }));
}

async function openVideo(req) {
  let res;
  try { res = await req; } catch { return setStatus("upload failed", "err"); }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    return setStatus(err.detail || `upload failed (${res.status})`, "err");
  }
  meta = await res.json();
  resetSession();
  $("empty").hidden = true;
  video.src = meta.url;
  $("modality").value = "auto";
  showModality(meta.modality.modality, meta.modality.reason);
  $("device").textContent = meta.device;
  setStatus(`${meta.name} · ${meta.info.width}×${meta.info.height} · ${meta.info.duration.toFixed(1)} s · ` +
            (meta.has_audio ? "audio track found" : "no audio track"), "ok");
  connect();
  video.play().catch(() => {});
}

function showModality(mod, reason) {
  const b = $("modBadge");
  b.hidden = false;
  b.textContent = `${mod.toUpperCase()} INPUT` + (reason ? ` · ${reason}` : "");
  b.style.color = MOD_COLORS[mod];
  $("detNote").textContent = mod === "thermal" ? "thermal input: fire/smoke boxes less reliable, no hazard zones" : "";
}

function resetSession() {
  if (ws) { ws.onclose = null; ws.close(); }
  inflight = false; latest = null; lastScene = lastVictim = lastTag = lastAgree = null; lastZones = new Set();
  $("disagree").hidden = true;
  doneTimes.length = 0; $("log").innerHTML = "";
  clearOverlay();
}

// ── websocket ─────────────────────────────────────────────────────────────
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/${meta.id}`);
  ws.onopen = () => { sendConfig(); pump(); };
  ws.onmessage = (ev) => {
    inflight = false;
    const r = JSON.parse(ev.data);
    if (r.error) return setStatus(r.error, "err");
    if (r.skip) return pump();
    latest = r;
    doneTimes.push(performance.now());
    while (doneTimes.length > 20) doneTimes.shift();
    updatePanel(r);
    pump();
  };
  ws.onclose = () => { inflight = false; setStatus("connection closed — re-select the video", "err"); };
}

function sendConfig() {
  if (!ws || ws.readyState !== 1) return;
  const m = $("modality").value;
  const mod = m === "auto" ? meta.modality.modality : m;
  ws.send(JSON.stringify({ type: "config", modality: mod,
    det_conf: +$("detConf").value, person_conf: +$("perConf").value }));
  showModality(mod, m === "auto" ? meta.modality.reason : "manual");
}
$("modality").addEventListener("change", () => { sendConfig(); sendFrame(); });
for (const [id, out] of [["detConf", "detConfV"], ["perConf", "perConfV"]]) {
  $(id).addEventListener("input", () => { $(out).textContent = (+$(id).value).toFixed(2); });
  $(id).addEventListener("change", () => { sendConfig(); sendFrame(); });
}

function sendFrame() {
  // mid-seek the canvas would still hold the old frame under the new currentTime
  if (!ws || ws.readyState !== 1 || inflight || !video.videoWidth || video.seeking) return false;
  const s = Math.min(1, SEND_WIDTH / video.videoWidth);
  grab.width = Math.round(video.videoWidth * s);
  grab.height = Math.round(video.videoHeight * s);
  gctx.drawImage(video, 0, 0, grab.width, grab.height);
  inflight = true;
  ws.send(JSON.stringify({ t: video.currentTime, seq: ++seq,
                           image: grab.toDataURL("image/jpeg", 0.8) }));
  return true;
}

// Analyze continuously while playing; analyze once after pause/seek.
// Playing: analyze continuously. Paused: re-analyze until the result matches the frame on screen
// (a seek can land while an earlier request is still in flight).
function pump() {
  if (!video.paused && !video.ended) sendFrame();
  else if (!latest || Math.abs(latest.t - video.currentTime) > 0.02) sendFrame();
}
video.addEventListener("play", pump);
video.addEventListener("seeking", () => { latest = null; });   // drop the old moment's overlay
video.addEventListener("seeked", pump);
video.addEventListener("loadeddata", () => sendFrame());
video.addEventListener("error", () => setStatus("browser cannot play this codec — try MP4 (H.264)", "err"));

// Persistence (2 of 3 analyzed frames) suppresses flicker during playback. A paused frame is
// analyzed once after the seek, so show that frame's raw detections instead.
const shownBoxes = (r) => (video.paused ? r.raw_boxes || r.boxes : r.boxes);

// ── overlay ───────────────────────────────────────────────────────────────
function contentRect() {
  // where the video pixels actually sit inside the element (object-fit: contain)
  const W = overlay.clientWidth, H = overlay.clientHeight;
  const vw = video.videoWidth || 16, vh = video.videoHeight || 9;
  const s = Math.min(W / vw, H / vh);
  return { x: (W - vw * s) / 2, y: (H - vh * s) / 2, w: vw * s, h: vh * s };
}

function clearOverlay() {
  const dpr = window.devicePixelRatio || 1;
  overlay.width = overlay.clientWidth * dpr;
  overlay.height = overlay.clientHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, overlay.clientWidth, overlay.clientHeight);
}

const heat = document.createElement("canvas");
const hctx = heat.getContext("2d");

function hexRgb(h) { const n = parseInt(h.slice(1), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }

function drawZones(r, res) {
  const g = res.zone_grid;
  if (g && res.zones && res.zones.length) {
    // soft heat layer: one pixel per patch, scaled up with smoothing
    heat.width = g.w; heat.height = g.h;
    const img = hctx.createImageData(g.w, g.h);
    for (let i = 0; i < g.cls.length; i++) {
      if (g.cls[i] < 0) continue;
      const zone = res.zones.find(z => z.event === EVENT_NAMES[g.cls[i]]);
      if (!zone) continue;
      const [cr, cg, cb] = hexRgb(ZONE_COLORS[zone.cls] || "#ffffff");
      img.data.set([cr, cg, cb, Math.round(Math.min(1, g.p[i]) * 85)], i * 4);
    }
    hctx.putImageData(img, 0, 0);
    ctx.save(); ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
    ctx.drawImage(heat, r.x, r.y, r.w, r.h); ctx.restore();
  }
  ctx.font = "600 12px system-ui, sans-serif";
  for (const z of res.zones || []) {
    const [x1, y1, x2, y2] = z.box;
    const x = r.x + x1 * r.w, y = r.y + y1 * r.h, w = (x2 - x1) * r.w, h = (y2 - y1) * r.h;
    const c = ZONE_COLORS[z.cls] || "#fff";
    ctx.setLineDash([7, 5]); ctx.lineWidth = 2; ctx.strokeStyle = c; ctx.strokeRect(x, y, w, h); ctx.setLineDash([]);
    const label = `${z.cls} ${z.conf.toFixed(2)}`;
    const tw = ctx.measureText(label).width + 8;
    const ly = y + h - 18 > y ? y + h - 18 : y;
    ctx.fillStyle = c; ctx.fillRect(x, ly, tw, 18);
    ctx.fillStyle = "#000"; ctx.fillText(label, x + 4, ly + 13);
  }
}

function drawOverlay() {
  clearOverlay();
  if (latest && video.videoWidth) {
    const r = contentRect();
    drawZones(r, latest);
    ctx.lineWidth = 2;
    ctx.font = "600 12px system-ui, sans-serif";
    for (const b of shownBoxes(latest)) {
      const [x1, y1, x2, y2] = b.box;
      const x = r.x + x1 * r.w, y = r.y + y1 * r.h, w = (x2 - x1) * r.w, h = (y2 - y1) * r.h;
      const c = COLORS[b.cls] || "#ffff00";
      ctx.strokeStyle = c; ctx.strokeRect(x, y, w, h);
      const label = `${b.cls} ${b.conf.toFixed(2)}`;
      const tw = ctx.measureText(label).width + 8;
      ctx.fillStyle = c; ctx.fillRect(x, Math.max(0, y - 18), tw, 18);
      ctx.fillStyle = b.cls === "smoke" ? "#000" : "#fff";
      ctx.fillText(label, x + 4, Math.max(13, y - 5));
    }
    // top-right scene banner
    const ev = latest.event || { label: latest.disaster.label, conf: latest.disaster.conf };
    const text = `${ev.label.toUpperCase()}  ${(ev.conf * 100).toFixed(0)}%`;
    ctx.font = "700 15px system-ui, sans-serif";
    const tw = ctx.measureText(text).width + 20;
    ctx.fillStyle = "rgba(0,0,0,.65)";
    ctx.fillRect(r.x + r.w - tw - 10, r.y + 10, tw, 28);
    ctx.fillStyle = EVENT_COLORS[ev.label] || "#fff";
    ctx.fillText(text, r.x + r.w - tw, r.y + 29);
    // sound banner under it
    const a = latest.audio || {};
    if (a.status === "ok" && a.headline && a.headline.name !== "background noise") {
      const st = `♪ ${a.headline.name}`;
      ctx.font = "600 13px system-ui, sans-serif";
      const sw = ctx.measureText(st).width + 20;
      ctx.fillStyle = "rgba(0,0,0,.65)";
      ctx.fillRect(r.x + r.w - sw - 10, r.y + 42, sw, 24);
      ctx.fillStyle = a.headline.hazard ? HAZARD_SOUND : "#e6edf3";
      ctx.fillText(st, r.x + r.w - sw, r.y + 59);
    }
  }
  requestAnimationFrame(drawOverlay);
}
requestAnimationFrame(drawOverlay);

// ── side panel ────────────────────────────────────────────────────────────
function bars(el, items) {
  if (el.children.length !== items.length || !el.querySelector(".tick")) {
    el.innerHTML = items.map(() =>
      `<div class="bar"><span class="name"></span><div class="track"><div class="fill"></div><div class="tick" hidden></div></div><span class="val"></span></div>`
    ).join("");
  }
  items.forEach((it, i) => {
    const row = el.children[i];
    row.className = "bar" + (it.on ? " on" : "");
    row.querySelector(".name").textContent = it.name;
    row.querySelector(".name").title = it.name;
    row.querySelector(".fill").style.width = `${Math.max(0, Math.min(1, it.value)) * 100}%`;
    row.querySelector(".fill").style.background = it.color;
    const tick = row.querySelector(".tick");
    tick.hidden = it.tick === undefined;
    if (it.tick !== undefined) tick.style.left = `${Math.min(1, it.tick) * 100}%`;
    row.querySelector(".val").textContent = it.value.toFixed(2);
  });
}

function pct(x) { return `${(x * 100).toFixed(0)}%`; }

function updatePanel(r) {
  const d = r.disaster, ev = r.event || { label: d.label, group: d.label, probs: {} };
  const sl = $("sceneLabel");
  sl.textContent = ev.label;
  sl.className = "scene-label";
  sl.style.color = EVENT_COLORS[ev.label] || "";
  $("sceneSub").textContent = ev.label === "normal" ? "no hazard in view" : ev.group;
  $("sceneChip").textContent = `conf ${pct(d.conf)}`;
  bars($("sceneBars"), Object.entries(d.probs).map(([k, v]) =>
    ({ name: k, value: v, color: SCENE_COLORS[k] })));
  bars($("eventBars"), Object.entries(ev.probs).filter(([k]) => k !== "normal")
    .sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, v]) => ({ name: k, value: v, color: EVENT_COLORS[k] })));
  $("victim").textContent = `${r.victim.label} (${pct(r.victim.conf)})`;
  $("nuisance").textContent = r.nuisance ? `${r.nuisance.label} (${pct(r.nuisance.conf)})` : "—";

  bars($("relBars"), ["rgb", "thermal", "audio"].map(k => ({
    name: k + (r.has[k] ? "" : " (absent)"), value: r.reliability[k], color: MOD_COLORS[k] })));

  const a = r.audio || {};
  $("audioChip").textContent = a.status === "ok" ? `${a.window_s.toFixed(0)} s window @ ${a.window_end.toFixed(1)} s`
    : a.status === "silent" ? "silent" : "no audio track";
  const snd = $("soundLabel");
  if (a.status === "ok") {
    const h = a.headline;
    snd.textContent = h.name;
    snd.className = "sound-label " + (h.hazard ? "hazard" : h.name === "background noise" ? "" : "context");
    const vc = a.victim_cue;
    $("audioVictim").textContent = vc && vc.active ? `distress voices (${vc.score.toFixed(2)})` : "none heard";
    $("audioTop").textContent = a.top_tags && a.top_tags[0] ? `${a.top_tags[0].name} ${a.top_tags[0].score.toFixed(2)}` : "—";
    $("audioTop").title = (a.top_tags || []).map(t => `${t.name} ${t.score.toFixed(2)}`).join("\n");
    bars($("tagBars"), a.events.slice(0, 5).map(e => ({ name: e.name, value: e.score, tick: e.thr, on: e.active,
      color: e.hazard ? (e.active ? HAZARD_SOUND : "#7a5a1e") : (e.active ? MOD_COLORS.audio : "#5b3f66") })));
  } else {
    snd.textContent = a.status === "silent" ? "silence" : "—"; snd.className = "sound-label";
    $("audioVictim").textContent = "—"; $("audioTop").textContent = "—";
    $("tagBars").innerHTML = `<div class="muted small">${a.status === "silent" ? "audio is silent here" : "no audio in this video"}</div>`;
  }

  const n = { fire: 0, smoke: 0, person: 0, vehicle: 0 };
  for (const b of shownBoxes(r)) {
    if (VEHICLES.includes(b.cls)) n.vehicle++; else n[b.cls] = (n[b.cls] || 0) + 1;
  }
  $("nFire").textContent = n.fire; $("nSmoke").textContent = n.smoke;
  $("nPerson").textContent = n.person; $("nVehicle").textContent = n.vehicle;
  const zc = {};
  for (const z of r.zones || []) zc[z.cls] = (zc[z.cls] || 0) + 1;
  $("zoneList").innerHTML = Object.keys(zc).length
    ? Object.entries(zc).map(([k, c]) => `<span class="zone" style="color:${ZONE_COLORS[k]};border-color:${ZONE_COLORS[k]}">${esc(k)}${c > 1 ? " ×" + c : ""}</span>`).join("")
    : `<span class="muted small">none</span>`;

  $("ms").textContent = `${r.timing_ms.total.toFixed(0)} ms`;
  if (doneTimes.length > 1) {
    const span = (doneTimes[doneTimes.length - 1] - doneTimes[0]) / 1000;
    $("fps").textContent = `${((doneTimes.length - 1) / span).toFixed(1)} /s`;
  }
  $("lag").textContent = `${Math.max(0, video.currentTime - r.t).toFixed(2)} s`;

  // event log: changes of scene event, victim flag, hazard zones, hazard sound
  if (ev.label !== lastScene) {
    logEvent(r.t, `Scene → <b style="color:${EVENT_COLORS[ev.label] || "#fff"}">${esc(ev.label)}</b> (${pct(d.conf)})`);
    lastScene = ev.label;
  }
  if (r.victim.index !== lastVictim) { if (lastVictim !== null || r.victim.index === 1) logEvent(r.t, `Victim → <b>${r.victim.label}</b>`); lastVictim = r.victim.index; }
  const zs = new Set(Object.keys(zc));
  for (const k of zs) if (!lastZones.has(k)) logEvent(r.t, `Zone → <b style="color:${ZONE_COLORS[k]}">${esc(k)}</b>`);
  lastZones = zs;
  const tag = a.status === "ok" && a.headline.hazard ? a.headline.name : null;
  if (tag && tag !== lastTag) logEvent(r.t, `Sound → <b style="color:${HAZARD_SOUND}">${esc(tag)}</b>`);
  lastTag = tag;

  const ag = r.agreement || { status: "agree", reason: "" };
  const badge = $("disagree");
  if (ag.status === "agree") badge.hidden = true;
  else { badge.hidden = false; badge.textContent = `⚠ models disagree — ${ag.reason}`; }
  if (ag.status !== lastAgree) {
    if (lastAgree !== null || ag.status !== "agree")
      logEvent(r.t, ag.status === "agree" ? "Scene and detector agree again"
        : `<b style="color:var(--warn)">Models disagree</b> — ${esc(ag.reason)}`);
    lastAgree = ag.status;
  }
}

function logEvent(t, html) {
  const li = document.createElement("li");
  li.innerHTML = `<span class="ts">${t.toFixed(1)} s</span><span>${html}</span>`;
  li.onclick = () => { video.currentTime = t; };
  $("log").prepend(li);
}
