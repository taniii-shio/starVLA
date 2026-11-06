"""collect_results.py の出力 (results.json と videos/) から、単独で開ける HTML レポートを作る。Mac で実行 (標準ライブラリのみ)。

  python3 scripts_local/make_report.py --export <展開先> --out ~/workspace/PAI/sv/papers/2610.19825/impl/report

<展開先> は export.tar を展開した場所 (results.json と videos/ がある)。
出力: <out>/index.html (データ埋め込み) と <out>/videos/ (成功動画のコピー)。
"""

import argparse
import json
import shutil
from pathlib import Path

# 学習ログの world_loss と比べる基準。wm_copy_baseline.py (n=16000, seed 0) の出力から転記した値
COPY_BASELINE = {"mean": 0.0882, "k": {"4": 0.0513, "8": 0.0807, "12": 0.1025, "16": 0.1185}}

TEMPLATE = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WM-Align 再現 結果</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a;
  --seq-0: #e1e0d9; --seq-1: #cde2fb; --seq-2: #9ec5f4; --seq-3: #5598e7; --seq-4: #256abf; --seq-5: #104281;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19;
    --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70;
    --seq-0: #2c2c2a; --seq-1: #184f95; --seq-2: #1c5cab; --seq-3: #2a78d6; --seq-4: #6da7ec; --seq-5: #b7d3f6;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19;
  --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70;
  --seq-0: #2c2c2a; --seq-1: #184f95; --seq-2: #1c5cab; --seq-3: #2a78d6; --seq-4: #6da7ec; --seq-5: #b7d3f6;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 14px/1.55 system-ui, -apple-system, "Hiragino Sans", "Noto Sans JP", sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 2px; }
.sub { color: var(--ink-2); margin: 0 0 16px; }
.card { background: var(--surface); border: 1px solid var(--grid); border-radius: 12px; padding: 16px; margin: 16px 0; }
.note { color: var(--ink-2); font-size: 13px; margin: 8px 0 0; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 8px 0; color: var(--ink-2); font-size: 13px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.key { width: 18px; height: 0; border-top: 2px solid; }
.key.dash { border-top-style: dashed; }
.controls { display: flex; gap: 8px; align-items: center; color: var(--ink-2); font-size: 13px; margin: 4px 0 8px; }
select { font: inherit; color: var(--ink); background: var(--surface); border: 1px solid var(--axis); border-radius: 6px; padding: 2px 6px; }
.chart { position: relative; width: 100%; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
svg text { fill: var(--muted); font-size: 11px; font-family: inherit; }
.tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--axis); border-radius: 8px; padding: 8px 10px; font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.12); display: none; z-index: 2; white-space: nowrap; }
.tip .t { color: var(--ink-2); margin-bottom: 4px; }
.tip .r { display: flex; align-items: center; gap: 8px; }
.tip b { font-size: 13px; color: var(--ink); }
.tip .n { color: var(--ink-2); }
table { border-collapse: collapse; width: 100%; font-size: 12px; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 8px; text-align: right; border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink-2); font-weight: 600; }
details { margin-top: 8px; } summary { cursor: pointer; color: var(--ink-2); font-size: 13px; }
.tasks { color: var(--ink-2); font-size: 12px; margin: 8px 0 0; padding-left: 20px; }
.videos { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; }
.videos figure { margin: 0; } .videos video { width: 100%; border-radius: 8px; background: #000; }
.videos figcaption { font-size: 12px; color: var(--ink-2); margin-top: 4px; word-break: break-all; }
.empty { color: var(--ink-2); padding: 12px 0; }
</style>
</head>
<body>
<main>
  <h1>WM-Align 再現: 学習と評価の結果</h1>
  <p class="sub">LIBERO / starVLA。3 アームを同じ 5,000 ステップ・実効バッチ 16 で学習し、step 5000 のチェックポイントを評価した。</p>
  <div class="legend" id="arms-legend"></div>

  <section class="card">
    <h2>アクション損失の推移</h2>
    <p class="sub">flow-matching の損失 (action_dit_loss)。各点は直近 W ステップの平均。</p>
    <div class="controls">平均の窓 <select id="win"><option>50</option><option selected>100</option><option>250</option></select> ステップ</div>
    <div class="chart" id="c-act"></div>
    <p class="note">窓の平均は、ログに残った step (10 ごと) から計算している。step 10 未満は除く。</p>
  </section>

  <section class="card">
    <h2>WM 損失 (world_loss) の推移: A と C</h2>
    <p class="sub">1 − cos(予測, DINOv3 特徴)。A は 4〜16 フレーム先、C は現在フレームを当てる。破線は「現在の特徴をそのままコピーした」ときの損失の平均 (A の目標に対して)。</p>
    <div class="chart" id="c-world"></div>
    <p class="note">基準値 __BASE_MEAN__ は wm_copy_baseline.py による平均 (k=4/8/12/16 ごとに __BASE_K__)。学習の進み方とは別に、目標の取り方の難しさの目安として見る。</p>
  </section>

  <section class="card">
    <h2>LIBERO 成功率</h2>
    <p class="sub">スイートごとの全試行の成功率。棒の上の値は成功率、ホバーで成功数 / 試行数。</p>
    <div class="chart" id="c-bar"></div>
    <details><summary>表で見る</summary><div id="t-suite"></div></details>
  </section>

  <section class="card">
    <h2>タスクごとの成功数</h2>
    <p class="sub">行がアーム、列がタスク。セルの濃さが成功率 (灰色は 0)。</p>
    <div id="heat"></div>
    <ol class="tasks" id="task-list"></ol>
  </section>

  <section class="card">
    <h2>成功した試行の動画</h2>
    <div id="videos"></div>
  </section>
</main>

<script>
const DATA = __DATA__;
const VIDEOS = __VIDEOS__;
const BASE = __BASE__;
const ARMS = [
  {id: "B", label: "B: QwenPI (ベースライン)", color: "var(--s1)", dash: ""},
  {id: "A", label: "A: QwenPIAlign (未来 t+4〜16)", color: "var(--s2)", dash: ""},
  {id: "C", label: "C: 対照 (現在フレーム)", color: "var(--s3)", dash: "6 4"},
];
const NS = "http://www.w3.org/2000/svg";
// 色は var(--...) で渡すので、属性ではなく style に入れる (属性の var() は環境によって効かない)
const el = (n, a = {}, parent) => {
  const e = document.createElementNS(NS, n);
  for (const k in a) { if (String(a[k]).includes("var(")) e.style.setProperty(k, a[k]); else e.setAttribute(k, a[k]); }
  if (parent) parent.appendChild(e);
  return e;
};

function legend(host, arms, extra) {
  host.textContent = "";
  for (const a of arms) {
    const s = document.createElement("span");
    const k = document.createElement("i"); k.className = "key" + (a.dash ? " dash" : ""); k.style.borderTopColor = a.color;
    s.appendChild(k); s.appendChild(document.createTextNode(a.label)); host.appendChild(s);
  }
  for (const x of extra || []) { const s = document.createElement("span"); s.textContent = x; host.appendChild(s); }
}
legend(document.getElementById("arms-legend"), ARMS);

function niceTicks(lo, hi, n) {
  const span = hi - lo, raw = span / n, p = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * p).find(s => s >= raw) || raw;
  const a = Math.floor(lo / step) * step, out = [];
  for (let v = a; v <= hi + step * 0.001; v += step) out.push(+v.toFixed(10));
  return out;
}

function windowMean(steps, vals, w) {
  const out = []; let acc = [], start = 0;
  for (let i = 0; i < steps.length; i++) {
    if (steps[i] < 10 || vals[i] == null) continue;
    acc.push(vals[i]);
    const bucket = Math.floor((steps[i] - 1) / w);
    if (i + 1 >= steps.length || Math.floor((steps[i + 1] - 1) / w) !== bucket) {
      out.push([(bucket + 1) * w, acc.reduce((a, b) => a + b, 0) / acc.length]);
      acc = [];
    }
  }
  return out;
}

/* 折れ線: series=[{arm, pts:[[x,y]]}], refs=[{y,label}] */
function lineChart(host, series, refs, fmt) {
  host.textContent = "";
  const W = 900, H = 320, m = {l: 52, r: 16, t: 12, b: 34};
  const svg = el("svg", {viewBox: `0 0 ${W} ${H}`}, host);
  const xs = series.flatMap(s => s.pts.map(p => p[0])), ys = series.flatMap(s => s.pts.map(p => p[1])).concat((refs || []).map(r => r.y));
  if (!xs.length) { host.innerHTML = '<div class="empty">データなし</div>'; return; }
  const x0 = 0, x1 = Math.max(...xs);
  let y0 = Math.min(...ys), y1 = Math.max(...ys); const pad = (y1 - y0) * 0.08 || 0.1; y0 = Math.max(0, y0 - pad); y1 += pad;
  const sx = x => m.l + (x - x0) / (x1 - x0) * (W - m.l - m.r), sy = y => H - m.b - (y - y0) / (y1 - y0) * (H - m.t - m.b);
  for (const v of niceTicks(y0, y1, 5)) {
    if (v < y0 || v > y1) continue;
    el("line", {x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), stroke: "var(--grid)", "stroke-width": 1}, svg);
    const t = el("text", {x: m.l - 8, y: sy(v) + 4, "text-anchor": "end"}, svg); t.textContent = fmt(v);
  }
  el("line", {x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "var(--axis)", "stroke-width": 1}, svg);
  for (const v of niceTicks(x0, x1, 5)) {
    const t = el("text", {x: sx(v), y: H - m.b + 18, "text-anchor": "middle"}, svg); t.textContent = v >= 1000 ? (v / 1000) + "K" : v;
  }
  const xl = el("text", {x: W - m.r, y: H - 2, "text-anchor": "end"}, svg); xl.textContent = "学習ステップ";
  for (const r of refs || []) {
    el("line", {x1: m.l, x2: W - m.r, y1: sy(r.y), y2: sy(r.y), stroke: "var(--ink-2)", "stroke-width": 1.5, "stroke-dasharray": "2 4"}, svg);
    const t = el("text", {x: W - m.r, y: sy(r.y) - 6, "text-anchor": "end"}, svg); t.textContent = r.label;
  }
  for (const s of series) {
    el("path", {d: s.pts.map((p, i) => (i ? "L" : "M") + sx(p[0]).toFixed(1) + " " + sy(p[1]).toFixed(1)).join(""), fill: "none",
      stroke: s.arm.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round", "stroke-dasharray": s.arm.dash}, svg);
    const last = s.pts[s.pts.length - 1];
    el("circle", {cx: sx(last[0]), cy: sy(last[1]), r: 4, fill: s.arm.color, stroke: "var(--surface)", "stroke-width": 2}, svg);
  }
  // ホバー: x の最近傍に縦線、全系列の値を 1 つの tooltip に出す
  const cross = el("line", {y1: m.t, y2: H - m.b, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden"}, svg);
  const tip = document.createElement("div"); tip.className = "tip"; host.appendChild(tip);
  const overlay = el("rect", {x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent"}, svg);
  const move = (ev) => {
    const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) / r.width * W;
    const xv = x0 + (px - m.l) / (W - m.l - m.r) * (x1 - x0);
    const base = series.reduce((b, s) => s.pts.length > b.length ? s.pts : b, []);
    const nearest = base.reduce((b, p) => Math.abs(p[0] - xv) < Math.abs(b[0] - xv) ? p : b, base[0])[0];
    cross.setAttribute("x1", sx(nearest)); cross.setAttribute("x2", sx(nearest)); cross.setAttribute("visibility", "visible");
    tip.textContent = "";
    const t = document.createElement("div"); t.className = "t"; t.textContent = "step " + nearest; tip.appendChild(t);
    for (const s of series) {
      const p = s.pts.find(q => q[0] === nearest); if (!p) continue;
      const row = document.createElement("div"); row.className = "r";
      const k = document.createElement("i"); k.className = "key" + (s.arm.dash ? " dash" : ""); k.style.borderTopColor = s.arm.color;
      const b = document.createElement("b"); b.textContent = fmt(p[1], true);
      const n = document.createElement("span"); n.className = "n"; n.textContent = s.arm.id;
      row.append(k, b, n); tip.appendChild(row);
    }
    tip.style.display = "block";
    const hr = host.getBoundingClientRect(), left = ev.clientX - hr.left + 14;
    tip.style.left = Math.min(left, hr.width - tip.offsetWidth - 4) + "px"; tip.style.top = (ev.clientY - hr.top + 14) + "px";
  };
  overlay.addEventListener("pointermove", move);
  overlay.addEventListener("pointerleave", () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); });
}

function drawTrain() {
  const w = +document.getElementById("win").value;
  const mk = (key, arms) => arms.filter(a => DATA.train[a.id] && DATA.train[a.id][key])
    .map(a => ({arm: a, pts: windowMean(DATA.train[a.id].steps, DATA.train[a.id][key], w)}));
  lineChart(document.getElementById("c-act"), mk("action_dit_loss", ARMS), [], (v, long) => v.toFixed(long ? 3 : 2));
  lineChart(document.getElementById("c-world"), mk("world_loss", ARMS.filter(a => a.id !== "B")),
    [{y: BASE.mean, label: "コピー基準 " + BASE.mean.toFixed(3)}], (v, long) => v.toFixed(long ? 4 : 2));
}
document.getElementById("win").addEventListener("change", drawTrain);
drawTrain();

/* 成功率: スイート × アームの縦棒 */
function suiteRows() {
  const suites = []; for (const a of ARMS) for (const s in (DATA.eval[a.id] || {})) if (!suites.includes(s)) suites.push(s);
  const order = ["libero_object", "libero_goal", "libero_spatial", "libero_10"];
  suites.sort((a, b) => order.indexOf(a) - order.indexOf(b));
  return suites;
}
function drawBars() {
  const host = document.getElementById("c-bar"); host.textContent = "";
  const suites = suiteRows();
  if (!suites.length) { host.innerHTML = '<div class="empty">評価結果なし</div>'; return; }
  const W = 900, H = 300, m = {l: 52, r: 16, t: 24, b: 40};
  const svg = el("svg", {viewBox: `0 0 ${W} ${H}`}, host);
  const rate = (a, s) => { const e = (DATA.eval[a.id] || {})[s]; return e && e.total ? e.success / e.total : null; };
  const all = suites.flatMap(s => ARMS.map(a => rate(a, s))).filter(v => v != null);
  const ymax = Math.max(0.1, Math.max(...all) * 1.25);
  const sy = v => H - m.b - v / ymax * (H - m.t - m.b);
  for (const v of niceTicks(0, ymax, 5)) {
    if (v > ymax) continue;
    el("line", {x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), stroke: "var(--grid)", "stroke-width": 1}, svg);
    const t = el("text", {x: m.l - 8, y: sy(v) + 4, "text-anchor": "end"}, svg); t.textContent = (v * 100).toFixed(0) + "%";
  }
  el("line", {x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "var(--axis)", "stroke-width": 1}, svg);
  const tip = document.createElement("div"); tip.className = "tip"; host.appendChild(tip);
  const band = (W - m.l - m.r) / suites.length, bw = Math.min(24, band / 5), gap = 2;
  suites.forEach((s, si) => {
    const cx = m.l + band * (si + 0.5), total = ARMS.length * bw + (ARMS.length - 1) * gap;
    const t = el("text", {x: cx, y: H - m.b + 20, "text-anchor": "middle"}, svg); t.textContent = s;
    ARMS.forEach((a, ai) => {
      const e = (DATA.eval[a.id] || {})[s]; if (!e) return;
      const r = e.success / e.total, x = cx - total / 2 + ai * (bw + gap), h = Math.max(r / ymax * (H - m.t - m.b), 0), y = H - m.b - h;
      const rr = Math.min(4, h);
      if (h > 0) el("path", {d: `M${x} ${H - m.b}V${y + rr}Q${x} ${y} ${x + rr} ${y}H${x + bw - rr}Q${x + bw} ${y} ${x + bw} ${y + rr}V${H - m.b}Z`, fill: a.color}, svg);
      const lab = el("text", {x: x + bw / 2, y: y - 6, "text-anchor": "middle"}, svg); lab.textContent = (r * 100).toFixed(1) + "%"; lab.style.fill = "var(--ink-2)";
      const hit = el("rect", {x: x - 1, y: m.t, width: bw + 2, height: H - m.t - m.b, fill: "transparent"}, svg);
      hit.addEventListener("pointermove", ev => {
        tip.textContent = "";
        const tt = document.createElement("div"); tt.className = "t"; tt.textContent = s + " / " + a.label; tip.appendChild(tt);
        const row = document.createElement("div"); row.className = "r";
        const b = document.createElement("b"); b.textContent = (r * 100).toFixed(1) + "%";
        const n = document.createElement("span"); n.className = "n"; n.textContent = e.success + " / " + e.total + " 試行";
        row.append(b, n); tip.appendChild(row); tip.style.display = "block";
        const hr = host.getBoundingClientRect();
        tip.style.left = Math.min(ev.clientX - hr.left + 14, hr.width - tip.offsetWidth - 4) + "px"; tip.style.top = (ev.clientY - hr.top + 14) + "px";
      });
      hit.addEventListener("pointerleave", () => { tip.style.display = "none"; });
    });
  });
  // 表
  const tb = document.getElementById("t-suite"); tb.textContent = "";
  const table = document.createElement("table");
  const head = table.insertRow(); ["スイート", ...ARMS.map(a => a.id + " 成功/試行"), ...ARMS.map(a => a.id + " 率")].forEach(h => { const th = document.createElement("th"); th.textContent = h; head.appendChild(th); });
  for (const s of suites) {
    const row = table.insertRow(); row.insertCell().textContent = s;
    for (const a of ARMS) { const e = (DATA.eval[a.id] || {})[s]; row.insertCell().textContent = e ? e.success + " / " + e.total : "-"; }
    for (const a of ARMS) { const r = rate(a, s); row.insertCell().textContent = r == null ? "-" : (r * 100).toFixed(1) + "%"; }
  }
  tb.appendChild(table);
}
drawBars();

/* タスクごと: 行=アーム、列=タスク (スイート別に 1 枚ずつ) */
function drawHeat() {
  const host = document.getElementById("heat"); host.textContent = "";
  const list = document.getElementById("task-list"); list.textContent = "";
  const suites = suiteRows();
  if (!suites.length) { host.innerHTML = '<div class="empty">評価結果なし</div>'; return; }
  const seq = ["var(--seq-1)", "var(--seq-2)", "var(--seq-3)", "var(--seq-4)", "var(--seq-5)"];
  for (const s of suites) {
    const tasks = [...new Set(ARMS.flatMap(a => Object.keys(((DATA.eval[a.id] || {})[s] || {tasks: {}}).tasks)))].sort();
    const h = document.createElement("h2"); h.style.margin = "12px 0 4px"; h.textContent = s; host.appendChild(h);
    const table = document.createElement("table");
    const head = table.insertRow(); head.insertCell().textContent = "";
    tasks.forEach((t, i) => { const th = document.createElement("th"); th.textContent = "T" + (i + 1); th.title = t.replace(/_/g, " "); head.appendChild(th); });
    for (const a of ARMS) {
      const row = table.insertRow(); const c0 = row.insertCell(); c0.textContent = a.id;
      for (const t of tasks) {
        const e = (((DATA.eval[a.id] || {})[s] || {tasks: {}}).tasks || {})[t]; const c = row.insertCell();
        if (!e) { c.textContent = "-"; continue; }
        const r = e.success / e.total;
        c.textContent = e.success; c.title = t.replace(/_/g, " ") + "\n" + a.id + ": " + e.success + " / " + e.total;
        c.style.background = r === 0 ? "var(--seq-0)" : seq[Math.min(4, Math.floor(r * 5))];
        c.style.color = r === 0 ? "var(--ink-2)" : (r >= 0.4 ? "#fff" : "#0b0b0b");
        c.style.textAlign = "center";
      }
    }
    host.appendChild(table);
    const li = document.createElement("li"); li.textContent = s + " " + tasks.map((t, i) => "T" + (i + 1) + "=" + t.replace(/_/g, " ")).join(" / "); list.appendChild(li);
  }
}
drawHeat();

/* 成功動画 */
(function () {
  const host = document.getElementById("videos"); host.textContent = "";
  if (!VIDEOS.length) { host.innerHTML = '<div class="empty">成功した試行がない (または動画が未取得)。</div>'; return; }
  const grid = document.createElement("div"); grid.className = "videos";
  for (const v of VIDEOS) {
    const f = document.createElement("figure"), vid = document.createElement("video");
    vid.controls = true; vid.preload = "none"; vid.muted = true; vid.src = v.path; vid.setAttribute("playsinline", "");
    const cap = document.createElement("figcaption"); cap.textContent = v.arm + " / " + v.suite + " / " + v.name.replace(/^rollout_/, "").replace(/_success\.mp4$/, "");
    f.append(vid, cap); grid.appendChild(f);
  }
  host.appendChild(grid);
})();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="export.tar を展開した場所 (results.json, videos/)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    export, out = Path(args.export).expanduser(), Path(args.out).expanduser()
    data = json.loads((export / "results.json").read_text())
    out.mkdir(parents=True, exist_ok=True)

    videos = []
    src_videos = export / "videos"
    if src_videos.is_dir():
        dst = out / "videos"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src_videos, dst)
        for p in sorted(dst.rglob("*.mp4")):
            rel = p.relative_to(out)
            videos.append({"arm": rel.parts[1], "suite": rel.parts[2], "name": p.name, "path": rel.as_posix()})

    def js(obj):
        return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")

    html = (
        TEMPLATE.replace("__DATA__", js(data))
        .replace("__VIDEOS__", js(videos))
        .replace("__BASE__", js(COPY_BASELINE))
        .replace("__BASE_MEAN__", f"{COPY_BASELINE['mean']:.3f}")
        .replace("__BASE_K__", " / ".join(f"{v:.3f}" for v in COPY_BASELINE["k"].values()))
    )
    (out / "index.html").write_text(html)
    print(f"wrote {out / 'index.html'} (動画 {len(videos)} 本)")


if __name__ == "__main__":
    main()
