// ==========================================================
//  Orbite d'Arenstorf — problème à trois corps restreint circulaire
//  Repère tournant, unités normalisées (distance Terre-Lune = 1,
//  masse totale = 1, vitesse angulaire = 1).
// ==========================================================
(function () {
  const MU = 0.012277471;      // masse réduite de la Lune
  const MUP = 1 - MU;          // masse réduite de la Terre
  const T = 17.0652165601579625588917206249;   // période de l'orbite
  const S0 = [0.994, 0, 0, -2.00158510637908252240537862224]; // [x, y, x', y']

  // Second membre du système du premier ordre  s' = f(s)
  function f(s) {
    const x = s[0], y = s[1], u = s[2], v = s[3];
    const r1 = Math.hypot(x + MU, y), r2 = Math.hypot(x - MUP, y);
    const d1 = r1 * r1 * r1, d2 = r2 * r2 * r2;
    return [
      u,
      v,
      x + 2 * v - MUP * (x + MU) / d1 - MU * (x - MUP) / d2,
      y - 2 * u - MUP * y / d1 - MU * y / d2
    ];
  }

  function add(s, k, h) {
    return [s[0] + h * k[0], s[1] + h * k[1], s[2] + h * k[2], s[3] + h * k[3]];
  }

  function eulerStep(s, h) {
    return add(s, f(s), h);
  }

  function rk4Step(s, h) {
    const k1 = f(s);
    const k2 = f(add(s, k1, h / 2));
    const k3 = f(add(s, k2, h / 2));
    const k4 = f(add(s, k3, h));
    return [
      s[0] + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
      s[1] + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
      s[2] + h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
      s[3] + h / 6 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
    ];
  }

  // Constante de Jacobi : intégrale première du mouvement dans le repère tournant
  function jacobi(s) {
    const x = s[0], y = s[1], u = s[2], v = s[3];
    const r1 = Math.hypot(x + MU, y), r2 = Math.hypot(x - MUP, y);
    return x * x + y * y + 2 * MUP / r1 + 2 * MU / r2 - (u * u + v * v);
  }

  // Intègre sur une période avec N pas fixes. Renvoie les positions et l'état final.
  function integrate(step, N) {
    const h = T / N;
    const pts = new Float64Array(2 * (N + 1));
    let s = S0.slice();
    pts[0] = s[0]; pts[1] = s[1];
    let n = N;
    for (let i = 1; i <= N; i++) {
      s = step(s, h);
      pts[2 * i] = s[0]; pts[2 * i + 1] = s[1];
      if (!isFinite(s[0]) || Math.abs(s[0]) > 50 || Math.abs(s[1]) > 50) { n = i; break; }
    }
    return { pts: pts, count: n + 1, final: s, escaped: n < N };
  }

  // ---------- Dessin ----------
  const VIEW = { xmin: -1.4, xmax: 1.2, ymin: -1.22, ymax: 1.22 };

  function makeMapper(w, h) {
    const sx = w / (VIEW.xmax - VIEW.xmin), sy = h / (VIEW.ymax - VIEW.ymin);
    const k = Math.min(sx, sy);
    const cx = w / 2 - k * (VIEW.xmin + VIEW.xmax) / 2;
    const cy = h / 2 + k * (VIEW.ymin + VIEW.ymax) / 2;
    return function (x, y) { return [cx + k * x, cy - k * y]; };
  }

  function drawBackground(ctx, w, h, P) {
    ctx.clearRect(0, 0, w, h);
    const line = themeColor("--line"), muted = themeColor("--muted"), fg = themeColor("--fg");
    // axes
    ctx.strokeStyle = line; ctx.lineWidth = 1;
    ctx.beginPath();
    let a = P(VIEW.xmin, 0), b = P(VIEW.xmax, 0); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
    a = P(0, VIEW.ymin); b = P(0, VIEW.ymax); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
    ctx.stroke();
    // Terre et Lune
    const e = P(-MU, 0), m = P(MUP, 0);
    ctx.fillStyle = themeColor("--accent");
    ctx.beginPath(); ctx.arc(e[0], e[1], 7, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = muted;
    ctx.beginPath(); ctx.arc(m[0], m[1], 4, 0, 2 * Math.PI); ctx.fill();
    ctx.font = "12px Inter, system-ui, sans-serif";
    ctx.fillStyle = fg;
    ctx.fillText("Terre", e[0] - 16, e[1] + 22);
    ctx.fillText("Lune", m[0] - 12, m[1] + 20);
  }

  function drawPath(ctx, P, pts, count, color, width) {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = "round";
    ctx.beginPath();
    let p = P(pts[0], pts[1]); ctx.moveTo(p[0], p[1]);
    for (let i = 1; i < count; i++) { p = P(pts[2 * i], pts[2 * i + 1]); ctx.lineTo(p[0], p[1]); }
    ctx.stroke();
  }

  // ---------- Figure 1 : animation de l'orbite ----------
  function figureOrbite() {
    const canvas = document.getElementById("fig-orbite");
    if (!canvas) return;
    const N = 20000;
    const traj = integrate(rk4Step, N);
    const DURATION = 9000; // ms pour une période
    let t0 = performance.now();

    function frame(now) {
      const { ctx, w, h } = setupCanvas(canvas);
      const P = makeMapper(w, h);
      drawBackground(ctx, w, h, P);
      const frac = ((now - t0) % DURATION) / DURATION;
      const k = Math.max(2, Math.floor(frac * N));
      // orbite complète en pointillés discrets
      ctx.globalAlpha = 0.25;
      drawPath(ctx, P, traj.pts, traj.count, themeColor("--accent-2"), 1);
      ctx.globalAlpha = 1;
      drawPath(ctx, P, traj.pts, k, themeColor("--accent-2"), 2);
      const p = P(traj.pts[2 * (k - 1)], traj.pts[2 * (k - 1) + 1]);
      ctx.fillStyle = themeColor("--fg");
      ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, 2 * Math.PI); ctx.fill();
      ctx.font = "12px JetBrains Mono, monospace";
      ctx.fillStyle = themeColor("--muted");
      ctx.fillText("t = " + (frac * T).toFixed(2) + " / " + T.toFixed(2), 12, 20);
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
    const btn = document.getElementById("replay-orbite");
    if (btn) btn.addEventListener("click", function () { t0 = performance.now(); });
  }

  // ---------- Figure 2 : Euler contre RK4 ----------
  function figureMethodes() {
    const canvas = document.getElementById("fig-methodes");
    const slider = document.getElementById("pas-slider");
    const label = document.getElementById("pas-label");
    const out = document.getElementById("methodes-readout");
    if (!canvas || !slider) return;

    let pending = false;
    function render() {
      pending = false;
      const N = Math.round(Math.pow(10, parseFloat(slider.value)));
      label.textContent = N.toLocaleString("fr-FR");
      const e = integrate(eulerStep, N), r = integrate(rk4Step, N);
      const { ctx, w, h } = setupCanvas(canvas);
      const P = makeMapper(w, h);
      drawBackground(ctx, w, h, P);
      drawPath(ctx, P, e.pts, e.count, themeColor("--muted"), 1.3);
      drawPath(ctx, P, r.pts, r.count, themeColor("--accent-2"), 1.8);

      const C0 = jacobi(S0);
      function report(name, res) {
        if (res.escaped) return name + " : le satellite est éjecté du système";
        const err = Math.hypot(res.final[0] - S0[0], res.final[1] - S0[1]);
        const dC = Math.abs(jacobi(res.final) - C0) / Math.abs(C0);
        return name + " : écart au point de départ = " + err.toExponential(2) +
               ", dérive relative de C_J = " + dC.toExponential(2);
      }
      out.innerHTML = report("Euler", e) + "<br>" + report("RK4  ", r);
    }
    function schedule() { if (!pending) { pending = true; requestAnimationFrame(render); } }
    slider.addEventListener("input", schedule);
    window.addEventListener("resize", schedule);
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", schedule);
    schedule();
  }

  document.addEventListener("DOMContentLoaded", function () {
    figureOrbite();
    figureMethodes();
  });
})();
