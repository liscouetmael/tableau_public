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

  // ---------- Figure 3 : sensibilité aux conditions initiales ----------
  function figurePerturbation() {
    const canvas = document.getElementById("fig-perturbe");
    const choixDelta = document.getElementById("perturbe-delta");
    const choixTours = document.getElementById("perturbe-tours");
    const out = document.getElementById("perturbe-readout");
    if (!canvas || !choixDelta) return;
    const PAS = 40000;   // pas RK4 par période : l'erreur numérique reste très en dessous des écarts étudiés

    function trajectoire(s0, tours) {
      const N = PAS * tours, h = T / N;
      const pts = new Float64Array(2 * (N + 1)), fins = [];
      let s = s0.slice(), n = N;
      pts[0] = s[0]; pts[1] = s[1];
      for (let i = 1; i <= N; i++) {
        s = rk4Step(s, h);
        pts[2 * i] = s[0]; pts[2 * i + 1] = s[1];
        if (i % PAS === 0) fins.push(s.slice());
        if (!isFinite(s[0]) || Math.abs(s[0]) > 50 || Math.abs(s[1]) > 50) { n = i; break; }
      }
      return { pts: pts, count: n + 1, fins: fins };
    }

    let pending = false;
    function render() {
      pending = false;
      const d0 = +choixDelta.value, tours = +choixTours.value;
      const s1 = S0.slice(); s1[0] += d0;
      const a = trajectoire(S0, tours), b = trajectoire(s1, tours);
      const { ctx, w, h } = setupCanvas(canvas);
      const P = makeMapper(w, h);
      drawBackground(ctx, w, h, P);
      drawPath(ctx, P, b.pts, b.count, themeColor("--accent"), 1.4);
      drawPath(ctx, P, a.pts, a.count, themeColor("--accent-2"), 1.8);
      const lignes = [];
      for (let k = 0; k < Math.min(a.fins.length, b.fins.length); k++) {
        const ea = a.fins[k], eb = b.fins[k];
        const d = Math.hypot(ea[0] - eb[0], ea[1] - eb[1], ea[2] - eb[2], ea[3] - eb[3]);
        lignes.push("après " + (k + 1) + " tour" + (k ? "s" : "") + " : " + d.toExponential(1).replace(".", ","));
      }
      if (b.fins.length < tours) lignes.push("la trajectoire perturbée est éjectée");
      out.textContent = "Écart initial " + d0.toExponential(0).replace(".", ",") + " · " + lignes.join(" · ");
    }
    function schedule() { if (!pending) { pending = true; requestAnimationFrame(render); } }
    choixDelta.addEventListener("change", schedule);
    choixTours.addEventListener("change", schedule);
    window.addEventListener("resize", schedule);
    schedule();
  }

  // ---------- Figure 4 : section de Poincaré y = 0 (passages vers le bas) ----------
  // Intégration par Dormand–Prince 5(4) à pas adaptatif ; au moment où y change de signe,
  // un pas de Hénon (on prend y comme variable d'intégration) tombe exactement sur y = 0.
  const DP_A = [[], [1 / 5], [3 / 40, 9 / 40], [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]];
  const DP_B5 = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0];
  const DP_B4 = [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40];
  const R_TERRE = 6371 / 384400, R_LUNE = 1737 / 384400;

  function omega(x, y) {
    return (x * x + y * y) / 2 + MUP / Math.hypot(x + MU, y) + MU / Math.hypot(x - MUP, y);
  }

  function pasHenon(s) {
    function g(z) { const d = f(z); return [d[0] / d[1], 1, d[2] / d[1], d[3] / d[1]]; }
    const h = -s[1];
    const k1 = g(s), k2 = g(add(s, k1, h / 2)), k3 = g(add(s, k2, h / 2)), k4 = g(add(s, k3, h));
    return [0, 1, 2, 3].map((i) => s[i] + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
  }

  function Trajet(s, couleur, taille) {
    this.s = s.slice(); this.t = 0; this.h = 1e-3; this.couleur = couleur; this.taille = taille;
    this.pts = [s[0], s[2]]; this.etat = "en cours";
  }
  // Avance de nbPas pas d'intégration au plus ; s'arrête sur collision, éjection ou assez de points
  Trajet.prototype.avance = function (nbPas) {
    const tol = 1e-10, K = new Array(7);
    for (let n = 0; n < nbPas && this.etat === "en cours"; n++) {
      const s = this.s, h = this.h;
      for (let i = 0; i < 7; i++) {
        const si = s.slice();
        for (let k = 0; k < i; k++) for (let j = 0; j < 4; j++) si[j] += h * DP_A[i][k] * K[k][j];
        K[i] = f(si);
      }
      const y5 = s.slice(), y4 = s.slice();
      let err = 0;
      for (let j = 0; j < 4; j++) {
        for (let k = 0; k < 7; k++) { y5[j] += h * DP_B5[k] * K[k][j]; y4[j] += h * DP_B4[k] * K[k][j]; }
        err = Math.max(err, Math.abs(y5[j] - y4[j]) / (tol + tol * Math.max(Math.abs(s[j]), Math.abs(y5[j]))));
      }
      if (err <= 1) {
        if (s[1] > 0 && y5[1] <= 0) {
          const c = pasHenon(s);
          this.pts.push(c[0], c[2]);
        }
        this.s = y5; this.t += h;
        const r1 = Math.hypot(y5[0] + MU, y5[1]), r2 = Math.hypot(y5[0] - MUP, y5[1]);
        if (r1 < R_TERRE) this.etat = "Terre";
        else if (r2 < R_LUNE) this.etat = "Lune";
        else if (r1 > 3) this.etat = "éjectée";
        else if (this.pts.length >= 2 * 400 || this.t > 3000) this.etat = "fini";
      }
      this.h = h * (err > 0 ? Math.min(5, Math.max(0.2, 0.9 * Math.pow(err, -0.2))) : 5);
      if (this.h < 1e-12) this.etat = "Lune";
    }
  };

  function figurePoincare() {
    const canvas = document.getElementById("fig-poincare");
    const curseur = document.getElementById("poincare-c"), val = document.getElementById("poincare-c-val");
    const btnArenstorf = document.getElementById("poincare-arenstorf");
    const btnEffacer = document.getElementById("poincare-effacer");
    const out = document.getElementById("poincare-readout");
    if (!canvas || !curseur) return;
    const C_A = jacobi(S0);
    const VUE = { xmin: -1.35, xmax: 1.45, vmin: -2.6, vmax: 2.6 };
    let trajets = [], visible = true;

    function couleur(k) {
      const sombre = matchMedia("(prefers-color-scheme: dark)").matches;
      return "hsl(" + Math.round((k * 137.5) % 360) + ", 55%, " + (sombre ? "66%" : "42%") + ")";
    }
    function depart(x, vx, C) {
      const v2 = 2 * omega(x, 0) - C - vx * vx;
      return v2 < 0 ? null : [x, 0, vx, -Math.sqrt(v2)];
    }
    function relancer() {
      const C = +curseur.value;
      val.textContent = C.toFixed(3).replace(".", ",");
      trajets = [];
      let k = 0;
      for (let x = -1.25; x <= 1.4; x += 0.075) {
        if (Math.abs(x + MU) < 0.06 || Math.abs(x - MUP) < 0.03) continue;
        const s = depart(x, 0, C);
        if (s) trajets.push(new Trajet(s, couleur(k++), 1.4));
      }
      if (Math.abs(C - C_A) < 2e-3) {
        const p = S0.slice(); p[0] += 1e-4;
        trajets.push(new Trajet(p, "#c9a227", 2.4));
        const exacte = new Trajet(S0, "#c9a227", 7);
        exacte.pts = [0.994, 0, -0.5776, -0.3576, -0.5776, 0.3576];
        exacte.etat = "fixe";
        trajets.push(exacte);
      }
    }

    function dessiner() {
      const { ctx, w, h } = setupCanvas(canvas);
      const muted = themeColor("--muted"), line = themeColor("--line"), fg = themeColor("--fg");
      const g = 40, bas = 26;
      const X = (x) => g + (x - VUE.xmin) / (VUE.xmax - VUE.xmin) * (w - g - 8);
      const V = (v) => 8 + (1 - (v - VUE.vmin) / (VUE.vmax - VUE.vmin)) * (h - 8 - bas);
      const C = +curseur.value;
      ctx.clearRect(0, 0, w, h);
      ctx.font = "11px " + themeColor("--sans"); ctx.lineWidth = 1;
      // axes et graduations
      ctx.strokeStyle = line;
      ctx.beginPath(); ctx.moveTo(g, V(0)); ctx.lineTo(w - 8, V(0)); ctx.stroke();
      ctx.fillStyle = muted; ctx.textAlign = "center"; ctx.textBaseline = "top";
      for (let x = -1; x <= 1.01; x += 0.5) ctx.fillText(x.toFixed(1).replace(".", ","), X(x), h - bas + 6);
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      for (let v = -2; v <= 2; v += 1) ctx.fillText(v, g - 6, V(v));
      ctx.textAlign = "right"; ctx.textBaseline = "alphabetic"; ctx.fillText("x →", w - 10, h - bas - 4);
      ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillText("ẋ", g + 4, 10);
      // Terre et Lune
      [[-MU, "Terre"], [MUP, "Lune"]].forEach(([x, nom]) => {
        ctx.strokeStyle = muted; ctx.setLineDash([2, 4]);
        ctx.beginPath(); ctx.moveTo(X(x), 8); ctx.lineTo(X(x), h - bas); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = muted; ctx.textAlign = "center"; ctx.fillText(nom, X(x), 10);
      });
      // limite de la zone accessible : ẋ² = 2Ω(x, 0) − C
      ctx.strokeStyle = fg; ctx.globalAlpha = 0.35; ctx.lineWidth = 1;
      [1, -1].forEach((sg) => {
        ctx.beginPath(); let enCours = false;
        for (let i = 0; i <= 600; i++) {
          const x = VUE.xmin + (VUE.xmax - VUE.xmin) * i / 600;
          const v2 = 2 * omega(x, 0) - C;
          if (v2 < 0 || Math.abs(x + MU) < 1e-3 || Math.abs(x - MUP) < 1e-3) { enCours = false; continue; }
          const v = sg * Math.min(Math.sqrt(v2), VUE.vmax + 1);
          enCours ? ctx.lineTo(X(x), V(v)) : ctx.moveTo(X(x), V(v));
          enCours = true;
        }
        ctx.stroke();
      });
      ctx.globalAlpha = 1;
      // points de la section
      trajets.forEach((tr) => {
        ctx.fillStyle = tr.couleur;
        const r = tr.taille;
        for (let k = 0; k < tr.pts.length; k += 2) {
          const x = tr.pts[k], v = tr.pts[k + 1];
          if (x < VUE.xmin || x > VUE.xmax || v < VUE.vmin || v > VUE.vmax) continue;
          if (r > 4) { ctx.beginPath(); ctx.arc(X(x), V(v), r / 2, 0, 2 * Math.PI); ctx.fill(); }
          else ctx.fillRect(X(x) - r / 2, V(v) - r / 2, r, r);
        }
      });
      // bilan
      const n = { "en cours": 0, fini: 0, Terre: 0, Lune: 0, "éjectée": 0 };
      trajets.forEach((tr) => { if (tr.etat in n) n[tr.etat]++; });
      out.textContent = (Math.abs(C - C_A) < 2e-3 ? "Énergie de l'orbite d'Arenstorf. " : "") +
        "Trajectoires restées dans le système : " + (n["en cours"] + n.fini) +
        " · écrasées sur la Terre : " + n.Terre + " · sur la Lune : " + n.Lune +
        " · éjectées : " + n["éjectée"] + (n["en cours"] ? " · calcul en cours…" : "");
    }

    function image() {
      if (visible && trajets.some((tr) => tr.etat === "en cours")) {
        trajets.forEach((tr) => tr.avance(150));
        dessiner();
      }
      requestAnimationFrame(image);
    }

    curseur.addEventListener("input", () => { relancer(); dessiner(); });
    btnArenstorf.addEventListener("click", () => { curseur.value = C_A.toFixed(4); relancer(); dessiner(); });
    btnEffacer.addEventListener("click", () => { trajets = trajets.filter((tr) => tr.taille !== 2); dessiner(); });
    canvas.addEventListener("click", (e) => {
      const r = canvas.getBoundingClientRect(), g = 40, bas = 26;
      const x = VUE.xmin + (e.clientX - r.left - g) / (r.width - g - 8) * (VUE.xmax - VUE.xmin);
      const v = VUE.vmin + (1 - (e.clientY - r.top - 8) / (r.height - 8 - bas)) * (VUE.vmax - VUE.vmin);
      const s = depart(x, v, +curseur.value);
      if (s) { trajets.push(new Trajet(s, fgClic(), 2)); dessiner(); }
    });
    function fgClic() { return themeColor("--fg"); }
    if ("IntersectionObserver" in window) {
      new IntersectionObserver((e) => { visible = e[0].isIntersecting; }).observe(canvas);
    }
    window.addEventListener("resize", dessiner);
    curseur.value = C_A.toFixed(4);
    relancer();
    dessiner();
    requestAnimationFrame(image);
  }

  document.addEventListener("DOMContentLoaded", function () {
    figureOrbite();
    figureMethodes();
    figurePerturbation();
    figurePoincare();
  });
})();
