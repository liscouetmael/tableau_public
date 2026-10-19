// ==========================================================
//  Problème à trois corps
//  1. Simulation de solutions connues, avec un « jumeau » légèrement perturbé
//  2. Application standard de Chirikov et tore au nombre d'or
//  3. Exposant de Lyapunov en fonction du rapport des périodes (données précalculées)
// ==========================================================
(function () {
  "use strict";

  // ---------- Dormand–Prince 5(4) à pas adaptatif (le RK45 de scipy, le ode45 de MATLAB) ----------
  const DP_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]
  ];
  const DP_B5 = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0];
  const DP_B4 = [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40];

  function Integrateur(f, y, tol) {
    this.f = f; this.y = y.slice(); this.t = 0; this.h = 1e-3; this.tol = tol; this.ok = true;
  }
  // Avance jusqu'au temps tFin. Renvoie false si le pas s'effondre (quasi-collision).
  Integrateur.prototype.avance = function (tFin) {
    const n = this.y.length, K = new Array(7);
    while (this.ok && this.t < tFin - 1e-12) {
      const h = Math.min(this.h, tFin - this.t), y = this.y;
      for (let i = 0; i < 7; i++) {
        const yi = new Array(n);
        for (let j = 0; j < n; j++) {
          let s = y[j];
          for (let k = 0; k < i; k++) s += h * DP_A[i][k] * K[k][j];
          yi[j] = s;
        }
        K[i] = this.f(yi);
      }
      let err = 0;
      const y5 = new Array(n);
      for (let j = 0; j < n; j++) {
        let a = y[j], b = y[j];
        for (let k = 0; k < 7; k++) { a += h * DP_B5[k] * K[k][j]; b += h * DP_B4[k] * K[k][j]; }
        y5[j] = a;
        const e = Math.abs(a - b) / (this.tol + this.tol * Math.max(Math.abs(y[j]), Math.abs(a)));
        if (e > err) err = e;
      }
      if (err <= 1) { this.t += h; this.y = y5; }
      this.h = h * (err > 0 ? Math.min(5, Math.max(0.2, 0.9 * Math.pow(err, -0.2))) : 5);
      if (this.h < 1e-9) this.ok = false;
    }
    return this.ok;
  };

  // ---------- Problème à trois corps dans le plan (G = 1) ----------
  // État : [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]
  function champTroisCorps(m) {
    return function (Y) {
      const out = new Array(12);
      for (let i = 0; i < 6; i++) out[i] = Y[6 + i];
      for (let i = 6; i < 12; i++) out[i] = 0;
      for (let a = 0; a < 3; a++) {
        for (let b = a + 1; b < 3; b++) {
          const dx = Y[2 * b] - Y[2 * a], dy = Y[2 * b + 1] - Y[2 * a + 1];
          const r2 = dx * dx + dy * dy, inv = 1 / (r2 * Math.sqrt(r2));
          out[6 + 2 * a] += m[b] * dx * inv; out[7 + 2 * a] += m[b] * dy * inv;
          out[6 + 2 * b] -= m[a] * dx * inv; out[7 + 2 * b] -= m[a] * dy * inv;
        }
      }
      return out;
    };
  }

  function suvakov(p1, p2) { return [-1, 0, 1, 0, 0, 0, p1, p2, p1, p2, -2 * p1, -2 * p2]; }
  function lagrange(m) {          // triangle équilatéral de côté 1 en rotation rigide
    const M = m[0] + m[1] + m[2], w = Math.sqrt(M);
    const pts = [[0, 0], [1, 0], [0.5, Math.sqrt(3) / 2]];
    const cx = (m[0] * 0 + m[1] * 1 + m[2] * 0.5) / M, cy = m[2] * Math.sqrt(3) / 2 / M;
    const pos = [], vit = [];
    pts.forEach((p) => {
      const x = p[0] - cx, y = p[1] - cy;
      pos.push(x, y); vit.push(-w * y, w * x);
    });
    return pos.concat(vit);
  }
  const wEuler = Math.sqrt(1.25);
  const vb = Math.SQRT2 / 2, vrel = Math.sqrt(3 / 10);

  const PRESETS = {
    chaos: {
      m: [1, 1, 1], y: [-1, 0, 1, 0, 0, 1, 0, 0.5, 0, -0.5, 0.5, 0], vue: 2.2, vitesse: 1,
      texte: "Trois masses égales, conditions initiales quelconques : le mouvement est chaotique, et le jumeau décroche très vite."
    },
    huit: {
      m: [1, 1, 1],
      y: [-0.97000436, 0.24308753, 0.97000436, -0.24308753, 0, 0,
          0.46620368, 0.43236573, 0.46620368, 0.43236573, -0.93240737, -0.86473146],
      vue: 1.3, vitesse: 1,
      texte: "Le huit de Moore, Chenciner et Montgomery : trois masses égales se poursuivent sur une même courbe. Il est stable : le jumeau reste collé."
    },
    troyen: {
      m: [1, 0.001, 1e-6], y: lagrange([1, 0.001, 1e-6]), vue: 1.3, vitesse: 1.5,
      texte: "Soleil, Jupiter et un astéroïde troyen au sommet du triangle équilatéral (point L4). Stable, comme les vrais troyens de Jupiter."
    },
    lagrange: {
      m: [1, 1, 1], y: lagrange([1, 1, 1]), vue: 1, vitesse: 0.6,
      texte: "La même configuration de Lagrange avec trois masses égales : elle est instable, le triangle se disloque au bout de quelques tours."
    },
    euler: {
      m: [1, 1, 1], y: [-1, 0, 0, 0, 1, 0, 0, -wEuler, 0, 0, 0, wEuler], vue: 1.6, vitesse: 0.6,
      texte: "La solution alignée d'Euler : trois corps en ligne qui tournent ensemble. Elle est toujours instable."
    },
    papillon: {
      m: [1, 1, 1], y: suvakov(0.306893, 0.125507), vue: 1.3, vitesse: 1,
      texte: "Le « papillon I » de Šuvakov et Dmitrašinović (2013) : une orbite périodique, mais instable."
    },
    hierarchique: {
      m: [1, 1, 1], y: [-0.5, 0, 0.5, 0, 10, 0, 0, -vb - vrel / 3, 0, vb - vrel / 3, 0, 2 * vrel / 3],
      vue: 11, vitesse: 6,
      texte: "Un système hiérarchique : un couple serré et un troisième corps lointain, comme la Terre, la Lune et le Soleil. C'est la façon la plus courante d'être stable."
    }
  };

  // ---------- Figure 1 : simulation et jumeau perturbé ----------
  function figureSimulation() {
    const canvas = document.getElementById("tc-sim");
    if (!canvas) return;
    const ecart = document.getElementById("tc-ecart");
    const choix = document.getElementById("tc-preset"), delta = document.getElementById("tc-delta");
    const info = document.getElementById("tc-info"), descr = document.getElementById("tc-descr");
    const btnRejouer = document.getElementById("tc-rejouer"), btnPause = document.getElementById("tc-pause");
    const TRAINE = 500;
    let P, ref, jumeau, d0, traines, journal, enPause = false, visible = true, dernier = null;

    function demarrer() {
      P = PRESETS[choix.value];
      d0 = +delta.value;
      const f = champTroisCorps(P.m);
      ref = new Integrateur(f, P.y, 1e-11);
      const y2 = P.y.slice(); y2[0] += d0;
      jumeau = new Integrateur(f, y2, 1e-11);
      traines = [[], [], []];
      journal = [];
      descr.textContent = P.texte;
    }

    function centreMasse(Y) {
      const M = P.m[0] + P.m[1] + P.m[2];
      return [(P.m[0] * Y[0] + P.m[1] * Y[2] + P.m[2] * Y[4]) / M,
              (P.m[0] * Y[1] + P.m[1] * Y[3] + P.m[2] * Y[5]) / M];
    }

    function distance() {
      let s = 0;
      for (let j = 0; j < 12; j++) { const d = ref.y[j] - jumeau.y[j]; s += d * d; }
      return Math.sqrt(s);
    }

    function dessiner() {
      const { ctx, w, h } = setupCanvas(canvas);
      const fg = themeColor("--fg"), muted = themeColor("--muted");
      const couleurs = [themeColor("--accent"), themeColor("--accent-2"), "#c9a227"];
      const echelle = Math.min(w, h) / (2 * P.vue);
      const cm = centreMasse(ref.y);
      const X = (x) => w / 2 + (x - cm[0]) * echelle, Y = (y) => h / 2 - (y - cm[1]) * echelle;
      ctx.clearRect(0, 0, w, h);
      for (let b = 0; b < 3; b++) {
        const tr = traines[b];
        ctx.strokeStyle = couleurs[b]; ctx.globalAlpha = 0.45; ctx.lineWidth = 1.2;
        ctx.beginPath();
        for (let k = 0; k < tr.length; k++) {
          const px = X(tr[k][0]), py = Y(tr[k][1]);
          k ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
        }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      for (let b = 0; b < 3; b++) {
        const r = Math.max(2.5, 6 * Math.cbrt(P.m[b] / Math.max(P.m[0], P.m[1], P.m[2])));
        // jumeau : cercle vide
        ctx.strokeStyle = muted; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.arc(X(jumeau.y[2 * b]), Y(jumeau.y[2 * b + 1]), r + 3, 0, 2 * Math.PI); ctx.stroke();
        ctx.fillStyle = couleurs[b];
        ctx.beginPath(); ctx.arc(X(ref.y[2 * b]), Y(ref.y[2 * b + 1]), r, 0, 2 * Math.PI); ctx.fill();
      }
      ctx.fillStyle = fg; ctx.font = "12px " + themeColor("--sans");
      ctx.fillText("t = " + ref.t.toFixed(1), 10, 18);
    }

    function dessinerEcart() {
      const { ctx, w, h } = setupCanvas(ecart);
      const muted = themeColor("--muted"), line = themeColor("--line"), c = themeColor("--accent-2"), fg = themeColor("--fg");
      const g = 44, d = 10, t = 10, bas = 24;
      const ymin = Math.floor(Math.log10(d0)) - 1, ymax = 1;
      const tmax = Math.max(10, journal.length ? journal[journal.length - 1][0] : 10);
      const X = (x) => g + (x / tmax) * (w - g - d), Y = (v) => t + (1 - (v - ymin) / (ymax - ymin)) * (h - t - bas);
      ctx.clearRect(0, 0, w, h);
      ctx.font = "11px " + themeColor("--sans"); ctx.lineWidth = 1;
      for (let v = ymin; v <= ymax; v++) {
        if ((v - ymin) % 2 && ymax - ymin > 8) continue;
        ctx.strokeStyle = line; ctx.beginPath(); ctx.moveTo(g, Y(v)); ctx.lineTo(w - d, Y(v)); ctx.stroke();
        ctx.fillStyle = muted; ctx.textAlign = "right"; ctx.textBaseline = "middle";
        ctx.fillText("10" + (v < 0 ? "⁻" : "") + String(Math.abs(v)).split("").map((ch) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[+ch]).join(""), g - 6, Y(v));
      }
      ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillStyle = fg;
      ctx.fillText("écart entre la simulation et son jumeau", g + 6, t);
      ctx.textAlign = "right"; ctx.fillStyle = muted; ctx.textBaseline = "alphabetic";
      ctx.fillText("temps →", w - d, h - 6);
      if (journal.length < 2) return;
      ctx.strokeStyle = c; ctx.lineWidth = 2; ctx.beginPath();
      journal.forEach((p, k) => {
        const v = Math.max(ymin, Math.min(ymax, Math.log10(p[1])));
        k ? ctx.lineTo(X(p[0]), Y(v)) : ctx.moveTo(X(p[0]), Y(v));
      });
      ctx.stroke();
    }

    function image(maintenant) {
      if (dernier === null) dernier = maintenant;
      const dt = Math.min(0.05, (maintenant - dernier) / 1000);
      dernier = maintenant;
      if (!enPause && visible && ref.ok && jumeau.ok) {
        const cible = ref.t + dt * P.vitesse;
        ref.avance(cible); jumeau.avance(cible);
        for (let b = 0; b < 3; b++) {
          traines[b].push([ref.y[2 * b], ref.y[2 * b + 1]]);
          if (traines[b].length > TRAINE) traines[b].shift();
        }
        const dist = distance();
        if (ref.t > 0) journal.push([ref.t, dist]);
        if (journal.length > 3000) journal = journal.filter((p, k) => k % 2 === 0);   // garde la courbe légère
        const lam = ref.t > 0.5 ? Math.log(dist / d0) / ref.t : NaN;
        info.textContent = (ref.ok && jumeau.ok ? "" : "Quasi-collision : simulation arrêtée. ") +
          "écart = " + dist.toExponential(1).replace(".", ",") +
          (isNaN(lam) ? "" : " · λ estimé = ln(écart / écart initial) / t = " + lam.toFixed(2).replace(".", ","));
        dessiner();
        dessinerEcart();
      }
      requestAnimationFrame(image);
    }

    choix.addEventListener("change", demarrer);
    delta.addEventListener("change", demarrer);
    btnRejouer.addEventListener("click", demarrer);
    btnPause.addEventListener("click", () => {
      enPause = !enPause;
      btnPause.textContent = enPause ? "Reprendre" : "Pause";
    });
    if ("IntersectionObserver" in window) {
      new IntersectionObserver((e) => { visible = e[0].isIntersecting; }).observe(canvas);
    }
    demarrer();
    requestAnimationFrame(image);
  }

  // ---------- Figure 2 : application standard p' = p + K sin x, x' = x + p' ----------
  function figureStandard() {
    const canvas = document.getElementById("tc-std");
    if (!canvas) return;
    const curseur = document.getElementById("tc-k"), val = document.getElementById("tc-k-val");
    const info = document.getElementById("tc-std-info"), effacer = document.getElementById("tc-std-effacer");
    const TAU = 2 * Math.PI, OR = (Math.sqrt(5) - 1) / 2, K_C = 0.971635;
    let ajoutees = [], minuteur = null;

    function rotation(x, p, K, N) {
      let X = x;
      for (let i = 0; i < N; i++) { p += K * Math.sin(X); X += p; }
      return (X - x) / (TAU * N);
    }
    // Orbite de nombre de rotation doré, cherchée par dichotomie sur la droite x = 0
    function orbiteDoree(K) {
      let lo = 0.5 * TAU, hi = 0.75 * TAU;
      for (let i = 0; i < 40; i++) {
        const mid = (lo + hi) / 2;
        if (rotation(0, mid, K, 1500) < OR) lo = mid; else hi = mid;
      }
      return (lo + hi) / 2;
    }
    function orbite(x, p, K, N) {
      const pts = new Float32Array(2 * N);
      for (let i = 0; i < N; i++) {
        p += K * Math.sin(x); x += p;
        pts[2 * i] = ((x % TAU) + TAU) % TAU; pts[2 * i + 1] = ((p % TAU) + TAU) % TAU;
      }
      return pts;
    }

    function dessiner() {
      const K = +curseur.value;
      val.textContent = K.toFixed(3).replace(".", ",");
      const { ctx, w, h } = setupCanvas(canvas);
      const X = (x) => (x / TAU) * w, Y = (p) => h - (p / TAU) * h;
      const muted = themeColor("--muted"), c1 = themeColor("--accent"), c2 = themeColor("--accent-2");
      ctx.clearRect(0, 0, w, h);
      const couleurs = [c1, c2, muted];
      const N0 = 28;
      for (let i = 0; i < N0; i++) {
        const pts = orbite(i % 2 ? Math.PI : 0.001, TAU * (i + 0.5) / N0, K, 700);
        ctx.fillStyle = couleurs[i % 3]; ctx.globalAlpha = 0.55;
        for (let k = 0; k < pts.length; k += 2) ctx.fillRect(X(pts[k]), Y(pts[k + 1]), 1.2, 1.2);
      }
      ajoutees.forEach((s) => {
        const pts = orbite(s[0], s[1], K, 3000);
        ctx.fillStyle = themeColor("--fg"); ctx.globalAlpha = 0.7;
        for (let k = 0; k < pts.length; k += 2) ctx.fillRect(X(pts[k]), Y(pts[k + 1]), 1.2, 1.2);
      });
      const p0 = orbiteDoree(K);
      const dore = orbite(0, p0, K, 4000);
      ctx.globalAlpha = 1; ctx.fillStyle = "#c9a227";
      for (let k = 0; k < dore.length; k += 2) ctx.fillRect(X(dore[k]) - 0.8, Y(dore[k + 1]) - 0.8, 2.2, 2.2);
      info.textContent = K < K_C
        ? "K < Kc ≈ 0,9716 : le tore de nombre de rotation doré (en jaune) existe encore et sépare le bas du haut."
        : "K > Kc ≈ 0,9716 : le tore doré est détruit. Les orbites chaotiques peuvent traverser, mais juste au-dessus du seuil elles mettent très longtemps à le faire.";
    }
    function planifier() { clearTimeout(minuteur); minuteur = setTimeout(dessiner, 30); }

    curseur.addEventListener("input", planifier);
    canvas.addEventListener("click", (e) => {
      const r = canvas.getBoundingClientRect();
      ajoutees.push([(e.clientX - r.left) / r.width * TAU, (1 - (e.clientY - r.top) / r.height) * TAU]);
      if (ajoutees.length > 12) ajoutees.shift();
      dessiner();
    });
    effacer.addEventListener("click", () => { ajoutees = []; dessiner(); });
    window.addEventListener("resize", planifier);
    dessiner();
  }

  // ---------- Figure 3 : exposant de Lyapunov selon le rapport des périodes ----------
  // Calculé hors ligne (Dormand–Prince, méthode de Benettin, t = 600 soit 95 orbites intérieures).
  // null : la simulation s'est arrêtée sur une quasi-collision.
  const SCAN = {
    m3: [[1.45,0.1121],[1.46,0.0697],[1.47,0.0902],[1.48,0.0664],[1.49,0.0793],[1.5,0.1034],[1.51,null],[1.52,0.1044],[1.53,0.014],[1.54,0.0433],[1.55,0.0816],[1.56,0.0455],[1.57,0.1076],[1.58,0.0832],[1.59,0.0659],[1.6,0.017],[1.61,0.0126],[1.6154,0.0126],[1.618,0.0125],[1.62,0.0125],[1.625,0.0125],[1.63,0.0125],[1.64,0.0127],[1.65,0.0128],[1.66,0.0129],[1.6667,0.0127],[1.67,0.0129],[1.68,0.013],[1.69,0.0128],[1.7,0.0131],[1.71,0.0128],[1.72,0.0132],[1.73,0.0129],[1.74,0.0132],[1.75,0.013],[1.76,0.013],[1.77,0.013],[1.78,0.0128],[1.79,0.013],[1.8,0.0251],[1.81,0.0277],[1.82,0.032],[1.83,0.0307],[1.84,0.0367],[1.85,0.0264],[1.86,0.0131],[1.87,0.0128],[1.88,0.0127],[1.89,0.0126],[1.9,0.0126],[1.91,0.0127],[1.92,0.0128],[1.93,0.0129],[1.94,0.0129],[1.95,0.0129],[1.96,0.0128],[1.97,0.0127],[1.98,0.0126],[1.99,0.0125],[2,0.0126],[2.01,0.0126],[2.02,0.0127],[2.03,0.0128],[2.04,0.013],[2.05,0.013],[2.06,0.0129],[2.07,0.0127],[2.08,0.0127],[2.09,0.0128],[2.1,0.0132]],
    m8: [[1.45,0.101],[1.46,0.0754],[1.47,0.0891],[1.48,0.0972],[1.49,0.0821],[1.5,0.0969],[1.51,0.1002],[1.52,0.1279],[1.53,0.1039],[1.54,0.1054],[1.55,0.1271],[1.56,0.0817],[1.57,0.1082],[1.58,0.1375],[1.59,0.0917],[1.6,0.0928],[1.61,0.085],[1.6154,0.0871],[1.618,0.046],[1.62,0.108],[1.625,0.1783],[1.63,0.1392],[1.64,null],[1.65,null],[1.66,0.1127],[1.6667,0.1216],[1.67,0.111],[1.68,0.0613],[1.69,0.0893],[1.7,0.1263],[1.71,0.0938],[1.72,0.1406],[1.73,0.1186],[1.74,0.1397],[1.75,0.0287],[1.76,0.0791],[1.77,0.0653],[1.78,0.0862],[1.79,0.0811],[1.8,0.0805],[1.81,0.0944],[1.82,0.0942],[1.83,0.0762],[1.84,0.0317],[1.85,0.1139],[1.86,0.0775],[1.87,0.0782],[1.88,0.0363],[1.89,0.0358],[1.9,0.0219],[1.91,0.013],[1.92,0.0122],[1.93,0.0122],[1.94,0.0122],[1.95,0.0123],[1.96,0.0125],[1.97,0.0127],[1.98,0.0129],[1.99,0.0129],[2,0.0131],[2.01,0.0131],[2.02,0.0127],[2.03,0.0124],[2.04,0.0124],[2.05,0.0123],[2.06,0.0124],[2.07,0.0124],[2.08,0.0125],[2.09,0.0126],[2.1,0.0128]]
  };

  function figureScan() {
    const canvas = document.getElementById("tc-scan");
    if (!canvas) return;
    function dessiner() {
      const { ctx, w, h } = setupCanvas(canvas);
      const fg = themeColor("--fg"), muted = themeColor("--muted"), line = themeColor("--line");
      const c1 = themeColor("--accent"), c2 = themeColor("--accent-2");
      const g = 46, d = 12, t = 26, bas = 30, xmin = 1.45, xmax = 2.1, ymax = 0.19;
      const X = (x) => g + (x - xmin) / (xmax - xmin) * (w - g - d);
      const Y = (v) => t + (1 - v / ymax) * (h - t - bas);
      ctx.clearRect(0, 0, w, h);
      ctx.font = "11px " + themeColor("--sans"); ctx.lineWidth = 1;
      for (let v = 0; v <= 0.18; v += 0.06) {
        ctx.strokeStyle = line; ctx.beginPath(); ctx.moveTo(g, Y(v)); ctx.lineTo(w - d, Y(v)); ctx.stroke();
        ctx.fillStyle = muted; ctx.textAlign = "right"; ctx.textBaseline = "middle";
        ctx.fillText(v.toFixed(2).replace(".", ","), g - 6, Y(v));
      }
      const reperes = [[1.5, "3/2"], [1.6, "8/5"], [(1 + Math.sqrt(5)) / 2, "φ"], [5 / 3, "5/3"], [1.8, "9/5"], [2, "2/1"]];
      reperes.forEach(([x, nom]) => {
        ctx.strokeStyle = nom === "φ" ? "#c9a227" : line; ctx.lineWidth = nom === "φ" ? 2 : 1;
        ctx.setLineDash(nom === "φ" ? [] : [3, 3]);
        ctx.beginPath(); ctx.moveTo(X(x), t); ctx.lineTo(X(x), h - bas); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = nom === "φ" ? "#c9a227" : muted; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
        ctx.fillText(nom, X(x), t - 4);
      });
      ctx.strokeStyle = muted; ctx.setLineDash([2, 4]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(g, Y(0.0127)); ctx.lineTo(w - d, Y(0.0127)); ctx.stroke(); ctx.setLineDash([]);
      [[SCAN.m3, c1], [SCAN.m8, c2]].forEach(([s, col]) => {
        ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1.6;
        ctx.beginPath();
        let enCours = false;
        s.forEach(([x, v]) => {
          if (v === null) { enCours = false; return; }
          enCours ? ctx.lineTo(X(x), Y(v)) : ctx.moveTo(X(x), Y(v));
          enCours = true;
        });
        ctx.stroke();
        s.forEach(([x, v]) => {
          if (v === null) {
            ctx.font = "bold 12px " + themeColor("--sans"); ctx.textAlign = "center"; ctx.textBaseline = "middle";
            ctx.fillText("×", X(x), Y(0.18));
          }
        });
      });
      ctx.font = "11px " + themeColor("--sans");
      ctx.fillStyle = muted; ctx.textAlign = "center"; ctx.textBaseline = "top";
      for (let x = 1.5; x <= 2.1 + 1e-9; x += 0.1) ctx.fillText(x.toFixed(1).replace(".", ","), X(x), h - bas + 6);
      ctx.fillStyle = fg; ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText("λ", 6, t - 18);
      ctx.textAlign = "right"; ctx.fillStyle = muted;
      ctx.fillText("rapport des périodes T₃ / T₂", w - d, h - 14);
    }
    let minuteur = null;
    window.addEventListener("resize", () => { clearTimeout(minuteur); minuteur = setTimeout(dessiner, 100); });
    dessiner();
  }

  document.addEventListener("DOMContentLoaded", function () {
    figureSimulation();
    figureStandard();
    figureScan();
  });
})();
