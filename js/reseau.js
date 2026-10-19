// ==========================================================
//  Perceptron multicouche écrit à la main
//  Couches cachées tanh, sortie sigmoïde, entropie croisée binaire,
//  rétropropagation et optimiseur Adam. Aucune bibliothèque.
// ==========================================================
(function () {

  // ---------- Données ----------
  function randn() {   // loi normale (Box-Muller)
    return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
  }

  function makeData(kind, n) {
    const X = [], Y = [];
    for (let i = 0; i < n; i++) {
      let x, y, c;
      if (kind === "cercles") {
        c = i % 2;
        const r = c ? 0.35 * Math.random() : 0.65 + 0.3 * Math.random();
        const a = 2 * Math.PI * Math.random();
        x = r * Math.cos(a); y = r * Math.sin(a);
      } else if (kind === "xor") {
        x = 2 * Math.random() - 1; y = 2 * Math.random() - 1;
        x += x > 0 ? 0.05 : -0.05; y += y > 0 ? 0.05 : -0.05;
        c = x * y > 0 ? 1 : 0;
      } else { // spirales
        c = i % 2;
        const t = Math.random();
        const r = 0.08 + 0.85 * t, a = 3.5 * Math.PI * t + c * Math.PI;
        x = r * Math.cos(a) + 0.03 * randn(); y = r * Math.sin(a) + 0.03 * randn();
      }
      X.push([x, y]); Y.push(c);
    }
    return { X: X, Y: Y };
  }

  // ---------- Réseau ----------
  function Network(sizes) {
    this.sizes = sizes;
    this.L = sizes.length - 1;
    this.W = []; this.b = [];
    this.mW = []; this.vW = []; this.mb = []; this.vb = [];
    for (let l = 0; l < this.L; l++) {
      const nin = sizes[l], nout = sizes[l + 1];
      const W = new Float64Array(nin * nout);
      const s = Math.sqrt(1 / nin);           // initialisation de Xavier
      for (let i = 0; i < W.length; i++) W[i] = s * randn();
      this.W.push(W); this.b.push(new Float64Array(nout));
      this.mW.push(new Float64Array(nin * nout)); this.vW.push(new Float64Array(nin * nout));
      this.mb.push(new Float64Array(nout)); this.vb.push(new Float64Array(nout));
    }
    this.t = 0;
  }

  // Propagation avant : renvoie les activations de chaque couche
  Network.prototype.forward = function (x) {
    const A = [Float64Array.from(x)];
    for (let l = 0; l < this.L; l++) {
      const nin = this.sizes[l], nout = this.sizes[l + 1];
      const a = A[l], W = this.W[l], b = this.b[l];
      const z = new Float64Array(nout);
      for (let j = 0; j < nout; j++) {
        let s = b[j];
        for (let i = 0; i < nin; i++) s += W[j * nin + i] * a[i];
        z[j] = l === this.L - 1 ? 1 / (1 + Math.exp(-s)) : Math.tanh(s);
      }
      A.push(z);
    }
    return A;
  };

  Network.prototype.predict = function (x) {
    const A = this.forward(x);
    return A[this.L][0];
  };

  // Un pas d'apprentissage sur un mini-lot ; renvoie la perte moyenne
  Network.prototype.trainBatch = function (X, Y, idx, lr) {
    const gW = this.W.map(function (w) { return new Float64Array(w.length); });
    const gb = this.b.map(function (b) { return new Float64Array(b.length); });
    let loss = 0;
    for (const k of idx) {
      const A = this.forward(X[k]);
      const yhat = A[this.L][0], y = Y[k];
      loss -= y * Math.log(yhat + 1e-12) + (1 - y) * Math.log(1 - yhat + 1e-12);
      // sigmoïde + entropie croisée : delta de sortie = yhat - y
      let delta = new Float64Array([yhat - y]);
      for (let l = this.L - 1; l >= 0; l--) {
        const nin = this.sizes[l], nout = this.sizes[l + 1], a = A[l], W = this.W[l];
        for (let j = 0; j < nout; j++) {
          gb[l][j] += delta[j];
          for (let i = 0; i < nin; i++) gW[l][j * nin + i] += delta[j] * a[i];
        }
        if (l > 0) {
          const prev = new Float64Array(nin);
          for (let i = 0; i < nin; i++) {
            let s = 0;
            for (let j = 0; j < nout; j++) s += W[j * nin + i] * delta[j];
            prev[i] = s * (1 - a[i] * a[i]);   // dérivée de tanh
          }
          delta = prev;
        }
      }
    }
    // Mise à jour Adam
    const m = idx.length, b1 = 0.9, b2 = 0.999, eps = 1e-8;
    this.t++;
    const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
    const self = this;
    function adam(P, G, M, V) {
      for (let i = 0; i < P.length; i++) {
        const g = G[i] / m;
        M[i] = b1 * M[i] + (1 - b1) * g;
        V[i] = b2 * V[i] + (1 - b2) * g * g;
        P[i] -= lr * (M[i] / c1) / (Math.sqrt(V[i] / c2) + eps);
      }
    }
    for (let l = 0; l < self.L; l++) {
      adam(self.W[l], gW[l], self.mW[l], self.vW[l]);
      adam(self.b[l], gb[l], self.mb[l], self.vb[l]);
    }
    return loss / m;
  };

  // ---------- Démo interactive ----------
  document.addEventListener("DOMContentLoaded", function () {
    const canvas = document.getElementById("fig-reseau");
    const lossCanvas = document.getElementById("fig-perte");
    if (!canvas) return;
    const selData = document.getElementById("nn-data");
    const sliderH = document.getElementById("nn-hidden");
    const labelH = document.getElementById("nn-hidden-label");
    const btnRun = document.getElementById("nn-run");
    const btnReset = document.getElementById("nn-reset");
    const out = document.getElementById("nn-readout");

    const GRID = 60, BATCH = 32, STEPS_PER_FRAME = 8, LR = 0.01;
    const off = document.createElement("canvas");
    off.width = GRID; off.height = GRID;
    const offCtx = off.getContext("2d");

    let data, net, running = false, epoch = 0, history = [];

    function reset() {
      const h = parseInt(sliderH.value, 10);
      labelH.textContent = h;
      data = makeData(selData.value, 400);
      net = new Network([2, h, h, 1]);
      epoch = 0; history = [];
      draw();
    }

    function hexToRgb(c) {
      const m = c.replace("#", "");
      return [parseInt(m.slice(0, 2), 16), parseInt(m.slice(2, 4), 16), parseInt(m.slice(4, 6), 16)];
    }

    function draw() {
      const { ctx, w, h } = setupCanvas(canvas);
      const c0 = hexToRgb(themeColor("--accent")), c1 = hexToRgb(themeColor("--accent-2"));
      const bg = hexToRgb(themeColor("--card"));
      // carte de décision : probabilité prédite sur une grille
      const img = offCtx.createImageData(GRID, GRID);
      for (let j = 0; j < GRID; j++) {
        for (let i = 0; i < GRID; i++) {
          const x = -1 + 2 * (i + 0.5) / GRID, y = 1 - 2 * (j + 0.5) / GRID;
          const p = net.predict([x, y]);
          const c = p > 0.5 ? c1 : c0, a = 0.12 + 0.3 * Math.abs(p - 0.5) * 2;
          const k = 4 * (j * GRID + i);
          img.data[k] = bg[0] + a * (c[0] - bg[0]);
          img.data[k + 1] = bg[1] + a * (c[1] - bg[1]);
          img.data[k + 2] = bg[2] + a * (c[2] - bg[2]);
          img.data[k + 3] = 255;
        }
      }
      offCtx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, 0, 0, w, h);
      // points de données
      for (let k = 0; k < data.X.length; k++) {
        const px = (data.X[k][0] + 1) / 2 * w, py = (1 - data.X[k][1]) / 2 * h;
        ctx.fillStyle = data.Y[k] ? themeColor("--accent-2") : themeColor("--accent");
        ctx.strokeStyle = themeColor("--card");
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(px, py, 3.2, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      }
      drawLoss();
      let correct = 0;
      for (let k = 0; k < data.X.length; k++) if ((net.predict(data.X[k]) > 0.5 ? 1 : 0) === data.Y[k]) correct++;
      const last = history.length ? history[history.length - 1] : NaN;
      out.textContent = "époque " + epoch + " · perte " + (isNaN(last) ? "—" : last.toFixed(4)) +
                        " · précision " + (100 * correct / data.X.length).toFixed(1) + " %";
    }

    function drawLoss() {
      if (!lossCanvas) return;
      const { ctx, w, h } = setupCanvas(lossCanvas);
      ctx.clearRect(0, 0, w, h);
      const pad = { l: 44, r: 12, t: 12, b: 26 };
      const ymin = Math.log10(0.005), ymax = Math.log10(1);
      ctx.strokeStyle = themeColor("--line"); ctx.lineWidth = 1;
      ctx.fillStyle = themeColor("--muted");
      ctx.font = "11px Inter, system-ui, sans-serif";
      [1, 0.1, 0.01].forEach(function (v) {
        const y = pad.t + (ymax - Math.log10(v)) / (ymax - ymin) * (h - pad.t - pad.b);
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
        ctx.fillText(String(v).replace(".", ","), 8, y + 4);
      });
      ctx.fillText("époques →", w - pad.r - 64, h - 8);
      if (history.length < 2) return;
      const n = Math.max(history.length, 50);
      ctx.strokeStyle = themeColor("--accent-2"); ctx.lineWidth = 1.8;
      ctx.beginPath();
      history.forEach(function (v, i) {
        const x = pad.l + i / (n - 1) * (w - pad.l - pad.r);
        const lv = Math.min(ymax, Math.max(ymin, Math.log10(v)));
        const y = pad.t + (ymax - lv) / (ymax - ymin) * (h - pad.t - pad.b);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.stroke();
    }

    function shuffle(a) {
      for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); const t = a[i]; a[i] = a[j]; a[j] = t; }
      return a;
    }

    function loop() {
      if (!running) return;
      for (let s = 0; s < STEPS_PER_FRAME; s++) {
        const order = shuffle(data.X.map(function (_, i) { return i; }));
        let tot = 0, nb = 0;
        for (let k = 0; k < order.length; k += BATCH) {
          tot += net.trainBatch(data.X, data.Y, order.slice(k, k + BATCH), LR); nb++;
        }
        epoch++;
        history.push(tot / nb);
      }
      draw();
      if (epoch >= 3000) { running = false; btnRun.textContent = "Entraîner"; return; }
      requestAnimationFrame(loop);
    }

    btnRun.addEventListener("click", function () {
      running = !running;
      btnRun.textContent = running ? "Pause" : "Entraîner";
      if (running) requestAnimationFrame(loop);
    });
    btnReset.addEventListener("click", function () {
      running = false; btnRun.textContent = "Entraîner"; reset();
    });
    selData.addEventListener("change", function () { btnReset.click(); });
    sliderH.addEventListener("input", function () { btnReset.click(); });
    window.addEventListener("resize", draw);
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);

    reset();
  });
})();
