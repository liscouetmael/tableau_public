// ==========================================================
//  Démo MNIST : entraîner un réseau dense ou convolutif
//  directement dans le navigateur, avec TensorFlow.js.
//  Les données sont un extrait de MNIST (data/mnist-extrait.js).
//  La bibliothèque et les données ne sont chargées qu'au premier clic.
// ==========================================================
(function () {
  const TFJS_URL = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
  const racine = document.getElementById("mnist-demo");
  if (!racine) return;
  const DATA_URL = racine.dataset.src;
  const P = 784;                          // pixels par image (28 x 28)
  const TAILLE_LOT = 64;
  const CHOIX_NEURONES = [8, 16, 32, 64, 128, 256];
  const FILTRES = [16, 32, 64];           // filtres de la 1re, 2e, 3e couche de convolution

  const $ = (id) => document.getElementById(id);
  const el = {
    modele: $("mn-modele"), optim: $("mn-optim"),
    denseOpts: $("mn-dense-opts"), couches: $("mn-couches"), couchesVal: $("mn-couches-val"),
    neurones: $("mn-neurones"),
    cnnOpts: $("mn-cnn-opts"), conv: $("mn-conv"), convVal: $("mn-conv-val"),
    epoques: $("mn-epoques"), epoquesVal: $("mn-epoques-val"), taille: $("mn-taille"),
    run: $("mn-run"), stop: $("mn-stop"),
    archi: $("mn-archi"), statut: $("mn-statut"), alerte: $("mn-alerte"),
    perte: $("mn-perte"), precision: $("mn-precision"),
    resultats: $("mn-resultats"), score: $("mn-score"), cm: $("mn-cm"),
    confusions: $("mn-confusions"), erreurs: $("mn-erreurs")
  };

  let neuronesParCouche = [128, 32, 32, 32];
  let donnees = null, modele = null, enCours = false;
  let dernier = { hist: null, epoques: 10, cm: null, erreurs: null };

  // ---------- Petits utilitaires ----------
  const pct = (x) => (100 * x).toFixed(1).replace(".", ",") + " %";
  const nombre = (x) => x.toLocaleString("fr-FR");
  function statut(txt) { el.statut.textContent = txt; }
  function hexVersRgb(hex) {
    const h = hex.replace("#", "");
    const v = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
    return [0, 2, 4].map((i) => parseInt(v.slice(i, i + 2), 16));
  }

  // ---------- Panneau de réglages ----------
  function majCouches() {
    const k = +el.couches.value;
    el.couchesVal.textContent = k;
    el.neurones.innerHTML = "";
    for (let i = 0; i < k; i++) {
      const s = document.createElement("select");
      s.setAttribute("aria-label", "Neurones de la couche cachée " + (i + 1));
      CHOIX_NEURONES.forEach((n) => {
        const o = document.createElement("option");
        o.value = n; o.textContent = n;
        if (n === neuronesParCouche[i]) o.selected = true;
        s.appendChild(o);
      });
      s.addEventListener("change", () => { neuronesParCouche[i] = +s.value; majArchi(); });
      el.neurones.appendChild(s);
    }
    majArchi();
  }

  function config() {
    return {
      type: el.modele.value,
      cachees: neuronesParCouche.slice(0, +el.couches.value),
      nConv: +el.conv.value,
      epoques: +el.epoques.value,
      nTrain: +el.taille.value,
      optim: el.optim.value
    };
  }

  // Description de l'architecture et nombre de paramètres (calculés sans TensorFlow)
  function majArchi() {
    const cfg = config();
    let texte, params = 0;
    if (cfg.type === "dense") {
      const tailles = [P].concat(cfg.cachees, [10]);
      for (let l = 1; l < tailles.length; l++) params += tailles[l - 1] * tailles[l] + tailles[l];
      texte = tailles.join(" → ");
    } else {
      let canaux = 1, cote = 28;
      const etapes = [];
      for (let i = 0; i < cfg.nConv; i++) {
        params += 9 * canaux * FILTRES[i] + FILTRES[i];
        canaux = FILTRES[i];
        cote = Math.floor(cote / 2);
        etapes.push("conv " + FILTRES[i] + " → pool");
      }
      const aplati = cote * cote * canaux;
      params += aplati * 128 + 128 + 128 * 10 + 10;
      texte = etapes.join(" → ") + " → " + aplati + " → 128 → 10";
    }
    el.archi.textContent = "Architecture : " + texte + " · " + nombre(params) + " paramètres";
  }

  function choisirModele() {
    const cnn = el.modele.value === "cnn";
    el.denseOpts.style.display = cnn ? "none" : "";
    el.cnnOpts.style.display = cnn ? "" : "none";
    el.optim.value = cnn ? "adam" : "sgd";   // les choix du notebook
    majArchi();
  }

  function verrouiller(oui) {
    [el.modele, el.optim, el.couches, el.conv, el.epoques, el.taille, el.run]
      .concat(Array.from(el.neurones.querySelectorAll("select")))
      .forEach((x) => { x.disabled = oui; });
    el.stop.disabled = !oui;
  }

  // ---------- Chargement de TensorFlow.js et des données ----------
  function chargerScript(src) {
    return new Promise((ok, ko) => {
      const s = document.createElement("script");
      s.src = src;
      s.onload = ok;
      s.onerror = () => ko(new Error("impossible de charger " + src));
      document.head.appendChild(s);
    });
  }

  async function charger() {
    if (donnees) return;
    statut("Chargement de TensorFlow.js…");
    if (!window.tf) await chargerScript(TFJS_URL);
    await tf.ready();
    statut("Chargement de l'extrait de MNIST (3 Mo)…");
    if (!window.MNIST_EXTRAIT) await chargerScript(DATA_URL);
    const m = window.MNIST_EXTRAIT;
    const chaine = atob(m.b64);
    const comprime = new Uint8Array(chaine.length);
    for (let i = 0; i < chaine.length; i++) comprime[i] = chaine.charCodeAt(i);
    const flux = new Blob([comprime]).stream().pipeThrough(new DecompressionStream("gzip"));
    const brut = new Uint8Array(await new Response(flux).arrayBuffer());
    const n = m.nTrain + m.nVal + m.nTest;
    const pixels = new Float32Array(n * P);
    for (let i = 0; i < n * P; i++) pixels[i] = brut[i] / 255;   // comme transforms.ToTensor()
    donnees = {
      nTrain: m.nTrain, nVal: m.nVal, nTest: m.nTest,
      brut: brut, pixels: pixels, etiquettes: brut.slice(n * P, n * P + n)
    };
    window.MNIST_EXTRAIT = null;
  }

  // Tenseurs (images, étiquettes one-hot) pour les images debut .. debut + nb - 1
  function tranche(debut, nb) {
    const x = tf.tensor4d(donnees.pixels.subarray(debut * P, (debut + nb) * P), [nb, 28, 28, 1]);
    const lab = Int32Array.from(donnees.etiquettes.subarray(debut, debut + nb));
    const y = tf.tidy(() => tf.oneHot(tf.tensor1d(lab, "int32"), 10));
    return { x: x, y: y };
  }

  // ---------- Le réseau ----------
  function construire(cfg) {
    const m = tf.sequential();
    if (cfg.type === "dense") {
      m.add(tf.layers.flatten({ inputShape: [28, 28, 1] }));
      cfg.cachees.forEach((n) => m.add(tf.layers.dense({ units: n, activation: "relu" })));
    } else {
      for (let i = 0; i < cfg.nConv; i++) {
        const conf = { filters: FILTRES[i], kernelSize: 3, padding: "same", activation: "relu" };
        if (i === 0) conf.inputShape = [28, 28, 1];
        m.add(tf.layers.conv2d(conf));
        m.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
      }
      m.add(tf.layers.flatten());
      m.add(tf.layers.dense({ units: 128, activation: "relu" }));
    }
    // Softmax suivi de l'entropie croisée : le même calcul que nn.CrossEntropyLoss sur les logits
    m.add(tf.layers.dense({ units: 10, activation: "softmax" }));
    m.compile({
      optimizer: cfg.optim === "adam" ? tf.train.adam(0.001) : tf.train.sgd(0.01),
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"]
    });
    return m;
  }

  // ---------- Entraînement ----------
  async function entrainer() {
    if (enCours) return;
    enCours = true;
    verrouiller(true);
    el.resultats.hidden = true;
    montrerAlerte("");
    let tr = null, va = null;
    try {
      await charger();
      const cfg = config();
      if (modele) {
        if (modele.optimizer) modele.optimizer.dispose();
        modele.dispose();
        modele = null;
      }
      tr = tranche(0, cfg.nTrain);
      va = tranche(donnees.nTrain, donnees.nVal);
      modele = construire(cfg);

      const hist = { loss: [], val_loss: [], acc: [], val_acc: [] };
      dernier = { hist: hist, epoques: cfg.epoques, cm: null, erreurs: null };
      dessinerCourbes();
      const lots = Math.ceil(cfg.nTrain / TAILLE_LOT);
      const lent = tf.getBackend() === "cpu" ? " (calcul sur le processeur : ce sera lent)" : "";
      let epoque = 0;
      const t0 = performance.now();

      await modele.fit(tr.x, tr.y, {
        epochs: cfg.epoques,
        batchSize: TAILLE_LOT,
        shuffle: true,
        validationData: [va.x, va.y],
        callbacks: {
          onEpochBegin: async (e) => { epoque = e; },
          onBatchEnd: async (lot) => {
            if (lot % 10 === 0) statut("Époque " + (epoque + 1) + "/" + cfg.epoques + " · lot " + (lot + 1) + "/" + lots + lent);
          },
          onEpochEnd: async (e, logs) => {
            hist.loss.push(logs.loss);
            hist.val_loss.push(logs.val_loss);
            hist.acc.push(logs.acc !== undefined ? logs.acc : logs.accuracy);
            hist.val_acc.push(logs.val_acc !== undefined ? logs.val_acc : logs.val_accuracy);
            dessinerCourbes();
            verifierSurapprentissage(hist);
          }
        }
      });
      const duree = (performance.now() - t0) / 1000;
      statut("Entraînement terminé en " + duree.toFixed(0) + " s. Test en cours…");
      tester(hist, duree);
    } catch (err) {
      statut("Erreur : " + err.message);
      console.error(err);
    } finally {
      [tr, va].forEach((t) => { if (t) { t.x.dispose(); t.y.dispose(); } });
      enCours = false;
      verrouiller(false);
    }
  }

  // ---------- Surapprentissage ----------
  function montrerAlerte(html) {
    el.alerte.innerHTML = html;
    el.alerte.hidden = !html;
  }

  function meilleureEpoque(hist) {
    let best = 0;
    hist.val_loss.forEach((v, i) => { if (v < hist.val_loss[best]) best = i; });
    return best;
  }

  function verifierSurapprentissage(hist) {
    const e = hist.loss.length - 1, best = meilleureEpoque(hist);
    const ecart = hist.acc[e] - hist.val_acc[e];
    if (e - best >= 2 && hist.loss[e] < hist.loss[best]) {
      montrerAlerte("<b>Surapprentissage détecté.</b> La perte de validation remonte depuis l'époque " +
        (best + 1) + ", alors que la perte d'entraînement continue de baisser : le réseau apprend " +
        "par cœur ses images d'entraînement au lieu de généraliser. Il aurait fallu s'arrêter à " +
        "l'époque " + (best + 1) + ".");
    } else if (ecart > 0.03) {
      montrerAlerte("<b>Premier signe de surapprentissage.</b> Le réseau réussit " +
        (100 * ecart).toFixed(1).replace(".", ",") + " points de mieux sur ses images " +
        "d'entraînement que sur celles de validation, qu'il n'a jamais vues.");
    } else {
      montrerAlerte("<b>Pas de surapprentissage pour l'instant.</b> Les performances sur les images " +
        "d'entraînement et de validation restent proches.");
    }
  }

  // ---------- Test final ----------
  function tester(hist, duree) {
    const debut = donnees.nTrain + donnees.nVal, n = donnees.nTest;
    const te = tranche(debut, n);
    const pred = tf.tidy(() => modele.predict(te.x, { batchSize: 256 }).argMax(-1));
    const p = pred.dataSync();
    pred.dispose(); te.x.dispose(); te.y.dispose();

    const vrai = donnees.etiquettes.subarray(debut, debut + n);
    const cm = Array.from({ length: 10 }, () => new Array(10).fill(0));
    const erreurs = [];
    let justes = 0;
    for (let i = 0; i < n; i++) {
      cm[vrai[i]][p[i]]++;
      if (vrai[i] === p[i]) justes++;
      else erreurs.push({ i: debut + i, vrai: vrai[i], predit: p[i] });
    }
    const e = hist.acc.length - 1;
    el.score.innerHTML = "Précision sur les " + nombre(n) + " images de test : <b>" + pct(justes / n) +
      "</b>" + (e >= 0 ? " <span class=\"readout\">(entraînement : " + pct(hist.acc[e]) +
      ", validation : " + pct(hist.val_acc[e]) + ", " + duree.toFixed(0) + " s)</span>" : "");

    const paires = [];
    for (let a = 0; a < 10; a++) for (let b = 0; b < 10; b++) if (a !== b && cm[a][b] > 0) paires.push([cm[a][b], a, b]);
    paires.sort((u, v) => v[0] - u[0]);
    el.confusions.innerHTML = paires.length
      ? paires.slice(0, 5).map((q) => "<li>un " + q[1] + " pris pour un " + q[2] + " : " + q[0] + " fois</li>").join("")
      : "<li>aucune erreur !</li>";

    dernier.cm = cm;
    dernier.erreurs = erreurs.slice(0, 24);
    el.resultats.hidden = false;
    dessinerCM();
    dessinerErreurs();
    statut("Terminé. Modifiez les réglages et relancez pour comparer.");
  }

  // ---------- Dessins ----------
  function dessinerCourbes() {
    const h = dernier.hist || { loss: [], val_loss: [], acc: [], val_acc: [] };
    const best = h.val_loss.length ? meilleureEpoque(h) : -1;
    courbe(el.perte, h.loss, h.val_loss, dernier.epoques, "perte", best);
    courbe(el.precision, h.acc, h.val_acc, dernier.epoques, "précision", best);
  }

  function courbe(canvas, a, b, xmax, nom, best) {
    const { ctx, w, h } = setupCanvas(canvas);
    const fg = themeColor("--fg"), muted = themeColor("--muted"), line = themeColor("--line");
    const c1 = themeColor("--accent"), c2 = themeColor("--accent-2");
    const g = 48, d = 12, t = 12, bas = 28;
    const tout = a.concat(b);
    let ymin, ymax;
    if (nom === "perte") { ymin = 0; ymax = tout.length ? Math.max.apply(null, tout) * 1.08 : 1; }
    else { ymax = 1; ymin = tout.length ? Math.max(0, Math.floor(Math.min.apply(null, tout) * 20) / 20) : 0; }
    if (ymax - ymin < 1e-6) ymax = ymin + 1;
    const X = (k) => g + (xmax > 1 ? (k / (xmax - 1)) : 0.5) * (w - g - d);
    const Y = (v) => t + (1 - (v - ymin) / (ymax - ymin)) * (h - t - bas);

    ctx.clearRect(0, 0, w, h);
    ctx.font = "11px " + themeColor("--sans");
    ctx.lineWidth = 1;
    for (let k = 0; k <= 4; k++) {
      const v = ymin + (k / 4) * (ymax - ymin), y = Y(v);
      ctx.strokeStyle = line; ctx.beginPath(); ctx.moveTo(g, y); ctx.lineTo(w - d, y); ctx.stroke();
      ctx.fillStyle = muted; ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText(nom === "perte" ? v.toFixed(2).replace(".", ",") : Math.round(100 * v) + " %", g - 6, y);
    }
    const pas = Math.max(1, Math.ceil(xmax / 10));
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let k = 0; k < xmax; k += pas) ctx.fillText(k + 1, X(k), h - bas + 6);
    ctx.textAlign = "left"; ctx.fillStyle = fg;
    ctx.fillText(nom.charAt(0).toUpperCase() + nom.slice(1), g + 4, t);

    if (best >= 0 && a.length > 1) {
      ctx.strokeStyle = muted; ctx.setLineDash([3, 4]);
      ctx.beginPath(); ctx.moveTo(X(best), t); ctx.lineTo(X(best), h - bas); ctx.stroke();
      ctx.setLineDash([]);
    }
    [[a, c1], [b, c2]].forEach(([s, col]) => {
      if (!s.length) return;
      ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2;
      ctx.beginPath();
      s.forEach((v, k) => { k ? ctx.lineTo(X(k), Y(v)) : ctx.moveTo(X(k), Y(v)); });
      ctx.stroke();
      s.forEach((v, k) => { ctx.beginPath(); ctx.arc(X(k), Y(v), 2.5, 0, 2 * Math.PI); ctx.fill(); });
    });
  }

  function dessinerCM() {
    const cm = dernier.cm;
    if (!cm) return;
    const { ctx, w, h } = setupCanvas(el.cm);
    const fg = themeColor("--fg"), muted = themeColor("--muted"), card = themeColor("--card");
    const c1 = themeColor("--accent"), c2 = themeColor("--accent-2");
    const marge = 40, taille = Math.min(w, h) - marge - 6, c = taille / 10;
    const x0 = marge + (w - marge - 6 - taille) / 2, y0 = marge;
    ctx.clearRect(0, 0, w, h);
    ctx.font = "11px " + themeColor("--sans");
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (let i = 0; i < 10; i++) {
      const total = cm[i].reduce((s, v) => s + v, 0) || 1;
      for (let j = 0; j < 10; j++) {
        const f = cm[i][j] / total;
        const fort = i === j ? f : Math.min(1, 4 * f);   // on grossit les erreurs, sinon invisibles
        ctx.globalAlpha = i === j ? 0.12 + 0.88 * f : fort;
        ctx.fillStyle = i === j ? c1 : c2;
        ctx.fillRect(x0 + j * c + 1, y0 + i * c + 1, c - 2, c - 2);
        ctx.globalAlpha = 1;
        if (cm[i][j] > 0) {
          ctx.fillStyle = (i === j ? f > 0.5 : fort > 0.6) ? card : fg;
          ctx.fillText(cm[i][j], x0 + (j + 0.5) * c, y0 + (i + 0.5) * c);
        }
      }
      ctx.fillStyle = muted;
      ctx.fillText(i, x0 + (i + 0.5) * c, y0 - 10);
      ctx.fillText(i, x0 - 10, y0 + (i + 0.5) * c);
    }
    ctx.fillStyle = fg;
    ctx.fillText("classe prédite →", x0 + taille / 2, 12);
    ctx.save();
    ctx.translate(12, y0 + taille / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText("vraie classe →", 0, 0);
    ctx.restore();
  }

  function dessinerErreurs() {
    const liste = dernier.erreurs;
    if (!liste) return;
    const largeur = el.erreurs.getBoundingClientRect().width;
    const cols = largeur < 500 ? 6 : 8, cell = largeur / cols, img = Math.min(cell * 0.62, 56);
    const ligne = img + 40, rangs = Math.max(1, Math.ceil(liste.length / cols));
    el.erreurs.style.height = rangs * ligne + "px";
    const { ctx, w, h } = setupCanvas(el.erreurs);
    const fg = hexVersRgb(themeColor("--fg")), fond = hexVersRgb(themeColor("--card"));
    const muted = themeColor("--muted"), c2 = themeColor("--accent-2");
    ctx.clearRect(0, 0, w, h);
    if (!liste.length) {
      ctx.fillStyle = muted; ctx.textAlign = "center"; ctx.fillText("Aucune erreur sur le jeu de test.", w / 2, h / 2);
      return;
    }
    const petit = document.createElement("canvas");
    petit.width = petit.height = 28;
    const pctx = petit.getContext("2d"), data = pctx.createImageData(28, 28);
    ctx.imageSmoothingEnabled = false;
    ctx.font = "11px " + themeColor("--sans");
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    liste.forEach((e, k) => {
      for (let p = 0; p < P; p++) {
        const v = donnees.brut[e.i * P + p] / 255;
        for (let c = 0; c < 3; c++) data.data[4 * p + c] = Math.round(fond[c] + v * (fg[c] - fond[c]));
        data.data[4 * p + 3] = 255;
      }
      pctx.putImageData(data, 0, 0);
      const cx = (k % cols) * cell + cell / 2, cy = Math.floor(k / cols) * ligne;
      ctx.drawImage(petit, cx - img / 2, cy + 4, img, img);
      ctx.fillStyle = c2;
      ctx.fillText("prédit " + e.predit, cx, cy + img + 7);
      ctx.fillStyle = muted;
      ctx.fillText("vrai " + e.vrai, cx, cy + img + 20);
    });
  }

  // ---------- Branchements ----------
  el.couches.addEventListener("input", majCouches);
  el.conv.addEventListener("input", () => { el.convVal.textContent = el.conv.value; majArchi(); });
  el.epoques.addEventListener("input", () => {
    el.epoquesVal.textContent = el.epoques.value;
    if (!enCours) { dernier.epoques = +el.epoques.value; if (!dernier.hist || !dernier.hist.loss.length) dessinerCourbes(); }
  });
  el.modele.addEventListener("change", choisirModele);
  el.run.addEventListener("click", entrainer);
  el.stop.addEventListener("click", () => {
    if (modele && enCours) { modele.stopTraining = true; statut("Arrêt demandé, fin du lot en cours…"); }
  });
  let minuteur = null;
  function redessiner() {
    clearTimeout(minuteur);
    minuteur = setTimeout(() => { dessinerCourbes(); dessinerCM(); dessinerErreurs(); }, 100);
  }
  window.addEventListener("resize", redessiner);
  if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redessiner);

  majCouches();
  choisirModele();
  verrouiller(false);
  window.addEventListener("load", dessinerCourbes);
})();
