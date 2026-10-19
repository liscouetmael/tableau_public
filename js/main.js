// Rendu des équations LaTeX avec KaTeX.
// Syntaxe dans le HTML :  $$ ... $$  (bloc)   et   \( ... \)  (en ligne)
document.addEventListener("DOMContentLoaded", function () {
  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\(", right: "\\)", display: false }
      ],
      throwOnError: false
    });
  }
});

// Petit utilitaire partagé par les figures : lit une couleur du thème CSS.
window.themeColor = function (name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
};

// Prépare un canvas net sur les écrans haute densité. Renvoie {ctx, w, h}.
window.setupCanvas = function (canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx: ctx, w: rect.width, h: rect.height };
};
