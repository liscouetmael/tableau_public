# Carnet scientifique

Site statique (HTML + CSS + un peu de JavaScript), sans framework ni étape de compilation.

## Structure

```
index.html              page d'accueil, liste des projets
a-propos.html           présentation
css/style.css           tout le style (clair/sombre automatique)
js/main.js              rendu des équations (KaTeX) + utilitaires pour les figures
js/arenstorf.js         simulation de l'orbite d'Arenstorf
js/reseau.js            réseau de neurones et démo
projets/                un fichier HTML par article
projets/_modele.html    modèle à copier pour un nouvel article
images/                 (à créer) figures exportées de Python/Matlab…
```

## Voir le site en local

Double-cliquer sur `index.html` suffit. Pour un rendu plus fidèle :

```
python -m http.server 8000
```

puis ouvrir http://localhost:8000.

## Ajouter un projet

1. Copier `projets/_modele.html` → `projets/mon-projet.html` et remplir.
2. Les figures Python : `plt.savefig("images/ma-figure.png", dpi=200, bbox_inches="tight")`
   (le format `.svg` fonctionne aussi et reste net à toutes les tailles).
3. Les équations : `\( ... \)` en ligne, `$$ ... $$` en bloc (syntaxe LaTeX).
4. Ajouter un `<li>` en haut de la liste dans `index.html`.

## Mettre en ligne (GitHub Pages, gratuit)

1. Créer un dépôt GitHub nommé `ton-pseudo.github.io`.
2. Y pousser le contenu de ce dossier.
3. Le site est en ligne à `https://ton-pseudo.github.io` après une minute.
