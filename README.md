# tableau_public
Mon projet de site internet perso sur lequel je mets les avancées de mes projets persos

## Carnet scientifique

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

Le dépôt est https://github.com/liscouetmael/tableau_public.

1. Une seule fois : sur GitHub, *Settings → Pages → Branch : `main` / `(root)` → Save*.
2. Après chaque modification :

   ```
   git add -A
   git commit -m "ce que j'ai changé"
   git push
   ```

3. Le site est en ligne à https://liscouetmael.github.io/tableau_public/ (mise à jour en une minute environ).
