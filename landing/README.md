# Hooksmith AI - Landing Page

Landing page moderne pour Hooksmith AI, générateur IA de scripts TikTok & Instagram avec démo live.

## 🚀 Démarrage rapide

### Option 1 : Serveur Python simple

```bash
cd landing
python3 -m http.server 8080
```

Puis ouvrez http://localhost:8080 dans votre navigateur.

### Option 2 : Serveur Node.js (si vous avez `npx`)

```bash
cd landing
npx serve .
```

### Option 3 : Ouvrir directement

Vous pouvez aussi ouvrir `index.html` directement dans votre navigateur, mais l'API ne fonctionnera pas à cause des restrictions CORS.

## 🔧 Configuration de l'API

Par défaut, la landing page pointe vers `http://localhost:8000/generate-script`.

Pour changer l'URL de l'API, modifiez la variable `API_URL` dans `script.js` :

```javascript
const API_URL = 'http://votre-serveur:8000/generate-script';
```

## 📋 Fonctionnalités

- ✅ Hero section avec stats
- ✅ Démo live intégrée avec formulaire
- ✅ Section fonctionnalités
- ✅ Section tarifs
- ✅ Design responsive
- ✅ Intégration API `/generate-script`
- ✅ Affichage formaté des résultats (hook, script, CTA, scores)

## 🎨 Personnalisation

Les couleurs principales sont définies dans `styles.css` via les variables CSS :

```css
:root {
    --primary: #6366f1;
    --secondary: #8b5cf6;
    --gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
}
```

## 📱 Responsive

La landing page est entièrement responsive et s'adapte aux mobiles, tablettes et desktop.

## 🔗 Intégration avec l'API

La landing page appelle l'endpoint `/generate-script` de votre API FastAPI avec les paramètres suivants :

```json
{
  "brand": "Froove",
  "platform": "tiktok",
  "audience": "étudiantes 18-22 FR",
  "tone": "trend « that girl but »",
  "angle_main": "that girl mais sans argent"
}
```

L'API doit retourner un JSON avec la structure attendue (voir `api/server.py`).

