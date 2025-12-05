# Déploiement sur GitHub

## Méthode rapide (3 commandes)

1. **Créez le repository sur GitHub** :
   - Allez sur https://github.com/new
   - Repository name: `marketing-ai-engine`
   - Description: `Marketing AI Engine - Fine-tuned Mistral 7B with LoRA`
   - Ne cochez **PAS** "Initialize with README"
   - Cliquez sur "Create repository"

2. **Exécutez ces commandes** :

```bash
cd "/Users/ja/Froove VS/Froove IA Marketing/marketing-ai-engine"
git remote add origin https://github.com/Froove/marketing-ai-engine.git
git branch -M main
git push -u origin main
```

3. **Vérifiez** : https://github.com/Froove/marketing-ai-engine

## Méthode automatique (avec token GitHub)

Si vous avez un Personal Access Token GitHub :

```bash
python3 create_github_repo.py --username Froove --token VOTRE_TOKEN
```

Cela créera automatiquement le repository ET poussera le code.

## Méthode script interactif

```bash
./push_to_github.sh Froove
```

Le script vous guidera étape par étape.

