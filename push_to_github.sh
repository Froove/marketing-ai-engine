#!/bin/bash

# Script pour créer et pousser le repository sur GitHub
# Usage: ./push_to_github.sh VOTRE_USERNAME [REPO_NAME]

set -e

USERNAME=${1:-""}
REPO_NAME=${2:-"marketing-ai-engine"}

if [ -z "$USERNAME" ]; then
    echo "❌ Erreur: Vous devez fournir votre username GitHub"
    echo "Usage: ./push_to_github.sh VOTRE_USERNAME [REPO_NAME]"
    exit 1
fi

echo "🚀 Création du repository GitHub: $USERNAME/$REPO_NAME"

# Vérifier si le remote existe déjà
if git remote get-url origin >/dev/null 2>&1; then
    echo "⚠️  Le remote 'origin' existe déjà. Suppression..."
    git remote remove origin
fi

# Ajouter le remote
REPO_URL="https://github.com/$USERNAME/$REPO_NAME.git"
git remote add origin "$REPO_URL"

# Vérifier la branche
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    git branch -M main
fi

echo ""
echo "📋 INSTRUCTIONS:"
echo "1. Allez sur https://github.com/new"
echo "2. Nom du repository: $REPO_NAME"
echo "3. Ne cochez PAS 'Initialize with README'"
echo "4. Cliquez sur 'Create repository'"
echo ""
echo "Une fois le repository créé, appuyez sur Entrée pour continuer..."
read -r

echo "📤 Push vers GitHub..."
git push -u origin main

echo ""
echo "✅ Repository créé et code poussé sur GitHub !"
echo "🔗 Voir sur: https://github.com/$USERNAME/$REPO_NAME"

