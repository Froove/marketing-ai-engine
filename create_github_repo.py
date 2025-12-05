#!/usr/bin/env python3
"""
Script pour créer automatiquement le repository GitHub et pousser le code.
Usage: python create_github_repo.py --username VOTRE_USERNAME --token VOTRE_TOKEN [--repo-name marketing-ai-engine]
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path

try:
    import requests
except ImportError:
    print("❌ Le module 'requests' n'est pas installé.")
    print("Installez-le avec: pip install requests")
    sys.exit(1)


def create_github_repo(username, token, repo_name, private=False):
    """Crée un repository sur GitHub via l'API"""
    url = "https://api.github.com/user/repos"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "name": repo_name,
        "description": "Marketing AI Engine - Fine-tuned Mistral 7B with LoRA for marketing content generation",
        "private": private,
        "auto_init": False  # On ne veut pas initialiser avec README
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 201:
        return response.json()["clone_url"]
    elif response.status_code == 422:
        print(f"⚠️  Le repository '{repo_name}' existe déjà sur GitHub.")
        return f"https://github.com/{username}/{repo_name}.git"
    else:
        print(f"❌ Erreur lors de la création: {response.status_code}")
        print(response.text)
        return None


def setup_git_remote(repo_url):
    """Configure le remote Git et pousse le code"""
    base_dir = Path(__file__).parent
    
    # Vérifier si on est dans un repo Git
    result = subprocess.run(["git", "rev-parse", "--git-dir"], 
                          cwd=base_dir, capture_output=True)
    if result.returncode != 0:
        print("❌ Ce n'est pas un repository Git. Exécutez 'git init' d'abord.")
        return False
    
    # Supprimer l'ancien remote s'il existe
    subprocess.run(["git", "remote", "remove", "origin"], 
                  cwd=base_dir, capture_output=True)
    
    # Ajouter le nouveau remote
    result = subprocess.run(["git", "remote", "add", "origin", repo_url],
                          cwd=base_dir, capture_output=True)
    if result.returncode != 0:
        print(f"❌ Erreur lors de l'ajout du remote: {result.stderr.decode()}")
        return False
    
    # S'assurer qu'on est sur la branche main
    subprocess.run(["git", "branch", "-M", "main"], 
                  cwd=base_dir, capture_output=True)
    
    # Push
    print("📤 Push du code vers GitHub...")
    result = subprocess.run(["git", "push", "-u", "origin", "main"],
                          cwd=base_dir)
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Crée un repository GitHub et pousse le code"
    )
    parser.add_argument("--username", required=True, help="Username GitHub")
    parser.add_argument("--token", required=True, help="Token GitHub (Personal Access Token)")
    parser.add_argument("--repo-name", default="marketing-ai-engine", 
                       help="Nom du repository (défaut: marketing-ai-engine)")
    parser.add_argument("--private", action="store_true", 
                       help="Créer un repository privé")
    
    args = parser.parse_args()
    
    print(f"🚀 Création du repository GitHub: {args.username}/{args.repo_name}")
    
    # Créer le repository
    repo_url = create_github_repo(args.username, args.token, args.repo_name, args.private)
    
    if not repo_url:
        print("❌ Impossible de créer/configurer le repository.")
        sys.exit(1)
    
    print(f"✅ Repository créé/configuré: {repo_url}")
    
    # Setup Git et push
    if setup_git_remote(repo_url):
        print(f"\n✅ Succès ! Voir sur: https://github.com/{args.username}/{args.repo_name}")
    else:
        print("\n⚠️  Le repository existe sur GitHub mais le push a échoué.")
        print(f"   Essayez manuellement: git push -u origin main")
        print(f"   URL: {repo_url}")


if __name__ == "__main__":
    main()

