# Marketing AI Engine

Moteur d'IA pour le marketing basé sur Mistral 7B et LoRA.

Ce projet permet de fine-tuner un modèle de langage (LLM) sur des données marketing spécifiques et de servir ce modèle via une API REST.

## Structure du projet

- `data/` : Contient les jeux de données d'entraînement et de validation au format JSONL.
- `models/` : Dossier de destination pour les poids du modèle LoRA après entraînement.
- `training/` : Scripts pour lancer le fine-tuning.
- `api/` : Serveur FastAPI pour utiliser le modèle en inférence.

## Prérequis

- Python 3.8+
- Un GPU NVIDIA (recommandé pour le fine-tuning et l'inférence rapide avec quantization).

## Installation

1.  Créer un environnement virtuel :
    ```bash
    python -m venv venv
    source venv/bin/activate  # Sur Windows: venv\Scripts\activate
    ```

2.  Installer les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

### 1. Préparation des données

Modifiez `data/train.jsonl` et `data/val.jsonl` avec vos propres exemples marketing (tweets, slogans, emails, articles de blog, etc.).

### 2. Entraînement (Fine-Tuning)

Lancez le script d'entraînement :

```bash
cd training
python train_lora_mistral.py
```

Cela va :
- Télécharger le modèle Mistral 7B (base).
- Appliquer la configuration LoRA.
- Entraîner sur vos données `data/`.
- Sauvegarder les poids légers (adaptateurs) dans `models/mistral-marketing-lora`.

### 3. Lancer l'API

Une fois l'entraînement terminé (ou même avant, pour tester le modèle de base), lancez le serveur :

```bash
cd api
python server.py
```

L'API sera accessible sur `http://localhost:8000`.

### 4. Tester l'API

Exemple de requête (avec curl) :

```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Instruction: Écris un slogan pour une agence de voyage.", "max_length": 100}'
```

## Notes

- Le chargement en 4-bit nécessite `bitsandbytes` et un GPU compatible CUDA. Si vous n'avez pas de GPU, le script essaiera de charger le modèle complet (ce qui peut demander beaucoup de RAM).
- Vous pouvez ajuster les hyperparamètres dans `training/train_lora_mistral.py`.

