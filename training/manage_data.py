import json
import os
import glob
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_FILE = DATA_DIR / "train.jsonl"
QUEUE_FILE = DATA_DIR / "queue_inputs.jsonl"
ARCHIVE_DIR = DATA_DIR / "archive"

def load_jsonl(file_path):
    data = []
    if not file_path.exists():
        return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data

def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def get_unique_key(entry):
    # Crée une clé unique basée sur task + brand + angle_main pour dédoublonner
    input_data = entry.get("input", entry) # Gère le cas où c'est déjà imbriqué ou plat
    return f"{input_data.get('task')}_{input_data.get('brand')}_{input_data.get('angle_main')}"

def consolidate_inputs():
    """
    Scanne tous les fichiers *briefs*.jsonl, les dédoublonne, 
    et les ajoute à queue_inputs.jsonl s'ils ne sont pas déjà dans train.jsonl
    """
    print("--- Consolidation des briefs ---")
    
    # 1. Charger l'existant (ce qui est déjà fait)
    existing_train = load_jsonl(TRAIN_FILE)
    existing_keys = set(get_unique_key(item) for item in existing_train)
    print(f"Déjà entraînés : {len(existing_train)}")

    # 2. Charger la file d'attente actuelle
    queue_data = load_jsonl(QUEUE_FILE)
    queue_keys = set(get_unique_key(item) for item in queue_data)
    print(f"Déjà en file d'attente : {len(queue_data)}")

    # 3. Trouver les nouveaux fichiers
    # On cherche tout ce qui ressemble à *brief*.jsonl ou user_*.jsonl
    # On exclut train, val, logged, queue
    all_files = glob.glob(str(DATA_DIR / "*.jsonl"))
    excluded_files = [str(TRAIN_FILE), str(DATA_DIR / "val.jsonl"), str(DATA_DIR / "logged_requests.jsonl"), str(QUEUE_FILE)]
    
    new_entries = []
    
    for file_path in all_files:
        if file_path in excluded_files:
            continue
            
        print(f"Traitement de : {os.path.basename(file_path)}")
        file_data = load_jsonl(Path(file_path))
        
        for entry in file_data:
            # Normalisation: si l'entrée est juste le brief (dict plat), on l'encapsule dans "input"
            # Si l'entrée est déjà {"input": ...}, on garde tel quel
            if "input" in entry:
                normalized_entry = entry
                brief = entry["input"]
            else:
                normalized_entry = {"input": entry}
                brief = entry
            
            key = get_unique_key(normalized_entry)
            
            # Vérification doublon global
            if key not in existing_keys and key not in queue_keys:
                new_entries.append(normalized_entry)
                queue_keys.add(key) # On l'ajoute au set pour éviter les doublons dans le même batch
    
    # 4. Sauvegarder la nouvelle queue
    if new_entries:
        total_queue = queue_data + new_entries
        save_jsonl(total_queue, QUEUE_FILE)
        print(f"Ajouté {len(new_entries)} nouveaux briefs à la file d'attente.")
        print(f"Total en attente de génération : {len(total_queue)}")
        
        # Optionnel: déplacer les fichiers traités dans archive
        ARCHIVE_DIR.mkdir(exist_ok=True)
        # (On ne le fait pas automatiquement pour l'instant pour ne pas perdre de données par erreur, 
        # mais c'est la bonne pratique)
    else:
        print("Aucun nouveau brief unique trouvé.")

if __name__ == "__main__":
    consolidate_inputs()

