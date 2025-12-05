from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import json
from pathlib import Path

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Chemin absolu vers le dossier du projet
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LORA_PATH = PROJECT_ROOT / "models" / "mistral-marketing-lora"

SYSTEM_PROMPT = (
    "Tu es un expert marketing TikTok. "
    "Tu reponds strictement avec un objet JSON unique, valide et complet. "
    "Aucun texte hors JSON n'est autorise. "
    "Tu dois inclure dans cet objet les champs: task, brand, platform, hook, script, cta, variants."
)


def extract_first_json_object(text: str) -> str | None:
    """Retourne la sous-chaîne correspondant au premier objet JSON { ... } de niveau 0."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    end = None
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None
    return text[start:end]


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, str(LORA_PATH))
    model.eval()

    prompt = (
        SYSTEM_PROMPT
        + " Genere un script TikTok pour Froove en francais, "
        "au format JSON unique avec les champs: task, brand, platform, "
        "hook, script, cta, variants."
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=700,   # on laisse plus de place pour fermer le JSON
            do_sample=True,
            temperature=0.5,      # un peu moins de blabla
        )

    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("==== RAW OUTPUT ====")
    print(raw)
    print("==== /RAW OUTPUT ====")

    json_str = extract_first_json_object(raw)
    if json_str is None:
        print("[ERREUR] Aucun JSON detecte.")
        return

    try:
        obj = json.loads(json_str)
    except Exception as e:
        print("[ERREUR] Echec du parsing JSON :", e)
        print("JSON brut :")
        print(json_str)
        return

    print("==== JSON PARSE ====")
    print(json.dumps(obj, ensure_ascii=False, indent=2))
    print("==== /JSON PARSE ====")


if __name__ == "__main__":
    main()
