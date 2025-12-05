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
    "Aucun texte hors JSON n'est autorise."
)

# Charger le modèle UNE seule fois au niveau module
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, str(LORA_PATH))
model.eval()


def extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None


def cleanup_json(json_str: str) -> str:
    """
    Supprime les lignes contenant '...' pour rendre l'objet JSON parsable.
    """
    lines = []
    for line in json_str.splitlines():
        if "..." in line:
            continue
        lines.append(line)
    return "\n".join(lines)


def generate_tiktok_script(params: dict) -> dict:
    """
    params = par ex :
    {
      "audience": "etudiantes 18-22 FR",
      "tone": "trend « that girl but »",
      "angle_main": "that girl mais sans argent"
    }
    """
    prompt = (
        SYSTEM_PROMPT
        + " Tu dois inclure les champs: task, brand, platform, hook, script, cta, variants. "
        + f"Contexte: {json.dumps(params, ensure_ascii=False)}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
        )

    raw = tokenizer.decode(output[0], skip_special_tokens=True)
    json_str = extract_first_json_object(raw)
    if not json_str:
        raise ValueError("Aucun JSON dans la sortie LoRA")

    cleaned = cleanup_json(json_str)
    return json.loads(cleaned)

