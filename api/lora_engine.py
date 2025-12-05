# api/lora_engine.py
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
    "Tu dois inclure les champs: task, brand, platform, hook, script, cta, variants."
)

# === Chargement du modèle UNE FOIS au démarrage ===
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, str(LORA_PATH))
model.eval()


def _extract_first_json_object(text: str) -> str | None:
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


def generate_tiktok_script(params: dict) -> dict:
    """
    params peut contenir : audience, tone, angle_main, etc.
    """
    prompt = (
        SYSTEM_PROMPT
        + " Contexte campagne: "
        + json.dumps(params, ensure_ascii=False)
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
    json_str = _extract_first_json_object(raw)
    if not json_str:
        raise ValueError("Aucun JSON valide detecte dans la sortie du modele.")

    obj = json.loads(json_str)
    return obj
