from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
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
        + "au format JSON avec les champs: task, brand, platform, hook, script, cta."
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.7,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()

