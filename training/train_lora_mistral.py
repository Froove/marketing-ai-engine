import json
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# --- CONFIG GLOBALE ---

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

TRAIN_PATH = BASE_DIR / "data" / "train.jsonl"
VAL_PATH = BASE_DIR / "data" / "val.jsonl"

LORA_OUTPUT_DIR = BASE_DIR / "models" / "mistral-marketing-lora"

SYSTEM_PROMPT = (
    "Tu es un expert senior en marketing et growth, spécialisé dans les publicités "
    "short-form (TikTok, Reels, Shorts, LinkedIn). "
    "Tu génères des hooks, scripts et structures d'annonces en JSON STRICT, "
    "sans texte superflu, exactement au format demandé."
)

MAX_LENGTH = 1024  # longueur max des séquences (tokens)


# --- 1. CHARGER DATASET JSONL (train/val) ---

def load_hf_dataset():
    data_files = {
        "train": str(TRAIN_PATH),
        "validation": str(VAL_PATH),
    }
    ds = load_dataset("json", data_files=data_files)
    return ds


# --- 2. BUILD PROMPT POUR CHAQUE EXEMPLE ---

def format_example(example):
    """
    Chaque exemple est du type :
    {
      "input": {...},
      "output": {...}
    }
    On construit un prompt style instruction :

    <s>[INST] SYSTEM_PROMPT + input_json [/INST]
    output_json</s>
    """
    input_obj = example["input"]
    output_obj = example["output"]

    input_json = json.dumps(input_obj, ensure_ascii=False, indent=2)
    output_json = json.dumps(output_obj, ensure_ascii=False, indent=2)

    text = (
        "<s>[INST]\n"
        + SYSTEM_PROMPT
        + "\n\nVoici la demande marketing au format JSON :\n"
        + input_json
        + "\n[/INST]\n"
        + output_json
        + "\n</s>"
    )

    return {"text": text}


def prepare_tokenizer_and_model():
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Modèle
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )

    # Config LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return tokenizer, model


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )


def main():
    # 1. Charger le dataset HF
    print("Chargement du dataset...")
    ds = load_hf_dataset()

    # 2. Ajouter le champ "text" formaté
    print("Formatage des exemples (prompt + réponse)...")
    ds = ds.map(format_example)

    # 3. Charger tokenizer + modèle LoRA
    print("Chargement tokenizer + modèle de base + LoRA...")
    tokenizer, model = prepare_tokenizer_and_model()

    # 4. Tokenisation
    print("Tokenisation...")
    tokenized_ds = ds.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        remove_columns=[col for col in ds["train"].column_names if col != "text"],
    )

    # 5. Data collator pour LM (dynamic padding)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 6. TrainingArguments
    training_args = TrainingArguments(
        output_dir=str(LORA_OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        report_to="none",
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=data_collator,
    )

    # 8. Entraînement
    print("Démarrage du fine-tune LoRA sur Mistral 7B...")
    trainer.train()

    # 9. Sauvegarde de l'adaptateur LoRA
    print(f"Sauvegarde de l'adaptateur LoRA dans {LORA_OUTPUT_DIR} ...")
    model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)
    print("Terminé.")


if __name__ == "__main__":
    main()
