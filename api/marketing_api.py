from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import json
import time
import logging
from pathlib import Path

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Marketing AI Engine (Froove/Bridgely)")

# Configuration
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
LORA_ADAPTER_PATH = "../models/mistral-marketing-lora"
SYSTEM_PROMPT = "You are a specialized marketing AI engine. Your goal is to generate high-converting ad scripts and hooks. Output must be STRICT VALID JSON only."
USE_DUMMY_MODEL = True  # on basculera à False quand le LoRA sera prêt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_LOG_PATH = BASE_DIR / "data" / "logged_requests.jsonl"
# DATA_LOG_PATH = Path("../data/logged_requests.jsonl")


# Globals
tokenizer = None
model = None

# --- Data Models ---

class MarketingInput(BaseModel):
    task: str = Field(..., description="Ex: tiktok_script|hooks_batch|optimize")
    brand: str = Field(..., description="Ex: Froove|Bridgely")
    platform: str = Field(..., description="Ex: tiktok|instagram|linkedin")
    audience: str = Field(..., description="Description de l'audience cible.")
    tone: str = Field(..., description="Ex: culotté mais safe|anti-gourou")
    language: str = "fr"
    length_sec: int | None = None
    goal: str | None = None
    angle_main: str | None = None
    constraints: list[str] = Field(default_factory=list)
    # si tu as ça dans le fichier, rends-le OPTIONNEL :
    product_features: list[str] = Field(default_factory=list)

    additional_context: Optional[str] = None

class ScriptVariant(BaseModel):
    id: Optional[str] = None # Added id field to match dummy output
    hook: str
    script: List[Dict[str, str]] # Updated to List[Dict] based on examples
    cta: str
    scores: Dict[str, Any] # Changed to Any to accommodate string values like "high"
    explanation: Optional[Dict[str, str]] = None # Updated to match examples, optional for dummy

class MarketingOutput(BaseModel):
    variants: List[ScriptVariant]

class GenerationResponse(BaseModel):
    input_processed: MarketingInput
    output: MarketingOutput
    meta: Dict[str, Any]

# --- Core Logic ---

def dummy_output() -> dict:
    return {
        "variants": [ # Fixed: structure should match MarketingOutput (directly variants list)
                {
                    "id": "v1",
                    "hook": "Et si ton temps libre avait enfin une vraie valeur ?",
                    "script": [
                        {
                            "timing_sec": "0-3",
                            "text": "« T’as déjà rêvé d’être payée juste pour un café ? »",
                            "visual": "fille en selfie dans un café"
                        },
                        {
                            "timing_sec": "3-10",
                            "text": "« Avec Froove, tu proposes une activité, tu fixes ton prix, et tu rencontres quelqu’un en vrai. »",
                            "visual": "capture d’écran floutée de l’app"
                        }
                    ],
                    "cta": "Télécharge Froove et crée ta première sortie.",
                    "scores": {
                        "hook_power_10": 8.9,
                        "retention_level": "high",
                        "risk_level": "low"
                    }
                }
            ]
    }

def load_model_and_tokenizer():
    global tokenizer, model
    logger.info(f"Loading tokenizer: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    # FIX: Ensure pad_token is defined for generation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


    logger.info(f"Loading base model: {BASE_MODEL_NAME}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map="auto",
            load_in_4bit=True,
            torch_dtype=torch.float16
        )
        logger.info("Model loaded in 4-bit mode (GPU).")
    except Exception as e:
        logger.warning(f"Failed to load 4-bit model: {e}. Falling back to standard load.")
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    adapter_path = os.path.join(script_dir, LORA_ADAPTER_PATH)

    if os.path.exists(adapter_path):
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        try:
            model = PeftModel.from_pretrained(base_model, adapter_path)
            logger.info("LoRA adapter loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading LoRA adapter: {e}")
            model = base_model
    else:
        logger.warning(f"LoRA adapter not found at {adapter_path}. Using base model.")
        model = base_model

def build_prompt_from_input(input_data: MarketingInput) -> str:
    input_dict = input_data.dict()
    input_json_str = json.dumps(input_dict, ensure_ascii=False)
    return f"<s>[INST] {SYSTEM_PROMPT}\n\nINPUT CONFIG:\n{input_json_str} [/INST]"

async def generate_marketing_content(req: MarketingInput):
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model not initialized")

    full_prompt = build_prompt_from_input(req)

    # Tokenize
    inputs = tokenizer(full_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    # Generate
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    # Decode
    generated_text_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract logic
    if "[/INST]" in generated_text_full:
        generated_content = generated_text_full.split("[/INST]")[-1].strip()
    else:
        # Fallback based on length of prompt (less reliable but needed if tokens skipped)
        prompt_len_char = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
        generated_content = generated_text_full[prompt_len_char:].strip()

    # JSON cleanup
    start_brace = generated_content.find("{")
    end_brace = generated_content.rfind("}")
    
    if start_brace != -1 and end_brace != -1:
        json_str = generated_content[start_brace:end_brace+1]
        try:
            output_data = json.loads(json_str)
            return MarketingOutput(**output_data)
        except Exception as e:
            logger.error(f"JSON parse error: {e} | Content: {json_str}")
            raise HTTPException(status_code=500, detail="Invalid JSON generated")
    else:
         raise HTTPException(status_code=500, detail="No JSON found in response")

@app.on_event("startup")
async def startup_event():
    if not USE_DUMMY_MODEL:
        load_model_and_tokenizer()

@app.post("/generate-marketing", response_model=GenerationResponse)
async def endpoint_generate_marketing(req: MarketingInput):
    start_time = time.time()
    
    if USE_DUMMY_MODEL:
        # On loguera plus tard le input pour s'en servir comme dataset
        dummy_data = dummy_output()
        output_obj = MarketingOutput(**dummy_data)

        # Log pour futur fine-tune : même schéma que train.jsonl
        log_entry = {
            "timestamp": time.time(),
            "input": req.dict(),
            "output": dummy_data
        }
        DATA_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DATA_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    else:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Le modèle n'est pas encore chargé.")
        try:
            output_obj = await generate_marketing_content(req)
        except HTTPException:
            raise
        except Exception as e:
            print(f"Erreur inattendue lors de la génération: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur interne du moteur IA: {str(e)}")

    process_time = time.time() - start_time
    return {
        "input_processed": req,
        "output": output_obj,
        "meta": {
            "model_version": "mistral-7b-instruct-v0.3-lora" if not USE_DUMMY_MODEL else "dummy-model",
            "latency_seconds": round(process_time, 2)
        }
    }

@app.get("/health")
def health_check():
    return {"status": "active", "gpu": torch.cuda.is_available(), "mode": "dummy" if USE_DUMMY_MODEL else "production"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
