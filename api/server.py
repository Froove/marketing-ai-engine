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

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Marketing AI Engine (Froove/Bridgely)")

# Configuration
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
LORA_ADAPTER_PATH = "../models/mistral-marketing-lora"
SYSTEM_PROMPT = "You are a specialized marketing AI engine. Your goal is to generate high-converting ad scripts and hooks. Output must be STRICT VALID JSON only."

# Globals
tokenizer = None
model = None

# --- Data Models ---

class MarketingInput(BaseModel):
    task: str = Field(..., example="create_ad_script")
    brand: str = Field(..., example="Froove")
    platform: str = Field(..., example="TikTok")
    audience: str = Field(..., example="Gen Z students")
    tone: str = Field(..., example="energetic")
    product_features: List[str] = Field(..., example=["AI note taking", "saves time"])
    additional_context: Optional[str] = None

class ScriptVariant(BaseModel):
    hook: str
    script: Dict[str, str]
    cta: str
    scores: Dict[str, float]
    explanation: str

class MarketingOutput(BaseModel):
    variants: List[ScriptVariant]

class GenerationResponse(BaseModel):
    input_processed: MarketingInput
    output: MarketingOutput
    meta: Dict[str, Any]

# --- Core Logic ---

def load_models():
    global tokenizer, model
    logger.info(f"Loading tokenizer: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

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

@app.on_event("startup")
async def startup_event():
    load_models()

@app.post("/generate-marketing", response_model=GenerationResponse)
async def generate_marketing(req: MarketingInput):
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model not initialized")

    start_time = time.time()
    
    # 1. Construct Prompt (Matching Training Format)
    # Convert input pydantic model to dict, then to json string
    input_dict = req.dict()
    input_json_str = json.dumps(input_dict, ensure_ascii=False)
    
    full_prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\nINPUT CONFIG:\n{input_json_str} [/INST]"

    # 2. Tokenize
    inputs = tokenizer(full_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    # 3. Generate
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024, # Enough for full JSON
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    # 4. Decode & Parse
    generated_text_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extraction robuste du JSON après la balise [/INST] (si présente dans le décodage) ou à la fin
    # On cherche le premier '{' qui suit la fin du prompt système ou on prend tout si on ne trouve pas de marqueur
    
    # Stratégie 1 : Chercher le pattern [/INST] si le tokenizer le garde (souvent non avec skip_special_tokens=True)
    # Stratégie 2 : Chercher le premier '{' après une certaine position, ou simplement le premier '{' du texte généré qui semble être le début de la réponse.
    
    # Le plus simple et robuste : On sait que la réponse est un JSON. On cherche la première occurrence de '{' 
    # APRES l'input config (qui contient aussi du JSON).
    # L'input config finit par 'INPUT CONFIG:\n... } [/INST]'
    
    # On va chercher la dernière occurrence de [/INST] dans le texte brut si possible, sinon on fait une recherche de JSON.
    
    # Amélioration : On utilise le texte généré complet et on cherche le JSON de sortie.
    # Comme l'input contient aussi du JSON, il faut être prudent.
    # Le prompt finit par "[/INST]". On peut essayer de split dessus.
    
    if "[/INST]" in generated_text_full:
        generated_content = generated_text_full.split("[/INST]")[-1].strip()
    else:
        # Si le tag a sauté, on assume que le modèle a généré la suite.
        # Risque : confondre avec le JSON de l'input.
        # On va essayer de parser le dernier objet JSON valide de la chaîne.
        generated_content = generated_text_full[prompt_len:].strip()

    # Nettoyage final pour ne garder que le JSON (au cas où le modèle blablate après)
    start_brace = generated_content.find("{")
    end_brace = generated_content.rfind("}")
    
    if start_brace != -1 and end_brace != -1:
        generated_content = generated_content[start_brace:end_brace+1]
    else:
        # Fallback: si on n'a pas trouvé de bon split, on cherche dans tout le texte le DERNIER gros bloc {...}
        # C'est risqué mais mieux que rien.
        pass
    
    # 5. JSON Parsing & Validation
    try:
        output_data = json.loads(generated_content)
        # Validate against output model
        validated_output = MarketingOutput(**output_data)
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON. Raw output: {generated_content}")
        # Fallback or Error
        raise HTTPException(status_code=500, detail="Model generated invalid JSON. Try again.")
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=f"Output validation failed: {e}")

    process_time = time.time() - start_time
    
    return {
        "input_processed": req,
        "output": validated_output,
        "meta": {
            "model_version": "mistral-7b-instruct-v0.3-lora",
            "latency_seconds": round(process_time, 2)
        }
    }

@app.get("/health")
def health_check():
    return {"status": "active", "gpu": torch.cuda.is_available()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
