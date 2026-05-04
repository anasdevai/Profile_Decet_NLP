import os
import re
import json
from huggingface_hub import InferenceClient
from google import genai
from dotenv import load_dotenv

# Try to load .env if it exists (local dev), but don't fail if it's missing (Docker)
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".env"))
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)

# Credentials
HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configuration
MODEL_ID = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
MAX_TOKENS = int(os.getenv("HF_MAX_TOKENS", "4096"))
TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.1"))

# Initialize Clients
hf_client = None
gemini_client = None

if HF_TOKEN:
    try:
        hf_client = InferenceClient(token=HF_TOKEN)
        print(f"[LLM] HuggingFace client initialized with model: {MODEL_ID}")
    except Exception as e:
        print(f"[LLM] Failed to initialize HuggingFace client: {e}")

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print(f"[LLM] Gemini client initialized with model: {GEMINI_MODEL}")
    except Exception as e:
        print(f"[LLM] Failed to initialize Gemini client: {e}")

# Global 'client' for legacy compatibility (points to HF if available)
client = hf_client


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting characters (**, #, etc.) from text."""
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    return text.strip()


def _invoke_llm(prompt: str, system_prompt: str = None):
    # Prioritize HuggingFace as the main provider
    if hf_client:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            response = hf_client.chat_completion(
                model=MODEL_ID,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            return _strip_thinking(response.choices[0].message.content)
        except Exception as e:
            print(f"[LLM] HuggingFace chat_completion failed: {e}")
            if not gemini_client: raise
            print("[LLM] Falling back to Gemini...")

    # Fallback to Gemini
    if gemini_client:
        try:
            config = {"temperature": TEMPERATURE}
            if system_prompt:
                config["system_instruction"] = system_prompt
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config
            )
            return _strip_thinking(response.text)
        except Exception as e:
            print(f"[LLM] Gemini generation failed: {e}")
            raise

    raise RuntimeError("No LLM client configured (HuggingFace and Gemini missing)")


def confirm_with_llm(text_sample: str, features: dict, prediction: dict) -> dict:
    if not gemini_client and not hf_client:
        return {
            "confirmed_genre": prediction.get("predicted_genre", "general"),
            "key_style_rules": ["Follow natural flow."],
            "do_not_change": ["Maintain structure."],
            "tone": "neutral"
        }

    user_prompt = f"""
    Analyze text sample: {text_sample[:800]}
    Features: {json.dumps(features)}
    Prediction: {json.dumps(prediction)}
    Respond ONLY in valid JSON.
    """
    system_prompt = "You are a style engine. Respond ONLY in valid JSON with keys: confirmed_genre, key_style_rules, do_not_change, tone."

    try:
        content = _invoke_llm(user_prompt, system_prompt=system_prompt)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return json.loads(content)
    except Exception:
        return {"confirmed_genre": prediction.get("predicted_genre", "general"), "tone": "neutral"}


REWRITE_PROMPT_TEMPLATE = """\
{base_user_prompt}
ID CHECKLIST: {id_checklist}
SECTION: {section_title}
CONTENT: {chunk_text}
"""

def rewrite_chunk(chunk_text: str, section_title: str, analysis_result: dict, similar_docs: list, mode="improve", **kwargs):
    if not gemini_client and not hf_client:
        return f"[LLM not configured] {chunk_text}"
    
    from nlp_pipeline import _normalize_mode
    mode = _normalize_mode(mode)
    prompts = analysis_result.get("llm_prompts", {}).get(mode, {})
    system_prompt = prompts.get("system_prompt", "SOP Intelligence Engine.")
    base_user_prompt = prompts.get("user_prompt", "Rewrite text.")

    from llm_service_utils import _extract_compliance_ids, _extract_id_lines, _verify_and_restore
    required_ids = _extract_compliance_ids(chunk_text)
    source_lines = _extract_id_lines(chunk_text, required_ids)
    id_checklist = ", ".join(required_ids) if required_ids else "None"
    
    user_prompt = REWRITE_PROMPT_TEMPLATE.format(
        base_user_prompt=base_user_prompt,
        id_checklist=id_checklist,
        section_title=section_title,
        chunk_text=chunk_text
    )

    try:
        rewritten = _invoke_llm(user_prompt, system_prompt=system_prompt)
        if required_ids and mode not in ["create_new", "generate", "summarize"]:
            rewritten = _verify_and_restore(rewritten, chunk_text, required_ids, source_lines)
        return _strip_markdown(rewritten)
    except Exception as e:
        return f"[Rewrite Failed]: {e}"

def build_task_prompts(style_profile: dict, modes: list = None) -> dict:
    from nlp_pipeline import _normalize_mode
    modes = modes or ["rewrite", "improve", "create_new"]
    return {m: style_profile.get("llm_prompts", {}).get(_normalize_mode(m), {}) for m in modes}
