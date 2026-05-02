import os
import re
import json
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(env_path, override=True)

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
MAX_TOKENS = int(os.getenv("HF_MAX_TOKENS", "4096"))
TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.1"))

client = None
if HF_TOKEN:
    try:
        client = InferenceClient(token=HF_TOKEN)
    except Exception as e:
        print("Failed to initialize HuggingFace client:", e)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output."""
    # Remove complete think blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove any dangling opening tag if model was cut off
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def _invoke_llm(prompt: str, system_prompt: str = None):
    if not client:
        raise RuntimeError("LLM client not configured")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat_completion(
            model=MODEL_ID,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return _strip_thinking(response.choices[0].message.content)
    except Exception as e:
        print(f"[LLM] chat_completion failed ({e})")
        raise


CONFIRM_PROMPT_TEMPLATE = """
You are a style analysis engine. Review the following text sample and its extracted features.
The math classifier predicted the genre as: {prediction}

Extracted Features:
{features}

Text Sample:
{sample}

Respond in valid JSON format with the following keys:
- confirmed_genre: (string) The corrected or confirmed genre.
- key_style_rules: (list of strings) Key rules to reproduce this style.
- do_not_change: (list of strings) Structural or stylistic elements that MUST NOT be changed in a rewrite.
- tone: (string) The overall tone of the text.
"""

CONFIRM_SYSTEM_PROMPT = (
    "You are a style analysis engine. Respond ONLY in valid JSON with no extra text."
)

def confirm_with_llm(text_sample: str, features: dict, prediction: dict) -> dict:
    if not client:
        return {
            "confirmed_genre": prediction["predicted_genre"],
            "key_style_rules": ["Follow the natural flow of the document."],
            "do_not_change": ["Maintain the original structure where possible."],
            "tone": "neutral"
        }

    user_prompt = CONFIRM_PROMPT_TEMPLATE.format(
        sample=text_sample[:800],
        features=json.dumps(features, indent=2),
        prediction=json.dumps(prediction)
    )

    try:
        content = _invoke_llm(user_prompt, system_prompt=CONFIRM_SYSTEM_PROMPT)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return json.loads(content)
    except Exception as e:
        print("LLM Error:", e)
        return {
            "confirmed_genre": prediction["predicted_genre"],
            "key_style_rules": ["Fallback: preserve exact tone."],
            "do_not_change": ["Fallback: preserve structure."],
            "tone": "neutral"
        }

REWRITE_MODES = {
    "rewrite":  "Rewrite the SOP while preserving meaning, traceability IDs, and procedural order.",
    "improve":  "Improve clarity, consistency, and compliance quality without changing intent or control outcomes.",
    "create_new": "Generate a new SOP in the same domain/style using the learned structure and control language.",
    "summarize": "Summarize SOP controls and workflow without omitting critical compliance constraints.",
    "gap_analysis": "Identify missing sections, weak controls, and traceability gaps against the learned SOP profile.",
    "translate": "Translate the SOP while preserving IDs, compliance modality, and procedural hierarchy.",
    "expand":   "Expand with more detail and examples (fallback)",
    "shorten":  "Compress while preserving all key points (fallback)",
    "generate": "Generate new content based on context (fallback)",
}

LLM_TASK_DIRECTIVES = {
    "rewrite": "Rewrite this section while preserving meaning, traceability IDs, and procedural order.",
    "improve": "Improve the clarity and compliance of this section without changing intent or control outcomes.",
    "create_new": "Generate new content for this section in the same domain/style.",
    "summarize": "Summarize the controls and workflow in this section.",
    "gap_analysis": "Identify gaps in this section against the learned SOP profile.",
    "translate": "Translate this section while preserving IDs and procedural hierarchy.",
    "expand": "Expand details in this section while keeping controls and IDs intact.",
    "shorten": "Shorten this section while preserving all compliance-critical constraints.",
    "generate": "Generate new SOP-aligned content for this section.",
}

def _normalize_mode(mode: str) -> str:
    if not mode:
        return "improve"
    mode = mode.strip().lower()
    if mode == "new":
        return "create_new"
    return mode if mode in REWRITE_MODES else "improve"



# Regex to find all compliance IDs (DEV, CAPA, AUD, DEC, SOP, REF, etc.)
_COMPLIANCE_ID_PATTERN = re.compile(
    r'\b([A-Z]{2,8}-[A-Z]{0,4}-?\d{3,})\b'
)

def _extract_compliance_ids(text: str) -> list:
    """Extract all compliance IDs from a block of text."""
    return list(dict.fromkeys(_COMPLIANCE_ID_PATTERN.findall(text)))


def _extract_id_lines(text: str, ids: list) -> dict:
    """For each ID, extract the full line from source text."""
    id_lines = {}
    for line in text.splitlines():
        for cid in ids:
            if cid in line and cid not in id_lines:
                id_lines[cid] = line.strip()
    return id_lines


def _verify_and_restore(rewritten: str, source_text: str, required_ids: list, source_lines: dict) -> str:
    """
    Post-generation completeness check.
    Any required ID missing from the rewrite output is appended verbatim from source.
    """
    missing = [cid for cid in required_ids if cid not in rewritten]
    if not missing:
        return rewritten

    restore_block = "\n\n<!-- Compliance Integrity Restore: the following entries were missing from the rewrite and have been restored verbatim -->\n"
    for cid in missing:
        restore_block += f"\n{source_lines.get(cid, cid)}"

    print(f"[ID Guard] Restored {len(missing)} missing IDs: {missing}")
    return rewritten + restore_block


REWRITE_PROMPT_TEMPLATE = """\
{base_user_prompt}

COMPLIANCE INTEGRITY RULES (NON-NEGOTIABLE):
1. DO NOT invent any information not in the '{input_label}'.
2. EVERY ID in the REQUIRED CHECKLIST below MUST appear in your output WITH its full original description.
3. DO NOT skip, merge, or summarize any ID entry.
4. DO NOT add sections not present in the '{input_label}'.
5. If the section is short, the rewrite MUST be proportionally short.

REQUIRED COMPLIANCE ID CHECKLIST (ALL must appear in output):
{id_checklist}

Reference texts for STYLE context ONLY (Do NOT copy facts from these):
1. {reference_doc_1}
2. {reference_doc_2}

---
Section: {section_title}
{input_label}:
{chunk_text}
"""

def rewrite_chunk(chunk_text: str, section_title: str, analysis_result: dict, similar_docs: list, mode="improve", **kwargs):
    if not client:
        return f"[LLM not configured] Original: {chunk_text}"
    
    mode = _normalize_mode(mode)
    prompts = analysis_result.get("llm_prompts", {}).get(mode, {})
    
    system_prompt = prompts.get("system_prompt", "You are the SOP Intelligence Engine.")
    base_user_prompt = prompts.get("user_prompt", "Rewrite the following text.")

    ref1 = similar_docs[0].payload.get("chunk_text", "") if similar_docs and len(similar_docs) > 0 else "N/A"
    ref2 = similar_docs[1].payload.get("chunk_text", "") if similar_docs and len(similar_docs) > 1 else "N/A"
    
    input_label = "Instruction" if mode in ["create_new", "generate"] else "Original Content"
    
    # --- ID Completeness Guard ---
    required_ids = _extract_compliance_ids(chunk_text)
    source_lines = _extract_id_lines(chunk_text, required_ids)
    
    if required_ids:
        id_checklist = "\n".join(f"  - {cid}: {source_lines.get(cid, '(see source)')}" for cid in required_ids)
    else:
        id_checklist = "  (no compliance IDs in this section)"
    
    user_prompt = REWRITE_PROMPT_TEMPLATE.format(
        base_user_prompt=base_user_prompt,
        reference_doc_1=ref1,
        reference_doc_2=ref2,
        section_title=section_title,
        input_label=input_label,
        chunk_text=chunk_text,
        id_checklist=id_checklist,
    )

    try:
        rewritten = _invoke_llm(user_prompt, system_prompt=system_prompt)
        # Post-generation integrity check: restore any dropped IDs verbatim
        if required_ids and mode not in ["create_new", "generate", "summarize"]:
            rewritten = _verify_and_restore(rewritten, chunk_text, required_ids, source_lines)
        return rewritten
    except Exception as e:
        return f"[Rewrite Failed]: {e}"



def build_task_prompts(style_profile: dict, modes: list = None) -> dict:
    """Pass through the pre-built prompts from the NLP pipeline."""
    modes = modes or ["rewrite", "improve", "create_new"]
    return {m: style_profile.get("llm_prompts", {}).get(_normalize_mode(m), {}) for m in modes}
