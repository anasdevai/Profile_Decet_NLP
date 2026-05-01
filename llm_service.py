import os
import re
import json
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen3.5-9B")
HF_PROVIDER = os.getenv("HF_PROVIDER", "auto")
# Set HF_ENABLE_THINKING=true in .env to activate Qwen3 reasoning mode.
# When enabled the model reasons internally via <think>...</think> before answering.
HF_ENABLE_THINKING = os.getenv("HF_ENABLE_THINKING", "true").strip().lower() in {"1", "true", "yes"}

llm = None
if HF_TOKEN and HF_TOKEN != "your_hf_token_here":
    try:
        llm = InferenceClient(
            provider=HF_PROVIDER,
            api_key=HF_TOKEN,
        )
    except Exception as e:
        print("Failed to initialize Hugging Face client:", e)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output."""
    # Remove complete think blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove any dangling opening tag if model was cut off
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def _invoke_llm(prompt: str, system_prompt: str = None):
    if not llm:
        raise RuntimeError("LLM client not configured")
    # HF Inference expects a valid repo id; drop router suffixes like ":fastest".
    model_id = HF_MODEL.split(":", 1)[0].strip()
    is_qwen3 = "qwen3" in model_id.lower() or "qwen/qwen3" in model_id.lower()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Prefer chat-completion style invocation for newer instruction models.
    try:
        if hasattr(llm, "chat_completion"):
            extra = {}
            if is_qwen3 and HF_ENABLE_THINKING:
                # Qwen3 native thinking mode: model reasons in <think> blocks
                # before producing the final structured answer.
                extra["enable_thinking"] = True
            resp = llm.chat_completion(
                model=model_id,
                messages=messages,
                temperature=0.6 if (is_qwen3 and HF_ENABLE_THINKING) else 0.1,
                max_tokens=3000 if (is_qwen3 and HF_ENABLE_THINKING) else 1200,
                **extra,
            )
            if resp and getattr(resp, "choices", None):
                raw = resp.choices[0].message.content or ""
                return _strip_thinking(raw)
    except Exception as chat_exc:
        print(f"[LLM] chat_completion failed ({chat_exc}), falling back to text_generation")

    # Fallback to text generation for providers/models that expose completions.
    raw = llm.text_generation(
        prompt,
        model=model_id,
        max_new_tokens=1200,
        temperature=0.1,
    )
    return _strip_thinking(raw)


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
    if not llm:
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
    "improve":  "Improve clarity, flow, and quality",
    "rewrite":  "Completely rewrite in your own words",
    "expand":   "Expand with more detail and examples",
    "shorten":  "Compress while preserving all key points",
    "generate": "Generate new content based on context",
    "create_new": "Generate a new SOP aligned with learned profile",
    "summarize": "Summarize while preserving compliance-critical constraints",
    "gap_analysis": "Identify structure/control/traceability gaps",
    "translate": "Translate while preserving SOP control intent and IDs",
}

LLM_TASK_DIRECTIVES = {
    "rewrite": "Rewrite while preserving meaning, SOP hierarchy, and traceability.",
    "improve": "Improve quality and clarity without changing compliance intent.",
    "create_new": "Create new SOP text that follows the learned SOP profile exactly.",
    "generate": "Generate new SOP-aligned content from provided context and profile.",
    "expand": "Expand details while keeping controls, IDs, and policy intent intact.",
    "shorten": "Shorten content while preserving all compliance-critical constraints.",
    "summarize": "Summarize procedural and compliance essentials without dropping rules.",
    "gap_analysis": "Produce a gap analysis versus expected SOP structure and controls.",
    "translate": "Translate content while preserving IDs, modality, domain, and legal force.",
}

def _normalize_mode(mode: str) -> str:
    if not mode:
        return "improve"
    mode = mode.strip().lower()
    if mode == "new":
        return "create_new"
    return mode if mode in REWRITE_MODES else "improve"

def _get_dynamic_profile(style_profile: dict) -> dict:
    # Accept both older shape and new NLP structured profile shape.
    structured = style_profile.get("structured_sop_profile", {}) or style_profile.get("sop_profile", {})
    guardrails = structured.get("guardrails", style_profile.get("guardrails", {}))
    identity = structured.get("document_identity", {})
    features = style_profile.get("features", {})
    return {
        "language": guardrails.get("preserve_language") or identity.get("language", "en"),
        "domain": guardrails.get("preserve_domain") or identity.get("domain", "general"),
        "traceability": structured.get("traceability", {}),
        "workflow": structured.get("workflow", {}),
        "structure": structured.get("structure", {}),
        "style_learning": structured.get("style_learning", {}),
        "compliance": structured.get("compliance", {}),
        "guardrails": guardrails,
        "features": features,
        "genre": style_profile.get("genre", identity.get("genre", "general")),
        "tone": style_profile.get("tone", "neutral"),
        "voice": style_profile.get("voice", "active"),
        "key_style_rules": style_profile.get("key_style_rules", []),
        "do_not_change": style_profile.get("do_not_change", []),
    }

def build_llm_system_prompt(mode: str, profile: dict) -> str:
    directive = LLM_TASK_DIRECTIVES.get(mode, LLM_TASK_DIRECTIVES["improve"])
    return f"""
You are the SOP Intelligence Engine.
Task: {mode}
Directive: {directive}

Hard constraints:
- Keep output language as: {profile["language"]}
- Keep output domain as: {profile["domain"]}
- Preserve SOP structure, workflow logic, and compliance force.
- Never drop or mutate traceability IDs (SOP-*, DEV-*, CAPA-*, AUD-*, DEC-*).
- Never switch to casual/creative/slang style.
- Preserve formal, instructional, compliance-oriented tone.
- STRICT STYLE LOCK: Match the detected tone and writing style exactly; do not soften, dramatize, or simplify compliance phrasing.
- STRICT FORMAT LOCK: Preserve section order, heading intent, numbering/bullets, and procedural sequencing.
- STRICT MODALITY LOCK: Keep mandatory/prohibitive language strength (e.g., shall/must/must not) at equivalent force.
- STRICT TERMINOLOGY LOCK: Preserve regulated technical terms and abbreviations exactly unless explicitly instructed otherwise.
- STRICT SCOPE LOCK: For mode "improve", improve clarity only; do not introduce new policy intent, new controls, or new exceptions.
""".strip()

REWRITE_PROMPT_TEMPLATE = """
System Prompt:
{system_prompt}

Task Details:
- Mode: {mode}
- Genre: {genre}
- Tone: {tone}
- Voice: {voice}

Style Rules to Follow:
{style_rules}

DO NOT CHANGE:
{do_not_change}

Dynamic SOP Profile:
{dynamic_profile}

Style Mechanics:
{style_mechanics}

SOP/QMS Mechanics:
{sop_mechanics}

Legal/Firm Mechanics:
{legal_mechanics}

Reference texts for style context:
1. {reference_doc_1}
2. {reference_doc_2}

Section: {section_title}
{input_label}:
{chunk_text}

Produce output for mode "{mode}" following all hard constraints above.
If any constraint conflicts with fluency, prioritize the constraints over fluency.
"""

def rewrite_chunk(chunk_text: str, section_title: str, style_profile: dict, similar_docs: list, mode="improve"):
    if not llm:
        return f"[LLM not configured] Original: {chunk_text}"
    mode = _normalize_mode(mode)
    dynamic = _get_dynamic_profile(style_profile)

    ref1 = similar_docs[0].payload.get("chunk_text", "") if len(similar_docs) > 0 else ""
    ref2 = similar_docs[1].payload.get("chunk_text", "") if len(similar_docs) > 1 else ""
    system_prompt = build_llm_system_prompt(mode, dynamic)

    # Split system_prompt out of the user prompt so it goes into the system role.
    # The REWRITE_PROMPT_TEMPLATE user payload no longer repeats the system block.
    input_label = "Instruction" if mode in ["create_new", "generate"] else "Original Content"
    user_prompt = REWRITE_PROMPT_TEMPLATE.format(
        system_prompt="(See system message above.)",
        input_label=input_label,
        mode=mode,
        genre=dynamic.get("genre", "general"),
        tone=dynamic.get("tone", "neutral"),
        voice=dynamic.get("voice", "active"),
        style_rules="\n".join(f"- {r}" for r in dynamic.get("key_style_rules", [])),
        do_not_change="\n".join(f"- {r}" for r in dynamic.get("do_not_change", [])),
        dynamic_profile=json.dumps({
            "language": dynamic.get("language"),
            "domain": dynamic.get("domain"),
            "structure": dynamic.get("structure"),
            "workflow": dynamic.get("workflow"),
            "traceability": dynamic.get("traceability"),
            "style_learning": dynamic.get("style_learning"),
            "compliance": dynamic.get("compliance"),
            "guardrails": dynamic.get("guardrails"),
        }, ensure_ascii=False, indent=2),
        style_mechanics=json.dumps(dynamic.get("features", {}).get("poetic_style", {}), ensure_ascii=False, indent=2),
        sop_mechanics=json.dumps(dynamic.get("features", {}).get("sop_style", {}), ensure_ascii=False, indent=2),
        legal_mechanics=json.dumps(dynamic.get("features", {}).get("legal_style", {}), ensure_ascii=False, indent=2),
        reference_doc_1=ref1,
        reference_doc_2=ref2,
        section_title=section_title,
        chunk_text=chunk_text,
    )

    try:
        return _invoke_llm(user_prompt, system_prompt=system_prompt)
    except Exception as e:
        return f"[Rewrite Failed]: {e}"

def build_task_prompts(style_profile: dict, modes: list = None) -> dict:
    modes = modes or ["rewrite", "improve", "create_new"]
    dynamic = _get_dynamic_profile(style_profile)
    prompts = {}
    for mode in modes:
        m = _normalize_mode(mode)
        prompts[m] = {
            "system_prompt": build_llm_system_prompt(m, dynamic),
            "task_directive": LLM_TASK_DIRECTIVES.get(m, LLM_TASK_DIRECTIVES["improve"]),
        }
    return prompts
