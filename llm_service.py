import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

llm = None
if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=GEMINI_API_KEY
        )
    except Exception as e:
        print("Failed to initialize Gemini:", e)

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

def confirm_with_llm(text_sample: str, features: dict, prediction: dict) -> dict:
    if not llm:
        return {
            "confirmed_genre": prediction["predicted_genre"],
            "key_style_rules": ["Follow the natural flow of the document."],
            "do_not_change": ["Maintain the original structure where possible."],
            "tone": "neutral"
        }
    
    prompt = CONFIRM_PROMPT_TEMPLATE.format(
        sample=text_sample[:800],
        features=json.dumps(features, indent=2),
        prediction=json.dumps(prediction)
    )
    
    try:
        response = llm.invoke(prompt)
        content = response.content
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
}

REWRITE_PROMPT_TEMPLATE = """
Task: {mode}
Genre: {genre}
Tone: {tone}
Voice: {voice}

Style Rules to Follow:
{style_rules}

DO NOT CHANGE:
{do_not_change}

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
Original Content:
{chunk_text}

Rewrite the original content strictly applying the style rules and tone above.
If the source is poetic or lineated, preserve line breaks, rhythm, rhyme/slant-rhyme,
compressed metaphor, and emotional delivery. Do not turn symbolic poetry into
explanatory prose unless the requested mode explicitly asks for explanation.
If the source is SOP/QMS, compliance, legal, audit, deviation, CAPA, or decision
content, do the opposite: preserve formal headings, traceability IDs, bullets,
numbered steps, defined terms, exact technical terminology, time limits, approvals,
retention periods, and constraints. Never convert SOP/QMS/legal content into rhyme,
poetry, metaphor, mnemonic verse, or narrative prose. Never drop IDs such as SOP-*,
DEV-*, CAPA-*, AUD-*, or DEC-*.

For German SOP/QMS source text, preserve regulated source precision:
- "Log nach 1h" means activity/logging checkpoint after 1 hour, not logout.
- "SPS" must be preserved as "SPS (PLC)" on first English use, not replaced by PLC only.
- Preserve 24-hour times such as 22:30 and 06:00-20:00.
- "15 Min Sperre" means 15-minute lockout.
- Keep KI/AI terminology consistent and never write phrases like "AI's line".
- Do not add empty sections such as "Responsibilities: No content provided" unless the original has that section.
"""

def rewrite_chunk(chunk_text: str, section_title: str, style_profile: dict, similar_docs: list, mode="improve"):
    if not llm:
        return f"[LLM not configured] Original: {chunk_text}"
        
    ref1 = similar_docs[0].payload.get("chunk_text", "") if len(similar_docs) > 0 else ""
    ref2 = similar_docs[1].payload.get("chunk_text", "") if len(similar_docs) > 1 else ""
    
    prompt = REWRITE_PROMPT_TEMPLATE.format(
        mode=REWRITE_MODES.get(mode, "Improve clarity"),
        genre=style_profile.get("genre", "general"),
        tone=style_profile.get("tone", "neutral"),
        voice=style_profile.get("voice", "active"),
        style_rules="\n".join(f"- {r}" for r in style_profile.get("key_style_rules", [])),
        do_not_change="\n".join(f"- {r}" for r in style_profile.get("do_not_change", [])),
        style_mechanics=json.dumps(style_profile.get("features", {}).get("poetic_style", {}), indent=2),
        sop_mechanics=json.dumps(style_profile.get("features", {}).get("sop_style", {}), indent=2),
        legal_mechanics=json.dumps(style_profile.get("features", {}).get("legal_style", {}), indent=2),
        reference_doc_1=ref1,
        reference_doc_2=ref2,
        section_title=section_title,
        chunk_text=chunk_text,
    )
    
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"[Rewrite Failed]: {e}"
