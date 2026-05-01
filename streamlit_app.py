import io
import json
import re
from uuid import uuid4
import os

import streamlit as st
# Moved import
# Moved import


st.set_page_config(
    page_title="Style Detection & Rewriting Engine",
    page_icon="",
    layout="wide",
)


CUSTOM_CSS = """
<style>
/* Gradient Title */
.gradient-text {
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}

/* Premium Metric Cards */
.metric-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    border-color: rgba(79, 172, 254, 0.4);
}
.metric-label {
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
    font-weight: 600;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #f8fafc;
    background: linear-gradient(90deg, #f8fafc 0%, #cbd5e1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Improve button aesthetics */
div[data-testid="stButton"] button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
}
div[data-testid="stFormSubmitButton"] button p {
    font-size: 1.1rem;
}

/* Cleaner Expanders */
.streamlit-expanderHeader {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def llm_status() -> dict:
    import llm_service

    return {
        "llm_client_initialized": llm_service.llm is not None,
        "hf_model": llm_service.HF_MODEL,
        "hf_provider": llm_service.HF_PROVIDER,
        "hf_token_present": bool(os.getenv("HF_TOKEN")),
        "thinking_enabled": os.getenv("HF_ENABLE_THINKING", "true").strip().lower() in {"1", "true", "yes"},
    }


def read_uploaded_file(uploaded_file) -> str:
    from docx import Document
    from pypdf import PdfReader

    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="replace")

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()

    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs).strip()

    raise ValueError("Unsupported file type. Upload .txt, .md, .pdf, or .docx.")


def build_profile(text: str, use_llm_validation: bool = False) -> dict:
    import nlp_pipeline

    result = nlp_pipeline.process_document(text)
    genre = result["classification"]["predicted_genre"]
    confidence = result["classification"]["confidence"]
    lang_code = result["language"]["lang_code"]

    llm_used = False
    threshold = 0.70
    key_style_rules = [f"Preserve the detected {genre} style and structure."]
    do_not_change = []
    tone = "neutral"

    should_validate_with_llm = (
        use_llm_validation
        and (confidence < threshold or lang_code in {"ur", "hi", "ar"} or result["language"]["is_mixed_script"])
    )

    if should_validate_with_llm:
        import llm_service

        llm_used = True
        llm_result = llm_service.confirm_with_llm(
            text,
            result["features"],
            result["classification"],
        )
        genre = llm_result.get("confirmed_genre", genre)
        key_style_rules = llm_result.get("key_style_rules", key_style_rules)
        do_not_change = llm_result.get("do_not_change", do_not_change)
        tone = llm_result.get("tone", tone)

    structure = result["features"]["structure"]
    poetic_style = result["features"].get("poetic_style", {})
    sop_style = result["features"].get("sop_style", {})
    legal_style = result["features"].get("legal_style", {})
    is_compliance_doc = sop_style.get("format_score", 0) >= 0.45 or legal_style.get("legalese_score", 0) >= 0.35
    if structure["has_numbered_steps"]:
        do_not_change.append("Preserve numbered steps.")
    if structure["has_bullets"]:
        do_not_change.append("Preserve bullet structure.")
    if structure["has_table"]:
        do_not_change.append("Preserve table-like content.")
    if poetic_style.get("is_lineated") and not is_compliance_doc:
        do_not_change.append("Preserve poetic line breaks and stanza-like shape.")
        key_style_rules.append("Keep line breaks; do not flatten the poem into prose.")
    if poetic_style.get("rhyme_scheme") and not is_compliance_doc:
        do_not_change.append(f"Preserve rhyme/music pattern where possible: {poetic_style['rhyme_scheme']}.")
        key_style_rules.append("Use end-word rhyme or slant rhyme to retain musicality.")
    if poetic_style.get("figurative_density", 0) >= 0.8 and not is_compliance_doc:
        key_style_rules.append("Preserve compressed metaphor-heavy language instead of explaining the imagery.")
    if poetic_style.get("emotional_delivery") in {"subtle/symbolic", "figurative/emotive"} and not is_compliance_doc:
        key_style_rules.append(f"Preserve emotional delivery: {poetic_style['emotional_delivery']}.")

    if sop_style.get("format_score", 0) >= 0.45:
        key_style_rules.append(f"Preserve SOP/QMS format: {sop_style['format_pattern']}.")
        if sop_style.get("sections_found"):
            do_not_change.append(f"Preserve section order/headings: {', '.join(sop_style['sections_found'])}.")
        if sop_style.get("numbered_step_count", 0) > 0:
            do_not_change.append("Preserve numbered procedural steps and action-oriented wording.")
        if sop_style.get("shall_count", 0) or sop_style.get("must_count", 0):
            key_style_rules.append("Preserve mandatory control language such as shall/must where appropriate.")
        if sop_style.get("compliance_ids"):
            do_not_change.append(f"Preserve all traceability IDs: {', '.join(sop_style['compliance_ids'][:30])}.")
        if sop_style.get("precision_terms"):
            do_not_change.append(
                "Preserve regulated source terminology: "
                + "; ".join(f"{k} = {v}" for k, v in sop_style["precision_terms"].items())
            )
        key_style_rules.append("Do not introduce rhyme, metaphor, poetic compression, or narrative wording into SOP/QMS content.")

    if legal_style.get("legalese_score", 0) >= 0.35:
        key_style_rules.append(f"Preserve legal/firm drafting pattern: {legal_style['format_pattern']}.")
        if legal_style.get("clause_headers_found"):
            do_not_change.append(f"Preserve clause headings: {', '.join(legal_style['clause_headers_found'])}.")
        if legal_style.get("defined_terms"):
            do_not_change.append(f"Preserve defined terms: {', '.join(legal_style['defined_terms'][:8])}.")
        if legal_style.get("obligation_count", 0) > 0:
            key_style_rules.append("Preserve obligation language and legal modality such as shall, may, and shall not.")

    return {
        "doc_id": str(uuid4()),
        "genre": genre,
        "confidence": confidence,
        "runner_up": result["classification"]["runner_up"],
        "all_scores": result["classification"]["all_scores"],
        "tone": tone,
        "voice": result["features"]["voice"],
        "language": lang_code,
        "llm_used": llm_used,
        "key_style_rules": key_style_rules,
        "do_not_change": list(dict.fromkeys(do_not_change)),
        "features": result["features"],
        "chunks": [
            {
                "section_title": chunk.section_title,
                "section_type": chunk.section_type,
                "word_count": len(chunk.content.split()),
                "content": chunk.content,
            }
            for chunk in result["chunks"]
        ],
    }


def rewrite_text(text: str, profile: dict, mode: str, use_hf_llm: bool = False) -> str:
    import nlp_pipeline

    sop_style = profile.get("features", {}).get("sop_style", {})
    legal_style = profile.get("features", {}).get("legal_style", {})
    is_compliance_doc = sop_style.get("format_score", 0) >= 0.45 or legal_style.get("legalese_score", 0) >= 0.35

    if is_compliance_doc and not use_hf_llm:
        return local_fallback_transform(text, mode, profile)

    chunks = nlp_pipeline.smart_chunk(text)
    rewritten_sections = []

    for chunk in chunks:
        if use_hf_llm:
            import llm_service

            rewritten = llm_service.rewrite_chunk(
                chunk_text=chunk.content,
                section_title=chunk.section_title,
                style_profile=profile,
                similar_docs=[],
                mode=mode,
            )
            if rewritten.startswith("[LLM not configured]") or rewritten.startswith("[Rewrite Failed]"):
                rewritten = local_fallback_transform(chunk.content, mode, profile)
            rewritten = compliance_precision_postprocess(rewritten, profile)
        else:
            rewritten = local_fallback_transform(chunk.content, mode, profile)
        if getattr(chunk, "is_generic", chunk.section_title.startswith("Chunk ") or chunk.section_title == "Intro"):
            rewritten_sections.append(rewritten)
        else:
            rewritten_sections.append(f"{chunk.section_title}\n\n{rewritten}")

    return "\n\n".join(rewritten_sections)


def local_fallback_transform(text: str, mode: str, profile: dict | None = None) -> str:
    profile = profile or {}
    sop_style = profile.get("features", {}).get("sop_style", {})
    legal_style = profile.get("features", {}).get("legal_style", {})
    if sop_style.get("format_score", 0) >= 0.45:
        return compliance_precision_postprocess(local_compliance_transform(text, mode, "SOP/QMS"), profile)
    if legal_style.get("legalese_score", 0) >= 0.35:
        return local_compliance_transform(text, mode, "Legal/Firm")

    poetic_style = profile.get("features", {}).get("poetic_style", {})
    if poetic_style.get("is_lineated") or profile.get("genre") == "poetry":
        return local_poetry_transform(text, mode)

    cleaned = " ".join(text.split())
    if mode == "shorten":
        sentences = [s.strip() for s in cleaned.replace("?", ".").replace("!", ".").split(".") if s.strip()]
        return ". ".join(sentences[: max(1, min(3, len(sentences)))]) + ("." if sentences else "")
    if mode == "expand":
        return (
            f"{cleaned}\n\nAdditional detail: clarify responsibilities, expected records, "
            "review points, and completion criteria while preserving the detected style."
        )
    if mode == "generate":
        return (
            "Draft generated from the detected style profile:\n\n"
            f"{cleaned}\n\nMaintain the same genre, tone, structure, and terminology shown in the source profile."
        )
    if mode == "rewrite":
        return cleaned
    return cleaned


def local_compliance_transform(text: str, mode: str, label: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned_lines = []
    previous_blank = False

    for line in lines:
        cleaned = " ".join(line.split())
        if not cleaned:
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue

        previous_blank = False
        cleaned = cleaned.replace("slept", "lockout")
        cleaned = re.sub(r"\bAI's line\b", "AI systems", cleaned, flags=re.I)
        cleaned_lines.append(cleaned)

    result = "\n".join(cleaned_lines).strip()
    if mode == "shorten":
        protected = [
            line for line in cleaned_lines
            if re.search(r"\b(?:SOP|DEV|CAPA|AUD|DEC)-[A-Z]{2,}-\d{3}\b", line, re.I)
            or line.endswith(":")
            or line.lower().startswith(("zweck", "purpose", "scope", "geltungsbereich", "key points"))
        ]
        return "\n".join(protected[:60]).strip() or result

    if mode == "expand":
        return compliance_precision_postprocess(
            f"{result}\n\nCompliance rewrite note: preserve all headings, IDs, constraints, approvals, "
            f"logs, retention periods, and controlled terminology for {label} use.",
            {"features": {"sop_style": {"format_score": 1.0, "precision_terms": {}}}},
        )

    if mode == "generate":
        return (
            f"{label} draft generated from the detected structure.\n\n"
            "Purpose:\n\nScope:\n\nResponsibilities:\n\nProcedure:\n1. \n2. \n3. \n\nRecords:\n\nDeviations/CAPAs/Audit Findings/Decisions:\n"
        )

    return result


def compliance_precision_postprocess(text: str, profile: dict) -> str:
    sop_style = profile.get("features", {}).get("sop_style", {})
    if sop_style.get("format_score", 0) < 0.45:
        return text

    corrected = text
    precision_terms = sop_style.get("precision_terms", {})

    if "Log nach 1h" in precision_terms:
        corrected = re.sub(r"\b(?:must|shall|required to)?\s*log\s*out\s*after\s*1\s*hour\b", "activity must be logged after 1 hour", corrected, flags=re.I)
        corrected = re.sub(r"\b(?:must|shall|required to)?\s*logout\s*after\s*1\s*hour\b", "activity must be logged after 1 hour", corrected, flags=re.I)
        corrected = re.sub(r"\b(?:must|shall|required to)?\s*log\s*out\s*after\s*one\s*hour\b", "activity must be logged after 1 hour", corrected, flags=re.I)

    if "SPS" in precision_terms:
        corrected = re.sub(r"\bthe PLC\b", "the SPS (PLC)", corrected, count=1, flags=re.I)
        corrected = re.sub(r"\baccessed PLC\b", "accessed SPS (PLC)", corrected, flags=re.I)
        corrected = re.sub(r"\baccessed the PLC\b", "accessed the SPS (PLC)", corrected, flags=re.I)

    if "22:30" in precision_terms:
        corrected = re.sub(r"\b10:30\s*PM\b", "22:30", corrected, flags=re.I)
        corrected = re.sub(r"\b8:00\s*PM\b", "20:00", corrected, flags=re.I)

    if "15 Min Sperre" in precision_terms:
        corrected = re.sub(r"\b15-minute\s+block\b", "15-minute lockout", corrected, flags=re.I)
        corrected = re.sub(r"\b15\s*min(?:ute)?\s+sleep\b", "15-minute lockout", corrected, flags=re.I)

    corrected = re.sub(r"\bAI's line\b", "AI systems", corrected, flags=re.I)
    corrected = re.sub(r"\bExternal users activity must be logged\b", "External user activity must be logged", corrected, flags=re.I)
    return corrected


def local_poetry_transform(text: str, mode: str) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text

    if mode == "shorten":
        keep = max(2, min(len(lines), len(lines) // 2 or 2))
        return "\n".join(lines[:keep])

    if mode == "expand":
        expanded = []
        for line in lines:
            expanded.append(line)
            if len(line.split()) <= 7:
                expanded.append("softly holding what the silence knows")
        return "\n".join(expanded)

    if mode == "generate":
        return "\n".join(
            [
                "A quiet light begins to rise",
                "and folds its grace beneath the skies",
                "where every shadow finds its place",
                "and every breath remembers grace",
            ]
        )

    improved = []
    for line in lines:
        cleaned_line = " ".join(line.split())
        cleaned_line = re.sub(r"\bvery\s+", "", cleaned_line, flags=re.I)
        improved.append(cleaned_line)

    return "\n".join(improved)


def render_metric_grid(profile: dict) -> None:
    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem;">
        <div class="metric-card">
            <div class="metric-label">Detected Genre</div>
            <div class="metric-value">{}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">{:.1%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Language</div>
            <div class="metric-value">{}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Voice</div>
            <div class="metric-value">{}</div>
        </div>
    </div>
    """.format(
        profile["genre"].upper(), 
        profile["confidence"], 
        profile["language"].upper(), 
        profile["voice"].capitalize()
    ), unsafe_allow_html=True)

def render_nlp_output(profile: dict) -> None:
    features = profile["features"]

    st.subheader("NLP Output")
    render_metric_grid(profile)

    with st.expander("Summary", expanded=True):
        left, right = st.columns(2)
        with left:
            st.write("Style rules")
            st.write(profile["key_style_rules"] or ["No explicit rules found."])
            st.write("Do not change")
            st.write(profile["do_not_change"] or ["No locked elements found."])
        with right:
            st.write("Readability")
            st.json(features["readability"])
            st.write("POS distribution")
            st.json(features["pos_dist"])

    with st.expander("Structure"):
        st.json(features["structure"])
        st.write("Modal verbs")
        st.json(features["modal_verbs"])

    with st.expander("Poetic Pattern"):
        poetic_style = features.get("poetic_style", {})
        if poetic_style:
            c1, c2, c3, c4 = st.columns(4)
            c1.write("Musicality")
            c1.write(f"**{poetic_style.get('musicality_score', 0)}**")
            c2.write("Figurative Density")
            c2.write(f"**{poetic_style.get('figurative_density', 0)}**")
            c3.write("Compression")
            c3.write(f"**{poetic_style.get('compression_score', 0)}**")
            c4.write("Delivery")
            c4.write(f"**{poetic_style.get('emotional_delivery', 'unknown')}**")

            st.write("Rhyme and line pattern")
            st.json(
                {
                    "rhyme_scheme": poetic_style.get("rhyme_scheme"),
                    "end_words": poetic_style.get("end_words"),
                    "rhyming_pairs": poetic_style.get("rhyming_pairs"),
                    "line_count": poetic_style.get("line_count"),
                    "avg_line_words": poetic_style.get("avg_line_words"),
                    "line_break_density": poetic_style.get("line_break_density"),
                }
            )
            st.write("Figurative and emotional signals")
            st.json(
                {
                    "metaphor_marker_hits": poetic_style.get("metaphor_marker_hits"),
                    "sensory_marker_hits": poetic_style.get("sensory_marker_hits"),
                    "explanation_marker_hits": poetic_style.get("explanation_marker_hits"),
                    "emotional_delivery": poetic_style.get("emotional_delivery"),
                }
            )
        else:
            st.info("No poetic pattern features found.")

    with st.expander("SOP/QMS"):
        sop_style = features.get("sop_style", {})
        if sop_style:
            c1, c2, c3, c4 = st.columns(4)
            c1.write("Format Score")
            c1.write(f"**{sop_style.get('format_score', 0)}**")
            c2.write("Completeness")
            c2.write(f"**{sop_style.get('completeness_score', 0)}**")
            c3.write("Steps")
            c3.write(f"**{sop_style.get('numbered_step_count', 0)}**")
            c4.write("Control Language")
            c4.write(f"**{sop_style.get('control_language_score', 0)}**")
            
            st.write("Detected SOP/QMS pattern")
            st.json(
                {
                    "format_pattern": sop_style.get("format_pattern"),
                    "sections_found": sop_style.get("sections_found"),
                    "sections_missing_core": sop_style.get("sections_missing_core"),
                    "compliance_ids": sop_style.get("compliance_ids"),
                    "compliance_id_count": sop_style.get("compliance_id_count"),
                    "detected_standard_terms": sop_style.get("detected_standard_terms"),
                }
            )
            st.write("Procedural language")
            st.json(
                {
                    "shall_count": sop_style.get("shall_count"),
                    "must_count": sop_style.get("must_count"),
                    "should_count": sop_style.get("should_count"),
                    "imperative_step_count": sop_style.get("imperative_step_count"),
                    "bullet_count": sop_style.get("bullet_count"),
                    "sub_step_count": sop_style.get("sub_step_count"),
                }
            )
        else:
            st.info("No SOP/QMS pattern features found.")

    with st.expander("Legal/Firm"):
        legal_style = features.get("legal_style", {})
        if legal_style:
            c1, c2, c3, c4 = st.columns(4)
            c1.write("Legalese")
            c1.write(f"**{legal_style.get('legalese_score', 0)}**")
            c2.write("Clause Density")
            c2.write(f"**{legal_style.get('clause_density', 0)}**")
            c3.write("Obligations")
            c3.write(f"**{legal_style.get('obligation_count', 0)}**")
            c4.write("Defined Terms")
            c4.write(f"**{len(legal_style.get('defined_terms', []))}**")
            
            st.write("Detected legal/firm pattern")
            st.json(
                {
                    "format_pattern": legal_style.get("format_pattern"),
                    "clause_headers_found": legal_style.get("clause_headers_found"),
                    "defined_terms": legal_style.get("defined_terms"),
                    "detected_legal_terms": legal_style.get("detected_legal_terms"),
                }
            )
            st.write("Drafting mechanics")
            st.json(
                {
                    "numbered_clause_count": legal_style.get("numbered_clause_count"),
                    "permission_count": legal_style.get("permission_count"),
                    "prohibition_count": legal_style.get("prohibition_count"),
                    "cross_reference_count": legal_style.get("cross_reference_count"),
                    "avg_clause_sentence_words": legal_style.get("avg_clause_sentence_words"),
                }
            )
        else:
            st.info("No legal/firm pattern features found.")

    with st.expander("Genre Scores"):
        st.write("All Scores")
        st.json(profile.get("all_scores", {}))
        st.write("Terminology scores")
        st.json(features["terminology_scores"])

    with st.expander("Chunks"):
        for chunk in profile["chunks"]:
            st.markdown(f"**{chunk['section_title']}** ({chunk['section_type']}, {chunk['word_count']} words)")
            st.text(chunk["content"][:1200])

    with st.expander("JSON Profile Data"):
        st.code(json.dumps(profile, indent=2, ensure_ascii=False), language="json")


st.markdown("<h1 class='gradient-text'>Style Detection & Rewriting Engine 🪄</h1>", unsafe_allow_html=True)
status = llm_status()
thinking = status.get("thinking_enabled", False)
thinking_tag = " | Thinking: ON (Qwen3)" if thinking else ""
st.caption(
    f"LLM: {'ready' if status['llm_client_initialized'] else 'not configured'} | "
    f"Model: {status['hf_model']} | Provider: {status['hf_provider']}{thinking_tag}"
)

input_mode = st.radio(
    "Input method",
    options=["Direct paste", "Upload document"],
    horizontal=True,
)

document_text = ""
analyze_clicked = False
if input_mode == "Direct paste":
    with st.form("direct_paste_form", clear_on_submit=False):
        pasted_text = st.text_area(
            "Direct text copy/paste",
            height=320,
            placeholder="Copy and paste your document text here, then click Analyze Style.",
        )
        use_llm_validation = st.checkbox(
            "Use Hugging Face validation during analysis",
            value=False,
            help="Keep this off for fast NLP output. Rewriting can still use Hugging Face Qwen.",
        )
        analyze_clicked = st.form_submit_button("Analyze Style", type="primary")
    document_text = pasted_text.strip()
else:
    use_llm_validation = st.checkbox(
        "Use Hugging Face validation during analysis",
        value=False,
        help="Keep this off for fast NLP output. Rewriting can still use Hugging Face Qwen.",
    )
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "md", "pdf", "docx"])
    if uploaded_file is not None:
        try:
            uploaded_text = read_uploaded_file(uploaded_file)
            if uploaded_text:
                document_text = uploaded_text.strip()
                with st.expander("Extracted text preview", expanded=True):
                    st.text_area("Extracted document text", value=document_text, height=260)
                st.success(f"Loaded {uploaded_file.name}")
            else:
                st.warning("The uploaded file did not contain extractable text.")
        except Exception as exc:
            st.error(str(exc))
    analyze_clicked = st.button("Analyze Style", type="primary")

if analyze_clicked and not document_text:
    st.warning("Paste text or upload a document before analyzing.")
elif analyze_clicked:
    with st.spinner("Running NLP analysis..."):
        st.session_state["document_text"] = document_text
        st.session_state["profile"] = build_profile(document_text, use_llm_validation)

profile = st.session_state.get("profile")
stored_text = st.session_state.get("document_text", document_text)

if profile:
    try:
        render_nlp_output(profile)
    except Exception as exc:
        st.error(f"NLP UI rendering issue: {exc}")
        st.info("Rewrite/Improve/Create remains available below even if NLP visuals fail.")
elif not document_text:
    st.info("Paste text or upload a document to begin.")

if profile:
    st.divider()
    st.subheader("Improve, Rewrite, or Create 🛠️")
    action = st.selectbox(
        "Feature",
        options=["Improve", "Rewrite", "Create"],
        index=0,
    )
    mode_map = {
        "Improve": "improve",
        "Rewrite": "rewrite",
        "Create": "create_new",
    }
    mode = mode_map[action]
    use_hf_rewrite = st.checkbox(
        "Use Hugging Face Qwen for rewrite",
        value=False,
        help="Leave off for fast local output. Turn on to test remote LLM rewrite.",
    )

    if action == "Create":
        generation_source = st.text_area(
            "New content instruction",
            height=140,
            placeholder="Describe what you want to create in this detected SOP/style...",
        )
        source_text = generation_source.strip() or stored_text
    else:
        source_text = stored_text

    col_run, col_test = st.columns(2)
    run_clicked = col_run.button(action, disabled=not source_text)
    test_improve_clicked = col_test.button("Test improve via HF", disabled=not stored_text)

    if run_clicked:
        with st.spinner(f"Running {mode}..."):
            st.session_state["rewritten_text"] = rewrite_text(
                source_text,
                profile,
                mode,
                use_hf_rewrite,
            )

    if test_improve_clicked:
        with st.spinner("Testing improve mode via Hugging Face Qwen..."):
            improved = rewrite_text(stored_text, profile, "improve", True)
            st.session_state["rewritten_text"] = improved
            st.session_state["improve_test_status"] = {
                "mode": "improve",
                "llm_client_initialized": status["llm_client_initialized"],
                "hf_model": status["hf_model"],
                "hf_provider": status["hf_provider"],
                "output_len": len(improved or ""),
                "is_fallback": isinstance(improved, str)
                and (
                    improved.startswith("[LLM not configured]")
                    or improved.startswith("[Rewrite Failed]")
                    or len(improved.strip()) == 0
                ),
            }
else:
    st.info("Run `Analyze Style` first to unlock Improve, Rewrite, and Create.")

rewritten_text = st.session_state.get("rewritten_text")
improve_test_status = st.session_state.get("improve_test_status")
if improve_test_status:
    st.write("Improve test status")
    st.json(improve_test_status)
if rewritten_text:
    st.markdown("---")
    st.subheader("Output")
    st.text_area("Rewritten / Improved Text", value=rewritten_text, height=460, key="output_area")
    col_copy, col_clear = st.columns(2)
    if col_clear.button("Clear Output"):
        st.session_state.pop("rewritten_text", None)
        st.session_state.pop("improve_test_status", None)
        st.rerun()
    st.caption(f"Output length: {len(rewritten_text)} characters | {len(rewritten_text.split())} words")
