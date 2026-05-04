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
        "llm_client_initialized": (llm_service.hf_client is not None) or (llm_service.gemini_client is not None),
        "model_id": llm_service.GEMINI_MODEL if llm_service.gemini_client else llm_service.MODEL_ID,
        "gemini_api_key_present": bool(os.getenv("GEMINI_API_KEY")),
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
    
    # Extraction from V2 profile
    sop_profile = result.get("structured_sop_profile") or {}
    doc_id_info = sop_profile.get("document_identity", {})
    classification = sop_profile.get("classification", {})
    
    genre = classification.get("primary_genre", "general")
    confidence = classification.get("genre_confidence", 0.0)
    lang_code = result.get("language", {}).get("lang_code", "en")

    style_rules = []
    if sop_profile.get("precision_blocklist"):
        style_rules.append(f"BLOCKLIST: {sop_profile['precision_blocklist']}")
    if sop_profile.get("domain_seeds"):
        style_rules.append(f"DOMAIN SEEDS: {', '.join(sop_profile['domain_seeds'])}")

    do_not_change = []
    traceability = sop_profile.get("traceability", {})
    if traceability.get("ids"):
        do_not_change.append(f"Preserve IDs: {', '.join(traceability['ids'][:10])}")

    return {
        "doc_id": str(uuid4()),
        "genre": genre,
        "confidence": confidence,
        "tone": result.get("tone_profile", {}).get("primary_tone", "neutral"),
        "voice": result.get("features", {}).get("voice", "active"),
        "language": lang_code,
        "key_style_rules": style_rules,
        "do_not_change": do_not_change,
        "features": result.get("writing_style", {}),
        "structured_profile": sop_profile,
        "analysis_result": result, # Store the full result for rewrite
        "chunks": [
            {
                "section_title": chunk.get("section_title", "Body"),
                "section_type": chunk.get("section_type", "general"),
                "word_count": len(chunk.get("content", "").split()),
                "content": chunk.get("content", ""),
                "is_generic": chunk.get("is_generic", False)
            }
            for chunk in result["chunks"]
        ],
    }


def rewrite_text(text: str, profile: dict, mode: str, use_hf_llm: bool = False) -> str:
    # If using HF LLM, use the V2 service
    if use_hf_llm:
        import llm_service
        import nlp_pipeline
        
        # We need the full analysis result for the prompt builder
        analysis = profile.get("analysis_result")
        if not analysis:
            analysis = nlp_pipeline.process_document(text)
            
        chunks = analysis["chunks"]
        rewritten_sections = []

        for chunk in chunks:
            rewritten = llm_service.rewrite_chunk(
                chunk_text=chunk.get("content", ""),
                section_title=chunk.get("section_title", ""),
                analysis_result=analysis,
                similar_docs=[],
                mode=mode,
            )
            
            if "[Rewrite Failed]" in rewritten:
                rewritten = chunk.get("content", "")
                
            title = chunk.get("section_title") or chunk.get("title") or "Body"
            rewritten_sections.append(rewritten)

        return "\n\n".join(rewritten_sections)
    else:
        # Local fallback
        return local_fallback_transform(text, mode, profile)


def local_fallback_transform(text: str, mode: str, profile: dict | None = None) -> str:
    profile = profile or {}
    sop_profile = profile.get("structured_profile", {})
    
    # Check if it's a compliance doc using V2 confidence
    is_sop = profile.get("genre") == "IT_OT_security" or profile.get("genre") == "QMS_general"
    
    if is_sop:
        return compliance_precision_postprocess(local_compliance_transform(text, mode, "SOP/QMS"), profile)

    cleaned = " ".join(text.split())
    if mode == "shorten":
        sentences = [s.strip() for s in cleaned.replace("?", ".").replace("!", ".").split(".") if s.strip()]
        return ". ".join(sentences[: max(1, min(3, len(sentences)))]) + ("." if sentences else "")
    if mode == "expand":
        return (
            f"{cleaned}\n\nAdditional detail: clarify responsibilities, expected records, "
            "review points, and completion criteria while preserving the detected style."
        )
    if mode in ["generate", "create_new"]:
        return (
            "### [LOCAL DRAFT] New Section Generated\n\n"
            f"Based on instruction: {cleaned}\n\n"
            "Style Note: This draft follows the detected profile's genre and tone. "
            "Please use the Gemini toggle for high-fidelity reasoning-based generation."
        )
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
        return f"{result}\n\nCompliance rewrite note: preserve all headings, IDs, constraints, approvals, logs, and controlled terminology."

    if mode == "generate":
        return (
            f"{label} draft generated from the detected structure.\n\n"
            "Purpose:\n\nScope:\n\nResponsibilities:\n\nProcedure:\n1. \n2. \n3. \n\nRecords:\n\nDeviations/CAPAs/Audit Findings/Decisions:\n"
        )

    return result


def compliance_precision_postprocess(text: str, profile: dict) -> str:
    sop_profile = profile.get("structured_profile", {})
    if not sop_profile:
        return text

    corrected = text
    vocabulary = sop_profile.get("vocabulary", {})

    # Apply precision terms from V2 vocabulary
    for term, definition in vocabulary.items():
        if term in ["SPS", "PLC"]:
            corrected = re.sub(r"\bthe PLC\b", "the SPS (PLC)", corrected, count=1, flags=re.I)
        if "22:30" in term:
            corrected = re.sub(r"\b10:30\s*PM\b", "22:30", corrected, flags=re.I)
            
    corrected = re.sub(r"\bAI's line\b", "AI systems", corrected, flags=re.I)
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
    st.subheader("NLP Intelligence Output (V2)")
    render_metric_grid(profile)

    sop_profile = profile.get("structured_profile", {})
    
    with st.expander("Summary & Strategy", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Detected Attributes**")
            st.write(f"Domain: `{profile['genre']}`")
            st.write(f"Tone: `{profile['tone']}`")
            st.write(f"Voice: `{profile['voice']}`")
        with col2:
            st.write("**Compliance Strategy**")
            st.write(profile["key_style_rules"] or ["No specific rules."])
            st.write(profile["do_not_change"] or ["No locked elements."])

    # V2 Step Audit
    step_audit = sop_profile.get("workflow", {}).get("step_audit", [])
    if step_audit:
        with st.expander("Procedural Step Audit", expanded=True):
            st.write(f"Total Steps Detected: {len(step_audit)}")
            for i, step in enumerate(step_audit):
                score = step.get("score", 0)
                color = "green" if score > 0.7 else "orange" if score > 0.4 else "red"
                st.markdown(f"""
                <div style="border-left: 5px solid {color}; padding: 10px; margin-bottom: 10px; background: rgba(255,255,255,0.05)">
                    <strong>Step {i+1}</strong> (Score: {score:.2f})<br/>
                    <em>Actor:</em> {step.get('actor')} | <em>Action:</em> {step.get('action')} | <em>Object:</em> {step.get('object')}
                </div>
                """, unsafe_allow_html=True)

    with st.expander("Compliance Metadata (V2)"):
        st.json({
            "traceability": sop_profile.get("traceability"),
            "vocabulary": sop_profile.get("vocabulary"),
            "domain_seeds": sop_profile.get("domain_seeds"),
            "precision_blocklist": sop_profile.get("precision_blocklist")
        })

    with st.expander("Chunks (Semantic)"):
        for chunk in profile["chunks"]:
            st.markdown(f"**{chunk['section_title']}** ({chunk['section_type']}, {chunk['word_count']} words)")
            st.text(chunk["content"][:800])

    with st.expander("Full JSON Profile"):
        st.code(json.dumps(profile, indent=2, ensure_ascii=False), language="json")


st.markdown("<h1 class='gradient-text'>Style Detection & Rewriting Engine 🪄</h1>", unsafe_allow_html=True)
try:
    status = llm_status()
    thinking = status.get("thinking_enabled", False)
    thinking_tag = " | Thinking: ON (Qwen3)" if thinking else ""
    st.caption(
        f"LLM: {'ready' if status['llm_client_initialized'] else 'not configured'} | "
        f"Model: {status['model_id']}"
    )
except Exception as _llm_err:
    st.caption(f"LLM: not configured | {_llm_err}")

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
            "Use Gemini validation during analysis",
            value=False,
            help="Keep this off for fast NLP output. Rewriting can still use Gemini.",
        )
        analyze_clicked = st.form_submit_button("Analyze Style", type="primary")
    document_text = pasted_text.strip()
else:
    use_llm_validation = st.checkbox(
        "Use Gemini validation during analysis",
        value=False,
        help="Keep this off for fast NLP output. Rewriting can still use Gemini.",
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
        "Use Gemini for rewrite",
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
    test_improve_clicked = col_test.button("Test improve via Gemini", disabled=not stored_text)

    if run_clicked:
        with st.spinner(f"Running {mode}..."):
            st.session_state["rewritten_text"] = rewrite_text(
                source_text,
                profile,
                mode,
                use_hf_rewrite,
            )

    if test_improve_clicked:
        with st.spinner("Testing improve mode via Gemini..."):
            improved = rewrite_text(stored_text, profile, "improve", True)
            st.session_state["rewritten_text"] = improved
            st.session_state["improve_test_status"] = {
                "mode": "improve",
                "llm_client_initialized": status["llm_client_initialized"],
                "model_id": status["model_id"],
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
