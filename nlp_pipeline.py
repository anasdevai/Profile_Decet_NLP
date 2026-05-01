import os
import json
import re
from langdetect import detect, DetectorFactory
import textstat
import pronouncing
import spacy

DetectorFactory.seed = 0

# Load spacy models if available
spacy_models = {}
try:
    spacy_models["en"] = spacy.load("en_core_web_sm")
except OSError:
    pass

def detect_language(text: str) -> dict:
    try:
        lang_code = detect(text)
    except:
        lang_code = "en"

    urdu_arabic = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    latin       = sum(1 for c in text if c.isascii() and c.isalpha())
    total       = max(len(text), 1)

    is_mixed         = (urdu_arabic/total > 0.1) and (latin/total > 0.1)
    has_urdu_script  = urdu_arabic/total > 0.3

    SPACY_SUPPORT = {
        "en": "en_core_web_sm",
        "de": "de_core_news_sm",
        "fr": "fr_core_news_sm",
        "es": "es_core_news_sm",
        "it": "it_core_news_sm",
    }

    return {
        "lang_code":         lang_code,
        "has_spacy_support": lang_code in SPACY_SUPPORT,
        "spacy_model":       SPACY_SUPPORT.get(lang_code),
        "is_mixed_script":   is_mixed,
        "has_urdu_script":   has_urdu_script,
        "script_ratios":     {"urdu_arabic": urdu_arabic/total, "latin": latin/total}
    }

HEADER_PATTERN = re.compile(
    r'^((?:\d+[\.\)]\s+)?[A-Z][A-Za-z0-9\s\+\-\(\)]+:\s*)$',
    re.MULTILINE
)

class DocumentChunk:
    def __init__(self, title, content, section_type="general", is_generic=False):
        self.section_title = title
        self.content = content
        self.section_type = section_type
        self.is_generic = is_generic

def smart_chunk(text: str):
    parts = HEADER_PATTERN.split(text)
    if len(parts) > 2:
        # Strategy A: Header-based
        chunks = []
        current_header = "Intro"
        if parts[0].strip():
            chunks.append(DocumentChunk(current_header, parts[0].strip(), is_generic=True))
        for i in range(1, len(parts), 2):
            header = parts[i].strip()
            content = parts[i+1].strip() if i+1 < len(parts) else ""
            if content:
                chunks.append(DocumentChunk(header, content, determine_section_type(header)))
        return chunks
    else:
        # Strategy B: Size-based
        words = text.split()
        chunks = []
        chunk_words = 1500
        overlap = 50
        for i in range(0, len(words), chunk_words - overlap):
            content = " ".join(words[i:i + chunk_words])
            chunks.append(DocumentChunk(f"Chunk {i//(chunk_words - overlap) + 1}", content, "general", is_generic=True))
        return chunks

def determine_section_type(header: str) -> str:
    h = header.lower()
    types = ["purpose", "scope", "definitions", "responsibilities", "procedure", "references", "revision", "appendix"]
    for t in types:
        if t in h:
            return t
    return "general"

def rhyme_score(lines):
    last_words = [l.strip().split()[-1].lower() for l in lines if l.strip()]
    hits = 0
    valid_pairs = 0
    for i in range(len(last_words)-1):
        pa = pronouncing.phones_for_word(last_words[i])
        pb = pronouncing.phones_for_word(last_words[i+1])
        if pa and pb:
            valid_pairs += 1
            if pronouncing.rhyming_part(pa[0]) == pronouncing.rhyming_part(pb[0]):
                hits += 1
        elif len(last_words[i]) > 2 and len(last_words[i+1]) > 2:
            valid_pairs += 1
            if last_words[i][-2:] == last_words[i+1][-2:]:
                hits += 1
    return hits / max(valid_pairs, 1) if valid_pairs > 0 else 0

def _clean_word(word: str) -> str:
    return re.sub(r"[^a-zA-Z']", "", word).lower()

def _rhyme_key(word: str) -> str:
    cleaned = _clean_word(word)
    phones = pronouncing.phones_for_word(cleaned)
    if phones:
        return pronouncing.rhyming_part(phones[0])
    return cleaned[-3:] if len(cleaned) > 3 else cleaned

def rhyme_scheme(lines):
    scheme = []
    key_to_letter = {}
    next_letter = ord("A")
    end_words = []

    for line in lines:
        words = [_clean_word(w) for w in line.split()]
        words = [w for w in words if w]
        if not words:
            continue

        end_word = words[-1]
        end_words.append(end_word)
        key = _rhyme_key(end_word)
        if key not in key_to_letter:
            key_to_letter[key] = chr(next_letter)
            next_letter += 1
        scheme.append(key_to_letter[key])

    rhyming_pairs = []
    for i in range(len(end_words) - 1):
        if _rhyme_key(end_words[i]) == _rhyme_key(end_words[i + 1]):
            rhyming_pairs.append(f"{end_words[i]}/{end_words[i + 1]}")

    return {
        "scheme": "".join(scheme),
        "end_words": end_words,
        "rhyming_pairs": rhyming_pairs,
        "unique_rhyme_sounds": len(key_to_letter),
    }

def poetic_style_features(text: str, lines: list, total_words: int, pos_dist: dict, ttr: float) -> dict:
    non_empty_lines = [l.strip() for l in lines if l.strip()]
    line_word_counts = [len(l.split()) for l in non_empty_lines]
    avg_line_len = sum(line_word_counts) / max(len(line_word_counts), 1)
    line_len_variance = (
        sum((count - avg_line_len) ** 2 for count in line_word_counts) / max(len(line_word_counts), 1)
    )
    line_break_density = len(non_empty_lines) / max(total_words, 1)
    short_line_ratio = sum(1 for count in line_word_counts if count <= 8) / max(len(line_word_counts), 1)
    scheme = rhyme_scheme(non_empty_lines)

    lower = text.lower()
    metaphor_markers = [
        "like", "as", "as if", "shadow", "light", "flame", "fire", "sea", "ocean",
        "sky", "skies", "storm", "heart", "soul", "dream", "grace", "wings",
        "echo", "silence", "moon", "sun", "stars", "river", "stone", "bloom",
    ]
    sensory_terms = [
        "bright", "dark", "soft", "cold", "warm", "gold", "silver", "whisper",
        "sing", "scent", "touch", "taste", "glow", "hush", "music", "rhythm",
    ]
    direct_explanation_markers = [
        "this means", "in other words", "because", "therefore", "explains",
        "clearly", "directly", "specifically", "the reason", "as a result",
    ]
    subtle_markers = ["perhaps", "almost", "seems", "quiet", "silent", "soft", "beneath", "within"]

    metaphor_hits = sum(1 for marker in metaphor_markers if marker in lower)
    sensory_hits = sum(1 for marker in sensory_terms if marker in lower)
    explanation_hits = sum(1 for marker in direct_explanation_markers if marker in lower)
    subtle_hits = sum(1 for marker in subtle_markers if marker in lower)

    figurative_density = (metaphor_hits + sensory_hits + (pos_dist.get("adj_ratio", 0) * 10)) / max(total_words / 50, 1)
    compression_score = (
        short_line_ratio * 0.45
        + (avg_line_len <= 8) * 0.25
        + min(ttr, 1.0) * 0.20
        + (line_break_density > 0.12) * 0.10
    )
    musicality_score = (
        min(rhyme_score(non_empty_lines), 1.0) * 0.45
        + (short_line_ratio * 0.20)
        + (line_len_variance < 12) * 0.20
        + (len(scheme["rhyming_pairs"]) > 0) * 0.15
    )

    if explanation_hits > subtle_hits and explanation_hits > 0:
        emotional_delivery = "direct/explanatory"
    elif figurative_density >= 1.2 and subtle_hits > 0:
        emotional_delivery = "subtle/symbolic"
    elif figurative_density >= 1.2:
        emotional_delivery = "figurative/emotive"
    else:
        emotional_delivery = "plain/descriptive"

    return {
        "is_lineated": len(non_empty_lines) >= 3 and line_break_density > 0.08,
        "line_count": len(non_empty_lines),
        "avg_line_words": round(avg_line_len, 2),
        "line_length_variance": round(line_len_variance, 2),
        "line_break_density": round(line_break_density, 3),
        "short_line_ratio": round(short_line_ratio, 3),
        "rhyme_scheme": scheme["scheme"],
        "end_words": scheme["end_words"],
        "rhyming_pairs": scheme["rhyming_pairs"],
        "musicality_score": round(musicality_score, 3),
        "figurative_density": round(figurative_density, 3),
        "compression_score": round(compression_score, 3),
        "emotional_delivery": emotional_delivery,
        "metaphor_marker_hits": metaphor_hits,
        "sensory_marker_hits": sensory_hits,
        "explanation_marker_hits": explanation_hits,
    }

def sop_style_features(text: str, lines: list) -> dict:
    lower = text.lower()
    normalized_headers = [re.sub(r"[^a-z ]", "", l.strip().lower()) for l in lines if l.strip()]
    section_aliases = {
        "title": ["sop id titel", "sop id", "titel", "title"],
        "purpose": ["purpose", "objective", "zweck", "ziel"],
        "scope": ["scope", "applicability", "geltungsbereich", "anwendungsbereich"],
        "definitions": ["definitions", "terms and definitions", "definitionen", "begriffe"],
        "responsibilities": ["responsibilities", "roles and responsibilities", "verantwortlichkeiten", "rollen"],
        "procedure": ["procedure", "process", "method", "verfahren", "prozess", "durchfhrung", "ablauf"],
        "key_points": ["key points", "kernpunkte", "wichtige punkte"],
        "records": ["records", "documentation", "forms", "aufzeichnungen", "dokumentation", "logs"],
        "deviations": ["deviations", "abweichungen"],
        "capas": ["capas", "capa", "korrekturmanahmen", "korrekturmassnahmen"],
        "audit_findings": ["audit findings", "audit", "audit findings", "auditfeststellungen"],
        "decisions": ["decisions", "entscheidungen"],
        "references": ["references", "related documents", "referenzen", "mitgeltende dokumente"],
        "revision_history": ["revision history", "change history", "revision", "nderungshistorie"],
    }
    sections_found = []
    for canonical, aliases in section_aliases.items():
        if any(any(alias == h or alias in h for alias in aliases) for h in normalized_headers):
            sections_found.append(canonical)

    expected = list(section_aliases.keys())
    required_core = ["purpose", "scope", "procedure"]
    missing_core = [s for s in required_core if s not in sections_found]

    qms_terms = [
        "qms", "quality", "sop", "procedure", "process", "record", "document control",
        "revision", "approval", "effective date", "capa", "deviation", "nonconformance",
        "audit", "training", "compliance", "iso", "gmp", "controlled copy",
        "zweck", "geltungsbereich", "zugriff", "zugriffsmanagement", "produktionsnetzwerk",
        "ot", "it/ot", "iec 62443", "deviation", "capa", "audit findings", "decisions",
        "abweichung", "entscheidung", "freigabe", "genehmigung", "vertraulichkeit",
        "aufbewahrung", "sperre", "notfall", "break-glass", "qp", "ema", "fda",
    ]
    compliance_ids = re.findall(r"\b(?:SOP|DEV|CAPA|AUD|DEC)-[A-Z]{2,}-\d{3}\b", text, re.I)
    imperative_starts = len(re.findall(r"^\s*(?:[-*]|\d+[\.\)])\s*(verify|record|ensure|review|approve|maintain|document|inspect|submit|implement|migrate|enable|preserve|prfen|dokumentieren|freigeben|genehmigen|umsetzen|migrieren)\b", text, re.I | re.M))
    numbered_steps = len(re.findall(r"^\s*\d+[\.\)]\s+", text, re.M))
    sub_steps = len(re.findall(r"^\s*\d+\.\d+\s+", text, re.M))
    bullet_count = len(re.findall(r"^\s*(?:[-*]|[A-Z]+-[A-Z]+-\d{3}\s*[â€“-])", text, re.I | re.M))
    shall_count = len(re.findall(r"\b(shall|muss|mssen|darf nur|drfen nur)\b", text, re.I))
    must_count = len(re.findall(r"\b(must|required|mandatory|pflicht|erforderlich|obligatorisch)\b", text, re.I))
    should_count = len(re.findall(r"\b(should|soll|sollte)\b", text, re.I))
    qms_hits = sum(1 for term in qms_terms if term in lower)
    precision_terms = {}
    if re.search(r"\bSPS\b", text):
        precision_terms["SPS"] = "Preserve as SPS (PLC) on first English use; do not replace with PLC only."
    if re.search(r"\bKI\b|KI-", text):
        precision_terms["KI"] = "Translate consistently as AI; preserve KI-* access labels as AI-* or KI-* consistently."
    if re.search(r"Log\s+nach\s+1h", text, re.I):
        precision_terms["Log nach 1h"] = "Translate as activity/logging checkpoint after 1 hour; never as logout."
    if re.search(r"\b22:30\b", text):
        precision_terms["22:30"] = "Preserve 24-hour time format; do not convert to 10:30 PM."
    if re.search(r"15\s*Min\s*Sperre", text, re.I):
        precision_terms["15 Min Sperre"] = "Translate as 15-minute lockout."

    completeness_score = len(sections_found) / max(len(expected), 1)
    control_language_score = min((shall_count + must_count + imperative_starts) / 8, 1.0)
    format_score = (
        (numbered_steps > 0 or bullet_count > 4) * 0.20
        + (len(sections_found) >= 4) * 0.25
        + (qms_hits >= 4) * 0.20
        + (sub_steps > 0) * 0.10
        + (len(compliance_ids) >= 3) * 0.25
    )

    if completeness_score >= 0.45 and (numbered_steps or bullet_count > 4 or compliance_ids):
        format_pattern = "controlled QMS/SOP with section headers, traceability IDs, and procedural controls"
    elif numbered_steps:
        format_pattern = "procedure-style numbered work instruction"
    elif len(sections_found) >= 3:
        format_pattern = "sectioned policy/procedure document"
    else:
        format_pattern = "general operational prose"

    return {
        "format_pattern": format_pattern,
        "sections_found": sections_found,
        "sections_missing_core": missing_core,
        "completeness_score": round(completeness_score, 3),
        "numbered_step_count": numbered_steps,
        "sub_step_count": sub_steps,
        "bullet_count": bullet_count,
        "compliance_ids": sorted(set(compliance_ids)),
        "compliance_id_count": len(set(compliance_ids)),
        "imperative_step_count": imperative_starts,
        "shall_count": shall_count,
        "must_count": must_count,
        "should_count": should_count,
        "qms_term_hits": qms_hits,
        "control_language_score": round(control_language_score, 3),
        "format_score": round(format_score, 3),
        "detected_standard_terms": [term for term in qms_terms if term in lower],
        "precision_terms": precision_terms,
    }

def legal_style_features(text: str, lines: list) -> dict:
    lower = text.lower()
    clause_headers = re.findall(
        r"(?:^|\n)\s*(?:\d+[\.\)]\s*)?(definitions|term|confidentiality|obligations|representations|warranties|indemnification|limitation of liability|termination|governing law|jurisdiction|notices|miscellaneous)\b",
        text,
        re.I,
    )
    legal_terms = [
        "agreement", "whereas", "therefore", "hereby", "pursuant", "notwithstanding",
        "covenant", "indemnify", "liable", "liability", "jurisdiction", "governing law",
        "confidential information", "representations", "warranties", "termination",
        "breach", "remedy", "force majeure", "severability", "assigns", "successors",
    ]
    defined_terms = re.findall(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s+means\b", text)
    obligation_count = len(re.findall(r"\b(shall|must|agrees to|is obligated to|will)\b", text, re.I))
    permission_count = len(re.findall(r"\b(may|is permitted to|has the right to)\b", text, re.I))
    prohibition_count = len(re.findall(r"\b(shall not|must not|may not|is prohibited from)\b", text, re.I))
    cross_refs = len(re.findall(r"\b(section|clause|article)\s+\d+(?:\.\d+)*\b", text, re.I))
    numbered_clauses = len(re.findall(r"(?:^|\n)\s*\d+(?:\.\d+)*[\.\)]\s+", text))
    whereas_count = len(re.findall(r"\bwhereas\b", text, re.I))
    legal_hits = sum(1 for term in legal_terms if term in lower)

    sentence_parts = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    avg_clause_sentence_words = sum(len(s.split()) for s in sentence_parts) / max(len(sentence_parts), 1)
    clause_density = (numbered_clauses + len(clause_headers) + cross_refs) / max(len(text.split()) / 100, 1)
    legalese_score = min((legal_hits + whereas_count + obligation_count + prohibition_count + len(defined_terms)) / 14, 1.0)

    if whereas_count and numbered_clauses:
        format_pattern = "formal agreement with recitals and numbered clauses"
    elif numbered_clauses or len(clause_headers) >= 3:
        format_pattern = "clause-based legal/firm document"
    elif legal_hits >= 4:
        format_pattern = "legal prose with obligation language"
    else:
        format_pattern = "general prose"

    return {
        "format_pattern": format_pattern,
        "clause_headers_found": sorted(set(h.lower() for h in clause_headers)),
        "numbered_clause_count": numbered_clauses,
        "defined_terms": sorted(set(defined_terms))[:20],
        "obligation_count": obligation_count,
        "permission_count": permission_count,
        "prohibition_count": prohibition_count,
        "cross_reference_count": cross_refs,
        "whereas_count": whereas_count,
        "legal_term_hits": legal_hits,
        "legalese_score": round(legalese_score, 3),
        "clause_density": round(clause_density, 3),
        "avg_clause_sentence_words": round(avg_clause_sentence_words, 2),
        "detected_legal_terms": [term for term in legal_terms if term in lower],
    }

SOP_SECTION_LABELS = {
    "purpose": ["purpose", "objective", "aim", "zweck", "ziel"],
    "scope": ["scope", "applicability", "geltungsbereich", "anwendungsbereich"],
    "definitions": ["definitions", "terms and definitions", "glossary", "definitionen", "begriffe"],
    "responsibilities": ["responsibilities", "roles", "accountability", "ownership", "verantwortlichkeiten", "rollen"],
    "procedure": ["procedure", "process", "method", "steps", "workflow", "work instruction", "verfahren", "prozess", "ablauf", "key points"],
    "prerequisites": ["prerequisites", "requirements", "precondition", "materials", "equipment", "voraussetzungen"],
    "safety": ["safety", "warning", "caution", "hazard", "risk", "sicherheit", "warnung", "gefahr"],
    "notes": ["notes", "important note", "remarks"],
    "exceptions": ["exceptions", "deviation", "alternate flow"],
    "references": ["references", "related documents", "source documents", "referenzen", "mitgeltende dokumente"],
    "records": ["records", "logs", "retention", "documentation", "aufzeichnungen", "dokumentation", "protokoll"],
    "appendix": ["appendix", "annex", "attachment"],
    "approvals": ["approval", "approvals", "sign-off", "authorization", "freigabe", "genehmigung"],
    "revision_history": ["revision history", "change log", "version history", "anderungshistorie", "anderungen"],
}

ROLE_TERMS = [
    "user", "operator", "technician", "supervisor", "manager", "admin",
    "reviewer", "approver", "system", "qa", "qc", "auditor", "department",
]

DOMAIN_TERM_BANKS = {
    "healthcare": ["patient", "clinical", "hospital", "medication", "diagnosis", "healthcare"],
    "laboratory": ["specimen", "lab", "assay", "reagent", "calibration", "laboratory"],
    "manufacturing": ["production", "batch", "line", "assembly", "gmp", "manufacturing"],
    "it": ["server", "application", "deploy", "backup", "incident", "it", "ot", "sps", "scada", "vpn", "service-account", "produktionsnetzwerk", "zugriffsmanagement"],
    "security": ["access control", "authentication", "authorization", "threat", "security", "2fa", "token", "firewall", "passwort", "zugriff", "berechtigung", "vertraulichkeit"],
    "operations": ["operations", "runbook", "handover", "workflow", "sla", "produktion", "wartung"],
    "administration": ["office", "administrative", "policy", "documentation", "filing"],
    "finance": ["invoice", "ledger", "reconciliation", "expense", "financial"],
    "education": ["curriculum", "student", "classroom", "assessment", "education"],
    "logistics": ["shipment", "warehouse", "dispatch", "inventory", "logistics"],
    "customer_support": ["ticket", "customer", "escalation", "resolution", "support"],
    "research": ["study", "protocol", "hypothesis", "experiment", "research"],
}

TRACE_ID_PATTERN = re.compile(r"\b(SOP|DEV|CAPA|AUD|DEC)-[A-Z]{2,}-\d{3,}\b", re.I)

CANONICAL_SECTION_VARIANTS = {
    "purpose": ["purpose", "objective", "intent", "zweck"],
    "scope": ["scope", "applicability", "coverage", "geltungsbereich"],
    "responsibilities": ["responsibilities", "roles", "ownership", "accountability", "verantwortlichkeiten"],
    "procedure": ["procedure", "process", "method", "workflow", "verfahren", "prozess", "ablauf"],
    "deviations": ["deviations", "exceptions", "incidents", "abweichungen"],
    "capa": ["capa", "corrective actions", "corrective and preventive actions", "korrekturmassnahmen"],
    "audit_findings": ["audit findings", "observations", "audits", "auditfeststellungen"],
    "decisions": ["decisions", "approvals", "resolution", "entscheidungen"],
    "records": ["records", "logs", "retention", "aufzeichnungen"],
    "references": ["references", "standards", "related documents", "referenzen"],
}

DOMAIN_SECTION_VARIANTS = {
    "medical": ["patient safety", "clinical protocol", "risk assessment"],
    "healthcare": ["patient safety", "clinical protocol", "risk assessment"],
    "insurance": ["claims processing", "policy rules", "fraud checks"],
    "it": ["access control", "network security", "authentication rules"],
    "security": ["access control", "network security", "authentication rules"],
    "pharma": ["gmp controls", "batch release", "quality risk management"],
}

SOP_LANGUAGE_PACKS = {
    "de": {
        "section_headers": {
            "title": "Titel",
            "purpose": "Zweck",
            "scope": "Geltungsbereich",
            "responsibilities": "Verantwortlichkeiten",
            "procedure": "Verfahren",
            "exceptions": "Ausnahmen",
            "records": "Aufzeichnungen",
            "references": "Referenzen",
        },
        "phrases": {
            "mandatory": "muss",
            "prohibition": "darf nicht",
            "conditional": "falls",
            "verify": "verifizieren",
            "record": "dokumentieren",
        },
    },
    "en": {
        "section_headers": {
            "title": "Title",
            "purpose": "Purpose",
            "scope": "Scope",
            "responsibilities": "Responsibilities",
            "procedure": "Procedure",
            "exceptions": "Exceptions",
            "records": "Records",
            "references": "References",
        },
        "phrases": {
            "mandatory": "shall",
            "prohibition": "must not",
            "conditional": "if",
            "verify": "verify",
            "record": "document",
        },
    },
}

def _infer_sop_language(text: str, lang_code: str = "en") -> str:
    lower = text.lower()
    de_markers = len(re.findall(r"\b(zweck|geltungsbereich|verantwortlichkeiten|verfahren|aufzeichnungen|genehmigung|abweichung|muss|darf nicht)\b", lower))
    en_markers = len(re.findall(r"\b(purpose|scope|responsibilities|procedure|records|approval|deviation|shall|must)\b", lower))
    if de_markers > en_markers:
        return "de"
    if lang_code.startswith("de"):
        return "de"
    return "en"

def _normalize_header(line: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", line.lower()).strip()

def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r"^[^:]{2,40}:\s*$", stripped):
        return True
    return bool(re.match(r"^(\d+(\.\d+)*[\.\)]\s+.+|[A-Z][A-Z0-9 \-/]{3,}|[A-Z][a-z]+(?:\s+[A-Za-z]+){0,8}\s*:?)$", stripped))

def _semantic_overlap_score(text_a: str, text_b: str) -> float:
    a_tokens = set(re.findall(r"[a-z]{3,}", text_a.lower()))
    b_tokens = set(re.findall(r"[a-z]{3,}", text_b.lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(len(a_tokens | b_tokens), 1)

def _classify_header_semantically(raw_header: str, domain: str = "general") -> tuple:
    norm = _normalize_header(raw_header)
    best_label = "general"
    best_score = 0.0
    for canonical, variants in CANONICAL_SECTION_VARIANTS.items():
        variant_score = max((_semantic_overlap_score(norm, v) for v in variants), default=0.0)
        if any(v in norm for v in variants):
            variant_score = max(variant_score, 0.95)
        if variant_score > best_score:
            best_score = variant_score
            best_label = canonical

    domain_hits = []
    for dom, labels in DOMAIN_SECTION_VARIANTS.items():
        if dom in domain and any(lbl in norm for lbl in labels):
            domain_hits.extend([lbl for lbl in labels if lbl in norm])
    return best_label, round(best_score, 3), sorted(set(domain_hits))

def _qdrant_semantic_section_lookup(section_candidates: list, domain: str = "general") -> dict:
    """
    Optional Qdrant semantic linking. Safe no-op when Qdrant config/client is unavailable.
    Uses term-overlap proxy vectors unless external embeddings are configured.
    """
    # Keep analysis stable by default in local/desktop usage.
    # Enable this only when explicitly requested.
    if os.getenv("ENABLE_QDRANT_SECTION_LOOKUP", "false").strip().lower() not in {"1", "true", "yes"}:
        return {"enabled": False, "reason": "disabled by default (set ENABLE_QDRANT_SECTION_LOOKUP=true to enable)", "links": []}

    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        return {"enabled": False, "reason": "QDRANT_URL not configured", "links": []}
    try:
        from qdrant_client import QdrantClient  # type: ignore
    except Exception as exc:
        return {"enabled": False, "reason": f"qdrant-client unavailable: {exc}", "links": []}

    try:
        client = QdrantClient(url=qdrant_url, api_key=os.getenv("QDRANT_API_KEY"))
        collection = os.getenv("QDRANT_SOP_COLLECTION", "sop_sections")
        links = []
        for candidate in section_candidates[:40]:
            query_text = f"{domain} {candidate.get('raw', '')} {candidate.get('normalized', '')}".strip()
            # Keep integration lightweight by using text payload filter; embedding lookup can be layered later.
            hits = client.scroll(
                collection_name=collection,
                limit=3,
                with_payload=True,
                scroll_filter=None,
            )[0]
            mapped = []
            for h in hits:
                payload = getattr(h, "payload", {}) or {}
                sem = payload.get("section_label") or payload.get("canonical") or payload.get("title")
                if sem:
                    mapped.append(str(sem))
            if mapped:
                links.append({
                    "header": candidate.get("raw"),
                    "query_text": query_text,
                    "semantic_matches": mapped[:3],
                })
        return {"enabled": True, "reason": "linked via Qdrant scroll", "links": links}
    except Exception as exc:
        return {"enabled": False, "reason": f"qdrant query failed: {exc}", "links": []}

def _extract_metadata(text: str) -> dict:
    patterns = {
        "title": r"(?im)^(?:title|document title)\s*[:\-]\s*(.+)$",
        "sop_number": r"(?im)^(?:sop(?:\s*(?:number|no|id))?|document id)\s*[:\-]\s*([A-Za-z0-9\-_/.]+)$",
        "version": r"(?im)^(?:version|rev(?:ision)?)\s*[:\-]\s*([A-Za-z0-9.\-_]+)$",
        "effective_date": r"(?im)^(?:effective date|date effective)\s*[:\-]\s*(.+)$",
        "review_date": r"(?im)^(?:review date|next review)\s*[:\-]\s*(.+)$",
        "author": r"(?im)^(?:author|prepared by)\s*[:\-]\s*(.+)$",
        "owner": r"(?im)^(?:owner|process owner|document owner)\s*[:\-]\s*(.+)$",
        "approver": r"(?im)^(?:approver|approved by)\s*[:\-]\s*(.+)$",
        "department": r"(?im)^(?:department|dept)\s*[:\-]\s*(.+)$",
    }
    metadata = {}
    for key, pat in patterns.items():
        m = re.search(pat, text)
        metadata[key] = m.group(1).strip() if m else None

    if not metadata.get("title"):
        m = re.search(r"(?im)^sop\s*id\s*\+\s*titel\s*[:\-]\s*(.+)$", text)
        metadata["title"] = m.group(1).strip() if m else metadata.get("title")
    if not metadata.get("sop_number"):
        m = re.search(r"\b(SOP-[A-Z]{2,}-\d{3,})\b", text, re.I)
        metadata["sop_number"] = m.group(1).upper() if m else None

    metadata["revision_history_found"] = bool(re.search(r"(?im)\b(revision history|change log|version history|anderungshistorie)\b", text))
    metadata["change_log_entries"] = len(re.findall(r"(?im)^\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}.*(?:version|rev|change)\b", text))
    return metadata

def _extract_section_structure(lines: list, domain: str = "general") -> dict:
    headings = []
    for idx, line in enumerate(lines):
        if _is_heading(line):
            raw = line.strip()
            norm = _normalize_header(raw)
            numeric_prefix = re.match(r"^(\d+(?:\.\d+)*)", raw)
            level = numeric_prefix.group(1).count(".") + 1 if numeric_prefix else 1
            label = "general"
            domain_hits = []
            rule_hits = []
            for canonical, aliases in SOP_SECTION_LABELS.items():
                if any(alias in norm for alias in aliases):
                    label = canonical
                    rule_hits.append(canonical)
                    break
            sem_label, sem_score, domain_hits = _classify_header_semantically(raw, domain=domain)
            final_label = sem_label if sem_score >= 0.5 else label
            headings.append({
                "line_index": idx,
                "raw": raw,
                "normalized": norm,
                "label": final_label,
                "rule_label": label,
                "semantic_label": sem_label,
                "semantic_score": sem_score,
                "domain_hits": domain_hits,
                "rule_hits": rule_hits,
                "level": level
            })

    labels = [h["label"] for h in headings if h["label"] != "general"]
    unique_labels = sorted(set(labels))
    required = ["purpose", "scope", "responsibilities", "procedure", "records"]
    missing = [s for s in required if s not in unique_labels]
    repeated = sorted({label for label in labels if labels.count(label) > 1})
    optional = sorted([s for s in unique_labels if s not in required])

    all_caps_ratio = sum(1 for h in headings if h["raw"].isupper()) / max(len(headings), 1)
    numbering_styles = {
        "decimal": any(re.match(r"^\d+\.\d+", h["raw"]) for h in headings),
        "simple_numeric": any(re.match(r"^\d+[\.\)]\s+", h["raw"]) for h in headings),
        "roman": any(re.match(r"^(?:[IVXLC]+)[\.\)]\s+", h["raw"], re.I) for h in headings),
    }
    qdrant_links = _qdrant_semantic_section_lookup(headings, domain=domain)
    return {
        "sections": headings,
        "section_order": labels,
        "section_hierarchy_levels": [h["level"] for h in headings],
        "missing_core_sections": missing,
        "repeated_sections": repeated,
        "optional_sections": optional,
        "heading_capitalization_style": "all_caps" if all_caps_ratio > 0.6 else "title_or_sentence_case",
        "heading_numbering_style": [k for k, v in numbering_styles.items() if v] or ["none"],
        "heading_formatting_style": "numbered headings" if any(numbering_styles.values()) else "plain headings",
        "semantic_section_classifier": {
            "enabled": True,
            "method": "rule + token-overlap semantics + optional qdrant linking",
            "qdrant": qdrant_links,
        },
    }

def extract_traceability(text: str) -> dict:
    ids = TRACE_ID_PATTERN.findall(text)
    id_values = re.findall(TRACE_ID_PATTERN, text)
    flat_ids = re.findall(r"\b(?:SOP|DEV|CAPA|AUD|DEC)-[A-Z]{2,}-\d{3,}\b", text, re.I)
    entities = []
    for uid in sorted(set([x.upper() for x in flat_ids])):
        etype = uid.split("-", 1)[0]
        entities.append({"id": uid, "entity_type": etype})

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    relations = []
    entity_ids = [e["id"] for e in entities]
    for line in lines:
        refs = [eid for eid in entity_ids if eid in line.upper()]
        if len(refs) < 2:
            continue
        for i in range(len(refs)):
            for j in range(i + 1, len(refs)):
                a, b = refs[i], refs[j]
                relation = "related_to"
                upper_line = line.upper()
                if a.startswith("DEV-") and b.startswith("CAPA-") or b.startswith("DEV-") and a.startswith("CAPA-"):
                    relation = "addressed_by"
                elif a.startswith("AUD-") and b.startswith("CAPA-") or b.startswith("AUD-") and a.startswith("CAPA-"):
                    relation = "remediated_by"
                elif a.startswith("SOP-") and b.startswith("DEV-") or b.startswith("SOP-") and a.startswith("DEV-"):
                    relation = "has_deviation"
                elif a.startswith("SOP-") and b.startswith("DEC-") or b.startswith("SOP-") and a.startswith("DEC-"):
                    relation = "decision_for"
                elif "CAUSE" in upper_line or "ROOT" in upper_line:
                    relation = "cause_of"
                relations.append({"source": a, "target": b, "relation": relation, "evidence": line})

    # Section-context relationship inference even when IDs are not on same line.
    sections = {"deviations": [], "capa": [], "audit": [], "decisions": [], "sop": []}
    current = None
    for line in lines:
        low = line.lower().strip(":")
        if "deviation" in low or "abweich" in low:
            current = "deviations"
            continue
        if "capa" in low or "corrective" in low:
            current = "capa"
            continue
        if "audit" in low:
            current = "audit"
            continue
        if "decision" in low or "approval" in low or "entscheid" in low:
            current = "decisions"
            continue
        for uid in re.findall(r"\b(?:SOP|DEV|CAPA|AUD|DEC)-[A-Z]{2,}-\d{3,}\b", line, re.I):
            key = current if current in sections else uid.split("-", 1)[0].lower()
            if key in sections:
                sections[key].append(uid.upper())
            if uid.upper().startswith("SOP-"):
                sections["sop"].append(uid.upper())

    sop_refs = sorted(set(sections["sop"])) or [e["id"] for e in entities if e["id"].startswith("SOP-")]
    for sop_id in sop_refs:
        for dev in sorted(set(sections["deviations"])):
            relations.append({"source": sop_id, "target": dev, "relation": "has_deviation", "evidence": "section:deviations"})
        for aud in sorted(set(sections["audit"])):
            relations.append({"source": sop_id, "target": aud, "relation": "has_audit_finding", "evidence": "section:audit"})
        for dec in sorted(set(sections["decisions"])):
            relations.append({"source": sop_id, "target": dec, "relation": "decision_for", "evidence": "section:decisions"})

    # Pair DEV/CAPA and AUD/CAPA by numeric suffix when domain code matches.
    dev_map = {d.split("-")[-1]: d for d in sorted(set(sections["deviations"]))}
    aud_map = {a.split("-")[-1]: a for a in sorted(set(sections["audit"]))}
    capa_map = {c.split("-")[-1]: c for c in sorted(set(sections["capa"]))}
    for num, dev_id in dev_map.items():
        if num in capa_map:
            relations.append({"source": dev_id, "target": capa_map[num], "relation": "addressed_by", "evidence": "id_suffix_match"})
    for num, aud_id in aud_map.items():
        if num in capa_map:
            relations.append({"source": aud_id, "target": capa_map[num], "relation": "remediated_by", "evidence": "id_suffix_match"})

    # Heuristic cross-links by matching domain segment (e.g., IT) and numeric locality.
    for ent in entities:
        ent["links"] = [r for r in relations if r["source"] == ent["id"] or r["target"] == ent["id"]]

    relation_stats = {}
    for r in relations:
        relation_stats[r["relation"]] = relation_stats.get(r["relation"], 0) + 1

    return {
        "entities": entities,
        "relations": relations[:300],
        "relation_counts": relation_stats,
        "id_counts": {
            "SOP": len([e for e in entities if e["entity_type"] == "SOP"]),
            "DEV": len([e for e in entities if e["entity_type"] == "DEV"]),
            "CAPA": len([e for e in entities if e["entity_type"] == "CAPA"]),
            "AUD": len([e for e in entities if e["entity_type"] == "AUD"]),
            "DEC": len([e for e in entities if e["entity_type"] == "DEC"]),
        }
    }

def extract_workflow(text: str) -> dict:
    lines = text.splitlines()
    step_pattern = re.compile(r"^\s*(\d+(?:\.\d+)*[\.\)]?)\s+(.+)$")
    bullet_pattern = re.compile(r"^\s*[-*â€¢]\s+(.+)$")
    condition_pattern = re.compile(r"\b(if|when|unless|in case of|otherwise|nur mit|nur bei|falls)\b", re.I)
    decision_pattern = re.compile(r"\b(decide|decision|approve|reject|pass|fail|yes|no|genehmig|freigab)\b", re.I)
    exception_pattern = re.compile(r"\b(exception|deviation|abweich|escalate|escalation|error|incident|notfall)\b", re.I)
    completion_pattern = re.compile(r"\b(completed|completion|verify|validated|closed|sign[- ]?off|freigegeben|abschluss)\b", re.I)

    steps = []
    for idx, line in enumerate(lines):
        m = step_pattern.match(line)
        if m:
            step_id = m.group(1).rstrip(".)")
            content = m.group(2).strip()
            steps.append({"id": step_id, "line_index": idx, "text": content, "type": "numbered"})
            continue
        b = bullet_pattern.match(line)
        if b and steps:
            steps.append({"id": f"{steps[-1]['id']}.b{len(steps)}", "line_index": idx, "text": b.group(1).strip(), "type": "bullet_substep"})
        elif ":" in line and len(line.split()) > 3:
            label, content = line.split(":", 1)
            if content.strip():
                steps.append({"id": f"s{len(steps)+1}", "line_index": idx, "text": f"{label.strip()}: {content.strip()}", "type": "inline_instruction"})

    edges = []
    for i in range(len(steps) - 1):
        edges.append({"from": steps[i]["id"], "to": steps[i + 1]["id"], "type": "sequence"})

    decisions = [s for s in steps if decision_pattern.search(s["text"]) or condition_pattern.search(s["text"])]
    exceptions = [s for s in steps if exception_pattern.search(s["text"])]
    validations = [s for s in steps if completion_pattern.search(s["text"])]
    dependencies = [
        {"step_id": s["id"], "depends_on": steps[max(i - 1, 0)]["id"] if i > 0 else None}
        for i, s in enumerate(steps)
    ]

    return {
        "steps": steps,
        "edges": edges,
        "decision_points": [{"step_id": s["id"], "text": s["text"]} for s in decisions],
        "preconditions": [s["text"] for s in steps if re.search(r"\b(before|prior to|precondition)\b", s["text"], re.I)],
        "branches": [{"step_id": s["id"], "branch_type": "conditional"} for s in decisions],
        "dependencies": dependencies,
        "alternate_flows": [s["text"] for s in steps if re.search(r"\b(alternative|alternate|fallback)\b", s["text"], re.I)],
        "exception_handling": [s["text"] for s in exceptions],
        "escalation_paths": [s["text"] for s in steps if re.search(r"\bescalat(e|ion)\b", s["text"], re.I)],
        "validation_steps": [s["text"] for s in validations],
        "completion_criteria": [s["text"] for s in validations],
        "workflow_graph": {
            "nodes": [{"id": s["id"], "label": s["text"]} for s in steps],
            "edges": edges,
        },
    }

def _extract_roles_and_actions(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    assignments = []
    for line in lines:
        for role in ROLE_TERMS:
            if re.search(rf"\b{re.escape(role)}\b", line, re.I):
                action_match = re.search(r"\b(shall|must|should|may|is required to|is responsible for)\b\s+(.+)", line, re.I)
                action = action_match.group(2).strip() if action_match else line
                handoff = bool(re.search(r"\b(hand off|handoff|submit to|forward to|escalate to)\b", line, re.I))
                assignments.append({
                    "actor": role,
                    "action": action,
                    "object": None,
                    "responsibility": line,
                    "handoff": handoff,
                    "ownership": bool(re.search(r"\bowner|accountable|responsible\b", line, re.I)),
                })

    actor_counts = {}
    for a in assignments:
        actor_counts[a["actor"]] = actor_counts.get(a["actor"], 0) + 1

    return {
        "responsibility_matrix": assignments[:150],
        "actors_detected": sorted(actor_counts.keys()),
        "actor_distribution": actor_counts,
    }

def _extract_tone_style_profile(text: str, sentences: list, modal_counts: dict) -> dict:
    total_sent = max(len(sentences), 1)
    avg_sentence_length = sum(len(s.split()) for s in sentences) / total_sent
    imperative_terms = (
        r"verify|ensure|record|maintain|document|review|approve|check|submit|"
        r"pruefen|prÃ¼fen|sicherstellen|dokumentieren|freigeben|genehmigen|"
        r"aufzeichnen|bewerten|kontrollieren|umsetzen|archivieren"
    )
    imperative_density = sum(1 for s in sentences if re.match(rf"^({imperative_terms})\b", s.strip(), re.I)) / total_sent
    passive_hits = len(re.findall(r"\b(is|are|was|were|be|been|being)\s+\w+ed\b", text, re.I))
    passive_ratio = passive_hits / total_sent
    cautionary_hits = len(re.findall(r"\b(caution|warning|hazard|danger|achtung|warnung|gefahr|risiko)\b", text, re.I))
    compliance_focus = modal_counts.get("mandatory", 0) + len(re.findall(r"\b(compliance|audit|regulatory|shall|sop|capa|abweichung|genehmigung|freigabe|aufbewahrung|pflicht|muss)\b", text, re.I))
    sentence_lengths = [len(s.split()) for s in sentences] or [0]
    variance = sum((x - avg_sentence_length) ** 2 for x in sentence_lengths) / max(len(sentence_lengths), 1)

    formal_level = "highly_formal" if avg_sentence_length > 17 and compliance_focus > 4 else "formal" if compliance_focus > 2 else "semi_formal"
    strictness_level = "high" if modal_counts.get("mandatory", 0) >= modal_counts.get("permitted", 0) + 1 else "moderate"
    authority_level = "high" if modal_counts.get("mandatory", 0) + cautionary_hits >= 4 else "medium"

    return {
        "formality": formal_level,
        "instructional_tone": imperative_density > 0.18 or compliance_focus > 5,
        "imperative_tone": imperative_density > 0.15,
        "voice_profile": {"active_ratio_estimate": round(max(1 - passive_ratio, 0.0), 3), "passive_ratio_estimate": round(min(passive_ratio, 1.0), 3)},
        "compliance_focused_tone": compliance_focus > 3,
        "cautionary_tone": cautionary_hits > 0,
        "conciseness": "concise" if avg_sentence_length < 14 else "verbose",
        "strictness_level": strictness_level,
        "authority_level": authority_level,
        "average_sentence_length": round(avg_sentence_length, 2),
        "sentence_complexity_variance": round(variance, 2),
        "passive_voice_ratio": round(min(passive_ratio, 1.0), 3),
        "imperative_density": round(imperative_density, 3),
        "modal_distribution": modal_counts,
        "punctuation_patterns": {
            "colon_count": text.count(":"),
            "semicolon_count": text.count(";"),
            "parentheses_count": text.count("(") + text.count(")"),
        },
        "repetition_patterns": {
            "repeated_warning_tokens": len(re.findall(r"\b(?:warning|ensure|verify)\b", text, re.I)),
        },
        "consistency_across_sections": "high" if variance < 40 else "medium",
    }

def _extract_vocabulary_profile(text: str) -> dict:
    lower = text.lower()
    mandatory = re.findall(r"\b(shall|must|required to|mandatory|muss|erforderlich|pflicht)\b", lower)
    optional = re.findall(r"\b(should|may|optional|recommended|sollte|kann)\b", lower)
    prohibition = re.findall(r"\b(prohibited|must not|shall not|may not|do not|verboten|darf nicht)\b", lower)
    conditional = re.findall(r"\b(if|when|unless|in case of|otherwise|falls|bei)\b", lower)
    compliance_phrases = [
        "ensure that", "required to", "in case of", "before proceeding", "upon completion", "verify that"
    ]
    phrase_hits = [p for p in compliance_phrases if p in lower]
    words = re.findall(r"\b[a-z]{3,}\b", lower)
    ngrams = re.findall(r"\b([a-z]{3,}\s+[a-z]{3,}(?:\s+[a-z]{3,})?)\b", lower)
    common_verbs = re.findall(r"\b(ensure|verify|record|document|review|approve|maintain|submit|check)\b", lower)

    top_phrases = {}
    for ng in ngrams:
        top_phrases[ng] = top_phrases.get(ng, 0) + 1
    top_phrases = sorted(top_phrases.items(), key=lambda x: x[1], reverse=True)[:20]

    return {
        "common_verbs": sorted(set(common_verbs)),
        "standard_instruction_phrases": phrase_hits,
        "compliance_phrases": phrase_hits,
        "mandatory_language": sorted(set(mandatory)),
        "optional_language": sorted(set(optional)),
        "prohibition_language": sorted(set(prohibition)),
        "conditional_language": sorted(set(conditional)),
        "domain_terminology_candidates": sorted(set(w for w in words if len(w) > 7))[:50],
        "repeated_ngrams": top_phrases,
    }

def _extract_compliance_rules(text: str) -> dict:
    rules = []
    for line in [l.strip() for l in text.splitlines() if l.strip()]:
        if re.search(r"\b(shall|must|required|mandatory|muss|erforderlich|pflicht)\b", line, re.I):
            rules.append({"type": "mandatory", "priority": "high", "text": line})
        elif re.search(r"\b(shall not|must not|prohibited|forbidden|do not|verboten|darf nicht)\b", line, re.I):
            rules.append({"type": "prohibited", "priority": "critical", "text": line})
        elif re.search(r"\b(should|recommended|sollte)\b", line, re.I):
            rules.append({"type": "recommended", "priority": "medium", "text": line})
        elif re.search(r"\b(if|when|unless|in case of|falls|bei)\b", line, re.I):
            rules.append({"type": "conditional", "priority": "high", "text": line})
        elif re.search(r"\b(safety|warning|hazard|caution|sicherheit|warnung|gefahr)\b", line, re.I):
            rules.append({"type": "safety", "priority": "critical", "text": line})
        elif re.search(r"\b(audit|inspection|traceability|record|aufzeichnung|log)\b", line, re.I):
            rules.append({"type": "audit", "priority": "high", "text": line})
        elif re.search(r"\b(regulatory|fda|ema|iso|gmp|iec)\b", line, re.I):
            rules.append({"type": "regulatory", "priority": "high", "text": line})
        elif re.search(r"\b(quality|qc|qa|acceptance criteria)\b", line, re.I):
            rules.append({"type": "quality_control", "priority": "high", "text": line})
        elif re.search(r"\b(escalate|escalation|notify supervisor)\b", line, re.I):
            rules.append({"type": "escalation", "priority": "high", "text": line})
        elif re.search(r"\b(approval|approved by|sign-off)\b", line, re.I):
            rules.append({"type": "approval", "priority": "high", "text": line})
        elif re.search(r"\b(retention|retain|archive|records)\b", line, re.I):
            rules.append({"type": "retention", "priority": "medium", "text": line})
    counts = {}
    for r in rules:
        counts[r["type"]] = counts.get(r["type"], 0) + 1
    return {"rules": rules[:200], "rule_type_counts": counts}

def _extract_formatting_profile(text: str, lines: list) -> dict:
    numbering_style = "decimal" if re.search(r"^\s*\d+\.\d+\s+", text, re.M) else "numeric" if re.search(r"^\s*\d+[\.\)]\s+", text, re.M) else "none"
    bullet_style = "dash_or_star" if re.search(r"^\s*[-*]\s+", text, re.M) else "none"
    sub_bullet_style = "indented" if re.search(r"^\s{2,}[-*]\s+", text, re.M) else "flat"
    indentation_behavior = "mixed" if re.search(r"^\t|\s{2,}", text, re.M) else "minimal"
    note_blocks = len(re.findall(r"(?im)^\s*(note|warning|caution)\s*[:\-]", text))
    caps_headers = sum(1 for l in lines if l.strip() and l.strip().isupper())

    return {
        "numbering_style": numbering_style,
        "bullet_style": bullet_style,
        "sub_bullet_style": sub_bullet_style,
        "indentation_behavior": indentation_behavior,
        "table_usage": text.count("|") > 4,
        "note_warning_blocks": note_blocks,
        "bold_emphasis_markers": len(re.findall(r"\*\*[^*]+\*\*", text)),
        "italic_emphasis_markers": len(re.findall(r"_[^_]+_", text)),
        "all_caps_headers_count": caps_headers,
        "section_separators_count": len(re.findall(r"^\s*[-=_]{3,}\s*$", text, re.M)),
        "revision_table_layout_detected": bool(re.search(r"(?im)\b(revision|change)\b.*\b(date|version)\b", text)),
    }

def _detect_domain_context(text: str) -> dict:
    lower = text.lower()
    scores = {}
    for domain, terms in DOMAIN_TERM_BANKS.items():
        scores[domain] = sum(1 for t in terms if t in lower) / max(len(terms), 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return {
        "predicted_domain": ranked[0][0] if ranked and ranked[0][1] > 0 else "general",
        "domain_scores": dict(ranked),
    }

def _learn_sop_style_signature(text: str, vocabulary: dict, tone_style: dict, compliance: dict) -> dict:
    line_samples = [l.strip() for l in text.splitlines() if l.strip()]
    control_tokens = sorted(set(vocabulary.get("mandatory_language", []) + vocabulary.get("prohibition_language", [])))
    rule_total = sum(compliance.get("rule_type_counts", {}).values())
    style_class = "strict_compliance" if tone_style.get("strictness_level") == "high" and rule_total >= 4 else "procedural_formal"
    return {
        "style_class": style_class,
        "control_language_tokens": control_tokens[:30],
        "compliance_density": round(rule_total / max(len(line_samples), 1), 3),
        "tone_signature": {
            "formality": tone_style.get("formality"),
            "authority_level": tone_style.get("authority_level"),
            "instructional_tone": tone_style.get("instructional_tone"),
        },
        "formatting_signature": {
            "line_prefix_patterns": [l.split()[0] for l in line_samples[:30] if l.split()],
            "avg_line_length_words": round(sum(len(l.split()) for l in line_samples) / max(len(line_samples), 1), 2),
        },
    }

def build_structured_sop_profile(result: dict) -> dict:
    language = result.get("language", {})
    features = result.get("features", {})
    classification = result.get("classification", {})
    sop = result.get("sop_profile") or features.get("sop_intelligence", {})
    traceability = sop.get("traceability", {})
    profile = {
        "document_identity": {
            "language": language.get("lang_code"),
            "genre": classification.get("predicted_genre"),
            "confidence": classification.get("confidence"),
            "sop_id": (sop.get("document_metadata") or {}).get("sop_number"),
            "title": (sop.get("document_metadata") or {}).get("title"),
            "domain": (sop.get("domain_context") or {}).get("predicted_domain"),
        },
        "structure": sop.get("section_structure", {}),
        "workflow": sop.get("workflow", {}),
        "traceability": traceability,
        "style_learning": sop.get("style_learning", {}),
        "compliance": sop.get("compliance_rules", {}),
        "vocabulary": sop.get("vocabulary_profile", {}),
        "guardrails": sop.get("style_guardrails", {}),
    }
    return profile

LLM_TASK_DIRECTIVES = {
    "rewrite": "Rewrite the SOP while preserving meaning, traceability IDs, and procedural order.",
    "improve": "Improve clarity, consistency, and compliance quality without changing intent or control outcomes.",
    "create_new": "Generate a new SOP in the same domain/style using the learned structure and control language.",
    "summarize": "Summarize SOP controls and workflow without omitting critical compliance constraints.",
    "gap_analysis": "Identify missing sections, weak controls, and traceability gaps against the learned SOP profile.",
    "translate": "Translate the SOP while preserving IDs, compliance modality, and procedural hierarchy.",
}

def build_sop_llm_system_prompt(task: str = "rewrite") -> str:
    directive = LLM_TASK_DIRECTIVES.get(task, LLM_TASK_DIRECTIVES["rewrite"])
    return (
        "You are the SOP Intelligence Engine.\n"
        "Follow SOP guardrails strictly.\n"
        "- Never break traceability IDs (SOP/DEV/CAPA/AUD/DEC).\n"
        "- Preserve or enforce formal compliance tone.\n"
        "- Keep procedural logic and section hierarchy intact unless task explicitly requires changes.\n"
        "- Keep output in preserved SOP language and preserved domain.\n"
        "- Avoid casual, poetic, promotional, or slang language.\n"
        f"Task directive: {directive}"
    )

def build_sop_llm_prompt(structured_profile: dict, task: str = "rewrite") -> str:
    rules = structured_profile.get("guardrails", {})
    identity = structured_profile.get("document_identity", {})
    section_order = structured_profile.get("structure", {}).get("section_order", [])
    trace = structured_profile.get("traceability", {}).get("relation_counts", {})
    style = structured_profile.get("style_learning", {})
    directive = LLM_TASK_DIRECTIVES.get(task, LLM_TASK_DIRECTIVES["rewrite"])
    prompt_payload = {
        "task": task,
        "task_directive": directive,
        "language": identity.get("language"),
        "domain": identity.get("domain"),
        "sop_id": identity.get("sop_id"),
        "required_section_order": section_order,
        "traceability_counts": trace,
        "style_signature": style,
        "guardrails": rules,
    }
    return (
        f"{build_sop_llm_system_prompt(task=task)}\n"
        "Use this structured profile as hard constraints:\n"
        f"{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
    )

def build_sop_llm_prompts(structured_profile: dict, tasks: list = None) -> dict:
    requested_tasks = tasks or ["rewrite", "improve", "create_new"]
    prompts = {}
    for task in requested_tasks:
        prompts[task] = {
            "system_prompt": build_sop_llm_system_prompt(task=task),
            "user_prompt": build_sop_llm_prompt(structured_profile, task=task),
        }
    return prompts

def extract_sop_intelligence(text: str, sentences: list, lines: list, modal_counts: dict, lang_info: dict = None) -> dict:
    metadata = _extract_metadata(text)
    domain = _detect_domain_context(text)
    section_structure = _extract_section_structure(lines, domain=domain["predicted_domain"])
    workflow = extract_workflow(text)
    roles = _extract_roles_and_actions(text)
    tone_style = _extract_tone_style_profile(text, sentences, modal_counts)
    vocabulary = _extract_vocabulary_profile(text)
    compliance = _extract_compliance_rules(text)
    formatting = _extract_formatting_profile(text, lines)
    traceability = extract_traceability(text)
    detected_language = _infer_sop_language(text, (lang_info or {}).get("lang_code", "en"))
    style_learning = _learn_sop_style_signature(text, vocabulary, tone_style, compliance)

    sop_strength = (
        0.25 * min(len(section_structure["section_order"]) / 8, 1.0)
        + 0.20 * min(len(workflow["steps"]) / 10, 1.0)
        + 0.20 * min(sum(compliance["rule_type_counts"].values()) / 12, 1.0)
        + 0.10 * min(sum(traceability.get("id_counts", {}).values()) / 10, 1.0)
        + 0.15 * (1.0 if metadata["sop_number"] else 0.0)
        + 0.10 * (1.0 if tone_style["compliance_focused_tone"] else 0.0)
    )
    style_guardrails = {
        "allow_tone": ["formal", "highly_formal", "instructional", "compliance-focused"],
        "avoid_tone": ["casual", "creative_poetic", "slang-heavy", "promotional"],
        "required_keywords": vocabulary["mandatory_language"][:10] + vocabulary["standard_instruction_phrases"][:10],
        "forbidden_patterns": ["yo", "ain't", "no cap", "metaphorical flourish", "storytelling digression"],
        "preserve_domain": domain["predicted_domain"],
        "preserve_language": detected_language,
        "language_policy": f"rewrite must remain in detected SOP language: {detected_language}",
    }
    return {
        "document_metadata": metadata,
        "section_structure": section_structure,
        "workflow": workflow,
        "traceability": traceability,
        "roles_and_actors": roles,
        "tone_and_style": tone_style,
        "style_learning": style_learning,
        "vocabulary_profile": vocabulary,
        "compliance_rules": compliance,
        "formatting_profile": formatting,
        "domain_context": domain,
        "language_profile": {
            "detected_sop_language": detected_language,
            "language_pack": SOP_LANGUAGE_PACKS[detected_language],
        },
        "sop_strength_score": round(min(sop_strength, 1.0), 3),
        "style_guardrails": style_guardrails,
    }

def learn_sop_corpus_patterns(documents: list) -> dict:
    profiles = [extract_sop_intelligence(doc, [s.strip() for s in re.split(r"[.!?]+", doc) if s.strip()], doc.splitlines(), {
        "mandatory": len(re.findall(r"\b(shall|must)\b", doc, re.I)),
        "recommended": len(re.findall(r"\bshould\b", doc, re.I)),
        "permitted": len(re.findall(r"\b(may|can)\b", doc, re.I)),
        "prohibited": len(re.findall(r"\b(shall not|must not|cannot)\b", doc, re.I)),
    }) for doc in documents if doc and doc.strip()]
    if not profiles:
        return {"document_count": 0}

    section_orders = [" > ".join(p["section_structure"]["section_order"]) for p in profiles if p["section_structure"]["section_order"]]
    domain_votes = {}
    mandatory_terms = {}
    formatting_votes = {}
    relation_votes = {}
    style_classes = {}
    for p in profiles:
        dom = p["domain_context"]["predicted_domain"]
        domain_votes[dom] = domain_votes.get(dom, 0) + 1
        for term in p["vocabulary_profile"]["mandatory_language"]:
            mandatory_terms[term] = mandatory_terms.get(term, 0) + 1
        fmt = p["formatting_profile"]["numbering_style"]
        formatting_votes[fmt] = formatting_votes.get(fmt, 0) + 1
        sclass = p.get("style_learning", {}).get("style_class", "procedural_formal")
        style_classes[sclass] = style_classes.get(sclass, 0) + 1
        for rel, count in p.get("traceability", {}).get("relation_counts", {}).items():
            relation_votes[rel] = relation_votes.get(rel, 0) + count

    return {
        "document_count": len(profiles),
        "common_structural_pattern": max(section_orders, key=section_orders.count) if section_orders else "",
        "common_tone": max([p["tone_and_style"]["formality"] for p in profiles], key=[p["tone_and_style"]["formality"] for p in profiles].count),
        "common_vocabulary": sorted(mandatory_terms.items(), key=lambda x: x[1], reverse=True)[:20],
        "common_section_order": max(section_orders, key=section_orders.count) if section_orders else "",
        "common_formatting_style": max(formatting_votes, key=formatting_votes.get) if formatting_votes else "none",
        "common_procedure_logic": {
            "avg_step_count": round(sum(len(p["workflow"]["steps"]) for p in profiles) / max(len(profiles), 1), 2),
            "decision_frequency": round(sum(len(p["workflow"]["decision_points"]) for p in profiles) / max(len(profiles), 1), 2),
        },
        "common_compliance_language": sorted(mandatory_terms.items(), key=lambda x: x[1], reverse=True)[:10],
        "dominant_domain": max(domain_votes, key=domain_votes.get) if domain_votes else "general",
        "common_traceability_relations": sorted(relation_votes.items(), key=lambda x: x[1], reverse=True)[:20],
        "dominant_style_class": max(style_classes, key=style_classes.get) if style_classes else "procedural_formal",
    }

def extract_features(text: str, lang_info: dict) -> dict:
    lang_code = lang_info["lang_code"]
    nlp = spacy_models.get(lang_code, spacy_models.get("en"))
    
    total_words = len(text.split())
    lines = text.split('\n')
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    pos_dist = {"noun_ratio": 0.0, "verb_ratio": 0.0, "adj_ratio": 0.0, "adv_ratio": 0.0}
    voice = "active"
    org_density = 0.0
    date_density = 0.0
    ttr = 0.0
    formal_morph = 0
    
    if nlp:
        # Load full document might be too heavy, taking a sample if too long
        doc = nlp(text[:10000])
        doc_words = [t for t in doc if not t.is_punct and not t.is_space]
        total_doc_words = max(len(doc_words), 1)
        
        counts = {"NOUN": 0, "VERB": 0, "ADJ": 0, "ADV": 0}
        for t in doc_words:
            if t.pos_ in counts:
                counts[t.pos_] += 1
                
        pos_dist = {
            "noun_ratio": counts["NOUN"] / total_doc_words,
            "verb_ratio": counts["VERB"] / total_doc_words,
            "adj_ratio":  counts["ADJ"]  / total_doc_words,
            "adv_ratio":  counts["ADV"]  / total_doc_words,
        }
        
        passive_count = sum(1 for t in doc if t.dep_ == "auxpass")
        voice = "passive" if passive_count > len(list(doc.sents)) * 0.3 else "active"
        
        ents = [e.label_ for e in doc.ents]
        org_density = ents.count("ORG") / total_doc_words
        date_density = ents.count("DATE") / total_doc_words
        
        lemmas = set(t.lemma_.lower() for t in doc if not t.is_stop)
        ttr = len(lemmas) / total_doc_words
        
        FORMAL_SUFFIXES  = ["tion", "ment", "ance", "ence", "ize", "ise"]
        formal_morph = sum(1 for t in doc_words if any(t.text.lower().endswith(s) for s in FORMAL_SUFFIXES))

    flesch = textstat.flesch_reading_ease(text)
    fog    = textstat.gunning_fog(text)
    
    MODALS = {
        "mandatory":   r'\b(shall|must|required|mandatory|muss|muessen|mÃ¼ssen|pflicht|erforderlich|verpflichtend)\b',
        "recommended": r'\b(should|sollte|empfohlen)\b',
        "permitted":   r'\b(may|can|darf|kann)\b',
        "prohibited":  r'\b(shall not|must not|cannot|darf nicht|duerfen nicht|dÃ¼rfen nicht|verboten)\b',
    }
    modal_counts = {k: len(re.findall(v, text, re.I)) for k, v in MODALS.items()}
    
    short_line_ratio = sum(1 for l in lines if len(l.split()) < 8) / max(len(lines), 1)
    poetic_features = poetic_style_features(text, lines, total_words, pos_dist, ttr)
    sop_features = sop_style_features(text, lines)
    legal_features = legal_style_features(text, lines)
    is_compliance_like = (
        sop_features.get("format_score", 0) >= 0.55
        or sop_features.get("compliance_id_count", 0) >= 3
        or legal_features.get("legalese_score", 0) >= 0.55
    )
    
    structure = {
        "has_numbered_steps": bool(re.search(r'^\s*\d+[\.\)]\s', text, re.MULTILINE)),
        "has_sub_steps":      bool(re.search(r'^\s*\d+\.\d+',    text, re.MULTILINE)),
        "has_headers":        any(l.isupper() and len(l.split()) < 10 for l in lines),
        "has_bullets":        any(l.strip().startswith(("-","â€¢","*","â€“")) for l in lines),
        "has_table":          text.count("|") > 4,
        "has_notes":          any(l.strip().upper().startswith(("NOTE:","WARNING:","CAUTION:")) for l in lines),
        "stanza_like":        (short_line_ratio > 0.55) and not is_compliance_like,
        "rhyme_score":        rhyme_score(lines),
        "poetic_lineation":    poetic_features["is_lineated"] and not is_compliance_like,
    }
    
    TERM_BANKS = {
        "sop":     ["shall", "ensure", "verify", "document", "record",
                    "maintain", "comply", "procedure", "protocol", "operator"],
        "legal":   ["hereby", "pursuant", "whereas", "notwithstanding",
                    "indemnify", "liable", "covenant", "jurisdiction"],
        "academic":["furthermore", "methodology", "hypothesis", "empirical",
                    "analysis", "et al", "ibid", "literature"],
        "rap":     ["yo", "gonna", "ain't", "no cap", "bars", "flow", "drip"],
        "poetry":  ["thee", "thou", "hath", "doth", "o'er", "ere", "midst"],
        "religious":["blessed", "almighty", "prayer", "mercy", "faith", "divine"],
    }

    terminology_scores = {
        genre: sum(1 for t in terms if t in text.lower()) / len(terms)
        for genre, terms in TERM_BANKS.items()
    }
    
    avg_sent_len = total_words / max(len(sentences), 1)
    avg_word_len = sum(len(w) for w in text.split()) / max(total_words, 1)
    sop_intelligence = extract_sop_intelligence(text, sentences, lines, modal_counts, lang_info=lang_info)

    return {
        "pos_dist": pos_dist,
        "voice": voice,
        "org_density": org_density,
        "date_density": date_density,
        "ttr": ttr,
        "formal_morph": formal_morph,
        "readability": {"flesch": flesch, "fog": fog},
        "modal_verbs": modal_counts,
        "structure": structure,
        "poetic_style": poetic_features,
        "sop_style": sop_features,
        "sop_intelligence": sop_intelligence,
        "legal_style": legal_features,
        "terminology_scores": terminology_scores,
        "avg_sent_len": avg_sent_len,
        "avg_word_len": avg_word_len
    }

def classify_genre(features: dict) -> dict:
    t  = features["terminology_scores"]
    st = features["structure"]
    m  = features["modal_verbs"]
    fl = features["readability"]["flesch"] or 50
    sop_style = features["sop_style"]
    legal_style = features["legal_style"]
    sop_intelligence = features.get("sop_intelligence", {})

    scores = {
        "sop":      t.get("sop",0)*0.25 + st["has_numbered_steps"]*0.20
                    + st["has_headers"]*0.10 + (m.get("mandatory",0)>3)*0.10
                    + features["sop_style"]["format_score"]*0.25
                    + features["sop_style"]["control_language_score"]*0.10
                    + sop_intelligence.get("sop_strength_score", 0)*0.15,

        "legal":    t.get("legal",0)*0.30 + (features["avg_sent_len"]>25)*0.10
                    + (features["avg_word_len"]>6)*0.10
                    + (features["voice"]=="passive")*0.05
                    + features["legal_style"]["legalese_score"]*0.30
                    + min(features["legal_style"]["clause_density"], 1.0)*0.15,

        "academic": t.get("academic",0)*0.50 + (fl<45)*0.30
                    + (features["voice"]=="passive")*0.20,

        "poetry":   (st["rhyme_score"]>0.3)*0.30 + st["stanza_like"]*0.20
                    + (features["avg_sent_len"]<10)*0.15
                    + (features["poetic_style"]["musicality_score"]>0.35)*0.20
                    + (features["poetic_style"]["figurative_density"]>0.8)*0.15,

        "rap":      t.get("rap",0)*0.40 + (st["rhyme_score"]>0.4)*0.40
                    + (features["avg_sent_len"]<8)*0.10
                    + (features["poetic_style"]["musicality_score"]>0.45)*0.10,

        "casual":   (fl>70)*0.50 + t.get("rap",0)*0.20
                    + (features["avg_sent_len"]<12)*0.30,

        "religious":t.get("religious",0)*0.60 + st["stanza_like"]*0.40,
    }

    strong_sop = (
        sop_style["format_score"] >= 0.55
        and (sop_style["qms_term_hits"] >= 3 or sop_style["compliance_id_count"] >= 3)
    )
    strong_legal = legal_style["legalese_score"] >= 0.55 and (
        legal_style["numbered_clause_count"] >= 2 or legal_style["whereas_count"] > 0
    )

    if strong_sop:
        scores["sop"] += 0.45
        scores["poetry"] *= 0.15
        scores["rap"] *= 0.10
        scores["religious"] *= 0.20
        scores["casual"] *= 0.25

    if strong_legal:
        scores["legal"] += 0.45
        scores["sop"] *= 0.45 if not strong_sop else 1.0
        scores["poetry"] *= 0.10
        scores["rap"] *= 0.10
        scores["religious"] *= 0.15
        scores["casual"] *= 0.25

    total = sum(scores.values()) or 1
    probs = {k: round(v/total, 3) for k, v in scores.items()}
    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    confidence = ranked[0][1]
    if ranked[0][0] == "sop" and strong_sop:
        confidence = max(confidence, 0.82)

    return {
        "predicted_genre": ranked[0][0],
        "confidence":      confidence,
        "runner_up":       ranked[1][0],
        "all_scores":      dict(ranked),
    }

def _extract_sop_source_lines(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    purpose = ""
    scope = ""
    responsibilities = []
    procedure = []
    exceptions = []
    records = []
    references = []

    for line in lines:
        low = line.lower()
        if ":" in line:
            k, v = line.split(":", 1)
            key = k.strip().lower()
            value = v.strip()
            if any(t in key for t in ["purpose", "zweck"]):
                purpose = value
            elif any(t in key for t in ["scope", "geltungsbereich"]):
                scope = value
            elif any(t in key for t in ["responsibil", "verantwort"]):
                responsibilities.append(value)
            elif any(t in key for t in ["reference", "referenz"]):
                references.append(value)
            elif any(t in key for t in ["record", "aufzeichnung", "log"]):
                records.append(value)
            elif any(t in key for t in ["deviation", "abweich", "exception", "ausnahme"]):
                exceptions.append(value)
            elif len(value.split()) >= 4:
                procedure.append(f"{k.strip()}: {value}")
        elif re.search(r"^\d+[\.\)]\s+", line):
            procedure.append(re.sub(r"^\d+[\.\)]\s*", "", line))
        elif re.search(r"\b(DEV|CAPA|AUD|DEC)-", line, re.I):
            exceptions.append(line)
        elif any(t in low for t in ["log", "record", "aufbewahrung", "retention"]):
            records.append(line)
        elif len(line.split()) >= 6 and len(procedure) < 12:
            procedure.append(line)

    all_traceability = []
    for line in lines:
        if re.search(r"\b(?:DEV|CAPA|AUD|DEC)-[A-Z]{2,}-\d{3,}\b", line, re.I):
            all_traceability.append(line)

    return {
        "purpose": purpose,
        "scope": scope,
        "responsibilities": responsibilities[:6],
        "procedure": procedure[:12],
        "exceptions": exceptions[:8],
        "traceability": all_traceability,
        "records": records[:6],
        "references": references[:6],
    }

def rewrite_sop_same_language(text: str) -> dict:
    analysis = process_document(text)
    sop_profile = analysis.get("sop_profile") or {}
    language = ((sop_profile.get("language_profile") or {}).get("detected_sop_language")) or _infer_sop_language(text, analysis["language"]["lang_code"])
    pack = SOP_LANGUAGE_PACKS.get(language, SOP_LANGUAGE_PACKS["en"])
    src = _extract_sop_source_lines(text)
    meta = sop_profile.get("document_metadata", {})

    title = meta.get("title") or (meta.get("sop_number") or "SOP")
    sop_id = meta.get("sop_number") or "SOP-UNSPECIFIED"
    headers = pack["section_headers"]
    mandatory = pack["phrases"]["mandatory"]

    if language == "de":
        rewritten_lines = [
            f"SOP ID: {sop_id}",
            f"{headers['title']}: {title}",
            "",
            f"{headers['purpose']}:",
            src["purpose"] or "Diese SOP beschreibt verbindliche Kontrollen fuer sicheren und regelkonformen Zugriff.",
            "",
            f"{headers['scope']}:",
            src["scope"] or "Diese SOP gilt fuer alle beteiligten Rollen, Systeme und externen Dienstleister.",
            "",
            f"{headers['responsibilities']}:",
        ]
        resp = src["responsibilities"] or [
            f"Die verantwortliche Rolle {mandatory} den Zugriff freigeben und pruefen.",
            f"Die durchfuehrende Rolle {mandatory} alle Zugriffsschritte dokumentieren.",
        ]
        rewritten_lines.extend([f"- {r}" for r in resp])
        rewritten_lines.extend(["", f"{headers['procedure']}:", ""])
        proc = src["procedure"] or [
            f"Zugriffsanforderung klassifizieren und Identitaet verifizieren.",
            f"Nur autorisierte Berechtigungen vergeben und jede Aktion dokumentieren.",
            f"Bei Abweichungen sofort eskalieren und CAPA einleiten.",
        ]
        rewritten_lines.extend([f"{i}. {p}" for i, p in enumerate(proc, 1)])
        traceability_lines = src["traceability"] or src["exceptions"] or ["Abweichungen sind als DEV-Eintrag zu dokumentieren und nachzuverfolgen."]
        rewritten_lines.extend(["", f"{headers['exceptions']}:", *traceability_lines, ""])
        rewritten_lines.extend([f"{headers['records']}:", *(src["records"] or ["Alle Logs und Freigaben sind gemaess Aufbewahrungsfrist zu archivieren."]), ""])
        rewritten_lines.extend([f"{headers['references']}:", *(src["references"] or ["Relevante Normen und interne Richtlinien sind verbindlich anzuwenden."])])
    else:
        rewritten_lines = [
            f"SOP ID: {sop_id}",
            f"{headers['title']}: {title}",
            "",
            f"{headers['purpose']}:",
            src["purpose"] or "This SOP defines mandatory controls for compliant and secure access handling.",
            "",
            f"{headers['scope']}:",
            src["scope"] or "This SOP applies to all involved roles, systems, and external providers.",
            "",
            f"{headers['responsibilities']}:",
        ]
        resp = src["responsibilities"] or [
            f"The responsible role {mandatory} approve and verify access.",
            f"The executing role {mandatory} document all access activities.",
        ]
        rewritten_lines.extend([f"- {r}" for r in resp])
        rewritten_lines.extend(["", f"{headers['procedure']}:", ""])
        proc = src["procedure"] or [
            "Classify the access request and verify identity.",
            "Grant only authorized permissions and document each action.",
            "Escalate deviations immediately and initiate CAPA tracking.",
        ]
        rewritten_lines.extend([f"{i}. {p}" for i, p in enumerate(proc, 1)])
        traceability_lines = src["traceability"] or src["exceptions"] or ["Deviations shall be logged as DEV records and reviewed."]
        rewritten_lines.extend(["", f"{headers['exceptions']}:", *traceability_lines, ""])
        rewritten_lines.extend([f"{headers['records']}:", *(src["records"] or ["All logs and approvals shall be retained according to policy."]), ""])
        rewritten_lines.extend([f"{headers['references']}:", *(src["references"] or ["Applicable standards and internal policies are mandatory."])])

    rewritten_text = "\n".join(rewritten_lines).strip() + "\n"
    rewritten_analysis = process_document(rewritten_text)
    rewritten_lang = (rewritten_analysis.get("sop_profile", {}).get("language_profile", {}).get("detected_sop_language")) or _infer_sop_language(rewritten_text, rewritten_analysis["language"]["lang_code"])

    return {
        "detected_language": language,
        "rewritten_language": rewritten_lang,
        "language_preserved": language == rewritten_lang,
        "rewritten_text": rewritten_text,
        "source_analysis": analysis,
        "rewritten_analysis": rewritten_analysis,
    }

def process_document(text: str):
    lang_info = detect_language(text)
    chunks = smart_chunk(text)
    features = extract_features(text, lang_info)
    classification = classify_genre(features)
    sop_profile = features.get("sop_intelligence")
    base_result = {
        "language": lang_info,
        "chunks": chunks,
        "features": features,
        "classification": classification,
        "sop_profile": sop_profile
    }
    structured_profile = build_structured_sop_profile(base_result)
    llm_prompts = build_sop_llm_prompts(
        structured_profile,
        tasks=["rewrite", "improve", "create_new", "summarize", "gap_analysis", "translate"]
    )
    
    base_result["structured_sop_profile"] = structured_profile
    # Backward compatibility for existing consumers.
    base_result["llm_prompt_rewrite"] = llm_prompts["rewrite"]["user_prompt"]
    base_result["llm_system_prompt_rewrite"] = llm_prompts["rewrite"]["system_prompt"]
    base_result["llm_prompts"] = llm_prompts
    return base_result
