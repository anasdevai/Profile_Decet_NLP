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
    r'^(\d+[\.\)]\s+[A-Z].+|[A-Z][A-Z\s]{3,})$',
    re.MULTILINE
)

class DocumentChunk:
    def __init__(self, title, content, section_type="general"):
        self.section_title = title
        self.content = content
        self.section_type = section_type

def smart_chunk(text: str):
    parts = HEADER_PATTERN.split(text)
    if len(parts) > 2:
        # Strategy A: Header-based
        chunks = []
        current_header = "Intro"
        if parts[0].strip():
            chunks.append(DocumentChunk(current_header, parts[0].strip()))
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
        chunk_words = 500
        overlap = 50
        for i in range(0, len(words), chunk_words - overlap):
            content = " ".join(words[i:i + chunk_words])
            chunks.append(DocumentChunk(f"Chunk {i//(chunk_words - overlap) + 1}", content, "general"))
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
    bullet_count = len(re.findall(r"^\s*(?:[-*]|[A-Z]+-[A-Z]+-\d{3}\s*[–-])", text, re.I | re.M))
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
        "mandatory":   r'\b(shall|must)\b',
        "recommended": r'\bshould\b',
        "permitted":   r'\b(may|can)\b',
        "prohibited":  r'\b(shall not|must not|cannot)\b',
    }
    modal_counts = {k: len(re.findall(v, text, re.I)) for k, v in MODALS.items()}
    
    short_line_ratio = sum(1 for l in lines if len(l.split()) < 8) / max(len(lines), 1)
    poetic_features = poetic_style_features(text, lines, total_words, pos_dist, ttr)
    sop_features = sop_style_features(text, lines)
    legal_features = legal_style_features(text, lines)
    
    structure = {
        "has_numbered_steps": bool(re.search(r'^\s*\d+[\.\)]\s', text, re.MULTILINE)),
        "has_sub_steps":      bool(re.search(r'^\s*\d+\.\d+',    text, re.MULTILINE)),
        "has_headers":        any(l.isupper() and len(l.split()) < 10 for l in lines),
        "has_bullets":        any(l.strip().startswith(("-","•","*","–")) for l in lines),
        "has_table":          text.count("|") > 4,
        "has_notes":          any(l.strip().upper().startswith(("NOTE:","WARNING:","CAUTION:")) for l in lines),
        "stanza_like":        short_line_ratio > 0.55,
        "rhyme_score":        rhyme_score(lines),
        "poetic_lineation":    poetic_features["is_lineated"],
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

    scores = {
        "sop":      t.get("sop",0)*0.25 + st["has_numbered_steps"]*0.20
                    + st["has_headers"]*0.10 + (m.get("mandatory",0)>3)*0.10
                    + features["sop_style"]["format_score"]*0.25
                    + features["sop_style"]["control_language_score"]*0.10,

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

    return {
        "predicted_genre": ranked[0][0],
        "confidence":      ranked[0][1],
        "runner_up":       ranked[1][0],
        "all_scores":      dict(ranked),
    }

def process_document(text: str):
    lang_info = detect_language(text)
    chunks = smart_chunk(text)
    features = extract_features(text, lang_info)
    classification = classify_genre(features)
    
    return {
        "language": lang_info,
        "chunks": chunks,
        "features": features,
        "classification": classification
    }
