---
name: style-detection-engine
description: >
  Full pipeline skill for detecting, analyzing, and rewriting documents based on their
  writing style, tone, structure, vocabulary, and language. Use this skill whenever the
  user wants to: analyze writing style of any document (SOP, QMS, legal contract, poem,
  rap, academic paper, casual email, religious text, etc.), detect language and genre,
  rewrite or improve a document while preserving its exact style, generate a new document
  in the style of existing ones, or build a style-aware document engine with NLP +
  embeddings + LLM. Trigger this skill even if the user just says "analyze this doc",
  "rewrite this in same style", "what kind of writing is this", "improve my SOP",
  "write a new SOP like my existing ones", or uploads any document and asks about its
  style, tone, or structure. This skill covers the FULL pipeline: language detection →
  NLP feature extraction → statistical analysis → embedding → Qdrant storage → genre
  classification → LLM confirmation → style-aware rewrite/generation.
---

# Style Detection & Rewriting Engine

A production-grade hybrid pipeline that recognizes ANY writing style — SOP, legal
contract, poetry, rap, academic, religious, casual — in ANY language, then rewrites
or generates new content in that exact style using NLP + embeddings + Gemini LLM.

## Tech Stack
- **NLP**: spaCy (en, de, fr, es models) + textstat + pronouncing + langdetect
- **Embeddings**: LangChain + `BAAI/bge-small-en-v1.5` (multilingual fallback: `paraphrase-multilingual-MiniLM-L12-v2`)
- **Vector Store**: Qdrant
- **LLM**: Gemini 2.0 Flash (via `langchain-google-genai`)
- **Backend**: FastAPI
- **DB**: SQLite for local development (PostgreSQL + Redis are planned production options)

---

## Pipeline Overview

```
Input Document(s)
      ↓
[STAGE 1]  Language Detection          → langdetect + script analysis
      ↓
[STAGE 2]  Smart Chunking              → header-based or size-based
      ↓
[STAGE 3]  NLP Feature Extraction      → spaCy POS, deps, NER, morphology
      ↓
[STAGE 4]  Statistical Analysis        → readability, rhyme, structure, modals
      ↓
[STAGE 5]  Embedding + Qdrant Store    → BGE-small → Qdrant with style payload
      ↓
[STAGE 6]  Dynamic Threshold Check     → genre-specific confidence thresholds
      ↓
[STAGE 7]  Math Genre Classifier       → weighted scoring, no LLM
      ↓
[STAGE 8]  Conditional LLM (Gemini)    → only if confidence below threshold
      ↓
[STAGE 9]  Style Profile Output        → JSON profile stored in SQLite
      ↓
[STAGE 10] Chunk-wise Rewriter         → style-locked, per-section rewrite
      ↓
[STAGE 11] Feedback Loop               → user corrections → profile update
      ↓
[STAGE 12] Runtime Learning            → new docs update Qdrant + term banks
```

Read the detailed reference files for each stage:
- `references/nlp-techniques.md` — All NLP methods, code, and when each fires
- `references/embedding-qdrant.md` — Embedding model, Qdrant schema, retrieval
- `references/genre-classifier.md` — Weighted scoring formulas per genre
- `references/llm-prompts.md` — Gemini prompt templates for confirm + rewrite
- `references/feedback-learning.md` — Feedback loop + runtime learning logic
- `references/language-support.md` — Per-language NLP routing table

---

## Quick Decision Guide

| User Request | What to Do |
|---|---|
| "Analyze this document" | Run Stages 1–9, return style profile JSON |
| "Rewrite this doc" | Run Stages 1–9, then Stage 10 |
| "Improve this SOP" | Run Stages 1–9, Stage 10 with `mode=improve` |
| "Write a new SOP like my existing ones" | Embed context → Qdrant retrieve → Stage 10 generate |
| "What language/genre is this?" | Run Stages 1–7 only, return classification |
| "This rewrite is wrong" | Run Stage 11 (feedback loop) |

---

## STAGE 1 — Language Detection

**Tools**: `langdetect`, Unicode script analysis, spaCy model router

```python
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

def detect_language(text: str) -> dict:
    lang_code = detect(text)

    # Script-level analysis
    urdu_arabic = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    latin       = sum(1 for c in text if c.isascii() and c.isalpha())
    total       = max(len(text), 1)

    is_mixed         = (urdu_arabic/total > 0.1) and (latin/total > 0.1)
    has_urdu_script  = urdu_arabic/total > 0.3

    # spaCy model routing
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
```

**Routing rules**:
- `en/de/fr/es` → full NLP pipeline (spaCy)
- `ur/hi/ar` → skip spaCy, use statistical + LLM only
- Mixed/unknown → statistical only + always call LLM

---

## STAGE 2 — Smart Chunking

**Strategy A — Header-based** (SOP, Legal, Academic):
```python
import re

HEADER_PATTERN = re.compile(
    r'^(\d+[\.\)]\s+[A-Z].+|[A-Z][A-Z\s]{3,})$',
    re.MULTILINE
)

def chunk_by_headers(text):
    parts = HEADER_PATTERN.split(text)
    # Returns list of (section_title, content) tuples
```

**Strategy B — Size-based** (Prose, Poetry, Rap):
```python
def chunk_by_size(text, chunk_words=500, overlap=50):
    words = text.split()
    for i in range(0, len(words), chunk_words - overlap):
        yield " ".join(words[i:i + chunk_words])
```

**Selection logic**: If `len(header_split) > 2` → Strategy A, else Strategy B.

**Section type tagging**: Each chunk tagged as one of:
`purpose | scope | definitions | responsibilities | procedure | references | revision | appendix | prose | stanza | verse | general`

---

## STAGE 3 — NLP Feature Extraction

> **Full NLP technique details → see `references/nlp-techniques.md`**

### 3a. POS Tagging (spaCy)
```python
doc = nlp(text)
words = [t for t in doc if not t.is_punct and not t.is_space]
pos_dist = {
    "noun_ratio": count("NOUN") / total,   # High → technical/SOP/legal
    "verb_ratio": count("VERB") / total,   # High → narrative/procedure
    "adj_ratio":  count("ADJ")  / total,   # High → poetry/descriptive
    "adv_ratio":  count("ADV")  / total,   # High → casual/informal
}
```

### 3b. Dependency Parsing — Voice Detection
```python
passive_count = sum(1 for t in doc if t.dep_ == "auxpass")
voice = "passive" if passive_count > len(sentences) * 0.3 else "active"
# passive → legal/SOP/academic
# active  → casual/rap/fiction
```

### 3c. Named Entity Recognition
```python
ents = [(e.text, e.label_) for e in doc.ents]
org_density  = count("ORG")  / total_words  # High → legal/corporate
date_density = count("DATE") / total_words  # High → contracts
```

### 3d. Lemmatization + Vocabulary Richness
```python
lemmas = set(t.lemma_.lower() for t in doc if not t.is_stop)
ttr = len(lemmas) / max(total_words, 1)
# High TTR → academic/legal (rich vocabulary)
# Low TTR  → rap/chat (repetitive)
```

### 3e. Morphological Analysis
```python
FORMAL_SUFFIXES  = ["tion", "ment", "ance", "ence", "ize", "ise"]
CASUAL_SUFFIXES  = ["nna", "ta", "ya", "in'"]

formal_morph = sum(1 for t in doc
    if any(t.text.lower().endswith(s) for s in FORMAL_SUFFIXES))
```

---

## STAGE 4 — Statistical Analysis

### 4a. Readability (textstat)
```python
import textstat
flesch = textstat.flesch_reading_ease(text)
fog    = textstat.gunning_fog(text)
# flesch < 40  → legal/academic/SOP
# flesch 40-70 → technical/news
# flesch > 70  → casual/rap/chat
```

### 4b. Phonological / Rhyme Detection
```python
import pronouncing

def rhyme_score(lines):
    last_words = [l.strip().split()[-1].lower() for l in lines if l.strip()]
    hits = 0
    for i in range(len(last_words)-1):
        pa = pronouncing.phones_for_word(last_words[i])
        pb = pronouncing.phones_for_word(last_words[i+1])
        if pa and pb:
            if pronouncing.rhyming_part(pa[0]) == pronouncing.rhyming_part(pb[0]):
                hits += 1
    return hits / max(len(last_words)-1, 1)
# score > 0.6 → poetry/rap
# score < 0.2 → prose
```

**Fallback rhyme** (for non-English — last 2 chars match):
```python
if last_words[i][-2:] == last_words[i+1][-2:]:
    hits += 1
```

### 4c. Modal Verb Analysis (SOP compliance)
```python
MODALS = {
    "mandatory":   r'\b(shall|must)\b',
    "recommended": r'\bshould\b',
    "permitted":   r'\b(may|can)\b',
    "prohibited":  r'\b(shall not|must not|cannot)\b',
}
modal_counts = {k: len(re.findall(v, text, re.I)) for k, v in MODALS.items()}
# mandatory > 5 → ISO-compliant SOP
```

### 4d. Structure Detection
```python
structure = {
    "has_numbered_steps": bool(re.search(r'^\s*\d+[\.\)]\s', text, re.M)),
    "has_sub_steps":      bool(re.search(r'^\s*\d+\.\d+',    text, re.M)),
    "has_headers":        any(l.isupper() and len(l.split()) < 10 for l in lines),
    "has_bullets":        any(l.strip().startswith(("-","•","*","–")) for l in lines),
    "has_table":          text.count("|") > 4,
    "has_notes":          any(l.strip().upper().startswith(("NOTE:","WARNING:","CAUTION:")) for l in lines),
    "stanza_like":        short_line_ratio > 0.55,
    "rhyme_score":        rhyme_score(lines),
}
```

### 4e. Terminology Scoring
```python
# See references/genre-classifier.md for full term banks
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
```

---

## STAGE 5 — Embedding + Qdrant

> **Full schema + retrieval logic → see `references/embedding-qdrant.md`**

```python
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Embedding model
embedder = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Multilingual fallback
multilingual_embedder = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

def get_embedder(lang_code: str):
    return multilingual_embedder if lang_code not in ["en"] else embedder

# Qdrant payload structure per chunk:
payload = {
    "doc_id":         "sop_001_chunk_3",
    "doc_name":       "Chemical_Storage_SOP.pdf",
    "genre":          "SOP",
    "tone":           "authoritative",
    "voice":          "imperative",
    "structure":      "numbered_steps",
    "wording_style":  "ISO_compliant",
    "language":       "en",
    "style_rules":    ["Use shall for mandatory", "..."],
    "do_not_change":  ["numbered structure", "passive voice"],
    "chunk_text":     "actual section text...",
    "section_type":   "procedure",
    "org_id":         "abc_pharma",
    "feedback_score": 5.0,
    "version":        1,
}
```

**Retrieval for new doc / context**:
```python
def retrieve_similar_styles(query_text, genre_filter=None, top_k=5):
    vector = embedder.embed_query(query_text)
    filters = {"must": [{"key": "genre", "match": {"value": genre_filter}}]} \
              if genre_filter else None
    return qdrant.search(
        collection_name="style_profiles",
        query_vector=vector,
        query_filter=filters,
        limit=top_k,
        with_payload=True
    )
```

---

## STAGE 6 — Dynamic Confidence Thresholds

```python
GENRE_THRESHOLDS = {
    "sop":      0.80,   # High — compliance critical
    "legal":    0.82,   # Highest — wrong style = wrong rewrite
    "academic": 0.75,   # Medium — stats usually reliable
    "poetry":   0.65,   # Lower — rhyme score strong enough
    "rap":      0.62,   # Lower — slang + rhyme = clear
    "casual":   0.60,   # Low — easy to detect
    "ur":       0.45,   # Very low — always call LLM for Urdu
    "hi":       0.50,   # Low — Stanza weak, LLM preferred
    "default":  0.70,
}

def get_threshold(genre: str, lang_code: str) -> float:
    # Urdu/Hindi → lower threshold (always LLM)
    if lang_code in ["ur", "hi", "ar"]:
        return GENRE_THRESHOLDS.get(lang_code, 0.45)
    return GENRE_THRESHOLDS.get(genre, GENRE_THRESHOLDS["default"])
```

**Auto-calibration**: After 50 user corrections on a genre → threshold raised by +2%.
Stored in the configured SQL database threshold table.

---

## STAGE 7 — Math Genre Classifier

> **Full scoring formulas → see `references/genre-classifier.md`**

```python
def classify_genre(features: dict) -> dict:
    t  = features["terminology_scores"]
    st = features["structure"]
    m  = features["modal_verbs"]
    fl = features["readability"]["flesch"] or 50

    scores = {
        "sop":      t.get("sop",0)*0.40 + st["has_numbered_steps"]*0.30
                    + st["has_headers"]*0.15 + (m.get("mandatory",0)>5)*0.15,

        "legal":    t.get("legal",0)*0.50 + (features["avg_sent_len"]>25)*0.20
                    + (features["avg_word_len"]>6)*0.20
                    + (features["voice"]=="passive")*0.10,

        "academic": t.get("academic",0)*0.50 + (fl<45)*0.30
                    + (features["voice"]=="passive")*0.20,

        "poetry":   (st["rhyme_score"]>0.3)*0.50 + st["stanza_like"]*0.30
                    + (features["avg_sent_len"]<10)*0.20,

        "rap":      t.get("rap",0)*0.40 + (st["rhyme_score"]>0.4)*0.40
                    + (features["avg_sent_len"]<8)*0.20,

        "casual":   (fl>70)*0.50 + t.get("rap",0)*0.20
                    + (features["avg_sent_len"]<12)*0.30,

        "religious":t.get("religious",0)*0.60 + st["stanza_like"]*0.40,
    }

    total = sum(scores.values()) or 1
    probs = {k: round(v/total, 3) for k, v in scores.items()}
    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    return {
        "predicted_genre": ranked[0][0],
        "confidence":      ranked[0][1],
        "runner_up":       ranked[1][0],
        "all_scores":      dict(ranked),
    }
```

---

## STAGE 8 — Conditional LLM (Gemini)

> **Full prompt templates → see `references/llm-prompts.md`**

**Only call Gemini when**: `confidence < genre_threshold`

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.1,
    google_api_key=GEMINI_API_KEY
)

def confirm_with_llm(text_sample: str, features: dict, prediction: dict) -> dict:
    # Send only 800 words + features — NOT full document
    prompt = CONFIRM_PROMPT_TEMPLATE.format(
        sample=text_sample[:800],
        features=json.dumps(features, indent=2),
        prediction=prediction
    )
    response = llm.invoke(prompt)
    return json.loads(response.content)
```

**Gemini is also used for**:
- Final style profile enrichment (key_style_rules, do_not_change)
- Rewriting (Stage 10) — per chunk
- New document generation with style context

---

## STAGE 9 — Style Profile Output

```python
# Final style profile JSON — stored in SQLite for local development
style_profile = {
    "genre":           "SOP",
    "confidence":      0.91,
    "tone":            "authoritative",
    "voice":           "imperative",
    "vocabulary_type": "technical_jargon",
    "structure":       "numbered_steps",
    "wording_style":   "ISO_compliant",
    "language":        "en",
    "llm_used":        False,
    "threshold_used":  0.80,

    # NLP-derived
    "avg_sentence_len": 18.4,
    "flesch_score":     34.2,
    "fog_index":        16.8,
    "rhyme_score":      0.02,
    "modal_verbs":      {"mandatory": 14, "recommended": 3, "permitted": 1},
    "pos_distribution": {"nouns": 0.38, "verbs": 0.22, "adj": 0.11},

    # LLM-enriched
    "key_style_rules": [
        "Use 'shall' for all mandatory actions",
        "Numbered main steps, sub-steps as 1.1, 1.2",
        "Section headers in ALL CAPS",
        "Imperative voice throughout procedures",
        "NOTE: prefix for all callouts"
    ],
    "do_not_change": [
        "numbered structure",
        "shall/must modal usage",
        "ALL CAPS headers",
        "ISO section order"
    ],

    # SOP-specific (if genre == SOP)
    "sop_analysis": {
        "sections_found":    ["purpose","scope","procedure","revision"],
        "sections_missing":  ["definitions","references"],
        "completeness_score": 0.71,
        "wording_compliance": "ISO_compliant",
        "terminology_gaps":   ["CAPA", "deviation", "NCR"]
    },

    # Meta
    "doc_id":            "sop_001",
    "org_id":            "abc_pharma",
    "version":           1,
    "feedback_corrections": 0,
    "qdrant_point_ids":  ["uuid1", "uuid2", "uuid3"],
}
```

---

## STAGE 10 — Chunk-wise Style-Aware Rewriter

```python
REWRITE_MODES = {
    "improve":  "Improve clarity, flow, and quality",
    "rewrite":  "Completely rewrite in your own words",
    "expand":   "Expand with more detail and examples",
    "shorten":  "Compress while preserving all key points",
    "generate": "Generate new content based on context",
}

def rewrite_chunk(chunk_text, section_title, style_profile,
                  similar_docs, mode="improve"):
    prompt = REWRITE_PROMPT_TEMPLATE.format(
        mode=REWRITE_MODES[mode],
        genre=style_profile["genre"],
        tone=style_profile["tone"],
        voice=style_profile["voice"],
        vocabulary=style_profile["vocabulary_type"],
        style_rules="\n".join(f"- {r}" for r in style_profile["key_style_rules"]),
        do_not_change="\n".join(f"- {r}" for r in style_profile["do_not_change"]),
        reference_doc_1=similar_docs[0].payload["chunk_text"] if similar_docs else "",
        reference_doc_2=similar_docs[1].payload["chunk_text"] if len(similar_docs)>1 else "",
        section_title=section_title,
        chunk_text=chunk_text,
    )
    return llm.invoke(prompt).content

def rewrite_full_document(chunks, style_profile, mode="improve"):
    similar = retrieve_similar_styles(
        chunks[0].content,
        genre_filter=style_profile["genre"],
        top_k=3
    )
    sections = [
        rewrite_chunk(c.content, c.section_title, style_profile, similar, mode)
        for c in chunks
    ]
    return "\n\n".join(sections)
```

---

## STAGE 11 — Feedback Loop

> **Full feedback schema → see `references/feedback-learning.md`**

```python
def apply_feedback(doc_id: str, correction: dict):
    """
    correction = {
        "wrong_genre":      "academic",
        "correct_genre":    "SOP",
        "wrong_tone":       "formal",
        "correct_tone":     "authoritative",
        "style_rules_add":  ["Use 'shall' not 'must'"],
        "do_not_change_add":["ALL CAPS headers"],
        "rating":           2,  # 1-5
    }
    """
    # 1. Update style profile
    profile = db.get_profile(doc_id)
    profile["genre"]  = correction.get("correct_genre", profile["genre"])
    profile["tone"]   = correction.get("correct_tone",  profile["tone"])
    profile["key_style_rules"]  += correction.get("style_rules_add", [])
    profile["do_not_change"]    += correction.get("do_not_change_add", [])
    profile["version"]          += 1
    profile["feedback_corrections"] += 1
    db.save_profile(profile)

    # 2. Update Qdrant payload
    qdrant.set_payload(
        collection_name="style_profiles",
        payload={"genre": profile["genre"], "tone": profile["tone"]},
        points=profile["qdrant_point_ids"]
    )

    # 3. Recalibrate threshold
    corrections_count = db.count_corrections(profile["genre"])
    if corrections_count % 50 == 0:
        GENRE_THRESHOLDS[profile["genre"]] = min(
            GENRE_THRESHOLDS[profile["genre"]] + 0.02, 0.95
        )
        db.save_threshold(profile["genre"], GENRE_THRESHOLDS[profile["genre"]])
```

---

## STAGE 12 — Runtime Learning

Every new document automatically:
1. Gets analyzed and embedded
2. Stored in Qdrant (becomes a reference for future similarity search)
3. Enriches terminology banks if new domain terms found
4. Updates the multi-doc master profile for its org

```python
def ingest_new_document(text, org_id, doc_name):
    lang    = detect_language(text)
    chunks  = smart_chunk(text)
    profile = full_pipeline(text, lang)

    for i, chunk in enumerate(chunks):
        vector = get_embedder(lang["lang_code"]).embed_query(chunk.content)
        qdrant.upsert(collection_name="style_profiles", points=[
            PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload={**profile, "chunk_text": chunk.content,
                         "doc_name": doc_name, "org_id": org_id}
            )
        ])

    # Update org master profile
    update_master_profile(org_id, profile)

    # Enrich terminology banks with new domain terms
    enrich_term_banks(text, profile["genre"])
```

---

## FastAPI Endpoints

```python
POST /analyze          → Returns full style profile JSON
POST /rewrite          → Rewrites document preserving style
POST /improve          → Improves quality, locks style
POST /generate         → Generates new doc from context + style
POST /feedback         → Applies user correction to profile
POST /ingest           → Indexes document into Qdrant
GET  /profile/{doc_id} → Returns saved style profile
GET  /similar/{doc_id} → Returns similar style documents
```

---

## Environment Variables Required

```env
GEMINI_API_KEY=your_key_here
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=style_profiles
DATABASE_URL=sqlite:///./style.db
# POSTGRES_URL=postgresql://user:pass@localhost/styledb
# REDIS_URL=redis://localhost:6379
```

---

## Installation

```bash
pip install spacy textstat pronouncing langdetect
pip install langchain langchain-google-genai langchain-community
pip install qdrant-client sentence-transformers
pip install fastapi uvicorn sqlalchemy
pip install python-dotenv

python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
python -m spacy download fr_core_news_sm
python -m spacy download es_core_news_sm
```

---

## Reference Files Index

| File | Contents |
|---|---|
| `references/nlp-techniques.md` | All 10 NLP techniques with full code |
| `references/embedding-qdrant.md` | BGE model setup, Qdrant schema, retrieval |
| `references/genre-classifier.md` | Full term banks, scoring formulas |
| `references/llm-prompts.md` | All Gemini prompt templates |
| `references/feedback-learning.md` | Feedback schema, DB tables, calibration |
| `references/language-support.md` | Language routing table, multilingual notes |
