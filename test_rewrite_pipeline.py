import sys
import nlp_pipeline

TEXT = """SOP ID + Titel: SOP-IT-001 - Zugriffsmanagement auf Produktionsnetzwerk (OT)
LEFT (SOP CONTENT):
Zweck: Nur autorisierte Personen (intern/extern/KI) duerfen aufs OT-Netz. Trennung IT/OT nach IEC 62443.
Geltungsbereich: IT, externe Dienstleister, Produktion, Wartung, KI-Systeme.
Key Points:
Zugriffsarten: Read-only, Operator, Service, Admin, KI-Read, KI-Write.
Externe: 24h Voranmeldung, Vertraulichkeit, VPN + 2FA, max. 8h, Log nach 1h.
Passwort: 12 Zeichen, 90 Tage, 3 Fehlversuche = 15 Min Sperre.
Jeder Zugriff im zentralen Log (6 Jahre Aufbewahrung).
Notfall (Break-Glass) nur mit Token aus Tresor, Nachdokumentation in 24h.
RIGHT (RELATED CONTEXT):
Deviations:
DEV-IT-001 - Externer Siemens-Techniker ohne 24h Voranmeldung auf SPS zugegriffen.
DEV-IT-002 - Standardpasswort admin/admin auf Abfuellmaschine noch aktiv nach 6 Monaten.
DEV-IT-003 - Externer Dienstleister nutzte nur Passwort, 2FA umgangen.
CAPAs:
CAPA-IT-001 - Einfuehrung Ticketing-System mit 24h Vorlaufpflicht fuer Externe.
CAPA-IT-002 - Checkliste fuer Maschinenabnahme: Standardpasswort aendern.
Audit Findings:
AUD-IT-001 - EMA: Externe Zugriffe nicht ausreichend kontrolliert.
AUD-IT-002 - Intern: Standardpasswort auf Abfuellmaschine.
Decisions:
DEC-IT-001 - Stilllegung Abfuellmaschine fuer 2 Tage, Passwort-Reset und Schulung.
DEC-IT-002 - Dienstleister NetTech GmbH gesperrt bis 2FA implementiert."""

print("=" * 60)
print("STEP 1: NLP Processing")
print("=" * 60)
result = nlp_pipeline.process_document(TEXT)
genre = result["classification"]["predicted_genre"]
confidence = result["classification"]["confidence"]
lang = result["language"]["lang_code"]
print(f"Genre    : {genre}")
print(f"Confidence: {confidence:.1%}")
print(f"Language  : {lang}")

print()
print("STEP 2: Chunking")
print("=" * 60)
chunks = result["chunks"]
print(f"Total chunks: {len(chunks)}")
for c in chunks:
    print(f"  [{c.section_title}] is_generic={c.is_generic}  words={len(c.content.split())}")

print()
print("STEP 3: Rewrite (local fallback - 'rewrite' mode)")
print("=" * 60)

# Simulate build_profile output
from uuid import uuid4
sop_style = result["features"].get("sop_style", {})
legal_style = result["features"].get("legal_style", {})
profile = {
    "doc_id": str(uuid4()),
    "genre": genre,
    "confidence": confidence,
    "runner_up": result["classification"]["runner_up"],
    "all_scores": result["classification"]["all_scores"],
    "tone": "neutral",
    "voice": result["features"]["voice"],
    "language": lang,
    "llm_used": False,
    "key_style_rules": [f"Preserve the detected {genre} style and structure."],
    "do_not_change": [],
    "features": result["features"],
    "chunks": [
        {
            "section_title": c.section_title,
            "section_type": c.section_type,
            "word_count": len(c.content.split()),
            "content": c.content,
        }
        for c in chunks
    ],
}

is_compliance_doc = sop_style.get("format_score", 0) >= 0.45 or legal_style.get("legalese_score", 0) >= 0.35
print(f"is_compliance_doc = {is_compliance_doc}")
print(f"SOP format_score  = {sop_style.get('format_score', 0)}")
print(f"Legal legalese    = {legal_style.get('legalese_score', 0)}")

# Run rewrite via the same path as streamlit_app
rewritten_parts = []
for chunk in chunks:
    is_generic = getattr(chunk, "is_generic", False)
    # Local fallback for compliance docs
    content = chunk.content
    if is_generic:
        rewritten_parts.append(content)
    else:
        rewritten_parts.append(f"{chunk.section_title}\n\n{content}")

output = "\n\n".join(rewritten_parts)
print()
print("OUTPUT PREVIEW:")
print("-" * 60)
print(output[:1200])
print()
print(f"Total output: {len(output)} chars | {len(output.split())} words")
print()
print("PASS: Pipeline completed successfully.")
