=== NLP PROCESSING ===
Genre: sop (Confidence: 0.82)
Language: de
SOP Number: SOP-IT-001
Domain: it
Traceability IDs: {'SOP': 1, 'DEV': 10, 'CAPA': 10, 'AUD': 6, 'DEC': 4}

=== NLP ACCURACY ===
Genre Detection: PASS
Confidence >= 80%: PASS
Language Detection: PASS
SOP Number Detection: PASS
Domain Detection: PASS
Traceability IDs: PASS

=== REWRITE MODE TEST ===
Total chunks: 7
Rewrite Result (Excerpt):
[Rewrite Failed]: 503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.', 'status': 'UN...
Rewrite Test: WARNING (ID SOP-IT-001 missing in first chunk)

=== IMPROVE MODE TEST ===
Improve Result (Excerpt):
SOP-Identifikation und Titel: SOP-IT-001 – Zugriffsmanagement auf Produktionsnetzwerk (OT)...
Improve Test: PASS (ID preserved)

=== CREATE MODE TEST ===
Create Result (Excerpt):
Backup & Disaster Recovery: Für das OT-Netzwerk muss ein Backup- und Disaster Recovery Plan erstellt und periodisch validiert werden....
Create Test: PASS (Generated content aligned with profile)