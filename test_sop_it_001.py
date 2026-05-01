from pathlib import Path

import nlp_pipeline


FIXTURE = Path(__file__).resolve().parent.parent / "SOP-IT-001.txt"


def test_sop_it_001_detect_rewrite_and_check():
    text = FIXTURE.read_text(encoding="utf-8")

    analysis = nlp_pipeline.process_document(text)
    profile = analysis["sop_profile"]

    assert analysis["classification"]["predicted_genre"] == "sop"
    assert analysis["classification"]["confidence"] >= 0.80
    assert profile["language_profile"]["detected_sop_language"] == "de"
    assert profile["document_metadata"]["sop_number"] == "SOP-IT-001"
    assert profile["domain_context"]["predicted_domain"] in {"it", "security"}
    assert profile["traceability"]["id_counts"] == {
        "SOP": 1,
        "DEV": 10,
        "CAPA": 10,
        "AUD": 6,
        "DEC": 4,
    }

    rewritten = nlp_pipeline.rewrite_sop_same_language(text)
    rewritten_analysis = rewritten["rewritten_analysis"]

    assert rewritten["language_preserved"] is True
    assert rewritten_analysis["classification"]["predicted_genre"] == "sop"
    assert rewritten_analysis["classification"]["confidence"] >= 0.80
    assert rewritten_analysis["sop_profile"]["traceability"]["id_counts"] == profile["traceability"]["id_counts"]
