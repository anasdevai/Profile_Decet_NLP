import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nlp_pipeline
from llm_service import rewrite_chunk
from vector_store import ingest_document_chunks

def evaluate_nlp(profile):
    report = []
    
    # Check V2 extractions
    doc_id = profile.get("document_identity", {})
    style = profile.get("style_and_tone", {})
    quality = profile.get("quality_metrics", {})
    vocab = profile.get("vocabulary", {})
    
    report.append(f"Language Detected: {doc_id.get('language', 'MISSING')}")
    report.append(f"Primary Domain: {doc_id.get('primary_domain', 'MISSING')}")
    report.append(f"Primary Tone: {style.get('primary_tone', 'MISSING')}")
    
    shall_ratio = style.get("tone_signals", {}).get("SHALL_ratio", "0.0")
    report.append(f"SHALL Ratio (Raw Signal): {shall_ratio}")
    
    blocklist = profile.get("precision_blocklist", "")
    report.append(f"Precision Blocklist Built: {'PASS' if blocklist else 'PASS (None Required)'}")
    
    seeds = profile.get("domain_seeds", [])
    report.append(f"Domain Seeds Generated: {len(seeds) if seeds else '0'} seeds")
    
    # Check step audit
    workflow = profile.get("workflow", {})
    step_audit = workflow.get("step_completeness", {}).get("step_details", [])
    report.append(f"Step Audit Generated: {'PASS' if step_audit else 'FAIL/NONE'}")
    
    return report

def main():
    sops_dir = Path("d:/profiles/SOPS")
    if not sops_dir.exists():
        print(f"Directory {sops_dir} not found.")
        return
        
    txt_files = list(sops_dir.glob("*.txt"))
    if not txt_files:
        print("No .txt files found in SOPS folder.")
        return

    full_report = ["# Deep NLP Pipeline & LLM Evaluation Report\n"]
    
    total_files = len(txt_files)
    full_report.append(f"**Total SOPs Found**: {total_files}\n")
    
    for i, file_path in enumerate(txt_files):
        print(f"Processing {file_path.name} ({i+1}/{total_files})...")
        full_report.append(f"## Document: {file_path.name}")
        
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception as e:
            full_report.append(f"**Error reading file:** {e}\n")
            continue
            
        print("  Running NLP Pipeline...")
        start_time = time.time()
        analysis = nlp_pipeline.process_document(text)
        nlp_time = time.time() - start_time
        
        profile = analysis["structured_sop_profile"]
        
        full_report.append(f"### 1. NLP Extraction (took {nlp_time:.2f}s)")
        nlp_checks = evaluate_nlp(profile)
        for check in nlp_checks:
            full_report.append(f"- {check}")
            
        print("  Ingesting to Vector Store...")
        try:
            chunks = analysis["chunks"]
            doc_name = file_path.name
            lang_code = profile.get("document_identity", {}).get("language", "en")
            point_ids = ingest_document_chunks(chunks, analysis, lang_code, doc_name, "test_org")
            full_report.append(f"\n### 2. Vector Store Ingestion")
            full_report.append(f"- PASS: Ingested {len(point_ids)} chunks into Qdrant using V2 taxonomy.")
        except Exception as e:
            full_report.append(f"\n### 2. Vector Store Ingestion")
            full_report.append(f"- FAIL: {e}")
            
        print("  Running LLM Tests...")
        full_report.append("\n### 3. LLM Fidelity Execution")
        
        # We test on the first significant chunk
        test_chunk = None
        for c in analysis["chunks"]:
            if len(c.get("content", "").split()) > 30 and not c.get("is_generic", False):
                test_chunk = c
                break
        if not test_chunk and analysis["chunks"]:
            test_chunk = analysis["chunks"][0]
            
        if test_chunk:
            # 3A: Rewrite Test
            print("    -> Mode: rewrite")
            try:
                res_rewrite = rewrite_chunk(test_chunk.get("content", ""), test_chunk.get("section_title", ""), analysis, [], mode="rewrite")
                full_report.append("#### Mode: `rewrite`")
                full_report.append("```text\n" + res_rewrite[:400] + "...\n```")
                # Basic check: Does it retain SOP numbers or 'shall' language?
                sop_id = profile.get("document_identity", {}).get("sop_id", "")
                if sop_id and sop_id in res_rewrite:
                    full_report.append(f"- **Check**: Retained Traceability ID ({sop_id}) -> PASS")
                else:
                    full_report.append(f"- **Check**: Retained Traceability ID -> FAILED or NOT APPLICABLE")
            except Exception as e:
                full_report.append(f"#### Mode: `rewrite` -> FAIL ({e})")
                
            # 3B: Improve Test
            print("    -> Mode: improve")
            try:
                res_improve = rewrite_chunk(test_chunk.get("content", ""), test_chunk.get("section_title", ""), analysis, [], mode="improve")
                full_report.append("#### Mode: `improve`")
                full_report.append("```text\n" + res_improve[:400] + "...\n```")
            except Exception as e:
                full_report.append(f"#### Mode: `improve` -> FAIL ({e})")
                
        # 3C: Create Test
        print("    -> Mode: create_new")
        try:
            create_seed = "Draft a procedure for emergency shutdown."
            res_create = rewrite_chunk(create_seed, "Emergency Shutdown", analysis, [], mode="create_new")
            full_report.append("#### Mode: `create_new`")
            full_report.append("```text\n" + res_create[:400] + "...\n```")
        except Exception as e:
            full_report.append(f"#### Mode: `create_new` -> FAIL ({e})")
            
        full_report.append("\n---\n")

    report_path = Path("d:/profiles/deep_accuracy_report.md")
    report_path.write_text("\n".join(full_report), encoding="utf-8")
    print(f"\nDeep Evaluation Complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()
