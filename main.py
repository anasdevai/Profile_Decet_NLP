import os
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import database
import nlp_pipeline
import vector_store
import llm_service
from uuid import uuid4
from sqlalchemy.orm import Session

app = FastAPI(title="Style Detection & Rewriting Engine")

class AnalyzeRequest(BaseModel):
    text: str
    doc_id: Optional[str] = None
    org_id: Optional[str] = "default_org"

class RewriteRequest(BaseModel):
    text: str
    doc_id: Optional[str] = None
    org_id: Optional[str] = "default_org"
    mode: str = "improve"
    debug: bool = False

class FeedbackRequest(BaseModel):
    doc_id: str
    correct_genre: Optional[str] = None
    correct_tone: Optional[str] = None
    style_rules_add: Optional[List[str]] = None
    do_not_change_add: Optional[List[str]] = None
    rating: int = 5

GENRE_THRESHOLDS = {
    "sop":      0.80,
    "legal":    0.82,
    "academic": 0.75,
    "poetry":   0.65,
    "rap":      0.62,
    "casual":   0.60,
    "ur":       0.45,
    "hi":       0.50,
    "default":  0.70,
}

def get_threshold(db, genre: str, lang_code: str) -> float:
    if lang_code in ["ur", "hi", "ar"]:
        return database.get_threshold(db, lang_code, GENRE_THRESHOLDS.get(lang_code, 0.45))
    default = GENRE_THRESHOLDS.get(genre, GENRE_THRESHOLDS["default"])
    return database.get_threshold(db, genre, default)

@app.post("/analyze")
def analyze_document(req: AnalyzeRequest, db: Session = Depends(database.get_db)):
    doc_id = req.doc_id or str(uuid4())
    
    # Run Stages 1-7
    result = nlp_pipeline.process_document(req.text)
    
    genre = result["classification"]["predicted_genre"]
    confidence = result["classification"]["confidence"]
    lang_code = result["language"]["lang_code"]
    
    threshold = get_threshold(db, genre, lang_code)
    llm_used = False
    
    # Stage 8: Conditional LLM validation
    if confidence < threshold or lang_code in ["ur", "hi", "ar"]:
        llm_used = True
        llm_result = llm_service.confirm_with_llm(req.text, result["features"], result["classification"])
        genre = llm_result.get("confirmed_genre", genre)
        key_style_rules = llm_result.get("key_style_rules", [])
        do_not_change = llm_result.get("do_not_change", [])
        tone = llm_result.get("tone", "neutral")
    else:
        key_style_rules = [f"Vocabulary heavily relies on {genre} terminology"]
        do_not_change = []
        if result["features"]["structure"]["has_numbered_steps"]:
            do_not_change.append("Numbered list structure")
        tone = "neutral"

    # Stage 9: Profile generation
    profile = {
        "doc_id": doc_id,
        "org_id": req.org_id,
        "genre": genre,
        "confidence": confidence,
        "tone": tone,
        "voice": result["features"]["voice"],
        "vocabulary_type": f"{genre}_jargon",
        "structure": "numbered_steps" if result["features"]["structure"]["has_numbered_steps"] else "prose",
        "wording_style": "formal" if result["features"]["formal_morph"] > 5 else "casual",
        "language": lang_code,
        "llm_used": llm_used,
        "threshold_used": threshold,
        "key_style_rules": key_style_rules,
        "do_not_change": do_not_change,
        "version": 1,
        "feedback_corrections": 0,
        "features": result["features"]
    }
    
    # Ingest document chunks to Qdrant (Stage 12 Runtime Learning)
    qdrant_ids = vector_store.ingest_document_chunks(
        result["chunks"], profile, lang_code, f"doc_{doc_id}", req.org_id
    )
    profile["qdrant_point_ids"] = qdrant_ids
    
    database.save_profile(db, profile)
    
    return profile

@app.post("/rewrite")
def rewrite_document(req: RewriteRequest, db: Session = Depends(database.get_db)):
    doc_id = req.doc_id or str(uuid4())
    
    # Stage 1-9
    profile = analyze_document(AnalyzeRequest(text=req.text, doc_id=doc_id, org_id=req.org_id), db)
    
    # Stage 10: Chunk-wise rewrite
    chunks = nlp_pipeline.smart_chunk(req.text)
    
    similar_docs = vector_store.retrieve_similar_styles(
        query_text=chunks[0].content if chunks else req.text,
        lang_code=profile["language"],
        genre_filter=profile["genre"],
        top_k=3
    )
    
    rewritten_sections = []
    fallback_count = 0
    for chunk in chunks:
        rewritten = llm_service.rewrite_chunk(
            chunk.content, chunk.section_title, profile, similar_docs, req.mode
        )
        if rewritten.startswith("[LLM not configured]") or rewritten.startswith("[Rewrite Failed]") or not rewritten.strip():
            fallback_count += 1
        rewritten_sections.append(f"## {chunk.section_title}\n\n{rewritten}")

    response = {
        "doc_id": doc_id,
        "rewritten_text": "\n\n".join(rewritten_sections),
        "style_profile": profile,
    }
    if req.debug:
        response["debug"] = {
            "mode": req.mode,
            "chunk_count": len(chunks),
            "fallback_count": fallback_count,
            "llm_client_initialized": llm_service.client is not None,
            "model_id": llm_service.MODEL_ID,
        }
    return response

@app.get("/llm-status")
def llm_status():
    return {
        "llm_client_initialized": llm_service.client is not None,
        "model_id": llm_service.MODEL_ID,
    }

@app.post("/feedback")
def apply_feedback(req: FeedbackRequest, db: Session = Depends(database.get_db)):
    profile = database.get_profile(db, req.doc_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
        
    profile["genre"] = req.correct_genre or profile.get("genre")
    profile["tone"] = req.correct_tone or profile.get("tone")
    if req.style_rules_add:
        profile.setdefault("key_style_rules", []).extend(req.style_rules_add)
    if req.do_not_change_add:
        profile.setdefault("do_not_change", []).extend(req.do_not_change_add)
        
    profile["version"] = profile.get("version", 1) + 1
    profile["feedback_corrections"] = profile.get("feedback_corrections", 0) + 1
    
    database.save_profile(db, profile)
    
    # Auto-calibration
    # This is a simplified version: raise threshold if multiple corrections happen
    # Ideally, we query count of corrections for this genre
    if profile["feedback_corrections"] % 5 == 0:
        current_thresh = get_threshold(db, profile["genre"], "en")
        new_thresh = min(current_thresh + 0.02, 0.95)
        database.save_threshold(db, profile["genre"], new_thresh)
        
    return {"status": "success", "message": "Feedback applied and profile updated", "profile": profile}

@app.get("/profile/{doc_id}")
def get_profile(doc_id: str, db: Session = Depends(database.get_db)):
    profile = database.get_profile(db, doc_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile
