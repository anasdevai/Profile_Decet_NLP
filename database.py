import os
import json
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL", "sqlite:///./style.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class StyleProfileModel(Base):
    __tablename__ = "style_profiles"

    doc_id = Column(String, primary_key=True, index=True)
    org_id = Column(String, index=True)
    genre = Column(String)
    tone = Column(String)
    voice = Column(String)
    vocabulary_type = Column(String)
    structure = Column(String)
    wording_style = Column(String)
    language = Column(String)
    version = Column(Integer, default=1)
    feedback_corrections = Column(Integer, default=0)
    
    # Store full JSON for all other nested properties
    full_profile_json = Column(Text)

class ThresholdModel(Base):
    __tablename__ = "genre_thresholds"

    genre = Column(String, primary_key=True)
    threshold = Column(Float)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_profile(db_session, profile: dict):
    existing = db_session.query(StyleProfileModel).filter_by(doc_id=profile["doc_id"]).first()
    if existing:
        existing.genre = profile.get("genre")
        existing.tone = profile.get("tone")
        existing.version = profile.get("version", 1)
        existing.feedback_corrections = profile.get("feedback_corrections", 0)
        existing.full_profile_json = json.dumps(profile)
    else:
        new_profile = StyleProfileModel(
            doc_id=profile["doc_id"],
            org_id=profile.get("org_id", "default_org"),
            genre=profile.get("genre"),
            tone=profile.get("tone"),
            voice=profile.get("voice"),
            vocabulary_type=profile.get("vocabulary_type"),
            structure=profile.get("structure"),
            wording_style=profile.get("wording_style"),
            language=profile.get("language"),
            version=profile.get("version", 1),
            feedback_corrections=profile.get("feedback_corrections", 0),
            full_profile_json=json.dumps(profile)
        )
        db_session.add(new_profile)
    db_session.commit()

def get_profile(db_session, doc_id: str) -> dict:
    model = db_session.query(StyleProfileModel).filter_by(doc_id=doc_id).first()
    if model and model.full_profile_json:
        return json.loads(model.full_profile_json)
    return None

def save_threshold(db_session, genre: str, threshold: float):
    existing = db_session.query(ThresholdModel).filter_by(genre=genre).first()
    if existing:
        existing.threshold = threshold
    else:
        new_thresh = ThresholdModel(genre=genre, threshold=threshold)
        db_session.add(new_thresh)
    db_session.commit()

def get_threshold(db_session, genre: str, default_thresh: float) -> float:
    existing = db_session.query(ThresholdModel).filter_by(genre=genre).first()
    return existing.threshold if existing else default_thresh
