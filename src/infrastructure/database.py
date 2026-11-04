from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, LargeBinary, ForeignKey, UniqueConstraint, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = "sqlite:///./data/oneapp.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Tenant(Base):
    __tablename__ = "tenants"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, index=True)
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False)
    password_hash = Column(String, nullable=False)
    display_name = Column(String, nullable=True)  # Human-readable name for UI

    tenant = relationship("Tenant")

class Enrollment(Base):
    __tablename__ = "enrollments"
    id = Column(String, primary_key=True, index=True)
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    status = Column(String, nullable=False)
    required_samples = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

class EnrollmentSample(Base):
    __tablename__ = "enrollment_samples"
    id = Column(String, primary_key=True, index=True)
    enrollment_id = Column(String, ForeignKey("enrollments.id"), nullable=False)
    embedding_vector = Column(LargeBinary, nullable=False)
    model_id = Column(String, nullable=False)
    dimension = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class ReferenceEmbeddingModel(Base):
    __tablename__ = "reference_embeddings"
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False)
    enrollment_id = Column(String, ForeignKey("enrollments.id"), nullable=True)
    vector_blob = Column(LargeBinary, nullable=False)
    model_id = Column(String, nullable=False)
    dimension = Column(Integer, nullable=False)
    is_active = Column(Boolean, default=True)

class AttendanceSessionModel(Base):
    __tablename__ = "attendance_sessions"
    id = Column(String, primary_key=True, index=True)
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False)
    # session_token: server-side credential submitted when marking attendance.
    # It is separate from 'id' so that the credential can be rotated independently
    # of the session identity. For this POC it is a stable readable string.
    session_token = Column(String, unique=True, index=True, nullable=False)
    # class_name: display name shown to the student. NOT derived from session_token.
    class_name = Column(String, nullable=False, server_default="Unnamed Class")
    starts_at = Column(DateTime, nullable=False)
    ends_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True)

class AttendanceRecord(Base):
    __tablename__ = "attendance_records"
    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    session_id = Column(String, ForeignKey("attendance_sessions.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("tenant_id", "user_id", "session_id", name="uix_tenant_user_session"),)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False)
    user_id = Column(String, nullable=False)
    session_token = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    request_id = Column(String, nullable=True)
    result = Column(String, nullable=False)
    reason = Column(String, nullable=False)
    similarity_score = Column(Float, nullable=True)
    liveness_score = Column(Float, nullable=True)
    model_id = Column(String, nullable=True)

class Challenge(Base):
    __tablename__ = "challenges"
    id = Column(String, primary_key=True, index=True)
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    session_id = Column(String, ForeignKey("attendance_sessions.id"), nullable=False)
    expected_phrase = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    consumed = Column(Boolean, default=False)

def init_db():
    import os
    os.makedirs("./data", exist_ok=True)
    Base.metadata.create_all(bind=engine)
    _migrate_existing_db()

def _migrate_existing_db():
    """Apply additive migrations for columns added after initial schema creation."""
    from sqlalchemy import text
    with engine.connect() as conn:
        # Add display_name to users if missing
        try:
            conn.execute(text("ALTER TABLE users ADD COLUMN display_name VARCHAR"))
            conn.commit()
        except Exception:
            pass  # Column already exists
        # Add class_name to attendance_sessions if missing
        try:
            conn.execute(text("ALTER TABLE attendance_sessions ADD COLUMN class_name VARCHAR NOT NULL DEFAULT 'Unnamed Class'"))
            conn.commit()
        except Exception:
            pass  # Column already exists

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
