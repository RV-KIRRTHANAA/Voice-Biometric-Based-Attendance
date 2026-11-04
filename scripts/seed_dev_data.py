r"""
Canonical development seed script with Demo Mode.
Usage:  venv\Scripts\python scripts/seed_dev_data.py

- Idempotent: safe to run multiple times.
- Demo Mode: Generates 6 daily class sessions spanning the full calendar day.
- All sessions are available immediately after login without inter-class wait times.
- Creates/refreshes tenant, students, and class sessions.
- Does NOT touch biometric enrollment data.
- Preserves all historical attendance records.
"""
import sys, os, uuid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime, timedelta, timezone
from typing import Optional
from passlib.context import CryptContext
from src.infrastructure.database import SessionLocal, init_db, Tenant, User, AttendanceSessionModel

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ==============================================================================
# DEVELOPMENT CONFIGURATION ONLY
#
# CRITICAL SECURITY NOTICE:
# The default demo credentials below (e.g., student_001 / password123) are provided
# strictly for local proof-of-concept testing and development environments.
# These credentials must NEVER be used or reused in staging, production, or
# public-facing deployments.
# ==============================================================================
TENANT_ID = "tenant_1"
TENANT_NAME = "Development University (POC Demo)"
DEFAULT_PASSWORD = "password123"  # DEVELOPMENT ONLY - DO NOT USE IN PRODUCTION

# IST = UTC + 5:30
IST = timezone(timedelta(hours=5, minutes=30))

STUDENTS = [
    ("student_001", "Student 001"),
    ("student_002", "Student 002"),
    ("student_003", "Student 003"),
    ("student_004", "Student 004"),
    ("student_005", "Student 005"),
]

# Demo Mode: 6 daily classes available throughout the local calendar day
DEMO_CLASSES = [
    ("Mathematics", "math"),
    ("Physics", "physics"),
    ("Chemistry", "chem"),
    ("Computer Science", "cs"),
    ("English", "eng"),
    ("Voice Biometrics Lab", "voicelab"),
]


def build_class_sessions(now_dt: Optional[datetime] = None):
    """
    Demo Mode: Generates 6 daily class sessions spanning the full calendar day.
    All sessions are available immediately after login without inter-class wait times.
    Tokens include date suffix to guarantee fresh unique sessions per day while
    maintaining idempotency for repeated seeding within the same day.
    """
    now_utc = now_dt or datetime.utcnow()
    now_ist = now_utc.replace(tzinfo=timezone.utc).astimezone(IST)
    date_suffix = now_ist.strftime("%Y%m%d")

    day_start_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end_utc = day_start_utc.replace(hour=23, minute=59, second=59)

    return [
        {
            "token": f"class_{slug}_{date_suffix}",
            "class_name": name,
            "starts_at": day_start_utc + timedelta(minutes=i),
            "ends_at": day_end_utc,
        }
        for i, (name, slug) in enumerate(DEMO_CLASSES)
    ]


def seed(db_session=None):
    if db_session is None:
        init_db()
        db = SessionLocal()
        close_on_exit = True
    else:
        db = db_session
        close_on_exit = False

    try:
        # --- Tenant ---
        tenant = db.query(Tenant).filter(Tenant.id == TENANT_ID).first()
        if not tenant:
            db.add(Tenant(id=TENANT_ID, name=TENANT_NAME))
            db.commit()
            print(f"[+] Tenant created: {TENANT_ID}")
        else:
            print(f"[=] Tenant exists:  {TENANT_ID}")

        # --- Students ---
        for sid, display_name in STUDENTS:
            user = db.query(User).filter(User.id == sid).first()
            if not user:
                db.add(User(
                    id=sid,
                    tenant_id=TENANT_ID,
                    password_hash=pwd_context.hash(DEFAULT_PASSWORD),
                    display_name=display_name
                ))
                print(f"[+] User created:  {sid} ({display_name})")
            else:
                user.display_name = display_name
                print(f"[=] User exists:   {sid} ({display_name})")
        db.commit()

        # --- Class Sessions (Demo Mode: 6 sessions available full day) ---
        for cls in build_class_sessions():
            row = db.query(AttendanceSessionModel).filter(
                AttendanceSessionModel.session_token == cls["token"],
                AttendanceSessionModel.tenant_id == TENANT_ID
            ).first()
            if row:
                row.class_name = cls["class_name"]
                row.starts_at  = cls["starts_at"]
                row.ends_at    = cls["ends_at"]
                row.is_active  = True
                print(f"[~] Session refreshed: {cls['class_name']} ({cls['token']})")
            else:
                db.add(AttendanceSessionModel(
                    id=str(uuid.uuid4()),
                    tenant_id=TENANT_ID,
                    session_token=cls["token"],
                    class_name=cls["class_name"],
                    starts_at=cls["starts_at"],
                    ends_at=cls["ends_at"],
                    is_active=True
                ))
                print(f"[+] Session created:   {cls['class_name']} ({cls['token']})")
        db.commit()

        print()
        print("=== CREDENTIALS ===")
        print(f"Password (all students): {DEFAULT_PASSWORD}")
        for sid, name in STUDENTS:
            print(f"  {sid}  ({name})")
    finally:
        if close_on_exit:
            db.close()


if __name__ == "__main__":
    seed()
