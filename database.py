"""
NaijaClimaGuard — Database Layer (database.py)
===============================================
SQLite backend. Swap DB_PATH for Postgres URI when deploying.

Tables:
  users          — accounts, hashed passwords, account type
  subscriptions  — active plan per user, dates, Paystack ref
  paystack_cards — tokenised card authorizations (no raw card data)
  borehole_usage — per-user report count for quota enforcement
  audit_log      — every login, payment, upgrade event
"""

import hashlib
import os
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path("naijaclimagard.db")

# ── Tier definitions ─────────────────────────────────────────────────────────
TIERS = {
    "free": {
        "label":           "Free",
        "price_monthly":   0,
        "price_kobo":      0,
        "locations":       1,
        "replay_days":     0,
        "borehole_quota":  0,
        "whatsapp_alerts": False,
        "api_access":      False,
        "color":           "#5a6a7e",
        "badge":           "FREE",
    },
    "individual": {
        "label":           "Individual",
        "price_monthly":   1000,
        "price_kobo":      100000,
        "locations":       5,
        "replay_days":     7,
        "borehole_quota":  3,
        "whatsapp_alerts": True,
        "api_access":      False,
        "color":           "#4da6ff",
        "badge":           "INDIVIDUAL",
    },
    "farmer": {
        "label":           "Farmer Pro",
        "price_monthly":   2500,
        "price_kobo":      250000,
        "locations":       5,
        "replay_days":     30,
        "borehole_quota":  10,
        "whatsapp_alerts": True,
        "api_access":      False,
        "color":           "#00c864",
        "badge":           "FARMER PRO",
    },
    "business": {
        "label":           "Business",
        "price_monthly":   15000,
        "price_kobo":      1500000,
        "locations":       5,
        "replay_days":     365,
        "borehole_quota":  -1,   # unlimited
        "whatsapp_alerts": True,
        "api_access":      True,
        "color":           "#ffd700",
        "badge":           "BUSINESS",
    },
    "government": {
        "label":           "Government",
        "price_monthly":   150000,
        "price_kobo":      15000000,
        "locations":       5,
        "replay_days":     365,
        "borehole_quota":  -1,
        "whatsapp_alerts": True,
        "api_access":      True,
        "color":           "#ff8c00",
        "badge":           "GOVERNMENT",
    },
}

ACCOUNT_TYPES = ["individual", "farmer", "ngo", "business", "government", "researcher"]


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_conn()
    c = conn.cursor()

    c.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        email         TEXT    UNIQUE NOT NULL,
        password_hash TEXT    NOT NULL,
        full_name     TEXT    NOT NULL,
        phone         TEXT,
        account_type  TEXT    NOT NULL DEFAULT 'individual',
        state         TEXT,
        organisation  TEXT,
        tier          TEXT    NOT NULL DEFAULT 'free',
        is_verified   INTEGER NOT NULL DEFAULT 0,
        is_vetted     INTEGER NOT NULL DEFAULT 0,
        created_at    TEXT    NOT NULL,
        last_login    TEXT
    );

    CREATE TABLE IF NOT EXISTS subscriptions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id         INTEGER NOT NULL REFERENCES users(id),
        tier            TEXT    NOT NULL,
        status          TEXT    NOT NULL DEFAULT 'active',
        paystack_ref    TEXT,
        started_at      TEXT    NOT NULL,
        expires_at      TEXT,
        auto_renew      INTEGER NOT NULL DEFAULT 1,
        amount_paid     INTEGER,
        created_at      TEXT    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS paystack_cards (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id             INTEGER NOT NULL REFERENCES users(id),
        authorization_code  TEXT    NOT NULL,
        card_type           TEXT,
        last4               TEXT,
        exp_month           TEXT,
        exp_year            TEXT,
        bank                TEXT,
        is_default          INTEGER NOT NULL DEFAULT 1,
        created_at          TEXT    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS borehole_usage (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id     INTEGER NOT NULL REFERENCES users(id),
        latitude    REAL,
        longitude   REAL,
        score       INTEGER,
        verdict     TEXT,
        used_at     TEXT    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS audit_log (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    INTEGER REFERENCES users(id),
        event      TEXT    NOT NULL,
        detail     TEXT,
        ip         TEXT,
        created_at TEXT    NOT NULL
    );
    """)

    conn.commit()
    conn.close()


# ── Password helpers ──────────────────────────────────────────────────────────
def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260000)
    return f"{salt}${h.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, h = stored_hash.split("$", 1)
        check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260000)
        return check.hex() == h
    except Exception:
        return False


# ── User CRUD ─────────────────────────────────────────────────────────────────
def create_user(email, password, full_name, phone, account_type, state, organisation=""):
    conn = get_conn()
    try:
        conn.execute(
            """INSERT INTO users
               (email, password_hash, full_name, phone, account_type, state,
                organisation, tier, created_at)
               VALUES (?,?,?,?,?,?,?,'free',?)""",
            (email.lower().strip(), _hash_password(password), full_name,
             phone, account_type, state, organisation,
             datetime.utcnow().isoformat()),
        )
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."
    finally:
        conn.close()


def authenticate_user(email, password):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE email = ?", (email.lower().strip(),)
    ).fetchone()
    conn.close()
    if not row:
        return None, "Email not found."
    if not _verify_password(password, row["password_hash"]):
        return None, "Incorrect password."
    # Update last_login
    conn = get_conn()
    conn.execute("UPDATE users SET last_login=? WHERE id=?",
                 (datetime.utcnow().isoformat(), row["id"]))
    conn.commit()
    conn.close()
    return dict(row), None


def get_user(user_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def upgrade_user_tier(user_id, tier):
    conn = get_conn()
    conn.execute("UPDATE users SET tier=? WHERE id=?", (tier, user_id))
    conn.commit()
    conn.close()


# ── Subscription CRUD ─────────────────────────────────────────────────────────
def create_subscription(user_id, tier, paystack_ref, amount_kobo):
    expires = (datetime.utcnow() + timedelta(days=30)).isoformat()
    conn = get_conn()
    # Deactivate previous
    conn.execute(
        "UPDATE subscriptions SET status='cancelled' WHERE user_id=? AND status='active'",
        (user_id,)
    )
    conn.execute(
        """INSERT INTO subscriptions
           (user_id, tier, status, paystack_ref, started_at, expires_at,
            amount_paid, created_at)
           VALUES (?,?,'active',?,?,?,?,?)""",
        (user_id, tier, paystack_ref,
         datetime.utcnow().isoformat(), expires,
         amount_kobo, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
    upgrade_user_tier(user_id, tier)


def get_active_subscription(user_id):
    conn = get_conn()
    row = conn.execute(
        """SELECT * FROM subscriptions
           WHERE user_id=? AND status='active'
           ORDER BY created_at DESC LIMIT 1""",
        (user_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ── Card storage ──────────────────────────────────────────────────────────────
def save_card(user_id, auth_code, card_type, last4, exp_month, exp_year, bank):
    conn = get_conn()
    conn.execute("UPDATE paystack_cards SET is_default=0 WHERE user_id=?", (user_id,))
    conn.execute(
        """INSERT INTO paystack_cards
           (user_id, authorization_code, card_type, last4, exp_month, exp_year,
            bank, is_default, created_at)
           VALUES (?,?,?,?,?,?,?,1,?)""",
        (user_id, auth_code, card_type, last4, exp_month, exp_year, bank,
         datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_default_card(user_id):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM paystack_cards WHERE user_id=? AND is_default=1",
        (user_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_cards(user_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM paystack_cards WHERE user_id=? ORDER BY created_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Borehole quota ────────────────────────────────────────────────────────────
def count_borehole_this_month(user_id):
    conn = get_conn()
    start = datetime.utcnow().replace(day=1).isoformat()
    n = conn.execute(
        "SELECT COUNT(*) FROM borehole_usage WHERE user_id=? AND used_at >= ?",
        (user_id, start)
    ).fetchone()[0]
    conn.close()
    return n


def log_borehole(user_id, lat, lon, score, verdict):
    conn = get_conn()
    conn.execute(
        "INSERT INTO borehole_usage (user_id, latitude, longitude, score, verdict, used_at) VALUES (?,?,?,?,?,?)",
        (user_id, lat, lon, score, verdict, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


# ── Audit ─────────────────────────────────────────────────────────────────────
def audit(user_id, event, detail=""):
    conn = get_conn()
    conn.execute(
        "INSERT INTO audit_log (user_id, event, detail, created_at) VALUES (?,?,?,?)",
        (user_id, event, detail, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


# Initialise on import
init_db()


# ── Demo user seeding ─────────────────────────────────────────────────────────
DEMO_USERS = [
    {
        "email":        "demo.free@naijaclimagard.com",
        "password":     "Demo@Free1",
        "full_name":    "Amina Yusuf",
        "phone":        "+2348011111111",
        "account_type": "individual",
        "state":        "FCT",
        "organisation": "",
        "tier":         "free",
        "card_last4":   None,
    },
    {
        "email":        "demo.individual@naijaclimagard.com",
        "password":     "Demo@Ind1",
        "full_name":    "Chukwuemeka Obi",
        "phone":        "+2348022222222",
        "account_type": "individual",
        "state":        "Anambra",
        "organisation": "",
        "tier":         "individual",
        "card_last4":   "4081",
    },
    {
        "email":        "demo.farmer@naijaclimagard.com",
        "password":     "Demo@Farm1",
        "full_name":    "Ibrahim Musa",
        "phone":        "+2348033333333",
        "account_type": "farmer",
        "state":        "Kogi",
        "organisation": "Musa Cooperative Farms",
        "tier":         "farmer",
        "card_last4":   "5512",
    },
    {
        "email":        "demo.business@naijaclimagard.com",
        "password":     "Demo@Biz1",
        "full_name":    "Ngozi Adeyemi",
        "phone":        "+2348044444444",
        "account_type": "business",
        "state":        "Lagos",
        "organisation": "PrimeSure Insurance Ltd",
        "tier":         "business",
        "card_last4":   "7823",
    },
    {
        "email":        "demo.government@naijaclimagard.com",
        "password":     "Demo@Gov1",
        "full_name":    "Abdullahi Sule",
        "phone":        "+2348055555555",
        "account_type": "government",
        "state":        "Benue",
        "organisation": "Benue State Emergency Management Agency",
        "tier":         "government",
        "card_last4":   "9234",
    },
]


def seed_demo_users():
    """Insert demo accounts if they don't already exist. Safe to call on every startup."""
    conn = get_conn()
    now  = datetime.utcnow().isoformat()
    expires = (datetime.utcnow() + timedelta(days=365)).isoformat()

    for u in DEMO_USERS:
        exists = conn.execute(
            "SELECT id FROM users WHERE email=?", (u["email"],)
        ).fetchone()
        if exists:
            continue

        # Create user
        conn.execute(
            """INSERT INTO users
               (email, password_hash, full_name, phone, account_type,
                state, organisation, tier, is_verified, is_vetted, created_at)
               VALUES (?,?,?,?,?,?,?,?,1,1,?)""",
            (u["email"], _hash_password(u["password"]), u["full_name"],
             u["phone"], u["account_type"], u["state"],
             u["organisation"], u["tier"], now),
        )
        user_id = conn.execute(
            "SELECT id FROM users WHERE email=?", (u["email"],)
        ).fetchone()["id"]

        # Create subscription (skip free tier)
        if u["tier"] != "free":
            tier_data = TIERS[u["tier"]]
            conn.execute(
                """INSERT INTO subscriptions
                   (user_id, tier, status, paystack_ref, started_at,
                    expires_at, auto_renew, amount_paid, created_at)
                   VALUES (?,?,'active',?,?,?,1,?,?)""",
                (user_id, u["tier"],
                 f"DEMO-{u['tier'].upper()}-{user_id:04d}",
                 now, expires,
                 tier_data["price_kobo"], now),
            )

        # Save a fake card token (non-reusable demo token)
        if u["card_last4"]:
            card_types = {"4081":"visa","5512":"mastercard",
                          "7823":"verve","9234":"mastercard"}
            banks      = {"4081":"Access Bank","5512":"GTBank",
                          "7823":"Zenith Bank","9234":"First Bank"}
            conn.execute(
                """INSERT INTO paystack_cards
                   (user_id, authorization_code, card_type, last4,
                    exp_month, exp_year, bank, is_default, created_at)
                   VALUES (?,?,?,?,?,?,?,1,?)""",
                (user_id,
                 f"AUTH_DEMO_{u['card_last4']}_{user_id}",
                 card_types.get(u["card_last4"], "visa"),
                 u["card_last4"], "12", "2027",
                 banks.get(u["card_last4"], "Demo Bank"), now),
            )

    conn.commit()
    conn.close()


# Seed demo users on every startup (idempotent)
seed_demo_users()
