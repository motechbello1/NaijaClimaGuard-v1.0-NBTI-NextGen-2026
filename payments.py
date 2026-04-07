"""
NaijaClimaGuard — Paystack Payments (payments.py)
==================================================
Handles:
  - Initialising a Paystack transaction (returns checkout URL)
  - Verifying a transaction after redirect
  - Charging a saved card (recurring billing)
  - Storing authorization_code (card token) — never raw card data

Test keys: replace with live keys from paystack.com/settings/developer
  PAYSTACK_SECRET_KEY = "sk_live_..."
  PAYSTACK_PUBLIC_KEY = "pk_live_..."

Paystack docs: https://paystack.com/docs/api/
"""

import hashlib
import hmac
import json
import secrets
from datetime import datetime

import requests

# ── Keys — loaded from Streamlit secrets in production ───────────────────────
# In Streamlit Cloud: Settings → Secrets → paste the block below
# [paystack]
# secret_key = "sk_live_your_real_key_here"
# public_key = "pk_live_your_real_key_here"
#
# Locally: create .streamlit/secrets.toml with the same block
# Falls back to test keys if secrets are not configured yet.

try:
    import streamlit as st
    PAYSTACK_SECRET_KEY = st.secrets["paystack"]["secret_key"]
    PAYSTACK_PUBLIC_KEY = st.secrets["paystack"]["public_key"]
except Exception:
    # Test keys — payments will not charge real cards
    PAYSTACK_SECRET_KEY = "sk_test_4fb1b2ef61fce8dfed5246e473f9af5c7c431d66"
    PAYSTACK_PUBLIC_KEY = "pk_test_2dc062c064bdd1c8b0a2260d64818b2c131beda4"

BASE_URL   = "https://api.paystack.co"
HEADERS    = {
    "Authorization": f"Bearer {PAYSTACK_SECRET_KEY}",
    "Content-Type":  "application/json",
}

# Your WhatsApp number for the alert system
OWNER_WHATSAPP = "+2348064425781"


def initialise_transaction(email: str, amount_kobo: int, user_id: int,
                            tier: str, callback_url: str) -> dict:
    """
    Start a Paystack payment.
    Returns {"status": True, "authorization_url": "...", "reference": "..."}
    or      {"status": False, "error": "..."}
    """
    reference = f"NCG-{user_id}-{tier}-{secrets.token_hex(6).upper()}"
    payload = {
        "email":        email,
        "amount":       amount_kobo,
        "reference":    reference,
        "callback_url": callback_url,
        "metadata": {
            "user_id":    user_id,
            "tier":       tier,
            "product":    "NaijaClimaGuard",
            "custom_fields": [
                {"display_name": "User ID",  "variable_name": "user_id",  "value": str(user_id)},
                {"display_name": "Plan",     "variable_name": "tier",     "value": tier},
            ],
        },
        "channels": ["card", "bank", "ussd", "qr", "mobile_money", "bank_transfer"],
    }
    try:
        r = requests.post(f"{BASE_URL}/transaction/initialize",
                          headers=HEADERS, json=payload, timeout=15)
        data = r.json()
        if data.get("status"):
            return {
                "status":            True,
                "authorization_url": data["data"]["authorization_url"],
                "reference":         data["data"]["reference"],
                "access_code":       data["data"]["access_code"],
            }
        return {"status": False, "error": data.get("message", "Paystack error")}
    except Exception as e:
        return {"status": False, "error": str(e)}


def verify_transaction(reference: str) -> dict:
    """
    Verify a completed transaction by reference.
    Returns full transaction data including authorization (card token).
    """
    try:
        r = requests.get(f"{BASE_URL}/transaction/verify/{reference}",
                         headers=HEADERS, timeout=15)
        data = r.json()
        if not data.get("status"):
            return {"status": False, "error": data.get("message", "Verification failed")}

        tx = data["data"]
        if tx["status"] != "success":
            return {"status": False, "error": f"Transaction status: {tx['status']}"}

        auth = tx.get("authorization", {})
        return {
            "status":             True,
            "amount":             tx["amount"],
            "reference":          tx["reference"],
            "paid_at":            tx.get("paid_at"),
            "channel":            tx.get("channel"),
            "currency":           tx.get("currency"),
            "authorization_code": auth.get("authorization_code"),
            "card_type":          auth.get("card_type"),
            "last4":              auth.get("last4"),
            "exp_month":          auth.get("exp_month"),
            "exp_year":           auth.get("exp_year"),
            "bank":               auth.get("bank"),
            "reusable":           auth.get("reusable", False),
            "customer_email":     tx.get("customer", {}).get("email"),
        }
    except Exception as e:
        return {"status": False, "error": str(e)}


def charge_saved_card(email: str, amount_kobo: int,
                       authorization_code: str, tier: str, user_id: int) -> dict:
    """
    Charge a previously saved card for recurring subscription renewal.
    This is the auto-renew mechanism — no user interaction needed.
    """
    reference = f"NCG-RENEW-{user_id}-{secrets.token_hex(6).upper()}"
    payload = {
        "email":              email,
        "amount":             amount_kobo,
        "authorization_code": authorization_code,
        "reference":          reference,
        "metadata": {
            "user_id": user_id,
            "tier":    tier,
            "type":    "auto_renewal",
        },
    }
    try:
        r = requests.post(f"{BASE_URL}/transaction/charge_authorization",
                          headers=HEADERS, json=payload, timeout=15)
        data = r.json()
        if data.get("status") and data["data"]["status"] == "success":
            return {"status": True, "reference": reference, "amount": amount_kobo}
        return {"status": False,
                "error": data.get("data", {}).get("gateway_response", "Charge failed")}
    except Exception as e:
        return {"status": False, "error": str(e)}


def verify_webhook(payload_bytes: bytes, signature: str) -> bool:
    """
    Verify that a webhook POST is genuinely from Paystack.
    Use in your FastAPI/Flask webhook endpoint when you add one.
    """
    computed = hmac.new(
        PAYSTACK_SECRET_KEY.encode(), payload_bytes, hashlib.sha512
    ).hexdigest()
    return hmac.compare_digest(computed, signature)


def format_naira(kobo: int) -> str:
    return f"₦{kobo / 100:,.0f}"
