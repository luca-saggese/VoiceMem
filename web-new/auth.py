"""Authentication for the web-new demo.

The module deliberately uses only Python's standard library. Configure SMTP and
Google OAuth with environment variables before enabling it outside localhost.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import smtplib
import sqlite3
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse


class AuthService:
    def __init__(self) -> None:
        default_db = Path(__file__).resolve().parent / "data" / "auth.sqlite3"
        self.db_path = Path(os.environ.get("VOICEMEM_AUTH_DB", str(default_db))).expanduser()
        self.frontend_url = os.environ.get("VOICEMEM_PUBLIC_URL", "http://127.0.0.1:5174").rstrip("/")
        self.cookie_name = "voicemem_session"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT,
                    google_sub TEXT UNIQUE,
                    display_name TEXT NOT NULL DEFAULT '',
                    email_verified INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tokens (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );
            """)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _stamp(value: datetime) -> str:
        return value.isoformat()

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    @staticmethod
    def _password_hash(password: str) -> str:
        iterations = 310_000
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
        return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"

    @staticmethod
    def _check_password(password: str, encoded: str | None) -> bool:
        try:
            algorithm, raw_iterations, salt_hex, digest_hex = (encoded or "").split("$", 3)
            if algorithm != "pbkdf2_sha256":
                return False
            digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(raw_iterations))
            return hmac.compare_digest(digest.hex(), digest_hex)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _validate_password(password: str, confirmation: str) -> None:
        if len(password) < 8:
            raise HTTPException(400, "La password deve contenere almeno 8 caratteri")
        if password != confirmation:
            raise HTTPException(400, "Le password non coincidono")

    @staticmethod
    def _validate_email(email: str) -> str:
        email = email.strip().lower()
        if "@" not in email or email.startswith("@") or email.endswith("@"):
            raise HTTPException(400, "Inserisci un indirizzo email valido")
        return email

    def _public_user(self, row: sqlite3.Row) -> dict:
        return {"id": row["id"], "email": row["email"], "display_name": row["display_name"] or row["email"]}

    def _issue_token(self, db: sqlite3.Connection, user_id: int, kind: str, hours: int) -> str:
        token = secrets.token_urlsafe(40)
        db.execute("INSERT INTO tokens(token_hash,user_id,kind,expires_at) VALUES(?,?,?,?)", (self._token_hash(token), user_id, kind, self._stamp(self._now() + timedelta(hours=hours))))
        return token

    def _send_email(self, recipient: str, subject: str, text: str, html: str) -> None:
        host = os.environ.get("VOICEMEM_SMTP_HOST", "").strip()
        if not host:
            raise RuntimeError("VOICEMEM_SMTP_HOST non configurato")
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = os.environ.get("VOICEMEM_SMTP_FROM", os.environ.get("VOICEMEM_SMTP_USER", ""))
        message["To"] = recipient
        message.set_content(text)
        message.add_alternative(html, subtype="html")
        port = int(os.environ.get("VOICEMEM_SMTP_PORT", "587"))
        with smtplib.SMTP(host, port, timeout=15) as smtp:
            if os.environ.get("VOICEMEM_SMTP_TLS", "1") != "0":
                smtp.starttls()
            user = os.environ.get("VOICEMEM_SMTP_USER", "")
            if user:
                smtp.login(user, os.environ.get("VOICEMEM_SMTP_PASSWORD", ""))
            smtp.send_message(message)

    def _send_verification(self, email: str, token: str) -> None:
        link = f"{self.frontend_url}/?verify={urllib.parse.quote(token)}"
        self._send_email(email, "Conferma il tuo account VoiceMem", f"Conferma il tuo account: {link}", f"<p>Conferma il tuo account VoiceMem:</p><p><a href=\"{link}\">Conferma email</a></p>")

    def _send_reset(self, email: str, token: str) -> None:
        link = f"{self.frontend_url}/?reset={urllib.parse.quote(token)}"
        self._send_email(email, "Reimposta la password VoiceMem", f"Reimposta la password: {link}", f"<p>Richiesta di reimpostazione password:</p><p><a href=\"{link}\">Reimposta password</a></p>")

    def register(self, email: str, password: str, confirmation: str) -> dict:
        email = self._validate_email(email)
        self._validate_password(password, confirmation)
        user_id = None
        with self._connect() as db:
            try:
                cursor = db.execute("INSERT INTO users(email,password_hash,created_at) VALUES(?,?,?)", (email, self._password_hash(password), self._stamp(self._now())))
            except sqlite3.IntegrityError:
                raise HTTPException(409, "Esiste già un account con questa email")
            user_id = cursor.lastrowid
            token = self._issue_token(db, cursor.lastrowid, "verify", 24)
        try:
            self._send_verification(email, token)
        except Exception as exc:
            with self._connect() as db:
                db.execute("DELETE FROM tokens WHERE user_id=?", (user_id,))
                db.execute("DELETE FROM users WHERE id=? AND email_verified=0", (user_id,))
            raise HTTPException(503, f"Impossibile inviare l'email di conferma: {exc}")
        return {"message": "Controlla la posta e conferma il tuo account prima di accedere"}

    def verify(self, token: str) -> None:
        with self._connect() as db:
            row = db.execute("SELECT user_id,expires_at FROM tokens WHERE token_hash=? AND kind='verify'", (self._token_hash(token),)).fetchone()
            if not row or datetime.fromisoformat(row["expires_at"]) < self._now():
                raise HTTPException(400, "Il link di conferma non è valido o è scaduto")
            db.execute("UPDATE users SET email_verified=1 WHERE id=?", (row["user_id"],))
            db.execute("DELETE FROM tokens WHERE token_hash=?", (self._token_hash(token),))

    def login(self, email: str, password: str) -> tuple[dict, str]:
        email = self._validate_email(email)
        with self._connect() as db:
            row = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
            if not row or not self._check_password(password, row["password_hash"]):
                raise HTTPException(401, "Email o password non corretti")
            if not row["email_verified"]:
                raise HTTPException(403, "Conferma prima il tuo indirizzo email")
            return self._public_user(row), self._create_session(db, row["id"])

    def _create_session(self, db: sqlite3.Connection, user_id: int) -> str:
        token = secrets.token_urlsafe(40)
        db.execute("INSERT INTO sessions(token_hash,user_id,expires_at) VALUES(?,?,?)", (self._token_hash(token), user_id, self._stamp(self._now() + timedelta(days=30))))
        return token

    def user_from_request(self, request: Request) -> dict | None:
        token = request.cookies.get(self.cookie_name)
        if not token:
            return None
        with self._connect() as db:
            row = db.execute("SELECT users.*,sessions.expires_at AS session_expires FROM sessions JOIN users ON users.id=sessions.user_id WHERE sessions.token_hash=?", (self._token_hash(token),)).fetchone()
            if not row or datetime.fromisoformat(row["session_expires"]) < self._now() or not row["email_verified"]:
                return None
            return self._public_user(row)

    def logout(self, request: Request) -> None:
        token = request.cookies.get(self.cookie_name)
        if token:
            with self._connect() as db:
                db.execute("DELETE FROM sessions WHERE token_hash=?", (self._token_hash(token),))

    def forgot(self, email: str) -> dict:
        email = self._validate_email(email)
        with self._connect() as db:
            row = db.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
            if row:
                token = self._issue_token(db, row["id"], "reset", 1)
        if row:
            try:
                self._send_reset(email, token)
            except Exception as exc:
                raise HTTPException(503, f"Impossibile inviare l'email di recupero: {exc}")
        return {"message": "Se l'email esiste, riceverai le istruzioni per recuperare l'account"}

    def reset(self, token: str, password: str, confirmation: str) -> dict:
        self._validate_password(password, confirmation)
        with self._connect() as db:
            row = db.execute("SELECT user_id,expires_at FROM tokens WHERE token_hash=? AND kind='reset'", (self._token_hash(token),)).fetchone()
            if not row or datetime.fromisoformat(row["expires_at"]) < self._now():
                raise HTTPException(400, "Il link di recupero non è valido o è scaduto")
            db.execute("UPDATE users SET password_hash=? WHERE id=?", (self._password_hash(password), row["user_id"]))
            db.execute("DELETE FROM sessions WHERE user_id=?", (row["user_id"],))
            db.execute("DELETE FROM tokens WHERE token_hash=?", (self._token_hash(token),))
        return {"message": "Password aggiornata. Ora puoi accedere"}

    def google_start(self) -> RedirectResponse:
        client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
        if not client_id:
            raise HTTPException(503, "Google login non configurato")
        state = secrets.token_urlsafe(32)
        redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI", f"{self.frontend_url}/api/auth/google/callback")
        query = urllib.parse.urlencode({"client_id": client_id, "redirect_uri": redirect_uri, "response_type": "code", "scope": "openid email profile", "state": state, "access_type": "offline", "prompt": "select_account"})
        response = RedirectResponse(f"https://accounts.google.com/o/oauth2/v2/auth?{query}")
        response.set_cookie("voicemem_google_state", state, max_age=600, httponly=True, secure=self.frontend_url.startswith("https://"), samesite="lax")
        return response

    def google_callback(self, request: Request, code: str, state: str) -> RedirectResponse:
        if not state or not hmac.compare_digest(state, request.cookies.get("voicemem_google_state", "")):
            raise HTTPException(400, "OAuth state non valido")
        redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI", f"{self.frontend_url}/api/auth/google/callback")
        payload = urllib.parse.urlencode({"code": code, "client_id": os.environ.get("GOOGLE_CLIENT_ID", ""), "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET", ""), "redirect_uri": redirect_uri, "grant_type": "authorization_code"}).encode()
        request_obj = urllib.request.Request("https://oauth2.googleapis.com/token", data=payload, headers={"Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(request_obj, timeout=15) as response:
                access = json.loads(response.read())
            info_request = urllib.request.Request("https://openidconnect.googleapis.com/v1/userinfo", headers={"Authorization": f"Bearer {access['access_token']}"})
            with urllib.request.urlopen(info_request, timeout=15) as response:
                info = json.loads(response.read())
        except Exception as exc:
            raise HTTPException(502, f"Google login non riuscito: {exc}")
        email = self._validate_email(info.get("email", ""))
        with self._connect() as db:
            row = db.execute("SELECT * FROM users WHERE google_sub=? OR email=?", (info.get("sub", ""), email)).fetchone()
            if row:
                db.execute("UPDATE users SET google_sub=?,display_name=?,email_verified=1 WHERE id=?", (info.get("sub", ""), info.get("name", ""), row["id"]))
                row = db.execute("SELECT * FROM users WHERE id=?", (row["id"],)).fetchone()
            else:
                cursor = db.execute("INSERT INTO users(email,google_sub,display_name,email_verified,created_at) VALUES(?,?,?,?,?)", (email, info.get("sub", ""), info.get("name", ""), 1, self._stamp(self._now())))
                row = db.execute("SELECT * FROM users WHERE id=?", (cursor.lastrowid,)).fetchone()
            session = self._create_session(db, row["id"])
        response = RedirectResponse(self.frontend_url)
        response.set_cookie(self.cookie_name, session, max_age=30 * 86400, httponly=True, secure=self.frontend_url.startswith("https://"), samesite="lax")
        response.delete_cookie("voicemem_google_state")
        return response


def attach_auth_routes(app) -> AuthService:
    from fastapi import Request
    auth = AuthService()

    @app.get("/api/auth/me")
    def auth_me(request: Request):
        user = auth.user_from_request(request)
        if not user:
            raise HTTPException(401, "Non autenticato")
        return user

    @app.post("/api/auth/register")
    async def auth_register(request: Request):
        body = await request.json()
        return auth.register(body.get("email", ""), body.get("password", ""), body.get("password_confirmation", ""))

    @app.get("/api/auth/verify")
    def auth_verify(request: Request, token: str):
        auth.verify(token)
        return RedirectResponse(f"{auth.frontend_url}/?verified=1")

    @app.post("/api/auth/login")
    async def auth_login(request: Request):
        body = await request.json()
        user, session = auth.login(body.get("email", ""), body.get("password", ""))
        response = JSONResponse(user)
        response.set_cookie(auth.cookie_name, session, max_age=30 * 86400, httponly=True, secure=auth.frontend_url.startswith("https://"), samesite="lax")
        return response

    @app.post("/api/auth/logout")
    def auth_logout(request: Request):
        auth.logout(request)
        response = JSONResponse({"ok": True})
        response.delete_cookie(auth.cookie_name)
        return response

    @app.post("/api/auth/forgot")
    async def auth_forgot(request: Request):
        body = await request.json()
        return auth.forgot(body.get("email", ""))

    @app.post("/api/auth/reset")
    async def auth_reset(request: Request):
        body = await request.json()
        return auth.reset(body.get("token", ""), body.get("password", ""), body.get("password_confirmation", ""))

    @app.get("/api/auth/google/start")
    def auth_google_start():
        return auth.google_start()

    @app.get("/api/auth/google/callback")
    def auth_google_callback(request: Request, code: str = "", state: str = ""):
        return auth.google_callback(request, code, state)

    return auth
