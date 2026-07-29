from flask import Flask, request, jsonify, Response, redirect, session, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import mediapipe as mp
import numpy as np

# Fix mediapipe solutions compatibility
try:
    mp.solutions.face_mesh
except AttributeError:
    pass
import os
import base64
import psycopg2
import psycopg2.extras
import psycopg2.pool
import threading
import time
from datetime import datetime, timedelta, timezone
import requests as http_requests
import smtplib
import secrets
from email.mime.text import MIMEText

# =========================================================
# EMAIL (SMTP) CONFIG — for school registration verification
# =========================================================
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("SMTP_FROM", SMTP_USER)
APP_BASE_URL = os.environ.get("APP_BASE_URL", "")  # e.g. https://attendance-app-1kwc.onrender.com

def email_is_configured():
    return bool(SMTP_HOST and SMTP_USER and SMTP_PASSWORD)

def send_email(to_email, subject, body):
    if not email_is_configured():
        print(f"[email_disabled] Would send to {to_email}: {subject}\n{body}", flush=True)
        return False, "Email is not configured on this server (missing SMTP_HOST/SMTP_USER/SMTP_PASSWORD)."
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = SMTP_FROM
        msg["To"] = to_email
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM, [to_email], msg.as_string())
        print(f"[email_sent] To {to_email}: {subject}", flush=True)
        return True, ""
    except Exception as e:
        print("Email send failed:", repr(e), flush=True)
        return False, str(e)

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set. Set it in Render's Environment tab.")

# =========================================================
# SUPABASE STORAGE CONFIG
# =========================================================
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://qsiedryjuusemdwkvcyf.supabase.co")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
SUPABASE_BUCKET = "student-images"

def supabase_upload(filename, image_bytes, content_type="image/jpeg"):
    try:
        url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{filename}"
        headers = {
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "apikey": SUPABASE_SERVICE_KEY,
            "Content-Type": content_type,
            "x-upsert": "true"
        }
        resp = http_requests.put(url, headers=headers, data=image_bytes, timeout=30)
        print(f"Supabase upload status: {resp.status_code}, response: {resp.text[:200]}")
        if resp.status_code in (200, 201):
            return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"
        # Try POST if PUT fails
        resp2 = http_requests.post(url, headers=headers, data=image_bytes, timeout=30)
        print(f"Supabase upload POST status: {resp2.status_code}, response: {resp2.text[:200]}")
        if resp2.status_code in (200, 201):
            return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"
        return None
    except Exception as e:
        print("Supabase upload exception:", e)
        return None

def supabase_delete(filename):
    try:
        url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{filename}"
        headers = {"Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
        http_requests.delete(url, headers=headers, timeout=10)
    except Exception as e:
        print("Supabase delete exception:", e)

def supabase_public_url(filename):
    if not filename:
        return ""
    if filename.startswith("http"):
        return filename
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}" 

def generate_video_thumbnail(video_bytes, ext):
    """
    Extract a cover-frame JPEG from raw video bytes using OpenCV.
    Writes to a temp file (cv2 can't read from an in-memory buffer reliably
    for compressed video containers), grabs an early frame, and returns
    JPEG-encoded bytes. Returns None on any failure so callers can fall
    back gracefully (video still works, just without a custom poster).
    """
    import tempfile
    tmp_path = None
    try:
        suffix = f".{ext}" if ext else ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return None

        # Skip a few frames in — the very first frame is sometimes black/blank
        # on some encoders. Fall back to frame 0 if the video is too short.
        frame = None
        for frame_idx in (5, 0):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, candidate = cap.read()
            if ok and candidate is not None:
                frame = candidate
                break
        cap.release()

        if frame is None:
            return None

        # Cap thumbnail resolution so it stays small/fast to load
        h, w = frame.shape[:2]
        max_dim = 480
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return None
        return buf.tobytes()
    except Exception as e:
        print("Video thumbnail generation error:", e, flush=True)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def maybe_generate_and_upload_thumb(storage_name, file_bytes, ext):
    """
    If the file is a video, generate a poster-frame thumbnail and upload it
    to Supabase next to the video. Returns the thumbnail's public URL, or
    "" if not a video or generation/upload failed (caller treats "" as "no
    custom poster" and falls back to native browser behavior).
    """
    video_exts = ("mp4", "mov", "avi", "webm", "mkv")
    if (ext or "").lower() not in video_exts:
        return ""
    thumb_bytes = generate_video_thumbnail(file_bytes, ext)
    if not thumb_bytes:
        return ""
    thumb_name = f"{storage_name}.thumb.jpg"
    thumb_url = supabase_upload(thumb_name, thumb_bytes, "image/jpeg")
    return thumb_url or ""


# MediaPipe face mesh for embeddings
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

def _get_face_embedding(rgb_image):
    """Extract a simple face embedding using MediaPipe landmarks as a feature vector."""
    try:
        import mediapipe as mp2
        face_mesh = mp2.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        results = face_mesh.process(rgb_image)
        face_mesh.close()
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32).flatten()
        coords = coords - coords.mean()
        norm = np.linalg.norm(coords)
        if norm == 0:
            return None
        return coords / norm
    except Exception as e:
        print("Embedding error:", e)
        return None

def _compare_embeddings(known_embedding, unknown_embedding, tolerance=0.6):
    """Compare two embeddings using cosine distance."""
    dist = np.linalg.norm(known_embedding - unknown_embedding)
    return dist < tolerance, dist

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY")
if not app.secret_key:
    raise RuntimeError("SECRET_KEY environment variable is not set. Set it in Render's Environment tab.")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = True  # app is served over HTTPS on Render

# =========================================================
# CSRF PROTECTION
# =========================================================
# Lightweight, dependency-free CSRF protection: a random per-session token
# is embedded in every state-changing form/AJAX call and verified on the
# server before any create/update/delete action is executed.
def get_csrf_token():
    if "_csrf_token" not in session:
        session["_csrf_token"] = secrets.token_hex(32)
    return session["_csrf_token"]

def verify_csrf():
    """Call at the top of every state-changing (POST/DELETE/etc) route."""
    token = session.get("_csrf_token")
    submitted = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
    if not submitted and request.is_json:
        submitted = (request.get_json(silent=True) or {}).get("csrf_token")
    return bool(token) and bool(submitted) and secrets.compare_digest(token, submitted)

def csrf_input_html():
    """HTML hidden-input snippet to drop inside every <form method="POST"> block."""
    return f'<input type="hidden" name="csrf_token" value="{get_csrf_token()}">'

import html as _html_mod
def esc(value):
    """Escape a value for safe interpolation into hand-built HTML strings.
    Use this around ANY user-supplied text (names, ids, messages, etc.)
    before dropping it into an f-string that produces HTML — this codebase
    does not use an auto-escaping template engine, so escaping is manual."""
    if value is None:
        return ""
    return _html_mod.escape(str(value), quote=True)

# =========================================================
# FILE UPLOAD CONTENT VALIDATION
# =========================================================
# Never trust a file's extension or the browser-supplied filename alone —
# check the actual file signature ("magic bytes") so someone can't upload
# an .html/.svg/.js payload renamed to photo.jpg to our public storage
# bucket (which would otherwise be hosted, publicly, on our own domain).
_IMAGE_MAGIC_BYTES = (
    (b"\xff\xd8\xff", "image/jpeg"),                # JPEG
    (b"\x89PNG\r\n\x1a\n", "image/png"),             # PNG
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),                          # WEBP (RIFF....WEBP)
)

def sniff_image_mime(file_bytes):
    """Returns a mimetype string if file_bytes looks like a real image,
    else None. Deliberately conservative — only the common web image
    formats this app actually needs are accepted."""
    if not file_bytes or len(file_bytes) < 12:
        return None
    for magic, mime in _IMAGE_MAGIC_BYTES:
        if file_bytes.startswith(magic):
            if magic == b"RIFF":
                if file_bytes[8:12] == b"WEBP":
                    return mime
                continue
            return mime
    return None

_VIDEO_MAGIC_CHECKS = (
    lambda b: b[4:8] in (b"ftyp",),           # mp4/mov (isobmff)
    lambda b: b[:4] == b"\x1aE\xdf\xa3",       # webm/mkv (EBML)
    lambda b: b[:4] == b"RIFF" and b[8:12] == b"AVI ",  # avi
)

def sniff_video_mime(file_bytes):
    if not file_bytes or len(file_bytes) < 12:
        return None
    for check in _VIDEO_MAGIC_CHECKS:
        try:
            if check(file_bytes):
                return "video/mp4"
        except Exception:
            continue
    return None

def is_allowed_upload(file_bytes, allow_video=False):
    """True if file_bytes sniffs as a real image (and, optionally, video)."""
    if sniff_image_mime(file_bytes):
        return True
    if allow_video and sniff_video_mime(file_bytes):
        return True
    return False

_DANGEROUS_SIGNATURES = (
    b"<script", b"<html", b"<!doctype html", b"<?php", b"MZ",  # PE/EXE header
)

def sniff_is_dangerous(file_bytes):
    """For general (non-image-restricted) attachment uploads: reject files
    that look like HTML/JS/PHP/executables regardless of their extension —
    these are the file types that can turn a storage bucket on our own
    domain into a phishing or malware host."""
    if not file_bytes:
        return False
    head = file_bytes[:512].lower()
    for sig in _DANGEROUS_SIGNATURES:
        if sig.lower() in head:
            return True
    return False

# Route name prefixes that are allowed to skip CSRF checks (none by default;
# add sparingly, e.g. for a webhook endpoint verified by its own signature).
CSRF_EXEMPT_ENDPOINTS = set()

@app.before_request
def _enforce_csrf():
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        if request.endpoint in CSRF_EXEMPT_ENDPOINTS:
            return None
        if not verify_csrf():
            if request.is_json or request.path.startswith("/api/"):
                return jsonify({"ok": False, "error": "Invalid or missing CSRF token."}), 400
            return "Invalid or missing CSRF token. Please refresh the page and try again.", 400
    return None

@app.after_request
def _set_csrf_cookie(resp):
    # Expose the token to JS (non-HttpOnly, small, low-value token) so
    # fetch()-based AJAX calls can send it back as a header.
    try:
        resp.set_cookie("csrf_token", get_csrf_token(), samesite="Lax", secure=True, httponly=False)
    except RuntimeError:
        pass
    return resp

DB_FILE = "attendance.db"

# =========================================================
# LOGIN RATE LIMITING (brute-force protection)
# =========================================================
# Simple in-memory sliding-window limiter. Not shared across multiple
# worker processes/dynos, but still meaningfully raises the cost of
# password guessing on a single-process deployment. For multi-worker
# deployments, back this with Redis/DB instead.
from collections import defaultdict, deque
import threading

_LOGIN_ATTEMPTS = defaultdict(deque)
_LOGIN_LOCK = threading.Lock()
LOGIN_MAX_ATTEMPTS = 8
LOGIN_WINDOW_SECONDS = 5 * 60

def _client_ip():
    fwd = request.headers.get("X-Forwarded-For", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.remote_addr or "unknown"

def login_rate_limited(bucket):
    """Returns True (and records the attempt) if this IP+bucket has exceeded
    the allowed number of login attempts in the current window."""
    key = (bucket, _client_ip())
    now = datetime.now(timezone.utc).timestamp()
    with _LOGIN_LOCK:
        q = _LOGIN_ATTEMPTS[key]
        while q and now - q[0] > LOGIN_WINDOW_SECONDS:
            q.popleft()
        if len(q) >= LOGIN_MAX_ATTEMPTS:
            return True
        q.append(now)
        return False

def too_many_attempts_response():
    return "Too many login attempts. Please wait a few minutes and try again.", 429
# On Vercel (and most serverless hosts) the project directory is read-only —
# only /tmp is writable, and it's wiped between cold starts. These local
# folders are only ever used as a best-effort fallback/cache; Supabase
# Storage (see supabase_upload/supabase_public_url above) is the actual
# source of truth for image files. Default to /tmp so writes don't crash
# on read-only filesystems; override with IMAGE_DIR/TEACHER_IMAGE_DIR env
# vars on hosts (like Render) with a persistent writable disk.
IMAGE_DIR = os.environ.get("IMAGE_DIR", "/tmp/student_images")
TEACHER_IMAGE_DIR = os.environ.get("TEACHER_IMAGE_DIR", "/tmp/teacher_images")


# Force HTTPS — camera (getUserMedia) requires a secure context in all browsers
@app.before_request
def force_https():
    # Only redirect on deployed platforms (they set X-Forwarded-Proto)
    if request.headers.get("X-Forwarded-Proto", "https") == "http":
        return redirect(request.url.replace("http://", "https://", 1), code=301)

known_encodings = []
known_students = []


# =========================================================
# DATABASE  (pooled — see notes below)
# =========================================================
# The app previously opened a brand-new psycopg2 connection (fresh TCP +
# TLS + Postgres auth handshake) on every single get_db() call, and there
# are 100+ call sites. A single QR check-in request chains through many
# helper functions (session lookup, GPS check, WiFi check, class lookup,
# mark_attendance, ...), each grabbing its own connection — so one tap on
# "Scan QR" could trigger 8-10+ handshakes to the database, which is what
# was causing the multi-second delay after scanning.
#
# Fix: keep a small pool of already-open connections around and hand them
# out instantly. To avoid touching every one of the ~106 existing
# `conn = get_db(); ...; conn.close()` call sites, get_db() returns a thin
# wrapper whose .close() RETURNS the connection to the pool instead of
# actually closing it. Everything else (cur = conn.cursor(), conn.commit(),
# row dicts via RealDictCursor, etc.) behaves exactly as before.
_db_pool = None
_db_pool_lock = threading.Lock()
_db_pool_last_failure = 0.0
_DB_POOL_FAILURE_COOLDOWN = 15  # seconds — how long to fail fast before retrying pool creation


def _get_pool():
    global _db_pool, _db_pool_last_failure
    if _db_pool is None:
        # If we tried very recently and it failed (e.g. bad DATABASE_URL, DB
        # unreachable), don't spend another 5-10s re-attempting on every
        # single incoming request — fail immediately until the cooldown
        # passes. This is what actually prevents the "stuck spinner": before
        # this check, EVERY request during an outage independently paid the
        # full connection-timeout cost.
        if _db_pool_last_failure and (time.time() - _db_pool_last_failure) < _DB_POOL_FAILURE_COOLDOWN:
            raise psycopg2.OperationalError(
                "Database is currently unreachable (failing fast; will retry pool creation automatically)."
            )
        with _db_pool_lock:
            if _db_pool is None:
                try:
                    _db_pool = psycopg2.pool.ThreadedConnectionPool(
                        minconn=2,
                        maxconn=30,
                        dsn=DATABASE_URL,
                        cursor_factory=psycopg2.extras.RealDictCursor,
                        connect_timeout=5,
                        # connect_timeout above only bounds the initial TCP/auth
                        # handshake. Without a statement_timeout, a query sent to
                        # a DB that's mid-wakeup, stalled, or on a dead-but-not-
                        # yet-detected connection can hang forever with no
                        # exception ever raised — which is what produces a
                        # "Submitting attendance…" spinner that never resolves,
                        # even though the client-side fetch has its own abort
                        # timer (that timer only helps once the server actually
                        # responds; it can't unstick a hung server-side query).
                        # 15s here is intentionally shorter than the client's
                        # 20s AbortController so the student gets a real error
                        # message ("Database error...") instead of silence.
                        options="-c statement_timeout=15000",
                        # TCP keepalives so a connection that's gone dead at the
                        # network level (host went to sleep, NAT/proxy dropped
                        # it, etc.) is detected within ~30-50s instead of the
                        # OS default (which can be hours), so it gets recycled
                        # by get_db()'s staleness check instead of handed out
                        # as a zombie connection.
                        keepalives=1,
                        keepalives_idle=20,
                        keepalives_interval=10,
                        keepalives_count=3,
                    )
                    _db_pool_last_failure = 0.0
                except Exception:
                    _db_pool_last_failure = time.time()
                    raise
    return _db_pool


class _PooledConnection:
    """Wraps a pooled psycopg2 connection so existing code (which calls
    conn.close() when it's done) returns the connection to the pool
    instead of tearing down the TCP/TLS session."""

    def __init__(self, conn, pool):
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "_pool", pool)
        object.__setattr__(self, "_returned", False)

    def close(self):
        if self._returned:
            return
        object.__setattr__(self, "_returned", True)
        try:
            # Never hand back a connection mid-transaction — leftover
            # uncommitted work from a caller that errored out before
            # calling commit() would otherwise leak into the next request
            # that picks up this connection from the pool.
            self._conn.rollback()
            self._pool.putconn(self._conn)
        except Exception:
            try:
                self._pool.putconn(self._conn, close=True)
            except Exception:
                try:
                    self._conn.close()
                except Exception:
                    pass

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __setattr__(self, name, value):
        setattr(self._conn, name, value)

    def __enter__(self):
        return self._conn.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._conn.__exit__(exc_type, exc_val, exc_tb)

    def __del__(self):
        # Safety net: if some code path forgets to call conn.close() (e.g. an
        # unexpected exception skips past it), don't let that connection be
        # lost forever — return it to the pool when this wrapper is garbage
        # collected instead of silently shrinking the pool's capacity over
        # time (which would eventually make every request queue/time out).
        try:
            self.close()
        except Exception:
            pass


def get_db():
    pool = _get_pool()
    last_err = None
    for attempt in range(2):
        try:
            raw_conn = pool.getconn()
            # A connection can go stale (DB restart, idle timeout from the
            # DB host, network blip) while sitting in the pool. Detect that
            # and discard it rather than handing back a dead connection.
            if raw_conn.closed:
                try:
                    pool.putconn(raw_conn, close=True)
                except Exception:
                    pass
                continue
            try:
                raw_conn.rollback()
            except Exception:
                try:
                    pool.putconn(raw_conn, close=True)
                except Exception:
                    pass
                continue
            return _PooledConnection(raw_conn, pool)
        except Exception as e:
            last_err = e
            print(f"get_db pool getconn attempt {attempt + 1} failed:", repr(e), flush=True)
    raise last_err


def init_db():
    try:
        os.makedirs(IMAGE_DIR, exist_ok=True)
        os.makedirs(TEACHER_IMAGE_DIR, exist_ok=True)
    except OSError as e:
        # Read-only filesystem (e.g. project root on Vercel) — fine, these
        # dirs are only a fallback cache; Supabase Storage is the source
        # of truth for image files.
        print("Skipping local image dir creation (read-only fs):", e, flush=True)
    conn = get_db()
    cur = conn.cursor()

    # ── SCHOOLS TABLE (multi-tenancy root) ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS schools (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            code TEXT UNIQUE NOT NULL,
            admin_username TEXT UNIQUE NOT NULL,
            admin_password TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    # Seed a default school so existing data isn't broken
    cur.execute("""
        INSERT INTO schools (name, code, admin_username, admin_password, created_at)
        VALUES ('Default School', 'DEFAULT', 'admin', 'admin123', %s)
        ON CONFLICT (code) DO NOTHING
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))

    # ── PENDING SCHOOL REGISTRATIONS (email verification) ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pending_school_registrations (
            id SERIAL PRIMARY KEY,
            token TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            code TEXT NOT NULL,
            admin_username TEXT NOT NULL,
            admin_password TEXT NOT NULL,
            email TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            expires_at TIMESTAMP NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            id SERIAL PRIMARY KEY,
            school_id INTEGER NOT NULL DEFAULT 1,
            teacher_name TEXT NOT NULL,
            username TEXT NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL,
            photo_path TEXT,
            UNIQUE(school_id, username),
            FOREIGN KEY (school_id) REFERENCES schools(id)
        )
    """)
    # Migrate existing teachers: assign to school 1 if column missing
    cur.execute("ALTER TABLE teachers ADD COLUMN IF NOT EXISTS school_id INTEGER NOT NULL DEFAULT 1")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id SERIAL PRIMARY KEY,
            school_id INTEGER NOT NULL DEFAULT 1,
            student_id TEXT NOT NULL,
            full_name TEXT NOT NULL,
            password TEXT NOT NULL,
            image_file TEXT NOT NULL,
            registered_at TEXT NOT NULL,
            UNIQUE(school_id, student_id),
            FOREIGN KEY (school_id) REFERENCES schools(id)
        )
    """)
    cur.execute("ALTER TABLE students ADD COLUMN IF NOT EXISTS school_id INTEGER NOT NULL DEFAULT 1")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS classes (
            id SERIAL PRIMARY KEY,
            school_id INTEGER NOT NULL DEFAULT 1,
            class_name TEXT NOT NULL,
            department TEXT,
            course TEXT,
            section_name TEXT,
            subject_name TEXT,
            teacher_id INTEGER NOT NULL,
            teacher_display_name TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (teacher_id) REFERENCES teachers(id),
            FOREIGN KEY (school_id) REFERENCES schools(id)
        )
    """)
    cur.execute("ALTER TABLE classes ADD COLUMN IF NOT EXISTS school_id INTEGER NOT NULL DEFAULT 1")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS student_classes (
            id SERIAL PRIMARY KEY,
            student_id_fk INTEGER NOT NULL,
            class_id_fk INTEGER NOT NULL,
            UNIQUE(student_id_fk, class_id_fk),
            FOREIGN KEY (student_id_fk) REFERENCES students(id),
            FOREIGN KEY (class_id_fk) REFERENCES classes(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id SERIAL PRIMARY KEY,
            school_id INTEGER NOT NULL DEFAULT 1,
            student_id TEXT NOT NULL,
            full_name TEXT NOT NULL,
            class_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            department TEXT,
            course TEXT,
            section_name TEXT,
            subject_name TEXT,
            teacher_name TEXT,
            status TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            UNIQUE(school_id, student_id, class_id, date),
            FOREIGN KEY (class_id) REFERENCES classes(id)
        )
    """)
    cur.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS school_id INTEGER NOT NULL DEFAULT 1")
    # admin_settings: migrate from old single-PK schema to (school_id, key) composite PK
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='admin_settings' AND column_name='school_id'
            ) THEN
                DROP TABLE IF EXISTS admin_settings;
            END IF;
        END$$;
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS admin_settings (
            school_id INTEGER NOT NULL DEFAULT 1,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (school_id, key)
        )
    """)
    # Seed default admin password for school 1 if not already set
    cur.execute("""
        INSERT INTO admin_settings (school_id, key, value)
        VALUES (1, 'admin_password', %s)
        ON CONFLICT (school_id, key) DO NOTHING
    """, (hash_password("admin123"),))

    # ── CLASS SESSIONS TABLE (QR/Session code check-in) ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS class_sessions (
            id SERIAL PRIMARY KEY,
            school_id INTEGER NOT NULL DEFAULT 1,
            class_id INTEGER NOT NULL,
            code TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            expires_at TIMESTAMP NOT NULL,
            active BOOLEAN NOT NULL DEFAULT TRUE,
            teacher_lat DOUBLE PRECISION,
            teacher_lng DOUBLE PRECISION,
            teacher_ip TEXT,
            rotate_seconds INTEGER NOT NULL DEFAULT 60,
            last_rotated TIMESTAMP NOT NULL DEFAULT NOW(),
            FOREIGN KEY (class_id) REFERENCES classes(id)
        )
    """)
    # Migrate existing class_sessions if columns missing
    for col, typedef in [
        ("teacher_lat", "DOUBLE PRECISION"),
        ("teacher_lng", "DOUBLE PRECISION"),
        ("teacher_ip", "TEXT"),
        ("rotate_seconds", "INTEGER NOT NULL DEFAULT 60"),
        ("last_rotated", "TIMESTAMP NOT NULL DEFAULT NOW()"),
    ]:
        cur.execute(f"ALTER TABLE class_sessions ADD COLUMN IF NOT EXISTS {col} {typedef}")

    # ── STUDENT GPS COLUMNS IN ATTENDANCE ──
    cur.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS student_lat DOUBLE PRECISION")
    cur.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS student_lng DOUBLE PRECISION")
    cur.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS distance_meters DOUBLE PRECISION")

    # ── SESSION TOKENS TABLE (one active device per user) ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_session_tokens (
            id SERIAL PRIMARY KEY,
            user_type TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            school_id INTEGER NOT NULL,
            token TEXT NOT NULL UNIQUE,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ust_user ON user_session_tokens(user_type, user_id, school_id)")

    # ── CLASS SOCIAL FEED: comments per class ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS class_comments (
            id SERIAL PRIMARY KEY,
            school_id INTEGER NOT NULL DEFAULT 1,
            class_id INTEGER NOT NULL,
            student_db_id INTEGER NOT NULL,
            student_name TEXT NOT NULL,
            student_image TEXT,
            comment TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            FOREIGN KEY (class_id) REFERENCES classes(id),
            FOREIGN KEY (student_db_id) REFERENCES students(id)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_cc_class ON class_comments(class_id)")

    # Student priority flag (teacher can mark a student as high priority per class)
    cur.execute("ALTER TABLE student_classes ADD COLUMN IF NOT EXISTS is_priority BOOLEAN NOT NULL DEFAULT FALSE")

    # Migrate class_comments to support teacher posts
    cur.execute("ALTER TABLE class_comments ADD COLUMN IF NOT EXISTS poster_type TEXT NOT NULL DEFAULT 'student'")
    cur.execute("ALTER TABLE class_comments ADD COLUMN IF NOT EXISTS teacher_id_fk INTEGER DEFAULT NULL")
    # Allow student_db_id to be 0 for teacher posts
    cur.execute("ALTER TABLE class_comments ALTER COLUMN student_db_id DROP NOT NULL")
    cur.execute("ALTER TABLE class_comments DROP CONSTRAINT IF EXISTS class_comments_student_db_id_fkey")
    # High priority / pinned flag set by teacher
    cur.execute("ALTER TABLE class_comments ADD COLUMN IF NOT EXISTS is_priority BOOLEAN NOT NULL DEFAULT FALSE")

    # ── DIRECT MESSAGES (student ↔ student within same class) ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS direct_messages (
            id SERIAL PRIMARY KEY,
            school_id INTEGER NOT NULL DEFAULT 1,
            class_id INTEGER NOT NULL,
            sender_db_id INTEGER NOT NULL,
            receiver_db_id INTEGER NOT NULL,
            sender_name TEXT NOT NULL,
            sender_image TEXT,
            message TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            is_read BOOLEAN NOT NULL DEFAULT FALSE
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_dm_receiver ON direct_messages(receiver_db_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_dm_sender ON direct_messages(sender_db_id)")

    # ── FILE ATTACHMENTS ──
    cur.execute("ALTER TABLE class_comments ADD COLUMN IF NOT EXISTS file_url TEXT DEFAULT NULL")
    cur.execute("ALTER TABLE class_comments ADD COLUMN IF NOT EXISTS file_name TEXT DEFAULT NULL")
    cur.execute("ALTER TABLE direct_messages ADD COLUMN IF NOT EXISTS file_url TEXT DEFAULT NULL")
    cur.execute("ALTER TABLE direct_messages ADD COLUMN IF NOT EXISTS file_name TEXT DEFAULT NULL")
    # Video cover/poster thumbnail (auto-generated server-side on upload)
    cur.execute("ALTER TABLE class_comments ADD COLUMN IF NOT EXISTS file_thumb_url TEXT DEFAULT NULL")
    cur.execute("ALTER TABLE direct_messages ADD COLUMN IF NOT EXISTS file_thumb_url TEXT DEFAULT NULL")

    conn.commit()
    conn.close()


# =========================================================
# HELPERS
# =========================================================
import math


# ── Single-device session token helpers ──
def create_user_session_token(user_type, user_id, school_id):
    """Invalidate all previous tokens for this user and issue a fresh one."""
    token = secrets.token_hex(32)
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("DELETE FROM user_session_tokens WHERE user_type=%s AND user_id=%s AND school_id=%s",
                    (user_type, user_id, school_id))
        cur.execute("INSERT INTO user_session_tokens (user_type, user_id, school_id, token) VALUES (%s,%s,%s,%s)",
                    (user_type, user_id, school_id, token))
        conn.commit()
        conn.close()
    except Exception as e:
        print("create_user_session_token error:", e)
    return token

def validate_user_session_token(user_type, user_id, school_id, token):
    """Return True only if this exact token is the current active one."""
    if not token:
        return False
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT id FROM user_session_tokens WHERE user_type=%s AND user_id=%s AND school_id=%s AND token=%s",
                    (user_type, user_id, school_id, token))
        row = cur.fetchone()
        conn.close()
        return row is not None
    except Exception as e:
        print("validate_user_session_token error:", e)
        return True  # on DB error, don't block

def revoke_user_session_token(user_type, user_id, school_id):
    """Remove all tokens (used on logout)."""
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("DELETE FROM user_session_tokens WHERE user_type=%s AND user_id=%s AND school_id=%s",
                    (user_type, user_id, school_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print("revoke_user_session_token error:", e)

def utc_iso(dt):
    """
    Serialize a datetime to an ISO string the browser will parse correctly.

    The server (datetime.now(), and Postgres TIMESTAMP-without-timezone columns
    read back via psycopg2) produces NAIVE datetimes that represent UTC wall-clock
    time, but carry no timezone marker. If we call .isoformat() on those directly,
    the string looks like "2026-06-28T15:32:00" with no "Z"/"+00:00" suffix.
    JavaScript's `new Date(...)` then parses that as LOCAL browser time, not UTC —
    so for any visitor in a timezone ahead of UTC (e.g. UTC+3), the parsed expiry
    time ends up hours in the past compared to the server's intent, and the
    attendance countdown shows "Time is up!" immediately on page load.
    Tagging the datetime as UTC before formatting fixes this for every timezone.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def render_file_html(file_url, file_name, file_thumb_url=None):
    """Return rich HTML preview for a file attachment (image/video/doc)."""
    if not file_url:
        return ""
    ext = (file_name or "").rsplit(".", 1)[-1].lower() if file_name else ""
    if ext in ("jpg", "jpeg", "png", "gif", "webp"):
        return (f'<div style="margin-top:5px;">'
                f'<a href="{file_url}" target="_blank">'
                f'<img src="{file_url}" style="max-width:220px;max-height:180px;border-radius:8px;display:block;object-fit:cover;">'
                f'</a></div>')
    elif ext in ("mp4", "mov", "avi", "webm", "mkv"):
        mime = "video/quicktime" if ext == "mov" else f"video/{ext}"
        poster_attr = f' poster="{file_thumb_url}"' if file_thumb_url else ""
        return (f'<div style="margin-top:5px;border-radius:8px;overflow:hidden;max-width:260px;background:#000;">'
                f'<video preload="metadata"{poster_attr} playsinline '
                f'onclick="if(this.requestFullscreen){{this.requestFullscreen();}}else if(this.webkitRequestFullscreen){{this.webkitRequestFullscreen();}}this.controls=true;this.play();" '
                f'style="display:block;max-width:260px;max-height:200px;border-radius:8px;width:100%;cursor:pointer;">'
                f'<source src="{file_url}" type="{mime}"></video></div>')
    else:
        icons = {"pdf":"📕","doc":"📝","docx":"📝","xls":"📊","xlsx":"📊","ppt":"📋","pptx":"📋",
                 "zip":"🗜️","rar":"🗜️","txt":"📄","mp3":"🎵","wav":"🎵"}
        icon = icons.get(ext, "📄")
        label = (file_name or "Download file").replace("&","&amp;").replace("<","&lt;")
        return (f'<div style="margin-top:5px;">'
                f'<a href="{file_url}" target="_blank" style="display:inline-flex;align-items:center;gap:8px;'
                f'background:rgba(0,0,0,0.2);border:1px solid rgba(91,155,217,0.3);border-radius:10px;'
                f'padding:8px 12px;text-decoration:none;max-width:220px;">'
                f'<span style="font-size:22px;flex-shrink:0;">{icon}</span>'
                f'<span style="color:#5b9bd9;font-size:12px;word-break:break-all;line-height:1.3;">{label}</span>'
                f'</a></div>')


def haversine_distance(lat1, lon1, lat2, lon2):
    """Returns distance in meters between two GPS coordinates."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def check_gps_valid(student_lat, student_lng, school_id):
    """Returns (passed, distance_meters). Passes if no classroom GPS set."""
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT value FROM admin_settings WHERE school_id=%s AND key='classroom_lat'", (school_id,))
        lat_row = cur.fetchone()
        cur.execute("SELECT value FROM admin_settings WHERE school_id=%s AND key='classroom_lng'", (school_id,))
        lng_row = cur.fetchone()
        cur.execute("SELECT value FROM admin_settings WHERE school_id=%s AND key='classroom_radius'", (school_id,))
        radius_row = cur.fetchone()
        conn.close()
        if not lat_row or not lng_row:
            return True, 0  # GPS not configured — skip check
        class_lat = float(lat_row["value"])
        class_lng = float(lng_row["value"])
        radius = float(radius_row["value"]) if radius_row else 100.0
        if student_lat is None or student_lng is None:
            return False, None
        dist = haversine_distance(float(student_lat), float(student_lng), class_lat, class_lng)
        return dist <= radius, dist
    except Exception as e:
        print("GPS check error:", e)
        return True, 0  # On error, don't block


def check_session_code_valid(code, class_id, school_id):
    """Returns True if the session code is valid and not expired for this class."""
    if not code:
        return False
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT id FROM class_sessions
            WHERE school_id=%s AND class_id=%s AND code=upper(%s)
            AND active=TRUE AND expires_at > NOW()
        """, (school_id, class_id, code.strip()))
        row = cur.fetchone()
        conn.close()
        return row is not None
    except Exception as e:
        print("Session code check error:", e)
        return False


def check_wifi_valid(request_obj, school_id, teacher_ip=None):
    """Returns True if student's IP matches the school's allowed IP prefix,
       OR if they share the same /24 subnet as the teacher (same WiFi enforcement)."""
    try:
        student_ip = request_obj.headers.get("X-Forwarded-For", request_obj.remote_addr or "").split(",")[0].strip()

        # Same-WiFi as teacher: compare first 3 octets (same /24 subnet)
        if teacher_ip:
            teacher_prefix = ".".join(teacher_ip.split(".")[:3]) + "."
            if student_ip.startswith(teacher_prefix):
                return True

        # Fallback: admin-configured IP prefix
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT value FROM admin_settings WHERE school_id=%s AND key='allowed_ip_prefix'", (school_id,))
        row = cur.fetchone()
        conn.close()
        if not row or not row["value"].strip():
            return True  # WiFi not configured — skip check
        allowed_prefix = row["value"].strip()
        return student_ip.startswith(allowed_prefix)
    except Exception as e:
        print("WiFi check error:", e)
        return True  # On error, don't block


def get_active_session_for_class(class_id, school_id):
    """Returns the active session row (with teacher GPS + IP) or None."""
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT code, teacher_lat, teacher_lng, teacher_ip, rotate_seconds, expires_at
            FROM class_sessions
            WHERE class_id=%s AND school_id=%s AND active=TRUE AND expires_at > NOW()
            ORDER BY id DESC LIMIT 1
        """, (class_id, school_id))
        row = cur.fetchone()
        conn.close()
        return row
    except Exception as e:
        print("get_active_session error:", e)
        return None


def is_attendance_session_active(class_id, school_id):
    """True only if the teacher has started (and not stopped/expired) an attendance session for this class.
    Attendance can never be marked while this is False, regardless of GPS/WiFi/session-code checks."""
    return get_active_session_for_class(class_id, school_id) is not None


def run_location_checks(student_lat, student_lng, session_code, class_id, school_id, request_obj):
    """
    Flexible check: passes if ANY ONE of GPS / Session Code / WiFi passes.
    Uses teacher GPS (from active session) as primary reference, then falls back to admin classroom GPS.
    Returns (passed: bool, reason: str, distance_meters: float|None)
    """
    reasons = []
    distance_out = None

    # Retrieve active session to get teacher GPS + IP for same-WiFi check
    active_sess = get_active_session_for_class(class_id, school_id)
    teacher_ip = active_sess["teacher_ip"] if active_sess else None

    # GPS check — prefer teacher's live location over fixed classroom coords
    if student_lat is not None and student_lng is not None:
        gps_checked = False
        if active_sess and active_sess["teacher_lat"] and active_sess["teacher_lng"]:
            try:
                dist = haversine_distance(float(student_lat), float(student_lng),
                                          float(active_sess["teacher_lat"]), float(active_sess["teacher_lng"]))
                distance_out = dist
                # Allow within 200 m of teacher
                if dist <= 200:
                    return True, f"GPS: {int(dist)}m from teacher", distance_out
                else:
                    reasons.append(f"GPS: {int(dist)}m from teacher (>200m)")
                gps_checked = True
            except Exception as e:
                print("Teacher GPS check error:", e)
        if not gps_checked:
            gps_ok, dist = check_gps_valid(student_lat, student_lng, school_id)
            if dist:
                distance_out = dist
            if gps_ok:
                return True, "GPS location verified", distance_out
            else:
                reasons.append(f"GPS failed ({int(dist)}m away)" if dist is not None else "GPS location unavailable")
    else:
        reasons.append("GPS location unavailable")

    # Session code check
    code_ok = check_session_code_valid(session_code, class_id, school_id)
    if code_ok:
        return True, "Session code verified", distance_out
    else:
        reasons.append("Session code invalid or expired")

    # WiFi check (same subnet as teacher or admin-configured prefix)
    wifi_ok = check_wifi_valid(request_obj, school_id, teacher_ip=teacher_ip)
    if wifi_ok:
        return True, "Same WiFi as teacher ✓", distance_out
    else:
        reasons.append("Not on same WiFi as teacher")

    return False, " | ".join(reasons), distance_out


def is_ajax():
    return request.headers.get('X-Requested-With') == 'XMLHttpRequest'

def ajax_ok(message="Done!", redirect_url=None):
    return jsonify({"ok": True, "message": message, "redirect": redirect_url})

def ajax_err(message="Something went wrong."):
    return jsonify({"ok": False, "error": message})

def format_time_12hr(time_str):
    """Convert HH:MM:SS or HH:MM to 12-hour AM/PM format."""
    if not time_str:
        return time_str
    try:
        # Already in 12hr format
        if 'AM' in str(time_str).upper() or 'PM' in str(time_str).upper():
            return time_str
        from datetime import datetime as dt
        for fmt in ('%H:%M:%S', '%H:%M'):
            try:
                return dt.strptime(time_str.strip(), fmt).strftime('%I:%M %p')
            except:
                continue
        return time_str
    except:
        return time_str

def sanitize_filename(text):
    text = "".join(c for c in text if c.isalnum() or c in (" ", "_", "-")).strip()
    return text.replace(" ", "_")


# =========================================================
# PASSWORD HASHING HELPERS
# =========================================================
# New passwords are always stored hashed (Werkzeug/pbkdf2). Older rows
# created before hashing was added may still contain plaintext values —
# verify_password() supports both so existing accounts keep working, and
# every successful legacy login is transparently upgraded to a hash.
def hash_password(plain_password):
    return generate_password_hash(plain_password)

def is_hashed_password(value):
    return bool(value) and value.startswith(("pbkdf2:", "scrypt:"))

def verify_password(stored_value, provided_password):
    if not stored_value or provided_password is None:
        return False
    if is_hashed_password(stored_value):
        try:
            return check_password_hash(stored_value, provided_password)
        except Exception:
            return False
    # Legacy plaintext row — compare directly (will be rehashed on success)
    return stored_value == provided_password


def get_admin_password():
    school_id = session.get("school_id", 1)
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT value FROM admin_settings WHERE school_id=%s AND key='admin_password'", (school_id,))
        row = cur.fetchone()
        conn.close()
        return row["value"] if row else "admin123"
    except:
        return "admin123"

def set_admin_password(new_password, school_id=None):
    if school_id is None:
        school_id = session.get("school_id", 1)
    hashed = hash_password(new_password)
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO admin_settings (school_id, key, value) VALUES (%s, 'admin_password', %s) ON CONFLICT (school_id, key) DO UPDATE SET value=%s",
        (school_id, hashed, hashed)
    )
    conn.commit()
    conn.close()

# ── SCHOOL HELPERS ──
def get_current_school_id():
    try:
        return session.get("school_id", 1)
    except RuntimeError:
        return 1  # No request context (startup)

def get_all_schools():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM schools ORDER BY name")
    rows = cur.fetchall()
    conn.close()
    return rows

def get_school_by_id(school_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM schools WHERE id=%s", (school_id,))
    row = cur.fetchone()
    conn.close()
    return row

def get_school_by_code(code):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM schools WHERE upper(code)=upper(%s)", (code,))
    row = cur.fetchone()
    conn.close()
    return row

def is_admin_logged_in():
    return session.get("admin_logged_in") is True


def is_teacher_logged_in():
    return session.get("teacher_logged_in") is True


def is_student_logged_in():
    return session.get("student_logged_in") is True


def get_logged_teacher_id():
    return session.get("teacher_id")


def get_logged_student_db_id():
    return session.get("student_db_id")


def admin_required():
    if not is_admin_logged_in():
        return redirect("/admin-login")
    return None


def teacher_required():
    if not is_teacher_logged_in():
        return redirect("/teacher-login")
    # One-device check
    teacher_id = session.get("teacher_id")
    school_id = session.get("school_id", 1)
    token = session.get("device_token")
    if teacher_id and not validate_user_session_token("teacher", teacher_id, school_id, token):
        session.clear()
        return redirect("/teacher-login?kicked=1")
    return None


def student_required():
    if not is_student_logged_in():
        return redirect("/student-login")
    # One-device check
    student_db_id = session.get("student_db_id")
    school_id = session.get("school_id", 1)
    token = session.get("device_token")
    if student_db_id and not validate_user_session_token("student", student_db_id, school_id, token):
        session.clear()
        return redirect("/student-login?kicked=1")
    return None



def student_exists(student_id, school_id=None):
    if school_id is None:
        school_id = get_current_school_id()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM students WHERE lower(student_id)=lower(%s) AND school_id=%s", (student_id.strip(), school_id))
    row = cur.fetchone()
    conn.close()
    return row is not None


def teacher_username_exists(username, school_id=None):
    if school_id is None:
        school_id = get_current_school_id()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM teachers WHERE lower(username)=lower(%s) AND school_id=%s", (username.strip(), school_id))
    row = cur.fetchone()
    conn.close()
    return row is not None


def get_all_students(school_id=None):
    if school_id is None:
        school_id = get_current_school_id()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM students WHERE school_id=%s ORDER BY id DESC", (school_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_all_teachers(school_id=None):
    if school_id is None:
        school_id = get_current_school_id()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM teachers WHERE school_id=%s ORDER BY id DESC", (school_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_all_classes(school_id=None):
    if school_id is None:
        school_id = get_current_school_id()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT c.*, t.teacher_name
        FROM classes c
        LEFT JOIN teachers t ON c.teacher_id = t.id
        WHERE c.school_id=%s
        ORDER BY c.id DESC
    """, (school_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_teacher_classes(teacher_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT c.*, t.teacher_name
        FROM classes c
        LEFT JOIN teachers t ON c.teacher_id = t.id
        WHERE c.teacher_id=%s
        ORDER BY c.id DESC
    """, (teacher_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_class_by_id(class_id, school_id=None):
    # SECURITY: always scope to a school so one school's admin/teacher can't
    # read or act on another school's class by guessing/incrementing the id.
    if school_id is None:
        school_id = get_current_school_id()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT c.*, t.teacher_name
        FROM classes c
        LEFT JOIN teachers t ON c.teacher_id = t.id
        WHERE c.id=%s AND c.school_id=%s
    """, (class_id, school_id))
    row = cur.fetchone()
    conn.close()
    return row


def get_student_row_by_student_id(student_id_text, school_id=None):
    if school_id is None:
        school_id = get_current_school_id()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM students WHERE student_id=%s AND school_id=%s", (student_id_text, school_id))
    row = cur.fetchone()
    conn.close()
    return row


def get_student_row_by_db_id(student_db_id, school_id=None):
    # SECURITY: always scope to a school so one school's admin/teacher can't
    # read or act on another school's student by guessing/incrementing the id.
    if school_id is None:
        school_id = get_current_school_id()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM students WHERE id=%s AND school_id=%s", (student_db_id, school_id))
    row = cur.fetchone()
    conn.close()
    return row


def get_students_in_class(class_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT s.*, sc.is_priority AS is_priority
        FROM students s
        INNER JOIN student_classes sc ON sc.student_id_fk = s.id
        WHERE sc.class_id_fk=%s
        ORDER BY sc.is_priority DESC, s.full_name
    """, (class_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_classes_for_student(student_db_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT c.*
        FROM classes c
        INNER JOIN student_classes sc ON sc.class_id_fk = c.id
        WHERE sc.student_id_fk=%s
        ORDER BY c.class_name
    """, (student_db_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def assign_student_to_class(student_db_id, class_id):
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO student_classes (student_id_fk, class_id_fk)
            VALUES (%s, %s)
        """, (student_db_id, class_id))
        conn.commit()
    except:
        pass
    conn.close()


def remove_student_from_class(student_db_id, class_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        DELETE FROM student_classes
        WHERE student_id_fk=%s AND class_id_fk=%s
    """, (student_db_id, class_id))
    conn.commit()
    conn.close()


def get_all_attendance(school_id=None):
    if school_id is None:
        school_id = get_current_school_id()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM attendance WHERE school_id=%s ORDER BY id DESC", (school_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_attendance_for_teacher(teacher_name, school_id=None):
    if school_id is None:
        school_id = get_current_school_id()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM attendance
        WHERE teacher_name=%s AND school_id=%s
        ORDER BY id DESC
    """, (teacher_name, school_id))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_attendance_for_class(class_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM attendance
        WHERE class_id=%s
        ORDER BY date DESC, time DESC
    """, (class_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_attendance_for_student(student_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM attendance
        WHERE student_id=%s
        ORDER BY date DESC, time DESC
    """, (student_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def load_known_faces(school_id=None):
    global known_encodings, known_students
    known_encodings = []
    known_students = []

    if school_id is None:
        try:
            school_id = session.get("school_id", 1)
        except RuntimeError:
            school_id = 1  # No request context (startup)

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, student_id, full_name, image_file FROM students WHERE school_id=%s", (school_id,))
    rows = cur.fetchall()
    conn.close()

    for row in rows:
        image_file = row["image_file"]
        # Support both Supabase URLs and local paths
        if image_file.startswith("http"):
            try:
                resp = http_requests.get(image_file, timeout=10)
                if resp.status_code != 200:
                    continue
                nparr = np.frombuffer(resp.content, np.uint8)
                bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
            except Exception as e:
                print("Face load from URL error:", e)
                continue
        else:
            image_path = os.path.join(IMAGE_DIR, image_file)
            if not os.path.exists(image_path):
                continue
            bgr = cv2.imread(image_path)
            if bgr is None:
                continue
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            embedding = _get_face_embedding(rgb)
            if embedding is not None:
                known_encodings.append(embedding)
                known_students.append({
                    "db_id": row["id"],
                    "student_id": row["student_id"],
                    "full_name": row["full_name"],
                    "image_file": row["image_file"]
                })
        except Exception as e:
            print("Face load error:", e)


def student_belongs_to_class(student_db_id, class_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT 1 FROM student_classes
        WHERE student_id_fk=%s AND class_id_fk=%s
    """, (student_db_id, class_id))
    row = cur.fetchone()
    conn.close()
    return row is not None


def mark_attendance(student_row, class_row, status="Present", student_lat=None, student_lng=None, distance_meters=None, force=False):
    # Fetches local system date and time
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%I:%M:%S %p")
    school_id = get_current_school_id()

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT * FROM attendance
        WHERE student_id=%s AND class_id=%s AND date=%s AND school_id=%s
    """, (student_row["student_id"], class_row["id"], today, school_id))
    existing = cur.fetchone()

    teacher_name = class_row["teacher_display_name"] or class_row["teacher_name"] or ""

    # ── Block re-scan: if already marked Present today, don't let a student's own
    # re-scan silently overwrite it. A teacher-initiated correction (force=True)
    # is always allowed to override, e.g. to mark a student Absent after they
    # already self-marked Present. ──
    if existing and existing["status"] == "Present" and not force:
        conn.close()
        return {"already_marked": True, "status": "Present", "time": existing["time"]}

    if existing:
        cur.execute("""
            UPDATE attendance
            SET status=%s, time=%s, teacher_name=%s,
                student_lat=%s, student_lng=%s, distance_meters=%s
            WHERE id=%s
        """, (status, now_time, teacher_name, student_lat, student_lng, distance_meters, existing["id"]))
    else:
        cur.execute("""
            INSERT INTO attendance (
                school_id, student_id, full_name, class_id, class_name,
                department, course, section_name, subject_name,
                teacher_name, status, date, time,
                student_lat, student_lng, distance_meters
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            school_id,
            student_row["student_id"],
            student_row["full_name"],
            class_row["id"],
            class_row["class_name"],
            class_row["department"] or "",
            class_row["course"] or "",
            class_row["section_name"] or "",
            class_row["subject_name"] or "",
            teacher_name,
            status,
            today,
            now_time,
            student_lat,
            student_lng,
            distance_meters
        ))

    conn.commit()
    conn.close()
    return {"already_marked": False, "status": status}


def get_attendance_count_for_student_class(student_id, class_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) as c
        FROM attendance
        WHERE student_id=%s AND class_id=%s AND status='Present'
    """, (student_id, class_id))
    present_count = cur.fetchone()["c"]

    cur.execute("""
        SELECT COUNT(*) as c
        FROM attendance
        WHERE student_id=%s AND class_id=%s
    """, (student_id, class_id))
    total_count = cur.fetchone()["c"]
    conn.close()
    return present_count, total_count


def get_percentage(student_id, class_id):
    present_count, total_count = get_attendance_count_for_student_class(student_id, class_id)
    if total_count == 0:
        return 0
    return round((present_count / total_count) * 100, 2)


def get_report_records(period="daily", school_id=None):
    if school_id is None:
        school_id = get_current_school_id()
    conn = get_db()
    cur = conn.cursor()

    today = datetime.now().date()

    if period == "daily":
        start_date = today
    elif period == "weekly":
        start_date = today - timedelta(days=7)
    elif period == "monthly":
        start_date = today - timedelta(days=30)
    else:
        start_date = today

    cur.execute("""
        SELECT * FROM attendance
        WHERE date >= %s AND school_id=%s
        ORDER BY date DESC, time DESC
    """, (start_date.strftime("%Y-%m-%d"), school_id))
    rows = cur.fetchall()
    conn.close()
    return rows


# =========================================================
# SYSTEM PAGE TEMPLATE LAYOUT ENGINE
# =========================================================
def page_wrapper(title, body, is_admin=False, is_student=False, student_context=None, is_teacher=False, teacher_name=""):
    sidebar_html = ""
    
    if is_admin:
        sidebar_html = f"""
        <div class="sb-brand">
            <div class="sb-brand-icon">🛡️</div>
            <div>
                <div class="sb-brand-title">Admin Console</div>
                <div class="sb-brand-sub">System Administrator</div>
            </div>
        </div>
        <nav class="sb-nav">
            <div class="sb-nav-label">MAIN</div>
            <a href="/admin" class="sb-link"><span class="sb-icon">⬛</span> Dashboard</a>
            <a href="/admin/charts" class="sb-link"><span class="sb-icon">📈</span> Charts</a>
            <a href="/student-register" class="sb-link"><span class="sb-icon">➕</span> Register Student</a>
            <a href="/admin/reports" class="sb-link"><span class="sb-icon">📋</span> Attendance Reports</a>
            <div class="sb-nav-label" style="margin-top:16px;">ACCOUNT</div>
            <a href="/admin/location-settings" class="sb-link"><span class="sb-icon">📍</span> Location Settings</a>
            <a href="/settings" class="sb-link"><span class="sb-icon">⚙️</span> Settings</a>
            <a href="/" class="sb-link sb-link-muted"><span class="sb-icon">🏠</span> Main Site</a>
            <form method="POST" action="/admin-logout" style="margin-top:8px;">
                            {csrf_input_html()}
                <button type="submit" class="sb-logout-btn">🚪 Sign Out</button>
            </form>
        </nav>
        """
    elif is_teacher:
        sidebar_html = f"""
        <div class="sb-brand">
            <div class="sb-brand-icon" style="background:linear-gradient(135deg,#7c3aed,#4f46e5);">👨‍🏫</div>
            <div>
                <div class="sb-brand-title">Instructor Panel</div>
                <div class="sb-brand-sub">{teacher_name}</div>
            </div>
        </div>
        <nav class="sb-nav">
            <div class="sb-nav-label">MAIN</div>
            <a href="/teacher" class="sb-link"><span class="sb-icon">📋</span> My Classes</a>
            <a href="/teacher/inbox" class="sb-link"><span class="sb-icon">💬</span> Student Inbox</a>
            <div class="sb-nav-label" style="margin-top:16px;">ACCOUNT</div>
            <a href="/teacher/edit-profile" class="sb-link"><span class="sb-icon">✏️</span> Edit Profile</a>
            <a href="/settings" class="sb-link"><span class="sb-icon">⚙️</span> Update Password</a>
            <a href="/" class="sb-link sb-link-muted"><span class="sb-icon">🏠</span> Main Site</a>
            <form method="POST" action="/teacher-logout" style="margin-top:8px;">
                            {csrf_input_html()}
                <button type="submit" class="sb-logout-btn">🚪 Sign Out</button>
            </form>
        </nav>
        """
    elif is_student and student_context:
        sidebar_html = f"""
        <div class="sb-student-profile">
            <img src="{supabase_public_url(student_context['image_file'])}" class="sb-avatar">
            <div class="sb-brand-title" style="margin-top:10px;">{student_context['full_name']}</div>
            <div class="sb-brand-sub" style="font-family:monospace;">ID: {student_context['student_id']}</div>
        </div>
        <nav class="sb-nav">
            <div class="sb-nav-label">STUDENT</div>
            <a href="/student" class="sb-link"><span class="sb-icon">📚</span> My Profile</a>
            <a href="/student/classes" class="sb-link"><span class="sb-icon">👥</span> My Classes</a>
            <a href="/student/qr-scan" class="sb-link sb-link-checkin"><span class="sb-icon">📷</span> Scan QR to Check In</a>
            <a href="/student/scan" class="sb-link"><span class="sb-icon">📸</span> Face Check-In</a>
            <a href="/student/edit-profile" class="sb-link"><span class="sb-icon">✏️</span> Edit Profile</a>
            <div class="sb-nav-label" style="margin-top:16px;">ACCOUNT</div>
            <a href="/" class="sb-link sb-link-muted"><span class="sb-icon">🏠</span> Main Site</a>
            <form method="POST" action="/student-logout" style="margin-top:8px;">
                            {csrf_input_html()}
                <button type="submit" class="sb-logout-btn">🚪 Sign Out</button>
            </form>
            <form method="POST" action="/student/delete-account" style="margin-top:6px;" onsubmit="return confirm('Permanently delete your account? This cannot be undone.')">
                            {csrf_input_html()}
                <button type="submit" class="sb-delete-btn">🗑️ Delete Account</button>
            </form>
        </nav>
        """

    if is_admin or is_student or is_teacher:
        content_html = f"""
        <div class="dw">
            <div class="sb" id="sidebarMenu">
                <div class="sb-inner">
                    {sidebar_html}
                </div>
            </div>
            <div class="sb-overlay" id="sbOverlay" onclick="closeSidebar()"></div>
            <div class="main-content">
                <div class="topbar">
                    <button class="topbar-menu-btn" onclick="toggleSidebar(event)" aria-label="Open menu">
                        <span></span><span></span><span></span>
                    </button>
                    <div class="topbar-title">{title}</div>
                    <div class="topbar-right">
                        <button class="dm-toggle" id="dmBtn" onclick="toggleDark()" title="Toggle dark mode">🌙</button>
                    </div>
                </div>
                <div class="content-area">
                    {body}
                </div>
            </div>
        </div>
        """
    else:
        content_html = f"""
        <div class="public-wrap">
            <button class="dm-toggle dm-float" id="dmBtn" onclick="toggleDark()" title="Toggle dark mode">🌙</button>
            {body}
        </div>
        """

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
        <title>{title} — EduTrack</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
        :root {{
            --c-bg: #f0f4ff;
            --c-surface: #ffffff;
            --c-border: #e2e8f0;
            --c-text: #0f172a;
            --c-muted: #64748b;
            --c-accent: #2563eb;
            --c-accent-dark: #1d4ed8;
            --c-accent-light: #eff6ff;
            --c-success: #059669;
            --c-warning: #d97706;
            --c-danger: #dc2626;
            --c-purple: #7c3aed;
            --sb-w: 272px;
            --topbar-h: 60px;
            --radius: 12px;
            --shadow-sm: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
            --shadow-md: 0 4px 16px rgba(0,0,0,0.08), 0 2px 6px rgba(0,0,0,0.04);
            --shadow-lg: 0 12px 32px rgba(0,0,0,0.12);
        }}
        html, body {{ height: 100%; }}
        body {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background: var(--c-bg);
            color: var(--c-text);
            font-size: 15px;
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }}

        /* ── DASHBOARD LAYOUT ── */
        .dw {{ display: flex; min-height: 100vh; }}

        /* ── SIDEBAR ── */
        .sb {{
            width: var(--sb-w);
            background: #0f172a;
            color: #f1f5f9;
            position: fixed;
            inset: 0 auto 0 0;
            z-index: 200;
            display: flex;
            flex-direction: column;
            transition: transform 0.28s cubic-bezier(0.4,0,0.2,1);
            box-shadow: 4px 0 24px rgba(0,0,0,0.18);
            overflow-y: auto;
            overflow-x: hidden;
        }}
        .sb-inner {{ padding: 0 0 32px; min-height: 100%; display: flex; flex-direction: column; }}

        .sb-brand {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 22px 20px 20px;
            border-bottom: 1px solid rgba(255,255,255,0.07);
        }}
        .sb-brand-icon {{
            width: 42px; height: 42px;
            border-radius: 10px;
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            display: flex; align-items: center; justify-content: center;
            font-size: 20px;
            flex-shrink: 0;
            box-shadow: 0 4px 12px rgba(37,99,235,0.35);
        }}
        .sb-brand-title {{ font-weight: 700; font-size: 15px; color: #f8fafc; letter-spacing: -0.01em; }}
        .sb-brand-sub {{ font-size: 12px; color: #94a3b8; margin-top: 1px; }}

        .sb-student-profile {{
            padding: 24px 20px 20px;
            border-bottom: 1px solid rgba(255,255,255,0.07);
            text-align: center;
        }}
        .sb-avatar {{
            width: 76px; height: 76px;
            border-radius: 50%;
            object-fit: cover;
            border: 3px solid #2563eb;
            box-shadow: 0 0 0 3px rgba(37,99,235,0.25);
            margin: 0 auto 10px;
            display: block;
        }}

        .sb-nav {{ padding: 16px 12px; flex: 1; }}
        .sb-nav-label {{
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.1em;
            color: #475569;
            padding: 0 12px;
            margin-bottom: 6px;
            text-transform: uppercase;
        }}
        .sb-link {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 12px;
            border-radius: 9px;
            text-decoration: none;
            color: #94a3b8;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.18s ease;
            margin-bottom: 2px;
        }}
        .sb-link:hover {{ background: rgba(255,255,255,0.07); color: #f1f5f9; }}
        .sb-link.active, .sb-link:focus {{ background: #2563eb; color: white; box-shadow: 0 2px 8px rgba(37,99,235,0.4); }}
        .sb-link-muted {{ color: #475569 !important; }}
        .sb-link-muted:hover {{ color: #94a3b8 !important; }}
        .sb-link-checkin {{ background: rgba(5,150,105,0.15) !important; color: #34d399 !important; border: 1px solid rgba(5,150,105,0.25); }}
        .sb-link-checkin:hover {{ background: #059669 !important; color: white !important; }}
        .sb-icon {{ font-size: 16px; width: 22px; text-align: center; flex-shrink: 0; }}
        .sb-logout-btn {{
            width: 100%;
            background: rgba(220,38,38,0.12);
            color: #fca5a5;
            border: 1px solid rgba(220,38,38,0.2);
            padding: 10px 12px;
            border-radius: 9px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            text-align: left;
            transition: all 0.18s;
        }}
        .sb-logout-btn:hover {{ background: #dc2626; color: white; border-color: #dc2626; }}
        .sb-delete-btn {{
            width: 100%;
            background: rgba(127,29,29,0.15);
            color: #f87171;
            border: 1px solid rgba(127,29,29,0.25);
            padding: 10px 12px;
            border-radius: 9px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            text-align: left;
            transition: all 0.18s;
        }}
        .sb-delete-btn:hover {{ background: #7f1d1d; color: white; }}

        /* ── OVERLAY ── */
        .sb-overlay {{
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.45);
            z-index: 150;
            backdrop-filter: blur(2px);
        }}

        /* ── MAIN CONTENT ── */
        .main-content {{
            flex: 1;
            margin-left: var(--sb-w);
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }}

        /* ── TOPBAR ── */
        .topbar {{
            height: var(--topbar-h);
            background: var(--c-surface);
            border-bottom: 1px solid var(--c-border);
            display: flex;
            align-items: center;
            padding: 0 24px;
            gap: 16px;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: var(--shadow-sm);
        }}
        .topbar-menu-btn {{
            display: none;
            flex-direction: column;
            gap: 5px;
            background: none;
            border: none;
            cursor: pointer;
            padding: 6px;
            border-radius: 8px;
            flex-shrink: 0;
        }}
        .topbar-menu-btn:hover {{ background: #f1f5f9; }}
        .topbar-menu-btn span {{
            display: block;
            width: 20px;
            height: 2px;
            background: #475569;
            border-radius: 2px;
            transition: all 0.2s;
        }}
        .topbar-title {{ font-weight: 600; font-size: 16px; color: var(--c-text); flex: 1; }}
        .topbar-right {{ display: flex; align-items: center; gap: 10px; }}

        /* ── CONTENT AREA ── */
        .content-area {{ padding: 28px 28px; flex: 1; }}

        /* ── PUBLIC PAGES ── */
        .public-wrap {{ max-width: 1100px; margin: 40px auto; padding: 0 20px; }}

        /* ── CARDS / BOX ── */
        .card {{
            background: var(--c-surface);
            border-radius: var(--radius);
            border: 1px solid var(--c-border);
            box-shadow: var(--shadow-sm);
        }}
        .card-header {{
            padding: 18px 22px;
            border-bottom: 1px solid var(--c-border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .card-header-title {{ font-weight: 700; font-size: 15px; color: var(--c-text); }}
        .card-body {{ padding: 22px; }}

        /* ── TABLES ── */
        .tbl-wrap {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        thead th {{
            background: #f8fafc;
            color: #475569;
            padding: 12px 14px;
            text-align: left;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            white-space: nowrap;
            border-bottom: 2px solid var(--c-border);
        }}
        tbody td {{
            padding: 12px 14px;
            border-bottom: 1px solid #f1f5f9;
            color: var(--c-text);
            vertical-align: middle;
        }}
        tbody tr:last-child td {{ border-bottom: none; }}
        tbody tr:hover td {{ background: #fafbff; }}

        /* ── FORMS ── */
        .form-group {{ margin-bottom: 16px; }}
        .form-label {{ display: block; font-size: 13px; font-weight: 600; color: #374151; margin-bottom: 6px; }}
        .form-input {{
            width: 100%;
            padding: 10px 14px;
            border: 1.5px solid var(--c-border);
            border-radius: 9px;
            font-size: 14px;
            font-family: inherit;
            color: var(--c-text);
            background: var(--c-surface);
            transition: border-color 0.18s, box-shadow 0.18s;
            outline: none;
            max-width: 100%;
        }}
        .form-input:focus {{ border-color: var(--c-accent); box-shadow: 0 0 0 3px rgba(37,99,235,0.12); }}
        select.form-input {{ cursor: pointer; }}
        input:not([class]), select:not([class]), textarea:not([class]) {{
            width: 100%;
            padding: 10px 14px;
            border: 1.5px solid var(--c-border);
            border-radius: 9px;
            font-size: 14px;
            font-family: inherit;
            color: var(--c-text);
            background: var(--c-surface);
            transition: border-color 0.18s, box-shadow 0.18s;
            outline: none;
            max-width: 100%;
            margin: 4px 0;
            box-sizing: border-box;
        }}
        input:not([class]):focus, select:not([class]):focus, textarea:not([class]):focus {{
            border-color: var(--c-accent);
            box-shadow: 0 0 0 3px rgba(37,99,235,0.12);
        }}

        /* ── BUTTONS ── */
        .btn, button:not(.topbar-menu-btn):not(.sb-logout-btn):not(.sb-delete-btn):not([style*="none"]) {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            padding: 10px 18px;
            border-radius: 9px;
            font-size: 14px;
            font-weight: 600;
            font-family: inherit;
            cursor: pointer;
            border: none;
            transition: all 0.18s ease;
            text-decoration: none;
            white-space: nowrap;
        }}
        .btn-primary, button[type="submit"]:not(.sb-logout-btn):not(.sb-delete-btn) {{
            background: var(--c-accent);
            color: white;
            box-shadow: 0 2px 6px rgba(37,99,235,0.3);
        }}
        .btn-primary:hover, button[type="submit"]:not(.sb-logout-btn):not(.sb-delete-btn):hover {{
            background: var(--c-accent-dark);
            box-shadow: 0 4px 12px rgba(37,99,235,0.4);
            transform: translateY(-1px);
        }}
        .btn.green {{ background: var(--c-success); color: white; box-shadow: 0 2px 6px rgba(5,150,105,0.3); }}
        .btn.green:hover {{ background: #047857; transform: translateY(-1px); }}
        .btn.orange {{ background: #f97316; color: white; }}
        .btn.red {{ background: var(--c-danger); color: white; }}
        .btn.dark {{ background: #1e293b; color: white; }}

        /* ── STAT CARDS ── */
        .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .stat-card {{
            background: var(--c-surface);
            border-radius: var(--radius);
            border: 1px solid var(--c-border);
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 6px;
            box-shadow: var(--shadow-sm);
            transition: box-shadow 0.2s, transform 0.2s;
        }}
        .stat-card:hover {{ box-shadow: var(--shadow-md); transform: translateY(-2px); }}
        .stat-icon {{ font-size: 26px; }}
        .stat-value {{ font-size: 32px; font-weight: 800; letter-spacing: -0.03em; color: var(--c-text); }}
        .stat-label {{ font-size: 13px; color: var(--c-muted); font-weight: 500; }}
        .stat-card.blue .stat-value {{ color: #2563eb; }}
        .stat-card.green .stat-value {{ color: #059669; }}
        .stat-card.purple .stat-value {{ color: #7c3aed; }}
        .stat-card.orange .stat-value {{ color: #d97706; }}

        /* ── BADGES ── */
        .badge {{
            display: inline-flex;
            align-items: center;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }}
        .badge-green {{ background: #d1fae5; color: #065f46; }}
        .badge-red {{ background: #fee2e2; color: #991b1b; }}
        .badge-blue {{ background: #dbeafe; color: #1e40af; }}
        .badge-gray {{ background: #f1f5f9; color: #475569; }}

        /* ── PAGE HEADER ── */
        .page-header {{ margin-bottom: 24px; }}
        .page-title {{ font-size: 26px; font-weight: 800; letter-spacing: -0.02em; color: var(--c-text); }}
        .page-sub {{ font-size: 14px; color: var(--c-muted); margin-top: 4px; }}

        /* ── SECTION SPACING ── */
        .section-stack {{ display: flex; flex-direction: column; gap: 24px; }}

        /* ── ALERT / FEEDBACK ── */
        .alert {{ padding: 12px 16px; border-radius: 9px; font-size: 14px; font-weight: 500; margin-bottom: 16px; }}
        .alert-error {{ background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; }}
        .alert-success {{ background: #f0fdf4; border: 1px solid #bbf7d0; color: #14532d; }}

        /* ── DARK MODE ── */
        html.dark {{
            --c-bg: #0d1117;
            --c-surface: #161b22;
            --c-border: #30363d;
            --c-text: #e6edf3;
            --c-muted: #8b949e;
            --c-accent: #4493f8;
            --c-accent-dark: #2f81f7;
            --c-accent-light: #1c2a3a;
            --c-success: #3fb950;
            --c-warning: #e3b341;
            --c-danger: #f85149;
        }}
        html.dark body {{ background: var(--c-bg); color: var(--c-text); }}
        html.dark .topbar {{ background: #161b22; border-color: #30363d; }}
        html.dark .topbar-menu-btn span {{ background: #8b949e; }}
        html.dark .topbar-menu-btn:hover {{ background: #21262d; }}
        html.dark .card {{ background: var(--c-surface); border-color: var(--c-border); }}
        html.dark .card-header {{ border-color: var(--c-border); }}
        html.dark thead th {{ background: #21262d; color: #8b949e; border-color: #30363d; }}
        html.dark tbody td {{ border-color: #21262d; color: var(--c-text); }}
        html.dark tbody tr:hover td {{ background: #1c2128; }}
        html.dark input:not([class]), html.dark select:not([class]), html.dark textarea:not([class]),
        html.dark .form-input {{
            background: #21262d; border-color: #30363d; color: var(--c-text);
        }}
        html.dark .stat-card {{ background: #161b22; border-color: #30363d; }}
        html.dark .stat-card:hover {{ box-shadow: 0 4px 20px rgba(0,0,0,0.4); }}
        html.dark .badge-green {{ background: #0d2818; color: #3fb950; }}
        html.dark .badge-red {{ background: #2d1216; color: #f85149; }}
        html.dark .badge-blue {{ background: #1c2a3a; color: #79c0ff; }}
        html.dark .alert-error {{ background: #2d1216; border-color: #f85149; color: #f85149; }}
        html.dark .alert-success {{ background: #0d2818; border-color: #3fb950; color: #3fb950; }}
        html.dark .public-wrap .feat-item {{ background: #161b22; border-color: #30363d; }}
        html.dark .public-wrap .feat-title {{ color: var(--c-text); }}
        html.dark .public-wrap .home-title {{ color: var(--c-text); }}
        html.dark .public-wrap .home-sub {{ color: var(--c-muted); }}
        html.dark .public-wrap {{ background: #0d1117; }}
        html.dark .page-title {{ color: var(--c-text); }}

        /* ── DARK MODE TOGGLE BUTTON ── */
        .dm-toggle {{
            display: flex; align-items: center; justify-content: center;
            width: 36px; height: 36px;
            border-radius: 8px;
            border: 1.5px solid var(--c-border);
            background: var(--c-surface);
            cursor: pointer;
            font-size: 17px;
            transition: all 0.18s;
            flex-shrink: 0;
            line-height: 1;
        }}
        .dm-toggle:hover {{ background: var(--c-accent-light); border-color: var(--c-accent); }}
        .dm-float {{
            position: fixed;
            top: 16px;
            right: 16px;
            z-index: 999;
            box-shadow: var(--shadow-md);
            width: 42px; height: 42px;
            border-radius: 10px;
            font-size: 19px;
        }}

        /* ── CHART CONTAINER ── */
        .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .chart-card {{ background: var(--c-surface); border: 1px solid var(--c-border); border-radius: var(--radius); padding: 20px; box-shadow: var(--shadow-sm); }}
        .chart-title {{ font-size: 14px; font-weight: 700; color: var(--c-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 16px; }}
        canvas {{ max-width: 100%; }}

        /* ── MOBILE RESPONSIVE ── */
        @media (max-width: 1023px) {{
            .sb {{ transform: translateX(-100%); }}
            .sb.open {{ transform: translateX(0); }}
            .sb-overlay.open {{ display: block; }}
            .main-content {{ margin-left: 0; }}
            .topbar-menu-btn {{ display: flex; }}
            .content-area {{ padding: 20px 16px; }}
            .stat-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .topbar {{ padding: 0 16px; }}
        }}
        @media (max-width: 480px) {{
            .stat-grid {{ grid-template-columns: repeat(2, 1fr); gap: 12px; }}
            .stat-value {{ font-size: 26px; }}
            .content-area {{ padding: 16px 12px; }}
            .chart-grid {{ grid-template-columns: 1fr; }}
        }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
        <script>
            // ── DARK MODE ──
            (function() {{
                const saved = localStorage.getItem('dm');
                if (saved === '1' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)) {{
                    document.documentElement.classList.add('dark');
                }}
            }})();
            function toggleDark() {{
                const isDark = document.documentElement.classList.toggle('dark');
                localStorage.setItem('dm', isDark ? '1' : '0');
                const btn = document.getElementById('dmBtn');
                if (btn) btn.textContent = isDark ? '☀️' : '🌙';
                // Refresh charts if present
                if (window._charts) window._charts.forEach(c => {{ c.options.plugins.legend.labels.color = isDark ? '#8b949e' : '#475569'; c.update(); }});
            }}
            // ── SIDEBAR ──
            function toggleSidebar(e) {{
                if(e) e.stopPropagation();
                document.getElementById('sidebarMenu').classList.toggle('open');
                document.getElementById('sbOverlay').classList.toggle('open');
            }}
            function closeSidebar() {{
                document.getElementById('sidebarMenu').classList.remove('open');
                document.getElementById('sbOverlay').classList.remove('open');
            }}
            document.addEventListener('DOMContentLoaded', function() {{
                // Active nav link
                const path = window.location.pathname;
                document.querySelectorAll('.sb-link').forEach(a => {{
                    if(a.getAttribute('href') === path) a.classList.add('active');
                }});
                // Set dark toggle icon
                const btn = document.getElementById('dmBtn');
                if (btn) btn.textContent = document.documentElement.classList.contains('dark') ? '☀️' : '🌙';
            }});
        </script>
        <meta name="theme-color" content="#0f172a">
        <style>
        #_toast {{
            position:fixed;bottom:24px;right:24px;z-index:9999;
            padding:13px 22px;border-radius:12px;font-size:14px;font-weight:600;
            box-shadow:0 4px 24px rgba(0,0,0,0.18);transition:opacity 0.4s,transform 0.3s;
            pointer-events:none;transform:translateY(0);
        }}
        #_toast.hidden {{ opacity:0; transform:translateY(12px); }}
        </style>
        <script>
        // ── GLOBAL CSRF PROTECTION FOR AJAX ──
        // Every fetch() call the app makes automatically carries the
        // CSRF token (read from the readable csrf_token cookie) as a
        // header, so no individual fetch() call site needs to be
        // touched by hand.
        (function() {{
            function getCookie(name) {{
                const m = document.cookie.match('(^|;)\\\\s*' + name + '\\\\s*=\\\\s*([^;]+)');
                return m ? decodeURIComponent(m.pop()) : '';
            }}
            const _origFetch = window.fetch;
            window.fetch = function(input, init) {{
                init = init || {{}};
                const method = (init.method || 'GET').toUpperCase();
                if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(method)) {{
                    init.headers = Object.assign({{}}, init.headers || {{}});
                    if (!init.headers['X-CSRF-Token'] && !init.headers['x-csrf-token']) {{
                        init.headers['X-CSRF-Token'] = getCookie('csrf_token');
                    }}
                }}
                return _origFetch(input, init);
            }};
        }})();
        // ── AJAX FORM INTERCEPTOR ──
        document.addEventListener('DOMContentLoaded', function() {{
            document.querySelectorAll('form[data-ajax]').forEach(function(form) {{
                form.addEventListener('submit', function(e) {{
                    e.preventDefault();
                    const btn = form.querySelector('[type=submit]');
                    const origLabel = btn ? btn.textContent : '';
                    if (btn) {{ btn.disabled = true; btn.textContent = 'Saving…'; }}
                    fetch(form.action || window.location.href, {{
                        method: (form.method || 'POST').toUpperCase(),
                        body: new FormData(form),
                        headers: {{ 'X-Requested-With': 'XMLHttpRequest' }}
                    }})
                    .then(r => r.json())
                    .then(data => {{
                        if (btn) {{ btn.disabled = false; btn.textContent = origLabel; }}
                        if (data.ok) {{
                            showToast(data.message || 'Saved!', 'success');
                            if (data.redirect) {{
                                setTimeout(() => window.location.href = data.redirect, 900);
                            }} else {{
                                setTimeout(() => window.location.reload(), 900);
                            }}
                        }} else {{
                            showToast(data.error || 'Something went wrong.', 'error');
                        }}
                    }})
                    .catch(() => {{
                        if (btn) {{ btn.disabled = false; btn.textContent = origLabel; }}
                        showToast('Network error — please try again.', 'error');
                    }});
                }});
            }});
        }});

        function showToast(msg, type) {{
            let t = document.getElementById('_toast');
            if (!t) {{
                t = document.createElement('div');
                t.id = '_toast';
                document.body.appendChild(t);
            }}
            t.textContent = msg;
            t.style.background = type === 'success' ? '#059669' : '#dc2626';
            t.style.color = 'white';
            t.classList.remove('hidden');
            t.style.opacity = '1';
            t.style.transform = 'translateY(0)';
            clearTimeout(t._timer);
            t._timer = setTimeout(() => {{
                t.style.opacity = '0';
                t.style.transform = 'translateY(12px)';
            }}, 3200);
        }}
        </script>
    </head>
    <body>
        {content_html}
    </body>
    </html>
    """


# =========================================================
# UNIVERSAL CONFIGURATION & SETTINGS
# =========================================================
@app.route("/settings", methods=["GET", "POST"])
def user_settings():
    role = None
    user_id = None
    display_name = ""
    student_ctx = None
    teacher_name_str = ""
    
    if is_admin_logged_in():
        role = "admin"
        display_name = "Administrator"
    elif is_teacher_logged_in():
        role = "teacher"
        user_id = get_logged_teacher_id()
        teacher_name_str = session.get("teacher_name", "Teacher")
        display_name = teacher_name_str
    elif is_student_logged_in():
        role = "student"
        user_id = get_logged_student_db_id()
        student_ctx = get_student_row_by_db_id(user_id)
        display_name = session.get("student_name", "Student")
    else:
        return redirect("/")

    if request.method == "POST":
        old_password = request.form.get("old_password", "").strip()
        new_password = request.form.get("new_password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if not old_password or not new_password or not confirm_password:
            return page_wrapper("Settings", "<p class='text-red-500 font-bold'>All fields are required.</p>", is_student=(role == "student"), student_context=student_ctx, is_teacher=(role == "teacher"), teacher_name=teacher_name_str)
        
        if new_password != confirm_password:
            return page_wrapper("Settings", "<p class='text-red-500 font-bold'>New entries do not match.</p>", is_student=(role == "student"), student_context=student_ctx, is_teacher=(role == "teacher"), teacher_name=teacher_name_str)

        conn = get_db()
        cur = conn.cursor()

        if role == "admin":
            if not verify_password(get_admin_password(), old_password):
                conn.close()
                if is_ajax(): return ajax_err("Incorrect existing password.")
                return page_wrapper("Settings", "<p class='text-red-500 font-bold'>Incorrect existing password.</p>")
            set_admin_password(new_password)
            conn.close()
            if is_ajax(): return ajax_ok("Admin password updated successfully!", redirect_url="/admin")
            return "<script>alert('Admin password updated successfully!'); window.location.href='/admin';</script>"

        elif role == "teacher":
            cur.execute("SELECT password FROM teachers WHERE id=%s", (user_id,))
            row = cur.fetchone()
            if not row or not verify_password(row["password"], old_password):
                conn.close()
                if is_ajax(): return ajax_err("Incorrect old password.")
                return page_wrapper("Settings", "<p class='text-red-500 font-bold'>Incorrect old password.</p>", is_teacher=True, teacher_name=teacher_name_str)
            cur.execute("UPDATE teachers SET password=%s WHERE id=%s", (hash_password(new_password), user_id))
            conn.commit()
            conn.close()
            if is_ajax(): return ajax_ok("Teacher password updated successfully!", redirect_url="/teacher")
            return "<script>alert('Teacher password updated successfully!'); window.location.href='/teacher';</script>"

        elif role == "student":
            cur.execute("SELECT password FROM students WHERE id=%s", (user_id,))
            row = cur.fetchone()
            if not row or not verify_password(row["password"], old_password):
                conn.close()
                if is_ajax(): return ajax_err("Incorrect old password.")
                return page_wrapper("Settings", "<p class='text-red-500 font-bold'>Incorrect old password.</p>", is_student=True, student_context=student_ctx)
            cur.execute("UPDATE students SET password=%s WHERE id=%s", (hash_password(new_password), user_id))
            conn.commit()
            conn.close()
            if is_ajax(): return ajax_ok("Password updated successfully!", redirect_url="/student")
            return "<script>alert('Student password updated successfully!'); window.location.href='/student';</script>"

    body = f"""
    <div class="max-w-xl">
        <h1 class="text-2xl font-bold text-slate-800 mb-2">Account Portal Settings</h1>
        <p class="text-sm text-slate-500 mb-6">Change local password credentials below for {display_name}</p>
        <form method="POST" action="/settings" class="space-y-4" data-ajax>
                            {csrf_input_html()}
            <div>
                <label class="block text-sm font-medium text-slate-700">Current Password</label>
                <input type="password" name="old_password" class="mt-1 block w-full px-3 py-2 border rounded-lg" required>
            </div>
            <div>
                <label class="block text-sm font-medium text-slate-700">New Password</label>
                <input type="password" name="new_password" class="mt-1 block w-full px-3 py-2 border rounded-lg" required>
            </div>
            <div>
                <label class="block text-sm font-medium text-slate-700">Confirm New Password</label>
                <input type="password" name="confirm_password" class="mt-1 block w-full px-3 py-2 border rounded-lg" required>
            </div>
            <div class="pt-2">
                <button type="submit" class="bg-blue-600 text-white font-semibold px-4 py-2 rounded-lg hover:bg-blue-700">Save Configuration</button>
                <a href="/{role}" class="inline-block ml-2 px-4 py-2 bg-slate-100 text-slate-700 font-semibold rounded-lg hover:bg-slate-200">Cancel</a>
            </div>
        </form>
    </div>
    """
    return page_wrapper("Account Settings", body, is_admin=(role == "admin"), is_student=(role == "student"), student_context=student_ctx, is_teacher=(role == "teacher"), teacher_name=teacher_name_str)


# =========================================================
# SYSTEM INDEX LANDING
# =========================================================
@app.route("/")
def home():
    return page_wrapper("EduTrack", """
    <style>
    .home-hero { text-align:center; padding:56px 20px 40px; max-width:680px; margin:0 auto; }
    .home-badge { display:inline-flex; align-items:center; gap:6px; background:#eff6ff; border:1px solid #bfdbfe; color:#1d4ed8; font-size:13px; font-weight:600; padding:6px 14px; border-radius:20px; margin-bottom:22px; }
    .home-title { font-size:clamp(32px,6vw,52px); font-weight:800; letter-spacing:-0.03em; color:#0f172a; line-height:1.15; margin-bottom:16px; }
    .home-title span { color:#2563eb; }
    .home-sub { font-size:17px; color:#475569; line-height:1.7; margin-bottom:40px; max-width:520px; margin-left:auto; margin-right:auto; }
    .portal-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:14px; max-width:560px; margin:0 auto 52px; }
    @media(max-width:480px){.portal-grid{grid-template-columns:1fr;}}
    .portal-card { display:flex; flex-direction:column; align-items:flex-start; gap:8px; padding:22px 20px; border-radius:14px; text-decoration:none; transition:all 0.2s ease; border:none; }
    .portal-card:hover { transform:translateY(-3px); box-shadow:0 12px 28px rgba(0,0,0,0.15); }
    .portal-card-icon { font-size:28px; }
    .portal-card-label { font-size:16px; font-weight:700; color:white; }
    .portal-card-desc { font-size:12px; opacity:0.82; color:white; }
    .pc-orange { background:linear-gradient(135deg,#f97316,#ea580c); }
    .pc-dark { background:linear-gradient(135deg,#1e293b,#0f172a); }
    .pc-purple { background:linear-gradient(135deg,#7c3aed,#6d28d9); }
    .pc-green { background:linear-gradient(135deg,#059669,#047857); }
    .pc-blue { background:linear-gradient(135deg,#2563eb,#1d4ed8); }
    .features-row { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; max-width:720px; margin:0 auto; }
    @media(max-width:600px){.features-row{grid-template-columns:1fr;}}
    .feat-item { background:white; border:1px solid #e2e8f0; border-radius:12px; padding:18px; text-align:left; }
    .feat-icon { font-size:22px; margin-bottom:8px; }
    .feat-title { font-size:14px; font-weight:700; color:#0f172a; margin-bottom:4px; }
    .feat-text { font-size:13px; color:#64748b; line-height:1.5; }
    </style>
    <div class="home-hero">
        <div class="home-badge">✨ AI-Powered Attendance System</div>
        <h1 class="home-title">Smart Attendance<br><span>Made Simple</span></h1>
        <p class="home-sub">Face recognition check-in, real-time reporting, and multi-role management — all in one place.</p>
        <div class="portal-grid">
            <a href="/student-register" class="portal-card pc-orange">
                <div class="portal-card-icon">🧑‍🎓</div>
                <div class="portal-card-label">Register Face</div>
                <div class="portal-card-desc">New student enrollment</div>
            </a>
            <a href="/admin-login" class="portal-card pc-dark">
                <div class="portal-card-icon">🔐</div>
                <div class="portal-card-label">Admin Login</div>
                <div class="portal-card-desc">System administration</div>
            </a>
            <a href="/teacher-login" class="portal-card pc-purple">
                <div class="portal-card-icon">👨‍🏫</div>
                <div class="portal-card-label">Teacher Login</div>
                <div class="portal-card-desc">Manage your classes</div>
            </a>
            <a href="/student-login" class="portal-card pc-green">
                <div class="portal-card-icon">📚</div>
                <div class="portal-card-label">Student Login</div>
                <div class="portal-card-desc">View your attendance</div>
            </a>
            <a href="/register-school" class="portal-card pc-blue" style="grid-column:1 / -1;">
                <div class="portal-card-icon">🏫</div>
                <div class="portal-card-label">Register Your School</div>
                <div class="portal-card-desc">Get started in minutes</div>
            </a>
        </div>
        <div class="features-row">
            <div class="feat-item"><div class="feat-icon">📸</div><div class="feat-title">Face Recognition</div><div class="feat-text">AI-powered check-in via live camera using MediaPipe</div></div>
            <div class="feat-item"><div class="feat-icon">📊</div><div class="feat-title">Live Reports</div><div class="feat-text">Real-time dashboards with CSV export</div></div>
            <div class="feat-item"><div class="feat-icon">🔒</div><div class="feat-title">Secure Roles</div><div class="feat-text">Isolated admin, teacher, and student sessions</div></div>
        </div>
    </div>
    """)


# =========================================================
# PORTAL AUTHENTICATION ROUTING
# =========================================================
def login_page(title, action, user_placeholder, pass_placeholder, error_message="", extra_fields=""):
    return page_wrapper(title, f"""
        <div class="max-w-md mx-auto my-8 p-6 bg-white border border-slate-200 rounded-xl shadow-sm">
            <h2 class="text-2xl font-bold text-slate-800 text-center mb-6">{title}</h2>
            <form method="POST" action="{action}" class="space-y-4">
                            {csrf_input_html()}
                <div>
                    <input type="text" name="username" placeholder="{user_placeholder}" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500" required>
                </div>
                <div>
                    <input type="password" name="password" placeholder="{pass_placeholder}" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500" required>
                </div>
                {extra_fields}
                <button class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition" type="submit">Sign In</button>
            </form>
            {f'<div class="mt-3 text-sm text-red-500 text-center font-semibold">{error_message}</div>' if error_message else ''}
            <div class="mt-4 border-t pt-4 text-center">
                <a class="text-sm text-blue-600 hover:underline" href="/">← Back to Landing Home</a>
            </div>
        </div>
    """)


ADMIN_LOGIN_EXTRA_FIELDS = '''
    <input type="text" name="school_code" placeholder="School Code (e.g. ABC123)" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500" required>
'''

@app.route("/admin-login", methods=["GET", "POST"])
def admin_login():
    if request.method == "GET":
        return login_page("Admin Login", "/admin-login", "School Name", "Admin Password",
                          extra_fields=ADMIN_LOGIN_EXTRA_FIELDS)

    if login_rate_limited("admin-login"):
        return too_many_attempts_response()

    school_name = request.form.get("username", "").strip()
    school_code = request.form.get("school_code", "").strip().upper()
    password = request.form.get("password", "").strip()

    school = get_school_by_code(school_code)
    if not school:
        return login_page("Admin Login", "/admin-login", "School Name", "Admin Password",
                          "School code not found.",
                          extra_fields=ADMIN_LOGIN_EXTRA_FIELDS)

    if school_name.strip().lower() != school["name"].strip().lower():
        return login_page("Admin Login", "/admin-login", "School Name", "Admin Password",
                          "School name does not match this school code.",
                          extra_fields=ADMIN_LOGIN_EXTRA_FIELDS)

    # Check school-specific admin password
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT value FROM admin_settings WHERE school_id=%s AND key='admin_password'", (school["id"],))
    row = cur.fetchone()
    conn.close()
    stored_pw = row["value"] if row else school["admin_password"]

    if not verify_password(stored_pw, password):
        return login_page("Admin Login", "/admin-login", "School Name", "Admin Password",
                          "Invalid password.",
                          extra_fields=ADMIN_LOGIN_EXTRA_FIELDS)

    if not is_hashed_password(stored_pw):
        set_admin_password(password, school_id=school["id"])

    session.clear()
    session["admin_logged_in"] = True
    session["school_id"] = school["id"]
    session["school_name"] = school["name"]
    return redirect("/admin")


@app.route("/teacher-login", methods=["GET", "POST"])
def teacher_login():
    school_code_field = '<input type="text" name="school_code" placeholder="School Code (e.g. ABC123)" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500" required>'
    if request.method == "GET":
        kicked_msg = "\u26a0\ufe0f Your account was logged in on another device. Please log in again." if request.args.get("kicked") else ""
        return login_page("Teacher Login", "/teacher-login", "Teacher Username", "Teacher Password",
                          error_message=kicked_msg, extra_fields=school_code_field)

    if login_rate_limited("teacher-login"):
        return too_many_attempts_response()

    school_code = request.form.get("school_code", "").strip().upper()
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    school = get_school_by_code(school_code)
    if not school:
        return login_page("Teacher Login", "/teacher-login", "Teacher Username", "Teacher Password",
                          "School code not found.", extra_fields=school_code_field)

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM teachers WHERE username=%s AND school_id=%s",
                (username, school["id"]))
    teacher = cur.fetchone()
    if teacher and not verify_password(teacher["password"], password):
        teacher = None
    elif teacher and not is_hashed_password(teacher["password"]):
        cur.execute("UPDATE teachers SET password=%s WHERE id=%s", (hash_password(password), teacher["id"]))
        conn.commit()
    conn.close()

    if teacher:
        session.clear()
        session["teacher_logged_in"] = True
        session["teacher_id"] = teacher["id"]
        session["teacher_name"] = teacher["teacher_name"]
        session["teacher_photo"] = teacher.get("photo_path") or ""
        session["school_id"] = school["id"]
        session["school_name"] = school["name"]
        # Issue a new device token — invalidates any other active session
        session["device_token"] = create_user_session_token("teacher", teacher["id"], school["id"])
        return redirect("/teacher")

    return login_page("Teacher Login", "/teacher-login", "Teacher Username", "Teacher Password",
                      "Invalid teacher username or password", extra_fields=school_code_field)


@app.route("/student-login", methods=["GET", "POST"])
def student_login():
    school_code_field = '<input type="text" name="school_code" placeholder="School Code (e.g. ABC123)" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500" required>'
    if request.method == "GET":
        kicked_msg = "\u26a0\ufe0f Your account was logged in on another device. Please log in again." if request.args.get("kicked") else ""
        return login_page("Student Login", "/student-login", "Student ID", "Student Password",
                          error_message=kicked_msg, extra_fields=school_code_field)

    if login_rate_limited("student-login"):
        return too_many_attempts_response()

    school_code = request.form.get("school_code", "").strip().upper()
    student_id = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    school = get_school_by_code(school_code)
    if not school:
        return login_page("Student Login", "/student-login", "Student ID", "Student Password",
                          "School code not found.", extra_fields=school_code_field)

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM students WHERE student_id=%s AND school_id=%s",
                (student_id, school["id"]))
    student = cur.fetchone()
    if student and not verify_password(student["password"], password):
        student = None
    elif student and not is_hashed_password(student["password"]):
        cur.execute("UPDATE students SET password=%s WHERE id=%s", (hash_password(password), student["id"]))
        conn.commit()
    conn.close()

    if student:
        session.clear()
        session["student_logged_in"] = True
        session["student_db_id"] = student["id"]
        session["student_id"] = student["student_id"]
        session["student_name"] = student["full_name"]
        session["school_id"] = school["id"]
        session["school_name"] = school["name"]
        # Issue a new device token — invalidates any other active session
        session["device_token"] = create_user_session_token("student", student["id"], school["id"])
        return redirect("/student")

    return login_page("Student Login", "/student-login", "Student ID", "Student Password",
                      "Invalid student ID or password", extra_fields=school_code_field)


@app.route("/admin-logout", methods=["GET", "POST"])
def admin_logout():
    session.clear()
    response = redirect("/admin-login")
    response.delete_cookie("session")
    return response


@app.route("/teacher-logout", methods=["GET", "POST"])
def teacher_logout():
    teacher_id = session.get("teacher_id")
    school_id = session.get("school_id", 1)
    if teacher_id:
        revoke_user_session_token("teacher", teacher_id, school_id)
    session.clear()
    response = redirect("/teacher-login")
    response.delete_cookie("session")
    return response


@app.route("/student-logout", methods=["GET", "POST"])
def student_logout():
    student_db_id = session.get("student_db_id")
    school_id = session.get("school_id", 1)
    if student_db_id:
        revoke_user_session_token("student", student_db_id, school_id)
    session.clear()
    response = redirect("/student-login")
    response.delete_cookie("session")
    return response



@app.route("/student/delete-account", methods=["POST"])
def student_delete_account():
    protect = student_required()
    if protect:
        return protect
    student_db_id = get_logged_student_db_id()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT student_id, image_file FROM students WHERE id=%s", (student_db_id,))
    row = cur.fetchone()
    if row:
        student_id = row["student_id"]
        img = row["image_file"]
        cur.execute("DELETE FROM attendance WHERE student_id=%s", (student_id,))
        cur.execute("DELETE FROM student_classes WHERE student_id_fk=%s", (student_db_id,))
        cur.execute("DELETE FROM students WHERE id=%s", (student_db_id,))
        conn.commit()
        try:
            if img and img.startswith("http"):
                supabase_delete(img.split("/")[-1])
            else:
                path = os.path.join(IMAGE_DIR, img)
                if os.path.exists(path):
                    os.remove(path)
        except:
            pass
    conn.close()
    load_known_faces()
    session.clear()
    response = redirect("/student-login")
    response.delete_cookie("session")
    return response

# =========================================================
# PUBLIC SCHOOL SELF-REGISTRATION
# =========================================================
@app.route("/register-school", methods=["GET", "POST"])
def register_school():
    if request.method == "GET":
        return page_wrapper("Register Your School", f"""
        <div class="max-w-lg mx-auto my-8 p-6 bg-white border border-slate-200 rounded-xl shadow-sm">
            <h2 class="text-2xl font-bold text-slate-800 mb-2">🏫 Register Your School</h2>
            <p class="text-sm text-slate-500 mb-6">Submit your school's details below. Once a system administrator reviews and approves your request, you'll be able to log in with your School Code.</p>
            <form method="POST" class="space-y-4">
                            {csrf_input_html()}
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">School Full Name</label>
                    <input type="text" name="name" placeholder="e.g. Green Hills Secondary School" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500" required>
                </div>
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">School Code <span class="text-slate-400">(short, unique, no spaces)</span></label>
                    <input type="text" name="code" placeholder="e.g. GREENHS" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500" required>
                </div>
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Admin Username</label>
                    <input type="text" name="admin_username" placeholder="e.g. principal" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500" required>
                </div>
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Admin Password</label>
                    <input type="password" name="admin_password" placeholder="Strong password" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500" required>
                </div>
                <button class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition" type="submit">Submit for Approval</button>
            </form>
            <div class="mt-4 border-t pt-4 text-center">
                <a class="text-sm text-blue-600 hover:underline" href="/">← Back to Home</a>
            </div>
        </div>
        """)

    name = request.form.get("name", "").strip()
    code = request.form.get("code", "").strip().upper().replace(" ", "")
    admin_username = request.form.get("admin_username", "").strip()
    admin_password = request.form.get("admin_password", "").strip()
    if not name or not code or not admin_username or not admin_password:
        return "<script>alert('All fields are required');window.location.href='/register-school';</script>"

    conn = get_db()
    cur = conn.cursor()
    try:
        # Check existing schools for code/username collisions
        cur.execute("SELECT 1 FROM schools WHERE code=%s", (code,))
        if cur.fetchone():
            conn.close()
            return "<script>alert('That School Code is already taken. Please choose another.');window.location.href='/register-school';</script>"
        cur.execute("SELECT 1 FROM schools WHERE admin_username=%s", (admin_username,))
        if cur.fetchone():
            conn.close()
            return "<script>alert('That Admin Username is already taken. Please choose another.');window.location.href='/register-school';</script>"
        # Check pending (not-yet-reviewed) requests for the same collisions
        cur.execute("SELECT 1 FROM pending_school_registrations WHERE code=%s", (code,))
        if cur.fetchone():
            conn.close()
            return "<script>alert('That School Code is already pending approval for another request. Please choose another.');window.location.href='/register-school';</script>"
        cur.execute("SELECT 1 FROM pending_school_registrations WHERE admin_username=%s", (admin_username,))
        if cur.fetchone():
            conn.close()
            return "<script>alert('That Admin Username is already pending approval for another request. Please choose another.');window.location.href='/register-school';</script>"

        token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(days=30)
        cur.execute("""
            INSERT INTO pending_school_registrations (token, name, code, admin_username, admin_password, email, expires_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (token, name, code, admin_username, hash_password(admin_password), "", expires_at))
        conn.commit()
    except Exception as e:
        conn.rollback()
        conn.close()
        msg = str(e).split('\n')[0].replace("'", "")
        return f"<script>alert('Error: {msg}');window.location.href='/register-school';</script>"
    conn.close()
    return ("<script>alert('Thanks! Your request has been submitted and is waiting for admin approval. "
            "You will be able to log in once it is approved.');window.location.href='/';</script>")


@app.route("/super-admin/approve-school/<int:pending_id>", methods=["GET", "POST"])
def super_admin_approve_school(pending_id):
    if not is_super_admin():
        return redirect("/super-admin-login")
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM pending_school_registrations WHERE id=%s", (pending_id,))
    pending = cur.fetchone()
    if not pending:
        conn.close()
        if is_ajax(): return ajax_err("Request not found (it may have already been handled).")
        return "<script>alert('Request not found (it may have already been handled).');window.location.href='/super-admin';</script>"

    try:
        cur.execute("""
            INSERT INTO schools (name, code, admin_username, admin_password, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (pending["name"], pending["code"], pending["admin_username"], pending["admin_password"],
              datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        cur.execute("SELECT id FROM schools WHERE code=%s", (pending["code"],))
        new_school = cur.fetchone()
        if new_school:
            cur.execute("""
                INSERT INTO admin_settings (school_id, key, value) VALUES (%s, 'admin_password', %s)
                ON CONFLICT (school_id, key) DO UPDATE SET value=%s
            """, (new_school["id"], pending["admin_password"], pending["admin_password"]))
        cur.execute("DELETE FROM pending_school_registrations WHERE id=%s", (pending_id,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        conn.close()
        msg = str(e).split('\n')[0].replace("'", "")
        if "schools_code_key" in msg:
            msg = "That School Code was already taken by another approved school. Reject this request and ask them to resubmit with a different code."
        elif "schools_admin_username_key" in msg:
            msg = "That Admin Username was already taken by another approved school. Reject this request and ask them to resubmit with a different username."
        if is_ajax(): return ajax_err(msg)
        return f"<script>alert('Error: {msg}');window.location.href='/super-admin';</script>"
    conn.close()
    if is_ajax(): return ajax_ok(f"Approved! {pending['name']} is now active with code {pending['code']}.")
    return f"<script>alert('Approved! {pending['name']} is now active with code {pending['code']}.');window.location.href='/super-admin';</script>"


@app.route("/super-admin/reject-school/<int:pending_id>", methods=["GET", "POST"])
def super_admin_reject_school(pending_id):
    if not is_super_admin():
        return redirect("/super-admin-login")
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM pending_school_registrations WHERE id=%s", (pending_id,))
    conn.commit()
    conn.close()
    if is_ajax(): return ajax_ok("Request rejected and removed.")
    return "<script>alert('Request rejected and removed.');window.location.href='/super-admin';</script>"



# =========================================================
# STUDENT REGISTRATION PAGE
# =========================================================
@app.route("/student-register")
def student_register():
    return page_wrapper("Student Registration", """
        <div class="max-w-2xl mx-auto text-center">
            <h1 class="text-3xl font-bold text-slate-800 mb-2">Student Face Registration</h1>
            <p class="text-sm text-slate-500 mb-6">Input student data fields and click register to create face vector profile mappings</p>
            
            <div class="space-y-3 max-w-md mx-auto mb-6 text-left">
                <input type="text" id="schoolName" placeholder="School Name (e.g. Green Hills Secondary School)" class="w-full px-3 py-2 border rounded-lg">
                <input type="text" id="schoolCode" placeholder="School Code (e.g. ABC123)" class="w-full px-3 py-2 border rounded-lg">
                <input type="text" id="studentId" placeholder="Student Alphanumeric ID" class="w-full px-3 py-2 border rounded-lg">
                <input type="text" id="fullName" placeholder="Full Registered Name" class="w-full px-3 py-2 border rounded-lg">
                <input type="password" id="password" placeholder="Roster Login Password" class="w-full px-3 py-2 border rounded-lg">
            </div>

            <video id="video" autoplay playsinline muted class="w-full max-w-md mx-auto bg-black rounded-xl shadow-inner mb-4 border border-slate-300"></video>
            
            <div class="flex justify-center gap-2 flex-wrap mb-4">
                <button class="bg-emerald-600 hover:bg-emerald-700 text-white px-4 py-2 rounded-lg font-semibold" onclick="registerFace()">Capture & Register Face</button>
                <button class="bg-orange-500 hover:bg-orange-600 text-white px-4 py-2 rounded-lg font-semibold" onclick="startCamera()">Restart Stream Feed</button>
                <button class="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg font-semibold" onclick="switchCamera()">Switch Orientation</button>
            </div>

            <div id="https-warning" style="display:none;background:#fef2f2;border:1px solid #fca5a5;color:#991b1b;padding:10px 16px;border-radius:8px;margin-bottom:12px;font-weight:600;">⚠️ Camera requires HTTPS. Please open this page using <b>https://</b> — the camera will not work over plain http://</div>
            <div id="status" class="text-sm font-semibold text-slate-700 mt-2">Starting camera...</div>
        </div>

<script>
const video = document.getElementById('video');
const statusDiv = document.getElementById('status');
let stream = null;
let currentFacingMode = "user";

async function startCamera() {
    try {
        statusDiv.innerText = "Starting camera...";
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            statusDiv.innerText = "Camera not supported. Please use HTTPS and a modern browser.";
            return;
        }
        if (stream) {
            stream.getTracks().forEach(t => t.stop());
            stream = null;
        }
        // Try exact facingMode first, then ideal, then any
        let constraints = [
            { video: { facingMode: { exact: currentFacingMode }, width: { ideal: 640 }, height: { ideal: 480 } }, audio: false },
            { video: { facingMode: { ideal: currentFacingMode }, width: { ideal: 640 }, height: { ideal: 480 } }, audio: false },
            { video: true, audio: false }
        ];
        let lastErr = null;
        for (let c of constraints) {
            try {
                stream = await navigator.mediaDevices.getUserMedia(c);
                break;
            } catch (e) {
                lastErr = e;
                stream = null;
            }
        }
        if (!stream) throw lastErr;
        video.srcObject = stream;
        await new Promise((resolve) => {
            video.onloadedmetadata = () => resolve();
        });
        try {
            await video.play();
            statusDiv.innerText = "✅ Camera ready (" + (currentFacingMode === "user" ? "Front" : "Back") + ")";
        } catch (playErr) {
            statusDiv.innerText = "Tap the video to start";
            video.onclick = async () => { await video.play(); statusDiv.innerText = "✅ Camera ready"; video.onclick = null; };
        }
    } catch (err) {
        if (err.name === "NotAllowedError") {
            statusDiv.innerText = "❌ Camera permission denied. Please allow camera access in your browser settings and reload.";
        } else if (err.name === "NotFoundError") {
            statusDiv.innerText = "❌ No camera found on this device.";
        } else if (location.protocol !== "https:") {
            statusDiv.innerText = "❌ Camera requires HTTPS. Please open this site using https://";
        } else {
            statusDiv.innerText = "❌ Camera error: " + err.message;
        }
    }
}

function switchCamera() {
    currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
    startCamera();
}

async function registerFace() {
    try {
        const schoolName = document.getElementById('schoolName').value.trim();
        const schoolCode = document.getElementById('schoolCode').value.trim();
        const studentId = document.getElementById('studentId').value.trim();
        const fullName = document.getElementById('fullName').value.trim();
        const password = document.getElementById('password').value.trim();
        
        if (!schoolName || !schoolCode || !studentId || !fullName || !password) {
            alert("All values are required prior to scanning");
            return;
        }

        statusDiv.innerText = "Capturing canvas frame matrix...";
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const image = canvas.toDataURL('image/jpeg');

        statusDiv.innerText = "Uploading credentials to database registry...";
        const response = await fetch('/register-face', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ school_name: schoolName, school_code: schoolCode, student_id: studentId, full_name: fullName, password: password, image: image })
        });
        const data = await response.json();
        statusDiv.innerText = data.message;
        if (data.success) {
            alert(data.message);
        }
    } catch (err) {
        statusDiv.innerText = "Registration failed: " + err.message;
    }
}
if (location.protocol !== "https:") { document.getElementById("https-warning").style.display = "block"; document.getElementById("status").innerText = "❌ Camera unavailable — HTTPS required."; } else { startCamera(); }
</script>
""", is_admin=is_admin_logged_in())


@app.route("/register-face", methods=["POST"])
def register_face():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No data received"})
        school_name = data.get("school_name", "").strip()
        school_code = data.get("school_code", "").strip().upper()
        student_id = data.get("student_id", "").strip()
        full_name = data.get("full_name", "").strip()
        password = data.get("password", "").strip()
        image_data = data.get("image", "")

        if not school_name or not school_code or not student_id or not full_name or not password or not image_data:
            return jsonify({"success": False, "message": "All database fields are required"})

        school = get_school_by_code(school_code)
        if not school:
            return jsonify({"success": False, "message": "School code not found"})
        if school_name.strip().lower() != school["name"].strip().lower():
            return jsonify({"success": False, "message": "School name does not match this school code"})
        school_id = school["id"]

        if student_exists(student_id, school_id=school_id):
            return jsonify({"success": False, "message": "This Student ID already holds a target map registry record"})

        safe_id = sanitize_filename(student_id)
        safe_name = sanitize_filename(full_name)
        filename = f"{safe_id}_{safe_name}.jpg"
        file_path = os.path.join(IMAGE_DIR, filename)

        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_data = image_data.replace(" ", "+")
        missing_padding = len(image_data) % 4
        if missing_padding:
            image_data += "=" * (4 - missing_padding)

        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "message": "Invalid image matrix array decoder mapping"})

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        embedding = _get_face_embedding(rgb_frame)

        if embedding is None:
            return jsonify({"success": False, "message": "No human faces found in frame context. Please try again."})

        # Encode frame to bytes and upload to Supabase Storage
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()
        public_url = supabase_upload(filename, img_bytes)
        if not public_url:
            return jsonify({"success": False, "message": "Failed to upload image to storage. Please try again."})

        # Also save locally as fallback for face recognition loading.
        # Best-effort only: on read-only filesystems (e.g. Vercel) this is
        # expected to fail, and that's fine — the Supabase upload above is
        # the real source of truth and load_known_faces() reads from the
        # public_url just fine.
        try:
            os.makedirs(IMAGE_DIR, exist_ok=True)
            cv2.imwrite(file_path, frame)
        except OSError as e:
            print("Local image cache save skipped (read-only fs):", e, flush=True)

        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO students (school_id, student_id, full_name, password, image_file, registered_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (school_id, student_id, full_name, hash_password(password), public_url, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()

        load_known_faces(school_id)
        return jsonify({"success": True, "message": f"Successfully registered face map vector profile for {full_name}!"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Internal mapping failure error: {str(e)}"})


# =========================================================
# ADMIN CONTROLLER DASHBOARD
# =========================================================
@app.route("/admin")
def admin_dashboard():
    protect = admin_required()
    if protect:
        return protect

    students = get_all_students()
    teachers = get_all_teachers()
    classes = get_all_classes()
    attendance = get_all_attendance()

    school_name = session.get("school_name", "")
    body = f"""
    <div class="section-stack">
        <div class="page-header">
            <div class="page-title">Admin Dashboard</div>
            <div class="page-sub">{f'🏫 {school_name} · ' if school_name else ''}Manage classes, instructors, students, and review attendance records.</div>
        </div>

        <div class="stat-grid">
            <div class="stat-card blue">
                <div class="stat-icon">👨‍🏫</div>
                <div class="stat-value">{len(teachers)}</div>
                <div class="stat-label">Teachers</div>
            </div>
            <div class="stat-card purple">
                <div class="stat-icon">📚</div>
                <div class="stat-value">{len(classes)}</div>
                <div class="stat-label">Classes</div>
            </div>
            <div class="stat-card green">
                <div class="stat-icon">🧑‍🎓</div>
                <div class="stat-value">{len(students)}</div>
                <div class="stat-label">Students</div>
            </div>
            <div class="stat-card orange">
                <div class="stat-icon">📋</div>
                <div class="stat-value">{len(attendance)}</div>
                <div class="stat-label">Records</div>
            </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="card card-body">
                <h2 class="text-xl font-bold text-slate-800 mb-4">Add New Instructor</h2>
                <form method="POST" action="/admin/create-teacher" class="space-y-3" data-ajax>
                            {csrf_input_html()}
                    <input type="text" name="teacher_name" placeholder="Full Name" class="form-input" required>
                    <input type="text" name="username" placeholder="Username" class="form-input" required>
                    <input type="text" name="password" placeholder="Password" class="form-input" required>
                    <button class="btn green" type="submit">Create Instructor</button>
                </form>
            </div>

            <div class="card card-body">
                <h2 class="text-xl font-bold text-slate-800 mb-4">Create New Class</h2>
                <form method="POST" action="/admin/create-class" class="space-y-2" data-ajax>
                            {csrf_input_html()}
                    <input type="text" name="class_name" placeholder="Class Name" class="form-input" required>
                    <input type="text" name="department" placeholder="Department" class="form-input">
                    <input type="text" name="course" placeholder="Course ID" class="form-input">
                    <input type="text" name="section_name" placeholder="Section" class="form-input">
                    <input type="text" name="subject_name" placeholder="Subject" class="form-input">
                    <select name="teacher_id" class="form-input" required>
                        <option value="">Assign Teacher</option>
    """
    for t in teachers:
        body += f'<option value="{t["id"]}">{t["teacher_name"]} ({t["username"]})</option>'
    body += f"""
                    </select>
                    <button class="btn green mt-2" type="submit">Create Class</button>
                </form>
            </div>
        </div>

        <div class="card card-body">
            <h2 class="text-xl font-bold text-slate-800 mb-4">Enroll Student in Class</h2>
            <form method="POST" action="/admin/assign-student-class" class="grid grid-cols-1 md:grid-cols-3 gap-3 items-end" data-ajax>
                            {csrf_input_html()}
                <select name="student_db_id" class="form-input" required>
                    <option value="">Select Student</option>
    """
    for s in students:
        body += f'<option value="{s["id"]}">{s["student_id"]} - {s["full_name"]}</option>'
    body += """
                </select>
                <select name="class_id" class="form-input" required>
                    <option value="">Select Class</option>
    """
    for c in classes:
        teacher_name = c["teacher_display_name"] or c["teacher_name"] or ""
        body += f'<option value="{c["id"]}">{c["class_name"]} | {c["subject_name"] or ""} | {teacher_name}</option>'
    body += """
                </select>
                <button class="btn green w-full md:w-auto" type="submit">Enroll Student</button>
            </form>
        </div>

        <div class="card">
            <div class="card-header">
                <span class="card-header-title">Teachers</span>
            </div>
            <div class="tbl-wrap">
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Name</th>
                        <th>Username</th>
                        <th>Password</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
    """
    if teachers:
        for t in teachers:
            body += f"""
                <tr>
                    <td class="text-slate-400 text-xs font-mono">#{t["id"]}</td>
                    <td class="font-semibold">{t["teacher_name"]}</td>
                    <td class="text-slate-600">{t["username"]}</td>
                    <td class="font-mono text-xs text-slate-500">••••••••</td>
                    <td>
                        <a class="text-blue-600 hover:underline font-medium mr-3" href="/admin/edit-teacher/{t['id']}">Edit</a>
                        <form method="POST" action="/admin/delete-teacher/{t['id']}" style="display:inline;" onsubmit="return confirm('Delete this teacher?')">
                            {csrf_input_html()}
                            <button type="submit" style="background:none;border:none;padding:0;color:#ef4444;font-weight:600;cursor:pointer;font-size:14px;">Delete</button>
                        </form>
                    </td>
                </tr>
            """
    else:
        body += "<tr><td colspan='5' style='padding:24px; text-align:center; color:#94a3b8;'>No teachers yet.</td></tr>"
    body += """
                </tbody>
            </table>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <span class="card-header-title">Classes</span>
            </div>
            <div class="tbl-wrap">
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Class Name</th>
                        <th>Subject</th>
                        <th>Teacher</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
    """
    if classes:
        for c in classes:
            t_name = c["teacher_display_name"] or c["teacher_name"] or "Unassigned"
            body += f"""
                <tr>
                    <td class="text-slate-400 text-xs font-mono">#{c["id"]}</td>
                    <td class="font-semibold"><a class="text-blue-600 hover:underline" href="/admin/class/{c["id"]}">{c["class_name"]}</a></td>
                    <td class="text-slate-600">{c["subject_name"] or "—"}</td>
                    <td class="text-slate-600">{t_name}</td>
                    <td>
                        <a class="text-blue-600 hover:underline font-medium mr-3" href="/admin/edit-class/{c['id']}">Edit</a>
                        <form method="POST" action="/admin/delete-class/{c['id']}" style="display:inline;" onsubmit="return confirm('Delete this class?')">
                            {csrf_input_html()}
                            <button type="submit" style="background:none;border:none;padding:0;color:#ef4444;font-weight:600;cursor:pointer;font-size:14px;">Delete</button>
                        </form>
                    </td>
                </tr>
            """
    else:
        body += "<tr><td colspan='5' style='padding:24px; text-align:center; color:#94a3b8;'>No classes yet.</td></tr>"
    body += """
                </tbody>
            </table>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <span class="card-header-title">Students</span>
            </div>
            <div class="tbl-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Photo</th>
                        <th>Student ID</th>
                        <th>Name</th>
                        <th>Password</th>
                        <th>Registered</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
    """
    if students:
        for s in students:
            body += f"""
                <tr>
                    <td><img class="w-10 h-10 object-cover rounded-full border-2 border-slate-200" src="{supabase_public_url(s["image_file"])}"></td>
                    <td class="font-mono text-xs text-slate-500">{s["student_id"]}</td>
                    <td class="font-semibold">{s["full_name"]}</td>
                    <td class="font-mono text-xs text-slate-400">••••••••</td>
                    <td class="text-xs text-slate-400">{s["registered_at"]}</td>
                    <td>
                        <a class="text-blue-600 hover:underline font-medium mr-3" href="/admin/edit-student/{s['id']}">Edit</a>
                        <form method="POST" action="/admin/delete-student/{s['id']}" style="display:inline;" onsubmit="return confirm('Delete this student?')">
                            {csrf_input_html()}
                            <button type="submit" style="background:none;border:none;padding:0;color:#ef4444;font-weight:600;cursor:pointer;font-size:14px;">Delete</button>
                        </form>
                    </td>
                </tr>
            """
    else:
        body += "<tr><td colspan='6' style='padding:24px; text-align:center; color:#94a3b8;'>No students registered yet.</td></tr>"
    body += """
                </tbody>
            </table>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <span class="card-header-title">Attendance Log</span>
                <a class="btn green" style="font-size:13px;padding:7px 14px;" href="/export-attendance">📥 Export CSV</a>
            </div>
            <div class="tbl-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Time</th>
                        <th>Student ID</th>
                        <th>Name</th>
                        <th>Class</th>
                        <th>Section</th>
                        <th>Subject</th>
                        <th>Status</th>
                        <th>Teacher</th>
                    </tr>
                </thead>
                <tbody>
    """
    if attendance:
        for a in attendance:
            badge_class = "badge badge-green" if a['status'] == 'Present' else "badge badge-red"
            body += f"""
                <tr>
                    <td class="text-xs font-mono whitespace-nowrap">{a["date"]}</td>
                    <td class="text-xs text-slate-400">{format_time_12hr(a["time"])}</td>
                    <td class="text-xs font-mono text-slate-500">{a["student_id"]}</td>
                    <td class="font-medium">{a["full_name"]}</td>
                    <td class="font-semibold">{a["class_name"]}</td>
                    <td class="text-slate-500 text-sm">{a["section_name"] or "—"}</td>
                    <td class="text-slate-500 text-sm">{a["subject_name"] or "—"}</td>
                    <td><span class="{badge_class}">{a["status"]}</span></td>
                    <td class="text-slate-400 text-sm">{a["teacher_name"] or "—"}</td>
                </tr>
            """
    else:
        body += "<tr><td colspan='9' style='padding:24px; text-align:center; color:#94a3b8;'>No attendance records yet.</td></tr>"
    body += """
                </tbody>
            </table>
            </div>
        </div>
    </div>
    """
    return page_wrapper("Admin Dashboard", body, is_admin=True)


# =========================================================
# ADMIN CHARTS DASHBOARD
# =========================================================
@app.route("/admin/charts")
def admin_charts():
    protect = admin_required()
    if protect:
        return protect

    conn = get_db()
    cur = conn.cursor()
    school_id = get_current_school_id()

    # Daily attendance last 14 days
    cur.execute("""
        SELECT date, COUNT(*) as total,
               SUM(CASE WHEN status='Present' THEN 1 ELSE 0 END) as present
        FROM attendance
        WHERE date >= (CURRENT_DATE - INTERVAL '14 days')::text AND school_id=%s
        GROUP BY date ORDER BY date ASC
    """, (school_id,))
    daily = cur.fetchall()

    # Per-class attendance rate
    cur.execute("""
        SELECT class_name,
               COUNT(*) as total,
               SUM(CASE WHEN status='Present' THEN 1 ELSE 0 END) as present
        FROM attendance
        WHERE school_id=%s
        GROUP BY class_name ORDER BY total DESC LIMIT 10
    """, (school_id,))
    by_class = cur.fetchall()

    # Present vs Absent overall
    cur.execute("SELECT status, COUNT(*) as c FROM attendance WHERE school_id=%s GROUP BY status", (school_id,))
    status_rows = cur.fetchall()

    # Top 5 students by attendance
    cur.execute("""
        SELECT full_name,
               COUNT(*) as total,
               SUM(CASE WHEN status='Present' THEN 1 ELSE 0 END) as present
        FROM attendance
        WHERE school_id=%s
        GROUP BY full_name ORDER BY present DESC LIMIT 8
    """, (school_id,))
    top_students = cur.fetchall()

    conn.close()

    import json
    daily_labels = json.dumps([r["date"] for r in daily])
    daily_present = json.dumps([int(r["present"]) for r in daily])
    daily_absent = json.dumps([int(r["total"]) - int(r["present"]) for r in daily])

    class_labels = json.dumps([r["class_name"] for r in by_class])
    class_rates = json.dumps([round(int(r["present"])/int(r["total"])*100,1) if r["total"] else 0 for r in by_class])

    status_map = {{r["status"]: int(r["c"]) for r in status_rows}}
    present_count = status_map.get("Present", 0)
    absent_count = status_map.get("Absent", 0)

    student_labels = json.dumps([r["full_name"] for r in top_students])
    student_rates = json.dumps([round(int(r["present"])/int(r["total"])*100,1) if r["total"] else 0 for r in top_students])

    body = f"""
    <div class="section-stack">
        <div class="page-header">
            <div class="page-title">Attendance Charts</div>
            <div class="page-sub">Visual overview of attendance trends across all classes and students.</div>
        </div>

        <div class="chart-grid">
            <div class="chart-card" style="grid-column: span 2;">
                <div class="chart-title">Daily Attendance — Last 14 Days</div>
                <canvas id="chartDaily" height="100"></canvas>
            </div>
            <div class="chart-card">
                <div class="chart-title">Overall Status</div>
                <canvas id="chartDonut" height="200"></canvas>
            </div>
            <div class="chart-card">
                <div class="chart-title">Attendance Rate by Class (%)</div>
                <canvas id="chartClass" height="200"></canvas>
            </div>
            <div class="chart-card" style="grid-column: span 2;">
                <div class="chart-title">Top Students by Attendance Rate (%)</div>
                <canvas id="chartStudents" height="100"></canvas>
            </div>
        </div>
    </div>

    <script>
    const isDark = () => document.documentElement.classList.contains('dark');
    const textColor = () => isDark() ? '#8b949e' : '#475569';
    const gridColor = () => isDark() ? '#21262d' : '#f1f5f9';

    window._charts = [];

    const dailyLabels = {daily_labels};
    const dailyPresent = {daily_present};
    const dailyAbsent = {daily_absent};

    const c1 = new Chart(document.getElementById('chartDaily'), {{
        type: 'bar',
        data: {{
            labels: dailyLabels,
            datasets: [
                {{ label: 'Present', data: dailyPresent, backgroundColor: '#3fb950', borderRadius: 5, stack: 'a' }},
                {{ label: 'Absent', data: dailyAbsent, backgroundColor: '#f85149', borderRadius: 5, stack: 'a' }}
            ]
        }},
        options: {{
            responsive: true, plugins: {{ legend: {{ labels: {{ color: textColor() }} }} }},
            scales: {{
                x: {{ stacked: true, grid: {{ color: gridColor() }}, ticks: {{ color: textColor() }} }},
                y: {{ stacked: true, grid: {{ color: gridColor() }}, ticks: {{ color: textColor() }} }}
            }}
        }}
    }});
    window._charts.push(c1);

    const c2 = new Chart(document.getElementById('chartDonut'), {{
        type: 'doughnut',
        data: {{
            labels: ['Present', 'Absent'],
            datasets: [{{ data: [{present_count}, {absent_count}], backgroundColor: ['#3fb950','#f85149'], borderWidth: 0 }}]
        }},
        options: {{
            responsive: true,
            plugins: {{ legend: {{ labels: {{ color: textColor() }} }} }}
        }}
    }});
    window._charts.push(c2);

    const c3 = new Chart(document.getElementById('chartClass'), {{
        type: 'bar',
        data: {{
            labels: {class_labels},
            datasets: [{{ label: 'Rate %', data: {class_rates}, backgroundColor: '#4493f8', borderRadius: 5 }}]
        }},
        options: {{
            indexAxis: 'y', responsive: true,
            plugins: {{ legend: {{ labels: {{ color: textColor() }} }} }},
            scales: {{
                x: {{ max: 100, grid: {{ color: gridColor() }}, ticks: {{ color: textColor() }} }},
                y: {{ grid: {{ color: gridColor() }}, ticks: {{ color: textColor() }} }}
            }}
        }}
    }});
    window._charts.push(c3);

    const c4 = new Chart(document.getElementById('chartStudents'), {{
        type: 'bar',
        data: {{
            labels: {student_labels},
            datasets: [{{ label: 'Attendance %', data: {student_rates}, backgroundColor: '#a371f7', borderRadius: 5 }}]
        }},
        options: {{
            responsive: true,
            plugins: {{ legend: {{ labels: {{ color: textColor() }} }} }},
            scales: {{
                x: {{ grid: {{ color: gridColor() }}, ticks: {{ color: textColor() }} }},
                y: {{ max: 100, grid: {{ color: gridColor() }}, ticks: {{ color: textColor() }} }}
            }}
        }}
    }});
    window._charts.push(c4);
    </script>
    """
    return page_wrapper("Charts", body, is_admin=True)



# =========================================================
# ADMIN CONTROLLERS IMPLEMENTATION MUTATION ROUTING
# =========================================================
@app.route("/admin/create-teacher", methods=["POST"])
def admin_create_teacher():
    protect = admin_required()
    if protect:
        return protect
    teacher_name = request.form.get("teacher_name", "").strip()
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    if not teacher_name or not username or not password:
        return "<script>alert('All instructor fields are required');window.location.href='/admin';</script>"
    if teacher_username_exists(username):
        return "<script>alert('Teacher login username already occupies registry namespace');window.location.href='/admin';</script>"

    conn = get_db()
    cur = conn.cursor()
    school_id = get_current_school_id()
    cur.execute("""
        INSERT INTO teachers (school_id, teacher_name, username, password, created_at)
        VALUES (%s, %s, %s, %s, %s)
    """, (school_id, teacher_name, username, hash_password(password), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    if is_ajax(): return ajax_ok("Instructor created successfully!", redirect_url="/admin")
    return "<script>alert('Instructor profile instantiated successfully');window.location.href='/admin';</script>"


@app.route("/admin/edit-teacher/<int:teacher_id>", methods=["GET", "POST"])
def admin_edit_teacher(teacher_id):
    protect = admin_required()
    if protect:
        return protect
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM teachers WHERE id=%s", (teacher_id,))
    teacher = cur.fetchone()
    if not teacher:
        conn.close()
        return "Teacher target row index match not located inside dynamic memory state", 404

    if request.method == "POST":
        teacher_name = request.form.get("teacher_name", "").strip()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if password:
            cur.execute("""
                UPDATE teachers SET teacher_name=%s, username=%s, password=%s WHERE id=%s
            """, (teacher_name, username, hash_password(password), teacher_id))
        else:
            cur.execute("""
                UPDATE teachers SET teacher_name=%s, username=%s WHERE id=%s
            """, (teacher_name, username, teacher_id))
        conn.commit()
        
        cur.execute("UPDATE classes SET teacher_display_name=%s WHERE teacher_id=%s", (teacher_name, teacher_id))
        conn.commit()
        
        conn.close()
        if is_ajax(): return ajax_ok("Teacher updated successfully!", redirect_url="/admin")
        return "<script>alert('Teacher entity parameters updated successfully');window.location.href='/admin';</script>"

    body = f"""
    <div class="max-w-md">
        <h1 class="text-2xl font-bold mb-4">Modify Instructor Record</h1>
        <form method="POST" class="space-y-4" data-ajax>
                            {csrf_input_html()}
            <div>
                <label class="block text-sm font-medium">Instructor Display Full Name</label>
                <input type="text" name="teacher_name" value="{esc(teacher["teacher_name"])}" class="w-full px-3 py-2 border rounded-lg" required>
            </div>
            <div>
                <label class="block text-sm font-medium">Login Username Reference</label>
                <input type="text" name="username" value="{teacher["username"]}" class="w-full px-3 py-2 border rounded-lg" required>
            </div>
            <div>
                <label class="block text-sm font-medium">Access Pass Key Token</label>
                <input type="text" name="password" placeholder="Leave blank to keep current password" class="w-full px-3 py-2 border rounded-lg">
            </div>
            <button type="submit" class="bg-blue-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-blue-700">Save Modifications</button>
            <a href="/admin" class="ml-2 inline-block bg-slate-100 text-slate-700 font-bold py-2 px-4 rounded-lg">Cancel</a>
        </form>
    </div>
    """
    conn.close()
    return page_wrapper("Edit Teacher Entity Matrix", body, is_admin=True)


@app.route("/admin/delete-teacher/<int:teacher_id>", methods=["POST"])
def admin_delete_teacher(teacher_id):
    protect = admin_required()
    if protect:
        return protect
    if not verify_csrf():
        return "Invalid or missing CSRF token.", 400
    school_id = get_current_school_id()
    conn = get_db()
    cur = conn.cursor()
    # SECURITY: scope to this admin's school so they can't delete another school's teacher
    cur.execute("DELETE FROM teachers WHERE id=%s AND school_id=%s", (teacher_id, school_id))
    conn.commit()
    conn.close()
    return "<script>alert('Teacher purged successfully');window.location.href='/admin';</script>"


@app.route("/admin/create-class", methods=["POST"])
def admin_create_class():
    protect = admin_required()
    if protect:
        return protect
    class_name = request.form.get("class_name", "").strip()
    department = request.form.get("department", "").strip()
    course = request.form.get("course", "").strip()
    section_name = request.form.get("section_name", "").strip()
    subject_name = request.form.get("subject_name", "").strip()
    teacher_id = request.form.get("teacher_id", "").strip()

    if not class_name or not teacher_id:
        return "<script>alert('Class name and teacher selection are mandatory');window.location.href='/admin';</script>"

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM teachers WHERE id=%s", (teacher_id,))
    teacher = cur.fetchone()
    if not teacher:
        conn.close()
        return "<script>alert('Selected proctor identity mismatch data reference code');window.location.href='/admin';</script>"

    cur.execute("""
        INSERT INTO classes (
            school_id, class_name, department, course, section_name, subject_name, teacher_id, teacher_display_name, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (get_current_school_id(), class_name, department, course, section_name, subject_name, teacher["id"], teacher["teacher_name"], datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    if is_ajax(): return ajax_ok("Class created successfully!", redirect_url="/admin")
    return "<script>alert('New classroom created successfully');window.location.href='/admin';</script>"


@app.route("/admin/edit-class/<int:class_id>", methods=["GET", "POST"])
def admin_edit_class(class_id):
    protect = admin_required()
    if protect:
        return protect
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM classes WHERE id=%s", (class_id,))
    class_row = cur.fetchone()
    if not class_row:
        conn.close()
        return "Class file matching matrix not detected index inside standard layout references", 404

    if request.method == "POST":
        class_name = request.form.get("class_name", "").strip()
        department = request.form.get("department", "").strip()
        course = request.form.get("course", "").strip()
        section_name = request.form.get("section_name", "").strip()
        subject_name = request.form.get("subject_name", "").strip()
        teacher_id = request.form.get("teacher_id", "").strip()

        cur.execute("SELECT * FROM teachers WHERE id=%s", (teacher_id,))
        t = cur.fetchone()
        t_name = t["teacher_name"] if t else ""

        cur.execute("""
            UPDATE classes SET class_name=%s, department=%s, course=%s, section_name=%s, subject_name=%s, teacher_id=%s, teacher_display_name=%s
            WHERE id=%s
        """, (class_name, department, course, section_name, subject_name, teacher_id, t_name, class_id))
        conn.commit()
        conn.close()
        if is_ajax(): return ajax_ok("Class updated successfully!", redirect_url="/admin")
        return "<script>alert('Classroom record adjustments committed!');window.location.href='/admin';</script>"

    teachers = get_all_teachers()
    body = f"""
    <div class="max-w-md">
        <h1 class="text-2xl font-bold mb-4">Edit Classroom Configuration</h1>
        <form method="POST" class="space-y-3" data-ajax>
                            {csrf_input_html()}
            <input type="text" name="class_name" value="{class_row["class_name"]}" class="w-full px-3 py-2 border rounded-lg" required>
            <input type="text" name="department" value="{class_row["department"] or ""}" class="w-full px-3 py-2 border rounded-lg">
            <input type="text" name="course" value="{class_row["course"] or ""}" class="w-full px-3 py-2 border rounded-lg">
            <input type="text" name="section_name" value="{class_row["section_name"] or ""}" class="w-full px-3 py-2 border rounded-lg">
            <input type="text" name="subject_name" value="{class_row["subject_name"] or ""}" class="w-full px-3 py-2 border rounded-lg">
            <select name="teacher_id" class="w-full px-3 py-2 border rounded-lg" required>
    """
    for t in teachers:
        sel = "selected" if t["id"] == class_row["teacher_id"] else ""
        body += f'<option value="{t["id"]}" {sel}>{t["teacher_name"]}</option>'
    body += f"""
            </select>
            <button class="bg-blue-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-blue-700" type="submit">Save Adjustments</button>
            <a class="ml-2 inline-block bg-slate-100 text-slate-700 font-bold py-2 px-4 rounded-lg" href="/admin">Cancel</a>
        </form>
    </div>
    """
    conn.close()
    return page_wrapper("Modify Course Class Configuration", body, is_admin=True)


@app.route("/admin/delete-class/<int:class_id>", methods=["POST"])
def admin_delete_class(class_id):
    protect = admin_required()
    if protect:
        return protect
    if not verify_csrf():
        return "Invalid or missing CSRF token.", 400
    school_id = get_current_school_id()
    conn = get_db()
    cur = conn.cursor()
    # SECURITY: scope to this admin's school so they can't delete another school's class
    cur.execute("DELETE FROM classes WHERE id=%s AND school_id=%s", (class_id, school_id))
    conn.commit()
    conn.close()
    return "<script>alert('Classroom record mapping purged successfully');window.location.href='/admin';</script>"


@app.route("/admin/assign-student-class", methods=["POST"])
def admin_assign_student_class():
    protect = admin_required()
    if protect:
        return protect
    student_db_id = request.form.get("student_db_id", "").strip()
    class_id = request.form.get("class_id", "").strip()
    if not student_db_id or not class_id:
        if is_ajax(): return ajax_err("All enrollment parameters are needed.")
        return "<script>alert('All enrollment parameters are needed');window.location.href='/admin';</script>"
    assign_student_to_class(int(student_db_id), int(class_id))
    if is_ajax(): return ajax_ok("Student enrolled successfully!", redirect_url="/admin")
    return "<script>alert('Student enrollment map updated dynamically!');window.location.href='/admin';</script>"


@app.route("/admin/edit-student/<int:student_db_id>", methods=["GET", "POST"])
def admin_edit_student(student_db_id):
    protect = admin_required()
    if protect:
        return protect
    student = get_student_row_by_db_id(student_db_id)
    if not student:
        return "Student not found", 404
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        password = request.form.get("password", "").strip()
        new_student_id = request.form.get("student_id", "").strip()
        conn = get_db()
        cur = conn.cursor()
        old_student_id = student["student_id"]
        old_image = student["image_file"]
        new_image = old_image

        # Handle photo upload
        photo = request.files.get("photo")
        if photo and photo.filename:
            safe_id = sanitize_filename(new_student_id or old_student_id)
            safe_name = sanitize_filename(full_name)
            ext = photo.filename.rsplit(".", 1)[-1].lower() if "." in photo.filename else "jpg"
            new_filename = f"{safe_id}_{safe_name}.{ext}"
            img_bytes = photo.read()
            if not is_allowed_upload(img_bytes):
                return "Uploaded file is not a valid image.", 400
            public_url = supabase_upload(new_filename, img_bytes)
            if public_url:
                if old_image and old_image.startswith("http"):
                    supabase_delete(old_image.split("/")[-1])
                elif old_image:
                    try:
                        old_path = os.path.join(IMAGE_DIR, old_image)
                        if os.path.exists(old_path): os.remove(old_path)
                    except: pass
                new_image = public_url

        # Update student_id in attendance too if changed
        if new_student_id and new_student_id != old_student_id:
            cur.execute("UPDATE attendance SET student_id=%s WHERE student_id=%s", (new_student_id, old_student_id))

        final_password = hash_password(password) if password else student["password"]
        cur.execute(
            "UPDATE students SET full_name=%s, password=%s, student_id=%s, image_file=%s WHERE id=%s",
            (full_name, final_password, new_student_id or old_student_id, new_image, student_db_id)
        )
        conn.commit()
        conn.close()
        load_known_faces()
        if is_ajax(): return ajax_ok("Student profile updated successfully!", redirect_url="/admin")
        return "<script>alert('Student profile updated successfully');window.location.href='/admin';</script>"

    body = f"""
    <div class="max-w-lg">
        <h1 class="text-2xl font-bold mb-1">Edit Student Profile</h1>
        <p class="text-sm text-slate-500 mb-5">Admin can update all fields including Student ID and photo.</p>

        <div class="flex items-center gap-4 mb-6 p-4 bg-slate-50 rounded-xl border">
            <img id="photoPreview" src="{supabase_public_url(student["image_file"])}" class="w-20 h-20 rounded-full object-cover border-2 border-blue-400 shadow">
            <div>
                <p class="font-semibold text-slate-700">{student["full_name"]}</p>
                <p class="text-xs text-slate-400 font-mono">ID: {student["student_id"]}</p>
            </div>
        </div>

        <form method="POST" enctype="multipart/form-data" class="space-y-4">
                            {csrf_input_html()}
            <div>
                <label class="block text-sm font-semibold text-slate-700 mb-1">Student ID</label>
                <input type="text" name="student_id" value="{student["student_id"]}" class="w-full px-3 py-2 border rounded-lg" required>
            </div>
            <div>
                <label class="block text-sm font-semibold text-slate-700 mb-1">Full Name</label>
                <input type="text" name="full_name" value="{student["full_name"]}" class="w-full px-3 py-2 border rounded-lg" required>
            </div>
            <div>
                <label class="block text-sm font-semibold text-slate-700 mb-1">Password</label>
                <input type="text" name="password" placeholder="Leave blank to keep current password" class="w-full px-3 py-2 border rounded-lg">
            </div>
            <div>
                <label class="block text-sm font-semibold text-slate-700 mb-1">Profile Photo (Upload from file)</label>
                <input type="file" name="photo" accept="image/*" class="w-full px-3 py-2 border rounded-lg bg-white"
                    onchange="document.getElementById('photoPreview').src = URL.createObjectURL(this.files[0])">
                <p class="text-xs text-slate-400 mt-1">Leave empty to keep current photo.</p>
            </div>
            <div class="flex gap-2 pt-2">
                <button class="bg-blue-600 text-white font-bold py-2 px-5 rounded-lg hover:bg-blue-700" type="submit">Save Changes</button>
                <a class="inline-block bg-slate-100 text-slate-700 font-bold py-2 px-5 rounded-lg hover:bg-slate-200" href="/admin">Cancel</a>
            </div>
        </form>
    </div>
    """
    return page_wrapper("Edit Student Profile", body, is_admin=True)


@app.route("/admin/delete-student/<int:db_id>", methods=["POST"])
def admin_delete_student(db_id):
    protect = admin_required()
    if protect:
        return protect
    if not verify_csrf():
        return "Invalid or missing CSRF token.", 400
    school_id = get_current_school_id()
    conn = get_db()
    cur = conn.cursor()
    # SECURITY: scope to this admin's school so they can't delete another school's student
    cur.execute("SELECT id, student_id, image_file FROM students WHERE id=%s AND school_id=%s", (db_id, school_id))
    row = cur.fetchone()
    if row:
        student_id = row["student_id"]
        img = row["image_file"]
        cur.execute("DELETE FROM attendance WHERE student_id=%s AND school_id=%s", (student_id, school_id))
        cur.execute("DELETE FROM student_classes WHERE student_id_fk=%s", (db_id,))
        cur.execute("DELETE FROM students WHERE id=%s AND school_id=%s", (db_id, school_id))
        conn.commit()
        try:
            if img and img.startswith('http'):
                supabase_delete(img.split('/')[-1])
            else:
                path = os.path.join(IMAGE_DIR, img)
                if os.path.exists(path):
                    os.remove(path)
        except:
            pass
    conn.close()
    load_known_faces()
    return "<script>alert('Student account fully deleted');window.location.href='/admin';</script>"


@app.route("/admin/class/<int:class_id>")
def admin_view_class(class_id):
    protect = admin_required()
    if protect:
        return protect
    class_row = get_class_by_id(class_id)
    if not class_row:
        return "Class code record mapping target not found in framework database state", 404
    students = get_students_in_class(class_id)
    attendance = get_attendance_for_class(class_id)

    body = f"""
    <div class="space-y-6">
        <div>
            <h1 class="text-3xl font-bold text-slate-800">Class: {class_row["class_name"]}</h1>
            <p class="text-sm text-slate-500">Subject: {class_row["subject_name"] or "None"} | Section: {class_row["section_name"] or "None"}</p>
        </div>
        <div class="flex gap-2">
            <a class="bg-slate-800 text-white font-bold py-2 px-4 rounded-lg" href="/admin">← Return Dashboard</a>
        </div>
        
        <div class="bg-white border rounded-xl overflow-hidden shadow-sm">
            <div class="p-4 bg-slate-50 font-bold text-slate-800">Enrolled Students Matrix</div>
            <table class="w-full text-left">
                <thead>
                    <tr class="bg-slate-100 text-slate-700">
                        <th class="p-3">Face</th>
                        <th class="p-3">Student ID</th>
                        <th class="p-3">Full Name</th>
                        <th class="p-3">Action Control</th>
                    </tr>
                </thead>
                <tbody>
    """
    if students:
        for s in students:
            body += f"""
                <tr class="border-b">
                    <td class="p-3"><img class="w-8 h-8 object-cover rounded-full border" src="{supabase_public_url(s["image_file"])}"></td>
                    <td class="p-3 font-mono">{s["student_id"]}</td>
                    <td class="p-3 font-medium">{s["full_name"]}</td>
                    <td class="p-3">
                        <form method="POST" action="/admin/remove-student-from-class/{class_id}/{s['id']}" style="display:inline;" onsubmit="return confirm('Unmap from current class framework matrix?')">
                            {csrf_input_html()}
                            <button type="submit" style="background:none;border:none;padding:0;color:#ef4444;text-decoration:underline;cursor:pointer;font-size:14px;">Drop Enrollment</button>
                        </form>
                    </td>
                </tr>
            """
    else:
        body += "<tr><td colspan='4' class='p-4 text-center text-slate-400'>No student data currently mapped inside classroom roster.</td></tr>"
    body += """
                </tbody>
            </table>
        </div>

        <div class="bg-white border rounded-xl overflow-hidden shadow-sm">
            <div class="p-4 bg-slate-50 font-bold text-slate-800">Attendance Log</div>
            <table class="w-full text-left">
                <thead>
                    <tr class="bg-slate-100 text-slate-700">
                        <th class="p-3">Date</th>
                        <th class="p-3">Time logged</th>
                        <th class="p-3">Student ID</th>
                        <th class="p-3">Full Name</th>
                        <th class="p-3">Verification Flag</th>
                    </tr>
                </thead>
                <tbody>
    """
    if attendance:
        for a in attendance:
            body += f"""
                <tr class="border-b">
                    <td class="p-3">{a["date"]}</td>
                    <td class="p-3 text-slate-500">{format_time_12hr(a["time"])}</td>
                    <td class="p-3 font-mono">{a["student_id"]}</td>
                    <td class="p-3 font-medium">{a["full_name"]}</td>
                    <td class="p-3"><span class="px-2 py-0.5 rounded text-xs font-bold {'bg-emerald-100 text-emerald-800' if a['status']=='Present' else 'bg-rose-100 text-rose-800'}">{a["status"]}</span></td>
                </tr>
            """
    else:
        body += "<tr><td colspan='5' class='p-4 text-center text-slate-400'>No scan iterations completed for this stream matrix.</td></tr>"
    body += """
                </tbody>
            </table>
        </div>
    </div>
    """
    return page_wrapper("Class Details Matrix View", body, is_admin=True)


@app.route("/admin/remove-student-from-class/<int:class_id>/<int:student_db_id>", methods=["POST"])
def admin_remove_student_from_class(class_id, student_db_id):
    protect = admin_required()
    if protect:
        return protect
    if not verify_csrf():
        return "Invalid or missing CSRF token.", 400
    # SECURITY: get_class_by_id/get_student_row_by_db_id are scoped to the
    # admin's own school_id; if either comes back empty the ids don't
    # belong to this admin's school, so refuse instead of proceeding.
    if not get_class_by_id(class_id) or not get_student_row_by_db_id(student_db_id):
        return "Not found.", 404
    remove_student_from_class(student_db_id, class_id)
    return f"<script>alert('Student unmapped from current class matrix');window.location.href='/admin/class/{class_id}';</script>"


@app.route("/admin/reports")
def admin_reports():
    protect = admin_required()
    if protect:
        return protect
    daily = get_report_records("daily")
    weekly = get_report_records("weekly")
    monthly = get_report_records("monthly")

    def render_report_table(title, rows):
        html = f"""
        <div class="bg-white border rounded-xl overflow-hidden shadow-sm mt-4">
            <div class="p-4 bg-slate-50 font-bold text-slate-800">{title}</div>
            <table class="w-full text-left">
                <thead>
                    <tr class="bg-slate-100 text-slate-700">
                        <th class="p-3">Calendar Date</th>
                        <th class="p-3">Student ID</th>
                        <th class="p-3">Student Name</th>
                        <th class="p-3">Course Subject Target</th>
                        <th class="p-3">Topic Description</th>
                        <th class="p-3">Status Flag</th>
                        <th class="p-3">Proctor Name</th>
                    </tr>
                </thead>
                <tbody>
        """
        if rows:
            for r in rows:
                html += f"""
                    <tr class="border-b text-sm">
                        <td class="p-3">{r["date"]}</td>
                        <td class="p-3 font-mono">{r["student_id"]}</td>
                        <td class="p-3 font-medium">{r["full_name"]}</td>
                        <td class="p-3 font-semibold">{r["class_name"]}</td>
                        <td class="p-3">{r["subject_name"] or ""}</td>
                        <td class="p-3"><span class="px-2 py-0.5 rounded text-xs font-bold {'bg-emerald-100 text-emerald-800' if r['status']=='Present' else 'bg-rose-100 text-rose-800'}">{r["status"]}</span></td>
                        <td class="p-3 text-slate-500">{r["teacher_name"] or ""}</td>
                    </tr>
                """
        else:
            html += "<tr><td colspan='7' class='p-4 text-center text-slate-400'>No records compiled inside this time range delta mapping state.</td></tr>"
        html += "</tbody></table></div>"
        return html

    body = """
    <div class="space-y-4">
        <div>
            <h1 class="text-3xl font-bold text-slate-800">System Time Frame Summaries</h1>
            <p class="text-sm text-slate-500">Review standard dynamic breakdowns mapped by date intervals.</p>
        </div>
    """
    body += render_report_table("Daily Metric Report Insights", daily)
    body += render_report_table("Extended Weekly Performance Matrix", weekly)
    body += render_report_table("Rolling Monthly Aggregated Logs Summary", monthly)
    body += "</div>"
    return page_wrapper("Reports Overview", body, is_admin=True)


# =========================================================
# BEAUTIFUL ORGANIZED TEACHER DASHBOARD PORTAL WITH SIDEBAR
# =========================================================
@app.route("/teacher")
def teacher_dashboard():
    protect = teacher_required()
    if protect:
        return protect

    teacher_id = get_logged_teacher_id()
    teacher_name = session.get("teacher_name", "Instructor")
    school_name = session.get("school_name", "")
    classes = get_teacher_classes(teacher_id)
    attendance = get_attendance_for_teacher(teacher_name)

    body = f"""
    <div class="space-y-6">
        <div class="bg-gradient-to-r from-blue-700 to-indigo-800 p-6 rounded-2xl text-white shadow-sm flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div>
                <h1 class="text-3xl font-extrabold tracking-tight">Welcome Back, {teacher_name}!</h1>
                <p class="text-blue-100 text-sm mt-1">{f'🏫 {school_name} · ' if school_name else ''}Manage active classrooms, launch live face-recognition streams, or complete fast manual manual ticking sheets.</p>
            </div>
            <div class="bg-white/10 px-4 py-2 rounded-xl text-xs font-mono backdrop-blur-sm flex items-center gap-3">
                System Session Verified ✓
                <a href="/teacher/edit-profile" class="bg-white/20 hover:bg-white/30 px-3 py-1.5 rounded-lg font-sans font-semibold no-underline text-white text-xs transition">✏️ Edit Profile</a>
            </div>
        </div>

        <div>
            <h2 class="text-xl font-bold text-slate-800 mb-4 flex items-center gap-2"><i class="fas fa-chalkboard text-blue-600"></i> My Assigned Academic Classrooms</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
    """
    if classes:
        for c in classes:
            body += f"""
                <div class="bg-white border border-slate-200 rounded-2xl shadow-sm hover:shadow-md transition flex flex-col justify-between overflow-hidden">
                    <div class="p-5">
                        <div class="flex justify-between items-start mb-3">
                            <span class="text-xs font-bold uppercase tracking-wider bg-blue-50 text-blue-700 px-2.5 py-1 rounded-full">{c["department"] or "General"}</span>
                            <span class="text-xs font-mono text-slate-400">#CLS-{c["id"]}</span>
                        </div>
                        <h3 class="text-xl font-bold text-slate-800 mb-1 flex items-center gap-2">
                            {c["class_name"]}
                            <a href="/teacher/class/{c["id"]}/edit-name" title="Rename class" class="text-slate-300 hover:text-blue-600 text-sm transition"><i class="fas fa-pen"></i></a>
                        </h3>
                        <p class="text-sm font-medium text-slate-600 mb-2">{c["subject_name"] or "No Topic Component Attached"}</p>
                        <div class="text-xs text-slate-400 space-y-1">
                            <div><i class="fas fa-layer-group w-4 text-slate-300"></i> <b>Section Block:</b> {c["section_name"] or "N/A"}</div>
                            <div><i class="fas fa-bookmark w-4 text-slate-300"></i> <b>Course Code:</b> {c["course"] or "N/A"}</div>
                        </div>
                    </div>
                    
                    <div class="p-4 bg-slate-50 border-t border-slate-100 grid grid-cols-2 gap-2 text-center">
                        <a href="/teacher/class/{c["id"]}" class="bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold py-2 px-3 rounded-lg flex items-center justify-center gap-1.5 transition">
                            <i class="fas fa-list-check"></i> Tracker Sheet
                        </a>
                        <a href="/teacher/class/{c["id"]}/scan" class="bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold py-2 px-3 rounded-lg flex items-center justify-center gap-1.5 transition">
                            <i class="fas fa-camera"></i> AI Scan
                        </a>
                    </div>
                </div>
            """
    else:
        body += """
            <div class="col-span-full bg-white border border-dashed rounded-2xl p-8 text-center text-slate-400">
                <i class="fas fa-folder-open text-4xl mb-2 text-slate-300"></i>
                <p class="font-medium">No classroom matrix mapping objects assigned to your proctor account identifier.</p>
            </div>
        """
    body += f"""
            </div>
        </div>

        <div class="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
            <div class="p-4 bg-slate-50 border-b border-slate-100 flex justify-between items-center">
                <h3 class="font-bold text-slate-800 flex items-center gap-2"><i class="fas fa-history text-slate-500"></i> Recent Attendance Mappings Log</h3>
                <span class="text-xs text-slate-500 bg-white border px-2 py-1 rounded-lg">Proctor: {teacher_name}</span>
            </div>
            <div class="overflow-x-auto">
                <table class="w-full text-left m-0 border-none shadow-none rounded-none">
                    <thead>
                        <tr class="bg-slate-100/70 border-b text-slate-700 text-xs uppercase font-bold tracking-wider">
                            <th class="p-3 border-none bg-transparent text-slate-700 font-bold">Date Logged</th>
                            <th class="p-3 border-none bg-transparent text-slate-700 font-bold">Timestamp</th>
                            <th class="p-3 border-none bg-transparent text-slate-700 font-bold">Student ID</th>
                            <th class="p-3 border-none bg-transparent text-slate-700 font-bold">Full Name</th>
                            <th class="p-3 border-none bg-transparent text-slate-700 font-bold">Class Framework</th>
                            <th class="p-3 border-none bg-transparent text-slate-700 font-bold">Section</th>
                            <th class="p-3 border-none bg-transparent text-slate-700 font-bold">Verification Status</th>
                        </tr>
                    </thead>
                    <tbody class="text-xs divide-y divide-slate-100">
    """
    if attendance:
        for a in attendance:
            body += f"""
                <tr class="hover:bg-slate-50/80 transition-colors">
                    <td class="p-3 font-medium whitespace-nowrap">{a["date"]}</td>
                    <td class="p-3 text-slate-400">{format_time_12hr(a["time"])}</td>
                    <td class="p-3 font-mono font-bold text-slate-600">{a["student_id"]}</td>
                    <td class="p-3 font-semibold text-slate-800">{a["full_name"]}</td>
                    <td class="p-3 text-slate-600">{a["class_name"]}</td>
                    <td class="p-3 text-slate-500">{a["section_name"] or "—"}</td>
                    <td class="p-3"><span class="px-2.5 py-1 rounded-full text-[11px] font-extrabold {'bg-emerald-50 text-emerald-700 border border-emerald-200' if a['status']=='Present' else 'bg-rose-50 text-rose-700 border border-rose-200'}">{a["status"]}</span></td>
                </tr>
            """
    else:
        body += "<tr><td colspan='7' class='p-6 text-center text-slate-400 font-medium'>No records processed via your instructor portal session.</td></tr>"
    body += """
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    """
    return page_wrapper("Teacher Dashboard", body, is_teacher=True, teacher_name=teacher_name)


# =========================================================
# TEACHER — EDIT MY PROFILE (name + photo)
# =========================================================
@app.route("/teacher/edit-profile", methods=["GET", "POST"])
def teacher_edit_profile():
    protect = teacher_required()
    if protect:
        return protect

    teacher_id = get_logged_teacher_id()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM teachers WHERE id=%s", (teacher_id,))
    teacher = cur.fetchone()
    if not teacher:
        conn.close()
        return redirect("/teacher-logout")

    error = ""
    if request.method == "POST":
        teacher_name = request.form.get("teacher_name", "").strip()
        if not teacher_name:
            conn.close()
            error = "Name cannot be empty."
            if is_ajax(): return ajax_err(error)
        else:
            new_photo = teacher.get("photo_path") or ""
            photo_file = request.files.get("photo")
            if photo_file and photo_file.filename:
                safe_name = sanitize_filename(teacher_name)
                ext = photo_file.filename.rsplit(".", 1)[-1].lower() if "." in photo_file.filename else "jpg"
                new_filename = f"teacher_{teacher_id}_{safe_name}.{ext}"
                img_bytes = photo_file.read()
                public_url = None
                if not is_allowed_upload(img_bytes):
                    error = "Uploaded file is not a valid image."
                    if is_ajax():
                        conn.close()
                        return ajax_err(error)
                else:
                    public_url = supabase_upload(new_filename, img_bytes)
                if public_url:
                    old_img = teacher.get("photo_path") or ""
                    if old_img and old_img.startswith("http"):
                        supabase_delete(old_img.split("/")[-1])
                    new_photo = public_url

            cur.execute(
                "UPDATE teachers SET teacher_name=%s, photo_path=%s WHERE id=%s",
                (teacher_name, new_photo, teacher_id)
            )
            # Keep class displays in sync with the teacher's current name
            cur.execute("UPDATE classes SET teacher_display_name=%s WHERE teacher_id=%s", (teacher_name, teacher_id))
            conn.commit()
            conn.close()
            session["teacher_name"] = teacher_name
            session["teacher_photo"] = new_photo
            if is_ajax(): return ajax_ok("Profile updated successfully!", redirect_url="/teacher")
            return "<script>alert('Profile updated successfully!');window.location.href='/teacher';</script>"
    else:
        conn.close()

    photo_url = supabase_public_url(teacher.get("photo_path") or "") if teacher.get("photo_path") else ""
    avatar_html = (f'<img id="photoPreview" src="{photo_url}" class="w-20 h-20 rounded-full object-cover border-2 border-indigo-400 shadow">'
                   if photo_url else
                   '<div id="photoPreview" class="w-20 h-20 rounded-full bg-indigo-500 text-white flex items-center justify-center text-2xl font-bold border-2 border-indigo-400 shadow">'
                   f'{(teacher["teacher_name"] or "?")[0].upper()}</div>')

    body = f"""
    <div class="max-w-lg mx-auto">
        <h1 class="text-2xl font-bold text-slate-800 mb-1">Edit My Profile</h1>
        <p class="text-sm text-slate-500 mb-5">Update your display name or profile photo.</p>
        {'<div class="mb-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm font-semibold">' + error + '</div>' if error else ''}

        <div class="flex items-center gap-4 mb-6 p-4 bg-slate-50 rounded-xl border">
            {avatar_html}
            <div>
                <p class="font-semibold text-slate-700">{esc(teacher["teacher_name"])}</p>
                <p class="text-xs text-slate-400 font-mono">Username: {teacher["username"]}</p>
            </div>
        </div>

        <form method="POST" enctype="multipart/form-data" class="space-y-4" data-ajax>
                            {csrf_input_html()}
            <div>
                <label class="block text-sm font-semibold text-slate-700 mb-1">Display Name</label>
                <input type="text" name="teacher_name" value="{esc(teacher["teacher_name"])}" class="w-full px-3 py-2 border rounded-lg" required>
            </div>
            <div class="border rounded-xl p-4 space-y-3 bg-slate-50">
                <p class="text-sm font-semibold text-slate-700">Profile Photo</p>
                <label class="inline-block bg-blue-600 hover:bg-blue-700 text-white font-semibold px-4 py-2 rounded-lg cursor-pointer text-sm">
                    📁 Upload Photo
                    <input type="file" name="photo" accept="image/*" class="hidden" onchange="if(this.files&&this.files[0]){{document.getElementById('photoPreview').src=URL.createObjectURL(this.files[0]);}}">
                </label>
            </div>
            <div class="flex gap-2 pt-2">
                <button type="submit" class="bg-blue-600 text-white font-bold py-2 px-5 rounded-lg hover:bg-blue-700">Save Changes</button>
                <a href="/teacher" class="inline-block bg-slate-100 text-slate-700 font-bold py-2 px-5 rounded-lg hover:bg-slate-200">Cancel</a>
            </div>
        </form>
    </div>
    """
    return page_wrapper("Edit My Profile", body, is_teacher=True, teacher_name=teacher["teacher_name"])


# =========================================================
# TEACHER — RENAME MY OWN CLASS
# =========================================================
@app.route("/teacher/class/<int:class_id>/edit-name", methods=["GET", "POST"])
def teacher_edit_class_name(class_id):
    protect = teacher_required()
    if protect:
        return protect

    teacher_id = get_logged_teacher_id()
    class_row = get_class_by_id(class_id)
    if not class_row or class_row["teacher_id"] != teacher_id:
        if is_ajax(): return ajax_err("Not authorized for this class.")
        return "<script>alert('Not authorized for this class.');window.location.href='/teacher';</script>"

    error = ""
    if request.method == "POST":
        class_name = request.form.get("class_name", "").strip()
        if not class_name:
            error = "Class name cannot be empty."
            if is_ajax(): return ajax_err(error)
        else:
            conn = get_db()
            cur = conn.cursor()
            cur.execute("UPDATE classes SET class_name=%s WHERE id=%s AND teacher_id=%s", (class_name, class_id, teacher_id))
            conn.commit()
            conn.close()
            if is_ajax(): return ajax_ok("Class name updated!", redirect_url="/teacher")
            return "<script>alert('Class name updated!');window.location.href='/teacher';</script>"

    body = f"""
    <div class="max-w-md mx-auto">
        <h1 class="text-2xl font-bold text-slate-800 mb-1">Rename Class</h1>
        <p class="text-sm text-slate-500 mb-5">Update the display name for this classroom.</p>
        {'<div class="mb-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm font-semibold">' + error + '</div>' if error else ''}
        <form method="POST" class="space-y-4" data-ajax>
                            {csrf_input_html()}
            <div>
                <label class="block text-sm font-semibold text-slate-700 mb-1">Class Name</label>
                <input type="text" name="class_name" value="{class_row['class_name']}" class="w-full px-3 py-2 border rounded-lg" required autofocus>
            </div>
            <div class="flex gap-2 pt-2">
                <button type="submit" class="bg-blue-600 text-white font-bold py-2 px-5 rounded-lg hover:bg-blue-700">Save</button>
                <a href="/teacher" class="inline-block bg-slate-100 text-slate-700 font-bold py-2 px-5 rounded-lg hover:bg-slate-200">Cancel</a>
            </div>
        </form>
    </div>
    """
    return page_wrapper("Rename Class", body, is_teacher=True, teacher_name=session.get("teacher_name"))


# =========================================================
# MANUAL ATTENDANCE ROSTER SHEET CONTROL WITH BATCH CLICK SUBMIT
# =========================================================
@app.route("/teacher/class/<int:class_id>")
def teacher_view_class(class_id):
    protect = teacher_required()
    if protect:
        return protect
    class_row = get_class_by_id(class_id)
    if not class_row:
        return "Class mismatch mapping object reference layer", 404
    if class_row["teacher_id"] != get_logged_teacher_id():
        return "Unauthorized proctor routing attempt override locked", 403

    students = get_students_in_class(class_id)
    today_str = datetime.now().strftime("%B %d, %Y")
    today_iso = datetime.now().strftime("%Y-%m-%d")

    # Look up what's already recorded today so the sheet reflects reality —
    # e.g. a student who self-checked-in shows as Present (with a badge),
    # and the teacher can still uncheck them to override to Absent.
    today_status = {
        row["student_id"]: row for row in get_attendance_for_class(class_id)
        if row["date"] == today_iso
    }

    body = f"""
    <div class="space-y-6">
        <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 bg-white p-4 border border-slate-200 rounded-xl shadow-sm">
            <div>
                <a href="/teacher" class="text-xs font-bold text-blue-600 hover:underline flex items-center gap-1 mb-1"><i class="fas fa-arrow-left"></i> Back to Dashboard Panel</a>
                <h1 class="text-2xl font-extrabold text-slate-800 tracking-tight">Roster Tracker: {class_row["class_name"]}</h1>
                <p class="text-xs text-slate-400 font-medium mt-0.5">Subject code: {class_row["subject_name"] or "N/A"} | Active Proctor Date: {today_str}</p>
            </div>
            <div class="flex gap-2 flex-wrap">
                <a class="bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold py-2.5 px-4 rounded-xl flex items-center gap-1.5 transition shadow-sm" href="/teacher/session-panel/{class_id}">
                    ⏱️ Attendance Timer
                </a>
                <a class="bg-purple-600 hover:bg-purple-700 text-white text-xs font-bold py-2.5 px-4 rounded-xl flex items-center gap-1.5 transition shadow-sm" href="/teacher/class/{class_id}/feed">
                    💬 Class Chat
                </a>
                <a class="bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold py-2.5 px-4 rounded-xl flex items-center gap-1.5 transition shadow-sm shadow-emerald-100" href="/teacher/class/{class_id}/scan">
                    <i class="fas fa-camera"></i> Launch AI Scanner
                </a>
            </div>
        </div>

        <div class="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
            <div class="px-5 py-4 bg-slate-50 border-b border-slate-100 flex justify-between items-center">
                <h3 class="font-bold text-slate-700 text-sm flex items-center gap-2"><i class="fas fa-user-check text-slate-400"></i> Interactive Manual Ticking Sheet</h3>
                <div class="flex items-center gap-3 text-xs">
                    <button onclick="checkAll(true)" type="button" class="text-blue-600 hover:text-blue-800 font-semibold bg-white px-2 py-1 rounded border shadow-sm">Mark All Present</button>
                    <span class="text-slate-300">|</span>
                    <button onclick="checkAll(false)" type="button" class="text-slate-500 hover:text-slate-700 font-semibold bg-white px-2 py-1 rounded border shadow-sm">Clear All</button>
                </div>
            </div>

            <form id="attendanceForm" method="POST" action="/teacher/class/{class_id}/manual-submit" data-ajax>
                <div class="divide-y divide-slate-100">
    """
    if students:
        for idx, s in enumerate(students, 1):
            pct = get_percentage(s["student_id"], class_id)
            existing_today = today_status.get(s["student_id"])
            # Default to checked (Present) if there's no record yet, otherwise
            # reflect whatever is actually recorded for today.
            is_checked = existing_today is None or existing_today["status"] == "Present"
            checked_attr = "checked" if is_checked else ""
            self_marked_badge = ""
            if existing_today is not None and existing_today["status"] == "Present":
                self_marked_badge = f' <span class="text-emerald-600 font-bold">· recorded {format_time_12hr(existing_today["time"])}</span>'
            elif existing_today is not None and existing_today["status"] == "Absent":
                self_marked_badge = ' <span class="text-rose-500 font-bold">· recorded Absent</span>'
            body += f"""
                    <div class="p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4 hover:bg-slate-50/50 transition-colors">
                        <div class="flex items-center gap-4">
                            <span class="text-xs font-mono font-bold text-slate-300 w-5 text-center">{idx:02d}</span>
                            <img class="w-10 h-10 object-cover rounded-xl border bg-slate-50" src="{supabase_public_url(s["image_file"])}">
                            <div>
                                <h4 class="font-bold text-slate-800 text-sm">{s["full_name"]}</h4>
                                <p class="text-[11px] font-mono text-slate-400">ID: #{s["student_id"]} | Agg. Ratio Score: <span class="text-blue-600 font-bold">{pct}%</span>{self_marked_badge}</p>
                            </div>
                        </div>

                        <div class="flex items-center gap-6">
                            <label class="relative inline-flex items-center cursor-pointer select-none">
                                <input type="checkbox" name="present_students" value="{s["student_id"]}" class="sr-only peer" {checked_attr}>
                                <div class="w-14 h-7 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[4px] after:left-[4px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-emerald-500"></div>
                                <span class="ml-3 text-xs font-bold text-slate-500 peer-checked:text-emerald-600 uppercase tracking-wider w-16">Present</span>
                            </label>
                        </div>
                    </div>
            """
    else:
        body += """
                    <div class="p-8 text-center text-slate-400 font-medium">
                        <i class="fas fa-users-slash text-3xl mb-1 text-slate-300"></i>
                        <p>No student accounts currently assigned to this classroom registry array block.</p>
                    </div>
        """
    body += """
                </div>

                <div class="p-5 bg-slate-50/60 border-t border-slate-100 flex justify-end">
                    <button type="submit" class="bg-indigo-600 hover:bg-indigo-700 text-white font-bold px-6 py-2.5 rounded-xl shadow-md shadow-indigo-100 flex items-center gap-2 transition-all">
                        <i class="fas fa-save text-xs"></i>
                        <span>Submit Attendance Sheet</span>
                    </button>
                </div>
            </form>
        </div>
    </div>

    <script>
    function checkAll(status) {
        const checkboxes = document.querySelectorAll('input[name="present_students"]');
        checkboxes.forEach(cb => cb.checked = status);
    }
    </script>
    """
    return page_wrapper("Classroom Ticking Roster Sheet", body, is_teacher=True, teacher_name=session.get("teacher_name"))


# =========================================================
# MANUAL BATCH TICKING SUBMISSION HANDLER ROUTE ACTION
# =========================================================
@app.route("/teacher/class/<int:class_id>/manual-submit", methods=["POST"])
def teacher_manual_submit_batch(class_id):
    protect = teacher_required()
    if protect:
        return protect
    class_row = get_class_by_id(class_id)
    if not class_row:
        return "Class file record mapping object mismatch context layer", 404
    if class_row["teacher_id"] != get_logged_teacher_id():
        return "Action forbidden access blocked context matrix", 403

    students = get_students_in_class(class_id)
    ticked_present_ids = request.form.getlist("present_students")

    for s in students:
        status = "Present" if s["student_id"] in ticked_present_ids else "Absent"
        mark_attendance(s, class_row, status=status, force=True)

    if is_ajax(): return ajax_ok("Attendance submitted successfully!", redirect_url="/teacher")
    return "<script>alert('Attendance roster processing batch committed successfully!'); window.location.href='/teacher';</script>"


# =========================================================
# BACKWARDS COMPATIBLE FORCE PILL OVERRIDES 
# =========================================================
@app.route("/teacher/manual-mark/<int:class_id>/<int:student_db_id>/<status>")
def teacher_manual_mark_override(class_id, student_db_id, status):
    protect = teacher_required()
    if protect:
        return protect
    class_row = get_class_by_id(class_id)
    if not class_row:
        return "Class mismatch row array object definition state error", 404
    if class_row["teacher_id"] != get_logged_teacher_id():
        return "Unauthorized action trigger attempt configuration locked", 403

    student_row = get_student_row_by_db_id(student_db_id)
    if student_row:
        mark_attendance(student_row, class_row, status, force=True)
    return f"<script>alert('Manually forced student record to {status}!');window.location.href='/teacher/class/{class_id}';</script>"


# =========================================================
# FACE SCAN LIVE CAMERA STREAM MODULE VISUAL ENGINE
# =========================================================
@app.route("/teacher/class/<int:class_id>/scan")
def teacher_scan_class(class_id):
    protect = teacher_required()
    if protect:
        return protect
    class_row = get_class_by_id(class_id)
    if not class_row:
        return "Class asset record code definition state error instance missing", 404
    if class_row["teacher_id"] != get_logged_teacher_id():
        return "Proctor verification sequence match locked mismatch exception", 403

    teacher_name = session.get("teacher_name", "Teacher Proctored Session")

    body = f"""
    <div class="max-w-3xl mx-auto text-center space-y-4">
        <div>
            <h1 class="text-3xl font-extrabold text-slate-800">📸 Automatic AI Face Recognition</h1>
            <p class="text-sm text-slate-500 mt-1">Live frame parser sequence linked to <b>{class_row["class_name"]}</b></p>
        </div>

        <video id="video" autoplay playsinline muted class="w-full max-w-lg mx-auto bg-black border border-slate-300 rounded-2xl shadow-lg"></video>
        
        <div id="result" class="text-2xl font-bold text-emerald-600 mt-4 tracking-tight animate-pulse">Scanning feed state framework...</div>
        <div id="status" class="text-xs font-semibold text-slate-400">Please provide camera hardware layout authorizations</div>
        
        <div class="flex justify-center gap-2 flex-wrap pt-2">
            <button class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-xl" onclick="switchCamera()">Switch Camera Orientation</button>
            <button class="bg-orange-500 hover:bg-orange-600 text-white font-bold py-2 px-4 rounded-xl" onclick="startCamera()">Reset Stream Connection</button>
            <a class="bg-slate-800 hover:bg-slate-900 text-white font-bold py-2 px-4 rounded-xl" href="/teacher/class/{class_id}">Return to Tracker Sheet</a>
        </div>
    </div>

<script>
const video = document.getElementById('video');
const resultDiv = document.getElementById('result');
const statusDiv = document.getElementById('status');
let currentFacingMode = "user";
let stream = null;
let intervalId = null;
let ambiguousActive = false;

async function startCamera() {{
    try {{
        statusDiv.innerText = "Starting camera...";
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {{
            statusDiv.innerText = "Camera not supported. Please use HTTPS and a modern browser.";
            return;
        }}
        if (stream) {{
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }}
        let constraints = [
            {{ video: {{ facingMode: {{ exact: currentFacingMode }}, width: {{ ideal: 640 }}, height: {{ ideal: 480 }} }}, audio: false }},
            {{ video: {{ facingMode: {{ ideal: currentFacingMode }}, width: {{ ideal: 640 }}, height: {{ ideal: 480 }} }}, audio: false }},
            {{ video: true, audio: false }}
        ];
        let lastErr = null;
        for (let c of constraints) {{
            try {{ stream = await navigator.mediaDevices.getUserMedia(c); break; }}
            catch (e) {{ lastErr = e; stream = null; }}
        }}
        if (!stream) throw lastErr;
        video.srcObject = stream;
        await new Promise((resolve) => {{ video.onloadedmetadata = () => resolve(); }});
        try {{
            await video.play();
            statusDiv.innerText = "✅ Camera ready (" + (currentFacingMode === "user" ? "Front" : "Back") + ")";
            startScanningLoops();
        }} catch (e) {{
            statusDiv.innerText = "Tap the video to start";
        }}
    }} catch (err) {{
        if (err.name === "NotAllowedError") {{ statusDiv.innerText = "❌ Camera permission denied. Please allow camera access and reload."; }}
        else if (err.name === "NotFoundError") {{ statusDiv.innerText = "❌ No camera found on this device."; }}
        else if (location.protocol !== "https:") {{ statusDiv.innerText = "❌ Camera requires HTTPS."; }}
        else {{ statusDiv.innerText = "❌ Camera error: " + err.message; }}
    }}
}}

function switchCamera() {{
    currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
    startCamera();
}}

function startScanningLoops() {{
    if(intervalId) clearInterval(intervalId);
    intervalId = setInterval(async () => {{
        if(video.paused || video.ended) return;
        if(ambiguousActive) return;  // pause scanning while teacher resolves a twin match
        try {{
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const image = canvas.toDataURL('image/jpeg');

            const res = await fetch('/teacher/class/{class_id}/scan-frame', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{ image: image }})
            }});
            const data = await res.json();
            if (data.ambiguous && Array.isArray(data.candidates) && data.candidates.length) {{
                showAmbiguousPicker(data.candidates, data.message);
            }} else if (data.name && data.name !== "Unknown") {{
                resultDiv.innerText = "🎉 " + data.name;
                statusDiv.innerText = data.message || "Match successfully logged.";
            }} else {{
                resultDiv.innerText = "Scanning framework loops state...";
            }}
        }} catch(err) {{
            console.log("Parsing framework error exception captured context loops", err);
        }}
    }}, 2000);
}}

function showAmbiguousPicker(candidates, message) {{
    ambiguousActive = true;
    statusDiv.innerText = message || "Multiple close matches found.";
    resultDiv.innerText = "⏸ Paused — confirm identity below";
    const picker = document.getElementById('ambiguousPicker');
    const list = document.getElementById('ambiguousCandidates');
    list.innerHTML = '';
    candidates.forEach(c => {{
        const enrolledNote = c.enrolled ? '' : ' <span class="text-rose-500">(not enrolled in this class)</span>';
        const btn = document.createElement('button');
        btn.className = "w-full text-left bg-white border border-amber-300 hover:border-amber-500 hover:bg-amber-100 rounded-xl px-4 py-2.5 font-semibold text-slate-800 text-sm transition";
        btn.innerHTML = c.full_name + ' <span class="text-slate-400 font-mono text-xs">(' + c.student_id + ')</span>' + enrolledNote;
        btn.onclick = () => confirmAmbiguous(c.db_id);
        list.appendChild(btn);
    }});
    picker.classList.remove('hidden');
}}

async function confirmAmbiguous(dbId) {{
    statusDiv.innerText = "Confirming...";
    try {{
        const res = await fetch('/teacher/class/{class_id}/scan-confirm', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{ db_id: dbId }})
        }});
        const data = await res.json();
        if (data.success) {{
            resultDiv.innerText = "🎉 " + data.name;
            statusDiv.innerText = data.message;
        }} else {{
            resultDiv.innerText = "Scanning framework loops state...";
            statusDiv.innerText = data.message || "Could not confirm identity.";
        }}
    }} catch(err) {{
        statusDiv.innerText = "Error confirming identity.";
    }}
    cancelAmbiguous();
}}

function cancelAmbiguous() {{
    ambiguousActive = false;
    document.getElementById('ambiguousPicker').classList.add('hidden');
    resultDiv.innerText = "Scanning framework loops state...";
}}

window.addEventListener('beforeunload', () => {{
    if(intervalId) clearInterval(intervalId);
    if(stream) stream.getTracks().forEach(t => t.stop());
}});

startCamera();
</script>
    """
    return page_wrapper("Face Scanner Live Canvas Pipeline Engine", body, is_teacher=True, teacher_name=teacher_name)


@app.route("/teacher/class/<int:class_id>/scan-frame", methods=["POST"])
def teacher_scan_frame_matrix_lookup(class_id):
    try:
        protect = teacher_required()
        if protect:
            return jsonify({"name": "Unknown", "message": "Authentication token missing"})

        class_row = get_class_by_id(class_id)
        if not class_row or class_row["teacher_id"] != get_logged_teacher_id():
            return jsonify({"name": "Unknown", "message": "Context scope mapping target mismatch access denied"})

        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"name": "Unknown"})

        image_data = data["image"]
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_data = image_data.replace(" ", "+")
        missing_padding = len(image_data) % 4
        if missing_padding:
            image_data += "=" * (4 - missing_padding)

        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"name": "Unknown"})

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # NOTE: this previously called the `face_recognition` library, which is
        # never imported in this app (it depends on dlib, a heavy native build
        # that's unreliable on most lightweight cloud hosts). That made this
        # route silently fail and always return "Unknown". It now uses the same
        # MediaPipe landmark-embedding engine that already powers student
        # self-check-in, so it actually works.
        face_embedding = _get_face_embedding(rgb_frame)

        if face_embedding is None:
            return jsonify({"name": "Unknown", "message": "No face detected — center your face in frame."})

        if len(known_encodings) == 0:
            return jsonify({"name": "Unknown", "message": "No registered faces found."})

        best_match_index = None
        best_dist = float("inf")
        second_match_index = None
        second_dist = float("inf")
        for i, known_emb in enumerate(known_encodings):
            matched, dist = _compare_embeddings(known_emb, face_embedding, tolerance=0.6)
            if matched and dist < best_dist:
                second_match_index = best_match_index
                second_dist = best_dist
                best_match_index = i
                best_dist = dist
            elif matched and dist < second_dist:
                second_match_index = i
                second_dist = dist

        if best_match_index is None:
            return jsonify({"name": "Unknown"})

        # ── TWIN / LOOKALIKE AMBIGUITY CHECK ──
        # If the runner-up match is nearly as close as the best match (and it's a
        # genuinely different student), the face alone can't reliably tell them
        # apart — e.g. twins or siblings. Rather than silently logging the closest
        # guess, surface both names so the teacher can pick the right one.
        AMBIGUITY_MARGIN = 0.12
        if (second_match_index is not None
                and known_students[second_match_index]["db_id"] != known_students[best_match_index]["db_id"]
                and (second_dist - best_dist) < AMBIGUITY_MARGIN):

            candidates = []
            seen_ids = set()
            for idx in (best_match_index, second_match_index):
                cand = known_students[idx]
                if cand["db_id"] in seen_ids:
                    continue
                seen_ids.add(cand["db_id"])
                candidates.append({
                    "db_id": cand["db_id"],
                    "student_id": cand["student_id"],
                    "full_name": cand["full_name"],
                    "enrolled": student_belongs_to_class(cand["db_id"], class_id)
                })

            return jsonify({
                "name": "Unknown",
                "ambiguous": True,
                "message": "This face closely matches more than one student (e.g. twins/siblings). Select the correct name.",
                "candidates": candidates
            })

        student = known_students[best_match_index]
        if not student_belongs_to_class(student["db_id"], class_id):
            return jsonify({
                "name": f"{student['student_id']} - {student['full_name']}",
                "message": "Recognized, but student is NOT enrolled in this specific roster mapping template"
            })

        student_row = get_student_row_by_db_id(student["db_id"])
        mark_attendance(student_row, class_row, "Present")
        return jsonify({
            "name": f"{student['student_id']} - {student['full_name']}",
            "message": f"Attendance verified & logged successfully into database for {class_row['class_name']}"
        })
    except Exception as e:
        print("SCAN BATCH ENCODING VECTOR LOOKUP EXCEPTION METRIC ERROR:", e)
        return jsonify({"name": "Unknown", "message": "Lookup processing matrix iteration break exception standard error"})


@app.route("/teacher/class/<int:class_id>/scan-confirm", methods=["POST"])
def teacher_scan_confirm_identity(class_id):
    """
    Called when the teacher resolves an ambiguous (twin/lookalike) match by
    picking the correct student name from the candidate list shown for
    scan-frame's "ambiguous" response. Marks that specific student Present.
    """
    try:
        protect = teacher_required()
        if protect:
            return jsonify({"success": False, "message": "Authentication token missing"})

        class_row = get_class_by_id(class_id)
        if not class_row or class_row["teacher_id"] != get_logged_teacher_id():
            return jsonify({"success": False, "message": "Context scope mapping target mismatch access denied"})

        data = request.get_json(silent=True) or {}
        db_id = data.get("db_id")
        try:
            db_id = int(db_id)
        except (TypeError, ValueError):
            return jsonify({"success": False, "message": "Invalid student selection."})

        if not student_belongs_to_class(db_id, class_id):
            return jsonify({"success": False, "message": "Selected student is not enrolled in this class."})

        student_row = get_student_row_by_db_id(db_id)
        if not student_row:
            return jsonify({"success": False, "message": "Student not found."})

        mark_attendance(student_row, class_row, "Present")
        return jsonify({
            "success": True,
            "name": f"{student_row['student_id']} - {student_row['full_name']}",
            "message": f"Attendance verified & logged successfully into database for {class_row['class_name']}"
        })
    except Exception as e:
        print("teacher_scan_confirm_identity error:", e)
        return jsonify({"success": False, "message": "Internal error confirming identity."})


# =========================================================
# PUBLIC QR CHECK-IN LANDING PAGE
# =========================================================
@app.route("/checkin/<code>")
def qr_checkin_landing(code):
    """
    Students land here after scanning the QR code.
    - If logged in: auto-fills session code and redirects to their check-in page.
    - If not logged in: redirects to student login with code saved in session.
    """
    code = code.upper().strip()
    # Find the class this code belongs to
    school_id = session.get("school_id", 1)
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT class_id FROM class_sessions
            WHERE code=%s AND active=TRUE AND expires_at > NOW()
            ORDER BY id DESC LIMIT 1
        """, (code,))
        row = cur.fetchone()
        conn.close()
    except Exception as e:
        print("qr_checkin_landing DB error:", repr(e), flush=True)
        return page_wrapper("QR Check-In", f"""
        <div class="max-w-md mx-auto text-center py-16 space-y-4">
            <div class="text-6xl">⚠️</div>
            <h1 class="text-2xl font-bold text-red-600">Something went wrong</h1>
            <p class="text-slate-500">Could not verify the QR code right now. Please try again in a moment.</p>
            <a href="/student" class="inline-block mt-4 bg-slate-800 text-white font-bold px-6 py-2.5 rounded-xl">Go to My Dashboard</a>
        </div>
        """)

    if not row:
        # Code expired or invalid
        msg = "This QR code has expired or is invalid. Ask your teacher to generate a new one."
        return page_wrapper("QR Check-In", f"""
        <div class="max-w-md mx-auto text-center py-16 space-y-4">
            <div class="text-6xl">⏰</div>
            <h1 class="text-2xl font-bold text-red-600">Code Expired</h1>
            <p class="text-slate-500">{msg}</p>
            <a href="/student" class="inline-block mt-4 bg-slate-800 text-white font-bold px-6 py-2.5 rounded-xl">Go to My Dashboard</a>
        </div>
        """)

    class_id = row["class_id"]

    # Save code in session so it survives a login redirect
    session["qr_session_code"] = code
    session["qr_class_id"] = class_id

    if not is_student_logged_in():
        return redirect(f"/student-login?next=/checkin/{code}")

    # Student is logged in — redirect straight to their check-in page with code pre-filled
    return redirect(f"/student/checkin/manual/{class_id}?code={code}")


# =========================================================
# ADDITIONS: STUDENT SELF ATTENDANCE (MANUAL & FACE CHECK-IN)
# =========================================================
@app.route("/student/checkin/manual/<int:class_id>", methods=["GET", "POST"])
def student_manual_checkin(class_id):
    protect = student_required()
    if protect:
        return protect

    student_db_id = get_logged_student_db_id()
    school_id = get_current_school_id()

    if not student_belongs_to_class(student_db_id, class_id):
        return "<script>alert('Error: You are not enrolled in this class.'); window.location.href='/student';</script>"

    student_row = get_student_row_by_db_id(student_db_id)
    class_row = get_class_by_id(class_id)

    if not student_row or not class_row:
        return "<script>alert('Error: Student or class not found.'); window.location.href='/student';</script>"

    # Attendance can only be marked while the teacher has an active session running.
    if not is_attendance_session_active(class_id, school_id):
        if request.method == "POST" and is_ajax():
            return jsonify({"ok": False, "message": "Attendance hasn't started yet. Wait for your teacher to start the session."})
        body = f"""
        <div class="max-w-md mx-auto space-y-5 text-center">
            <div class="text-5xl">⏳</div>
            <h1 class="text-2xl font-bold text-slate-800">Attendance Not Started</h1>
            <p class="text-sm text-slate-500">
                Your teacher hasn't started attendance for <strong>{class_row['class_name']}</strong> yet.
                You can't mark attendance until they begin the session.
            </p>
            <a href="/student" class="inline-block bg-blue-600 hover:bg-blue-700 text-white font-bold py-2.5 px-5 rounded-xl transition text-sm">
                ← Back to Dashboard
            </a>
        </div>
        """
        return page_wrapper(f"Attendance Not Started — {class_row['class_name']}", body, is_student=True, student_context=student_row)

    # GET — show the check-in form (collect GPS + session code)
    if request.method == "GET":
        # Pre-fill code from QR scan (URL param or saved in session)
        prefill_code = request.args.get("code", "") or session.pop("qr_session_code", "")
        body = f"""
        <div class="max-w-md mx-auto space-y-5">
            <h1 class="text-2xl font-bold text-slate-800">📍 Mark Attendance</h1>
            <p class="text-sm text-slate-500">Class: <strong>{class_row['class_name']}</strong></p>
            <div class="bg-amber-50 border border-amber-200 rounded-xl p-4 text-sm text-amber-800">
                ⚠️ At least one verification must pass: <strong>GPS location</strong>, <strong>session code</strong>, or <strong>school WiFi</strong>.
            </div>
            <form id="checkinForm" method="POST" class="space-y-4 bg-white border rounded-xl p-5 shadow-sm">
                {csrf_input_html()}
                <input type="hidden" name="latitude" id="lat">
                <input type="hidden" name="longitude" id="lng">

                <div>
                    <label class="block text-sm font-semibold text-slate-700 mb-1">📟 Session Code <span class="font-normal text-slate-400">(from your teacher or QR scan)</span></label>
                    <input type="text" name="session_code" id="sessionCodeField" value="{prefill_code}" placeholder="e.g. A4X9" maxlength="10"
                        class="w-full px-3 py-2 border rounded-lg uppercase tracking-widest font-mono text-lg text-center focus:ring-2 focus:ring-blue-400 {'border-emerald-400 bg-emerald-50' if prefill_code else ''}">
                    {'<p class="text-xs text-emerald-600 font-semibold text-center mt-1">✅ Code filled from QR scan</p>' if prefill_code else ''}
                </div>

                <div id="gpsStatus" class="text-xs text-slate-400 text-center">📡 Detecting your location...</div>

                <button type="submit" id="checkinSubmitBtn" class="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-3 rounded-xl text-base transition">
                    ✅ Submit Attendance
                </button>
                <a href="/student" class="block text-center text-sm text-slate-400 hover:underline mt-1">Cancel</a>
            </form>

            <div id="checkinResult" class="hidden rounded-xl p-5 border-2 text-center space-y-1"></div>
        </div>
        <script>
        // ── GPS capture: watch for a fix and keep the best one until accuracy is good ──
        let _bestAcc = Infinity;
        let _gpsWatchId = null;
        const _GPS_GOOD_ENOUGH_M = 50;
        const _GPS_MAX_WAIT_MS = 20000;

        function _updateGpsStatus(acc) {{
            const el = document.getElementById('gpsStatus');
            if (!el) return;
            el.innerText = '✅ GPS: ' + document.getElementById('lat').value + ', ' + document.getElementById('lng').value + ' (±' + Math.round(acc) + 'm)';
            el.className = acc <= _GPS_GOOD_ENOUGH_M ? 'text-xs text-emerald-600 text-center font-semibold' : 'text-xs text-amber-500 text-center font-semibold';
        }}

        if (navigator.geolocation) {{
            document.getElementById('gpsStatus').innerText = '🔄 Acquiring GPS fix…';
            _gpsWatchId = navigator.geolocation.watchPosition(function(pos) {{
                const acc = pos.coords.accuracy;
                if (acc < _bestAcc) {{
                    _bestAcc = acc;
                    document.getElementById('lat').value = pos.coords.latitude;
                    document.getElementById('lng').value = pos.coords.longitude;
                    _updateGpsStatus(acc);
                    if (acc <= _GPS_GOOD_ENOUGH_M) {{
                        navigator.geolocation.clearWatch(_gpsWatchId);
                    }}
                }}
            }}, function(err) {{
                document.getElementById('gpsStatus').innerText = '⚠️ GPS unavailable — use session code or school WiFi.';
                document.getElementById('gpsStatus').className = 'text-xs text-amber-600 text-center font-semibold';
            }}, {{ enableHighAccuracy: true, timeout: _GPS_MAX_WAIT_MS, maximumAge: 0 }});
            setTimeout(function() {{ if (_gpsWatchId !== null) navigator.geolocation.clearWatch(_gpsWatchId); }}, _GPS_MAX_WAIT_MS);
        }} else {{
            document.getElementById('gpsStatus').innerText = '⚠️ GPS not supported on this device.';
        }}

        document.getElementById('checkinForm').addEventListener('submit', function(e) {{
            e.preventDefault();
            const form = e.target;
            const btn = document.getElementById('checkinSubmitBtn');
            const resultBox = document.getElementById('checkinResult');
            btn.disabled = true;
            btn.textContent = 'Submitting…';

            fetch(form.action || window.location.href, {{
                method: 'POST',
                body: new FormData(form),
                headers: {{ 'X-Requested-With': 'XMLHttpRequest' }}
            }})
            .then(r => r.json())
            .then(data => {{
                form.classList.add('hidden');
                resultBox.classList.remove('hidden');
                if (data.ok) {{
                    const isInfo = data.already_marked;
                    resultBox.className = 'rounded-xl p-5 border-2 text-center space-y-1 ' +
                        (isInfo ? 'bg-blue-50 border-blue-200' : 'bg-emerald-50 border-emerald-200');
                    resultBox.innerHTML =
                        '<div class="text-3xl">' + (isInfo ? 'ℹ️' : '✅') + '</div>' +
                        '<div class="font-bold ' + (isInfo ? 'text-blue-800' : 'text-emerald-800') + '">' + data.message + '</div>' +
                        '<div class="text-xs text-slate-400 mt-2">Redirecting to your dashboard…</div>';
                }} else {{
                    resultBox.className = 'rounded-xl p-5 border-2 text-center space-y-1 bg-rose-50 border-rose-200';
                    resultBox.innerHTML =
                        '<div class="text-3xl">❌</div>' +
                        '<div class="font-bold text-rose-800">' + (data.message || 'Could not verify attendance.') + '</div>' +
                        '<div class="text-xs text-slate-400 mt-2">Redirecting to your dashboard…</div>';
                }}
                setTimeout(() => {{ window.location.href = '/student'; }}, 2200);
            }})
            .catch(() => {{
                btn.disabled = false;
                btn.textContent = '✅ Submit Attendance';
                resultBox.classList.remove('hidden');
                resultBox.className = 'rounded-xl p-5 border-2 text-center space-y-1 bg-rose-50 border-rose-200';
                resultBox.innerHTML = '<div class="text-3xl">⚠️</div><div class="font-bold text-rose-800">Network error — please try again.</div>';
            }});
        }});
        </script>
        """
        return page_wrapper(f"Mark Attendance — {class_row['class_name']}", body, is_student=True, student_context=student_row)

    # POST — validate and mark
    lat = request.form.get("latitude") or None
    lng = request.form.get("longitude") or None
    session_code = request.form.get("session_code", "").strip()

    try:
        lat_f = float(lat) if lat else None
        lng_f = float(lng) if lng else None
    except:
        lat_f = lng_f = None

    passed, reason, dist_m = run_location_checks(lat_f, lng_f, session_code, class_id, school_id, request)

    if passed:
        dist_str = f' ({int(dist_m)}m from teacher)' if dist_m is not None else ''
        result = mark_attendance(student_row, class_row, 'Present', student_lat=lat_f, student_lng=lng_f, distance_meters=dist_m)
        if result.get('already_marked'):
            msg = "ℹ️ Attendance already recorded as Present today. No changes made."
            if is_ajax(): return jsonify({"ok": True, "already_marked": True, "message": msg})
            return f"<script>alert('{msg}'); window.location.href='/student';</script>"
        msg = f"Attendance marked Present! ({reason}){dist_str}"
        if is_ajax(): return jsonify({"ok": True, "already_marked": False, "message": msg})
        return f"<script>alert('✅ {msg}'); window.location.href='/student';</script>"
    else:
        result = mark_attendance(student_row, class_row, 'Absent', student_lat=lat_f, student_lng=lng_f, distance_meters=dist_m)
        if result.get('already_marked'):
            msg = "ℹ️ Attendance already recorded as Present today. No changes made."
            if is_ajax(): return jsonify({"ok": True, "already_marked": True, "message": msg})
            return f"<script>alert('{msg}'); window.location.href='/student';</script>"
        msg = f"Marked Absent — none of the verification checks passed. Reasons: {reason}"
        if is_ajax(): return jsonify({"ok": False, "message": msg})
        return f"<script>alert('❌ {msg}'); window.location.href='/student';</script>"



@app.route("/student/checkin/api", methods=["POST"])
def student_checkin_api():
    """JSON endpoint used by the in-app QR scanner — no page reload."""
    try:
        protect = student_required()
        if protect:
            return jsonify({"ok": False, "error": "Not logged in."})

        student_db_id = get_logged_student_db_id()
        school_id = get_current_school_id()

        data = request.get_json(silent=True) or {}
        code = data.get("session_code", "").strip().upper()
        lat_raw = data.get("latitude")
        lng_raw = data.get("longitude")
        gps_acc = data.get("gps_accuracy")

        # Resolve class from session code
        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute("""
                SELECT class_id FROM class_sessions
                WHERE code=%s AND school_id=%s AND active=TRUE AND expires_at > NOW()
                ORDER BY id DESC LIMIT 1
            """, (code, school_id))
            row = cur.fetchone()
            conn.close()
        except Exception as e:
            return jsonify({"ok": False, "error": "Database error: " + str(e)})

        if not row:
            return jsonify({"ok": False, "error": "QR code has expired or is invalid. Ask your teacher to refresh it."})

        class_id = row["class_id"]

        if not is_attendance_session_active(class_id, school_id):
            return jsonify({"ok": False, "error": "Attendance hasn't started yet. Ask your teacher to start the session."})

        if not student_belongs_to_class(student_db_id, class_id):
            return jsonify({"ok": False, "error": "You are not enrolled in this class."})

        student_row = get_student_row_by_db_id(student_db_id)
        class_row = get_class_by_id(class_id)

        if not student_row or not class_row:
            return jsonify({"ok": False, "error": "Student or class not found."})

        try:
            lat_f = float(lat_raw) if lat_raw is not None else None
            lng_f = float(lng_raw) if lng_raw is not None else None
            if gps_acc is not None and float(gps_acc) > 200:
                lat_f = lng_f = None
        except:
            lat_f = lng_f = None

        passed, reason, dist_m = run_location_checks(lat_f, lng_f, code, class_id, school_id, request)

        if passed:
            result = mark_attendance(student_row, class_row, "Present",
                                     student_lat=lat_f, student_lng=lng_f, distance_meters=dist_m)
            if result.get("already_marked"):
                return jsonify({"ok": True, "already_marked": True,
                                "message": "Attendance already recorded as Present today. No changes made.",
                                "class_name": class_row["class_name"]})
            dist_str = f" ({int(dist_m)}m from teacher)" if dist_m is not None else ""
            return jsonify({"ok": True, "already_marked": False,
                            "message": f"Marked Present! {reason}{dist_str}",
                            "class_name": class_row["class_name"]})
        else:
            result = mark_attendance(student_row, class_row, "Absent",
                                     student_lat=lat_f, student_lng=lng_f, distance_meters=dist_m)
            if result.get("already_marked"):
                return jsonify({"ok": True, "already_marked": True,
                                "message": "Attendance already recorded as Present today. No changes made.",
                                "class_name": class_row["class_name"]})
            return jsonify({"ok": False, "absent": True,
                            "message": f"Marked Absent — verification failed. {reason}",
                            "class_name": class_row["class_name"]})
    except Exception as e:
        print("student_checkin_api error:", e)
        return jsonify({"ok": False, "error": "Something went wrong recording your attendance. Please try again."})


@app.route("/student/attendance-status")
def student_attendance_status():
    """
    Lightweight JSON poll used by the student dashboard to know (a) which classes
    the student has already marked Present for today, and (b) which classes
    currently have a teacher-started session running — so both the "time left
    to scan" countdown and the Mark Present / Not Started buttons can update
    live, without a full page reload, the moment a teacher starts a session or
    attendance is recorded elsewhere.
    """
    protect = student_required()
    if protect:
        return jsonify({"ok": False, "error": "Not logged in.", "marked_class_ids": [], "active_sessions": []})

    student_db_id = get_logged_student_db_id()
    student_id = session.get("student_id")
    school_id = get_current_school_id()
    today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT class_id FROM attendance
            WHERE student_id=%s AND date=%s AND status='Present'
        """, (student_id, today_str))
        rows = cur.fetchall()
        conn.close()
        marked_class_ids = [r["class_id"] for r in rows]
    except Exception as e:
        print("student_attendance_status error:", e)
        return jsonify({"ok": False, "error": "Something went wrong loading attendance status.", "marked_class_ids": [], "active_sessions": []})

    active_sessions = []
    try:
        for c in get_classes_for_student(student_db_id):
            active_sess = get_active_session_for_class(c["id"], school_id)
            if active_sess and active_sess.get("expires_at"):
                active_sessions.append({
                    "class_id": c["id"],
                    "class_name": c["class_name"],
                    "expires_at": utc_iso(active_sess["expires_at"])
                })
    except Exception as e:
        print("student_attendance_status active-session lookup error:", e)

    return jsonify({"ok": True, "marked_class_ids": marked_class_ids, "active_sessions": active_sessions})


@app.route("/student/qr-scan")
def student_qr_scan_portal():
    """
    In-app camera QR scanner. Opens the device's back camera right inside the
    website (no need to back out to the phone's native camera app) and uses
    the jsQR library to decode QR codes from the live video feed in real time.
    Once a /checkin/<code> URL is detected, it's handled exactly like a normal
    QR scan: the code is parsed out and sent straight into the same check-in
    flow used by qr_checkin_landing().
    """
    protect = student_required()
    if protect:
        return protect

    student_db_id = get_logged_student_db_id()
    student_ctx = get_student_row_by_db_id(student_db_id)

    body = f"""
    <div class="max-w-lg mx-auto text-center space-y-4">
        <div>
            <h1 class="text-2xl font-extrabold text-slate-800">📷 Scan Attendance QR Code</h1>
            <p class="text-sm text-slate-500 mt-1">Point your camera at the QR code your teacher is showing.</p>
        </div>

        <!-- Camera viewport: touch-action none so pinch doesn't zoom the page -->
        <div id="qrViewport" class="relative max-w-md mx-auto bg-black rounded-2xl overflow-hidden shadow-lg border border-slate-300"
             style="aspect-ratio:1/1; touch-action:none;">
            <video id="qrVideo" autoplay playsinline muted class="w-full h-full object-cover"></video>
            <canvas id="qrCanvas" class="hidden"></canvas>
            <!-- Scan-frame overlay -->
            <div class="absolute inset-0 pointer-events-none flex items-center justify-center">
                <div class="w-2/3 h-2/3 border-4 border-emerald-400 rounded-2xl" style="box-shadow:0 0 0 2000px rgba(0,0,0,0.35);"></div>
            </div>
            <!-- Zoom level badge -->
            <div id="zoomBadge" class="absolute top-2 right-2 bg-black/60 text-white text-xs font-bold px-2 py-1 rounded-lg hidden">1.0×</div>
        </div>

        <!-- Zoom slider — shown after camera starts -->
        <div id="zoomRow" class="hidden max-w-md mx-auto flex items-center gap-3 px-2">
            <span class="text-slate-400 text-lg select-none">🔍</span>
            <input id="zoomSlider" type="range" min="1" max="5" step="0.1" value="1"
                   class="flex-1 h-2 accent-emerald-500 cursor-pointer"
                   oninput="applyZoom(parseFloat(this.value))">
            <span class="text-slate-400 text-lg select-none">🔎</span>
        </div>

        <div id="qrStatus" class="text-sm font-semibold text-slate-500">📡 Starting camera…</div>
        <div id="gpsStatus" class="text-xs text-slate-400">📡 Detecting your GPS location…</div>

        <div id="qrResultBox" class="max-w-md mx-auto"></div>

        <div class="flex justify-center gap-2 flex-wrap pt-2">
            <button class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-xl text-sm" onclick="switchCamera()">🔄 Switch Camera</button>
            <button class="bg-orange-500 hover:bg-orange-600 text-white font-bold py-2 px-4 rounded-xl text-sm" onclick="startCamera()">↺ Restart Camera</button>
            <a class="bg-slate-100 hover:bg-slate-200 text-slate-700 font-bold py-2 px-4 rounded-xl text-sm" href="/student">⌨️ Enter Code Manually Instead</a>
            <a class="bg-slate-800 hover:bg-slate-900 text-white font-bold py-2 px-4 rounded-xl text-sm" href="/student">← Back</a>
        </div>
    </div>

    <!-- jsQR: lightweight pure-JS QR decoder. Note: jsQR is NOT on cdnjs —
         it's only published via jsDelivr (verified working URL below). -->
    <script src="https://cdn.jsdelivr.net/npm/jsqr@1.4.0/dist/jsQR.min.js"></script>
    <script>
    // ── CSRF helper: every POST in this app is checked against the
    // per-session csrf_token by a global before_request hook. This page's
    // /student/checkin/api call was missing that token, so the server was
    // rejecting every submission with 400 "Invalid or missing CSRF token"
    // before it ever reached the attendance logic. ──
    function _qrGetCsrfCookie() {{
        const m = document.cookie.match('(^|;)\\s*csrf_token\\s*=\\s*([^;]+)');
        return m ? decodeURIComponent(m.pop()) : '';
    }}
    const qrVideo = document.getElementById('qrVideo');
    const qrCanvas = document.getElementById('qrCanvas');
    const qrCtx = qrCanvas.getContext('2d', {{ willReadFrequently: true }});
    const qrStatus = document.getElementById('qrStatus');
    const zoomSlider = document.getElementById('zoomSlider');
    const zoomBadge = document.getElementById('zoomBadge');
    const zoomRow = document.getElementById('zoomRow');
    let qrStream = null;
    let qrScanLoopId = null;
    let qrCurrentFacingMode = "environment";  // back camera by default
    let qrHandledCode = false;
    let qrStudentLat = null;
    let qrAbortController = null;  // top-level so the inline Cancel button (onclick, global scope) can reach it
    let qrStudentLng = null;

    // ── Zoom state ──
    let currentZoom = 1.0;
    let maxNativeZoom = 5.0;  // updated once camera starts

    // ── Apply zoom via Camera API (native zoom) or CSS transform fallback ──
    async function applyZoom(level) {{
        currentZoom = level;
        zoomBadge.innerText = level.toFixed(1) + '×';
        zoomBadge.classList.remove('hidden');
        if (qrStream) {{
            const track = qrStream.getVideoTracks()[0];
            const caps = track.getCapabilities ? track.getCapabilities() : {{}};
            if (caps.zoom) {{
                // Native hardware/software zoom — keeps full resolution, best for QR
                const clamped = Math.min(Math.max(level, caps.zoom.min), caps.zoom.max);
                try {{ await track.applyConstraints({{ advanced: [{{ zoom: clamped }}] }}); }} catch(e) {{}}
            }} else {{
                // Fallback: CSS scale transform on the video element
                qrVideo.style.transform = 'scale(' + level + ')';
                qrVideo.style.transformOrigin = 'center center';
            }}
        }}
        zoomSlider.value = level;
    }}

    // ── Pinch-to-zoom on the camera viewport ──
    (function() {{
        const viewport = document.getElementById('qrViewport');
        let initialDist = null;
        let zoomAtPinchStart = 1.0;

        function dist(t) {{
            const dx = t[0].clientX - t[1].clientX;
            const dy = t[0].clientY - t[1].clientY;
            return Math.sqrt(dx*dx + dy*dy);
        }}

        viewport.addEventListener('touchstart', function(e) {{
            if (e.touches.length === 2) {{
                e.preventDefault();
                initialDist = dist(e.touches);
                zoomAtPinchStart = currentZoom;
            }}
        }}, {{ passive: false }});

        viewport.addEventListener('touchmove', function(e) {{
            if (e.touches.length === 2 && initialDist) {{
                e.preventDefault();
                const scale = dist(e.touches) / initialDist;
                const newZoom = Math.min(Math.max(zoomAtPinchStart * scale, 1.0), maxNativeZoom);
                applyZoom(newZoom);
            }}
        }}, {{ passive: false }});

        viewport.addEventListener('touchend', function(e) {{
            if (e.touches.length < 2) {{ initialDist = null; }}
        }});
    }})();

    // GPS capture (sent along once a code is found, same as the manual check-in form)
    const gpsStatusEl = document.getElementById('gpsStatus');
    if (navigator.geolocation) {{
        if (gpsStatusEl) {{ gpsStatusEl.innerText = '🔄 Acquiring GPS fix…'; }}
        let _qrBestAcc = Infinity;
        const _qrWatcher = navigator.geolocation.watchPosition(function(pos) {{
            const acc = pos.coords.accuracy;
            if (acc < _qrBestAcc) {{
                _qrBestAcc = acc;
                qrStudentLat = pos.coords.latitude;
                qrStudentLng = pos.coords.longitude;
                if (gpsStatusEl) {{
                    gpsStatusEl.innerText = '✅ GPS: ' + qrStudentLat.toFixed(5) + ', ' + qrStudentLng.toFixed(5) + ' (±' + Math.round(acc) + 'm)';
                    gpsStatusEl.className = acc <= 50 ? 'text-xs text-emerald-600 font-semibold' : 'text-xs text-amber-500 font-semibold';
                    if (acc <= 50) {{ navigator.geolocation.clearWatch(_qrWatcher); }}
                }}
            }}
        }}, function(err) {{
            if (gpsStatusEl) {{
                gpsStatusEl.innerText = '⚠️ GPS unavailable — session code from the QR will still work.';
                gpsStatusEl.className = 'text-xs text-amber-600 font-semibold';
            }}
        }}, {{ enableHighAccuracy: true, timeout: 20000, maximumAge: 0 }});
        setTimeout(function() {{ navigator.geolocation.clearWatch(_qrWatcher); }}, 20000);
    }} else {{
        if (gpsStatusEl) gpsStatusEl.innerText = '⚠️ GPS not supported on this device.';
    }}

    async function startCamera() {{
        try {{
            qrHandledCode = false;
            currentZoom = 1.0;
            zoomSlider.value = 1;
            zoomBadge.classList.add('hidden');
            zoomRow.classList.add('hidden');
            qrVideo.style.transform = '';
            if (qrScanLoopId) {{ cancelAnimationFrame(qrScanLoopId); qrScanLoopId = null; }}
            qrStatus.innerText = "Starting camera…";
            qrStatus.className = "text-sm font-semibold text-slate-500";
            if (typeof jsQR === 'undefined') {{
                qrStatus.innerText = "❌ QR scanner library failed to load. Check your internet connection and reload the page.";
                qrStatus.className = "text-sm font-semibold text-red-600";
                return;
            }}
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {{
                qrStatus.innerText = "Camera not supported. Please use HTTPS and a modern browser.";
                return;
            }}
            if (qrStream) {{
                qrStream.getTracks().forEach(track => track.stop());
                qrStream = null;
            }}
            let constraints = [
                {{ video: {{ facingMode: {{ exact: qrCurrentFacingMode }}, width: {{ ideal: 1920 }}, height: {{ ideal: 1920 }} }}, audio: false }},
                {{ video: {{ facingMode: {{ ideal: qrCurrentFacingMode }}, width: {{ ideal: 1280 }}, height: {{ ideal: 1280 }} }}, audio: false }},
                {{ video: true, audio: false }}
            ];
            let lastErr = null;
            for (let c of constraints) {{
                try {{ qrStream = await navigator.mediaDevices.getUserMedia(c); break; }}
                catch (e) {{ lastErr = e; qrStream = null; }}
            }}
            if (!qrStream) throw lastErr;
            qrVideo.srcObject = qrStream;
            await new Promise((resolve) => {{ qrVideo.onloadedmetadata = () => resolve(); }});
            await qrVideo.play();

            // ── Detect zoom capability and configure slider ──
            const track = qrStream.getVideoTracks()[0];
            const caps = track.getCapabilities ? track.getCapabilities() : {{}};
            if (caps.zoom) {{
                maxNativeZoom = caps.zoom.max || 5.0;
                zoomSlider.min = caps.zoom.min || 1;
                zoomSlider.max = maxNativeZoom;
                zoomSlider.step = 0.1;
            }} else {{
                maxNativeZoom = 5.0;  // CSS fallback range
                zoomSlider.min = 1;
                zoomSlider.max = 5;
            }}
            zoomRow.classList.remove('hidden');

            qrStatus.innerText = "✅ Camera ready — point at the QR code (" + (qrCurrentFacingMode === "environment" ? "Back" : "Front") + "). Pinch or use slider to zoom.";
            qrStatus.className = "text-sm font-semibold text-emerald-600";
            scanLoop();
        }} catch (err) {{
            if (err && err.name === "NotAllowedError") {{ qrStatus.innerText = "❌ Camera permission denied. Please allow camera access and reload."; }}
            else if (err && err.name === "NotFoundError") {{ qrStatus.innerText = "❌ No camera found on this device."; }}
            else if (location.protocol !== "https:") {{ qrStatus.innerText = "❌ Camera requires HTTPS."; }}
            else {{ qrStatus.innerText = "❌ Camera error: " + (err && err.message ? err.message : err); }}
            qrStatus.className = "text-sm font-semibold text-red-600";
        }}
    }}

    function switchCamera() {{
        qrCurrentFacingMode = qrCurrentFacingMode === "environment" ? "user" : "environment";
        startCamera();
    }}

    function scanLoop() {{
        if (qrHandledCode) return;
        if (qrVideo.readyState === qrVideo.HAVE_ENOUGH_DATA && typeof jsQR !== 'undefined') {{
            qrCanvas.width = qrVideo.videoWidth;
            qrCanvas.height = qrVideo.videoHeight;
            qrCtx.drawImage(qrVideo, 0, 0, qrCanvas.width, qrCanvas.height);
            try {{
                const imageData = qrCtx.getImageData(0, 0, qrCanvas.width, qrCanvas.height);
                const result = jsQR(imageData.data, imageData.width, imageData.height, {{ inversionAttempts: "dontInvert" }});
                if (result && result.data) {{
                    handleScannedText(result.data);
                    return;  // stop the loop — handleScannedText takes over
                }}
            }} catch (e) {{ /* ignore transient decode errors and keep scanning */ }}
        }}
        qrScanLoopId = requestAnimationFrame(scanLoop);
    }}

    function handleScannedText(text) {{
        if (qrHandledCode) return;
        qrHandledCode = true;

        // Extract the session code from a full /checkin/<code> URL or bare code
        let code = null;
        try {{
            const url = new URL(text);
            const parts = url.pathname.split('/').filter(Boolean);
            if (parts[0] === 'checkin' && parts[1]) code = parts[1];
        }} catch (e) {{
            code = text.trim();
        }}

        if (!code) {{
            qrStatus.innerText = "⚠️ That QR code isn't a valid attendance code. Try again.";
            qrStatus.className = "text-sm font-semibold text-amber-600";
            qrHandledCode = false;
            qrScanLoopId = requestAnimationFrame(scanLoop);
            return;
        }}

        // Stop camera
        if (qrStream) {{ qrStream.getTracks().forEach(t => t.stop()); qrStream = null; }}

        // Show spinner inline — no page reload
        qrStatus.innerText = "⏳ Submitting attendance…";
        qrStatus.className = "text-sm font-semibold text-blue-500";

        // Safety net: give the student fast, honest feedback instead of a
        // spinner that can sit there for over a minute. A 70s abort felt like
        // "frozen" to students even though it was technically bounded — by
        // 8s we tell them it's slow, by 25s we give up and show a real Retry
        // button instead of making them wonder if the app is broken.
        const qrSubmitTimeout = setTimeout(() => {{
            qrStatus.innerText = "⏳ Still working — hang tight a few more seconds…";
            qrStatus.className = "text-sm font-semibold text-amber-600";
            // Give the student a way out instead of just watching a spinner —
            // if they'd rather bail and try again (or enter the code by hand)
            // than keep waiting, let them.
            document.getElementById('qrResultBox').innerHTML =
                '<button onclick="qrAbortController.abort(); retryQR();" ' +
                'class="bg-slate-200 hover:bg-slate-300 text-slate-700 font-bold px-4 py-2 rounded-xl text-xs">✕ Cancel &amp; Try Again</button>';
        }}, 8000);

        qrAbortController = new AbortController();
        const qrAbortTimer = setTimeout(() => qrAbortController.abort(), 25000);

        fetch('/student/checkin/api', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json', 'X-CSRF-Token': _qrGetCsrfCookie() }},
            signal: qrAbortController.signal,
            body: JSON.stringify({{
                session_code: code,
                latitude: qrStudentLat,
                longitude: qrStudentLng,
                gps_accuracy: _qrBestAcc < Infinity ? _qrBestAcc : null
            }})
        }})
        .then(r => r.json().catch(() => {{
            // Server returned something that isn't valid JSON (e.g. an error page) —
            // surface that clearly instead of leaving the screen blank/silent.
            throw new Error('bad_json');
        }}))
        .then(data => {{
            clearTimeout(qrSubmitTimeout);
            clearTimeout(qrAbortTimer);
            const resultBox = document.getElementById('qrResultBox');
            if (data.ok) {{
                const icon = data.already_marked ? 'ℹ️' : '✅';
                const title = data.already_marked ? 'Already Marked Present' : 'Attendance Recorded';
                const color = data.already_marked ? 'bg-blue-50 border-blue-200 text-blue-800' : 'bg-emerald-50 border-emerald-200 text-emerald-800';
                resultBox.innerHTML = `
                    <div class="rounded-2xl border-2 p-5 ${{color}} text-center space-y-2">
                        <div class="text-4xl">${{icon}}</div>
                        <div class="font-extrabold text-xl">${{title}}</div>
                        <div class="font-semibold">${{data.message}}</div>
                        ${{data.class_name ? '<div class="text-sm opacity-70">Class: ' + data.class_name + '</div>' : ''}}
                        <div class="text-xs opacity-60 mt-2">Redirecting to your dashboard…</div>
                    </div>`;
                qrStatus.innerText = '';
                setTimeout(() => {{ window.location.href = '/student'; }}, 2000);
            }} else if (data.absent) {{
                resultBox.innerHTML = `
                    <div class="rounded-2xl border-2 bg-red-50 border-red-200 text-red-800 text-center p-5 space-y-2">
                        <div class="text-4xl">\u274c</div>
                        <div class="font-bold text-lg">Marked Absent</div>
                        <div class="text-sm opacity-80">${{data.message}}</div>
                        ${{data.class_name ? '<div class="text-sm opacity-70">Class: ' + data.class_name + '</div>' : ''}}
                        <div class="flex justify-center gap-3 mt-3 flex-wrap">
                            <button onclick="retryQR()" class="bg-blue-600 text-white font-bold px-5 py-2 rounded-xl text-sm">↺ Try Again</button>
                            <a href="/student" class="bg-slate-800 text-white font-bold px-5 py-2 rounded-xl text-sm">Dashboard</a>
                        </div>
                    </div>`;
                qrStatus.innerText = '';
            }} else {{
                resultBox.innerHTML = `
                    <div class="rounded-2xl border-2 bg-amber-50 border-amber-200 text-amber-800 text-center p-5 space-y-2">
                        <div class="text-4xl">\u26a0\ufe0f</div>
                        <div class="font-bold">${{data.error || 'Something went wrong.'}}</div>
                        <div class="flex justify-center gap-3 mt-3 flex-wrap">
                            <button onclick="retryQR()" class="bg-blue-600 text-white font-bold px-5 py-2 rounded-xl text-sm">↺ Try Again</button>
                            <a href="/student" class="bg-slate-800 text-white font-bold px-5 py-2 rounded-xl text-sm">Dashboard</a>
                        </div>
                    </div>`;
                qrStatus.innerText = '';
            }}
        }})
        .catch(err => {{
            clearTimeout(qrSubmitTimeout);
            clearTimeout(qrAbortTimer);
            const timedOut = err && err.name === 'AbortError';
            const badJson = err && err.message === 'bad_json';
            let msg = 'Network error. Please check your connection.';
            if (timedOut) msg = 'Request timed out — the server may have been waking up from being idle. Please tap Try Again.';
            else if (badJson) msg = 'Server error while recording attendance. Please try again.';
            document.getElementById('qrResultBox').innerHTML = `
                <div class="rounded-2xl border-2 bg-amber-50 border-amber-200 text-amber-800 text-center p-5">
                    <div class="font-bold">${{msg}}</div>
                    <div class="flex justify-center gap-3 mt-3 flex-wrap">
                        <button onclick="retryQR()" class="bg-blue-600 text-white font-bold px-5 py-2 rounded-xl text-sm">↺ Try Again</button>
                        <a href="/student" class="bg-slate-800 text-white font-bold px-5 py-2 rounded-xl text-sm">Dashboard</a>
                    </div>
                </div>`;
            qrStatus.innerText = '';
        }});
    }}

    function retryQR() {{
        document.getElementById('qrResultBox').innerHTML = '';
        qrHandledCode = false;
        startCamera();
    }}

    startCamera();
    </script>
    """
    return page_wrapper("Scan Attendance QR Code", body, is_student=True, student_context=student_ctx)


@app.route("/student/scan")
def student_scan_portal():
    protect = student_required()
    if protect:
        return protect

    student_db_id = get_logged_student_db_id()
    student_ctx = get_student_row_by_db_id(student_db_id)
    
    body = f"""
    <div class="max-w-3xl mx-auto text-center space-y-4">
        <div>
            <h1 class="text-3xl font-extrabold text-slate-800">📸 Student Face Check-In</h1>
            <p class="text-sm text-slate-500 mt-1">Look into your camera to verify your identity</p>
        </div>

        <div class="max-w-md mx-auto bg-amber-50 border border-amber-200 rounded-xl p-3 text-sm text-amber-800 text-left">
            ⚠️ Attendance requires at least one: <strong>GPS location</strong>, <strong>session code</strong>, or <strong>school WiFi</strong>.
        </div>

        <div class="max-w-md mx-auto">
            <input type="text" id="sessionCodeInput" placeholder="Enter session code (optional)" maxlength="10"
                class="w-full px-3 py-2 border rounded-lg uppercase tracking-widest font-mono text-base text-center focus:ring-2 focus:ring-blue-400">
            <div id="gpsStatus" class="text-xs text-slate-400 mt-1">📡 Detecting GPS location...</div>
        </div>

        <video id="video" autoplay playsinline muted class="w-full max-w-lg mx-auto bg-black border border-slate-300 rounded-2xl shadow-lg"></video>
        
        <div id="result" class="text-2xl font-bold text-emerald-600 mt-4 tracking-tight animate-pulse">Initializing face capture...</div>
        <div id="status" class="text-xs font-semibold text-slate-400">Please allow camera access</div>

        <div id="ambiguousPicker" class="hidden max-w-md mx-auto bg-amber-50 border-2 border-amber-300 rounded-2xl p-4 space-y-3 text-left">
            <div class="text-sm font-bold text-amber-800">⚠️ This face matches more than one student. Who is this?</div>
            <div id="ambiguousCandidates" class="space-y-2"></div>
            <button onclick="cancelAmbiguous()" class="text-xs text-slate-400 hover:text-slate-600 font-semibold">Cancel — keep scanning</button>
        </div>
        
        <div class="flex justify-center gap-2 flex-wrap pt-2">
            <button class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-xl" onclick="switchCamera()">Switch Camera</button>
            <button class="bg-orange-500 hover:bg-orange-600 text-white font-bold py-2 px-4 rounded-xl" onclick="startCamera()">Reset Feed</button>
            <a class="bg-slate-800 hover:bg-slate-900 text-white font-bold py-2 px-4 rounded-xl" href="/student">Back to Profile</a>
        </div>
    </div>

<script>
const video = document.getElementById('video');
const resultDiv = document.getElementById('result');
const statusDiv = document.getElementById('status');
let currentFacingMode = "user";
let stream = null;
let intervalId = null;
let studentLat = null;
let studentLng = null;

// GPS detection — watch for a fix and keep the best one until accuracy is good
let _bestAcc2 = Infinity;
let _gpsWatchId2 = null;
const _GPS_GOOD_ENOUGH_M2 = 50;
const _GPS_MAX_WAIT_MS2 = 20000;

if (navigator.geolocation) {{
    document.getElementById('gpsStatus').innerText = '🔄 Acquiring GPS fix…';
    _gpsWatchId2 = navigator.geolocation.watchPosition(function(pos) {{
        const acc = pos.coords.accuracy;
        if (acc < _bestAcc2) {{
            _bestAcc2 = acc;
            studentLat = pos.coords.latitude;
            studentLng = pos.coords.longitude;
            document.getElementById('gpsStatus').innerText = '✅ GPS: ' + studentLat.toFixed(5) + ', ' + studentLng.toFixed(5) + ' (±' + Math.round(acc) + 'm)';
            document.getElementById('gpsStatus').className = acc <= _GPS_GOOD_ENOUGH_M2 ? 'text-xs text-emerald-600 mt-1 font-semibold' : 'text-xs text-amber-500 mt-1 font-semibold';
            if (acc <= _GPS_GOOD_ENOUGH_M2) {{
                navigator.geolocation.clearWatch(_gpsWatchId2);
            }}
        }}
    }}, function() {{
        document.getElementById('gpsStatus').innerText = '⚠️ GPS unavailable — use session code or school WiFi.';
        document.getElementById('gpsStatus').className = 'text-xs text-amber-600 mt-1 font-semibold';
    }}, {{ enableHighAccuracy: true, timeout: _GPS_MAX_WAIT_MS2, maximumAge: 0 }});
    setTimeout(function() {{ if (_gpsWatchId2 !== null) navigator.geolocation.clearWatch(_gpsWatchId2); }}, _GPS_MAX_WAIT_MS2);
}}

async function startCamera() {{
    try {{
        statusDiv.innerText = "Starting camera...";
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {{
            statusDiv.innerText = "Camera not supported. Please use HTTPS and a modern browser.";
            return;
        }}
        if (stream) {{
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }}
        let constraints = [
            {{ video: {{ facingMode: {{ exact: currentFacingMode }}, width: {{ ideal: 640 }}, height: {{ ideal: 480 }} }}, audio: false }},
            {{ video: {{ facingMode: {{ ideal: currentFacingMode }}, width: {{ ideal: 640 }}, height: {{ ideal: 480 }} }}, audio: false }},
            {{ video: true, audio: false }}
        ];
        let lastErr = null;
        for (let c of constraints) {{
            try {{ stream = await navigator.mediaDevices.getUserMedia(c); break; }}
            catch (e) {{ lastErr = e; stream = null; }}
        }}
        if (!stream) throw lastErr;
        video.srcObject = stream;
        await new Promise((resolve) => {{ video.onloadedmetadata = () => resolve(); }});
        try {{
            await video.play();
            statusDiv.innerText = "✅ Camera ready (" + (currentFacingMode === "user" ? "Front" : "Back") + ")";
            startScanningLoops();
        }} catch (e) {{
            statusDiv.innerText = "Tap the video to start";
        }}
    }} catch (err) {{
        if (err.name === "NotAllowedError") {{ statusDiv.innerText = "❌ Camera permission denied. Please allow camera access and reload."; }}
        else if (err.name === "NotFoundError") {{ statusDiv.innerText = "❌ No camera found on this device."; }}
        else if (location.protocol !== "https:") {{ statusDiv.innerText = "❌ Camera requires HTTPS."; }}
        else {{ statusDiv.innerText = "❌ Camera error: " + err.message; }}
    }}
}}

function switchCamera() {{
    currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
    startCamera();
}}

function startScanningLoops() {{
    if(intervalId) clearInterval(intervalId);
    intervalId = setInterval(async () => {{
        if(video.paused || video.ended) return;
        try {{
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const image = canvas.toDataURL('image/jpeg');
            const sessionCode = document.getElementById('sessionCodeInput').value.trim();

            const res = await fetch('/student/scan-frame', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{
                    image: image,
                    latitude: studentLat,
                    longitude: studentLng,
                    session_code: sessionCode
                }})
            }});
            const data = await res.json();
            if (data.success) {{
                resultDiv.innerText = "✓ Verified!";
                statusDiv.innerText = data.message;
                clearInterval(intervalId);
                setTimeout(() => {{ window.location.href = '/student'; }}, 2500);
            }} else {{
                resultDiv.innerText = "Scanning...";
                if(data.message) statusDiv.innerText = data.message;
            }}
        }} catch(err) {{
            console.log("Scan error:", err);
        }}
    }}, 2000);
}}

window.addEventListener('beforeunload', () => {{
    if(intervalId) clearInterval(intervalId);
    if(stream) stream.getTracks().forEach(t => t.stop());
}});

startCamera();
</script>
    """
    return page_wrapper("Student Face Check-In", body, is_student=True, student_context=student_ctx)


@app.route("/student/scan-frame", methods=["POST"])
def student_scan_frame_matrix_lookup():
    try:
        protect = student_required()
        if protect:
            return jsonify({"success": False, "message": "Session verification check failure"})

        student_db_id = get_logged_student_db_id()
        student_row = get_student_row_by_db_id(student_db_id)
        classes = get_classes_for_student(student_db_id)

        if not student_row or not classes:
            return jsonify({"success": False, "message": "No course classrooms assigned roster registry blocks"})

        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"success": False})

        image_data = data["image"]
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_data = image_data.replace(" ", "+")
        missing_padding = len(image_data) % 4
        if missing_padding:
            image_data += "=" * (4 - missing_padding)

        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False})

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face using MediaPipe
        face_embedding = _get_face_embedding(rgb_frame)

        if face_embedding is None:
            return jsonify({"success": False, "message": "Position your face clearly within the camera frame layout matrix"})

        if len(known_encodings) == 0:
            return jsonify({"success": False, "message": "No registered faces found"})

        best_match_index = None
        best_dist = float("inf")
        for i, known_emb in enumerate(known_encodings):
            matched, dist = _compare_embeddings(known_emb, face_embedding, tolerance=0.6)
            if matched and dist < best_dist:
                best_dist = dist
                best_match_index = i

        if best_match_index is not None:
            matched_student = known_students[best_match_index]

            if matched_student["db_id"] != student_db_id:
                return jsonify({"success": False, "message": "Face profile does not match logged-in student."})

            # ── LOCATION CHECKS ──
            school_id = get_current_school_id()
            lat = data.get("latitude")
            lng = data.get("longitude")
            session_code = data.get("session_code", "")

            try:
                lat_f = float(lat) if lat else None
                lng_f = float(lng) if lng else None
            except:
                lat_f = lng_f = None

            any_passed = False
            already_marked_all = True
            any_class_had_active_session = False
            for c in classes:
                if not is_attendance_session_active(c["id"], school_id):
                    # Teacher hasn't started attendance for this class — skip entirely,
                    # don't mark Present or Absent.
                    continue
                any_class_had_active_session = True
                passed, reason, dist_m = run_location_checks(lat_f, lng_f, session_code, c["id"], school_id, request)
                if passed:
                    result = mark_attendance(student_row, c, "Present", student_lat=lat_f, student_lng=lng_f, distance_meters=dist_m)
                    if result.get('already_marked'):
                        pass  # already Present, don't overwrite
                    else:
                        any_passed = True
                        already_marked_all = False
                else:
                    result = mark_attendance(student_row, c, "Absent", student_lat=lat_f, student_lng=lng_f, distance_meters=dist_m)
                    if not result.get('already_marked'):
                        already_marked_all = False

            if not any_class_had_active_session:
                return jsonify({
                    "success": False,
                    "message": "Attendance hasn't started yet for any of your classes. Wait for your teacher to start the session."
                })

            # If every class was already marked Present, tell the student
            if already_marked_all and not any_passed:
                return jsonify({
                    "success": True,
                    "already_marked": True,
                    "message": f"\u2139\ufe0f Attendance already recorded as Present today for {student_row['full_name']}. No changes made."
                })
            if any_passed:
                return jsonify({
                    "success": True,
                    "message": f"Identity verified for {student_row['full_name']}. Attendance recorded!"
                })
            else:
                return jsonify({
                    "success": True,
                    "message": f"Face verified but location check failed — marked Absent."
                })

        return jsonify({"success": False, "message": "Face trace vector lookup match mismatch code error"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Internal runtime matrix loop failure error: {str(e)}"})


# =========================================================
# STUDENT PROFILE PORTAL VIEW 
# =========================================================
@app.route("/student")
def student_dashboard_portal():
    protect = student_required()
    if protect:
        return protect

    student_db_id = get_logged_student_db_id()
    student_id = session.get("student_id")
    school_name = session.get("school_name", "")
    student_ctx = get_student_row_by_db_id(student_db_id)

    classes = get_classes_for_student(student_db_id)
    history = get_attendance_for_student(student_id)
    school_id = get_current_school_id()

    # Build a quick lookup of classes the student has already marked Present for *today*,
    # so the "time left to scan" countdown disappears once they've checked in.
    today_str = datetime.now().strftime("%Y-%m-%d")
    marked_present_today_class_ids = {
        h["class_id"] for h in history
        if h["date"] == today_str and h["status"] == "Present"
    }

    # Check which of the student's classes currently have an open attendance window,
    # so we can show a live "time left to scan" countdown for each — but only for
    # classes this student hasn't already checked into today.
    open_sessions = []
    active_session_class_ids = set()
    for c in classes:
        active_sess = get_active_session_for_class(c["id"], school_id)
        if active_sess and active_sess.get("expires_at"):
            active_session_class_ids.add(c["id"])
        if c["id"] in marked_present_today_class_ids:
            continue
        if active_sess and active_sess.get("expires_at"):
            open_sessions.append({
                "class_id": c["id"],
                "class_name": c["class_name"],
                "expires_at": utc_iso(active_sess["expires_at"])
            })

    # Build the "attendance open" banner — one row per class with a live window right now.
    open_sessions_html = ""
    if open_sessions:
        rows_html = ""
        for s in open_sessions:
            rows_html += f"""
                <div class="flex items-center justify-between gap-3 bg-white/70 rounded-xl px-4 py-2.5" data-class-id="{s['class_id']}" data-role="session-row">
                    <div class="flex items-center gap-2 min-w-0">
                        <span class="relative flex h-2.5 w-2.5 flex-shrink-0">
                            <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                            <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
                        </span>
                        <span class="font-bold text-slate-800 text-sm truncate">{s["class_name"]}</span>
                    </div>
                    <div class="flex items-center gap-3 flex-shrink-0">
                        <span class="font-mono font-black text-emerald-700 text-lg tabular-nums"
                            data-expires="{s['expires_at']}" data-role="countdown">--:--</span>
                        <a href="/student/qr-scan" class="bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold px-3 py-1.5 rounded-lg transition whitespace-nowrap">
                            <i class="fas fa-qrcode mr-1"></i> Scan Now
                        </a>
                    </div>
                </div>
            """
        open_sessions_html = f"""
        <div class="lg:col-span-3 bg-gradient-to-r from-emerald-50 to-teal-50 border-2 border-emerald-300 rounded-2xl p-4 space-y-2 shadow-sm" id="openSessionsBanner">
            <div class="flex items-center gap-2 px-1">
                <span class="text-emerald-700 font-extrabold text-sm uppercase tracking-wide">⏳ Attendance Open — Time Left to Scan</span>
            </div>
            <div id="openSessionsRows" class="space-y-2">
                {rows_html}
            </div>
        </div>
        """

    body = f"""
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6" id="dashboardGrid">
        {open_sessions_html}
        <div class="lg:col-span-1 space-y-6">
            <div class="bg-white border rounded-xl p-6 text-center shadow-sm">
                <img class="w-24 h-24 object-cover rounded-full mx-auto border-2 border-blue-500 mb-3 shadow-inner" src="{supabase_public_url(student_ctx["image_file"])}">
                <h2 class="text-xl font-bold text-slate-800">{student_ctx["full_name"]}</h2>
                <p class="text-xs font-mono text-slate-400 mt-1">ID: {student_ctx["student_id"]}</p>
                {f'<p class="text-xs font-semibold text-blue-600 mt-1">🏫 {school_name}</p>' if school_name else ''}
                <div class="mt-3 inline-block bg-emerald-50 text-emerald-700 text-xs font-bold px-3 py-1 rounded-full border border-emerald-200">✓ Active Enrolled Student</div>
                
                <div class="mt-6 border-t pt-4 space-y-2">
                    <a href="/student/qr-scan" class="w-full inline-block text-center bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2.5 px-4 rounded-xl transition shadow-md shadow-indigo-100 text-xs">
                        <i class="fas fa-qrcode mr-1"></i> Scan QR to Check In
                    </a>
                    <a href="/student/scan" class="w-full inline-block text-center bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-2.5 px-4 rounded-xl transition shadow-md shadow-emerald-100 text-xs">
                        <i class="fas fa-camera mr-1"></i> Check-In with Face Scanner
                    </a>
                    <a href="/student/edit-profile" class="w-full inline-block text-center bg-blue-600 hover:bg-blue-700 text-white font-bold py-2.5 px-4 rounded-xl transition text-xs">
                        ✏️ Edit My Profile
                    </a>
                </div>
            </div>
        </div>

        <div class="lg:col-span-2 space-y-6">
            <div class="bg-white border rounded-xl overflow-hidden shadow-sm">
                <div class="p-4 bg-slate-50 border-b font-bold text-slate-700">📚 Registered Course Classes Summary Matrix</div>
                <div class="overflow-x-auto">
                    <table class="w-full text-left m-0 border-none shadow-none rounded-none">
                        <thead>
                            <tr class="bg-slate-100 border-b text-xs font-bold text-slate-600">
                                <th class="p-3">Class Target</th>
                                <th class="p-3">Department</th>
                                <th class="p-3">Course Catalog ID</th>
                                <th class="p-3">Section Identity</th>
                                <th class="p-3">Attendance Ratio</th>
                                <th class="p-3 text-right">Self Check-In</th>
                            </tr>
                        </thead>
                        <tbody class="text-xs">
    """
    if classes:
        for c in classes:
            pct = get_percentage(student_id, c["id"])
            if c["id"] in active_session_class_ids:
                checkin_action = f"""<a href="/student/checkin/manual/{c["id"]}" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-1 px-2.5 rounded text-[11px] transition inline-block">
                            <i class="fas fa-check mr-1"></i> Mark Present
                        </a>"""
            else:
                checkin_action = """<span class="bg-slate-100 text-slate-400 font-bold py-1 px-2.5 rounded text-[11px] inline-block cursor-not-allowed" title="Your teacher hasn't started attendance for this class yet">
                            <i class="fas fa-lock mr-1"></i> Not Started
                        </span>"""
            body += f"""
                <tr class="border-b" data-class-id="{c["id"]}">
                    <td class="p-3 font-semibold text-slate-800">{c["class_name"]}</td>
                    <td class="p-3 text-slate-500">{c["department"] or ""}</td>
                    <td class="p-3 font-mono">{c["course"] or ""}</td>
                    <td class="p-3">{c["section_name"] or ""}</td>
                    <td class="p-3"><strong class="text-blue-600 font-extrabold">{pct}%</strong></td>
                    <td class="p-3 text-right" data-role="checkin-cell" data-class-id="{c["id"]}">
                        {checkin_action}
                    </td>
                </tr>
            """
    else:
        body += "<tr><td colspan='6' class='p-4 text-center text-slate-400 font-medium'>No dynamic tracking assignments detected mapped to profile matrix.</td></tr>"
    body += """
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="bg-white border rounded-xl overflow-hidden shadow-sm">
                <div class="p-4 bg-slate-50 border-b font-bold text-slate-700">📋 My Historical Verification Check-In Logs</div>
                <div class="overflow-x-auto">
                    <table class="w-full text-left m-0 border-none shadow-none rounded-none">
                        <thead>
                            <tr class="bg-slate-100 border-b text-xs font-bold text-slate-600">
                                <th class="p-3">Calendar Date</th>
                                <th class="p-3">Time logged</th>
                                <th class="p-3">Course Target</th>
                                <th class="p-3">Subject Topic Description</th>
                                <th class="p-3">Status Pillar Flag</th>
                                <th class="p-3">Authorized Proctor</th>
                            </tr>
                        </thead>
                        <tbody class="text-xs">
    """
    if history:
        for h in history:
            body += f"""
                <tr class="border-b">
                    <td class="p-3 font-medium">{h["date"]}</td>
                    <td class="p-3 text-slate-400 font-mono">{format_time_12hr(h["time"])}</td>
                    <td class="p-3 font-bold text-slate-700">{h["class_name"]}</td>
                    <td class="p-3">{h["subject_name"] or ""}</td>
                    <td class="p-3"><span class="px-2 py-0.5 rounded text-[11px] font-bold {'bg-emerald-100 text-emerald-800' if h['status']=='Present' else 'bg-rose-100 text-rose-800'}">{h['status']}</span></td>
                    <td class="p-3 text-slate-500">{h["teacher_name"] or ""}</td>
                </tr>
            """
    else:
        body += "<tr><td colspan='6' class='p-4 text-center text-slate-400 font-medium'>No scan verification histories logged into backend databases rows.</td></tr>"
    body += """
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    """

    body += """
    <script>
    (function() {
        function removeRow(classId) {
            var row = document.querySelector('[data-role="session-row"][data-class-id="' + classId + '"]');
            if (row) row.remove();
            checkBannerEmpty();
        }

        function tick() {
            document.querySelectorAll('[data-role="countdown"]').forEach(function(el) {
                var expires = new Date(el.getAttribute('data-expires'));
                var now = new Date();
                var diffMs = expires - now;
                if (diffMs <= 0) {
                    el.innerText = "Closed";
                    el.classList.remove('text-emerald-700');
                    el.classList.add('text-rose-500');
                    var row = el.closest('[data-role="session-row"]');
                    if (row) {
                        // Give the student a beat to see "Closed" before the row disappears
                        setTimeout(function() { row.remove(); checkBannerEmpty(); }, 2000);
                    }
                    return;
                }
                var totalSecs = Math.floor(diffMs / 1000);
                var mins = Math.floor(totalSecs / 60);
                var secs = totalSecs % 60;
                el.innerText = mins + ":" + String(secs).padStart(2, "0");
                // Flip to an urgent color in the final 30 seconds
                if (totalSecs <= 30) {
                    el.classList.remove('text-emerald-700');
                    el.classList.add('text-rose-500');
                }
            });
        }

        function checkBannerEmpty() {
            var rows = document.getElementById('openSessionsRows');
            if (rows && rows.children.length === 0) {
                var banner = document.getElementById('openSessionsBanner');
                if (banner) banner.remove();
            }
        }

        // Create the "Attendance Open" banner on the fly if it doesn't exist yet
        // (e.g. the page loaded before the teacher had started any session).
        function ensureBanner() {
            var existingRows = document.getElementById('openSessionsRows');
            if (existingRows) return existingRows;
            var grid = document.getElementById('dashboardGrid');
            if (!grid) return null;
            var wrapper = document.createElement('div');
            wrapper.id = 'openSessionsBanner';
            wrapper.className = 'lg:col-span-3 bg-gradient-to-r from-emerald-50 to-teal-50 border-2 border-emerald-300 rounded-2xl p-4 space-y-2 shadow-sm';
            wrapper.innerHTML = '<div class="flex items-center gap-2 px-1">' +
                '<span class="text-emerald-700 font-extrabold text-sm uppercase tracking-wide">⏳ Attendance Open — Time Left to Scan</span>' +
                '</div><div id="openSessionsRows" class="space-y-2"></div>';
            grid.insertBefore(wrapper, grid.firstChild);
            return document.getElementById('openSessionsRows');
        }

        function upsertSessionRow(classId, className, expiresAtIso) {
            var existing = document.querySelector('[data-role="session-row"][data-class-id="' + classId + '"]');
            if (existing) {
                var cd = existing.querySelector('[data-role="countdown"]');
                if (cd) {
                    cd.setAttribute('data-expires', expiresAtIso);
                    cd.classList.remove('text-rose-500');
                    cd.classList.add('text-emerald-700');
                }
                return;
            }
            var rows = ensureBanner();
            if (!rows) return;
            var div = document.createElement('div');
            div.className = 'flex items-center justify-between gap-3 bg-white/70 rounded-xl px-4 py-2.5';
            div.setAttribute('data-class-id', classId);
            div.setAttribute('data-role', 'session-row');
            div.innerHTML = '<div class="flex items-center gap-2 min-w-0">' +
                '<span class="relative flex h-2.5 w-2.5 flex-shrink-0">' +
                '<span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>' +
                '<span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>' +
                '</span><span class="font-bold text-slate-800 text-sm truncate">' + className + '</span></div>' +
                '<div class="flex items-center gap-3 flex-shrink-0">' +
                '<span class="font-mono font-black text-emerald-700 text-lg tabular-nums" data-expires="' + expiresAtIso + '" data-role="countdown">--:--</span>' +
                '<a href="/student/qr-scan" class="bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold px-3 py-1.5 rounded-lg transition whitespace-nowrap">' +
                '<i class="fas fa-qrcode mr-1"></i> Scan Now</a></div>';
            rows.appendChild(div);
        }

        // Flip a class row's action cell between "Mark Present" and "Not Started"
        // based on whether the teacher currently has a live session running.
        function setRowButtonState(classId, isActive) {
            var cell = document.querySelector('[data-role="checkin-cell"][data-class-id="' + classId + '"]');
            if (!cell) return;
            if (isActive) {
                cell.innerHTML = '<a href="/student/checkin/manual/' + classId + '" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-1 px-2.5 rounded text-[11px] transition inline-block">' +
                    '<i class="fas fa-check mr-1"></i> Mark Present</a>';
            } else {
                cell.innerHTML = '<span class="bg-slate-100 text-slate-400 font-bold py-1 px-2.5 rounded text-[11px] inline-block cursor-not-allowed" title="Your teacher hasn\\'t started attendance for this class yet">' +
                    '<i class="fas fa-lock mr-1"></i> Not Started</span>';
            }
        }

        // Poll the server for (a) attendance just marked in another tab/device, and
        // (b) whether the teacher has started/stopped a session for any class —
        // and update the buttons + countdown banner live, with no page reload.
        function pollAttendanceStatus() {
            fetch('/student/attendance-status')
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    if (!data || !data.ok) return;
                    var markedIds = Array.isArray(data.marked_class_ids) ? data.marked_class_ids : [];
                    var activeSessions = Array.isArray(data.active_sessions) ? data.active_sessions : [];
                    var markedSet = {};
                    markedIds.forEach(function(cid) { markedSet[cid] = true; });
                    var activeMap = {};
                    activeSessions.forEach(function(s) { activeMap[s.class_id] = s; });

                    // Update every "Mark Present" / "Not Started" button on the page
                    document.querySelectorAll('[data-role="checkin-cell"]').forEach(function(cell) {
                        var cid = parseInt(cell.getAttribute('data-class-id'), 10);
                        setRowButtonState(cid, !!activeMap[cid]);
                    });

                    // Add/refresh countdown rows for newly-opened sessions
                    activeSessions.forEach(function(s) {
                        if (markedSet[s.class_id]) { removeRow(s.class_id); return; }
                        upsertSessionRow(s.class_id, s.class_name, s.expires_at);
                    });

                    // Drop rows for sessions the teacher has since stopped, or that
                    // just got marked present
                    document.querySelectorAll('[data-role="session-row"]').forEach(function(row) {
                        var cid = parseInt(row.getAttribute('data-class-id'), 10);
                        if (!activeMap[cid] || markedSet[cid]) removeRow(cid);
                    });
                })
                .catch(function() { /* silent — non-critical background poll */ });
        }

        tick();
        setInterval(tick, 1000);
        pollAttendanceStatus();
        setInterval(pollAttendanceStatus, 5000);
    })();
    </script>
    """

    return page_wrapper("Student Personal Hub Portal", body, is_student=True, student_context=student_ctx)


# =========================================================
# STUDENT EDIT PROFILE
# =========================================================
@app.route("/student/edit-profile", methods=["GET", "POST"])
def student_edit_profile():
    protect = student_required()
    if protect:
        return protect

    student_db_id = get_logged_student_db_id()
    student = get_student_row_by_db_id(student_db_id)
    error = ""

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        current_password = request.form.get("current_password", "").strip()
        new_password = request.form.get("new_password", "").strip()

        if not full_name:
            error = "Full name cannot be empty."
        elif current_password and not verify_password(student["password"], current_password):
            error = "Current password is incorrect."
        else:
            conn = get_db()
            cur = conn.cursor()
            new_image = student["image_file"]

            photo_file = request.files.get("photo")
            photo_b64 = request.form.get("photo_b64", "").strip()

            if photo_file and photo_file.filename:
                safe_id = sanitize_filename(student["student_id"])
                safe_name = sanitize_filename(full_name)
                ext = photo_file.filename.rsplit(".", 1)[-1].lower() if "." in photo_file.filename else "jpg"
                new_filename = f"{safe_id}_{safe_name}.{ext}"
                img_bytes = photo_file.read()
                public_url = None
                if not is_allowed_upload(img_bytes):
                    if is_ajax():
                        conn.close()
                        return ajax_err("Uploaded file is not a valid image.")
                else:
                    public_url = supabase_upload(new_filename, img_bytes)
                if public_url:
                    old_img = student["image_file"]
                    if old_img and old_img.startswith("http"):
                        supabase_delete(old_img.split("/")[-1])
                    new_image = public_url
            elif photo_b64:
                img_data = photo_b64
                if "," in img_data:
                    img_data = img_data.split(",")[1]
                img_data = img_data.replace(" ", "+")
                pad = len(img_data) % 4
                if pad:
                    img_data += "=" * (4 - pad)
                img_bytes = base64.b64decode(img_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    safe_id = sanitize_filename(student["student_id"])
                    safe_name = sanitize_filename(full_name)
                    new_filename = f"{safe_id}_{safe_name}.jpg"
                    _, enc = cv2.imencode('.jpg', frame)
                    public_url = supabase_upload(new_filename, enc.tobytes())
                    if public_url:
                        old_img = student["image_file"]
                        if old_img and old_img.startswith("http"):
                            supabase_delete(old_img.split("/")[-1])
                        new_image = public_url

            final_password = hash_password(new_password) if new_password else student["password"]
            cur.execute(
                "UPDATE students SET full_name=%s, password=%s, image_file=%s WHERE id=%s",
                (full_name, final_password, new_image, student_db_id)
            )
            conn.commit()
            conn.close()
            session["student_name"] = full_name
            load_known_faces()
            if is_ajax(): return ajax_ok("Profile updated successfully!", redirect_url="/student")
            return "<script>alert('Profile updated successfully!');window.location.href='/student';</script>"

    body = f"""
    <div class="max-w-lg mx-auto">
        <h1 class="text-2xl font-bold text-slate-800 mb-1">Edit My Profile</h1>
        <p class="text-sm text-slate-500 mb-5">Update your name, password, or profile photo.</p>
        {'<div class="mb-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm font-semibold">' + error + '</div>' if error else ''}

        <div class="flex items-center gap-4 mb-6 p-4 bg-slate-50 rounded-xl border">
            <img id="photoPreview" src="{supabase_public_url(student["image_file"])}" class="w-20 h-20 rounded-full object-cover border-2 border-blue-400 shadow">
            <div>
                <p class="font-semibold text-slate-700">{student["full_name"]}</p>
                <p class="text-xs text-slate-400 font-mono">ID: {student["student_id"]}</p>
            </div>
        </div>

        <form method="POST" enctype="multipart/form-data" class="space-y-4" id="profileForm" data-ajax>
                            {csrf_input_html()}
            <input type="hidden" name="photo_b64" id="photo_b64">
            <div>
                <label class="block text-sm font-semibold text-slate-700 mb-1">Full Name</label>
                <input type="text" name="full_name" value="{student["full_name"]}" class="w-full px-3 py-2 border rounded-lg" required>
            </div>
            <div>
                <label class="block text-sm font-semibold text-slate-700 mb-1">Current Password <span class="text-slate-400 font-normal">(required only to change password)</span></label>
                <input type="password" name="current_password" class="w-full px-3 py-2 border rounded-lg" placeholder="Enter current password to change it">
            </div>
            <div>
                <label class="block text-sm font-semibold text-slate-700 mb-1">New Password <span class="text-slate-400 font-normal">(leave blank to keep current)</span></label>
                <input type="password" name="new_password" class="w-full px-3 py-2 border rounded-lg" placeholder="Leave blank to keep current password">
            </div>
            <div class="border rounded-xl p-4 space-y-3 bg-slate-50">
                <p class="text-sm font-semibold text-slate-700">Profile Photo</p>
                <div class="flex flex-wrap gap-2">
                    <label class="bg-blue-600 hover:bg-blue-700 text-white font-semibold px-4 py-2 rounded-lg cursor-pointer text-sm">
                        📁 Upload from File
                        <input type="file" name="photo" accept="image/*" class="hidden" onchange="previewFile(this)">
                    </label>
                    <button type="button" onclick="openCamera()" class="bg-emerald-600 hover:bg-emerald-700 text-white font-semibold px-4 py-2 rounded-lg text-sm">📸 Take Selfie</button>
                </div>
                <div id="cameraArea" style="display:none;" class="space-y-2">
                    <video id="camVideo" autoplay playsinline muted class="w-full max-w-xs rounded-xl border bg-black"></video>
                    <div class="flex gap-2">
                        <button type="button" onclick="capturePhoto()" class="bg-emerald-600 text-white font-bold px-4 py-2 rounded-lg text-sm">✅ Capture</button>
                        <button type="button" onclick="closeCamera()" class="bg-slate-400 text-white font-bold px-4 py-2 rounded-lg text-sm">Cancel</button>
                    </div>
                    <div id="camStatus" class="text-xs text-slate-500"></div>
                </div>
            </div>
            <div class="flex gap-2 pt-2">
                <button type="submit" class="bg-blue-600 text-white font-bold py-2 px-5 rounded-lg hover:bg-blue-700">Save Changes</button>
                <a href="/student" class="inline-block bg-slate-100 text-slate-700 font-bold py-2 px-5 rounded-lg hover:bg-slate-200">Cancel</a>
            </div>
        </form>
    </div>
<script>
let camStream = null;
function previewFile(input) {{
    if (input.files && input.files[0]) {{
        document.getElementById('photoPreview').src = URL.createObjectURL(input.files[0]);
        document.getElementById('photo_b64').value = '';
    }}
}}
async function openCamera() {{
    document.getElementById('cameraArea').style.display = 'block';
    const camStatus = document.getElementById('camStatus');
    try {{
        let constraints = [
            {{ video: {{ facingMode: {{ exact: 'user' }} }}, audio: false }},
            {{ video: {{ facingMode: {{ ideal: 'user' }} }}, audio: false }},
            {{ video: true, audio: false }}
        ];
        let lastErr = null;
        for (let c of constraints) {{
            try {{ camStream = await navigator.mediaDevices.getUserMedia(c); break; }}
            catch(e) {{ lastErr = e; camStream = null; }}
        }}
        if (!camStream) throw lastErr;
        document.getElementById('camVideo').srcObject = camStream;
        await document.getElementById('camVideo').play();
        camStatus.innerText = '✅ Camera ready — click Capture when ready';
    }} catch(e) {{
        camStatus.innerText = '❌ Camera error: ' + e.message;
    }}
}}
function capturePhoto() {{
    const video = document.getElementById('camVideo');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL('image/jpeg');
    document.getElementById('photoPreview').src = dataUrl;
    document.getElementById('photo_b64').value = dataUrl;
    document.querySelector('input[name="photo"]').value = '';
    closeCamera();
}}
function closeCamera() {{
    if (camStream) {{ camStream.getTracks().forEach(t => t.stop()); camStream = null; }}
    document.getElementById('cameraArea').style.display = 'none';
}}
</script>
    """
    return page_wrapper("Edit My Profile", body, is_student=True, student_context=student)





# =========================================================
# STUDENT — TELEGRAM-STYLE CLASS GROUP CHAT
# =========================================================

def _tg_avatar(image_file, name, size=40):
    """Return an <img> tag or a letter-avatar div if image is missing."""
    url = supabase_public_url(image_file) if image_file else ""
    letter = (name or "?")[0].upper()
    colors = ["#2196F3","#E91E63","#9C27B0","#FF9800","#4CAF50","#00BCD4","#F44336","#3F51B5"]
    color = colors[sum(ord(c) for c in name) % len(colors)] if name else "#607D8B"
    if url:
        return f'<img src="{url}" style="width:{size}px;height:{size}px;border-radius:50%;object-fit:cover;flex-shrink:0;" onerror="this.style.display=\'none\';this.nextSibling.style.display=\'flex\'">' \
               f'<div class="tg-letter-av" style="display:none;width:{size}px;height:{size}px;background:{color};">{letter}</div>'
    return f'<div class="tg-letter-av" style="width:{size}px;height:{size}px;background:{color};">{letter}</div>'


@app.route("/student/classes")
def student_classes_hub():
    protect = student_required()
    if protect:
        return protect

    student_db_id = get_logged_student_db_id()
    student_ctx = get_student_row_by_db_id(student_db_id)
    school_id = get_current_school_id()
    classes = get_classes_for_student(student_db_id)

    # Build class list items for the Telegram left sidebar
    class_items = []
    for c in classes:
        classmates = get_students_in_class(c["id"])
        count = len(classmates)
        # Last message preview
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT student_name, comment, created_at FROM class_comments
            WHERE class_id=%s AND school_id=%s ORDER BY created_at DESC LIMIT 1
        """, (c["id"], school_id))
        last = cur.fetchone()
        conn.close()

        preview = ""
        ts_badge = ""
        if last:
            comment_text = last['comment'] or ''
            if comment_text:
                preview = f"{last['student_name'].split()[0]}: {comment_text[:40]}{'…' if len(comment_text)>40 else ''}"
            else:
                preview = f"{last['student_name'].split()[0]}: 📎 File"
            ts = last["created_at"]
            if hasattr(ts, "strftime"):
                from datetime import date as _date
                if ts.date() == _date.today():
                    ts_badge = ts.strftime("%I:%M %p")
                else:
                    ts_badge = ts.strftime("%b %d")
        else:
            preview = "No messages yet"

        letter = c["class_name"][0].upper()
        colors = ["#2196F3","#E91E63","#9C27B0","#FF9800","#4CAF50","#00BCD4","#F44336","#3F51B5"]
        color = colors[sum(ord(ch) for ch in c["class_name"]) % len(colors)]

        class_items.append({
            "id": c["id"],
            "name": c["class_name"],
            "subject": c.get("subject_name") or "",
            "teacher": c.get("teacher_display_name") or c.get("teacher_name") or "",
            "count": count,
            "preview": preview,
            "ts_badge": ts_badge,
            "letter": letter,
            "color": color,
        })

    list_html = ""
    for ci in class_items:
        list_html += f"""
        <a href="/student/class/{ci['id']}/feed" class="tg-chat-item" data-class-id="{ci['id']}">
            <div class="tg-chat-av" style="background:{ci['color']};">{ci['letter']}</div>
            <div class="tg-chat-info">
                <div class="tg-chat-top">
                    <span class="tg-chat-name">{ci['name']}</span>
                    <span class="tg-chat-ts">{ci['ts_badge']}</span>
                </div>
                <div class="tg-chat-preview">{ci['preview']}</div>
                <div class="tg-chat-sub">{ci['count']} members · {ci['subject']}</div>
            </div>
        </a>
        """

    if not list_html:
        list_html = """
        <div style="padding:40px 20px;text-align:center;color:#8e9aaf;">
            <div style="font-size:48px;margin-bottom:12px;">📭</div>
            <div style="font-weight:600;">No classes yet</div>
            <div style="font-size:13px;margin-top:6px;">Ask your admin to add you to a class</div>
        </div>
        """

    my_name = student_ctx["full_name"]
    my_img = supabase_public_url(student_ctx["image_file"])
    letter = my_name[0].upper()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Class Groups</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#1c2130;height:100vh;display:flex;overflow:hidden;}}

/* ── TG LEFT PANEL ── */
.tg-left{{width:340px;min-width:260px;background:#17212b;display:flex;flex-direction:column;border-right:1px solid #0d1117;flex-shrink:0;}}
.tg-header{{padding:12px 16px;background:#17212b;border-bottom:1px solid #0f1923;display:flex;align-items:center;gap:12px;}}
.tg-header-av{{width:38px;height:38px;border-radius:50%;object-fit:cover;flex-shrink:0;border:2px solid #2b5278;}}
.tg-header-letter{{width:38px;height:38px;border-radius:50%;background:#2b5278;display:flex;align-items:center;justify-content:center;font-weight:700;color:#fff;font-size:16px;flex-shrink:0;}}
.tg-header-name{{font-weight:600;color:#e4e7eb;font-size:14px;flex:1;}}
.tg-back-btn{{color:#5b9bd9;font-size:13px;font-weight:600;text-decoration:none;padding:6px 10px;border-radius:8px;transition:background 0.15s;}}
.tg-back-btn:hover{{background:rgba(91,155,217,0.12);}}
.tg-search{{padding:8px 12px;background:#17212b;border-bottom:1px solid #0f1923;}}
.tg-search input{{width:100%;background:#242f3d;border:none;border-radius:20px;padding:8px 14px;color:#e4e7eb;font-size:13px;outline:none;}}
.tg-search input::placeholder{{color:#5a6a7a;}}
.tg-chat-list{{flex:1;overflow-y:auto;}}
.tg-chat-item{{display:flex;align-items:center;gap:12px;padding:10px 14px;cursor:pointer;text-decoration:none;border-bottom:1px solid #1b2633;transition:background 0.12s;}}
.tg-chat-item:hover,.tg-chat-item.active{{background:#2b3c4e;}}
.tg-chat-av{{width:48px;height:48px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:20px;font-weight:700;color:#fff;flex-shrink:0;}}
.tg-chat-info{{flex:1;min-width:0;}}
.tg-chat-top{{display:flex;justify-content:space-between;align-items:center;}}
.tg-chat-name{{font-weight:600;color:#e4e7eb;font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:180px;}}
.tg-chat-ts{{font-size:11px;color:#5a6a7a;flex-shrink:0;}}
.tg-chat-preview{{font-size:13px;color:#7a8a9a;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:2px;}}
.tg-chat-sub{{font-size:11px;color:#4a5a6a;margin-top:2px;}}

/* ── TG RIGHT PANEL (welcome) ── */
.tg-right{{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;background:#0e1621;gap:16px;}}
.tg-right-icon{{font-size:80px;opacity:0.3;}}
.tg-right-title{{color:#4a5a6a;font-size:22px;font-weight:700;}}
.tg-right-sub{{color:#3a4a5a;font-size:14px;}}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{{width:4px;}}
::-webkit-scrollbar-track{{background:transparent;}}
::-webkit-scrollbar-thumb{{background:#2b3c4e;border-radius:4px;}}

/* ── MOBILE ── */
@media(max-width:640px){{
    .tg-left{{width:100%;}}
    .tg-right{{display:none;}}
}}
</style>
</head>
<body>
<!-- LEFT: class list -->
<div class="tg-left">
    <div class="tg-header">
        <a href="/student" class="tg-back-btn">← Back</a>
        <div style="flex:1;font-weight:700;color:#e4e7eb;font-size:15px;">Class Groups</div>
        {'<img class="tg-header-av" src="' + my_img + '">' if my_img else '<div class="tg-header-letter">' + letter + '</div>'}
    </div>
    <div class="tg-search">
        <input type="text" id="searchInput" placeholder="Search classes…" oninput="filterChats(this.value)">
    </div>
    <div class="tg-chat-list" id="chatList">
        {list_html}
    </div>
</div>

<!-- RIGHT: welcome -->
<div class="tg-right">
    <div class="tg-right-icon">💬</div>
    <div class="tg-right-title">Select a class group</div>
    <div class="tg-right-sub">Choose a class from the left to open the group chat</div>
</div>

<script>
function filterChats(q) {{
    q = q.toLowerCase();
    document.querySelectorAll('.tg-chat-item').forEach(el => {{
        el.style.display = el.textContent.toLowerCase().includes(q) ? '' : 'none';
    }});
}}
</script>
</body>
</html>"""
    return html


@app.route("/student/class/<int:class_id>/feed")
def student_class_feed(class_id):
    protect = student_required()
    if protect:
        return protect

    student_db_id = get_logged_student_db_id()
    school_id = get_current_school_id()
    student_ctx = get_student_row_by_db_id(student_db_id)

    if not student_belongs_to_class(student_db_id, class_id):
        return "<script>alert('You are not enrolled in this class.');window.location.href='/student/classes';</script>"

    class_row = get_class_by_id(class_id)
    if not class_row or class_row.get("school_id", school_id) != school_id:
        return "Class not found", 404

    classmates = get_students_in_class(class_id)
    my_name = student_ctx["full_name"]
    my_img = supabase_public_url(student_ctx["image_file"])

    # Fetch latest 80 messages oldest→newest
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, student_db_id, student_name, student_image, comment, created_at, poster_type, teacher_id_fk, is_priority, file_url, file_name, file_thumb_url
        FROM class_comments
        WHERE class_id=%s AND school_id=%s
        ORDER BY created_at ASC
        LIMIT 80
    """, (class_id, school_id))
    messages = cur.fetchall()
    cur.execute("""
        SELECT id, student_name, comment FROM class_comments
        WHERE class_id=%s AND school_id=%s AND is_priority=TRUE ORDER BY id DESC LIMIT 1
    """, (class_id, school_id))
    priority_row = cur.fetchone()
    conn.close()
    priority_banner_html = ""
    if priority_row:
        ptxt = priority_row["comment"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")[:120]
        pname = priority_row["student_name"]
        pid = priority_row["id"]
        priority_banner_html = (
            '<div id="priorityBanner" data-pin-id="' + str(pid) + '" '
            'style="background:linear-gradient(90deg,#92400e,#78350f);border-bottom:1px solid #d97706;'
            'padding:8px 16px;display:flex;align-items:center;gap:10px;flex-shrink:0;">'
            '<span style="font-size:16px;cursor:pointer;" onclick="scrollToPriority(' + str(pid) + ')">📌</span>'
            '<div style="flex:1;min-width:0;cursor:pointer;" onclick="scrollToPriority(' + str(pid) + ')">'
            '<div style="font-size:10px;color:#fbbf24;font-weight:700;text-transform:uppercase;">Pinned by teacher — ' + pname + '</div>'
            '<div style="font-size:13px;color:#fef3c7;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">' + ptxt + '</div>'
            '</div>'
            '<button onclick="dismissServerPin(' + str(pid) + ')" title="Dismiss for me" '
            'style="background:none;border:none;color:#92400e;cursor:pointer;font-size:18px;padding:0 4px;flex-shrink:0;line-height:1;" '
            'onmouseenter="this.style.color=\'#fef3c7\'" onmouseleave="this.style.color=\'#92400e\'">✕</button>'
            '</div>'
        )

    # Build initial messages HTML (with delete, priority ring, DM button)
    def _msg_html(m, is_me):
        ts = m["created_at"]
        if hasattr(ts, "strftime"):
            from datetime import date as _date
            if ts.date() == __import__("datetime").date.today():
                ts_str = ts.strftime("%I:%M %p")
            else:
                ts_str = ts.strftime("%b %d, %I:%M %p")
        else:
            ts_str = str(ts)[:16]
        txt = (m["comment"] or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        mid = m["id"]
        is_teacher = (m.get("poster_type") == "teacher")
        pr = m.get("is_priority", False)
        pring = "border:2px solid #f59e0b;box-shadow:0 0 10px rgba(245,158,11,0.3);" if pr else ""
        # File attachment
        furl = m.get("file_url") or ""
        fname = m.get("file_name") or ""
        fthumb = m.get("file_thumb_url") or ""
        fhtml = render_file_html(furl, fname, fthumb)
        txt_html = f'<div class="tg-bubble-text">{txt}</div>' if txt else ''
        if is_me and not is_teacher:
            return (f'<div class="tg-msg-row tg-mine" id="msg-{mid}">' +
                    f'<div class="tg-bubble tg-bubble-mine" style="position:relative;{pring}" ' +
                    'onmousedown="startLongPress(event,this)" onmouseup="cancelLongPress()" onmouseleave="cancelLongPress()" ' +
                    'ontouchstart="startLongPress(event,this)" ontouchend="cancelLongPress()" ontouchcancel="cancelLongPress()" ontouchmove="markPressMoved()">' +
                    f'{txt_html}{fhtml}' +
                    f'<div class="tg-bubble-ts">{ts_str} ✓✓</div>' +
                    f'<button onclick="event.stopPropagation();deleteMsg({mid})" class="chat-del-btn">✕</button>' +
                    f'</div></div>')
        elif is_teacher:
            t_profile_href = f"/student/teacher-profile/{m['teacher_id_fk']}" if m.get('teacher_id_fk') else "javascript:void(0)"
            return (f'<div class="tg-msg-row tg-theirs" id="msg-{mid}">' +
                    f'<div class="tg-av-wrap"><a href="{t_profile_href}" style="display:contents;"><div class="tg-letter-av" style="width:34px;height:34px;background:#7c3aed;cursor:pointer;">&#128104;&#8205;&#127979;</div></a></div>' +
                    f'<div><div class="tg-sender-name" style="color:#a78bfa;"><a href="{t_profile_href}" style="color:inherit;text-decoration:none;">{m["student_name"]}</a> ' +
                    f'<span style="font-size:10px;background:#4c1d95;color:#c4b5fd;padding:1px 6px;border-radius:6px;margin-left:4px;">Teacher</span></div>' +
                    f'<div class="tg-bubble" style="max-width:min(420px,72vw);padding:8px 12px 4px;border-radius:4px 16px 16px 16px;background:#2d1b69;border:1px solid #4c1d95;word-break:break-word;box-shadow:0 1px 4px rgba(0,0,0,0.25);{pring}">' +
                    f'{txt_html}{fhtml}' +
                    f'<div class="tg-bubble-ts">{ts_str}</div></div></div></div>')
        else:
            av = _tg_avatar(m["student_image"], m["student_name"], 34)
            nc = _name_color(m["student_name"])
            sid = m["student_db_id"]
            return (f'<div class="tg-msg-row tg-theirs" id="msg-{mid}">' +
                    f'<div class="tg-av-wrap"><a href="/student/classmate/{sid}" style="display:contents;">{av}</a></div>' +
                    f'<div><div class="tg-sender-name" style="color:{nc};">{m["student_name"]} ' +
                    f'<a href="/student/dm/{sid}" style="margin-left:6px;font-size:10px;background:#1e3a5f;color:#5b9bd9;padding:1px 7px;border-radius:6px;text-decoration:none;">&#128172; DM</a></div>' +
                    f'<div class="tg-bubble tg-bubble-theirs" style="{pring}">' +
                    f'{txt_html}{fhtml}' +
                    f'<div class="tg-bubble-ts">{ts_str}</div></div></div></div>')

    def _name_color(name):
        colors = ["#5b9bd9","#e8699a","#a876d8","#f4a623","#52c97f","#4db8d4","#e8645b","#7986cb"]
        return colors[sum(ord(c) for c in name) % len(colors)]

    msgs_html = "".join(_msg_html(m, m["student_db_id"] == student_db_id) for m in messages)
    last_id = messages[-1]["id"] if messages else 0
    student_db_id = int(student_db_id) if student_db_id else 0

    # Members sidebar HTML — priority star + DM button
    members_html = ""
    for s in classmates:
        is_me = s["id"] == student_db_id
        av = _tg_avatar(s["image_file"], s["full_name"], 36)
        me_badge = '<span class="tg-me-badge">You</span>' if is_me else ''
        is_p = s.get("is_priority", False)
        star = f'<span id="member-priority-{s["id"]}"><span style="color:#f59e0b;font-size:13px;margin-left:3px;" title="High Priority Student">⭐</span></span>' if is_p else f'<span id="member-priority-{s["id"]}"></span>'
        dm_link = '' if is_me else f'<a href="/student/dm/{s["id"]}" style="margin-left:auto;font-size:11px;background:#1e3a5f;color:#5b9bd9;padding:3px 9px;border-radius:8px;text-decoration:none;flex-shrink:0;" id="dmbadge-{s["id"]}">&#128172; DM</a>'
        link = "javascript:void(0)" if is_me else f"/student/classmate/{s['id']}"
        members_html += (f'<div style="display:flex;align-items:center;padding:4px 12px;">' +
            f'<a href="{link}" class="tg-member-item" style="flex:1;padding:6px 0;background:none;border-radius:0;{"cursor:default;" if is_me else ""}">' +
            f'<div class="tg-member-av">{av}</div>' +
            f'<div class="tg-member-info"><div class="tg-member-name">{s["full_name"]}{star} {me_badge}</div>' +
            f'<div class="tg-member-id">ID: {s["student_id"]}</div></div></a>{dm_link}</div>')
    csubject = class_row.get("subject_name") or ""
    cteacher = class_row.get("teacher_display_name") or class_row.get("teacher_name") or ""
    cdept = class_row.get("department") or ""
    cname = class_row.get("class_name") or "Class"
    letter = cname[0].upper()
    colors = ["#2196F3","#E91E63","#9C27B0","#FF9800","#4CAF50","#00BCD4","#F44336","#3F51B5"]
    color = colors[sum(ord(c) for c in cname) % len(colors)]

    EMOJIS = ["😀","😂","🥰","😎","🤔","👍","👏","🙌","🔥","💯","❤️","🎉","📚","✅","🤝","😅","🙏","💪","😴","🎓"]
    emoji_btns = "".join(f'<button class="tg-emoji-btn" onclick="insertEmoji(\'{e}\')">{e}</button>' for e in EMOJIS)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<title>{cname} · Group Chat</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
html{{height:100%;height:-webkit-fill-available;}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0e1621;height:100vh;height:100dvh;display:flex;flex-direction:column;overflow:hidden;color:#e4e7eb;}}

/* ── TOPBAR ── */
.tg-topbar{{
    height:56px;background:#17212b;border-bottom:1px solid #0f1923;
    display:flex;align-items:center;padding:0 16px;gap:12px;flex-shrink:0;
    box-shadow:0 1px 8px rgba(0,0,0,0.3);
}}
.tg-topbar-av{{
    width:40px;height:40px;border-radius:50%;display:flex;align-items:center;
    justify-content:center;font-size:18px;font-weight:700;color:#fff;flex-shrink:0;
    background:{color};cursor:pointer;
}}
.tg-topbar-info{{flex:1;min-width:0;cursor:pointer;}}
.tg-topbar-title{{font-weight:700;font-size:15px;color:#e4e7eb;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.tg-topbar-sub{{font-size:12px;color:#5a8ebd;margin-top:1px;}}
.tg-back{{color:#5b9bd9;font-size:13px;font-weight:600;text-decoration:none;padding:6px 10px;border-radius:8px;transition:background 0.15s;white-space:nowrap;}}
.tg-back:hover{{background:rgba(91,155,217,0.12);}}
.tg-members-toggle{{
    background:none;border:none;cursor:pointer;color:#5a8ebd;font-size:22px;
    padding:6px 8px;border-radius:8px;transition:background 0.15s;flex-shrink:0;
}}
.tg-members-toggle:hover{{background:rgba(91,155,217,0.12);}}

/* ── LAYOUT ── */
.tg-body{{flex:1;display:flex;overflow:hidden;}}

/* ── MESSAGES AREA ── */
.tg-messages-wrap{{flex:1;display:flex;flex-direction:column;background:#0e1621;min-width:0;}}
.tg-chat-bg{{
    flex:1;overflow-y:auto;padding:16px 12px;display:flex;flex-direction:column;gap:2px;
    background-image:radial-gradient(ellipse at 20% 80%,rgba(37,99,235,0.04) 0%,transparent 60%),
                     radial-gradient(ellipse at 80% 20%,rgba(139,92,246,0.04) 0%,transparent 60%);
}}

/* ── MESSAGE ROWS ── */
.tg-msg-row{{display:flex;align-items:flex-end;gap:8px;margin-bottom:2px;}}
.tg-mine{{flex-direction:row-reverse;}}
.tg-theirs{{flex-direction:row;}}
.tg-av-wrap{{flex-shrink:0;width:34px;align-self:flex-end;}}
.tg-letter-av{{border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;color:#fff;font-size:13px;flex-shrink:0;}}
.chat-del-btn{{position:absolute;top:-10px;right:-10px;background:#f87171;border:2px solid #0e1621;color:#fff;font-size:11px;cursor:pointer;opacity:0;transform:scale(0.7);transition:opacity 0.15s,transform 0.15s;padding:0;width:22px;height:22px;border-radius:50%;line-height:1;display:flex;align-items:center;justify-content:center;pointer-events:none;}}
.tg-bubble:hover .chat-del-btn,.tg-bubble.selected .chat-del-btn{{opacity:1;transform:scale(1);pointer-events:auto;}}
.tg-bubble.selected{{outline:2px solid #f87171;outline-offset:2px;}}
.tg-sender-name{{font-size:12px;font-weight:700;margin-bottom:3px;padding-left:2px;}}

/* ── BUBBLES ── */
.tg-bubble{{
    max-width:min(420px,72vw);padding:8px 12px 4px;border-radius:16px;
    position:relative;word-break:break-word;
    box-shadow:0 1px 4px rgba(0,0,0,0.25);
}}
.tg-bubble-mine{{
    background:#2b5278;border-radius:16px 4px 16px 16px;
}}
.tg-bubble-theirs{{
    background:#182533;border:1px solid #1e3048;border-radius:4px 16px 16px 16px;
}}
.tg-bubble-text{{font-size:14px;line-height:1.5;color:#e4e7eb;}}
.tg-bubble-ts{{
    font-size:10px;color:#7a9bbf;text-align:right;margin-top:4px;
}}

/* ── DATE DIVIDER ── */
.tg-date-divider{{
    text-align:center;margin:12px 0 8px;
}}
.tg-date-divider span{{
    background:#17212b;color:#5a8ebd;font-size:11px;font-weight:600;
    padding:4px 12px;border-radius:12px;border:1px solid #1e3048;
}}

/* ── COMPOSER ── */
.tg-composer{{
    background:#17212b;border-top:1px solid #0f1923;padding:10px 12px;padding-bottom:max(10px,env(safe-area-inset-bottom));flex-shrink:0;
}}
.tg-composer-row{{display:flex;align-items:flex-end;gap:8px;}}
.tg-emoji-toggle{{
    background:none;border:none;font-size:22px;cursor:pointer;padding:6px;
    border-radius:50%;transition:background 0.15s;flex-shrink:0;line-height:1;
}}
.tg-emoji-toggle:hover{{background:rgba(255,255,255,0.07);}}
.tg-input{{
    flex:1;background:#242f3d;border:none;border-radius:20px;
    padding:10px 16px;color:#e4e7eb;font-size:16px;outline:none;
    resize:none;max-height:120px;min-height:42px;line-height:1.4;
    font-family:inherit;
}}
.tg-input::placeholder{{color:#4a5a6a;}}
.tg-send-btn{{
    width:42px;height:42px;background:#2b5278;border:none;border-radius:50%;
    cursor:pointer;display:flex;align-items:center;justify-content:center;
    flex-shrink:0;transition:background 0.15s;font-size:18px;
}}
.tg-send-btn:hover{{background:#3a6a9a;}}
.tg-send-btn:disabled{{background:#1e2f3d;cursor:not-allowed;}}
.tg-emoji-panel{{
    display:none;background:#1a2635;border:1px solid #1e3048;border-radius:12px;
    padding:10px;margin-bottom:8px;flex-wrap:wrap;gap:4px;max-height:140px;overflow-y:auto;
}}
.tg-emoji-panel.open{{display:flex;}}
.tg-emoji-btn{{
    background:none;border:none;font-size:22px;cursor:pointer;
    padding:4px;border-radius:8px;transition:background 0.12s;line-height:1;
}}
.tg-emoji-btn:hover{{background:rgba(255,255,255,0.08);}}

/* ── MEMBERS PANEL ── */
.tg-members-panel{{
    width:300px;background:#17212b;border-left:1px solid #0f1923;
    display:flex;flex-direction:column;flex-shrink:0;
    transition:transform 0.25s ease;
}}
.tg-members-panel.hidden{{display:none;}}
.tg-members-header{{padding:16px;border-bottom:1px solid #0f1923;}}
.tg-members-class-av{{
    width:64px;height:64px;border-radius:50%;display:flex;align-items:center;justify-content:center;
    font-size:28px;font-weight:700;color:#fff;margin:0 auto 10px;background:{color};
}}
.tg-members-class-name{{font-weight:700;font-size:15px;text-align:center;color:#e4e7eb;}}
.tg-members-class-info{{font-size:12px;color:#5a8ebd;text-align:center;margin-top:4px;line-height:1.5;}}
.tg-members-list{{flex:1;overflow-y:auto;padding:8px;}}
.tg-members-title{{font-size:11px;font-weight:700;color:#5a8ebd;text-transform:uppercase;letter-spacing:0.05em;padding:8px 8px 4px;}}
.tg-member-item{{display:flex;align-items:center;gap:10px;padding:8px;border-radius:10px;text-decoration:none;transition:background 0.12s;cursor:pointer;}}
.tg-member-item:hover{{background:#1e3048;}}
.tg-member-av{{flex-shrink:0;}}
.tg-member-info{{min-width:0;}}
.tg-member-name{{font-size:13px;font-weight:600;color:#c8d8e8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:180px;}}
.tg-member-id{{font-size:11px;color:#4a5a6a;font-family:monospace;}}
.tg-me-badge{{
    display:inline-block;background:#2b5278;color:#7bb8ef;font-size:10px;font-weight:700;
    padding:1px 6px;border-radius:6px;margin-left:4px;
}}

/* ── SCROLL SNAP ── */
::-webkit-scrollbar{{width:4px;}}
::-webkit-scrollbar-track{{background:transparent;}}
::-webkit-scrollbar-thumb{{background:#2b3c4e;border-radius:4px;}}

/* ── TYPING INDICATOR ── */
.tg-typing{{display:none;padding:4px 12px;}}
.tg-typing.show{{display:block;}}
.tg-typing span{{font-size:12px;color:#5a8ebd;font-style:italic;}}

/* ── TOAST ── */
.tg-toast{{
    position:fixed;bottom:80px;left:50%;transform:translateX(-50%);
    background:#2b5278;color:#e4e7eb;font-size:13px;font-weight:600;
    padding:8px 20px;border-radius:20px;opacity:0;transition:opacity 0.3s;pointer-events:none;z-index:999;
}}
.tg-toast.show{{opacity:1;}}

/* ── MOBILE ── */
@media(max-width:700px){{
    .tg-members-panel{{
        position:fixed;right:0;top:0;bottom:0;z-index:500;width:85vw;max-width:300px;
        transform:translateX(100%);
    }}
    .tg-members-panel.open{{transform:translateX(0);display:flex;}}
    .tg-members-panel.hidden{{transform:translateX(100%);}}
    .tg-bubble{{max-width:80vw;}}
}}
</style>
</head>
<body>

<!-- TOPBAR -->
<div class="tg-topbar">
    <a href="/student/classes" class="tg-back">← Back</a>
    <div class="tg-topbar-av" onclick="toggleMembers()" title="View members">{letter}</div>
    <div class="tg-topbar-info" onclick="toggleMembers()">
        <div class="tg-topbar-title">{cname}</div>
        <div class="tg-topbar-sub">{len(classmates)} members{(' · ' + csubject) if csubject else ''}</div>
    </div>
    <button class="tg-members-toggle" onclick="toggleMembers()" title="Members">☰</button>
</div>

<div class="tg-body">
    <!-- MESSAGES -->
    <div class="tg-messages-wrap">
        {priority_banner_html}
        <div class="tg-chat-bg" id="chatBg">
            {'<div class="tg-date-divider"><span>Welcome to ' + cname + '</span></div>' if not messages else ''}
            {msgs_html if msgs_html else '<div style="text-align:center;color:#3a4a5a;padding:40px 20px;"><div style="font-size:40px;margin-bottom:10px;">👋</div><div style="font-size:14px;">No messages yet. Say hello!</div></div>'}
            <div id="messagesEnd"></div>
        </div>
        <div class="tg-typing" id="typingIndicator"><span>someone is typing…</span></div>

        <!-- COMPOSER -->
        <div class="tg-composer">
            <div class="tg-emoji-panel" id="emojiPanel">{emoji_btns}</div>
            <div id="classFilePreview" style="display:none;background:#182533;margin:0 12px 4px;border-radius:8px;padding:6px 10px;font-size:12px;color:#5b9bd9;align-items:center;gap:6px;"></div>
            <div class="tg-composer-row">
                <button class="tg-emoji-toggle" onclick="toggleEmoji()" title="Emoji">🙂</button>
                <input type="file" id="classFileInput" style="display:none" onchange="previewClassFile(this)">
                <button class="tg-emoji-toggle" onclick="document.getElementById('classFileInput').click()" title="Attach file" style="font-size:18px;">📎</button>
                <textarea class="tg-input" id="msgInput" placeholder="Message {cname}…" rows="1"
                    oninput="autoResize(this)" onkeydown="handleKey(event)"></textarea>
                <button class="tg-send-btn" id="sendBtn" onclick="sendMessage()" title="Send">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
                </button>
            </div>
        </div>
    </div>

    <!-- MEMBERS PANEL -->
    <div class="tg-members-panel" id="membersPanel">
        <div class="tg-members-header">
            <div class="tg-members-class-av">{letter}</div>
            <div class="tg-members-class-name">{cname}</div>
            <div class="tg-members-class-info">
                {'👨‍🏫 ' + cteacher + '<br>' if cteacher else ''}
                {'📖 ' + csubject + '<br>' if csubject else ''}
                {'🏛️ ' + cdept if cdept else ''}
            </div>
        </div>
        <div class="tg-members-list">
            <div class="tg-members-title">Members · {len(classmates)}</div>
            {members_html}
        </div>
    </div>
</div>

<div class="tg-toast" id="toast"></div>

<script>
// ── GLOBAL CSRF PROTECTION FOR AJAX ──
(function() {{
    function getCookie(name) {{
        const m = document.cookie.match('(^|;)\\s*' + name + '\\s*=\\s*([^;]+)');
        return m ? decodeURIComponent(m.pop()) : '';
    }}
    const _origFetch = window.fetch;
    window.fetch = function(input, init) {{
        init = init || {{}};
        const method = (init.method || 'GET').toUpperCase();
        if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(method)) {{
            init.headers = Object.assign({{}}, init.headers || {{}});
            if (!init.headers['X-CSRF-Token'] && !init.headers['x-csrf-token']) {{
                init.headers['X-CSRF-Token'] = getCookie('csrf_token');
            }}
        }}
        return _origFetch(input, init);
    }};
}})();
const MY_DB_ID = {student_db_id};
const CLASS_ID = {class_id};
let lastId = {last_id};
let membersOpen = false;
let pollTimer = null;

// ── Auto-resize textarea ──
function autoResize(el) {{
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}}

// ── Handle Enter key ──
function handleKey(e) {{
    if (e.key === 'Enter' && !e.shiftKey) {{
        e.preventDefault();
        sendMessage();
    }}
}}

// ── Emoji panel ──
function toggleEmoji() {{
    const p = document.getElementById('emojiPanel');
    p.classList.toggle('open');
    if (p.classList.contains('open')) document.getElementById('msgInput').focus();
}}
function insertEmoji(em) {{
    const inp = document.getElementById('msgInput');
    const start = inp.selectionStart, end = inp.selectionEnd;
    inp.value = inp.value.slice(0,start) + em + inp.value.slice(end);
    inp.selectionStart = inp.selectionEnd = start + em.length;
    inp.focus();
    autoResize(inp);
}}

// ── Members panel ──
function toggleMembers() {{
    membersOpen = !membersOpen;
    const p = document.getElementById('membersPanel');
    if (membersOpen) {{ p.classList.add('open'); p.classList.remove('hidden'); }}
    else {{ p.classList.remove('open'); }}
}}

// ── Toast ──
function showToast(msg) {{
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 2200);
}}

// ── Name color ──
const NAME_COLORS = ["#5b9bd9","#e8699a","#a876d8","#f4a623","#52c97f","#4db8d4","#e8645b","#7986cb"];
function nameColor(name) {{
    let s = 0; for(let c of name) s += c.charCodeAt(0);
    return NAME_COLORS[s % NAME_COLORS.length];
}}

// ── Render a single message bubble ──
function buildFileHtml(file_url, file_name, file_thumb_url) {{
    if (!file_url) return '';
    const ext = (file_name || '').split('.').pop().toLowerCase();
    const isImage = ['jpg','jpeg','png','gif','webp'].includes(ext);
    const isVideo = ['mp4','mov','avi','webm','mkv'].includes(ext);
    if (isImage) {{
        return `<div style="margin-top:5px;">
            <a href="${{file_url}}" target="_blank">
                <img src="${{file_url}}" style="max-width:220px;max-height:180px;border-radius:8px;display:block;object-fit:cover;">
            </a>
        </div>`;
    }} else if (isVideo) {{
        const posterAttr = file_thumb_url ? ` poster="${{file_thumb_url}}"` : '';
        return `<div style="margin-top:5px;border-radius:8px;overflow:hidden;max-width:260px;background:#000;position:relative;">
            <video preload="metadata" playsinline${{posterAttr}} style="display:block;max-width:260px;max-height:200px;border-radius:8px;width:100%;cursor:pointer;"
                onclick="if(this.requestFullscreen){{this.requestFullscreen();}}else if(this.webkitRequestFullscreen){{this.webkitRequestFullscreen();}}this.controls=true;this.play();">
                <source src="${{file_url}}" type="video/${{ext === 'mov' ? 'quicktime' : ext}}">
            </video>
        </div>`;
    }} else {{
        const icons = {{pdf:'📕',doc:'📝',docx:'📝',xls:'📊',xlsx:'📊',ppt:'📋',pptx:'📋',zip:'🗜️',rar:'🗜️',txt:'📄',mp3:'🎵',wav:'🎵'}};
        const icon = icons[ext] || '📄';
        return `<div style="margin-top:5px;">
            <a href="${{file_url}}" target="_blank" style="display:inline-flex;align-items:center;gap:8px;background:rgba(0,0,0,0.2);border:1px solid rgba(91,155,217,0.3);border-radius:10px;padding:8px 12px;text-decoration:none;max-width:220px;">
                <span style="font-size:22px;flex-shrink:0;">${{icon}}</span>
                <span style="color:#5b9bd9;font-size:12px;word-break:break-all;line-height:1.3;">${{file_name || 'Download file'}}</span>
            </a>
        </div>`;
    }}
}}

function renderMsg(m) {{
    const isMe = m.student_db_id === MY_DB_ID && m.poster_type !== 'teacher';
    const ts = m.ts_display || '';
    const txt = m.comment ? m.comment.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') : '';
    const txtHtml = txt ? `<div class="tg-bubble-text">${{txt}}</div>` : '';
    const fHtml = buildFileHtml(m.file_url, m.file_name, m.file_thumb_url);

    if (isMe) {{
        return `<div class="tg-msg-row tg-mine" id="msg-${{m.id}}">
            <div class="tg-bubble tg-bubble-mine" style="position:relative;" onmousedown="startLongPress(event,this)" onmouseup="cancelLongPress()" onmouseleave="cancelLongPress()" ontouchstart="startLongPress(event,this)" ontouchend="cancelLongPress()" ontouchcancel="cancelLongPress()" ontouchmove="markPressMoved()">
                ${{txtHtml}}${{fHtml}}
                <div class="tg-bubble-ts">${{ts}} ✓✓</div>
                <button onclick="event.stopPropagation();deleteMsg(${{m.id}})" class="chat-del-btn">✕</button>
            </div>
        </div>`;
    }} else if (m.poster_type === 'teacher') {{
        const teacherProfileUrl = m.teacher_id_fk ? `/student/teacher-profile/${{m.teacher_id_fk}}` : 'javascript:void(0)';
        return `<div class="tg-msg-row tg-theirs" id="msg-${{m.id}}">
            <div class="tg-av-wrap">
                <a href="${{teacherProfileUrl}}" style="display:contents;">
                    <div class="tg-letter-av" style="width:34px;height:34px;background:#7c3aed;cursor:pointer;">👨‍🏫</div>
                </a>
            </div>
            <div>
                <div class="tg-sender-name" style="color:#a78bfa;">
                    <a href="${{teacherProfileUrl}}" style="color:inherit;text-decoration:none;">${{m.student_name}}</a>
                    <span style="font-size:10px;background:#4c1d95;color:#c4b5fd;padding:1px 6px;border-radius:6px;margin-left:4px;">Teacher</span>
                </div>
                <div class="tg-bubble" style="max-width:min(420px,72vw);padding:8px 12px 4px;border-radius:4px 16px 16px 16px;background:#2d1b69;border:1px solid #4c1d95;word-break:break-word;box-shadow:0 1px 4px rgba(0,0,0,0.25);">
                    ${{txtHtml}}${{fHtml}}
                    <div class="tg-bubble-ts">${{ts}}</div>
                </div>
            </div>
        </div>`;
    }} else {{
        const color = nameColor(m.student_name);
        const avStyle = m.student_image
            ? `<img src="${{m.student_image}}" style="width:34px;height:34px;border-radius:50%;object-fit:cover;" onerror="this.style.display='none'">`
            : `<div class="tg-letter-av" style="width:34px;height:34px;background:${{color}};">${{m.student_name[0].toUpperCase()}}</div>`;
        return `<div class="tg-msg-row tg-theirs" id="msg-${{m.id}}">
            <div class="tg-av-wrap">
                <a href="/student/classmate/${{m.student_db_id}}" style="display:contents;">${{avStyle}}</a>
            </div>
            <div>
                <div class="tg-sender-name" style="color:${{color}};">${{m.student_name}}</div>
                <div class="tg-bubble tg-bubble-theirs">
                    ${{txtHtml}}${{fHtml}}
                    <div class="tg-bubble-ts">${{ts}}</div>
                </div>
            </div>
        </div>`;
    }}
}}

// ── File attach for class chat ──
let pendingClassFile = null;
function previewClassFile(input) {{
    const f = input.files[0];
    if (!f) return;
    pendingClassFile = f;
    const preview = document.getElementById('classFilePreview');
    preview.style.display = 'flex';
    preview.innerHTML = `📎 ${{f.name}} <button onclick="clearClassFile()" style="background:none;border:none;color:#f87171;cursor:pointer;margin-left:auto;">✕</button>`;
}}
function clearClassFile() {{
    pendingClassFile = null;
    document.getElementById('classFileInput').value = '';
    const preview = document.getElementById('classFilePreview');
    preview.style.display = 'none';
    preview.innerHTML = '';
}}

// ── Send message ──
async function sendMessage() {{
    const input = document.getElementById('msgInput');
    const btn = document.getElementById('sendBtn');
    const text = input.value.trim();
    if (!text && !pendingClassFile) return;

    input.value = '';
    autoResize(input);
    btn.disabled = true;
    document.getElementById('emojiPanel').classList.remove('open');

    const fd = new FormData();
    if (text) fd.append('comment', text);
    if (pendingClassFile) fd.append('file', pendingClassFile);
    clearClassFile();

    try {{
        const res = await fetch('/student/class/{class_id}/comment', {{
            method: 'POST',
            body: fd
        }});
        const data = await res.json();
        if (!data.ok) {{ showToast(data.error || 'Could not send'); input.value = text; }}
        else {{
            if (data.msg && !document.getElementById('msg-' + data.msg.id)) {{
                const bg = document.getElementById('chatBg');
                const end = document.getElementById('messagesEnd');
                const div = document.createElement('div');
                div.innerHTML = renderMsg(data.msg);
                bg.insertBefore(div.firstElementChild, end);
                lastId = Math.max(lastId, data.msg.id);
                scrollToBottom();
            }}
        }}
    }} catch(e) {{
        showToast('Network error');
        input.value = text;
    }} finally {{
        btn.disabled = false;
        input.focus();
    }}
}}

// ── Poll for new messages (and removals) ──
async function pollMessages() {{
    try {{
        const bg = document.getElementById('chatBg');
        const visibleIds = Array.from(bg.querySelectorAll('[id^="msg-"]'))
            .map(el => el.id.slice(4))
            .filter(id => /^\d+$/.test(id));
        const url = '/student/class/{class_id}/messages?after=' + lastId +
            (visibleIds.length ? '&ids=' + visibleIds.join(',') : '');
        const res = await fetch(url);
        const data = await res.json();

        if (data.removed_ids && data.removed_ids.length) {{
            data.removed_ids.forEach(id => {{
                document.getElementById('msg-' + id)?.remove();
            }});
        }}

        if (data.messages && data.messages.length) {{
            const end = document.getElementById('messagesEnd');
            data.messages.forEach(m => {{
                if (document.getElementById('msg-' + m.id)) return;
                const div = document.createElement('div');
                div.innerHTML = renderMsg(m);
                bg.insertBefore(div.firstElementChild, end);
                lastId = Math.max(lastId, m.id);
            }});
            scrollToBottom();
        }}
    }} catch(e) {{ /* silent */ }}
}}

function scrollToBottom() {{
    const bg = document.getElementById('chatBg');
    bg.scrollTop = bg.scrollHeight;
}}

// ── Init ──
scrollToBottom();
pollMessages();
pollTimer = setInterval(pollMessages, 1000);

// Poll for priority message changes every 8s
// ── Teacher pin dismiss helpers (localStorage, per-class) ──
const DISMISS_KEY = 'class_pin_dismissed_{class_id}';
function getDismissedPinId() {{
    try {{ return parseInt(localStorage.getItem(DISMISS_KEY) || '0', 10) || null; }} catch(e) {{ return null; }}
}}
function dismissServerPin(msgId) {{
    try {{ localStorage.setItem(DISMISS_KEY, String(msgId)); }} catch(e) {{}}
    const banner = document.getElementById('priorityBanner');
    if (banner) banner.remove();
}}

async function pollPriority() {{
    try {{
        const res = await fetch('/student/class/{class_id}/priority-message');
        const data = await res.json();
        const banner = document.getElementById('priorityBanner');
        const wrap = document.querySelector('.tg-messages-wrap');
        if (data.msg) {{
            const mid = data.msg.id;
            // If the user dismissed this exact pin, don't re-show it
            if (getDismissedPinId() === mid) {{
                if (banner) banner.remove();
                return;
            }}
            // If teacher pinned a NEW message, clear the old dismiss so it shows
            const prevDismissed = getDismissedPinId();
            if (prevDismissed && prevDismissed !== mid) {{
                try {{ localStorage.removeItem(DISMISS_KEY); }} catch(e) {{}}
            }}
            const ptxt = data.msg.text.substring(0, 120);
            const newHtml = `<div id="priorityBanner" data-pin-id="${{mid}}" style="background:linear-gradient(90deg,#92400e,#78350f);border-bottom:1px solid #d97706;padding:8px 16px;display:flex;align-items:center;gap:10px;flex-shrink:0;"><span style="font-size:16px;cursor:pointer;" onclick="scrollToPriority(${{mid}})">📌</span><div style="flex:1;min-width:0;cursor:pointer;" onclick="scrollToPriority(${{mid}})"><div style="font-size:10px;color:#fbbf24;font-weight:700;text-transform:uppercase;">Pinned by teacher — ${{data.msg.name}}</div><div style="font-size:13px;color:#fef3c7;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${{ptxt}}</div></div><button onclick="dismissServerPin(${{mid}})" title="Dismiss for me" style="background:none;border:none;color:#92400e;cursor:pointer;font-size:18px;padding:0 4px;flex-shrink:0;line-height:1;" onmouseenter="this.style.color='#fef3c7'" onmouseleave="this.style.color='#92400e'">✕</button></div>`;
            if (banner) {{
                banner.outerHTML = newHtml;
            }} else {{
                const chatBg = document.getElementById('chatBg');
                if (chatBg && wrap) wrap.insertBefore(document.createRange().createContextualFragment(newHtml), chatBg);
            }}
        }} else {{
            if (banner) banner.remove();
        }}
    }} catch(e) {{}}
}}
setInterval(pollPriority, 8000);

// On load: hide the banner immediately if this pin was already dismissed by this user
(function() {{
    const banner = document.getElementById('priorityBanner');
    if (!banner) return;
    const pinId = parseInt(banner.getAttribute('data-pin-id') || '0', 10);
    if (pinId && getDismissedPinId() === pinId) banner.remove();
}})();

function scrollToPriority(msgId) {{
    const el = document.getElementById('msg-' + msgId);
    if (el) {{
        el.scrollIntoView({{behavior:'smooth', block:'center'}});
        el.style.outline = '2px solid #f59e0b';
        setTimeout(() => el.style.outline = '', 2000);
    }}
}}

async function deleteMsg(msgId) {{
    if (!confirm('Delete this message?')) return;
    try {{
        const res = await fetch('/chat/delete-message/' + msgId, {{method:'POST'}});
        const data = await res.json();
        if (data.ok) {{
            document.getElementById('msg-' + msgId)?.remove();
        }} else {{
            alert(data.error || 'Could not delete message.');
        }}
    }} catch(e) {{ alert('Error deleting message.'); }}
}}

// ── Long-press a message bubble to select it and reveal its delete button (mobile-friendly) ──
let _pressTimer = null;
let _pressMoved = false;
function startLongPress(evt, bubbleEl) {{
    _pressMoved = false;
    clearTimeout(_pressTimer);
    _pressTimer = setTimeout(() => {{
        if (_pressMoved) return;
        const wasSelected = bubbleEl.classList.contains('selected');
        document.querySelectorAll('.tg-bubble.selected').forEach(b => b.classList.remove('selected'));
        if (!wasSelected) bubbleEl.classList.add('selected');
        if (navigator.vibrate) navigator.vibrate(15);
    }}, 500);
}}
function cancelLongPress() {{
    clearTimeout(_pressTimer);
}}
function markPressMoved() {{
    _pressMoved = true;
    clearTimeout(_pressTimer);
}}
document.addEventListener('click', () => {{
    document.querySelectorAll('.tg-bubble.selected').forEach(b => b.classList.remove('selected'));
}});
// ── Right-click context menu (desktop) ──────────────────────────────────
(function() {{
    // Inject the menu element once
    if (!document.getElementById('_ctxMenu')) {{
        const m = document.createElement('div');
        m.id = '_ctxMenu';
        m.style.cssText = [
            'position:fixed','z-index:9999','background:#1e2d3d','border:1px solid #2b4a6a',
            'border-radius:10px','box-shadow:0 8px 28px rgba(0,0,0,0.55)',
            'padding:4px 0','min-width:160px','display:none','user-select:none',
            'backdrop-filter:blur(4px)'
        ].join(';');
        m.innerHTML = `
            <div class="_ctx-item" id="_ctxSelect" style="display:flex;align-items:center;gap:10px;padding:9px 16px;cursor:pointer;font-size:14px;color:#c8d8e8;transition:background 0.12s;">
                <span style="font-size:15px;">✅</span> Select
            </div>
            <div style="height:1px;background:#2b3c4e;margin:3px 0;"></div>
            <div class="_ctx-item" id="_ctxPin" style="display:flex;align-items:center;gap:10px;padding:9px 16px;cursor:pointer;font-size:14px;color:#fbbf24;transition:background 0.12s;">
                <span style="font-size:15px;">📌</span> <span id="_ctxPinLabel">Pin for me</span>
            </div>
            <div style="height:1px;background:#2b3c4e;margin:3px 0;"></div>
            <div class="_ctx-item" id="_ctxDelete" style="display:flex;align-items:center;gap:10px;padding:9px 16px;cursor:pointer;font-size:14px;color:#f87171;transition:background 0.12s;">
                <span style="font-size:15px;">🗑️</span> Delete
            </div>`;
        // Hover effect
        m.querySelectorAll('._ctx-item').forEach(el => {{
            el.addEventListener('mouseenter', () => el.style.background = '#253647');
            el.addEventListener('mouseleave', () => el.style.background = '');
        }});
        document.body.appendChild(m);
    }}

    let _ctxTargetBubble = null;
    let _ctxTargetMsgId = null;
    const menu = document.getElementById('_ctxMenu');

    // ── Personal pin helpers (localStorage, per-class) ──
    const PIN_KEY = 'class_pin_{class_id}';
    function getMyPin() {{
        try {{ return JSON.parse(localStorage.getItem(PIN_KEY) || 'null'); }} catch(e) {{ return null; }}
    }}
    function setMyPin(msgId, text, name) {{
        localStorage.setItem(PIN_KEY, JSON.stringify({{id: msgId, text, name}}));
        renderMyPinBanner();
    }}
    function clearMyPin() {{
        localStorage.removeItem(PIN_KEY);
        renderMyPinBanner();
    }}
    function renderMyPinBanner() {{
        const pin = getMyPin();
        let existing = document.getElementById('_myPinBanner');
        const wrap = document.querySelector('.tg-messages-wrap');
        const chatBg = document.getElementById('chatBg');
        if (pin) {{
            const ptxt = (pin.text || '').substring(0, 100);
            const html = `<div id="_myPinBanner" style="background:linear-gradient(90deg,#1e3a5f,#162c47);border-bottom:1px solid #2b5278;padding:8px 16px;display:flex;align-items:center;gap:10px;flex-shrink:0;">
                <span style="font-size:16px;cursor:pointer;" onclick="scrollToPriority(${{pin.id}})">📌</span>
                <div style="flex:1;min-width:0;cursor:pointer;" onclick="scrollToPriority(${{pin.id}})">
                    <div style="font-size:10px;color:#5b9bd9;font-weight:700;text-transform:uppercase;">Pinned by you — ${{pin.name}}</div>
                    <div style="font-size:13px;color:#c8d8e8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${{ptxt}}</div>
                </div>
                <button onclick="clearMyPin()" title="Unpin" style="background:none;border:none;color:#5a6a7a;cursor:pointer;font-size:18px;padding:0 4px;flex-shrink:0;line-height:1;" onmouseenter="this.style.color='#f87171'" onmouseleave="this.style.color='#5a6a7a'">✕</button>
            </div>`;
            if (existing) {{
                existing.outerHTML = html;
            }} else {{
                // Insert above chatBg but below the server priority banner (if any)
                const serverBanner = document.getElementById('priorityBanner');
                const ref = serverBanner ? serverBanner.nextSibling : chatBg;
                if (wrap && ref) wrap.insertBefore(document.createRange().createContextualFragment(html), ref);
                else if (wrap && chatBg) wrap.insertBefore(document.createRange().createContextualFragment(html), chatBg);
            }}
        }} else {{
            if (existing) existing.remove();
        }}
    }}
    // Show personal pin on load
    renderMyPinBanner();

    function showCtxMenu(x, y, bubbleEl, msgId) {{
        _ctxTargetBubble = bubbleEl;
        _ctxTargetMsgId = msgId;

        // Update pin label: "Unpin" if this msg is already personally pinned
        const pin = getMyPin();
        const pinLabel = document.getElementById('_ctxPinLabel');
        if (pinLabel) pinLabel.textContent = (pin && pin.id === msgId) ? 'Unpin for me' : 'Pin for me';

        // Position: keep inside viewport
        menu.style.display = 'block';
        const vw = window.innerWidth, vh = window.innerHeight;
        const mw = menu.offsetWidth || 160, mh = menu.offsetHeight || 110;
        menu.style.left = (x + mw > vw ? vw - mw - 8 : x) + 'px';
        menu.style.top  = (y + mh > vh ? vh - mh - 8 : y) + 'px';

        // Highlight the bubble
        document.querySelectorAll('.tg-bubble.selected').forEach(b => b.classList.remove('selected'));
        if (bubbleEl) bubbleEl.classList.add('selected');
    }}

    function hideCtxMenu() {{
        menu.style.display = 'none';
        _ctxTargetBubble = null;
        _ctxTargetMsgId = null;
    }}

    // Select action
    document.getElementById('_ctxSelect').addEventListener('click', () => {{
        if (_ctxTargetBubble) {{
            const wasSelected = _ctxTargetBubble.classList.contains('selected');
            document.querySelectorAll('.tg-bubble.selected').forEach(b => b.classList.remove('selected'));
            if (!wasSelected) _ctxTargetBubble.classList.add('selected');
        }}
        hideCtxMenu();
    }});

    // Pin action (personal, localStorage)
    document.getElementById('_ctxPin').addEventListener('click', () => {{
        const id = _ctxTargetMsgId;
        const bubble = _ctxTargetBubble;
        hideCtxMenu();
        if (!id) return;
        const pin = getMyPin();
        if (pin && pin.id === id) {{
            clearMyPin();
        }} else {{
            // Extract text from the bubble
            const textEl = bubble ? bubble.querySelector('.tg-bubble-text') : null;
            const text = textEl ? textEl.textContent.trim() : '';
            // Extract sender name from the row
            const row = document.getElementById('msg-' + id);
            const nameEl = row ? row.querySelector('.tg-sender-name') : null;
            const name = nameEl ? nameEl.textContent.trim().split('\\n')[0].trim() : 'Message';
            setMyPin(id, text, name);
        }}
    }});

    // Delete action
    document.getElementById('_ctxDelete').addEventListener('click', async () => {{
        const id = _ctxTargetMsgId;
        hideCtxMenu();
        if (id && typeof deleteMsg === 'function') await deleteMsg(id);
    }});

    // Attach contextmenu to all bubbles (current + future via delegation)
    document.addEventListener('contextmenu', e => {{
        const bubble = e.target.closest('.tg-bubble');
        if (!bubble) {{ hideCtxMenu(); return; }}
        e.preventDefault();
        e.stopPropagation();
        // Extract msg id from the parent row's id ("msg-123")
        const row = bubble.closest('[id^="msg-"]');
        const msgId = row ? parseInt(row.id.replace('msg-', ''), 10) : null;
        showCtxMenu(e.clientX, e.clientY, bubble, msgId);
    }});

    // Close on outside click or scroll
    document.addEventListener('click', e => {{
        if (!menu.contains(e.target)) hideCtxMenu();
    }});
    document.addEventListener('scroll', hideCtxMenu, true);
    document.addEventListener('keydown', e => {{ if (e.key === 'Escape') hideCtxMenu(); }});
}})();


// Poll unread DM counts and update sidebar badges
async function pollDmUnread() {{
    try {{
        const res = await fetch('/student/dm/unread-counts');
        const counts = await res.json();
        for (const [sid, cnt] of Object.entries(counts)) {{
            const badge = document.getElementById('dmbadge-' + sid);
            if (badge && cnt > 0) {{
                badge.innerHTML = `&#128172; DM <span style="background:#ef4444;color:#fff;border-radius:9999px;padding:0 5px;font-size:10px;margin-left:3px;">${{cnt}}</span>`;
            }}
        }}
    }} catch(e) {{}}
}}
setInterval(pollDmUnread, 5000);
pollDmUnread();

// Poll member priority badges every 6s
async function pollMembers() {{
    try {{
        const res = await fetch('/student/class/{class_id}/members');
        const members = await res.json();
        members.forEach(m => {{
            const el = document.getElementById('member-priority-' + m.id);
            if (!el) return;
            el.innerHTML = m.is_priority
                ? '<span style="color:#f59e0b;font-size:13px;margin-left:3px;" title="High Priority Student">⭐</span>'
                : '';
        }});
    }} catch(e) {{}}
}}
setInterval(pollMembers, 6000);
pollMembers();

// Click outside to close members on mobile
document.addEventListener('click', e => {{
    const panel = document.getElementById('membersPanel');
    const toggle = document.querySelector('.tg-members-toggle');
    const av = document.querySelector('.tg-topbar-av');
    if (membersOpen && !panel.contains(e.target) && e.target !== toggle && e.target !== av) {{
        membersOpen = false;
        panel.classList.remove('open');
    }}
}});

window.addEventListener('beforeunload', () => clearInterval(pollTimer));
</script>
</body>
</html>"""
    return html


@app.route("/student/class/<int:class_id>/messages")
def student_class_messages(class_id):
    """Polling endpoint — returns new messages after a given id."""
    protect = student_required()
    if protect:
        return jsonify({"messages": []})

    student_db_id = get_logged_student_db_id()
    school_id = get_current_school_id()

    if not student_belongs_to_class(student_db_id, class_id):
        return jsonify({"messages": []})

    after = int(request.args.get("after", 0))

    # IDs the client currently has rendered on screen — used to detect
    # messages that were deleted (by anyone) since the client's last poll.
    visible_ids = []
    raw_ids = request.args.get("ids", "")
    if raw_ids:
        for part in raw_ids.split(","):
            part = part.strip()
            if part.isdigit():
                visible_ids.append(int(part))
        visible_ids = visible_ids[:500]  # safety cap

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, student_db_id, student_name, student_image, comment, created_at, poster_type, teacher_id_fk, is_priority, file_url, file_name, file_thumb_url
        FROM class_comments
        WHERE class_id=%s AND school_id=%s AND id>%s
        ORDER BY created_at ASC LIMIT 30
    """, (class_id, school_id, after))
    rows = cur.fetchall()

    removed_ids = []
    if visible_ids:
        placeholders = ",".join(["%s"] * len(visible_ids))
        cur.execute(
            f"SELECT id FROM class_comments WHERE class_id=%s AND school_id=%s AND id IN ({placeholders})",
            [class_id, school_id] + visible_ids
        )
        still_exist = {r["id"] for r in cur.fetchall()}
        removed_ids = [i for i in visible_ids if i not in still_exist]

    conn.close()

    from datetime import date as _date
    result = []
    for r in rows:
        ts = r["created_at"]
        if hasattr(ts, "strftime"):
            ts_display = ts.strftime("%I:%M %p") if ts.date() == _date.today() else ts.strftime("%b %d, %I:%M %p")
        else:
            ts_display = str(ts)[:16]
        result.append({
            "id": r["id"],
            "student_db_id": r["student_db_id"],
            "student_name": r["student_name"],
            "student_image": supabase_public_url(r["student_image"] or ""),
            "comment": r["comment"],
            "ts_display": ts_display,
            "poster_type": r.get("poster_type") or "student",
            "teacher_id_fk": r.get("teacher_id_fk"),
            "is_priority": r.get("is_priority") or False,
            "file_url": r.get("file_url") or "",
            "file_name": r.get("file_name") or "",
            "file_thumb_url": r.get("file_thumb_url") or "",
        })
    return jsonify({"messages": result, "removed_ids": removed_ids})


@app.route("/student/class/<int:class_id>/comment", methods=["POST"])
def student_post_comment(class_id):
    protect = student_required()
    if protect:
        return ajax_err("Not logged in.")

    student_db_id = get_logged_student_db_id()
    school_id = get_current_school_id()

    if not student_belongs_to_class(student_db_id, class_id):
        return ajax_err("You are not enrolled in this class.")

    # Support both JSON (text only) and multipart (file + optional text)
    file_url = None
    file_name = None
    file_thumb_url = None
    if request.content_type and 'multipart' in request.content_type:
        comment = (request.form.get("comment") or "").strip()
        f = request.files.get("file")
        if f and f.filename:
            filename = secure_filename(f.filename)
            file_bytes = f.read()
            if sniff_is_dangerous(file_bytes):
                return ajax_err("That file type is not allowed.")
            content_type = f.content_type or "application/octet-stream"
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            storage_name = f"class_{class_id}/{student_db_id}_{int(datetime.now().timestamp())}_{filename}"
            file_url = supabase_upload(storage_name, file_bytes, content_type)
            file_name = filename
            file_thumb_url = maybe_generate_and_upload_thumb(storage_name, file_bytes, ext)
    else:
        data = request.get_json(silent=True) or {}
        comment = (data.get("comment") or "").strip()

    if not comment and not file_url:
        return ajax_err("Message cannot be empty.")
    if len(comment) > 500:
        return ajax_err("Message too long (max 500 characters).")
    if not comment:
        comment = ""

    student_row = get_student_row_by_db_id(student_db_id)
    if not student_row:
        return ajax_err("Student account not found. Please log in again.")
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO class_comments (school_id, class_id, student_db_id, student_name, student_image, comment, poster_type, file_url, file_name, file_thumb_url)
            VALUES (%s, %s, %s, %s, %s, %s, 'student', %s, %s, %s)
        """, (school_id, class_id, student_db_id, student_row["full_name"], student_row["image_file"], comment, file_url, file_name, file_thumb_url))
        conn.commit()
        cur.execute("SELECT id, created_at FROM class_comments WHERE school_id=%s AND class_id=%s AND student_db_id=%s ORDER BY id DESC LIMIT 1", (school_id, class_id, student_db_id))
        new_row = cur.fetchone()
    except Exception as e:
        conn.rollback()
        conn.close()
        print("student_post_comment DB error:", repr(e), flush=True)
        return ajax_err("Failed to post message. Please try again.")
    conn.close()
    from datetime import date as _date
    new_id = new_row["id"] if new_row else 0
    ts = new_row["created_at"] if new_row else datetime.now()
    ts_display = ts.strftime("%I:%M %p") if hasattr(ts, "strftime") else ""
    return jsonify({"ok": True, "msg": {
        "id": new_id,
        "student_db_id": student_db_id,
        "student_name": student_row["full_name"],
        "student_image": supabase_public_url(student_row["image_file"] or ""),
        "comment": comment,
        "ts_display": ts_display,
        "poster_type": "student",
        "file_url": file_url or "",
        "file_name": file_name or "",
        "file_thumb_url": file_thumb_url or "",
        "is_priority": False,
    }})



# =========================================================
# STUDENT — MEMBERS POLL (for live priority badge updates)
# =========================================================
@app.route("/student/class/<int:class_id>/members")
def student_class_members_poll(class_id):
    protect = student_required()
    if protect:
        return jsonify([])
    student_db_id = get_logged_student_db_id()
    school_id = get_current_school_id()
    if not student_belongs_to_class(student_db_id, class_id):
        return jsonify([])
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT s.id, s.full_name, s.student_id, s.image_file, sc.is_priority
        FROM students s
        INNER JOIN student_classes sc ON sc.student_id_fk = s.id
        WHERE sc.class_id_fk=%s AND s.school_id=%s
        ORDER BY s.full_name ASC
    """, (class_id, school_id))
    rows = cur.fetchall()
    conn.close()
    return jsonify([{
        "id": r["id"],
        "full_name": r["full_name"],
        "student_id": r["student_id"],
        "image_file": supabase_public_url(r["image_file"] or ""),
        "is_priority": bool(r["is_priority"]),
    } for r in rows])


# =========================================================
# TEACHER — CLASS GROUP CHAT FEED
# =========================================================

def _teacher_belongs_to_class(teacher_id, class_id):
    """Check if the teacher owns this class."""
    row = get_class_by_id(class_id)
    return row is not None and row["teacher_id"] == teacher_id


@app.route("/teacher/class/<int:class_id>/feed")
def teacher_class_feed(class_id):
    protect = teacher_required()
    if protect:
        return protect

    teacher_id = get_logged_teacher_id()
    school_id = get_current_school_id()
    teacher_name = session.get("teacher_name", "Teacher")
    teacher_photo = session.get("teacher_photo", "")

    class_row = get_class_by_id(class_id)
    if not class_row or class_row["teacher_id"] != teacher_id:
        return "<script>alert('Access denied.');window.location.href='/teacher';</script>"

    classmates = get_students_in_class(class_id)

    # Fetch latest 80 messages oldest→newest
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, student_db_id, student_name, student_image, comment, created_at, poster_type, teacher_id_fk, file_url, file_name, file_thumb_url
        FROM class_comments
        WHERE class_id=%s AND school_id=%s
        ORDER BY created_at ASC LIMIT 80
    """, (class_id, school_id))
    messages = cur.fetchall()
    conn.close()

    cname = class_row["class_name"]
    csubject = class_row.get("subject_name") or ""
    cteacher = class_row.get("teacher_display_name") or class_row.get("teacher_name") or teacher_name
    letter = cname[0].upper()
    colors = ["#2196F3","#E91E63","#9C27B0","#FF9800","#4CAF50","#00BCD4","#F44336","#3F51B5"]
    color = colors[sum(ord(c) for c in cname) % len(colors)]

    def _msg_html_teacher(m):
        ts = m["created_at"]
        if hasattr(ts, "strftime"):
            from datetime import date as _date
            if ts.date() == __import__("datetime").date.today():
                ts_str = ts.strftime("%I:%M %p")
            else:
                ts_str = ts.strftime("%b %d, %I:%M %p")
        else:
            ts_str = str(ts)[:16]
        txt = m["comment"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        mid = m["id"]
        fhtml = render_file_html(m.get("file_url"), m.get("file_name"), m.get("file_thumb_url"))
        is_me = (m.get("poster_type") == "teacher" and m.get("teacher_id_fk") == teacher_id)
        pr = m.get("is_priority", False)
        pring = "border:2px solid #f59e0b;box-shadow:0 0 10px rgba(245,158,11,0.3);" if pr else ""
        pin_lbl = '<span style="font-size:10px;background:#92400e;color:#fbbf24;padding:1px 6px;border-radius:6px;margin-left:4px;">📌 Priority</span>' if pr else ''
        priority_btn = f'<button onclick="event.stopPropagation();setPriority({mid},{0 if pr else 1})" style="position:absolute;top:4px;right:4px;background:none;border:none;cursor:pointer;font-size:13px;opacity:0;transition:opacity 0.15s;" class="t-pin-btn" title="{"Remove priority" if pr else "Set High Priority"}">{"📌" if pr else "📍"}</button>'
        del_btn = f'<button onclick="event.stopPropagation();deleteMsg({mid})" style="position:absolute;top:4px;right:26px;background:none;border:none;cursor:pointer;font-size:11px;color:#f87171;opacity:0;transition:opacity 0.15s;" class="t-pin-btn" title="Delete">✕</button>'
        if is_me:
            return (f'<div class="tg-msg-row tg-mine" id="msg-{mid}">' +
                    f'<div class="tg-bubble tg-bubble-mine" style="position:relative;{pring}" ' +
                    'onmousedown="startLongPress(event,this)" onmouseup="cancelLongPress()" onmouseleave="cancelLongPress()" ' +
                    'ontouchstart="startLongPress(event,this)" ontouchend="cancelLongPress()" ontouchcancel="cancelLongPress()" ontouchmove="markPressMoved()">' +
                    f'<div class="tg-bubble-text">{txt}{pin_lbl}</div>{fhtml}' +
                    f'<div class="tg-bubble-ts">{ts_str} ✓✓</div>{priority_btn}{del_btn}</div></div>')
        else:
            name = m["student_name"]
            colors2 = ["#5b9bd9","#e8699a","#a876d8","#f4a623","#52c97f","#4db8d4","#e8645b","#7986cb"]
            nc = colors2[sum(ord(c) for c in name) % len(colors2)]
            av = _tg_avatar(m["student_image"], name, 34)
            sid = m["student_db_id"]
            return (f'<div class="tg-msg-row tg-theirs" id="msg-{mid}">' +
                    f'<div class="tg-av-wrap"><a href="/teacher/class/{class_id}/student-profile/{sid}" style="display:contents;">{av}</a></div>' +
                    f'<div><div class="tg-sender-name" style="color:{nc};">{name}</div>' +
                    f'<div class="tg-bubble tg-bubble-theirs" style="position:relative;{pring}" ' +
                    'onmousedown="startLongPress(event,this)" onmouseup="cancelLongPress()" onmouseleave="cancelLongPress()" ' +
                    'ontouchstart="startLongPress(event,this)" ontouchend="cancelLongPress()" ontouchcancel="cancelLongPress()" ontouchmove="markPressMoved()">' +
                    f'<div class="tg-bubble-text">{txt}{pin_lbl}</div>{fhtml}' +
                    f'<div class="tg-bubble-ts">{ts_str}</div>{priority_btn}{del_btn}</div></div></div>')

    msgs_html = "".join(_msg_html_teacher(m) for m in messages)
    last_id = messages[-1]["id"] if messages else 0

    # Members sidebar — with student priority star toggle
    members_html = ""
    for s in classmates:
        av = _tg_avatar(s["image_file"], s["full_name"], 36)
        is_p = s.get("is_priority", False)
        star = '<span style="color:#f59e0b;font-size:14px;margin-left:3px;" title="High Priority">⭐</span>' if is_p else ''
        star_btn = (
            f'<button onclick="toggleStudentPriority({s["id"]}, {0 if is_p else 1}, this)" ' +
            f'style="background:none;border:none;cursor:pointer;font-size:18px;padding:2px 6px;border-radius:6px;color:{"#f59e0b" if is_p else "#3a4a5a"};transition:color 0.2s;" ' +
            f'title="{"Remove priority" if is_p else "Set High Priority"}">{"⭐" if is_p else "☆"}</button>'
        )
        members_html += (
            f'<div style="display:flex;align-items:center;padding:4px 8px 4px 12px;">' +
            f'<a href="/teacher/class/{class_id}/student-profile/{s["id"]}" class="tg-member-item" style="flex:1;padding:6px 0;background:none;border-radius:0;">' +
            f'<div class="tg-member-av">{av}</div>' +
            f'<div class="tg-member-info">' +
            f'<div class="tg-member-name">{s["full_name"]}{star}</div>' +
            f'<div class="tg-member-id">ID: {s["student_id"]}</div>' +
            f'</div></a>{star_btn}</div>'
        )

    EMOJIS = ["😀","😂","🥰","😎","🤔","👍","👏","🙌","🔥","💯","❤️","🎉","📚","✅","🤝","😅","🙏","💪","😴","🎓"]
    emoji_btns = "".join(f'<button class="tg-emoji-btn" onclick="insertEmoji(\'{e}\')">{e}</button>' for e in EMOJIS)

    teacher_av_url = supabase_public_url(teacher_photo) if teacher_photo else ""
    teacher_letter = teacher_name[0].upper() if teacher_name else "T"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{cname} · Teacher Chat</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0e1621;height:100vh;display:flex;flex-direction:column;overflow:hidden;color:#e4e7eb;}}
.tg-topbar{{height:56px;background:#17212b;border-bottom:1px solid #0f1923;display:flex;align-items:center;padding:0 16px;gap:12px;flex-shrink:0;box-shadow:0 1px 8px rgba(0,0,0,0.3);}}
.tg-topbar-av{{width:40px;height:40px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:700;color:#fff;flex-shrink:0;background:{color};cursor:pointer;}}
.tg-topbar-info{{flex:1;min-width:0;cursor:pointer;}}
.tg-topbar-title{{font-weight:700;font-size:15px;color:#e4e7eb;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.tg-topbar-sub{{font-size:12px;color:#5a8ebd;margin-top:1px;}}
.tg-back{{color:#5b9bd9;font-size:13px;font-weight:600;text-decoration:none;padding:6px 10px;border-radius:8px;transition:background 0.15s;white-space:nowrap;}}
.tg-back:hover{{background:rgba(91,155,217,0.12);}}
.tg-members-toggle{{background:none;border:none;cursor:pointer;color:#5a8ebd;font-size:22px;padding:6px 8px;border-radius:8px;transition:background 0.15s;flex-shrink:0;}}
.tg-members-toggle:hover{{background:rgba(91,155,217,0.12);}}
.tg-teacher-badge{{background:#7c3aed;color:#e9d5ff;font-size:10px;font-weight:700;padding:2px 8px;border-radius:8px;margin-left:6px;}}
.tg-body{{flex:1;display:flex;overflow:hidden;}}
.tg-messages-wrap{{flex:1;display:flex;flex-direction:column;background:#0e1621;min-width:0;}}
.tg-chat-bg{{flex:1;overflow-y:auto;padding:16px 12px;display:flex;flex-direction:column;gap:2px;background-image:radial-gradient(ellipse at 20% 80%,rgba(124,58,237,0.04) 0%,transparent 60%),radial-gradient(ellipse at 80% 20%,rgba(139,92,246,0.04) 0%,transparent 60%);}}
.tg-msg-row{{display:flex;align-items:flex-end;gap:8px;margin-bottom:2px;}}
.tg-mine{{flex-direction:row-reverse;}}
.tg-theirs{{flex-direction:row;}}
.tg-av-wrap{{flex-shrink:0;width:34px;align-self:flex-end;}}
.tg-letter-av{{border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;color:#fff;font-size:13px;flex-shrink:0;}}
.tg-sender-name{{font-size:12px;font-weight:700;margin-bottom:3px;padding-left:2px;}}
.tg-bubble{{max-width:min(420px,72vw);padding:8px 12px 4px;border-radius:16px;position:relative;word-break:break-word;box-shadow:0 1px 4px rgba(0,0,0,0.25);}}
.tg-bubble-mine{{background:#4c1d95;border-radius:16px 4px 16px 16px;}}
.tg-bubble-theirs{{background:#182533;border:1px solid #1e3048;border-radius:4px 16px 16px 16px;}}
.tg-bubble-text{{font-size:14px;line-height:1.5;color:#e4e7eb;}}
.tg-bubble-ts{{font-size:10px;color:#7a9bbf;text-align:right;margin-top:4px;}}
.tg-bubble:hover .t-pin-btn,.tg-bubble.selected .t-pin-btn{{opacity:1;}}
.tg-bubble.selected{{outline:2px solid #f87171;outline-offset:2px;}}
.tg-input-bar{{background:#17212b;border-top:1px solid #0f1923;padding:10px 12px;display:flex;align-items:flex-end;gap:8px;flex-shrink:0;}}
.tg-input-wrap{{flex:1;background:#242f3d;border-radius:20px;display:flex;align-items:center;padding:6px 14px;gap:8px;}}
.tg-emoji-toggle{{background:none;border:none;font-size:20px;cursor:pointer;flex-shrink:0;opacity:0.6;transition:opacity 0.15s;}}
.tg-emoji-toggle:hover{{opacity:1;}}
.tg-input{{flex:1;background:none;border:none;outline:none;color:#e4e7eb;font-size:14px;resize:none;max-height:120px;min-height:22px;line-height:1.5;}}
.tg-input::placeholder{{color:#4a5a6a;}}
.tg-send{{width:42px;height:42px;border-radius:50%;background:#7c3aed;border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0;transition:background 0.15s;}}
.tg-send:hover{{background:#6d28d9;}}
.tg-send:disabled{{background:#2b3c4e;cursor:not-allowed;}}
.tg-emoji-panel{{position:absolute;bottom:72px;left:12px;right:12px;background:#17212b;border:1px solid #243447;border-radius:16px;padding:12px;display:none;z-index:10;}}
.tg-emoji-panel.open{{display:flex;flex-wrap:wrap;gap:4px;}}
.tg-emoji-btn{{background:none;border:none;font-size:22px;cursor:pointer;padding:4px;border-radius:8px;transition:background 0.12s;}}
.tg-emoji-btn:hover{{background:rgba(255,255,255,0.08);}}
.tg-members-panel{{width:260px;background:#17212b;border-left:1px solid #0d1117;display:flex;flex-direction:column;transition:transform 0.28s cubic-bezier(0.4,0,0.2,1);flex-shrink:0;overflow:hidden;}}
.tg-members-panel.open{{transform:translateX(0);}}
.tg-members-header{{padding:12px 14px;border-bottom:1px solid #0f1923;font-size:12px;font-weight:700;color:#5a8ebd;text-transform:uppercase;letter-spacing:0.06em;display:flex;align-items:center;justify-content:space-between;}}
.tg-members-list{{flex:1;overflow-y:auto;padding:6px;}}
.tg-member-item{{display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:10px;text-decoration:none;cursor:pointer;transition:background 0.12s;}}
.tg-member-item:hover{{background:#1e3048;}}
.tg-member-av{{flex-shrink:0;}}
.tg-member-info{{min-width:0;}}
.tg-member-name{{font-size:13px;font-weight:600;color:#c8d8e8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.tg-member-id{{font-size:11px;color:#4a6a8a;}}
.tg-me-badge{{background:#2b5278;color:#7bb8ef;font-size:10px;font-weight:700;padding:1px 6px;border-radius:6px;margin-left:4px;}}
::-webkit-scrollbar{{width:4px;}}::-webkit-scrollbar-track{{background:transparent;}}::-webkit-scrollbar-thumb{{background:#2b3c4e;border-radius:4px;}}
@media(max-width:640px){{.tg-members-panel{{position:fixed;right:0;top:0;bottom:0;z-index:200;transform:translateX(100%);}}}}
</style>
</head>
<body>
<!-- Topbar -->
<div class="tg-topbar">
    <a href="/teacher/class/{class_id}" class="tg-back">← Back</a>
    <div class="tg-topbar-av">{letter}</div>
    <div class="tg-topbar-info">
        <div class="tg-topbar-title">{cname} <span class="tg-teacher-badge">👨‍🏫 Teacher</span></div>
        <div class="tg-topbar-sub">{len(classmates)} students · {csubject}</div>
    </div>
    <button class="tg-members-toggle" onclick="toggleMembers()" title="Members">👥</button>
</div>

<div class="tg-body" style="position:relative;">
    <!-- Messages -->
    <div class="tg-messages-wrap">
        <div class="tg-chat-bg" id="chatBg">
            {msgs_html}
            <div id="messagesEnd"></div>
        </div>

        <!-- Emoji panel -->
        <div class="tg-emoji-panel" id="emojiPanel">{emoji_btns}</div>

        <!-- Input bar -->
        <div class="tg-input-bar">
            <div class="tg-input-wrap">
                <button class="tg-emoji-toggle" onclick="toggleEmoji()" title="Emoji">😊</button>
                <input type="file" id="teacherFileInput" style="display:none" onchange="previewTeacherFile(this)">
                <button class="tg-emoji-toggle" onclick="document.getElementById('teacherFileInput').click()" title="Attach file" style="font-size:18px;">📎</button>
                <div style="flex:1;display:flex;flex-direction:column;">
                    <div id="teacherFilePreview" style="display:none;background:#182533;border-radius:8px;padding:5px 10px;font-size:12px;color:#5b9bd9;align-items:center;gap:6px;margin-bottom:4px;"></div>
                    <textarea class="tg-input" id="msgInput" placeholder="Message {cname}…" rows="1"
                        oninput="autoResize(this)"
                        onkeydown="if(event.key==='Enter'&&!event.shiftKey){{event.preventDefault();sendMessage();}}"></textarea>
                </div>
            </div>
            <button class="tg-send" id="sendBtn" onclick="sendMessage()" title="Send">➤</button>
        </div>
    </div>

    <!-- Members panel -->
    <div class="tg-members-panel" id="membersPanel">
        <div class="tg-members-header">
            <span>Students ({len(classmates)})</span>
            <button onclick="toggleMembers()" style="background:none;border:none;color:#5a8ebd;cursor:pointer;font-size:16px;">✕</button>
        </div>
        <div class="tg-members-list">{members_html}</div>
    </div>
</div>

<script>
// ── GLOBAL CSRF PROTECTION FOR AJAX ──
(function() {{
    function getCookie(name) {{
        const m = document.cookie.match('(^|;)\\s*' + name + '\\s*=\\s*([^;]+)');
        return m ? decodeURIComponent(m.pop()) : '';
    }}
    const _origFetch = window.fetch;
    window.fetch = function(input, init) {{
        init = init || {{}};
        const method = (init.method || 'GET').toUpperCase();
        if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(method)) {{
            init.headers = Object.assign({{}}, init.headers || {{}});
            if (!init.headers['X-CSRF-Token'] && !init.headers['x-csrf-token']) {{
                init.headers['X-CSRF-Token'] = getCookie('csrf_token');
            }}
        }}
        return _origFetch(input, init);
    }};
}})();
const TEACHER_ID = {teacher_id};
let lastId = {last_id};
let pollTimer;
let membersOpen = false;

function toggleMembers() {{
    membersOpen = !membersOpen;
    const panel = document.getElementById('membersPanel');
    panel.classList.toggle('open', membersOpen);
}}

function toggleEmoji() {{
    document.getElementById('emojiPanel').classList.toggle('open');
}}

function insertEmoji(e) {{
    const inp = document.getElementById('msgInput');
    const pos = inp.selectionStart;
    inp.value = inp.value.slice(0, pos) + e + inp.value.slice(inp.selectionEnd);
    inp.selectionStart = inp.selectionEnd = pos + e.length;
    inp.focus();
    autoResize(inp);
}}

function autoResize(el) {{
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}}

function buildFileHtml(file_url, file_name, file_thumb_url) {{
    if (!file_url) return '';
    const ext = (file_name || '').split('.').pop().toLowerCase();
    const isImage = ['jpg','jpeg','png','gif','webp'].includes(ext);
    const isVideo = ['mp4','mov','avi','webm','mkv'].includes(ext);
    if (isImage) {{
        return `<div style="margin-top:5px;"><a href="${{file_url}}" target="_blank"><img src="${{file_url}}" style="max-width:220px;max-height:180px;border-radius:8px;display:block;object-fit:cover;"></a></div>`;
    }} else if (isVideo) {{
        const posterAttr = file_thumb_url ? ` poster="${{file_thumb_url}}"` : '';
        return `<div style="margin-top:5px;border-radius:8px;overflow:hidden;max-width:260px;background:#000;">
            <video preload="metadata" playsinline${{posterAttr}} style="display:block;max-width:260px;max-height:200px;border-radius:8px;width:100%;cursor:pointer;"
                onclick="if(this.requestFullscreen){{this.requestFullscreen();}}else if(this.webkitRequestFullscreen){{this.webkitRequestFullscreen();}}this.controls=true;this.play();">
                <source src="${{file_url}}" type="video/${{ext === 'mov' ? 'quicktime' : ext}}">
            </video>
        </div>`;
    }} else {{
        const icons = {{pdf:'📕',doc:'📝',docx:'📝',xls:'📊',xlsx:'📊',ppt:'📋',pptx:'📋',zip:'🗜️',rar:'🗜️',txt:'📄',mp3:'🎵',wav:'🎵'}};
        const icon = icons[ext] || '📄';
        return `<div style="margin-top:5px;"><a href="${{file_url}}" target="_blank" style="display:inline-flex;align-items:center;gap:8px;background:rgba(0,0,0,0.2);border:1px solid rgba(91,155,217,0.3);border-radius:10px;padding:8px 12px;text-decoration:none;max-width:220px;">
            <span style="font-size:22px;flex-shrink:0;">${{icon}}</span>
            <span style="color:#5b9bd9;font-size:12px;word-break:break-all;line-height:1.3;">${{file_name || 'Download file'}}</span>
        </a></div>`;
    }}
}}

function renderMsg(m) {{
    const ts = m.ts_display || '';
    const txt = (m.comment || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    const txtHtml = txt ? `<div class="tg-bubble-text">${{txt}}</div>` : '';
    const fHtml = buildFileHtml(m.file_url, m.file_name, m.file_thumb_url);
    const isMe = m.poster_type === 'teacher' && m.teacher_id_fk === TEACHER_ID;

    if (isMe) {{
        return `<div class="tg-msg-row tg-mine" id="msg-${{m.id}}">
            <div class="tg-bubble tg-bubble-mine" style="position:relative;" onmousedown="startLongPress(event,this)" onmouseup="cancelLongPress()" onmouseleave="cancelLongPress()" ontouchstart="startLongPress(event,this)" ontouchend="cancelLongPress()" ontouchcancel="cancelLongPress()" ontouchmove="markPressMoved()">
                ${{txtHtml}}${{fHtml}}
                <div class="tg-bubble-ts">${{ts}} ✓✓</div>
                <button onclick="event.stopPropagation();deleteMsg(${{m.id}})" style="position:absolute;top:4px;right:4px;background:none;border:none;cursor:pointer;font-size:11px;color:#f87171;opacity:0;transition:opacity 0.15s;" class="t-pin-btn" title="Delete">✕</button>
            </div></div>`;
    }} else {{
        const COLORS = ["#5b9bd9","#e8699a","#a876d8","#f4a623","#52c97f","#4db8d4","#e8645b","#7986cb"];
        let s = 0; for(let c of m.student_name) s += c.charCodeAt(0);
        const nc = COLORS[s % COLORS.length];
        const avStyle = m.student_image
            ? `<img src="${{m.student_image}}" style="width:34px;height:34px;border-radius:50%;object-fit:cover;">`
            : `<div class="tg-letter-av" style="width:34px;height:34px;background:${{nc}};">${{m.student_name[0].toUpperCase()}}</div>`;
        return `<div class="tg-msg-row tg-theirs" id="msg-${{m.id}}">
            <div class="tg-av-wrap"><a href="/teacher/class/{class_id}/student-profile/${{m.student_db_id}}" style="display:contents;">${{avStyle}}</a></div>
            <div>
                <div class="tg-sender-name" style="color:${{nc}};">${{m.student_name}}</div>
                <div class="tg-bubble tg-bubble-theirs" style="position:relative;" onmousedown="startLongPress(event,this)" onmouseup="cancelLongPress()" onmouseleave="cancelLongPress()" ontouchstart="startLongPress(event,this)" ontouchend="cancelLongPress()" ontouchcancel="cancelLongPress()" ontouchmove="markPressMoved()">
                    ${{txtHtml}}${{fHtml}}
                    <div class="tg-bubble-ts">${{ts}}</div>
                    <button onclick="event.stopPropagation();deleteMsg(${{m.id}})" style="position:absolute;top:4px;right:4px;background:none;border:none;cursor:pointer;font-size:11px;color:#f87171;opacity:0;transition:opacity 0.15s;" class="t-pin-btn" title="Delete">✕</button>
                </div>
            </div></div>`;
    }}
}}

let pendingTeacherFile = null;

function previewTeacherFile(input) {{
    const f = input.files[0];
    if (!f) return;
    pendingTeacherFile = f;
    const preview = document.getElementById('teacherFilePreview');
    preview.style.display = 'flex';
    preview.innerHTML = `📎 ${{f.name}} <button onclick="clearTeacherFile()" style="background:none;border:none;color:#f87171;cursor:pointer;margin-left:auto;">✕</button>`;
}}

function clearTeacherFile() {{
    pendingTeacherFile = null;
    document.getElementById('teacherFileInput').value = '';
    const preview = document.getElementById('teacherFilePreview');
    preview.style.display = 'none';
    preview.innerHTML = '';
}}

async function sendMessage() {{
    const input = document.getElementById('msgInput');
    const btn = document.getElementById('sendBtn');
    const text = input.value.trim();
    if (!text && !pendingTeacherFile) return;
    input.value = '';
    autoResize(input);
    btn.disabled = true;
    document.getElementById('emojiPanel').classList.remove('open');

    const fd = new FormData();
    if (text) fd.append('comment', text);
    if (pendingTeacherFile) fd.append('file', pendingTeacherFile);
    clearTeacherFile();

    try {{
        const res = await fetch('/teacher/class/{class_id}/comment', {{
            method: 'POST',
            body: fd
        }});
        const data = await res.json();
        if (!data.ok) {{ input.value = text; }}
        else {{
            if (data.msg && !document.getElementById('msg-' + data.msg.id)) {{
                const bg = document.getElementById('chatBg');
                const end = document.getElementById('messagesEnd');
                const div = document.createElement('div');
                div.innerHTML = renderMsg(data.msg);
                bg.insertBefore(div.firstElementChild, end);
                lastId = Math.max(lastId, data.msg.id);
                scrollToBottom();
            }}
        }}
    }} catch(e) {{ input.value = text; }}
    finally {{ btn.disabled = false; input.focus(); }}
}}

async function pollMessages() {{
    try {{
        const bg = document.getElementById('chatBg');
        const visibleIds = Array.from(bg.querySelectorAll('[id^="msg-"]'))
            .map(el => el.id.slice(4))
            .filter(id => /^\d+$/.test(id));
        const url = '/teacher/class/{class_id}/messages?after=' + lastId +
            (visibleIds.length ? '&ids=' + visibleIds.join(',') : '');
        const res = await fetch(url);
        const data = await res.json();

        if (data.removed_ids && data.removed_ids.length) {{
            data.removed_ids.forEach(id => {{
                document.getElementById('msg-' + id)?.remove();
            }});
        }}

        if (data.messages && data.messages.length) {{
            const end = document.getElementById('messagesEnd');
            data.messages.forEach(m => {{
                if (document.getElementById('msg-' + m.id)) return;
                const div = document.createElement('div');
                div.innerHTML = renderMsg(m);
                bg.insertBefore(div.firstElementChild, end);
                lastId = Math.max(lastId, m.id);
            }});
            scrollToBottom();
        }}
    }} catch(e) {{}}
}}

function scrollToBottom() {{
    const bg = document.getElementById('chatBg');
    bg.scrollTop = bg.scrollHeight;
}}

scrollToBottom();
pollMessages();
pollTimer = setInterval(pollMessages, 1000);

async function toggleStudentPriority(studentId, priority, btn) {{
    try {{
        const res = await fetch('/teacher/class/{class_id}/student-priority/' + studentId, {{
            method: 'POST',
            headers: {{'Content-Type':'application/json'}},
            body: JSON.stringify({{priority: !!priority}})
        }});
        const data = await res.json();
        if (data.ok) {{
            btn.textContent = data.priority ? '⭐' : '☆';
            btn.style.color = data.priority ? '#f59e0b' : '#3a4a5a';
            btn.title = data.priority ? 'Remove priority' : 'Set High Priority';
            btn.setAttribute('onclick', `toggleStudentPriority(${{studentId}}, ${{data.priority ? 0 : 1}}, this)`);
            // Update name badge
            const memberName = btn.closest('div').querySelector('.tg-member-name');
            if (memberName) {{
                const existing = memberName.querySelector('.priority-star');
                if (data.priority) {{
                    if (!existing) {{
                        const span = document.createElement('span');
                        span.className = 'priority-star';
                        span.style.cssText = 'color:#f59e0b;font-size:14px;margin-left:3px;';
                        span.title = 'High Priority';
                        span.textContent = '⭐';
                        memberName.appendChild(span);
                    }}
                }} else {{
                    existing?.remove();
                }}
            }}
        }} else {{
            alert(data.error || 'Failed.');
        }}
    }} catch(e) {{ alert('Error.'); }}
}}

async function setPriority(msgId, priority) {{
    try {{
        const res = await fetch('/teacher/class/{class_id}/set-priority/' + msgId, {{
            method: 'POST',
            headers: {{'Content-Type':'application/json'}},
            body: JSON.stringify({{priority: !!priority}})
        }});
        const data = await res.json();
        if (data.ok) {{
            // Reload to reflect new priority styling
            window.location.reload();
        }} else {{
            alert(data.error || 'Failed to set priority.');
        }}
    }} catch(e) {{ alert('Error setting priority.'); }}
}}

async function deleteMsg(msgId) {{
    if (!confirm('Delete this message?')) return;
    try {{
        const res = await fetch('/chat/delete-message/' + msgId, {{method:'POST'}});
        const data = await res.json();
        if (data.ok) {{
            document.getElementById('msg-' + msgId)?.remove();
        }} else {{
            alert(data.error || 'Could not delete message.');
        }}
    }} catch(e) {{ alert('Error.'); }}
}}

// ── Long-press a message bubble to select it and reveal its delete/priority buttons (mobile-friendly) ──
let _pressTimer = null;
let _pressMoved = false;
function startLongPress(evt, bubbleEl) {{
    _pressMoved = false;
    clearTimeout(_pressTimer);
    _pressTimer = setTimeout(() => {{
        if (_pressMoved) return;
        const wasSelected = bubbleEl.classList.contains('selected');
        document.querySelectorAll('.tg-bubble.selected').forEach(b => b.classList.remove('selected'));
        if (!wasSelected) bubbleEl.classList.add('selected');
        if (navigator.vibrate) navigator.vibrate(15);
    }}, 500);
}}
function cancelLongPress() {{
    clearTimeout(_pressTimer);
}}
function markPressMoved() {{
    _pressMoved = true;
    clearTimeout(_pressTimer);
}}

// ── Right-click context menu (desktop + long-press on mobile) ──
(function() {{
    if (!document.getElementById('_ctxMenu')) {{
        const m = document.createElement('div');
        m.id = '_ctxMenu';
        m.style.cssText = [
            'position:fixed','z-index:9999','background:#1e2d3d','border:1px solid #2b4a6a',
            'border-radius:10px','box-shadow:0 8px 28px rgba(0,0,0,0.55)',
            'padding:4px 0','min-width:170px','display:none','user-select:none',
            'backdrop-filter:blur(4px)'
        ].join(';');
        m.innerHTML = `
            <div class="_ctx-item" id="_ctxSelect" style="display:flex;align-items:center;gap:10px;padding:9px 16px;cursor:pointer;font-size:14px;color:#c8d8e8;transition:background 0.12s;">
                <span style="font-size:15px;">✅</span> Select
            </div>
            <div style="height:1px;background:#2b3c4e;margin:3px 0;"></div>
            <div class="_ctx-item" id="_ctxPin" style="display:flex;align-items:center;gap:10px;padding:9px 16px;cursor:pointer;font-size:14px;color:#fbbf24;transition:background 0.12s;">
                <span style="font-size:15px;">📌</span> <span id="_ctxPinLabel">Pin message</span>
            </div>
            <div style="height:1px;background:#2b3c4e;margin:3px 0;"></div>
            <div class="_ctx-item" id="_ctxDelete" style="display:flex;align-items:center;gap:10px;padding:9px 16px;cursor:pointer;font-size:14px;color:#f87171;transition:background 0.12s;">
                <span style="font-size:15px;">🗑️</span> Delete
            </div>`;
        m.querySelectorAll('._ctx-item').forEach(el => {{
            el.addEventListener('mouseenter', () => el.style.background = '#253647');
            el.addEventListener('mouseleave', () => el.style.background = '');
        }});
        document.body.appendChild(m);
    }}

    let _ctxTargetBubble = null;
    let _ctxTargetMsgId = null;
    const menu = document.getElementById('_ctxMenu');

    // Track currently pinned msg id (read from server banner rendered on load)
    let _pinnedMsgId = null;
    const _serverBanner = document.getElementById('priorityBanner');
    if (_serverBanner) {{
        const _bm = (_serverBanner.getAttribute('onclick') || '').match(/\d+/);
        if (_bm) _pinnedMsgId = parseInt(_bm[0], 10);
    }}

    function showCtxMenu(x, y, bubbleEl, msgId) {{
        _ctxTargetBubble = bubbleEl;
        _ctxTargetMsgId = msgId;
        // Show "Unpin" if already pinned, else "Pin message"
        const pinLabel = document.getElementById('_ctxPinLabel');
        if (pinLabel) pinLabel.textContent = (_pinnedMsgId === msgId) ? 'Unpin message' : 'Pin message';
        menu.style.display = 'block';
        const vw = window.innerWidth, vh = window.innerHeight;
        const mw = menu.offsetWidth || 170, mh = menu.offsetHeight || 120;
        menu.style.left = (x + mw > vw ? vw - mw - 8 : x) + 'px';
        menu.style.top  = (y + mh > vh ? vh - mh - 8 : y) + 'px';
        document.querySelectorAll('.tg-bubble.selected').forEach(b => b.classList.remove('selected'));
        if (bubbleEl) bubbleEl.classList.add('selected');
    }}

    function hideCtxMenu() {{
        menu.style.display = 'none';
        _ctxTargetBubble = null;
        _ctxTargetMsgId = null;
    }}

    document.getElementById('_ctxSelect').addEventListener('click', () => {{
        if (_ctxTargetBubble) {{
            const wasSelected = _ctxTargetBubble.classList.contains('selected');
            document.querySelectorAll('.tg-bubble.selected').forEach(b => b.classList.remove('selected'));
            if (!wasSelected) _ctxTargetBubble.classList.add('selected');
        }}
        hideCtxMenu();
    }});

    // Pin/Unpin — calls server setPriority endpoint (teacher-only backend)
    document.getElementById('_ctxPin').addEventListener('click', async () => {{
        const id = _ctxTargetMsgId;
        hideCtxMenu();
        if (!id || typeof setPriority !== 'function') return;
        const isAlreadyPinned = (_pinnedMsgId === id);
        _pinnedMsgId = isAlreadyPinned ? null : id;
        await setPriority(id, isAlreadyPinned ? 0 : 1);
    }});

    document.getElementById('_ctxDelete').addEventListener('click', async () => {{
        const id = _ctxTargetMsgId;
        hideCtxMenu();
        if (id && typeof deleteMsg === 'function') await deleteMsg(id);
    }});

    document.addEventListener('contextmenu', e => {{
        const bubble = e.target.closest('.tg-bubble');
        if (!bubble) {{ hideCtxMenu(); return; }}
        e.preventDefault();
        e.stopPropagation();
        const row = bubble.closest('[id^="msg-"]');
        const msgId = row ? parseInt(row.id.replace('msg-', ''), 10) : null;
        showCtxMenu(e.clientX, e.clientY, bubble, msgId);
    }});

    document.addEventListener('click', e => {{
        if (!menu.contains(e.target)) hideCtxMenu();
    }});
    document.addEventListener('scroll', hideCtxMenu, true);
    document.addEventListener('keydown', e => {{ if (e.key === 'Escape') hideCtxMenu(); }});
}})();

document.addEventListener('click', e => {{
    const panel = document.getElementById('membersPanel');
    const toggle = document.querySelector('.tg-members-toggle');
    if (membersOpen && !panel.contains(e.target) && e.target !== toggle) {{
        membersOpen = false;
        panel.classList.remove('open');
    }}
    document.querySelectorAll('.tg-bubble.selected').forEach(b => b.classList.remove('selected'));
}});
window.addEventListener('beforeunload', () => clearInterval(pollTimer));
</script>
</body></html>"""
    return html


@app.route("/teacher/class/<int:class_id>/messages")
def teacher_class_messages(class_id):
    """Teacher polling endpoint — returns new messages after a given id."""
    protect = teacher_required()
    if protect:
        return jsonify({"messages": []})

    teacher_id = get_logged_teacher_id()
    school_id = get_current_school_id()

    class_row = get_class_by_id(class_id)
    if not class_row or class_row["teacher_id"] != teacher_id:
        return jsonify({"messages": []})

    after = int(request.args.get("after", 0))

    # IDs the client currently has rendered on screen — used to detect
    # messages that were deleted (by anyone) since the client's last poll.
    visible_ids = []
    raw_ids = request.args.get("ids", "")
    if raw_ids:
        for part in raw_ids.split(","):
            part = part.strip()
            if part.isdigit():
                visible_ids.append(int(part))
        visible_ids = visible_ids[:500]  # safety cap

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, student_db_id, student_name, student_image, comment, created_at, poster_type, teacher_id_fk, file_url, file_name, file_thumb_url
        FROM class_comments
        WHERE class_id=%s AND school_id=%s AND id>%s
        ORDER BY created_at ASC LIMIT 30
    """, (class_id, school_id, after))
    rows = cur.fetchall()

    removed_ids = []
    if visible_ids:
        placeholders = ",".join(["%s"] * len(visible_ids))
        cur.execute(
            f"SELECT id FROM class_comments WHERE class_id=%s AND school_id=%s AND id IN ({placeholders})",
            [class_id, school_id] + visible_ids
        )
        still_exist = {r["id"] for r in cur.fetchall()}
        removed_ids = [i for i in visible_ids if i not in still_exist]

    conn.close()

    from datetime import date as _date
    result = []
    for r in rows:
        ts = r["created_at"]
        if hasattr(ts, "strftime"):
            ts_display = ts.strftime("%I:%M %p") if ts.date() == _date.today() else ts.strftime("%b %d, %I:%M %p")
        else:
            ts_display = str(ts)[:16]
        result.append({
            "id": r["id"],
            "student_db_id": r["student_db_id"],
            "student_name": r["student_name"],
            "student_image": supabase_public_url(r["student_image"] or ""),
            "comment": r["comment"],
            "ts_display": ts_display,
            "poster_type": r.get("poster_type") or "student",
            "teacher_id_fk": r.get("teacher_id_fk"),
            "file_url": r.get("file_url") or "",
            "file_name": r.get("file_name") or "",
            "file_thumb_url": r.get("file_thumb_url") or "",
        })
    return jsonify({"messages": result, "removed_ids": removed_ids})


# =========================================================
# TEACHER — TOGGLE STUDENT HIGH PRIORITY IN CLASS
# =========================================================
@app.route("/teacher/class/<int:class_id>/student-priority/<int:student_db_id>", methods=["POST"])
def teacher_toggle_student_priority(class_id, student_db_id):
    protect = teacher_required()
    if protect:
        return jsonify({"ok": False, "error": "Not logged in."})
    teacher_id = get_logged_teacher_id()
    class_row = get_class_by_id(class_id)
    if not class_row or class_row["teacher_id"] != teacher_id:
        return jsonify({"ok": False, "error": "Not authorized."})
    data = request.get_json(silent=True) or {}
    priority = bool(data.get("priority", True))
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        UPDATE student_classes SET is_priority=%s
        WHERE student_id_fk=%s AND class_id_fk=%s
    """, (priority, student_db_id, class_id))
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "priority": priority})


# =========================================================
# CHAT — DELETE OWN MESSAGE (student) / ANY MESSAGE (teacher)
# =========================================================
@app.route("/chat/delete-message/<int:msg_id>", methods=["POST"])
def delete_chat_message(msg_id):
    school_id = get_current_school_id()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, student_db_id, poster_type, class_id FROM class_comments WHERE id=%s AND school_id=%s", (msg_id, school_id))
    row = cur.fetchone()
    if not row:
        conn.close()
        return jsonify({"ok": False, "error": "Message not found."})

    # Student: can only delete their own messages
    if is_student_logged_in():
        student_db_id = get_logged_student_db_id()
        if row["poster_type"] != "student" or row["student_db_id"] != student_db_id:
            conn.close()
            return jsonify({"ok": False, "error": "You can only delete your own messages."})

    # Teacher: can delete any message in their class
    elif is_teacher_logged_in():
        teacher_id = get_logged_teacher_id()
        class_row = get_class_by_id(row["class_id"])
        if not class_row or class_row["teacher_id"] != teacher_id:
            conn.close()
            return jsonify({"ok": False, "error": "Not authorized."})
    else:
        conn.close()
        return jsonify({"ok": False, "error": "Not logged in."})

    cur.execute("DELETE FROM class_comments WHERE id=%s", (msg_id,))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


# =========================================================
# CHAT — TEACHER TOGGLE HIGH PRIORITY on a message
# =========================================================
@app.route("/teacher/class/<int:class_id>/set-priority/<int:msg_id>", methods=["POST"])
def teacher_set_priority(class_id, msg_id):
    protect = teacher_required()
    if protect:
        return jsonify({"ok": False, "error": "Not logged in."})
    teacher_id = get_logged_teacher_id()
    school_id = get_current_school_id()
    class_row = get_class_by_id(class_id)
    if not class_row or class_row["teacher_id"] != teacher_id:
        return jsonify({"ok": False, "error": "Not authorized."})
    data = request.get_json(silent=True) or {}
    priority = bool(data.get("priority", True))
    conn = get_db()
    cur = conn.cursor()
    # Clear existing priority in this class first (only one pinned at a time)
    cur.execute("UPDATE class_comments SET is_priority=FALSE WHERE class_id=%s AND school_id=%s", (class_id, school_id))
    if priority:
        cur.execute("UPDATE class_comments SET is_priority=TRUE WHERE id=%s AND class_id=%s", (msg_id, class_id))
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "priority": priority})




# =========================================================
# CHAT — PRIORITY MESSAGE POLL (get current pinned message)
# =========================================================
@app.route("/student/class/<int:class_id>/priority-message")
def student_class_priority_message(class_id):
    protect = student_required()
    if protect:
        return jsonify({"msg": None})
    student_db_id = get_logged_student_db_id()
    school_id = get_current_school_id()
    if not student_belongs_to_class(student_db_id, class_id):
        return jsonify({"msg": None})
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, student_name, comment FROM class_comments
        WHERE class_id=%s AND school_id=%s AND is_priority=TRUE
        ORDER BY id DESC LIMIT 1
    """, (class_id, school_id))
    row = cur.fetchone()
    conn.close()
    if row:
        return jsonify({"msg": {"id": row["id"], "name": row["student_name"], "text": row["comment"]}})
    return jsonify({"msg": None})


# =========================================================
# DIRECT MESSAGES — Student ↔ Student within class
# =========================================================
# ── FIX: static-path DM routes MUST come before the dynamic <int:classmate_db_id> route
# so Flask doesn't try to cast "delete" or "unread-counts" as integers and 404.

@app.route("/student/dm/delete/<int:msg_id>", methods=["POST"])
def student_dm_delete(msg_id):
    protect = student_required()
    if protect:
        return jsonify({"ok": False})
    student_db_id = get_logged_student_db_id()
    school_id = get_current_school_id()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT sender_db_id FROM direct_messages WHERE id=%s AND school_id=%s", (msg_id, school_id))
    row = cur.fetchone()
    if not row or row["sender_db_id"] != student_db_id:
        conn.close()
        return jsonify({"ok": False, "error": "Not authorized."})
    cur.execute("DELETE FROM direct_messages WHERE id=%s", (msg_id,))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/student/dm/unread-counts")
def student_dm_unread_counts():
    """Returns {sender_db_id: count} for unread DMs for the logged-in student."""
    protect = student_required()
    if protect:
        return jsonify({})
    student_db_id = get_logged_student_db_id()
    school_id = get_current_school_id()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT sender_db_id, COUNT(*) as cnt
        FROM direct_messages
        WHERE receiver_db_id=%s AND school_id=%s AND is_read=FALSE
        GROUP BY sender_db_id
    """, (student_db_id, school_id))
    rows = cur.fetchall()
    conn.close()
    return jsonify({str(r["sender_db_id"]): r["cnt"] for r in rows})


@app.route("/student/dm/<int:classmate_db_id>")
def student_dm_page(classmate_db_id):
    protect = student_required()
    if protect:
        return protect
    student_db_id = get_logged_student_db_id()
    school_id = get_current_school_id()
    if student_db_id == classmate_db_id:
        return redirect("/student")

    # Verify they share a class (scoped to this school)
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT sc1.class_id_fk AS class_id FROM student_classes sc1
        JOIN student_classes sc2 ON sc1.class_id_fk = sc2.class_id_fk
        JOIN classes c ON c.id = sc1.class_id_fk
        WHERE sc1.student_id_fk=%s AND sc2.student_id_fk=%s AND c.school_id=%s
        LIMIT 1
    """, (student_db_id, classmate_db_id, school_id))
    shared = cur.fetchone()
    if not shared:
        conn.close()
        return "<script>alert('You are not in the same class.');window.location.href='/student';</script>"
    class_id = shared["class_id"]

    # Mark messages as read
    cur.execute("""
        UPDATE direct_messages SET is_read=TRUE
        WHERE receiver_db_id=%s AND sender_db_id=%s AND school_id=%s
    """, (student_db_id, classmate_db_id, school_id))

    # Fetch conversation
    cur.execute("""
        SELECT id, sender_db_id, sender_name, sender_image, message, created_at, is_read, file_url, file_name, file_thumb_url
        FROM direct_messages
        WHERE school_id=%s
          AND ((sender_db_id=%s AND receiver_db_id=%s) OR (sender_db_id=%s AND receiver_db_id=%s))
        ORDER BY created_at ASC LIMIT 100
    """, (school_id, student_db_id, classmate_db_id, classmate_db_id, student_db_id))
    messages = cur.fetchall()
    conn.commit()
    conn.close()

    me = get_student_row_by_db_id(student_db_id)
    them = get_student_row_by_db_id(classmate_db_id)
    if not them:
        return "Student not found", 404

    them_av = _tg_avatar(them["image_file"], them["full_name"], 40)
    them_name = them["full_name"]

    def _dm_html(m):
        is_me = m["sender_db_id"] == student_db_id
        ts = m["created_at"]
        ts_str = ts.strftime("%I:%M %p") if hasattr(ts, "strftime") else str(ts)[:16]
        txt = m["message"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        mid = m["id"]
        furl = m.get("file_url") or ""
        fname = m.get("file_name") or ""
        fthumb = m.get("file_thumb_url") or ""
        fhtml = render_file_html(furl, fname, fthumb)
        txt_html = f'<div class="tg-bubble-text">{txt}</div>' if txt else ''
        if is_me:
            return f"""<div class="tg-msg-row tg-mine" id="dm-{mid}">
                <div class="tg-bubble tg-bubble-mine" style="position:relative;" onmousedown="startLongPress(event,this)" onmouseup="cancelLongPress()" onmouseleave="cancelLongPress()" ontouchstart="startLongPress(event,this)" ontouchend="cancelLongPress()" ontouchcancel="cancelLongPress()" ontouchmove="markPressMoved()">
                    {txt_html}{fhtml}
                    <div class="tg-bubble-ts">{ts_str} ✓✓</div>
                    <button onclick="event.stopPropagation();deleteDM({mid})" class="del-btn" title="Delete">✕</button>
                </div>
            </div>"""
        else:
            av = _tg_avatar(m["sender_image"], m["sender_name"], 34)
            return f"""<div class="tg-msg-row tg-theirs" id="dm-{mid}">
                <div class="tg-av-wrap"><a href="/student/classmate/{classmate_db_id}" style="display:contents;">{av}</a></div>
                <div>
                    <div class="tg-bubble tg-bubble-theirs">
                        {txt_html}{fhtml}
                        <div class="tg-bubble-ts">{ts_str}</div>
                    </div>
                </div>
            </div>"""

    msgs_html = "".join(_dm_html(m) for m in messages)
    last_id = int(messages[-1]["id"]) if messages else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<title>DM · {them_name}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
*{{box-sizing:border-box;}}
html{{height:100%;height:-webkit-fill-available;}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0e1621;height:100vh;height:100dvh;display:flex;flex-direction:column;overflow:hidden;color:#e4e7eb;}}
.tg-topbar{{height:56px;background:#17212b;border-bottom:1px solid #0f1923;display:flex;align-items:center;padding:0 16px;gap:12px;flex-shrink:0;box-shadow:0 1px 8px rgba(0,0,0,0.3);}}
.tg-back{{color:#5b9bd9;font-size:13px;font-weight:600;text-decoration:none;padding:6px 10px;border-radius:8px;}}
.tg-topbar-info{{flex:1;min-width:0;}}
.tg-topbar-title{{font-weight:700;font-size:15px;color:#e4e7eb;}}
.tg-topbar-sub{{font-size:12px;color:#5a8ebd;}}
.tg-chat-bg{{flex:1;overflow-y:auto;overflow-x:hidden;padding:16px 12px;display:flex;flex-direction:column;gap:2px;-webkit-overflow-scrolling:touch;}}
.tg-msg-row{{display:flex;align-items:flex-end;gap:8px;margin-bottom:2px;}}
.tg-mine{{flex-direction:row-reverse;}}
.tg-theirs{{flex-direction:row;}}
.tg-av-wrap{{flex-shrink:0;width:34px;align-self:flex-end;}}
.tg-letter-av{{border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;color:#fff;font-size:13px;flex-shrink:0;}}
.tg-bubble{{max-width:min(380px,72vw);padding:8px 12px 4px;border-radius:4px 16px 16px 16px;word-break:break-word;box-shadow:0 1px 4px rgba(0,0,0,0.25);}}
.tg-bubble-mine{{background:#2b5278;border-radius:16px 4px 16px 16px;}}
.tg-bubble-theirs{{background:#182533;border:1px solid #1f3344;}}
.tg-bubble-text{{font-size:14px;line-height:1.45;color:#e4e7eb;}}
.tg-bubble-ts{{font-size:10px;color:#5a8ebd;margin-top:3px;text-align:right;}}
.tg-input-bar{{background:#17212b;border-top:1px solid #0f1923;padding:10px 12px;padding-bottom:max(10px,env(safe-area-inset-bottom));display:flex;align-items:center;gap:8px;flex-shrink:0;}}
.tg-input{{flex:1;background:#0e1621;border:1px solid #243447;border-radius:22px;padding:10px 16px;color:#e4e7eb;font-size:16px;outline:none;resize:none;max-height:100px;overflow-y:auto;}}
.tg-send-btn{{width:42px;height:42px;border-radius:50%;background:#2b5278;border:none;cursor:pointer;color:#5b9bd9;font-size:20px;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:background 0.15s;}}
.tg-send-btn:hover{{background:#3a6a96;}}
.tg-attach-btn{{width:36px;height:36px;border-radius:50%;background:none;border:none;cursor:pointer;font-size:20px;display:flex;align-items:center;justify-content:center;flex-shrink:0;opacity:0.7;transition:opacity 0.15s;}}
.tg-attach-btn:hover{{opacity:1;}}
.del-btn{{position:absolute;top:-10px;right:-10px;background:#f87171;border:2px solid #0e1621;color:#fff;font-size:11px;cursor:pointer;opacity:0;transform:scale(0.7);transition:opacity 0.15s,transform 0.15s;padding:0;width:22px;height:22px;border-radius:50%;line-height:1;display:flex;align-items:center;justify-content:center;pointer-events:none;}}
.tg-bubble:hover .del-btn,.tg-bubble.selected .del-btn{{opacity:1;transform:scale(1);pointer-events:auto;}}
.tg-bubble.selected{{outline:2px solid #f87171;outline-offset:2px;}}
</style>
</head>
<body>
<div class="tg-topbar">
    <a href="/student/classmate/{classmate_db_id}" class="tg-back">←</a>
    <a href="/student/classmate/{classmate_db_id}" style="display:contents;text-decoration:none;color:inherit;">
        {them_av}
        <div class="tg-topbar-info">
            <div class="tg-topbar-title">{them_name}</div>
            <div class="tg-topbar-sub">Direct Message</div>
        </div>
    </a>
</div>

<div class="tg-chat-bg" id="chatBox">
    {msgs_html}
</div>

<div class="tg-input-bar">
    <input type="file" id="fileInput" style="display:none" onchange="previewFile(this)">
    <button class="tg-attach-btn" onclick="document.getElementById('fileInput').click()" title="Attach file">📎</button>
    <div style="flex:1;display:flex;flex-direction:column;gap:4px;">
        <div id="filePreview" style="display:none;background:#182533;border-radius:8px;padding:6px 10px;font-size:12px;color:#5b9bd9;align-items:center;gap:6px;"></div>
        <textarea class="tg-input" id="msgInput" placeholder="Message {them_name}..." rows="1"
            onkeydown="if(event.key==='Enter'&&!event.shiftKey){{event.preventDefault();sendMsg();}}"></textarea>
    </div>
    <button class="tg-send-btn" onclick="sendMsg()">➤</button>
</div>

<script>
// ── GLOBAL CSRF PROTECTION FOR AJAX ──
(function() {{
    function getCookie(name) {{
        const m = document.cookie.match('(^|;)\\s*' + name + '\\s*=\\s*([^;]+)');
        return m ? decodeURIComponent(m.pop()) : '';
    }}
    const _origFetch = window.fetch;
    window.fetch = function(input, init) {{
        init = init || {{}};
        const method = (init.method || 'GET').toUpperCase();
        if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(method)) {{
            init.headers = Object.assign({{}}, init.headers || {{}});
            if (!init.headers['X-CSRF-Token'] && !init.headers['x-csrf-token']) {{
                init.headers['X-CSRF-Token'] = getCookie('csrf_token');
            }}
        }}
        return _origFetch(input, init);
    }};
}})();
let lastId = {last_id};
const chatBox = document.getElementById('chatBox');
chatBox.scrollTop = chatBox.scrollHeight;
let pendingFile = null;

function previewFile(input) {{
    const f = input.files[0];
    if (!f) return;
    pendingFile = f;
    const preview = document.getElementById('filePreview');
    preview.style.display = 'flex';
    preview.style.alignItems = 'center';
    preview.innerHTML = `📎 ${{f.name}} <button onclick="clearFile()" style="background:none;border:none;color:#f87171;cursor:pointer;margin-left:auto;">✕</button>`;
}}

function clearFile() {{
    pendingFile = null;
    document.getElementById('fileInput').value = '';
    const preview = document.getElementById('filePreview');
    preview.style.display = 'none';
    preview.innerHTML = '';
}}

async function sendMsg() {{
    const inp = document.getElementById('msgInput');
    const msg = inp.value.trim();
    if (!msg && !pendingFile) return;
    inp.value = '';
    inp.style.height = 'auto';

    const fd = new FormData();
    if (msg) fd.append('message', msg);
    if (pendingFile) fd.append('file', pendingFile);
    clearFile();

    try {{
        const res = await fetch('/student/dm/{classmate_db_id}/send', {{
            method: 'POST',
            body: fd
        }});
        const data = await res.json();
        if (data.ok) {{
            // Instantly show the sent message without waiting for poll
            const now = new Date();
            const hh = now.getHours() % 12 || 12;
            const mm = String(now.getMinutes()).padStart(2, '0');
            const ampm = now.getHours() >= 12 ? 'PM' : 'AM';
            const ts = `${{hh}}:${{mm}} ${{ampm}}`;
            const txt = msg ? msg.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') : '';
            const txtHtml = txt ? `<div class="tg-bubble-text">${{txt}}</div>` : '';
            const div = document.createElement('div');
            div.innerHTML = `<div class="tg-msg-row tg-mine" id="dm-${{data.id}}">
                <div class="tg-bubble tg-bubble-mine" style="position:relative;" onmousedown="startLongPress(event,this)" onmouseup="cancelLongPress()" onmouseleave="cancelLongPress()" ontouchstart="startLongPress(event,this)" ontouchend="cancelLongPress()" ontouchcancel="cancelLongPress()" ontouchmove="markPressMoved()">
                    ${{txtHtml}}
                    <div class="tg-bubble-ts">${{ts}} ✓✓</div>
                    <button onclick="event.stopPropagation();deleteDM(${{data.id}})" class="del-btn" title="Delete">✕</button>
                </div>
            </div>`;
            chatBox.appendChild(div.firstElementChild);
            chatBox.scrollTop = chatBox.scrollHeight;
            lastId = Math.max(lastId, data.id);
        }}
    }} catch(e) {{}}
}}

async function deleteDM(id) {{
    if (!confirm('Delete this message?')) return;
    const res = await fetch('/student/dm/delete/' + id, {{method:'POST'}});
    const data = await res.json();
    if (data.ok) document.getElementById('dm-' + id)?.remove();
}}

// ── Long-press a message bubble to select it and reveal its delete button (mobile-friendly) ──
let _pressTimer = null;
let _pressMoved = false;
function startLongPress(evt, bubbleEl) {{
    _pressMoved = false;
    clearTimeout(_pressTimer);
    _pressTimer = setTimeout(() => {{
        if (_pressMoved) return;
        const wasSelected = bubbleEl.classList.contains('selected');
        document.querySelectorAll('.tg-bubble.selected').forEach(b => b.classList.remove('selected'));
        if (!wasSelected) bubbleEl.classList.add('selected');
        if (navigator.vibrate) navigator.vibrate(15);
    }}, 500);
}}
function cancelLongPress() {{
    clearTimeout(_pressTimer);
}}
function markPressMoved() {{
    _pressMoved = true;
    clearTimeout(_pressTimer);
}}
document.addEventListener('click', () => {{
    document.querySelectorAll('.tg-bubble.selected').forEach(b => b.classList.remove('selected'));
}});
// ── Right-click context menu (desktop) ──────────────────────────────────
(function() {{
    // Inject the menu element once
    if (!document.getElementById('_ctxMenu')) {{
        const m = document.createElement('div');
        m.id = '_ctxMenu';
        m.style.cssText = [
            'position:fixed','z-index:9999','background:#1e2d3d','border:1px solid #2b4a6a',
            'border-radius:10px','box-shadow:0 8px 28px rgba(0,0,0,0.55)',
            'padding:4px 0','min-width:160px','display:none','user-select:none',
            'backdrop-filter:blur(4px)'
        ].join(';');
        m.innerHTML = `
            <div class="_ctx-item" id="_ctxSelect" style="display:flex;align-items:center;gap:10px;padding:9px 16px;cursor:pointer;font-size:14px;color:#c8d8e8;transition:background 0.12s;">
                <span style="font-size:15px;">✅</span> Select
            </div>
            <div style="height:1px;background:#2b3c4e;margin:3px 0;"></div>
            <div class="_ctx-item" id="_ctxDelete" style="display:flex;align-items:center;gap:10px;padding:9px 16px;cursor:pointer;font-size:14px;color:#f87171;transition:background 0.12s;">
                <span style="font-size:15px;">🗑️</span> Delete
            </div>`;
        // Hover effect
        m.querySelectorAll('._ctx-item').forEach(el => {{
            el.addEventListener('mouseenter', () => el.style.background = '#253647');
            el.addEventListener('mouseleave', () => el.style.background = '');
        }});
        document.body.appendChild(m);
    }}

    let _ctxTargetBubble = null;
    let _ctxTargetMsgId = null;
    const menu = document.getElementById('_ctxMenu');

    function showCtxMenu(x, y, bubbleEl, msgId) {{
        _ctxTargetBubble = bubbleEl;
        _ctxTargetMsgId = msgId;

        // Position: keep inside viewport
        menu.style.display = 'block';
        const vw = window.innerWidth, vh = window.innerHeight;
        const mw = menu.offsetWidth || 160, mh = menu.offsetHeight || 90;
        menu.style.left = (x + mw > vw ? vw - mw - 8 : x) + 'px';
        menu.style.top  = (y + mh > vh ? vh - mh - 8 : y) + 'px';

        // Highlight the bubble
        document.querySelectorAll('.tg-bubble.selected').forEach(b => b.classList.remove('selected'));
        if (bubbleEl) bubbleEl.classList.add('selected');
    }}

    function hideCtxMenu() {{
        menu.style.display = 'none';
        _ctxTargetBubble = null;
        _ctxTargetMsgId = null;
    }}

    // Select action
    document.getElementById('_ctxSelect').addEventListener('click', () => {{
        if (_ctxTargetBubble) {{
            const wasSelected = _ctxTargetBubble.classList.contains('selected');
            document.querySelectorAll('.tg-bubble.selected').forEach(b => b.classList.remove('selected'));
            if (!wasSelected) _ctxTargetBubble.classList.add('selected');
        }}
        hideCtxMenu();
    }});

    // Delete action
    document.getElementById('_ctxDelete').addEventListener('click', async () => {{
        const id = _ctxTargetMsgId;
        hideCtxMenu();
        if (id && typeof deleteMsg === 'function') await deleteMsg(id);
    }});

    // Attach contextmenu to all bubbles (current + future via delegation)
    document.addEventListener('contextmenu', e => {{
        const bubble = e.target.closest('.tg-bubble');
        if (!bubble) {{ hideCtxMenu(); return; }}
        e.preventDefault();
        e.stopPropagation();
        // Extract msg id from the parent row's id ("msg-123")
        const row = bubble.closest('[id^="msg-"]');
        const msgId = row ? parseInt(row.id.replace('msg-', ''), 10) : null;
        showCtxMenu(e.clientX, e.clientY, bubble, msgId);
    }});

    // Close on outside click or scroll
    document.addEventListener('click', e => {{
        if (!menu.contains(e.target)) hideCtxMenu();
    }});
    document.addEventListener('scroll', hideCtxMenu, true);
    document.addEventListener('keydown', e => {{ if (e.key === 'Escape') hideCtxMenu(); }});
}})();


async function pollMessages() {{
    try {{
        const visibleIds = Array.from(chatBox.querySelectorAll('[id^="dm-"]'))
            .map(el => el.id.slice(3))
            .filter(id => /^\d+$/.test(id));
        const url = '/student/dm/{classmate_db_id}/poll?since=' + lastId +
            (visibleIds.length ? '&ids=' + visibleIds.join(',') : '');
        const res = await fetch(url);
        const data = await res.json();

        if (data.removed_ids && data.removed_ids.length) {{
            data.removed_ids.forEach(id => {{
                document.getElementById('dm-' + id)?.remove();
            }});
        }}

        const msgs = data.messages || [];
        for (const m of msgs) {{
            if (document.getElementById('dm-' + m.id)) continue;
            lastId = Math.max(lastId, m.id);
            const div = document.createElement('div');
            div.innerHTML = m.html;
            chatBox.appendChild(div.firstElementChild);
            chatBox.scrollTop = chatBox.scrollHeight;
        }}
    }} catch(e) {{}}
}}

setInterval(pollMessages, 1000);
</script>
</body>
</html>"""
    return html


@app.route("/student/dm/<int:classmate_db_id>/send", methods=["POST"])
def student_dm_send(classmate_db_id):
    protect = student_required()
    if protect:
        return jsonify({"ok": False})
    student_db_id = get_logged_student_db_id()
    school_id = get_current_school_id()
    if student_db_id == classmate_db_id:
        return jsonify({"ok": False, "error": "Cannot message yourself."})

    # Verify shared class (scoped to this school)
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT sc1.class_id_fk AS class_id FROM student_classes sc1
        JOIN student_classes sc2 ON sc1.class_id_fk = sc2.class_id_fk
        JOIN classes c ON c.id = sc1.class_id_fk
        WHERE sc1.student_id_fk=%s AND sc2.student_id_fk=%s AND c.school_id=%s LIMIT 1
    """, (student_db_id, classmate_db_id, school_id))
    shared = cur.fetchone()
    if not shared:
        conn.close()
        return jsonify({"ok": False, "error": "Not in same class."})
    class_id = shared["class_id"]

    # Support multipart (file) or JSON (text)
    file_url = None
    file_name = None
    file_thumb_url = None
    if request.content_type and 'multipart' in request.content_type:
        message = (request.form.get("message") or "").strip()
        f = request.files.get("file")
        if f and f.filename:
            fname = secure_filename(f.filename)
            file_bytes = f.read()
            if sniff_is_dangerous(file_bytes):
                return ajax_err("That file type is not allowed.")
            content_type = f.content_type or "application/octet-stream"
            ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
            storage_name = f"dm_{student_db_id}_{classmate_db_id}_{int(datetime.now().timestamp())}_{fname}"
            file_url = supabase_upload(storage_name, file_bytes, content_type)
            file_name = fname
            file_thumb_url = maybe_generate_and_upload_thumb(storage_name, file_bytes, ext)
    else:
        data = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()

    if not message and not file_url:
        conn.close()
        return jsonify({"ok": False, "error": "Message cannot be empty."})
    if len(message) > 1000:
        conn.close()
        return jsonify({"ok": False, "error": "Message too long."})
    if not message:
        message = ""

    me = get_student_row_by_db_id(student_db_id)
    cur.execute("""
        INSERT INTO direct_messages (school_id, class_id, sender_db_id, receiver_db_id, sender_name, sender_image, message, file_url, file_name, file_thumb_url)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
    """, (school_id, class_id, student_db_id, classmate_db_id, me["full_name"], me["image_file"], message, file_url, file_name, file_thumb_url))
    new_id = cur.fetchone()["id"]
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "id": new_id})


@app.route("/student/dm/<int:classmate_db_id>/poll")
def student_dm_poll(classmate_db_id):
    protect = student_required()
    if protect:
        return jsonify({"messages": [], "removed_ids": []})
    student_db_id = get_logged_student_db_id()
    school_id = get_current_school_id()
    since = int(request.args.get("since", 0))

    # IDs the client currently has rendered on screen — used to detect
    # messages that were deleted (by either party) since the last poll.
    visible_ids = []
    raw_ids = request.args.get("ids", "")
    if raw_ids:
        for part in raw_ids.split(","):
            part = part.strip()
            if part.isdigit():
                visible_ids.append(int(part))
        visible_ids = visible_ids[:500]  # safety cap

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, sender_db_id, sender_name, sender_image, message, created_at, file_url, file_name, file_thumb_url
        FROM direct_messages
        WHERE school_id=%s
          AND ((sender_db_id=%s AND receiver_db_id=%s) OR (sender_db_id=%s AND receiver_db_id=%s))
          AND id > %s
        ORDER BY created_at ASC LIMIT 50
    """, (school_id, student_db_id, classmate_db_id, classmate_db_id, student_db_id, since))
    rows = cur.fetchall()

    removed_ids = []
    if visible_ids:
        placeholders = ",".join(["%s"] * len(visible_ids))
        cur.execute(
            f"""SELECT id FROM direct_messages
                WHERE school_id=%s
                  AND ((sender_db_id=%s AND receiver_db_id=%s) OR (sender_db_id=%s AND receiver_db_id=%s))
                  AND id IN ({placeholders})""",
            [school_id, student_db_id, classmate_db_id, classmate_db_id, student_db_id] + visible_ids
        )
        still_exist = {r["id"] for r in cur.fetchall()}
        removed_ids = [i for i in visible_ids if i not in still_exist]

    # Mark as read
    cur.execute("""
        UPDATE direct_messages SET is_read=TRUE
        WHERE receiver_db_id=%s AND sender_db_id=%s AND school_id=%s AND id > %s
    """, (student_db_id, classmate_db_id, school_id, since))
    conn.commit()
    conn.close()

    result = []
    for m in rows:
        is_me = m["sender_db_id"] == student_db_id
        ts = m["created_at"]
        ts_str = ts.strftime("%I:%M %p") if hasattr(ts, "strftime") else str(ts)[:16]
        txt = m["message"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        mid = m["id"]
        fhtml = render_file_html(m.get("file_url"), m.get("file_name"), m.get("file_thumb_url"))
        if is_me:
            html = f"""<div class="tg-msg-row tg-mine" id="dm-{mid}">
                <div class="tg-bubble tg-bubble-mine" style="position:relative;" onmousedown="startLongPress(event,this)" onmouseup="cancelLongPress()" onmouseleave="cancelLongPress()" ontouchstart="startLongPress(event,this)" ontouchend="cancelLongPress()" ontouchcancel="cancelLongPress()" ontouchmove="markPressMoved()">
                    {'<div class="tg-bubble-text">' + txt + '</div>' if txt else ''}
                    {fhtml}
                    <div class="tg-bubble-ts">{ts_str} ✓✓</div>
                    <button onclick="event.stopPropagation();deleteDM({mid})" class="del-btn" title="Delete">✕</button>
                </div>
            </div>"""
        else:
            av = _tg_avatar(m["sender_image"], m["sender_name"], 34)
            html = f"""<div class="tg-msg-row tg-theirs" id="dm-{mid}">
                <div class="tg-av-wrap"><a href="/student/classmate/{classmate_db_id}" style="display:contents;">{av}</a></div>
                <div>
                    <div class="tg-bubble tg-bubble-theirs">
                        {'<div class="tg-bubble-text">' + txt + '</div>' if txt else ''}
                        {fhtml}
                        <div class="tg-bubble-ts">{ts_str}</div>
                    </div>
                </div>
            </div>"""
        result.append({"id": mid, "html": html})
    return jsonify({"messages": result, "removed_ids": removed_ids})


@app.route("/teacher/class/<int:class_id>/comment", methods=["POST"])
def teacher_post_comment(class_id):
    protect = teacher_required()
    if protect:
        return ajax_err("Not logged in.")

    teacher_id = get_logged_teacher_id()
    school_id = get_current_school_id()

    class_row = get_class_by_id(class_id)
    if not class_row or class_row["teacher_id"] != teacher_id:
        return ajax_err("Not authorized for this class.")

    file_url = None
    file_name = None
    file_thumb_url = None
    if request.content_type and 'multipart' in request.content_type:
        comment = (request.form.get("comment") or "").strip()
        f = request.files.get("file")
        if f and f.filename:
            filename = secure_filename(f.filename)
            file_bytes = f.read()
            if sniff_is_dangerous(file_bytes):
                return ajax_err("That file type is not allowed.")
            content_type = f.content_type or "application/octet-stream"
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            storage_name = f"class_{class_id}/teacher_{teacher_id}_{int(datetime.now().timestamp())}_{filename}"
            file_url = supabase_upload(storage_name, file_bytes, content_type)
            file_name = filename
            file_thumb_url = maybe_generate_and_upload_thumb(storage_name, file_bytes, ext)
    else:
        data = request.get_json(silent=True) or {}
        comment = (data.get("comment") or "").strip()

    if not comment and not file_url:
        return ajax_err("Message cannot be empty.")
    if len(comment) > 500:
        return ajax_err("Comment too long (max 500 characters).")
    if not comment:
        comment = ""

    teacher_name = session.get("teacher_name", "Teacher")
    teacher_photo = session.get("teacher_photo", "")

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO class_comments (school_id, class_id, student_db_id, student_name, student_image, comment, poster_type, teacher_id_fk, file_url, file_name, file_thumb_url)
            VALUES (%s, %s, %s, %s, %s, %s, 'teacher', %s, %s, %s, %s)
        """, (school_id, class_id, 0, teacher_name, teacher_photo, comment, teacher_id, file_url, file_name, file_thumb_url))
        conn.commit()
        cur.execute("SELECT id, created_at FROM class_comments WHERE school_id=%s AND class_id=%s AND teacher_id_fk=%s ORDER BY id DESC LIMIT 1", (school_id, class_id, teacher_id))
        new_row = cur.fetchone()
    except Exception as e:
        conn.rollback()
        conn.close()
        print("teacher_post_comment DB error:", repr(e), flush=True)
        return ajax_err("Failed to post message. Please try again.")
    conn.close()
    new_id = new_row["id"] if new_row else 0
    ts = new_row["created_at"] if new_row else datetime.now()
    ts_display = ts.strftime("%I:%M %p") if hasattr(ts, "strftime") else ""
    return jsonify({"ok": True, "msg": {
        "id": new_id,
        "student_db_id": 0,
        "student_name": teacher_name,
        "student_image": supabase_public_url(teacher_photo or ""),
        "comment": comment,
        "ts_display": ts_display,
        "poster_type": "teacher",
        "file_url": file_url or "",
        "file_name": file_name or "",
        "file_thumb_url": file_thumb_url or "",
        "is_priority": False,
    }})


@app.route("/teacher/class/<int:class_id>/student-profile/<int:student_db_id>")
def teacher_view_student_profile(class_id, student_db_id):
    """Teacher can tap on any student in their class to see their profile."""
    protect = teacher_required()
    if protect:
        return protect

    teacher_id = get_logged_teacher_id()
    school_id = get_current_school_id()

    class_row = get_class_by_id(class_id)
    if not class_row or class_row["teacher_id"] != teacher_id:
        return "<script>alert('Access denied.');window.location.href='/teacher';</script>"

    student = get_student_row_by_db_id(student_db_id)
    if not student or student.get("school_id", school_id) != school_id:
        return "Student not found", 404

    # Attendance stats for this student in this class
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT status, date, time FROM attendance WHERE student_id=%s AND class_id=%s ORDER BY date DESC LIMIT 20",
                (student["student_id"], class_id))
    recent_att = cur.fetchall()
    conn.close()

    present_count, total_count = get_attendance_count_for_student_class(student["student_id"], class_id)
    pct = round((present_count / total_count * 100), 1) if total_count else 0

    av_html = _tg_avatar(student["image_file"], student["full_name"], 90)

    att_rows = "".join(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 14px;border-bottom:1px solid #0f1923;">
        <span style="font-size:13px;color:#c8d8e8;">{a['date']}</span>
        <span style="font-size:11px;color:#7a9bbf;">{format_time_12hr(str(a['time']))}</span>
        <span style="font-size:11px;font-weight:700;padding:2px 10px;border-radius:8px;{'background:#1a3a2a;color:#52c97f;' if a['status']=='Present' else 'background:#3a1a1a;color:#f87171;'}">{a['status']}</span>
    </div>""" for a in recent_att) or '<p style="text-align:center;color:#3a4a5a;padding:16px;font-size:13px;">No records yet</p>'

    color_bar = "#52c97f" if pct >= 75 else ("#f4a623" if pct >= 50 else "#f87171")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{student['full_name']}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0e1621;color:#e4e7eb;min-height:100vh;}}
.tg-letter-av{{border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;color:#fff;font-size:32px;flex-shrink:0;}}
</style>
</head>
<body>
<div style="max-width:480px;margin:0 auto;padding:20px 16px;">
    <a href="/teacher/class/{class_id}/feed" style="display:inline-flex;align-items:center;gap:6px;color:#5b9bd9;font-size:14px;font-weight:600;text-decoration:none;margin-bottom:20px;">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="15 18 9 12 15 6"/></svg>
        Back to Chat
    </a>

    <!-- Profile card -->
    <div style="background:#17212b;border-radius:20px;overflow:hidden;margin-bottom:16px;">
        <div style="height:80px;background:linear-gradient(135deg,#1a3a5c,#2b1a5c);"></div>
        <div style="display:flex;justify-content:center;margin-top:-45px;margin-bottom:12px;">
            <div style="border:4px solid #17212b;border-radius:50%;">{av_html}</div>
        </div>
        <div style="text-align:center;padding:0 20px 24px;">
            <h2 style="font-size:20px;font-weight:700;color:#e4e7eb;">{student['full_name']}</h2>
            <div style="font-size:13px;color:#5a8ebd;margin-top:8px;font-family:monospace;">ID: {student['student_id']}</div>
            <div style="font-size:12px;color:#3a4a5a;margin-top:4px;">Registered: {str(student.get('registered_at',''))[:10]}</div>
        </div>
    </div>

    <!-- Attendance summary -->
    <div style="background:#17212b;border-radius:16px;overflow:hidden;margin-bottom:16px;">
        <div style="padding:14px 16px;border-bottom:1px solid #0f1923;font-size:12px;font-weight:700;color:#5a8ebd;text-transform:uppercase;letter-spacing:0.06em;">
            Attendance in {class_row['class_name']}
        </div>
        <div style="padding:16px;display:flex;align-items:center;gap:16px;">
            <div style="flex:1;">
                <div style="font-size:28px;font-weight:800;color:{color_bar};">{pct}%</div>
                <div style="font-size:12px;color:#5a8ebd;margin-top:2px;">{present_count} present / {total_count} total</div>
                <div style="height:6px;background:#1e3048;border-radius:4px;margin-top:10px;overflow:hidden;">
                    <div style="height:100%;width:{pct}%;background:{color_bar};border-radius:4px;transition:width 0.5s;"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Recent attendance -->
    <div style="background:#17212b;border-radius:16px;overflow:hidden;">
        <div style="padding:14px 16px;border-bottom:1px solid #0f1923;font-size:12px;font-weight:700;color:#5a8ebd;text-transform:uppercase;letter-spacing:0.06em;">
            Recent Records
        </div>
        {att_rows}
    </div>
</div>
</body>
</html>"""
    return html


@app.route("/student/classmate/<int:classmate_db_id>")
def student_view_classmate(classmate_db_id):
    protect = student_required()
    if protect:
        return protect

    viewer_db_id = get_logged_student_db_id()
    school_id = get_current_school_id()
    student_ctx = get_student_row_by_db_id(viewer_db_id)

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT sc1.class_id_fk FROM student_classes sc1
        INNER JOIN student_classes sc2 ON sc1.class_id_fk = sc2.class_id_fk
        WHERE sc1.student_id_fk=%s AND sc2.student_id_fk=%s LIMIT 1
    """, (viewer_db_id, classmate_db_id))
    shared = cur.fetchone()
    conn.close()

    if not shared and viewer_db_id != classmate_db_id:
        return "<script>alert('You can only view profiles of students in your classes.');window.location.href='/student/classes';</script>"

    classmate = get_student_row_by_db_id(classmate_db_id)
    if not classmate or classmate.get("school_id", school_id) != school_id:
        return "Student not found", 404

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT c.* FROM classes c
        INNER JOIN student_classes sc1 ON sc1.class_id_fk = c.id AND sc1.student_id_fk=%s
        INNER JOIN student_classes sc2 ON sc2.class_id_fk = c.id AND sc2.student_id_fk=%s
        ORDER BY c.class_name
    """, (viewer_db_id, classmate_db_id))
    shared_classes = cur.fetchall()
    conn.close()

    is_me = viewer_db_id == classmate_db_id
    av_html = _tg_avatar(classmate["image_file"], classmate["full_name"], 90)

    shared_html = "".join(f"""
    <a href="/student/class/{sc['id']}/feed" style="display:flex;align-items:center;gap:10px;padding:10px 16px;
        background:#1a2635;border-radius:12px;text-decoration:none;transition:background 0.15s;"
        onmouseover="this.style.background='#1e3048'" onmouseout="this.style.background='#1a2635'">
        <span style="font-size:20px;">📚</span>
        <div>
            <div style="font-weight:600;color:#c8d8e8;font-size:14px;">{sc['class_name']}</div>
            <div style="font-size:12px;color:#4a6a8a;">{sc.get('subject_name') or ''}</div>
        </div>
        <svg style="margin-left:auto;color:#4a6a8a;" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"/></svg>
    </a>""" for sc in shared_classes)

    dm_btn = "" if is_me else (
        f'<div style="margin-top:16px;">'
        f'<a href="/student/dm/{classmate_db_id}" '
        f'style="display:inline-flex;align-items:center;gap:8px;background:#2b5278;color:#e4e7eb;'
        f'font-size:14px;font-weight:700;padding:10px 24px;border-radius:50px;text-decoration:none;">'
        f'&#128172; Send Message</a></div>'
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{classmate['full_name']}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0e1621;color:#e4e7eb;min-height:100vh;}}
.tg-letter-av{{border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;color:#fff;font-size:32px;flex-shrink:0;}}
.class-link{{display:flex;align-items:center;gap:10px;padding:10px 16px;background:#1a2635;border-radius:12px;text-decoration:none;transition:background 0.15s;}}
.class-link:hover{{background:#1e3048;}}
.dm-btn{{display:inline-flex;align-items:center;gap:8px;background:#2b5278;color:#e4e7eb;font-size:14px;font-weight:700;padding:10px 24px;border-radius:50px;text-decoration:none;transition:background 0.15s;margin-top:16px;}}
.dm-btn:hover{{background:#3a6a96;}}
</style>
</head>
<body>
<div style="max-width:480px;margin:0 auto;padding:20px 16px;">
    <!-- Back -->
    <a href="javascript:history.back()" style="display:inline-flex;align-items:center;gap:6px;color:#5b9bd9;font-size:14px;font-weight:600;text-decoration:none;margin-bottom:20px;">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="15 18 9 12 15 6"/></svg>
        Back
    </a>

    <!-- Profile card -->
    <div style="background:#17212b;border-radius:20px;overflow:hidden;margin-bottom:16px;">
        <div style="height:80px;background:linear-gradient(135deg,#1a3a5c,#2b1a5c);"></div>
        <div style="display:flex;justify-content:center;margin-top:-45px;margin-bottom:12px;">
            <div style="border:4px solid #17212b;border-radius:50%;">{av_html}</div>
        </div>
        <div style="text-align:center;padding:0 20px 24px;">
            <h2 style="font-size:20px;font-weight:700;color:#e4e7eb;">{classmate['full_name']}</h2>
            {('<span style="display:inline-block;background:#2b5278;color:#7bb8ef;font-size:11px;font-weight:700;padding:2px 10px;border-radius:8px;margin-top:4px;">You</span>' if is_me else '')}
            <div style="font-size:13px;color:#5a8ebd;margin-top:8px;font-family:monospace;">ID: {classmate['student_id']}</div>
            <div style="font-size:12px;color:#3a4a5a;margin-top:4px;">Member since {str(classmate.get('registered_at',''))[:10]}</div>
            <div style="display:inline-flex;align-items:center;gap:6px;background:#1a3a2a;color:#52c97f;font-size:12px;font-weight:700;padding:4px 14px;border-radius:20px;margin-top:12px;border:1px solid #1e4a2a;">
                <span style="width:7px;height:7px;background:#52c97f;border-radius:50%;display:inline-block;"></span>
                Active Student
            </div>
            {dm_btn}
        </div>
    </div>

    <!-- Shared classes -->
    <div style="background:#17212b;border-radius:16px;overflow:hidden;margin-bottom:16px;">
        <div style="padding:14px 16px;border-bottom:1px solid #0f1923;font-size:12px;font-weight:700;color:#5a8ebd;text-transform:uppercase;letter-spacing:0.06em;">
            Shared Classes ({len(shared_classes)})
        </div>
        <div style="padding:10px;display:flex;flex-direction:column;gap:6px;">
            {shared_html if shared_html else '<p style="text-align:center;color:#3a4a5a;padding:20px;font-size:13px;">No shared classes</p>'}
        </div>
    </div>
</div>
</body>
</html>"""
    return html


# =========================================================
# TEACHER — CLASS SESSION CODE MANAGEMENT
# =========================================================
@app.route("/teacher/start-session/<int:class_id>", methods=["POST"])
def teacher_start_session(class_id):
    protect = teacher_required()
    if protect:
        return protect
    school_id = get_current_school_id()
    # Teacher can choose duration in minutes (default 30, max 180)
    try:
        duration_minutes = max(1, min(180, int(request.form.get("duration_minutes", 30))))
    except:
        duration_minutes = 30
    # QR rotation interval (default 60 seconds, min 10, max 300)
    try:
        rotate_seconds = max(10, min(300, int(request.form.get("rotate_seconds", 60))))
    except:
        rotate_seconds = 60
    # Teacher GPS (sent from browser)
    try:
        teacher_lat = float(request.form.get("teacher_lat", ""))
    except:
        teacher_lat = None
    try:
        teacher_lng = float(request.form.get("teacher_lng", ""))
    except:
        teacher_lng = None
    # Capture teacher IP for same-WiFi enforcement
    teacher_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[0].strip()

    code = secrets.token_hex(2).upper()  # e.g. "A4F9"
    expires_at = datetime.now() + timedelta(minutes=duration_minutes)
    now = datetime.now()
    try:
        conn = get_db()
        cur = conn.cursor()
        # Expire any existing active sessions for this class
        cur.execute("UPDATE class_sessions SET active=FALSE WHERE class_id=%s AND school_id=%s", (class_id, school_id))
        cur.execute("""
            INSERT INTO class_sessions
                (school_id, class_id, code, expires_at, active,
                 teacher_lat, teacher_lng, teacher_ip, rotate_seconds, last_rotated)
            VALUES (%s, %s, %s, %s, TRUE, %s, %s, %s, %s, %s)
        """, (school_id, class_id, code, expires_at, teacher_lat, teacher_lng, teacher_ip, rotate_seconds, now))
        conn.commit()
        conn.close()
    except Exception as e:
        try:
            conn.rollback()
            conn.close()
        except Exception:
            pass
        err_msg = str(e).split("\n")[0].replace("'", "")
        print("teacher_start_session DB error:", repr(e), flush=True)
        if is_ajax():
            return jsonify({"ok": False, "message": f"Could not start session: {err_msg}"}), 500
        return f"<script>alert('Could not start session: {err_msg}');window.location.href='/teacher/class/{class_id}';</script>"

    if is_ajax():
        return jsonify({"ok": True, "code": code, "duration_minutes": duration_minutes,
                        "rotate_seconds": rotate_seconds,
                        "expires_at": utc_iso(expires_at), "message": "Session started!"})
    return f"<script>alert('Session code: {code}  (valid {duration_minutes} minutes)');window.location.href='/teacher/class/{class_id}';</script>"


@app.route("/teacher/rotate-session/<int:class_id>", methods=["POST"])
def teacher_rotate_session(class_id):
    """Auto-rotate the QR code — called by JS timer on the teacher session panel."""
    protect = teacher_required()
    if protect:
        return jsonify({"ok": False, "message": "Unauthorized"})
    school_id = get_current_school_id()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, expires_at FROM class_sessions
        WHERE class_id=%s AND school_id=%s AND active=TRUE AND expires_at > NOW()
        ORDER BY id DESC LIMIT 1
    """, (class_id, school_id))
    row = cur.fetchone()
    if not row:
        conn.close()
        return jsonify({"ok": False, "message": "No active session"})
    new_code = secrets.token_hex(2).upper()
    cur.execute("""
        UPDATE class_sessions SET code=%s, last_rotated=NOW()
        WHERE id=%s
    """, (new_code, row["id"]))
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "code": new_code})


@app.route("/teacher/student-locations/<int:class_id>")
def teacher_student_locations(class_id):
    """Return today's attendance with GPS data for the teacher's live location panel."""
    protect = teacher_required()
    if protect:
        return jsonify({"error": "Unauthorized"})
    school_id = get_current_school_id()
    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_db()
    cur = conn.cursor()
    # Get teacher GPS from active session
    cur.execute("""
        SELECT teacher_lat, teacher_lng FROM class_sessions
        WHERE class_id=%s AND school_id=%s AND active=TRUE AND expires_at > NOW()
        ORDER BY id DESC LIMIT 1
    """, (class_id, school_id))
    sess = cur.fetchone()
    teacher_lat = sess["teacher_lat"] if sess else None
    teacher_lng = sess["teacher_lng"] if sess else None

    # LEFT JOIN every student enrolled in this class against today's attendance
    # (rather than SELECT-ing only from the attendance table) so students who
    # never checked in still show up as "Absent" and can be manually marked
    # Present by the teacher — previously they were simply missing from this
    # list entirely, which made the panel look like overrides weren't working.
    cur.execute("""
        SELECT s.id AS student_db_id, s.full_name, s.student_id,
               a.status AS status, a.time AS time,
               a.student_lat AS student_lat, a.student_lng AS student_lng,
               a.distance_meters AS distance_meters
        FROM students s
        INNER JOIN student_classes sc ON sc.student_id_fk = s.id
        LEFT JOIN attendance a
               ON a.student_id = s.student_id
              AND a.class_id = %s
              AND a.date = %s
              AND a.school_id = %s
        WHERE sc.class_id_fk = %s AND s.school_id = %s
        ORDER BY (a.time IS NULL) ASC, a.time DESC, s.full_name ASC
    """, (class_id, today, school_id, class_id, school_id))
    rows = cur.fetchall()
    conn.close()
    students = []
    for r in rows:
        dist = r["distance_meters"]
        if dist is None and teacher_lat and teacher_lng and r["student_lat"] and r["student_lng"]:
            try:
                dist = haversine_distance(float(r["student_lat"]), float(r["student_lng"]), teacher_lat, teacher_lng)
            except:
                dist = None
        students.append({
            "student_db_id": r["student_db_id"],
            "full_name": r["full_name"],
            "student_id": r["student_id"],
            "status": r["status"] or "Absent",
            "time": r["time"] or "—",
            "lat": r["student_lat"],
            "lng": r["student_lng"],
            "distance_meters": round(dist, 1) if dist is not None else None,
        })
    return jsonify({
        "teacher_lat": teacher_lat,
        "teacher_lng": teacher_lng,
        "students": students
    })


@app.route("/teacher/live-mark/<int:class_id>/<int:student_db_id>/<status>", methods=["POST"])
def teacher_live_mark(class_id, student_db_id, status):
    """AJAX-friendly Present/Absent override used by the Live Student
    Check-ins panel — returns JSON and never redirects, so the panel can
    update in place without a page reload (unlike the older
    /teacher/manual-mark/... route, which is designed for full-page use)."""
    protect = teacher_required()
    if protect:
        return jsonify({"ok": False, "error": "Not logged in."}), 401

    if status not in ("Present", "Absent"):
        return jsonify({"ok": False, "error": "Invalid status."}), 400

    class_row = get_class_by_id(class_id)
    if not class_row:
        return jsonify({"ok": False, "error": "Class not found."}), 404
    if class_row["teacher_id"] != get_logged_teacher_id():
        return jsonify({"ok": False, "error": "Not authorized for this class."}), 403

    student_row = get_student_row_by_db_id(student_db_id)
    if not student_row:
        return jsonify({"ok": False, "error": "Student not found."}), 404

    # Confirm the student is actually enrolled in this class before allowing
    # the teacher to write an attendance record for them.
    enrolled = [s for s in get_students_in_class(class_id) if s["id"] == student_db_id]
    if not enrolled:
        return jsonify({"ok": False, "error": "Student is not enrolled in this class."}), 403

    result = mark_attendance(student_row, class_row, status, force=True)
    return jsonify({"ok": True, "status": result["status"], "student_db_id": student_db_id})


@app.route("/teacher/session-panel/<int:class_id>")
def teacher_session_panel(class_id):
    """Dedicated attendance timer & session code panel for teachers."""
    protect = teacher_required()
    if protect:
        return protect
    class_row = get_class_by_id(class_id)
    if not class_row or class_row["teacher_id"] != get_logged_teacher_id():
        return redirect("/teacher")
    school_id = get_current_school_id()

    # Get active session if any
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT code, expires_at, created_at, rotate_seconds, last_rotated FROM class_sessions
        WHERE class_id=%s AND school_id=%s AND active=TRUE AND expires_at > NOW()
        ORDER BY id DESC LIMIT 1
    """, (class_id, school_id))
    active = cur.fetchone()
    conn.close()

    active_code = active["code"] if active else None
    expires_iso = utc_iso(active["expires_at"]) if active else None
    rotate_seconds_init = active["rotate_seconds"] if active else 60
    last_rotated_iso = utc_iso(active["last_rotated"]) if active else None

    body = f"""
    <div class="max-w-2xl mx-auto space-y-6">
        <div class="flex items-center gap-3">
            <a href="/teacher/class/{class_id}" class="text-blue-600 hover:underline text-sm font-semibold">← Back to Class</a>
        </div>
        <h1 class="text-2xl font-extrabold text-slate-800">⏱️ Attendance Timer — {class_row['class_name']}</h1>
        <p class="text-sm text-slate-500">Start a timed attendance window. Students scan the QR code or enter the code manually to check in. Attendance closes automatically when the timer expires.</p>

        <!-- Active session display -->
        <div id="sessionPanel" class="bg-white border-2 rounded-2xl shadow-sm p-6 space-y-5 {'border-emerald-400' if active_code else 'border-slate-200'}">
            <div id="noSession" class="{'hidden' if active_code else ''} text-center py-6">
                <div class="text-5xl mb-3">🔒</div>
                <p class="text-slate-500 font-medium">No active attendance window.</p>
                <p class="text-xs text-slate-400 mt-1">Start a session below to allow students to check in.</p>
            </div>

            <div id="activeSession" class="{'hidden' if not active_code else ''} space-y-5">
                <div class="text-center">
                    <div class="text-xs font-bold text-emerald-600 uppercase tracking-widest mb-2">✅ Attendance Open</div>
                    <div class="text-6xl font-black font-mono tracking-widest text-slate-800 mb-1" id="codeDisplay">{active_code or '----'}</div>
                    <p class="text-xs text-slate-400">Students can scan the QR code below or type this code manually</p>
                </div>

                <!-- QR Code + Countdown side by side -->
                <div class="flex flex-col sm:flex-row gap-4 items-center justify-center">

                    <!-- QR Code box -->
                    <div class="flex flex-col items-center gap-2">
                        <div id="qrBox" class="bg-white border-2 border-slate-200 rounded-2xl p-3 shadow-sm flex items-center justify-center transition-all" style="width:200px;height:200px;">
                            <canvas id="qrCanvas"></canvas>
                        </div>
                        <div class="flex items-center gap-2 w-full max-w-[220px]">
                            <span class="text-xs text-slate-400">🔍</span>
                            <input type="range" id="qrSizeSlider" min="120" max="500" value="174" step="2"
                                oninput="resizeQR(this.value)"
                                class="flex-1 h-1.5 accent-indigo-600 cursor-pointer">
                            <span class="text-xs text-slate-400 font-mono w-10 text-right" id="qrSizeLabel">174px</span>
                        </div>
                        <button onclick="openFullscreen()" class="text-xs bg-slate-800 hover:bg-slate-900 text-white font-bold px-4 py-2 rounded-xl transition">
                            🖥️ Fullscreen for Class
                        </button>
                    </div>

                    <!-- Countdown -->
                    <div class="flex-1 bg-slate-50 rounded-2xl p-5 border text-center space-y-2">
                        <div class="text-xs text-slate-500 font-semibold uppercase tracking-widest">Time Remaining</div>
                        <div id="countdown" class="text-5xl font-black font-mono text-indigo-600">--:--</div>
                        <div class="h-2 bg-slate-200 rounded-full overflow-hidden mt-2">
                            <div id="progressBar" class="h-2 bg-indigo-500 rounded-full transition-all duration-1000" style="width:100%"></div>
                        </div>
                        <div class="text-xs text-slate-400 mt-1" id="expiryLabel"></div>
                    </div>
                </div>

                <div class="flex justify-center pt-1">
                    <form method="POST" action="/teacher/stop-session/{class_id}" data-ajax>
                            {csrf_input_html()}
                        <button type="submit" class="bg-red-500 hover:bg-red-600 text-white font-bold px-8 py-2.5 rounded-xl transition">
                            ⏹ Stop Attendance Now
                        </button>
                    </form>
                </div>
            </div>
        </div>

        <!-- Start new session form -->
        <div id="startForm" class="bg-white border rounded-2xl shadow-sm p-6 space-y-4 {'hidden' if active_code else ''}">
            <h2 class="font-bold text-slate-700">🕐 Set Attendance Window</h2>
            <p class="text-xs text-slate-400">Choose duration and QR rotation speed. Your GPS and WiFi are captured automatically to verify student proximity.</p>

            <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <button type="button" onclick="setDuration(5)" class="preset-btn border rounded-xl py-3 font-bold text-slate-700 hover:bg-indigo-50 hover:border-indigo-400 text-sm transition">5 min</button>
                <button type="button" onclick="setDuration(10)" class="preset-btn border rounded-xl py-3 font-bold text-slate-700 hover:bg-indigo-50 hover:border-indigo-400 text-sm transition">10 min</button>
                <button type="button" onclick="setDuration(15)" class="preset-btn border rounded-xl py-3 font-bold text-slate-700 hover:bg-indigo-50 hover:border-indigo-400 text-sm transition">15 min</button>
                <button type="button" onclick="setDuration(30)" class="preset-btn border rounded-xl py-3 font-bold text-slate-700 hover:bg-indigo-50 hover:border-indigo-400 text-sm transition bg-indigo-100 border-indigo-400 text-indigo-700">30 min</button>
            </div>

            <div class="flex flex-wrap items-center gap-4">
                <div class="flex items-center gap-2">
                    <span class="text-sm text-slate-500 font-medium">Duration:</span>
                    <input type="number" id="customDuration" min="1" max="180" value="30"
                        class="w-20 px-2 py-2 border rounded-lg text-center font-bold text-slate-800 focus:ring-2 focus:ring-indigo-400"
                        oninput="syncCustom()">
                    <span class="text-sm text-slate-500">min</span>
                </div>
                <div class="flex items-center gap-2">
                    <span class="text-sm text-slate-500 font-medium">🔄 QR rotates every:</span>
                    <input type="number" id="rotateSeconds" min="10" max="300" value="60"
                        class="w-20 px-2 py-2 border rounded-lg text-center font-bold text-slate-800 focus:ring-2 focus:ring-indigo-400">
                    <span class="text-sm text-slate-500">sec</span>
                </div>
            </div>

            <div id="gpsStatus" class="text-xs text-amber-600 font-semibold bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
                📡 Detecting your GPS location before start…
            </div>

            <button onclick="startSession(event)" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 rounded-xl text-base transition shadow-md shadow-indigo-100">
                ▶ Start Attendance Timer
            </button>
        </div>

        <!-- Live Student Locations Panel -->
        <div id="locationsPanel" class="bg-white border rounded-2xl shadow-sm p-5 space-y-3 {'hidden' if not active_code else ''}">
            <div class="flex items-center justify-between">
                <h2 class="font-bold text-slate-700">📍 Live Student Check-ins</h2>
                <span class="text-xs text-slate-400" id="lastRefreshed">Refreshing…</span>
            </div>
            <div id="teacherGpsInfo" class="text-xs text-slate-500 font-mono"></div>
            <div class="overflow-x-auto">
                <table class="w-full text-sm" id="locTable">
                    <thead>
                        <tr class="text-left text-xs text-slate-400 uppercase border-b">
                            <th class="pb-2 pr-3">Student</th>
                            <th class="pb-2 pr-3">Status</th>
                            <th class="pb-2 pr-3">Distance</th>
                            <th class="pb-2">Time</th>
                        </tr>
                    </thead>
                    <tbody id="locTableBody">
                        <tr><td colspan="4" class="py-4 text-center text-slate-400 text-xs">Waiting for check-ins…</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- FULLSCREEN OVERLAY -->
    <div id="fullscreenOverlay" class="bg-white z-50 hidden flex-col items-center text-center" style="position:fixed;top:0;left:0;right:0;bottom:0;height:100vh;height:100dvh;overflow-y:scroll;-webkit-overflow-scrolling:touch;">
        <button onclick="closeFullscreen()" class="fixed top-5 right-6 z-[60] bg-slate-800 hover:bg-slate-900 text-white text-sm font-bold px-4 py-2 rounded-full shadow-lg">✕ Close</button>
        <div class="w-full flex flex-col items-center px-8 py-8" style="min-height:100%;box-sizing:border-box;">
        <div class="text-slate-400 text-sm font-semibold uppercase tracking-widest mb-2 mt-10">{class_row['class_name']} — Scan to Mark Attendance</div>
        <div id="fsCode" class="text-8xl font-black font-mono tracking-[0.2em] text-slate-800 mb-4"></div>
        <canvas id="fsQrCanvas" class="rounded-2xl shadow-lg border-4 border-slate-100" style="width:300px;height:300px;"></canvas>
        <div class="flex items-center gap-2 w-64 mt-3">
            <span class="text-xs text-slate-400">🔍</span>
            <input type="range" id="fsQrSizeSlider" min="150" max="700" value="300" step="2"
                oninput="resizeFsQR(this.value)"
                class="flex-1 h-1.5 accent-indigo-600 cursor-pointer">
            <span class="text-xs text-slate-400 font-mono w-10 text-right" id="fsQrSizeLabel">300px</span>
        </div>
        <div class="mt-4 text-slate-400 text-sm">Scan with your phone camera · Or type the code above · QR rotates automatically</div>
        <div id="fsCountdown" class="mt-3 text-4xl font-black font-mono text-indigo-600"></div>
        <div class="mt-2 w-64 h-2 bg-slate-200 rounded-full overflow-hidden">
            <div id="fsProgressBar" class="h-2 bg-indigo-500 rounded-full transition-all duration-1000" style="width:100%"></div>
        </div>
        <div id="fsRotateBar" class="mt-2 w-64 h-1.5 bg-slate-100 rounded-full overflow-hidden">
            <div id="fsRotateProgress" class="h-1.5 bg-emerald-400 rounded-full transition-all duration-1000" style="width:100%"></div>
        </div>
        <div class="text-xs text-slate-400 mt-1 mb-16">🔄 QR refreshes automatically</div>
        </div>
    </div>

<!-- qrcode.js from CDN (note: uses the soldair/node-qrcode "qrcode" package,
     which exposes QRCode.toCanvas — NOT the davidshimjs "qrcodejs" package,
     which does not have a toCanvas method and was silently breaking QR
     rendering, which in turn was preventing the countdown timer from
     starting since renderQR() threw before startCountdown() ran. -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/qrcode/1.4.4/qrcode.min.js"></script>
<script>
const CLASS_ID = {class_id};
let countdownInterval = null;
let rotateInterval = null;
let locRefreshInterval = null;
let totalSeconds = 0;
let currentRotateSecs = 60;
let rotateSecsLeft = 60;
let currentCode = '{active_code or ""}';
let currentExpiresAt = {f'new Date("{expires_iso}")' if expires_iso else 'null'};
let initialRotateSecs = {rotate_seconds_init};
let initialLastRotated = {f'new Date("{last_rotated_iso}")' if last_rotated_iso else 'null'};
let teacherLat = null;
let teacherLng = null;

// ── GPS capture on page load — watch for a fix and keep the best one until accuracy is good ──
const gpsStatusEl = document.getElementById('gpsStatus');
let _teacherBestAcc = Infinity;
let _teacherGpsWatchId = null;
const _TEACHER_GPS_GOOD_ENOUGH_M = 50;
const _TEACHER_GPS_MAX_WAIT_MS = 20000;

if (navigator.geolocation) {{
    if (gpsStatusEl) {{ gpsStatusEl.innerText = '🔄 Acquiring GPS fix…'; }}
    _teacherGpsWatchId = navigator.geolocation.watchPosition(function(pos) {{
        const acc = pos.coords.accuracy;
        if (acc < _teacherBestAcc) {{
            _teacherBestAcc = acc;
            teacherLat = pos.coords.latitude;
            teacherLng = pos.coords.longitude;
            if (gpsStatusEl) {{
                gpsStatusEl.className = acc <= _TEACHER_GPS_GOOD_ENOUGH_M ? 'text-xs text-emerald-700 font-semibold bg-emerald-50 border border-emerald-200 rounded-lg px-3 py-2' : 'text-xs text-amber-600 font-semibold bg-amber-50 border border-amber-200 rounded-lg px-3 py-2';
                gpsStatusEl.innerText = '✅ GPS ready: ' + teacherLat.toFixed(5) + ', ' + teacherLng.toFixed(5) + ' (±' + Math.round(acc) + 'm)';
            }}
            if (acc <= _TEACHER_GPS_GOOD_ENOUGH_M) {{
                navigator.geolocation.clearWatch(_teacherGpsWatchId);
            }}
        }}
    }}, function(err) {{
        if (gpsStatusEl) {{
            gpsStatusEl.className = 'text-xs text-amber-600 font-semibold bg-amber-50 border border-amber-200 rounded-lg px-3 py-2';
            gpsStatusEl.innerText = '⚠️ GPS unavailable — WiFi/code check will still work. (' + err.message + ')';
        }}
    }}, {{ enableHighAccuracy: true, timeout: _TEACHER_GPS_MAX_WAIT_MS, maximumAge: 0 }});
    setTimeout(function() {{ if (_teacherGpsWatchId !== null) navigator.geolocation.clearWatch(_teacherGpsWatchId); }}, _TEACHER_GPS_MAX_WAIT_MS);
}} else {{
    if (gpsStatusEl) gpsStatusEl.innerText = '⚠️ GPS not supported on this device.';
}}

function checkinUrl(code) {{
    return window.location.origin + '/checkin/' + code;
}}

let qrCurrentSize = 174;
let fsQrCurrentSize = 300;

function renderQR(code) {{
    try {{
        const canvas = document.getElementById('qrCanvas');
        if (!canvas || typeof QRCode === 'undefined' || !QRCode.toCanvas) {{
            console.error('QR library not available — check the qrcode CDN script tag.');
            return;
        }}
        const size = qrCurrentSize;
        canvas.width = size; canvas.height = size;
        QRCode.toCanvas(canvas, checkinUrl(code), {{
            width: size, margin: 1,
            color: {{ dark: '#1e293b', light: '#ffffff' }}
        }}, function(err) {{ if(err) console.error(err); }});
    }} catch(e) {{ console.error('renderQR failed:', e); }}
}}

function renderFsQR(code) {{
    try {{
        const canvas = document.getElementById('fsQrCanvas');
        if (!canvas || typeof QRCode === 'undefined' || !QRCode.toCanvas) {{
            console.error('QR library not available — check the qrcode CDN script tag.');
            return;
        }}
        const size = fsQrCurrentSize;
        canvas.width = size; canvas.height = size;
        canvas.style.width = size + 'px';
        canvas.style.height = size + 'px';
        QRCode.toCanvas(canvas, checkinUrl(code), {{
            width: size, margin: 1,
            color: {{ dark: '#1e293b', light: '#ffffff' }}
        }}, function(err) {{ if(err) console.error(err); }});
    }} catch(e) {{ console.error('renderFsQR failed:', e); }}
}}

function resizeQR(size) {{
    size = parseInt(size, 10);
    qrCurrentSize = size;
    const box = document.getElementById('qrBox');
    if (box) {{
        // box padding (p-3 = 12px each side) so the QR sits comfortably inside
        box.style.width = (size + 24) + 'px';
        box.style.height = (size + 24) + 'px';
    }}
    const label = document.getElementById('qrSizeLabel');
    if (label) label.innerText = size + 'px';
    if (typeof currentCode !== 'undefined' && currentCode) {{
        renderQR(currentCode);
    }}
}}

function resizeFsQR(size) {{
    size = parseInt(size, 10);
    fsQrCurrentSize = size;
    const label = document.getElementById('fsQrSizeLabel');
    if (label) label.innerText = size + 'px';
    if (typeof currentCode !== 'undefined' && currentCode) {{
        renderFsQR(currentCode);
    }}
}}

function setDuration(min) {{
    document.getElementById('customDuration').value = min;
    document.querySelectorAll('.preset-btn').forEach(b => {{
        b.classList.remove('bg-indigo-100','border-indigo-400','text-indigo-700');
    }});
    event.target.classList.add('bg-indigo-100','border-indigo-400','text-indigo-700');
}}

function syncCustom() {{
    document.querySelectorAll('.preset-btn').forEach(b => {{
        b.classList.remove('bg-indigo-100','border-indigo-400','text-indigo-700');
    }});
}}

async function startSession(e) {{
    const duration = parseInt(document.getElementById('customDuration').value) || 30;
    currentRotateSecs = parseInt(document.getElementById('rotateSeconds').value) || 60;
    rotateSecsLeft = currentRotateSecs;
    const btn = e.target;
    btn.disabled = true; btn.innerText = 'Starting…';
    try {{
        const fd = new FormData();
        fd.append('duration_minutes', duration);
        fd.append('rotate_seconds', currentRotateSecs);
        if (teacherLat !== null) fd.append('teacher_lat', teacherLat);
        if (teacherLng !== null) fd.append('teacher_lng', teacherLng);
        const res = await fetch('/teacher/start-session/' + CLASS_ID, {{
            method: 'POST',
            headers: {{ 'X-Requested-With': 'XMLHttpRequest' }},
            body: fd
        }});
        let data;
        try {{
            data = await res.json();
        }} catch (parseErr) {{
            const text = await res.text().catch(() => '');
            console.error('Non-JSON response from start-session:', res.status, text.slice(0, 300));
            alert('Server error (status ' + res.status + '). Check server logs for details.');
            btn.disabled = false; btn.innerText = '▶ Start Attendance Timer';
            return;
        }}
        if (data.ok) {{
            currentCode = data.code;
            currentRotateSecs = data.rotate_seconds || 60;
            rotateSecsLeft = currentRotateSecs;
            currentExpiresAt = new Date(data.expires_at);

            const codeDisplayEl = document.getElementById('codeDisplay');
            if (codeDisplayEl) codeDisplayEl.innerText = data.code;
            const noSessionEl = document.getElementById('noSession');
            if (noSessionEl) noSessionEl.classList.add('hidden');
            const activeSessionEl = document.getElementById('activeSession');
            if (activeSessionEl) activeSessionEl.classList.remove('hidden');
            const expiredMsg = document.getElementById('sessionExpiredMsg');
            if (expiredMsg) expiredMsg.classList.add('hidden');
            const startFormEl = document.getElementById('startForm');
            if (startFormEl) startFormEl.classList.add('hidden');
            const locationsPanelEl = document.getElementById('locationsPanel');
            if (locationsPanelEl) locationsPanelEl.classList.remove('hidden');
            const sessionPanelEl = document.getElementById('sessionPanel');
            if (sessionPanelEl) {{
                sessionPanelEl.classList.remove('border-slate-200');
                sessionPanelEl.classList.add('border-emerald-400');
            }}
            btn.disabled = false; btn.innerText = '▶ Start Attendance Timer';

            renderQR(data.code);
            startCountdown(currentExpiresAt);
            startQRRotation(currentRotateSecs);
            startLocRefresh();
        }} else {{
            alert(data.message || 'Error starting session.');
            btn.disabled = false; btn.innerText = '▶ Start Attendance Timer';
        }}
    }} catch(e2) {{
        console.error('startSession network error:', e2);
        alert('Network error: ' + (e2 && e2.message ? e2.message : 'request failed') + '. Check your internet connection and try again.');
        btn.disabled = false; btn.innerText = '▶ Start Attendance Timer';
    }}
}}

// ── Dynamic QR rotation ──
function startQRRotation(rotateSecs, elapsedSecs) {{
    if (rotateInterval) clearInterval(rotateInterval);
    currentRotateSecs = rotateSecs;
    elapsedSecs = elapsedSecs || 0;
    // Sync the countdown to how much of the current rotation window has already passed
    rotateSecsLeft = Math.max(1, rotateSecs - (elapsedSecs % rotateSecs));
    // Update the rotate progress bar every second
    rotateInterval = setInterval(async () => {{
        rotateSecsLeft -= 1;
        const pct = Math.max(0, (rotateSecsLeft / currentRotateSecs) * 100);
        const rp = document.getElementById('fsRotateProgress');
        if (rp) rp.style.width = pct + '%';
        if (rotateSecsLeft <= 0) {{
            rotateSecsLeft = currentRotateSecs;
            // Call server to rotate QR code
            try {{
                const res = await fetch('/teacher/rotate-session/' + CLASS_ID, {{ method: 'POST' }});
                const data = await res.json();
                if (data.ok && data.code) {{
                    currentCode = data.code;
                    const cdEl = document.getElementById('codeDisplay');
                    if (cdEl) cdEl.innerText = data.code;
                    renderQR(data.code);
                    // Update fullscreen if open
                    const overlay = document.getElementById('fullscreenOverlay');
                    if (overlay && overlay.style.display !== 'none') {{
                        const fsCodeEl = document.getElementById('fsCode');
                        if (fsCodeEl) fsCodeEl.innerText = data.code;
                        renderFsQR(data.code);
                    }}
                }}
            }} catch(err) {{ console.log('QR rotate error', err); }}
        }}
    }}, 1000);
}}

// ── Live student locations refresh ──
function startLocRefresh() {{
    if (locRefreshInterval) clearInterval(locRefreshInterval);
    refreshLocations();
    locRefreshInterval = setInterval(refreshLocations, 10000);
}}

async function refreshLocations() {{
    try {{
        const res = await fetch('/teacher/student-locations/' + CLASS_ID);
        const data = await res.json();
        const tbody = document.getElementById('locTableBody');
        const teacherInfo = document.getElementById('teacherGpsInfo');
        if (teacherInfo) {{
            if (data.teacher_lat && data.teacher_lng) {{
                teacherInfo.innerText = '📍 Your GPS: ' + data.teacher_lat.toFixed(5) + ', ' + data.teacher_lng.toFixed(5);
            }} else {{
                teacherInfo.innerText = '📍 Teacher GPS: not captured (WiFi/code check active)';
            }}
        }}
        if (!tbody) return;
        if (!data.students || data.students.length === 0) {{
            tbody.innerHTML = '<tr><td colspan="4" class="py-4 text-center text-slate-400 text-xs">No check-ins yet today.</td></tr>';
        }} else {{
            tbody.innerHTML = data.students.map(s => {{
                const distStr = s.distance_meters !== null && s.distance_meters !== undefined
                    ? Math.round(s.distance_meters) + 'm'
                    : '—';
                const distColor = (s.distance_meters !== null && s.distance_meters <= 200)
                    ? 'text-emerald-600 font-bold' : s.distance_meters !== null ? 'text-amber-600 font-semibold' : 'text-slate-400';
                // Status is now a clickable toggle button (was a static <span> with
                // no click handler at all, which is why teachers couldn't flip an
                // Absent student to Present from this panel) — clicking flips the
                // status via /teacher/live-mark/... and updates in place, no reload.
                const badge = s.status === 'Present'
                    ? `<button type="button" class="live-status-btn px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100 cursor-pointer transition" data-id="${{s.student_db_id}}" data-status="Present" onclick="toggleLiveStatus(this)" title="Click to mark Absent">Present</button>`
                    : `<button type="button" class="live-status-btn px-2 py-0.5 rounded-full text-[10px] font-bold bg-red-50 text-red-600 border border-red-200 hover:bg-red-100 cursor-pointer transition" data-id="${{s.student_db_id}}" data-status="Absent" onclick="toggleLiveStatus(this)" title="Click to mark Present">Absent</button>`;
                return `<tr class="border-b border-slate-50" data-student-row="${{s.student_db_id}}">
                    <td class="py-2 pr-3 font-semibold text-slate-800">${{s.full_name}}<br><span class="text-xs font-mono text-slate-400">${{s.student_id}}</span></td>
                    <td class="py-2 pr-3">${{badge}}</td>
                    <td class="py-2 pr-3 ${{distColor}}">${{distStr}}</td>
                    <td class="py-2 text-xs text-slate-400">${{s.time}}</td>
                </tr>`;
            }}).join('');
        }}
        const lastRefreshedEl = document.getElementById('lastRefreshed');
        if (lastRefreshedEl) lastRefreshedEl.innerText = 'Updated: ' + new Date().toLocaleTimeString();
    }} catch(err) {{ console.log('loc refresh error', err); }}
}}

// Flip a student's attendance status from the Live Student Check-ins panel.
// Optimistically updates the button in place so it feels instant, then
// confirms against the server and re-syncs the whole panel — no page
// reload involved either way.
async function toggleLiveStatus(btn) {{
    const studentDbId = btn.getAttribute('data-id');
    const currentStatus = btn.getAttribute('data-status');
    const newStatus = currentStatus === 'Present' ? 'Absent' : 'Present';

    btn.disabled = true;
    const originalText = btn.innerText;
    btn.innerText = '…';

    try {{
        const res = await fetch(`/teacher/live-mark/${{CLASS_ID}}/${{studentDbId}}/${{newStatus}}`, {{
            method: 'POST'
        }});
        const data = await res.json();
        if (data.ok) {{
            // Re-render this row's button immediately instead of waiting for
            // the next 10s poll.
            if (data.status === 'Present') {{
                btn.className = 'live-status-btn px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100 cursor-pointer transition';
                btn.title = 'Click to mark Absent';
                btn.innerText = 'Present';
                btn.setAttribute('data-status', 'Present');
            }} else {{
                btn.className = 'live-status-btn px-2 py-0.5 rounded-full text-[10px] font-bold bg-red-50 text-red-600 border border-red-200 hover:bg-red-100 cursor-pointer transition';
                btn.title = 'Click to mark Present';
                btn.innerText = 'Absent';
                btn.setAttribute('data-status', 'Absent');
            }}
            btn.disabled = false;
            // Pull a fresh snapshot (updates the time column, distance, etc.)
            refreshLocations();
        }} else {{
            btn.innerText = originalText;
            btn.disabled = false;
            alert(data.error || 'Could not update attendance. Please try again.');
        }}
    }} catch(err) {{
        console.log('toggleLiveStatus error', err);
        btn.innerText = originalText;
        btn.disabled = false;
        alert('Network error updating attendance. Please try again.');
    }}
}}

function startCountdown(expiresDate) {{
    if (countdownInterval) clearInterval(countdownInterval);
    totalSeconds = Math.max(1, Math.floor((expiresDate - new Date()) / 1000));

    function tick() {{
        const remaining = Math.floor((expiresDate - new Date()) / 1000);
        if (remaining <= 0) {{
            clearInterval(countdownInterval);
            if (rotateInterval) clearInterval(rotateInterval);
            if (locRefreshInterval) clearInterval(locRefreshInterval);
            const cd = document.getElementById('countdown');
            if (cd) cd.innerText = '00:00';
            const pb = document.getElementById('progressBar');
            if (pb) pb.style.width = '0%';
            // Hide the active-session UI and show a "time's up" message,
            // WITHOUT destroying #codeDisplay / #countdown / #progressBar via innerHTML —
            // those elements must still exist next time startSession() runs.
            const activeSessionEl = document.getElementById('activeSession');
            if (activeSessionEl) activeSessionEl.classList.add('hidden');
            let expiredMsg = document.getElementById('sessionExpiredMsg');
            if (!expiredMsg) {{
                expiredMsg = document.createElement('div');
                expiredMsg.id = 'sessionExpiredMsg';
                expiredMsg.className = 'text-red-500 font-bold text-lg py-6 text-center';
                expiredMsg.innerText = '⏰ Time is up! Attendance window closed.';
                const panel = document.getElementById('sessionPanel');
                if (panel) panel.appendChild(expiredMsg);
            }}
            expiredMsg.classList.remove('hidden');
            const locPanel = document.getElementById('locationsPanel');
            if (locPanel) locPanel.classList.add('hidden');
            const startFormEl = document.getElementById('startForm');
            if (startFormEl) startFormEl.classList.remove('hidden');
            const sessionPanelEl = document.getElementById('sessionPanel');
            if (sessionPanelEl) {{
                sessionPanelEl.classList.remove('border-emerald-400');
                sessionPanelEl.classList.add('border-slate-200');
            }}
            const overlay = document.getElementById('fullscreenOverlay');
            if (overlay && overlay.style.display !== 'none') {{
                const fsCd = document.getElementById('fsCountdown');
                if (fsCd) fsCd.innerText = '⏰ Closed';
            }}
            fetch('/teacher/stop-session/' + CLASS_ID, {{ method: 'POST', headers: {{ 'X-Requested-With': 'XMLHttpRequest' }} }});
            return;
        }}
        const mins = Math.floor(remaining / 60).toString().padStart(2, '0');
        const secs = (remaining % 60).toString().padStart(2, '0');
        const timeStr = mins + ':' + secs;
        const cdEl = document.getElementById('countdown');
        if (cdEl) cdEl.innerText = timeStr;
        const pct = Math.max(0, (remaining / totalSeconds) * 100);
        const pbEl = document.getElementById('progressBar');
        if (pbEl) pbEl.style.width = pct + '%';

        if (document.getElementById('fsCountdown')) {{
            document.getElementById('fsCountdown').innerText = timeStr;
            document.getElementById('fsProgressBar').style.width = pct + '%';
        }}

        if (remaining <= 60) {{
            document.getElementById('countdown').className = 'text-5xl font-black font-mono text-red-500';
            document.getElementById('progressBar').className = 'h-2 bg-red-500 rounded-full transition-all duration-1000';
            if(document.getElementById('fsCountdown')) document.getElementById('fsCountdown').className = 'mt-3 text-4xl font-black font-mono text-red-500';
            if(document.getElementById('fsProgressBar')) document.getElementById('fsProgressBar').className = 'h-2 bg-red-500 rounded-full transition-all duration-1000';
        }} else if (remaining <= 180) {{
            document.getElementById('countdown').className = 'text-5xl font-black font-mono text-amber-500';
            document.getElementById('progressBar').className = 'h-2 bg-amber-400 rounded-full transition-all duration-1000';
        }}
    }}
    tick();
    countdownInterval = setInterval(tick, 1000);
}}

function openFullscreen() {{
    if (!currentCode) return;
    document.getElementById('fsCode').innerText = currentCode;
    renderFsQR(currentCode);
    const overlay = document.getElementById('fullscreenOverlay');
    overlay.classList.remove('hidden');
    overlay.style.display = 'flex';
    overlay.scrollTop = 0;
    document.addEventListener('keydown', _fsEscHandler);
}}

function closeFullscreen() {{
    const overlay = document.getElementById('fullscreenOverlay');
    overlay.classList.add('hidden');
    overlay.style.display = 'none';
    document.removeEventListener('keydown', _fsEscHandler);
}}

function _fsEscHandler(e) {{
    if (e.key === 'Escape') closeFullscreen();
}}

// On load: if session already active, render QR, start countdown, rotation and locations
if (currentCode && currentExpiresAt) {{
    renderQR(currentCode);
    startCountdown(currentExpiresAt);
    const elapsedSinceRotate = initialLastRotated ? Math.floor((new Date() - initialLastRotated) / 1000) : 0;
    startQRRotation(initialRotateSecs || 60, elapsedSinceRotate);
    startLocRefresh();
    document.getElementById('locationsPanel').classList.remove('hidden');
}}
</script>
    """
    return page_wrapper(f"Attendance Timer — {class_row['class_name']}", body, is_teacher=True, teacher_name=session.get("teacher_name"))


@app.route("/teacher/stop-session/<int:class_id>", methods=["POST"])
def teacher_stop_session(class_id):
    protect = teacher_required()
    if protect:
        return protect
    school_id = get_current_school_id()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE class_sessions SET active=FALSE WHERE class_id=%s AND school_id=%s", (class_id, school_id))
    conn.commit()
    conn.close()
    if is_ajax():
        return ajax_ok("Session ended. Code is no longer valid.")
    return "<script>alert('Session ended.');window.location.href='/teacher';</script>"


@app.route("/teacher/active-session/<int:class_id>")
def teacher_active_session(class_id):
    """Returns current active session code for a class (AJAX)."""
    protect = teacher_required()
    if protect:
        return jsonify({"code": None})
    school_id = get_current_school_id()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT code, expires_at FROM class_sessions
        WHERE class_id=%s AND school_id=%s AND active=TRUE AND expires_at > NOW()
        ORDER BY id DESC LIMIT 1
    """, (class_id, school_id))
    row = cur.fetchone()
    conn.close()
    if row:
        return jsonify({"code": row["code"], "expires_at": utc_iso(row["expires_at"])})
    return jsonify({"code": None})


# =========================================================
# ADMIN — LOCATION SETTINGS (GPS + WiFi)
# =========================================================
@app.route("/admin/location-settings", methods=["GET", "POST"])
def admin_location_settings():
    protect = admin_required()
    if protect:
        return protect
    school_id = get_current_school_id()

    if request.method == "POST":
        lat = request.form.get("classroom_lat", "").strip()
        lng = request.form.get("classroom_lng", "").strip()
        radius = request.form.get("classroom_radius", "100").strip()
        ip_prefix = request.form.get("allowed_ip_prefix", "").strip()
        conn = get_db()
        cur = conn.cursor()
        for key, val in [("classroom_lat", lat), ("classroom_lng", lng),
                         ("classroom_radius", radius), ("allowed_ip_prefix", ip_prefix)]:
            if val:
                cur.execute("""
                    INSERT INTO admin_settings (school_id, key, value) VALUES (%s, %s, %s)
                    ON CONFLICT (school_id, key) DO UPDATE SET value=%s
                """, (school_id, key, val, val))
            else:
                cur.execute("DELETE FROM admin_settings WHERE school_id=%s AND key=%s", (school_id, key))
        conn.commit()
        conn.close()
        if is_ajax():
            return ajax_ok("Location settings saved!", redirect_url="/admin/location-settings")
        return "<script>alert('Location settings saved!');window.location.href='/admin/location-settings';</script>"

    # Load current settings
    conn = get_db()
    cur = conn.cursor()
    def get_setting(key):
        cur.execute("SELECT value FROM admin_settings WHERE school_id=%s AND key=%s", (school_id, key))
        r = cur.fetchone()
        return r["value"] if r else ""
    lat = get_setting("classroom_lat")
    lng = get_setting("classroom_lng")
    radius = get_setting("classroom_radius") or "100"
    ip_prefix = get_setting("allowed_ip_prefix")
    conn.close()

    body = f"""
    <div class="max-w-xl space-y-6">
        <div>
            <h1 class="text-2xl font-bold text-slate-800">📍 Attendance Location Settings</h1>
            <p class="text-sm text-slate-500 mt-1">Configure GPS, WiFi, and session code restrictions. Students need to pass <strong>at least one</strong> check to be marked Present.</p>
        </div>

        <form method="POST" class="space-y-6 bg-white border rounded-xl p-6 shadow-sm" data-ajax>
                            {csrf_input_html()}

            <div class="space-y-3">
                <h2 class="font-bold text-slate-700">🛰️ GPS Location</h2>
                <p class="text-xs text-slate-400">Set your classroom's GPS coordinates. Leave blank to disable GPS check.</p>
                <div class="grid grid-cols-2 gap-3">
                    <div>
                        <label class="block text-xs font-semibold text-slate-600 mb-1">Latitude</label>
                        <input type="text" name="classroom_lat" value="{lat}" placeholder="e.g. 9.0247" class="form-input w-full">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-600 mb-1">Longitude</label>
                        <input type="text" name="classroom_lng" value="{lng}" placeholder="e.g. 38.7469" class="form-input w-full">
                    </div>
                </div>
                <div>
                    <label class="block text-xs font-semibold text-slate-600 mb-1">Allowed Radius (meters)</label>
                    <input type="number" name="classroom_radius" value="{radius}" min="10" max="500" class="form-input w-full">
                </div>
                <button type="button" onclick="detectLocation()" class="text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 font-semibold px-3 py-1.5 rounded-lg">
                    📡 Use My Current Location
                </button>
                <div id="detectStatus" class="text-xs text-slate-400"></div>
            </div>

            <hr>

            <div class="space-y-3">
                <h2 class="font-bold text-slate-700">📶 WiFi / Network Check</h2>
                <p class="text-xs text-slate-400">Enter the IP prefix of your school's WiFi network. Leave blank to disable. Example: <code class="bg-slate-100 px-1 rounded">192.168.1.</code></p>
                <input type="text" name="allowed_ip_prefix" value="{ip_prefix}" placeholder="e.g. 192.168.1." class="form-input w-full font-mono">
            </div>

            <hr>

            <div class="space-y-2">
                <h2 class="font-bold text-slate-700">📟 Session Codes (QR)</h2>
                <p class="text-xs text-slate-400">Teachers generate session codes from their class panel. No configuration needed here — codes expire after 30 minutes.</p>
            </div>

            <button type="submit" class="btn green w-full">💾 Save Settings</button>
        </form>
    </div>
    <script>
    function detectLocation() {{
        const status = document.getElementById('detectStatus');
        status.innerText = 'Detecting...';
        navigator.geolocation.getCurrentPosition(function(pos) {{
            document.querySelector('[name=classroom_lat]').value = pos.coords.latitude.toFixed(6);
            document.querySelector('[name=classroom_lng]').value = pos.coords.longitude.toFixed(6);
            status.innerText = '✅ Location captured: ' + pos.coords.latitude.toFixed(6) + ', ' + pos.coords.longitude.toFixed(6);
            status.className = 'text-xs text-emerald-600 font-semibold';
        }}, function(err) {{
            status.innerText = '❌ Could not detect location: ' + err.message;
            status.className = 'text-xs text-red-500';
        }}, {{ enableHighAccuracy: true }});
    }}
    </script>
    """
    return page_wrapper("Location Settings", body, is_admin=True)


# =========================================================
# IMAGES ASSET LOADING HANDLER DISPATCH CONTROLLERS
# =========================================================
@app.route("/student-image/<path:filename>")
def student_image(filename):
    # If it's a full URL (Supabase), redirect directly
    if filename.startswith("http"):
        return redirect(filename)
    file_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(file_path):
        return "Image not found", 404
    ext = filename.lower().split(".")[-1]
    mime = "image/png" if ext == "png" else "image/jpeg"
    with open(file_path, "rb") as f:
        return Response(f.read(), mimetype=mime)


@app.route("/export-attendance")
def export_attendance():
    protect = admin_required()
    if protect:
        return protect

    attendance = get_all_attendance()
    csv_data = "StudentID,FullName,ClassName,Department,Course,Section,Subject,Teacher,Status,Date,Time\n"
    for r in attendance:
        csv_data += (
            f'{r["student_id"]},{r["full_name"]},{r["class_name"]},'
            f'{r["department"] or ""},{r["course"] or ""},{r["section_name"] or ""},'
            f'{r["subject_name"] or ""},{r["teacher_name"] or ""},{r["status"]},{r["date"]},{r["time"]}\n'
        )
    return Response(csv_data, mimetype="text/csv", headers={"Content-disposition": "attachment; filename=attendance_sheet.csv"})




# =========================================================


# SUPER-ADMIN — SCHOOL MANAGEMENT
# Password is stored in the database (super_admin_settings table).
# Default on first run: "superadmin123"
# =========================================================

def get_super_admin_setting(key, default=""):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT value FROM super_admin_settings WHERE key=%s", (key,))
        row = cur.fetchone()
        conn.close()
        return row["value"] if row else default
    except:
        return default

def set_super_admin_setting(key, value):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO super_admin_settings (key, value) VALUES (%s, %s)
        ON CONFLICT (key) DO UPDATE SET value=%s
    """, (key, value, value))
    conn.commit()
    conn.close()

def get_super_admin_password():
    return get_super_admin_setting("password", "superadmin123")

def get_super_admin_name():
    return get_super_admin_setting("name", "Super Admin")

def set_super_admin_password(new_pw):
    set_super_admin_setting("password", hash_password(new_pw))

def init_super_admin_table():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS super_admin_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    cur.execute("""
        INSERT INTO super_admin_settings (key, value) VALUES ('password', %s)
        ON CONFLICT (key) DO NOTHING
    """, (hash_password("superadmin123"),))
    cur.execute("""
        INSERT INTO super_admin_settings (key, value) VALUES ('name', 'Super Admin')
        ON CONFLICT (key) DO NOTHING
    """)
    conn.commit()
    conn.close()

def is_super_admin():
    return session.get("super_admin_logged_in") is True

@app.route("/super-admin-login", methods=["GET", "POST"])
def super_admin_login():
    err = ""
    if request.method == "POST":
        if login_rate_limited("super-admin-login"):
            return too_many_attempts_response()
        pw = request.form.get("password", "").strip()
        stored_pw = get_super_admin_password()
        if verify_password(stored_pw, pw):
            if not is_hashed_password(stored_pw):
                set_super_admin_password(pw)
            session["super_admin_logged_in"] = True
            session["super_admin_name"] = get_super_admin_name()
            return redirect("/super-admin")
        err = "Incorrect password."
    return page_wrapper("Super Admin Login", f"""
        <div class="max-w-md mx-auto my-8 p-6 bg-white border rounded-xl shadow-sm">
            <h2 class="text-2xl font-bold text-slate-800 text-center mb-2">🏫 Super Admin</h2>
            <p class="text-sm text-slate-500 text-center mb-5">Manage all schools on this platform</p>
            <form method="POST" class="space-y-4">
                            {csrf_input_html()}
                <input type="password" name="password" placeholder="Super Admin Password"
                    class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-slate-500" required>
                <button class="w-full bg-slate-800 hover:bg-slate-900 text-white font-bold py-2 rounded-lg" type="submit">Sign In</button>
            </form>
            {'<p class="mt-3 text-sm text-red-500 text-center font-semibold">' + err + '</p>' if err else ''}
            <div class="mt-4 text-center"><a class="text-sm text-blue-600 hover:underline" href="/">← Back to Home</a></div>
        </div>""")

@app.route("/super-admin/change-credentials", methods=["POST"])
def super_admin_change_credentials():
    if not is_super_admin():
        return redirect("/super-admin-login")
    new_name = request.form.get("new_name", "").strip()
    current = request.form.get("current_password", "").strip()
    new_pw = request.form.get("new_password", "").strip()
    confirm = request.form.get("confirm_password", "").strip()
    if not verify_password(get_super_admin_password(), current):
        if is_ajax(): return ajax_err("Incorrect current password.")
        return "<script>alert('Incorrect current password.');window.location.href='/super-admin';</script>"
    if new_name:
        set_super_admin_setting("name", new_name)
        session["super_admin_name"] = new_name
    if new_pw:
        if new_pw != confirm:
            if is_ajax(): return ajax_err("New passwords do not match.")
            return "<script>alert('New passwords do not match.');window.location.href='/super-admin';</script>"
        set_super_admin_password(new_pw)
    if is_ajax(): return ajax_ok("Credentials updated successfully!", redirect_url="/super-admin")
    return "<script>alert('Credentials updated successfully!');window.location.href='/super-admin';</script>"

@app.route("/super-admin")
def super_admin_dashboard():
    if not is_super_admin():
        return redirect("/super-admin-login")
    schools = get_all_schools()
    super_name = session.get("super_admin_name", get_super_admin_name())

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM pending_school_registrations ORDER BY created_at DESC")
    pending_requests = cur.fetchall()
    conn.close()

    pending_rows_html = ""
    for p in pending_requests:
        safe_pname = p['name'].replace("'", "\\'")
        pending_rows_html += f"""
        <tr id="pending-row-{p['id']}">
            <td class="p-3"><input type="checkbox" class="pending-cb w-4 h-4 rounded accent-emerald-600" value="{p['id']}"></td>
            <td class="p-3 font-semibold">{p['name']}</td>
            <td class="p-3 font-mono font-bold text-blue-700">{p['code']}</td>
            <td class="p-3 text-slate-500 text-sm">{p['admin_username']}</td>
            <td class="p-3 text-xs text-slate-400">{p['created_at']}</td>
            <td class="p-3 whitespace-nowrap">
                <button class="text-emerald-600 hover:text-emerald-800 text-sm font-bold mr-3"
                   onclick="handlePending({p['id']}, 'approve', '{safe_pname}')">✅ Approve</button>
                <button class="text-red-500 hover:text-red-700 text-sm font-medium"
                   onclick="handlePending({p['id']}, 'reject', '{safe_pname}')">Reject</button>
            </td>
        </tr>"""

    rows_html = ""
    for s in schools:
        safe_name = s['name'].replace("'", "\\'")
        rows_html += f"""
        <tr id="school-row-{s['id']}">
            <td class="p-3 font-mono text-xs text-slate-500">#{s['id']}</td>
            <td class="p-3 font-semibold">{s['name']}</td>
            <td class="p-3 font-mono font-bold text-blue-700 text-lg">{s['code']}</td>
            <td class="p-3 text-slate-500 text-sm">{s['admin_username']}</td>
            <td class="p-3 font-mono text-xs text-slate-400">••••••••</td>
            <td class="p-3 text-xs text-slate-400">{s['created_at']}</td>
            <td class="p-3 whitespace-nowrap">
                <a class="text-blue-600 hover:underline text-sm font-medium mr-3"
                   href="/super-admin/edit-school/{s['id']}">Edit</a>
                <button class="text-red-500 hover:text-red-700 text-sm font-medium"
                   onclick="deleteSchool({s['id']}, '{safe_name}')">Delete</button>
            </td>
        </tr>"""
    body = f"""
    <div class="space-y-6">
        <div class="flex items-center justify-between flex-wrap gap-3">
            <div>
                <h1 class="text-3xl font-bold text-slate-800">🏫 School Manager</h1>
                <p class="text-sm text-slate-500 mt-1">Logged in as <b>{super_name}</b> · {len(schools)} school(s) on this platform</p>
            </div>
            <form method="POST" action="/super-admin-logout">
                            {csrf_input_html()}
                <button class="bg-slate-700 hover:bg-slate-900 text-white font-bold py-2 px-4 rounded-lg text-sm">Sign Out</button>
            </form>
        </div>

        <div class="card card-body">
            <h2 class="text-xl font-bold text-slate-800 mb-1">🔐 Update Credentials</h2>
            <p class="text-sm text-slate-500 mb-4">Change your display name and/or password. Leave new password blank to keep current.</p>
            <form method="POST" action="/super-admin/change-credentials" class="grid grid-cols-1 md:grid-cols-2 gap-3" data-ajax>
                            {csrf_input_html()}
                <div class="md:col-span-2">
                    <label class="block text-sm font-medium text-slate-700 mb-1">Display Name</label>
                    <input type="text" name="new_name" value="{super_name}" class="form-input">
                </div>
                <div class="md:col-span-2">
                    <label class="block text-sm font-medium text-slate-700 mb-1">Current Password <span class="text-red-500">*</span></label>
                    <input type="password" name="current_password" class="form-input" required>
                </div>
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">New Password <span class="text-slate-400">(leave blank to keep)</span></label>
                    <input type="password" name="new_password" class="form-input">
                </div>
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Confirm New Password</label>
                    <input type="password" name="confirm_password" class="form-input">
                </div>
                <div class="md:col-span-2 pt-1">
                    <button class="btn green" type="submit">Save Changes</button>
                </div>
            </form>
        </div>

        <div class="card card-body">
            <h2 class="text-xl font-bold text-slate-800 mb-4">➕ Register New School</h2>
            <p class="text-sm text-slate-500 mb-4">Each school gets a unique <b>School Code</b>. Staff and students enter this code when logging in to access their school's data.</p>
            <form method="POST" action="/super-admin/create-school" class="grid grid-cols-1 md:grid-cols-2 gap-3" data-ajax>
                            {csrf_input_html()}
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">School Full Name</label>
                    <input type="text" name="name" placeholder="e.g. Green Hills Secondary School" class="form-input" required>
                </div>
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">School Code <span class="text-slate-400">(short, unique, no spaces)</span></label>
                    <input type="text" name="code" placeholder="e.g. GREENHS or ABC01" class="form-input" required>
                </div>
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Admin Username</label>
                    <input type="text" name="admin_username" placeholder="e.g. principal" class="form-input" required>
                </div>
                <div>
                    <label class="block text-sm font-medium text-slate-700 mb-1">Admin Password</label>
                    <input type="text" name="admin_password" placeholder="Strong password" class="form-input" required>
                </div>
                <div class="md:col-span-2 pt-1">
                    <button class="btn green" type="submit">Create School</button>
                </div>
            </form>
        </div>
        <div class="card">
            <div class="card-header">
                <span class="card-header-title">📥 Pending School Requests {f'<span class="ml-2 inline-block bg-amber-100 text-amber-700 text-xs font-bold px-2 py-0.5 rounded-full">{len(pending_requests)}</span>' if pending_requests else ''}</span>
                <div class="flex gap-2" id="pending-bulk-bar" style="display:none!important;">
                    <span class="text-xs text-slate-500 self-center" id="pending-sel-count"></span>
                    <button onclick="bulkPending('approve')" class="bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold px-3 py-1.5 rounded-lg">✅ Approve Selected</button>
                    <button onclick="bulkPending('reject')" class="bg-red-500 hover:bg-red-600 text-white text-xs font-bold px-3 py-1.5 rounded-lg">✗ Reject Selected</button>
                </div>
            </div>
            <div class="tbl-wrap">
            <table>
                <thead><tr>
                    <th class="w-8"><input type="checkbox" id="pending-select-all" class="w-4 h-4 rounded accent-emerald-600" onclick="toggleAllPending(this)"></th>
                    <th>School Name</th><th>Requested Code</th>
                    <th>Admin Username</th><th>Submitted</th><th>Action</th>
                </tr></thead>
                <tbody id="pending-tbody">{pending_rows_html if pending_rows_html else "<tr><td colspan='6' style='padding:24px;text-align:center;color:#94a3b8;'>No pending requests.</td></tr>"}</tbody>
            </table>
            </div>
        </div>
        <div class="card">
            <div class="card-header"><span class="card-header-title">All Schools</span></div>
            <div class="tbl-wrap">
            <table>
                <thead><tr>
                    <th>ID</th><th>School Name</th><th>Code</th>
                    <th>Admin Username</th><th>Admin Password</th><th>Created</th><th>Action</th>
                </tr></thead>
                <tbody>{rows_html if rows_html else "<tr><td colspan='7' style='padding:24px;text-align:center;color:#94a3b8;'>No schools yet.</td></tr>"}</tbody>
            </table>
            </div>
        </div>
    </div>
    """
    # JS is outside the f-string to avoid escaping issues with arrow functions and braces
    body += """
    <script>
    // ── PENDING CHECKBOXES ──
    function toggleAllPending(master) {
        document.querySelectorAll('.pending-cb').forEach(function(cb) { cb.checked = master.checked; });
        updateBulkBar();
    }
    document.addEventListener('change', function(e) {
        if (e.target.classList.contains('pending-cb')) updateBulkBar();
    });
    function getCheckedPending() {
        return Array.from(document.querySelectorAll('.pending-cb:checked')).map(function(cb) { return cb.value; });
    }
    function updateBulkBar() {
        var ids = getCheckedPending();
        var bar = document.getElementById('pending-bulk-bar');
        var count = document.getElementById('pending-sel-count');
        if (ids.length > 0) {
            bar.style.cssText = 'display:flex!important;align-items:center;gap:8px;';
            count.textContent = ids.length + ' selected';
        } else {
            bar.style.cssText = 'display:none!important;';
        }
        var all = document.querySelectorAll('.pending-cb');
        var selAll = document.getElementById('pending-select-all');
        selAll.indeterminate = ids.length > 0 && ids.length < all.length;
        selAll.checked = ids.length === all.length && all.length > 0;
    }

    // ── SINGLE APPROVE / REJECT ──
    async function handlePending(id, action, name) {
        var msg = action === 'approve'
            ? 'Approve "' + name + '"? It will go live immediately.'
            : 'Reject and delete the request for "' + name + '"?';
        if (!confirm(msg)) return;
        var url = action === 'approve'
            ? '/super-admin/approve-school/' + id
            : '/super-admin/reject-school/' + id;
        try {
            var res = await fetch(url, { method: 'POST', headers: { 'X-Requested-With': 'XMLHttpRequest' } });
            var data = await res.json();
            if (data.ok) {
                showToast(data.message, 'success');
                var row = document.getElementById('pending-row-' + id);
                if (row) row.remove();
                updateBulkBar();
            } else {
                showToast(data.error || 'Something went wrong.', 'error');
            }
        } catch(e) { showToast('Network error.', 'error'); }
    }

    // ── BULK APPROVE / REJECT ──
    async function bulkPending(action) {
        var ids = getCheckedPending();
        if (!ids.length) return;
        var label = action === 'approve' ? 'Approve' : 'Reject';
        if (!confirm(label + ' ' + ids.length + ' selected request(s)?')) return;
        try {
            var res = await fetch('/super-admin/bulk-pending', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
                body: JSON.stringify({ ids: ids, action: action })
            });
            var data = await res.json();
            if (data.ok) {
                showToast(data.message, 'success');
                ids.forEach(function(id) {
                    var row = document.getElementById('pending-row-' + id);
                    if (row) row.remove();
                });
                updateBulkBar();
            } else {
                showToast(data.error || 'Something went wrong.', 'error');
            }
        } catch(e) { showToast('Network error.', 'error'); }
    }

    // ── DELETE SCHOOL ──
    async function deleteSchool(id, name) {
        if (!confirm('Delete "' + name + '" and ALL its data? This cannot be undone.')) return;
        try {
            var res = await fetch('/super-admin/delete-school/' + id, { method: 'POST', headers: { 'X-Requested-With': 'XMLHttpRequest' } });
            var data = await res.json();
            if (data.ok) {
                showToast(data.message, 'success');
                var row = document.getElementById('school-row-' + id);
                if (row) row.remove();
            } else {
                showToast(data.error || 'Something went wrong.', 'error');
            }
        } catch(e) { showToast('Network error.', 'error'); }
    }
    </script>
    """
    return page_wrapper("School Manager", body, is_admin=True)

@app.route("/super-admin/create-school", methods=["POST"])
def super_admin_create_school():
    if not is_super_admin():
        return redirect("/super-admin-login")
    name = request.form.get("name", "").strip()
    code = request.form.get("code", "").strip().upper().replace(" ", "")
    admin_username = request.form.get("admin_username", "").strip()
    admin_password = request.form.get("admin_password", "").strip()
    if not name or not code or not admin_username or not admin_password:
        if is_ajax(): return ajax_err("All fields are required.")
        return "<script>alert('All fields are required');window.location.href='/super-admin';</script>"
    conn = get_db()
    cur = conn.cursor()
    hashed_admin_password = hash_password(admin_password)
    try:
        cur.execute("""
            INSERT INTO schools (name, code, admin_username, admin_password, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (name, code, admin_username, hashed_admin_password, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        cur.execute("SELECT id FROM schools WHERE code=%s", (code,))
        new_school = cur.fetchone()
        if new_school:
            cur.execute("""
                INSERT INTO admin_settings (school_id, key, value) VALUES (%s, 'admin_password', %s)
                ON CONFLICT (school_id, key) DO UPDATE SET value=%s
            """, (new_school["id"], hashed_admin_password, hashed_admin_password))
        conn.commit()
    except Exception as e:
        conn.rollback()
        conn.close()
        msg = str(e).split('\n')[0].replace("'", "")
        if is_ajax(): return ajax_err(f"Error: {msg}")
        return f"<script>alert('Error: {msg}');window.location.href='/super-admin';</script>"
    conn.close()
    if is_ajax(): return ajax_ok(f"School '{name}' created! Code: {code}", redirect_url="/super-admin")
    return f"<script>alert('School {name} created! Code: {code}');window.location.href='/super-admin';</script>"

@app.route("/super-admin/edit-school/<int:school_id>", methods=["GET", "POST"])
def super_admin_edit_school(school_id):
    if not is_super_admin():
        return redirect("/super-admin-login")
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM schools WHERE id=%s", (school_id,))
    school = cur.fetchone()
    if not school:
        conn.close()
        return "School not found", 404

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        code = request.form.get("code", "").strip().upper().replace(" ", "")
        admin_username = request.form.get("admin_username", "").strip()
        admin_password = request.form.get("admin_password", "").strip()

        if not name or not code or not admin_username:
            if is_ajax(): return ajax_err("All fields are required.")
            return "<script>alert('All fields are required');window.location.href='/super-admin';</script>"

        try:
            if admin_password:
                hashed_admin_password = hash_password(admin_password)
                cur.execute("""
                    UPDATE schools SET name=%s, code=%s, admin_username=%s, admin_password=%s
                    WHERE id=%s
                """, (name, code, admin_username, hashed_admin_password, school_id))
                # Also keep admin_settings in sync
                cur.execute("""
                    INSERT INTO admin_settings (school_id, key, value) VALUES (%s, 'admin_password', %s)
                    ON CONFLICT (school_id, key) DO UPDATE SET value=%s
                """, (school_id, hashed_admin_password, hashed_admin_password))
            else:
                cur.execute("""
                    UPDATE schools SET name=%s, code=%s, admin_username=%s
                    WHERE id=%s
                """, (name, code, admin_username, school_id))
            conn.commit()
        except Exception as e:
            conn.rollback()
            conn.close()
            msg = str(e).split('\n')[0].replace("'", "")
            if is_ajax(): return ajax_err(f"Error: {msg}")
            return f"<script>alert('Error: {msg}');window.location.href='/super-admin';</script>"
        conn.close()
        if is_ajax(): return ajax_ok(f"School '{name}' updated successfully!", redirect_url="/super-admin")
        return f"<script>alert('School updated successfully!');window.location.href='/super-admin';</script>"

    conn.close()
    body = f"""
    <div class="max-w-lg">
        <div class="flex items-center gap-3 mb-6">
            <a href="/super-admin" class="text-slate-400 hover:text-slate-700 text-sm font-medium">← Back to School Manager</a>
        </div>
        <h1 class="text-2xl font-bold text-slate-800 mb-1">✏️ Edit School</h1>
        <p class="text-sm text-slate-500 mb-6">Update any detail for <b>{school['name']}</b>. Changes take effect immediately.</p>

        <div class="card card-body space-y-4">
            <form method="POST" class="space-y-4" data-ajax>
                            {csrf_input_html()}
                <div>
                    <label class="block text-sm font-semibold text-slate-700 mb-1">School Full Name</label>
                    <input type="text" name="name" value="{school['name']}" class="form-input" required>
                </div>
                <div>
                    <label class="block text-sm font-semibold text-slate-700 mb-1">School Code <span class="text-slate-400 font-normal">(short, unique, no spaces)</span></label>
                    <input type="text" name="code" value="{school['code']}" class="form-input" required>
                    <p class="text-xs text-amber-600 mt-1">⚠️ Changing the code means all users must use the new code to log in.</p>
                </div>
                <div>
                    <label class="block text-sm font-semibold text-slate-700 mb-1">Admin Username</label>
                    <input type="text" name="admin_username" value="{school['admin_username']}" class="form-input" required>
                </div>
                <div>
                    <label class="block text-sm font-semibold text-slate-700 mb-1">Admin Password</label>
                    <input type="text" name="admin_password" placeholder="Leave blank to keep current password" class="form-input">
                </div>
                <div class="pt-2 flex gap-3">
                    <button type="submit" class="btn green">Save Changes</button>
                    <a href="/super-admin" class="inline-flex items-center px-4 py-2 bg-slate-100 text-slate-700 font-semibold rounded-lg hover:bg-slate-200 text-sm">Cancel</a>
                </div>
            </form>
        </div>
    </div>
    """
    return page_wrapper(f"Edit School — {school['name']}", body, is_admin=True)


@app.route("/super-admin/delete-school/<int:school_id>", methods=["GET", "POST"])
def super_admin_delete_school(school_id):
    if not is_super_admin():
        return redirect("/super-admin-login")
    if school_id == 1:
        if is_ajax(): return ajax_err("Cannot delete the default school.")
        return "<script>alert('Cannot delete the default school.');window.location.href='/super-admin';</script>"
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM attendance WHERE school_id=%s", (school_id,))
    cur.execute("SELECT id FROM classes WHERE school_id=%s", (school_id,))
    class_ids = [r["id"] for r in cur.fetchall()]
    if class_ids:
        cur.execute("DELETE FROM student_classes WHERE class_id_fk = ANY(%s)", (class_ids,))
    cur.execute("DELETE FROM classes WHERE school_id=%s", (school_id,))
    cur.execute("DELETE FROM students WHERE school_id=%s", (school_id,))
    cur.execute("DELETE FROM teachers WHERE school_id=%s", (school_id,))
    cur.execute("DELETE FROM admin_settings WHERE school_id=%s", (school_id,))
    cur.execute("DELETE FROM schools WHERE id=%s", (school_id,))
    conn.commit()
    conn.close()
    if is_ajax(): return ajax_ok("School and all its data deleted.")
    return "<script>alert('School and all its data deleted.');window.location.href='/super-admin';</script>"

@app.route("/super-admin/bulk-pending", methods=["POST"])
def super_admin_bulk_pending():
    if not is_super_admin():
        return ajax_err("Not authorized.")
    data = request.get_json()
    if not data:
        return ajax_err("No data received.")
    ids = data.get("ids", [])
    action = data.get("action", "")
    if not ids or action not in ("approve", "reject"):
        return ajax_err("Invalid request.")

    conn = get_db()
    cur = conn.cursor()
    approved = 0
    rejected = 0
    errors = []

    for pid in ids:
        try:
            pid = int(pid)
        except:
            continue
        if action == "approve":
            cur.execute("SELECT * FROM pending_school_registrations WHERE id=%s", (pid,))
            pending = cur.fetchone()
            if not pending:
                errors.append(f"Request #{pid} not found.")
                continue
            try:
                cur.execute("""
                    INSERT INTO schools (name, code, admin_username, admin_password, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                """, (pending["name"], pending["code"], pending["admin_username"], pending["admin_password"],
                      datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                cur.execute("SELECT id FROM schools WHERE code=%s", (pending["code"],))
                new_school = cur.fetchone()
                if new_school:
                    cur.execute("""
                        INSERT INTO admin_settings (school_id, key, value) VALUES (%s, 'admin_password', %s)
                        ON CONFLICT (school_id, key) DO UPDATE SET value=%s
                    """, (new_school["id"], pending["admin_password"], pending["admin_password"]))
                cur.execute("DELETE FROM pending_school_registrations WHERE id=%s", (pid,))
                conn.commit()
                approved += 1
            except Exception as e:
                conn.rollback()
                errors.append(f"{pending['name']}: {str(e).split(chr(10))[0]}")
        else:
            cur.execute("DELETE FROM pending_school_registrations WHERE id=%s", (pid,))
            conn.commit()
            rejected += 1

    conn.close()
    parts = []
    if approved: parts.append(f"{approved} approved")
    if rejected: parts.append(f"{rejected} rejected")
    msg = ", ".join(parts) + " successfully."
    if errors:
        msg += " Errors: " + "; ".join(errors)
    return ajax_ok(msg)


@app.route("/super-admin-logout", methods=["POST", "GET"])
def super_admin_logout():
    session.pop("super_admin_logged_in", None)
    return redirect("/super-admin-login")


# =========================================================
# TEACHER–STUDENT DIRECT MESSAGES (teacher inbox + reply)
# =========================================================
# We use a separate table so teacher IDs (teachers.id) and student IDs (students.id)
# don't collide with the student–student direct_messages table.
#
# Schema (created lazily via init_db extension below):
#   teacher_student_messages(id, school_id, class_id,
#       sender_type  TEXT  -- 'student' | 'teacher'
#       student_db_id INTEGER, teacher_db_id INTEGER,
#       sender_name TEXT, sender_image TEXT,
#       message TEXT, file_url TEXT, file_name TEXT,
#       is_read_by_teacher BOOL, is_read_by_student BOOL,
#       created_at TIMESTAMP)

def _ensure_teacher_msg_table():
    """Create the teacher–student DM table if it doesn't exist yet."""
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS teacher_student_messages (
                id SERIAL PRIMARY KEY,
                school_id INTEGER NOT NULL DEFAULT 1,
                class_id INTEGER NOT NULL,
                sender_type TEXT NOT NULL,
                student_db_id INTEGER NOT NULL,
                teacher_db_id INTEGER NOT NULL,
                sender_name TEXT NOT NULL,
                sender_image TEXT,
                message TEXT NOT NULL,
                file_url TEXT,
                file_name TEXT,
                is_read_by_teacher BOOLEAN NOT NULL DEFAULT FALSE,
                is_read_by_student BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tsm_teacher ON teacher_student_messages(teacher_db_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tsm_student ON teacher_student_messages(student_db_id)")
        conn.commit()
        conn.close()
    except Exception as e:
        print("_ensure_teacher_msg_table error:", e)


# ── STUDENT: view teacher profile page ──────────────────────────────────────
@app.route("/student/teacher-profile/<int:teacher_db_id>")
def student_view_teacher_profile(teacher_db_id):
    protect = student_required()
    if protect:
        return protect

    student_db_id = get_logged_student_db_id()
    school_id = get_current_school_id()
    student_ctx = get_student_row_by_db_id(student_db_id)

    # Fetch teacher row
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM teachers WHERE id=%s AND school_id=%s", (teacher_db_id, school_id))
    teacher = cur.fetchone()
    conn.close()

    if not teacher:
        return "<script>alert('Teacher not found.');window.location.href='/student/classes';</script>"

    # Check student shares at least one class with this teacher
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT 1 FROM student_classes sc
        INNER JOIN classes c ON c.id = sc.class_id_fk
        WHERE sc.student_id_fk=%s AND c.teacher_id=%s AND c.school_id=%s
        LIMIT 1
    """, (student_db_id, teacher_db_id, school_id))
    shared = cur.fetchone()

    # Classes this teacher teaches that the student is in
    cur.execute("""
        SELECT c.* FROM classes c
        INNER JOIN student_classes sc ON sc.class_id_fk = c.id
        WHERE sc.student_id_fk=%s AND c.teacher_id=%s AND c.school_id=%s
        ORDER BY c.class_name
    """, (student_db_id, teacher_db_id, school_id))
    shared_classes = cur.fetchall()
    conn.close()

    if not shared:
        return "<script>alert('You can only view teachers of your classes.');window.location.href='/student/classes';</script>"

    photo_url = supabase_public_url(teacher.get("photo_path") or "") if teacher.get("photo_path") else ""
    letter = (teacher["teacher_name"] or "T")[0].upper()

    if photo_url:
        av_html = f'<img src="{photo_url}" style="width:90px;height:90px;border-radius:50%;object-fit:cover;flex-shrink:0;">'
    else:
        av_html = f'<div style="width:90px;height:90px;border-radius:50%;background:linear-gradient(135deg,#7c3aed,#4f46e5);display:flex;align-items:center;justify-content:center;font-size:36px;font-weight:700;color:#fff;">{letter}</div>'

    shared_html = "".join(f"""
    <a href="/student/class/{sc['id']}/feed" style="display:flex;align-items:center;gap:10px;padding:10px 16px;
        background:#1a2635;border-radius:12px;text-decoration:none;transition:background 0.15s;"
        onmouseover="this.style.background='#1e3048'" onmouseout="this.style.background='#1a2635'">
        <span style="font-size:20px;">📚</span>
        <div>
            <div style="font-weight:600;color:#c8d8e8;font-size:14px;">{sc['class_name']}</div>
            <div style="font-size:12px;color:#4a6a8a;">{sc.get('subject_name') or ''}</div>
        </div>
        <svg style="margin-left:auto;color:#4a6a8a;" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"/></svg>
    </a>""" for sc in shared_classes)

    # Pick any shared class for DM context (use the first one)
    first_class_id = shared_classes[0]["id"] if shared_classes else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{esc(teacher['teacher_name'])}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0e1621;color:#e4e7eb;min-height:100vh;}}
</style>
</head>
<body>
<div style="max-width:480px;margin:0 auto;padding:20px 16px;">
    <a href="javascript:history.back()" style="display:inline-flex;align-items:center;gap:6px;color:#5b9bd9;font-size:14px;font-weight:600;text-decoration:none;margin-bottom:20px;">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="15 18 9 12 15 6"/></svg>
        Back
    </a>

    <!-- Profile card -->
    <div style="background:#17212b;border-radius:20px;overflow:hidden;margin-bottom:16px;">
        <div style="height:80px;background:linear-gradient(135deg,#2d1b69,#4c1d95);"></div>
        <div style="display:flex;justify-content:center;margin-top:-45px;margin-bottom:12px;">
            <div style="border:4px solid #17212b;border-radius:50%;">{av_html}</div>
        </div>
        <div style="text-align:center;padding:0 20px 24px;">
            <h2 style="font-size:20px;font-weight:700;color:#e4e7eb;">{esc(teacher['teacher_name'])}</h2>
            <div style="display:inline-flex;align-items:center;gap:6px;background:#2d1b69;color:#a78bfa;font-size:12px;font-weight:700;padding:4px 14px;border-radius:20px;margin-top:8px;border:1px solid #4c1d95;">
                👨‍🏫 Instructor
            </div>
            <div style="margin-top:16px;">
                <a href="/student/teacher-dm/{teacher_db_id}?class_id={first_class_id}"
                   style="display:inline-flex;align-items:center;gap:8px;background:#7c3aed;color:#fff;
                   font-size:14px;font-weight:700;padding:10px 28px;border-radius:50px;text-decoration:none;
                   box-shadow:0 4px 14px rgba(124,58,237,0.4);">
                    ✉️ Message
                </a>
            </div>
        </div>
    </div>

    <!-- Classes taught to this student -->
    <div style="background:#17212b;border-radius:16px;overflow:hidden;margin-bottom:16px;">
        <div style="padding:14px 16px;border-bottom:1px solid #0f1923;font-size:12px;font-weight:700;color:#a78bfa;text-transform:uppercase;letter-spacing:0.06em;">
            Your Classes with {esc(teacher['teacher_name'])} ({len(shared_classes)})
        </div>
        <div style="padding:10px;display:flex;flex-direction:column;gap:6px;">
            {shared_html if shared_html else '<p style="text-align:center;color:#3a4a5a;padding:20px;font-size:13px;">No shared classes</p>'}
        </div>
    </div>
</div>
</body>
</html>"""
    return html


# ── STUDENT: DM page with teacher ──────────────────────────────────────────
@app.route("/student/teacher-dm/<int:teacher_db_id>")
def student_teacher_dm_page(teacher_db_id):
    protect = student_required()
    if protect:
        return protect

    _ensure_teacher_msg_table()

    student_db_id = get_logged_student_db_id()
    school_id = get_current_school_id()
    student_ctx = get_student_row_by_db_id(student_db_id)
    if not student_ctx:
        return redirect("/student-logout")

    # resolve class_id (from query param or first shared class)
    class_id = request.args.get("class_id", type=int) or 0
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM teachers WHERE id=%s AND school_id=%s", (teacher_db_id, school_id))
    teacher = cur.fetchone()
    if not teacher:
        conn.close()
        return "<script>alert('Teacher not found.');window.location.href='/student/classes';</script>"

    if not class_id:
        cur.execute("""
            SELECT c.id FROM classes c
            INNER JOIN student_classes sc ON sc.class_id_fk = c.id
            WHERE sc.student_id_fk=%s AND c.teacher_id=%s AND c.school_id=%s
            ORDER BY c.id LIMIT 1
        """, (student_db_id, teacher_db_id, school_id))
        row = cur.fetchone()
        class_id = row["id"] if row else 0

    # Mark teacher's messages as read
    cur.execute("""
        UPDATE teacher_student_messages SET is_read_by_student=TRUE
        WHERE student_db_id=%s AND teacher_db_id=%s AND school_id=%s AND sender_type='teacher'
    """, (student_db_id, teacher_db_id, school_id))
    conn.commit()

    # Fetch conversation
    cur.execute("""
        SELECT id, sender_type, sender_name, sender_image, message, created_at
        FROM teacher_student_messages
        WHERE student_db_id=%s AND teacher_db_id=%s AND school_id=%s
        ORDER BY created_at ASC LIMIT 100
    """, (student_db_id, teacher_db_id, school_id))
    msgs = cur.fetchall()
    conn.close()

    def _ts(dt):
        if hasattr(dt, "strftime"):
            from datetime import date as _date
            if dt.date() == __import__("datetime").date.today():
                return dt.strftime("%I:%M %p")
            return dt.strftime("%b %d, %I:%M %p")
        return str(dt)[:16]

    msgs_html = ""
    for m in msgs:
        is_me = m["sender_type"] == "student"
        ts = _ts(m["created_at"])
        txt = (m["message"] or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        mid = m["id"]
        if is_me:
            msgs_html += f"""<div class="tg-msg-row tg-mine" id="tsm-{mid}">
    <div class="tg-bubble tg-bubble-mine">
        <div class="tg-bubble-text">{txt}</div>
        <div class="tg-bubble-ts">{ts} ✓✓</div>
    </div></div>"""
        else:
            t_letter = (teacher["teacher_name"] or "T")[0].upper()
            t_photo = supabase_public_url(teacher.get("photo_path") or "") if teacher.get("photo_path") else ""
            av = f'<img src="{t_photo}" style="width:34px;height:34px;border-radius:50%;object-fit:cover;">' if t_photo else f'<div style="width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#7c3aed,#4f46e5);display:flex;align-items:center;justify-content:center;font-weight:700;color:#fff;font-size:14px;">{t_letter}</div>'
            msgs_html += f"""<div class="tg-msg-row tg-theirs" id="tsm-{mid}">
    <div class="tg-av-wrap">{av}</div>
    <div>
        <div class="tg-sender-name" style="color:#a78bfa;">{esc(teacher['teacher_name'])} <span style="font-size:10px;background:#4c1d95;color:#c4b5fd;padding:1px 6px;border-radius:6px;">Teacher</span></div>
        <div class="tg-bubble tg-bubble-theirs" style="background:#2d1b69;border:1px solid #4c1d95;">
            <div class="tg-bubble-text">{txt}</div>
            <div class="tg-bubble-ts">{ts}</div>
        </div>
    </div></div>"""

    last_id = msgs[-1]["id"] if msgs else 0
    t_photo_url = supabase_public_url(teacher.get("photo_path") or "") if teacher.get("photo_path") else ""
    t_letter_big = (teacher["teacher_name"] or "T")[0].upper()
    topbar_av = f'<img src="{t_photo_url}" style="width:40px;height:40px;border-radius:50%;object-fit:cover;">' if t_photo_url else f'<div style="width:40px;height:40px;border-radius:50%;background:linear-gradient(135deg,#7c3aed,#4f46e5);display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:700;color:#fff;">{t_letter_big}</div>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<title>Chat with {esc(teacher['teacher_name'])}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
html{{height:100%;}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0e1621;height:100vh;height:100dvh;display:flex;flex-direction:column;overflow:hidden;color:#e4e7eb;}}
.tg-topbar{{height:56px;background:#17212b;border-bottom:1px solid #0f1923;display:flex;align-items:center;padding:0 12px;gap:10px;flex-shrink:0;box-shadow:0 1px 8px rgba(0,0,0,0.3);}}
.tg-topbar-info{{flex:1;min-width:0;}}
.tg-topbar-title{{font-weight:700;font-size:15px;color:#e4e7eb;}}
.tg-topbar-sub{{font-size:12px;color:#5a8ebd;}}
.tg-back{{color:#5b9bd9;font-size:14px;font-weight:600;text-decoration:none;padding:6px;}}
.tg-msgs{{flex:1;overflow-y:auto;padding:12px 12px 8px;display:flex;flex-direction:column;gap:6px;background:linear-gradient(180deg,#0e1621 0%,#111d2c 100%);}}
.tg-msg-row{{display:flex;align-items:flex-end;gap:6px;}}
.tg-mine{{justify-content:flex-end;}}
.tg-theirs{{justify-content:flex-start;}}
.tg-av-wrap{{flex-shrink:0;width:34px;display:flex;align-items:flex-end;}}
.tg-bubble{{max-width:min(360px,72vw);padding:8px 12px 4px;border-radius:16px;word-break:break-word;position:relative;}}
.tg-bubble-mine{{background:#2b5278;border-radius:16px 16px 4px 16px;box-shadow:0 1px 4px rgba(0,0,0,0.25);}}
.tg-bubble-theirs{{border-radius:4px 16px 16px 16px;background:#1e2d3d;box-shadow:0 1px 4px rgba(0,0,0,0.25);}}
.tg-bubble-text{{font-size:14px;color:#e4e7eb;line-height:1.5;}}
.tg-bubble-ts{{font-size:10px;color:#5a7a9a;text-align:right;margin-top:3px;}}
.tg-sender-name{{font-size:12px;font-weight:700;margin-bottom:3px;}}
.tg-input-bar{{background:#17212b;border-top:1px solid #0f1923;padding:10px 12px;display:flex;align-items:center;gap:8px;flex-shrink:0;padding-bottom:max(10px,env(safe-area-inset-bottom));}}
.tg-input{{flex:1;background:#242f3d;border:none;border-radius:20px;padding:10px 16px;color:#e4e7eb;font-size:14px;outline:none;resize:none;max-height:100px;line-height:1.4;font-family:inherit;}}
.tg-input::placeholder{{color:#5a6a7a;}}
.tg-send-btn{{width:40px;height:40px;border-radius:50%;background:#2b5278;border:none;color:#5b9bd9;font-size:18px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:background 0.15s;flex-shrink:0;}}
.tg-send-btn:hover{{background:#3a6a96;}}
::-webkit-scrollbar{{width:4px;}}::-webkit-scrollbar-thumb{{background:#2b3c4e;border-radius:4px;}}
</style>
</head>
<body>
<div class="tg-topbar">
    <a href="/student/teacher-profile/{teacher_db_id}" class="tg-back">←</a>
    <a href="/student/teacher-profile/{teacher_db_id}" style="display:contents;text-decoration:none;">
        {topbar_av}
        <div class="tg-topbar-info">
            <div class="tg-topbar-title">{esc(teacher['teacher_name'])}</div>
            <div class="tg-topbar-sub">👨‍🏫 Instructor</div>
        </div>
    </a>
</div>
<div class="tg-msgs" id="msgList">{msgs_html}</div>
<div class="tg-input-bar">
    <textarea class="tg-input" id="msgInput" placeholder="Message {esc(teacher['teacher_name'])}…" rows="1"
        onkeydown="if(event.key==='Enter'&&!event.shiftKey){{event.preventDefault();sendMsg();}}"></textarea>
    <button class="tg-send-btn" onclick="sendMsg()">➤</button>
</div>
<script>
// ── GLOBAL CSRF PROTECTION FOR AJAX ──
(function() {{
    function getCookie(name) {{
        const m = document.cookie.match('(^|;)\\s*' + name + '\\s*=\\s*([^;]+)');
        return m ? decodeURIComponent(m.pop()) : '';
    }}
    const _origFetch = window.fetch;
    window.fetch = function(input, init) {{
        init = init || {{}};
        const method = (init.method || 'GET').toUpperCase();
        if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(method)) {{
            init.headers = Object.assign({{}}, init.headers || {{}});
            if (!init.headers['X-CSRF-Token'] && !init.headers['x-csrf-token']) {{
                init.headers['X-CSRF-Token'] = getCookie('csrf_token');
            }}
        }}
        return _origFetch(input, init);
    }};
}})();
const msgList = document.getElementById('msgList');
msgList.scrollTop = msgList.scrollHeight;
let lastId = {last_id};
const STUDENT_DB_ID = {student_db_id};
const TEACHER_DB_ID = {teacher_db_id};

async function sendMsg() {{
    const input = document.getElementById('msgInput');
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    input.style.height = '';
    const res = await fetch('/student/teacher-dm/{teacher_db_id}/send', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{message: text, class_id: {class_id}}})
    }});
    const data = await res.json();
    if (data.ok && data.msg) appendMsg(data.msg, true);
}}

function appendMsg(m, isMe) {{
    const div = document.createElement('div');
    div.id = 'tsm-' + m.id;
    div.className = 'tg-msg-row ' + (isMe ? 'tg-mine' : 'tg-theirs');
    const txt = (m.message || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    const ts = m.ts || '';
    if (isMe) {{
        div.innerHTML = `<div class="tg-bubble tg-bubble-mine"><div class="tg-bubble-text">${{txt}}</div><div class="tg-bubble-ts">${{ts}} ✓✓</div></div>`;
    }} else {{
        const avHtml = '{t_photo_url}' ? `<img src="{t_photo_url}" style="width:34px;height:34px;border-radius:50%;object-fit:cover;">` : `<div style="width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#7c3aed,#4f46e5);display:flex;align-items:center;justify-content:center;font-weight:700;color:#fff;font-size:14px;">{t_letter_big}</div>`;
        div.innerHTML = `<div class="tg-av-wrap">${{avHtml}}</div><div><div class="tg-sender-name" style="color:#a78bfa;">{esc(teacher['teacher_name'])} <span style="font-size:10px;background:#4c1d95;color:#c4b5fd;padding:1px 6px;border-radius:6px;">Teacher</span></div><div class="tg-bubble tg-bubble-theirs" style="background:#2d1b69;border:1px solid #4c1d95;"><div class="tg-bubble-text">${{txt}}</div><div class="tg-bubble-ts">${{ts}}</div></div></div>`;
    }}
    msgList.appendChild(div);
    if (m.id > lastId) lastId = m.id;
    msgList.scrollTop = msgList.scrollHeight;
}}

async function pollMsgs() {{
    try {{
        const res = await fetch(`/student/teacher-dm/{teacher_db_id}/poll?since=${{lastId}}`);
        const data = await res.json();
        if (data.msgs) data.msgs.forEach(m => {{
            if (!document.getElementById('tsm-' + m.id)) appendMsg(m, m.sender_type === 'student');
        }});
    }} catch(e) {{}}
    setTimeout(pollMsgs, 3000);
}}
pollMsgs();

// Auto-resize textarea
document.getElementById('msgInput').addEventListener('input', function() {{
    this.style.height = '';
    this.style.height = Math.min(this.scrollHeight, 100) + 'px';
}});
</script>
</body>
</html>"""
    return html


@app.route("/student/teacher-dm/<int:teacher_db_id>/send", methods=["POST"])
def student_teacher_dm_send(teacher_db_id):
    protect = student_required()
    if protect:
        return jsonify({"ok": False, "error": "Not logged in."})

    _ensure_teacher_msg_table()
    student_db_id = get_logged_student_db_id()
    school_id = get_current_school_id()
    student_row = get_student_row_by_db_id(student_db_id)

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    class_id = data.get("class_id") or 0
    if not message:
        return jsonify({"ok": False, "error": "Message cannot be empty."})

    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT id FROM teachers WHERE id=%s AND school_id=%s", (teacher_db_id, school_id))
        if not cur.fetchone():
            conn.close()
            return jsonify({"ok": False, "error": "Teacher not found."})

        cur.execute("""
            INSERT INTO teacher_student_messages
                (school_id, class_id, sender_type, student_db_id, teacher_db_id, sender_name, sender_image, message, is_read_by_teacher, is_read_by_student)
            VALUES (%s, %s, 'student', %s, %s, %s, %s, %s, FALSE, TRUE)
            RETURNING id, created_at
        """, (school_id, class_id, student_db_id, teacher_db_id,
              student_row["full_name"], student_row.get("image_file") or "",
              message))
        row = cur.fetchone()
        conn.commit()
        conn.close()
        ts = row["created_at"].strftime("%I:%M %p") if hasattr(row["created_at"], "strftime") else str(row["created_at"])[:16]
        return jsonify({"ok": True, "msg": {"id": row["id"], "sender_type": "student", "message": message, "ts": ts}})
    except Exception as e:
        print("chat send error:", e, flush=True)
        return jsonify({"ok": False, "error": "Could not send message. Please try again."})


@app.route("/student/teacher-dm/<int:teacher_db_id>/poll")
def student_teacher_dm_poll(teacher_db_id):
    protect = student_required()
    if protect:
        return jsonify({"msgs": []})

    _ensure_teacher_msg_table()
    student_db_id = get_logged_student_db_id()
    school_id = get_current_school_id()
    since = request.args.get("since", 0, type=int)

    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, sender_type, sender_name, message, created_at
            FROM teacher_student_messages
            WHERE student_db_id=%s AND teacher_db_id=%s AND school_id=%s AND id > %s
            ORDER BY id ASC LIMIT 30
        """, (student_db_id, teacher_db_id, school_id, since))
        rows = cur.fetchall()
        # Mark teacher messages as read
        if any(r["sender_type"] == "teacher" for r in rows):
            cur.execute("""
                UPDATE teacher_student_messages SET is_read_by_student=TRUE
                WHERE student_db_id=%s AND teacher_db_id=%s AND school_id=%s AND sender_type='teacher' AND id > %s
            """, (student_db_id, teacher_db_id, school_id, since))
            conn.commit()
        conn.close()

        def _ts(dt):
            if hasattr(dt, "strftime"):
                from datetime import date as _d
                if dt.date() == __import__("datetime").date.today():
                    return dt.strftime("%I:%M %p")
                return dt.strftime("%b %d, %I:%M %p")
            return str(dt)[:16]

        return jsonify({"msgs": [{"id": r["id"], "sender_type": r["sender_type"],
                                   "message": r["message"], "ts": _ts(r["created_at"])} for r in rows]})
    except Exception as e:
        print("chat poll error:", e, flush=True)
        return jsonify({"msgs": [], "error": "Could not load messages."})


# ── TEACHER: inbox (conversations with students) ──────────────────────────
@app.route("/teacher/inbox")
def teacher_inbox():
    protect = teacher_required()
    if protect:
        return protect

    _ensure_teacher_msg_table()
    teacher_id = get_logged_teacher_id()
    school_id = get_current_school_id()
    teacher_name = session.get("teacher_name", "Teacher")

    conn = get_db()
    cur = conn.cursor()
    # One row per student conversation: latest message + unread count
    cur.execute("""
        SELECT DISTINCT ON (student_db_id)
            student_db_id,
            sender_name,
            sender_image,
            message,
            created_at,
            sender_type,
            SUM(CASE WHEN is_read_by_teacher=FALSE AND sender_type='student' THEN 1 ELSE 0 END)
                OVER (PARTITION BY student_db_id) AS unread_cnt
        FROM teacher_student_messages
        WHERE teacher_db_id=%s AND school_id=%s
        ORDER BY student_db_id, created_at DESC
    """, (teacher_id, school_id))
    conversations = cur.fetchall()
    conn.close()

    if not conversations:
        convs_html = """
        <div style="text-align:center;padding:60px 20px;color:#4a6a8a;">
            <div style="font-size:56px;margin-bottom:12px;">📭</div>
            <div style="font-weight:600;font-size:16px;">No messages yet</div>
            <div style="font-size:13px;margin-top:6px;">Students can message you from the class chat feed</div>
        </div>"""
    else:
        convs_html = ""
        for conv in conversations:
            sid = conv["student_db_id"]
            student_row = get_student_row_by_db_id(sid)
            name = student_row["full_name"] if student_row else conv["sender_name"]
            img = supabase_public_url(student_row.get("image_file") or "") if student_row else ""
            letter = (name or "?")[0].upper()
            colors = ["#2196F3","#E91E63","#9C27B0","#FF9800","#4CAF50","#00BCD4","#F44336","#3F51B5"]
            color = colors[sum(ord(c) for c in name) % len(colors)]
            av = f'<img src="{img}" style="width:48px;height:48px;border-radius:50%;object-fit:cover;flex-shrink:0;">' if img else f'<div style="width:48px;height:48px;border-radius:50%;background:{color};display:flex;align-items:center;justify-content:center;font-size:20px;font-weight:700;color:#fff;flex-shrink:0;">{letter}</div>'
            preview = (conv["message"] or "")[:50]
            if conv["sender_type"] == "teacher":
                preview = "You: " + preview
            unread = int(conv["unread_cnt"] or 0)
            unread_badge = f'<span style="background:#ef4444;color:#fff;font-size:11px;font-weight:700;padding:2px 7px;border-radius:20px;flex-shrink:0;">{unread}</span>' if unread else ""
            ts_raw = conv["created_at"]
            if hasattr(ts_raw, "strftime"):
                from datetime import date as _d
                ts_str = ts_raw.strftime("%I:%M %p") if ts_raw.date() == __import__("datetime").date.today() else ts_raw.strftime("%b %d")
            else:
                ts_str = str(ts_raw)[:10]
            convs_html += f"""
            <a href="/teacher/inbox/{sid}" style="display:flex;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid #1b2633;text-decoration:none;transition:background 0.12s;" onmouseover="this.style.background='#1e2d3d'" onmouseout="this.style.background=''">
                {av}
                <div style="flex:1;min-width:0;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-weight:600;color:#e4e7eb;font-size:14px;">{name}</span>
                        <span style="font-size:11px;color:#5a6a7a;flex-shrink:0;">{ts_str}</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:6px;margin-top:2px;">
                        <span style="font-size:13px;color:#7a8a9a;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1;">{preview}</span>
                        {unread_badge}
                    </div>
                </div>
            </a>"""

    body = f"""
    <div style="max-width:600px;">
        <h1 style="font-size:22px;font-weight:800;color:#e4e7eb;margin-bottom:4px;">💬 Student Messages</h1>
        <p style="font-size:13px;color:#5a8ebd;margin-bottom:20px;">Private conversations with students across your classes.</p>
        <div style="background:#17212b;border-radius:16px;overflow:hidden;border:1px solid #1b2633;">
            {convs_html}
        </div>
    </div>
    """
    return page_wrapper("Inbox", body, is_teacher=True, teacher_name=teacher_name)


# ── TEACHER: individual DM thread with a student ──────────────────────────
@app.route("/teacher/inbox/<int:student_db_id>")
def teacher_inbox_thread(student_db_id):
    protect = teacher_required()
    if protect:
        return protect

    _ensure_teacher_msg_table()
    teacher_id = get_logged_teacher_id()
    school_id = get_current_school_id()
    teacher_name_str = session.get("teacher_name", "Teacher")
    teacher_photo = session.get("teacher_photo", "")

    student_row = get_student_row_by_db_id(student_db_id)
    if not student_row or student_row.get("school_id", school_id) != school_id:
        return "<script>alert('Student not found.');window.location.href='/teacher/inbox';</script>"

    conn = get_db()
    cur = conn.cursor()
    # Get a shared class
    cur.execute("""
        SELECT c.id FROM classes c
        INNER JOIN student_classes sc ON sc.class_id_fk = c.id
        WHERE sc.student_id_fk=%s AND c.teacher_id=%s AND c.school_id=%s
        ORDER BY c.id LIMIT 1
    """, (student_db_id, teacher_id, school_id))
    row = cur.fetchone()
    class_id = row["id"] if row else 0

    # Mark student messages as read
    cur.execute("""
        UPDATE teacher_student_messages SET is_read_by_teacher=TRUE
        WHERE student_db_id=%s AND teacher_db_id=%s AND school_id=%s AND sender_type='student'
    """, (student_db_id, teacher_id, school_id))
    conn.commit()

    cur.execute("""
        SELECT id, sender_type, sender_name, message, created_at
        FROM teacher_student_messages
        WHERE student_db_id=%s AND teacher_db_id=%s AND school_id=%s
        ORDER BY created_at ASC LIMIT 100
    """, (student_db_id, teacher_id, school_id))
    msgs = cur.fetchall()
    conn.close()

    def _ts(dt):
        if hasattr(dt, "strftime"):
            from datetime import date as _d
            if dt.date() == __import__("datetime").date.today():
                return dt.strftime("%I:%M %p")
            return dt.strftime("%b %d, %I:%M %p")
        return str(dt)[:16]

    msgs_html = ""
    for m in msgs:
        is_me = m["sender_type"] == "teacher"
        ts = _ts(m["created_at"])
        txt = (m["message"] or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        mid = m["id"]
        if is_me:
            msgs_html += f"""<div class="tg-msg-row tg-mine" id="tsm-{mid}">
<div class="tg-bubble tg-bubble-mine"><div class="tg-bubble-text">{txt}</div><div class="tg-bubble-ts">{ts} ✓✓</div></div></div>"""
        else:
            s_img = supabase_public_url(student_row.get("image_file") or "") if student_row.get("image_file") else ""
            s_letter = (student_row["full_name"] or "?")[0].upper()
            colors2 = ["#2196F3","#E91E63","#9C27B0","#FF9800","#4CAF50","#00BCD4","#F44336","#3F51B5"]
            s_color = colors2[sum(ord(c) for c in student_row["full_name"]) % len(colors2)]
            av = f'<img src="{s_img}" style="width:34px;height:34px;border-radius:50%;object-fit:cover;">' if s_img else f'<div style="width:34px;height:34px;border-radius:50%;background:{s_color};display:flex;align-items:center;justify-content:center;font-weight:700;color:#fff;font-size:14px;">{s_letter}</div>'
            msgs_html += f"""<div class="tg-msg-row tg-theirs" id="tsm-{mid}">
<div class="tg-av-wrap">{av}</div>
<div>
    <div class="tg-sender-name" style="color:#5b9bd9;">{student_row['full_name']}</div>
    <div class="tg-bubble tg-bubble-theirs"><div class="tg-bubble-text">{txt}</div><div class="tg-bubble-ts">{ts}</div></div>
</div></div>"""

    last_id = msgs[-1]["id"] if msgs else 0
    t_photo_url = supabase_public_url(teacher_photo) if teacher_photo else ""
    t_letter_big = (teacher_name_str or "T")[0].upper()
    s_name = student_row["full_name"]
    s_name_js = s_name.replace("'", "\\'")
    s_img_url = supabase_public_url(student_row.get("image_file") or "") if student_row.get("image_file") else ""
    s_letter_big = (s_name or "?")[0].upper()
    topbar_av = f'<img src="{s_img_url}" style="width:40px;height:40px;border-radius:50%;object-fit:cover;">' if s_img_url else f'<div style="width:40px;height:40px;border-radius:50%;background:#2b5278;display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:700;color:#fff;">{s_letter_big}</div>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<title>Chat with {s_name}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
html{{height:100%;}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0e1621;height:100vh;height:100dvh;display:flex;flex-direction:column;overflow:hidden;color:#e4e7eb;}}
.tg-topbar{{height:56px;background:#17212b;border-bottom:1px solid #0f1923;display:flex;align-items:center;padding:0 12px;gap:10px;flex-shrink:0;box-shadow:0 1px 8px rgba(0,0,0,0.3);}}
.tg-topbar-info{{flex:1;min-width:0;}}
.tg-topbar-title{{font-weight:700;font-size:15px;color:#e4e7eb;}}
.tg-topbar-sub{{font-size:12px;color:#5a8ebd;}}
.tg-back{{color:#5b9bd9;font-size:14px;font-weight:600;text-decoration:none;padding:6px;}}
.tg-msgs{{flex:1;overflow-y:auto;padding:12px 12px 8px;display:flex;flex-direction:column;gap:6px;background:linear-gradient(180deg,#0e1621 0%,#111d2c 100%);}}
.tg-msg-row{{display:flex;align-items:flex-end;gap:6px;}}
.tg-mine{{justify-content:flex-end;}}
.tg-theirs{{justify-content:flex-start;}}
.tg-av-wrap{{flex-shrink:0;width:34px;display:flex;align-items:flex-end;}}
.tg-bubble{{max-width:min(360px,72vw);padding:8px 12px 4px;border-radius:16px;word-break:break-word;}}
.tg-bubble-mine{{background:#7c3aed;border-radius:16px 16px 4px 16px;box-shadow:0 1px 4px rgba(0,0,0,0.25);}}
.tg-bubble-theirs{{border-radius:4px 16px 16px 16px;background:#1e2d3d;box-shadow:0 1px 4px rgba(0,0,0,0.25);}}
.tg-bubble-text{{font-size:14px;color:#e4e7eb;line-height:1.5;}}
.tg-bubble-ts{{font-size:10px;color:#a0a8b8;text-align:right;margin-top:3px;}}
.tg-sender-name{{font-size:12px;font-weight:700;margin-bottom:3px;}}
.tg-input-bar{{background:#17212b;border-top:1px solid #0f1923;padding:10px 12px;display:flex;align-items:center;gap:8px;flex-shrink:0;padding-bottom:max(10px,env(safe-area-inset-bottom));}}
.tg-input{{flex:1;background:#242f3d;border:none;border-radius:20px;padding:10px 16px;color:#e4e7eb;font-size:14px;outline:none;resize:none;max-height:100px;line-height:1.4;font-family:inherit;}}
.tg-input::placeholder{{color:#5a6a7a;}}
.tg-send-btn{{width:40px;height:40px;border-radius:50%;background:#7c3aed;border:none;color:#fff;font-size:18px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:background 0.15s;flex-shrink:0;}}
.tg-send-btn:hover{{background:#6d28d9;}}
::-webkit-scrollbar{{width:4px;}}::-webkit-scrollbar-thumb{{background:#2b3c4e;border-radius:4px;}}
</style>
</head>
<body>
<div class="tg-topbar">
    <a href="/teacher/inbox" class="tg-back">←</a>
    {topbar_av}
    <div class="tg-topbar-info">
        <div class="tg-topbar-title">{s_name}</div>
        <div class="tg-topbar-sub">Student · ID: {student_row['student_id']}</div>
    </div>
</div>
<div class="tg-msgs" id="msgList">{msgs_html}</div>
<div class="tg-input-bar">
    <textarea class="tg-input" id="msgInput" placeholder="Reply to {s_name}…" rows="1"
        onkeydown="if(event.key==='Enter'&&!event.shiftKey){{event.preventDefault();sendMsg();}}"></textarea>
    <button class="tg-send-btn" onclick="sendMsg()">➤</button>
</div>
<script>
// ── GLOBAL CSRF PROTECTION FOR AJAX ──
(function() {{
    function getCookie(name) {{
        const m = document.cookie.match('(^|;)\\s*' + name + '\\s*=\\s*([^;]+)');
        return m ? decodeURIComponent(m.pop()) : '';
    }}
    const _origFetch = window.fetch;
    window.fetch = function(input, init) {{
        init = init || {{}};
        const method = (init.method || 'GET').toUpperCase();
        if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(method)) {{
            init.headers = Object.assign({{}}, init.headers || {{}});
            if (!init.headers['X-CSRF-Token'] && !init.headers['x-csrf-token']) {{
                init.headers['X-CSRF-Token'] = getCookie('csrf_token');
            }}
        }}
        return _origFetch(input, init);
    }};
}})();
const msgList = document.getElementById('msgList');
msgList.scrollTop = msgList.scrollHeight;
let lastId = {last_id};
const T_PHOTO = '{t_photo_url}';
const T_LETTER = '{t_letter_big}';
const S_IMG = '{s_img_url}';
const S_LETTER = '{s_letter_big}';
const S_NAME = '{s_name_js}';

async function sendMsg() {{
    const input = document.getElementById('msgInput');
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    input.style.height = '';
    const res = await fetch('/teacher/inbox/{student_db_id}/send', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{message: text, class_id: {class_id}}})
    }});
    const data = await res.json();
    if (data.ok && data.msg) appendMsg(data.msg, true);
}}

function appendMsg(m, isMe) {{
    const div = document.createElement('div');
    div.id = 'tsm-' + m.id;
    div.className = 'tg-msg-row ' + (isMe ? 'tg-mine' : 'tg-theirs');
    const txt = (m.message||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    const ts = m.ts || '';
    if (isMe) {{
        div.innerHTML = `<div class="tg-bubble tg-bubble-mine"><div class="tg-bubble-text">${{txt}}</div><div class="tg-bubble-ts">${{ts}} ✓✓</div></div>`;
    }} else {{
        const avHtml = S_IMG ? `<img src="${{S_IMG}}" style="width:34px;height:34px;border-radius:50%;object-fit:cover;">` : `<div style="width:34px;height:34px;border-radius:50%;background:#2b5278;display:flex;align-items:center;justify-content:center;font-weight:700;color:#fff;font-size:14px;">${{S_LETTER}}</div>`;
        div.innerHTML = `<div class="tg-av-wrap">${{avHtml}}</div><div><div class="tg-sender-name" style="color:#5b9bd9;">${{S_NAME}}</div><div class="tg-bubble tg-bubble-theirs"><div class="tg-bubble-text">${{txt}}</div><div class="tg-bubble-ts">${{ts}}</div></div></div>`;
    }}
    msgList.appendChild(div);
    if (m.id > lastId) lastId = m.id;
    msgList.scrollTop = msgList.scrollHeight;
}}

async function pollMsgs() {{
    try {{
        const res = await fetch(`/teacher/inbox/{student_db_id}/poll?since=${{lastId}}`);
        const data = await res.json();
        if (data.msgs) data.msgs.forEach(m => {{
            if (!document.getElementById('tsm-' + m.id)) appendMsg(m, m.sender_type === 'teacher');
        }});
    }} catch(e) {{}}
    setTimeout(pollMsgs, 3000);
}}
pollMsgs();

document.getElementById('msgInput').addEventListener('input', function() {{
    this.style.height = '';
    this.style.height = Math.min(this.scrollHeight, 100) + 'px';
}});
</script>
</body>
</html>"""
    return html


@app.route("/teacher/inbox/<int:student_db_id>/send", methods=["POST"])
def teacher_inbox_send(student_db_id):
    protect = teacher_required()
    if protect:
        return jsonify({"ok": False, "error": "Not logged in."})

    _ensure_teacher_msg_table()
    teacher_id = get_logged_teacher_id()
    school_id = get_current_school_id()
    teacher_name_str = session.get("teacher_name", "Teacher")
    teacher_photo = session.get("teacher_photo", "")

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    class_id = data.get("class_id") or 0
    if not message:
        return jsonify({"ok": False, "error": "Message cannot be empty."})

    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO teacher_student_messages
                (school_id, class_id, sender_type, student_db_id, teacher_db_id, sender_name, sender_image, message, is_read_by_teacher, is_read_by_student)
            VALUES (%s, %s, 'teacher', %s, %s, %s, %s, %s, TRUE, FALSE)
            RETURNING id, created_at
        """, (school_id, class_id, student_db_id, teacher_id, teacher_name_str, teacher_photo or "", message))
        row = cur.fetchone()
        conn.commit()
        conn.close()
        ts = row["created_at"].strftime("%I:%M %p") if hasattr(row["created_at"], "strftime") else str(row["created_at"])[:16]
        return jsonify({"ok": True, "msg": {"id": row["id"], "sender_type": "teacher", "message": message, "ts": ts}})
    except Exception as e:
        print("chat send error:", e, flush=True)
        return jsonify({"ok": False, "error": "Could not send message. Please try again."})


@app.route("/teacher/inbox/<int:student_db_id>/poll")
def teacher_inbox_poll(student_db_id):
    protect = teacher_required()
    if protect:
        return jsonify({"msgs": []})

    _ensure_teacher_msg_table()
    teacher_id = get_logged_teacher_id()
    school_id = get_current_school_id()
    since = request.args.get("since", 0, type=int)

    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, sender_type, sender_name, message, created_at
            FROM teacher_student_messages
            WHERE student_db_id=%s AND teacher_db_id=%s AND school_id=%s AND id > %s
            ORDER BY id ASC LIMIT 30
        """, (student_db_id, teacher_id, school_id, since))
        rows = cur.fetchall()
        if any(r["sender_type"] == "student" for r in rows):
            cur.execute("""
                UPDATE teacher_student_messages SET is_read_by_teacher=TRUE
                WHERE student_db_id=%s AND teacher_db_id=%s AND school_id=%s AND sender_type='student' AND id > %s
            """, (student_db_id, teacher_id, school_id, since))
            conn.commit()
        conn.close()

        def _ts(dt):
            if hasattr(dt, "strftime"):
                from datetime import date as _d
                if dt.date() == __import__("datetime").date.today():
                    return dt.strftime("%I:%M %p")
                return dt.strftime("%b %d, %I:%M %p")
            return str(dt)[:16]

        return jsonify({"msgs": [{"id": r["id"], "sender_type": r["sender_type"],
                                   "message": r["message"], "ts": _ts(r["created_at"])} for r in rows]})
    except Exception as e:
        print("chat poll error:", e, flush=True)
        return jsonify({"msgs": [], "error": "Could not load messages."})


# =========================================================
# STARTUP INITIALIZATION
# =========================================================
# Runs unconditionally at import time (not just under `if __name__ ==
# '__main__'`) because WSGI/serverless hosts — Vercel included — import this
# module as a Flask `app` object rather than executing it as a script, so the
# __main__ block below never runs there. Wrapped in try/except so a slow or
# briefly-unreachable database doesn't fail the whole cold start; routes that
# need the DB will surface their own errors, and load_known_faces() safely
# results in an empty cache if it can't run yet.
try:
    init_db()
    init_super_admin_table()
    _ensure_teacher_msg_table()
    load_known_faces()
except Exception as e:
    print("Startup initialization error (will not block the app from serving):", e, flush=True)

if __name__ == '__main__':
    # threaded=True lets Flask's dev server handle multiple students scanning
    # the QR at the same time concurrently instead of queueing requests one
    # at a time. Without this, a class of students scanning within the same
    # few seconds get serialized — each one waits for everyone ahead of them,
    # which is what produces the "still working, this is taking longer than
    # usual" message and eventual timeouts/failed check-ins.
    # Note: on Vercel this block never runs — Vercel imports `app` above and
    # serves it directly; this is only used for local dev / non-serverless
    # hosts like Render that execute `python app.py`.
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
