# Deploying this app to Vercel

## What changed from the original file

1. **Renamed** `app__81_.py` → `app.py` (Vercel looks for `app.py`/`index.py`/`server.py`/`main.py` at the project root, exposing a top-level `app` object — this file already had `app = Flask(__name__)`, so no framework code needed to change).
2. **Local image fallback dirs** (`IMAGE_DIR`/`TEACHER_IMAGE_DIR`) now default to `/tmp/...` instead of a relative path, and every local write is wrapped in `try/except`. Vercel's project filesystem is **read-only**; only `/tmp` is writable, and it's wiped on every cold start. This is safe because Supabase Storage is already the real source of truth for images (`supabase_upload`/`supabase_public_url`) — the local copy was always just a best-effort cache.
3. **Startup init moved out of `if __name__ == '__main__':`.** Vercel imports this module as a WSGI app; it never executes it as a script, so `init_db()`, `init_super_admin_table()`, `_ensure_teacher_msg_table()`, and `load_known_faces()` now also run unconditionally at import time (wrapped in `try/except` so a slow DB doesn't fail the cold start).
4. Added `requirements.txt`, `vercel.json`, `.vercelignore`, `.python-version`.

## Required environment variables

Set these in **Vercel → Project → Settings → Environment Variables**:

| Variable | Required | Notes |
|---|---|---|
| `DATABASE_URL` | Yes | Postgres connection string. **Use a pooled connection string** (e.g. Supabase's "Transaction pooler" URI on port 6543), not a direct connection — see caveat below. |
| `SECRET_KEY` | Yes | Flask session secret. |
| `SUPABASE_URL` | Recommended | Defaults to a hardcoded URL in the source; set explicitly. |
| `SUPABASE_SERVICE_KEY` | Yes | Needed for image upload/delete. |
| `APP_BASE_URL` | Recommended | Set to your `https://your-app.vercel.app` URL. |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM` | Optional | Only needed if you want registration verification emails to actually send. |

## Steps


1. `vercel link` (or import the repo in the Vercel dashboard).
2. Add the environment variables above.
3. If your Vercel project was created **before June 30, 2026**, also set `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` — this bundle (opencv + mediapipe) is likely to land between 250–500 MB, and that flag opts existing projects into the higher 500 MB–5 GB limit. New projects get this automatically.
4. `vercel deploy --prod`.

## Real caveats to know about (this is not a Render/Railway-style always-on box)

- **Database connections.** The app's `ThreadedConnectionPool` (`minconn=2, maxconn=30`) assumes one long-lived process. On Vercel, multiple concurrent serverless instances can each spin up their own pool, which can add up to more connections than a small Postgres instance allows under load. Point `DATABASE_URL` at a **connection pooler** (Supabase's pgbouncer endpoint, or similar) rather than a direct DB connection to avoid `too many connections` errors under concurrent traffic (e.g. a class scanning QR codes at once).
- **Cold starts will be slower than on Render.** `load_known_faces()` re-downloads every registered student's photo over HTTP and re-runs MediaPipe face embedding on each one, every time a fresh container starts. Fine for a class of dozens; will get slow with hundreds of students unless you add caching.
- **No real background threads.** The in-memory login rate limiter and face cache are per-instance — with multiple concurrent Vercel instances, rate limiting and the face cache are no longer globally consistent the way they are on a single Render dyno.
- **mediapipe is pinned to `0.10.18` on purpose** — this app uses the legacy `mediapipe.solutions.face_mesh` API, which newer mediapipe releases have removed. I verified this pin still works; don't let `pip` auto-upgrade it.

If this app sees real classroom traffic, Render/Railway/Fly.io (an always-on process, not per-request cold starts) will genuinely behave more predictably for the DB pool and face cache than serverless — but the above should get you a working Vercel deployment.
