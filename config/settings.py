"""Django settings for the Chronos Transit Engine."""
import os
import secrets
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from cte.config import DATA_DIR

BASE_DIR = Path(__file__).resolve().parent.parent

_key_file = DATA_DIR / ".django_secret_key"
if not _key_file.exists():
    _key_file.write_text(secrets.token_urlsafe(64))
    _key_file.chmod(0o600)
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY") or _key_file.read_text().strip()

DEBUG = os.environ.get("CTE_DEBUG", "1") == "1"
ALLOWED_HOSTS = os.environ.get("CTE_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver").split(",")

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "engine",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"

TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [BASE_DIR / "templates"],
    "APP_DIRS": True,
    "OPTIONS": {"context_processors": [
        "django.template.context_processors.request",
        "django.contrib.messages.context_processors.messages",
        "engine.context.shell",
    ]},
}]


def _database() -> dict:
    """PostgreSQL when CTE_DATABASE_URL is set (PRD §3); SQLite is the zero-setup local default."""
    url = os.environ.get("CTE_DATABASE_URL", "")
    if url.startswith(("postgres://", "postgresql://", "postgresql+psycopg://")):
        u = urlparse(url)
        socket_dir = parse_qs(u.query).get("host", [""])[0]  # postgresql://user@/db?host=/path/to/socket
        return {"ENGINE": "django.db.backends.postgresql", "NAME": u.path.lstrip("/"), "USER": unquote(u.username or ""),
                "PASSWORD": unquote(u.password or ""), "HOST": u.hostname or socket_dir or "localhost",
                "PORT": str(u.port or "") if socket_dir and not u.port else str(u.port or 5432)}
    return {"ENGINE": "django.db.backends.sqlite3", "NAME": DATA_DIR / "cte.sqlite3",
            "OPTIONS": {"timeout": 30, "init_command": "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;"},
            "TEST": {"NAME": DATA_DIR / "test_cte.sqlite3"}}


DATABASES = {"default": _database()}
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = False
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
MESSAGE_STORAGE = "django.contrib.messages.storage.session.SessionStorage"
DATA_UPLOAD_MAX_MEMORY_SIZE = 64 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 64 * 1024 * 1024

# Run pipeline jobs in the request thread instead of the worker pool (used by the test suite).
CTE_RUN_INLINE = os.environ.get("CTE_RUN_INLINE", "0") == "1"
