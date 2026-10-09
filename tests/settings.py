"""Test settings: an isolated scratch data directory, inline job execution, the trained checkpoint reused."""
import os
import shutil
import tempfile
from pathlib import Path

_real = Path(__file__).resolve().parent.parent / "data" / "models"
_tmp = Path(tempfile.mkdtemp(prefix="cte-test-"))
os.environ["CTE_DATA_DIR"] = str(_tmp)
os.environ["CTE_RUN_INLINE"] = "1"
# Tests use SQLite unless CTE_TEST_DATABASE_URL points at a PostgreSQL server (Django then creates test_<name> there).
if os.environ.get("CTE_TEST_DATABASE_URL"):
    os.environ["CTE_DATABASE_URL"] = os.environ["CTE_TEST_DATABASE_URL"]
else:
    os.environ.pop("CTE_DATABASE_URL", None)
if (_real / "cte_cnn.pt").exists():
    shutil.copytree(_real, _tmp / "models")

from config.settings import *  # noqa: E402,F401,F403
