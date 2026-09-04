import os
import tempfile
from pathlib import Path


_TEST_CACHE = Path(tempfile.gettempdir()) / "adora-test-cache"
(_TEST_CACHE / "config").mkdir(parents=True, exist_ok=True)
(_TEST_CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
# Test runs must not write into the checkout.  Override rather than using
# setdefault: a malformed inherited MPLCONFIGDIR (for example, one with a
# leading space) is otherwise interpreted as a relative path by Matplotlib.
os.environ["XDG_CONFIG_HOME"] = str(_TEST_CACHE / "config")
os.environ["MPLCONFIGDIR"] = str(_TEST_CACHE / "matplotlib")
