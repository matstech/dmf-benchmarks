import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The executable Mem0 path is server-only. Keep the test process from importing
# Mem0 with its default telemetry store enabled before individual tests run.
os.environ.setdefault("MEM0_TELEMETRY", "false")
