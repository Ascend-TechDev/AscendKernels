import sys
import pathlib

# tests -> kda -> fla -> attention -> triton_ascend -> <repo root>
ROOT = pathlib.Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
