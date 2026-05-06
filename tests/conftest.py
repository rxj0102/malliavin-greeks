"""Pytest configuration — ensures both src/ and root packages are importable."""
import sys
import os

# Make src/malliavin_greeks importable as `malliavin_greeks`
_src = os.path.join(os.path.dirname(__file__), "..", "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
