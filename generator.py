# hunyuan_t2i_turbo_generator.py
# Shim module so Modly can import a top-level module path without package __init__ files.
# It dynamically loads the real generator module file and exposes `Generator` at module scope.

import importlib.util
import os
import sys
from types import ModuleType

_SHIM_TARGET = os.path.join(os.path.dirname(__file__), "models", "hunyuan_t2i_turbo", "generator.py")

def _load_target_module(path: str) -> ModuleType:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Generator file not found at {path}")
    spec = importlib.util.spec_from_file_location("models_hunyuan_t2i_turbo_generator_module", path)
    module = importlib.util.module_from_spec(spec)
    # Ensure the module can import relative dependencies if any by adding repo root to sys.path
    repo_root = os.path.dirname(__file__)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    loader = spec.loader
    if loader is None:
        raise ImportError("Cannot load generator module (no loader)")
    loader.exec_module(module)
    return module

# Load once at import time so Modly can import this shim and find Generator
_target_mod = _load_target_module(_SHIM_TARGET)

# Expose Generator at top-level for Modly to instantiate
if not hasattr(_target_mod, "Generator"):
    raise AttributeError("The target generator module does not define class 'Generator'")

Generator = getattr(_target_mod, "Generator")
