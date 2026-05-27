import sys
from pathlib import Path
from typing import Callable, Optional
import threading

# venv injection
_here = Path(__file__).parent
for _v in (_here / "venv", _here / ".venv"):
    if _v.exists():
        _sp = _v / "Lib" / "site-packages"
        if _sp.exists() and str(_sp) not in sys.path:
            sys.path.insert(0, str(_sp))
        _lib = _v / "lib"
        if _lib.exists():
            for _d in _lib.glob("python3.*"):
                _s = _d / "site-packages"
                if _s.exists() and str(_s) not in sys.path:
                    sys.path.insert(0, str(_s))
        break

from services.generators.base import BaseGenerator


class HunyuanT2IGenerator(BaseGenerator):
    # required by the registry — actual generation runs through processor.js + t2i_worker.py
    MODEL_ID     = "hunyuan_t2i_turbo"
    DISPLAY_NAME = "Hunyuan T2I Turbo"
    VRAM_GB      = 6

    def load(self):
        pass

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        raise NotImplementedError("T2I runs through processor.js")
