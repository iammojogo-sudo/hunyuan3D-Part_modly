#!/usr/bin/env python3
"""
Patch generator.py to use the Tencent-Hunyuan/HunyuanDiT repo and request subfolder "t2i".
Usage: run this from the extension root where generator.py lives:
    python patch_generator_for_t2i.py
This script edits generator.py in place (makes a .bak backup).
"""
import re
from pathlib import Path
import sys

GEN = Path("generator.py")
if not GEN.exists():
    print("generator.py not found in current directory.", file=sys.stderr)
    sys.exit(2)

text = GEN.read_text(encoding="utf-8")

# 1) Replace default repo assignment
text, n1 = re.subn(
    r'(_DEFAULT_HF_REPO\s*=\s*)["\'].*?["\']',
    r'\1"Tencent-Hunyuan/HunyuanDiT"',
    text,
    count=1,
)

# 2) Ensure snapshot_download calls include subfolder="t2i" when missing
# This will insert subfolder="t2i", after the opening parenthesis if the call does not already include subfolder=
def add_subfolder(match):
    call = match.group(0)
    # if subfolder already present, return unchanged
    if re.search(r'\bsubfolder\s*=', call):
        return call
    # insert subfolder="t2i", after the opening parenthesis (preserve whitespace)
    return call.replace("snapshot_download(", 'snapshot_download(subfolder="t2i", ', 1)

# Apply to multi-line calls as well
pattern = re.compile(r'snapshot_download\s*\(\s*', flags=re.MULTILINE)
text, n2 = pattern.subn(lambda m: add_subfolder(m), text)

# 3) As a safety: also handle cases where snapshot_download is called with named args on multiple lines.
# We'll look for 'snapshot_download(' up to the matching ')' and add subfolder if missing.
def ensure_subfolder_in_calls(s):
    out = []
    i = 0
    L = len(s)
    while True:
        m = re.search(r'snapshot_download\s*\(', s[i:])
        if not m:
            out.append(s[i:])
            break
        start = i + m.start()
        out.append(s[i:start])
        # find matching closing parenthesis (simple stack)
        j = start + m.end() - m.start()  # position after '('
        depth = 1
        while j < L and depth > 0:
            if s[j] == '(':
                depth += 1
            elif s[j] == ')':
                depth -= 1
            j += 1
        call = s[start:j]
        if 'subfolder' not in call:
            # insert subfolder="t2i", after the first '('
            call = call.replace('snapshot_download(', 'snapshot_download(subfolder="t2i", ', 1)
        out.append(call)
        i = j
    return "".join(out)

text2 = ensure_subfolder_in_calls(text)

# If ensure_subfolder_in_calls made changes, count them
changed = text2 != text
text = text2

# Backup and write
bak = GEN.with_suffix(GEN.suffix + ".bak")
GEN.rename(bak)
GEN.write_text(text, encoding="utf-8")

print(f"Patched generator.py: default repo set to Tencent-Hunyuan/HunyuanDiT.")
if changed:
    print("Inserted subfolder=\"t2i\" into snapshot_download calls where missing.")
else:
    print("No snapshot_download calls needed modification (subfolder already present).")
print(f"Backup saved as: {bak}")
