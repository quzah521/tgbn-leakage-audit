"""Create navis_audit.py from NAVIS's navis.py and navis_audit.patch.

Usage:
    python navis/apply_patch.py <NAVIS checkout>/NAVIS

Reads <dir>/navis.py (NAVIS commit 2b503ef), writes <dir>/navis_audit.py next to it and
copies ltr_loss.py there. Neither git nor `patch` is needed. The script stops if a hunk does
not match, which means that navis.py is not the expected version.
"""
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def apply(src_lines, patch_lines):
    """Apply a unified diff to a list of lines.

    Args:
        src_lines: lines of the original file, with line endings.
        patch_lines: lines of a unified diff against that file.

    Returns:
        The lines of the patched file.

    Raises:
        ValueError: if the diff is malformed or a hunk does not match the original file.
    """
    out, pos, i = [], 0, 0
    while i < len(patch_lines) and not patch_lines[i].startswith("@@"):
        i += 1  # skip the --- / +++ header
    while i < len(patch_lines):
        m = HUNK.match(patch_lines[i])
        if not m:
            raise ValueError(f"unexpected patch line {i + 1}: {patch_lines[i]!r}")
        # an empty old range ("-n,0") inserts after line n; otherwise the hunk starts at line n
        start = int(m.group(1)) - 1 if int(m.group(2) or 1) > 0 else int(m.group(1))
        out += src_lines[pos:start]
        pos = start
        i += 1
        while i < len(patch_lines) and not patch_lines[i].startswith("@@"):
            line = patch_lines[i]
            tag, text = (line[:1], line[1:]) if line else (" ", "")
            if tag == "\\":  # "\ No newline at end of file"
                pass
            elif tag in " -":
                if pos >= len(src_lines) or src_lines[pos].rstrip("\r\n") != text.rstrip("\r\n"):
                    raise ValueError(f"hunk does not match navis.py at line {pos + 1}")
                if tag == " ":
                    out.append(src_lines[pos])
                pos += 1
            elif tag == "+":
                out.append(text if text.endswith("\n") else text + "\n")
            i += 1
    return out + src_lines[pos:]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    navis_dir = sys.argv[1]
    with open(os.path.join(navis_dir, "navis.py"), encoding="utf-8") as f:
        src = f.readlines()
    with open(os.path.join(HERE, "navis_audit.patch"), encoding="utf-8") as f:
        patch = f.readlines()
    with open(os.path.join(navis_dir, "navis_audit.py"), "w", encoding="utf-8", newline="\n") as f:
        f.writelines(apply(src, patch))
    shutil.copy(os.path.join(HERE, "ltr_loss.py"), os.path.join(navis_dir, "ltr_loss.py"))
    print("written", os.path.join(navis_dir, "navis_audit.py"), "(+ ltr_loss.py)")


if __name__ == "__main__":
    main()
