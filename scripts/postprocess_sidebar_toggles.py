#!/usr/bin/env python3
"""Append sidebar toggle behavior to the built MyST client bundle."""

from __future__ import annotations

import sys
from pathlib import Path


MARKER = "/* opt-sidebar-toggle-script */"

SCRIPT = f"""

{MARKER}
(() => {{
  if (window.optSidebarToggleInitialized) return;
  window.optSidebarToggleInitialized = true;

  document.addEventListener(
    "click",
    (event) => {{
      const target =
        event.target instanceof Element
          ? event.target.closest(".opt-sidebar-toggle")
          : null;
      if (!target) return;

      const className = target.classList.contains("opt-sidebar-toggle-left")
        ? "opt-sidebar-left-collapsed"
        : target.classList.contains("opt-sidebar-toggle-right")
          ? "opt-sidebar-right-collapsed"
          : "";
      if (!className) return;

      event.preventDefault();
      event.stopPropagation();
      document.body.classList.toggle(className);
    }},
    true
  );
}})();
"""


def append_toggle_script(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    if MARKER in source:
        return False
    path.write_text(f"{source.rstrip()}{SCRIPT}", encoding="utf-8")
    return True


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: postprocess_sidebar_toggles.py HTML_ROOT", file=sys.stderr)
        return 2

    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 1

    entrypoints = sorted((root / "build").glob("entry.client-*.js"))
    if not entrypoints:
        print(f"error: no entry.client-*.js found under {root / 'build'}", file=sys.stderr)
        return 1

    changed = sum(append_toggle_script(path) for path in entrypoints)
    print(f"Appended sidebar toggle script to {changed} client bundle(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
