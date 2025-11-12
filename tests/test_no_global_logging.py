# -*- coding: utf-8 -*-
import importlib
import logging
import sys
import subprocess
import textwrap

def test_import_does_not_mess_with_root_logger():
    # running import on fresh so we can see before/after
    code = textwrap.dedent("""
        import logging
        before_handlers = len(logging.getLogger().handlers)
        import consenrich.matching.matching_alg  # triggers module import
        after_handlers = len(logging.getLogger().handlers)
        print(f"{before_handlers},{after_handlers}")
    """)
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    before, after = [int(x) for x in out.split(",")]
    assert after == before
