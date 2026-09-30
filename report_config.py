"""Where installed copies send their analysis reports.

The upload token is injected at build time by CI from the AI_GRAVER_REPORT_TOKEN secret
(see .github/workflows/release.yml). A plain source checkout has no token and never uploads.
"""
from __future__ import annotations

import base64

REPORT_URL = "https://reports.82-27-77-56.nip.io/upload"
_TOKEN = ""


def get_upload_token() -> str:
    return base64.b64decode(_TOKEN).decode("utf-8") if _TOKEN else ""
