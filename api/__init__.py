"""API package initialization.

Avoid eagerly importing heavy inference stack (api.main) so that alternative
entrypoints like `api.production_main:app` can select lightweight/heavy model
profiles without triggering transformers/datasets imports.

If you need the original `api.main` app, import it explicitly:
	from api.main import app

Set environment variable API_AUTO_IMPORT_MAIN=1 to restore eager import.
"""
import os

if os.getenv("API_AUTO_IMPORT_MAIN") == "1":
	from .main import app  # type: ignore  # noqa: F401
	__all__ = ["app"]
else:
	__all__ = []