"""
File: /main.py
Project: chatbot
Created Date: Sunday May 31st 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday May 31st 2026 6:10:42 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import os
from pathlib import Path

from src.core.plugins.catalog import PluginManifestCatalog
from src.core.plugins.context import PluginContext
from src.core.plugins.manifest import MANIFEST_FILENAME

MEMORY_PLUGIN_NAME = "chatbot-memory"


def _plugins_dir() -> Path:
    project_root = Path(__file__).resolve().parent.parent.parent
    return Path(os.getenv("PLUGINS_DIR", str(project_root / "plugins")))


def is_memory_plugin_available() -> bool:
    plugins_dir = _plugins_dir()
    manifest_path = plugins_dir / MEMORY_PLUGIN_NAME / MANIFEST_FILENAME
    if manifest_path.exists():
        return True
    return any(
        m.plugin_dir is not None and m.plugin_dir.name == MEMORY_PLUGIN_NAME
        for m in PluginManifestCatalog(plugins_dir).discover(validate=True)
    )


memory_plugin_available = is_memory_plugin_available()


def register(context: PluginContext):
    if context._app:
        from routes.inference import create_inference_router

        context.include_router(create_inference_router(context))

