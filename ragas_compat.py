"""Compatibilidad de RAGAS 0.4.3 con LangChain 1.x."""

import os
import sys
import types

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

for ruta, clase in [
    ("langchain_community.chat_models.vertexai", "ChatVertexAI"),
    ("langchain_community.llms", "VertexAI")
]:
    try:
        modulo = __import__(ruta, fromlist=[clase])
        if hasattr(modulo, clase):
            continue
    except ImportError:
        modulo = types.ModuleType(ruta)
        sys.modules[ruta] = modulo

    setattr(modulo, clase, type(clase, (), {}))
