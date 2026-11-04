"""Conexion a PostgreSQL y embeddings locales mediante Ollama."""

import json
import os
import urllib.request

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()

MODELO_EMBEDDINGS = "nomic-embed-text"
DIMENSIONES = 768
URL_OLLAMA = "http://localhost:11434/api/embed"
TAMANO_LOTE = 4


def conectar() -> psycopg.Connection:
    """Conecta con Neon y habilita pgvector."""
    conn = psycopg.connect(
        os.environ["DATABASE_URL"],
        autocommit=True,
    )
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def _embedding_lote(textos):
    """Genera embeddings para un lote de textos."""
    datos = json.dumps({
        "model": MODELO_EMBEDDINGS,
        "input": textos,
    }).encode("utf-8")

    solicitud = urllib.request.Request(
        URL_OLLAMA,
        data=datos,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(solicitud, timeout=300) as respuesta:
        resultado = json.loads(respuesta.read().decode("utf-8"))

    vectores = resultado["embeddings"]

    if len(vectores) != len(textos):
        raise ValueError(
            f"Se esperaban {len(textos)} vectores, "
            f"pero Ollama devolvio {len(vectores)}."
        )

    for vector in vectores:
        if len(vector) != DIMENSIONES:
            raise ValueError(
                f"Dimension incorrecta: {len(vector)}. "
                f"Se esperaban {DIMENSIONES}."
            )

    return vectores


def embedding(textos):
    """Genera embeddings por lotes para evitar peticiones demasiado grandes."""
    if isinstance(textos, str):
        textos = [textos]

    textos = list(textos)

    if not textos:
        return []

    todos_los_vectores = []
    total = len(textos)

    for inicio in range(0, total, TAMANO_LOTE):
        lote = textos[inicio:inicio + TAMANO_LOTE]

        print(
            f"Generando embeddings: "
            f"{min(inicio + len(lote), total)}/{total}",
            flush=True,
        )

        vectores = _embedding_lote(lote)
        todos_los_vectores.extend(vectores)

    return todos_los_vectores
