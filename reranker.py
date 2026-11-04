
"""Reranking de fragmentos mediante BAAI/bge-reranker-v2-m3."""

from sentence_transformers import CrossEncoder


MODELO_RERANKER = "BAAI/bge-reranker-v2-m3"

# Cargar el modelo una sola vez en CPU.
reranker = CrossEncoder(
    MODELO_RERANKER,
    max_length=512,
    device="cpu",
)


def rerankear(
    pregunta: str,
    chunks: list,
    k: int = 4,
) -> list:
    """
    Reordena los fragmentos según su relevancia
    respecto a la pregunta mediante un cross-encoder.
    """

    if not chunks:
        return []

    pares = [
        (pregunta, chunk["texto"])
        for chunk in chunks
    ]

    puntuaciones = reranker.predict(pares)

    resultados = []

    for chunk, puntuacion in zip(chunks, puntuaciones):
        resultados.append({
            **chunk,
            "puntuacion_relevancia": float(puntuacion),
        })

    resultados.sort(
        key=lambda x: x["puntuacion_relevancia"],
        reverse=True,
    )

    return resultados[:k]
