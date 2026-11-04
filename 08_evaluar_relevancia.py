
"""Compara la relevancia de la recuperación original y BGE."""

from rag import recuperar
from reranker import rerankear


PREGUNTA = "¿Qué cubre la garantía de lunas?"

print("=" * 65)
print("EVALUACIÓN COMPARATIVA DE RELEVANCIA")
print("=" * 65)
print(f"\nPregunta: {PREGUNTA}")

# 1. RAG original: recuperar directamente los cuatro primeros.
original = recuperar(PREGUNTA, k=4)

# 2. RAG mejorado: recuperar 12 candidatos y seleccionar cuatro con BGE.
candidatos = recuperar(PREGUNTA, k=12)
mejorados = rerankear(PREGUNTA, candidatos, k=4)


def mostrar_resultados(titulo, fragmentos):
    print(f"\n{'=' * 65}")
    print(titulo)
    print(f"{'=' * 65}")

    for i, chunk in enumerate(fragmentos, start=1):
        print(f"\nFragmento {i}")
        print(f"Documento: {chunk['documento']}")
        print(f"Página: {chunk['pagina']}")

        if "puntuacion_relevancia" in chunk:
            print(
                "Puntuación BGE: "
                f"{chunk['puntuacion_relevancia']:.6f}"
            )

        print(f"Texto: {chunk['texto'][:350].replace(chr(10), ' ')}")


mostrar_resultados("RAG ORIGINAL: TOP 4", original)
mostrar_resultados("RAG MEJORADO: TOP 4 CON BGE", mejorados)

print("\n" + "=" * 65)
print("COMPARACIÓN FINALIZADA")
print("=" * 65)
