"""Genera una respuesta RAG utilizando el reranker BGE."""

from rag import recuperar, generar
from reranker import rerankear


pregunta = "¿Qué cubre la garantía de lunas?"

print("PREGUNTA:")
print(pregunta)

# Recuperar los mismos 10 candidatos iniciales.
candidatos = recuperar(pregunta, k=12)

print(f"\nCandidatos recuperados: {len(candidatos)}")

# Reordenar y seleccionar los 4 fragmentos más relevantes.
fragmentos = rerankear(
    pregunta,
    candidatos,
    k=4,
)

print(f"Fragmentos seleccionados por BGE: {len(fragmentos)}")

print("\n--- CONTEXTO SELECCIONADO ---")

for i, chunk in enumerate(fragmentos, start=1):
    print(
        f"\n{i}. Puntuación BGE: "
        f"{chunk['puntuacion_relevancia']:.6f}"
    )
    print(f"   Documento: {chunk['documento']}")
    print(f"   Página: {chunk['pagina']}")
    print(f"   Texto: {chunk['texto'][:250].replace(chr(10), ' ')}")

# Generar la respuesta con el mismo modelo y prompt del RAG original.
respuesta = generar(pregunta, fragmentos)

print("\n--- RESPUESTA CON RERANKER BGE ---")
print(respuesta)