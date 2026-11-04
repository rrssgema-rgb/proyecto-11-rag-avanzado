from rag import recuperar
from reranker import rerankear

pregunta = "¿Qué cubre la garantía de lunas?"

print("RECUPERANDO LOS 10 CANDIDATOS...")

candidatos = recuperar(pregunta, k=10)

print(f"Candidatos recuperados: {len(candidatos)}")

print("\n--- RESULTADOS DEL RERANKER BGE ---")

resultados = rerankear(
    pregunta,
    candidatos,
    k=4,
)

print(f"\nFragmentos seleccionados: {len(resultados)}")

for i, chunk in enumerate(resultados, start=1):
    print(f"\n{i}. Puntuación BGE: {chunk['puntuacion_relevancia']:.6f}")
    print(f"   Documento: {chunk['documento']}")
    print(f"   Página: {chunk['pagina']}")
    print(f"   Texto: {chunk['texto'][:250].replace(chr(10), ' ')}")
