"""Paso 3 · Recuperar: los 4 chunks mas relevantes para una pregunta y de que documento vienen."""
import sys

from rag import recuperar

pregunta = sys.argv[1] if len(sys.argv) > 1 else "¿Cuántos días libres tengo al año?"
chunks = recuperar(pregunta, k=4)

print(f"Pregunta: {pregunta}\n")
for i, c in enumerate(chunks, 1):
    print(f"{i}. distancia {c['distancia']:.3f} · {c['documento']} p.{c['pagina']}")
    print(f"   {c['texto'][:230].replace(chr(10), ' ')}...\n")
