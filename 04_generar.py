"""Paso 4: generar respuestas basadas exclusivamente en documentos de seguros."""

import sys

from rag import generar, recuperar


preguntas = sys.argv[1:] or [
    "¿Qué cubre la responsabilidad civil obligatoria?",
    "¿Cuál es el precio exacto de mi póliza personal?",
]

for pregunta in preguntas:
    chunks = recuperar(pregunta, k=4)

    print(f"PREGUNTA: {pregunta}")

    if chunks:
        fuentes = list(dict.fromkeys(
            f"{c['documento']} p.{c['pagina']}" for c in chunks
        ))
        print(f"FUENTES RECUPERADAS: {', '.join(fuentes)}")
    else:
        print("FUENTES RECUPERADAS: ninguna")

    respuesta = generar(pregunta, chunks)
    print(f"RESPUESTA: {respuesta}\n")
