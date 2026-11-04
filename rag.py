
"""Recuperacion y generacion de respuestas para el RAG de seguros de automovil."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pgvector import Vector

from db import conectar, embedding


NOMBRE_TABLA = "fragmentos_rag10"
TENANT_ID = "seguros_auto_demo"


# Modelo local para reformular preguntas y generar respuestas.
llm = ChatOllama(model="llama3.1:8b", temperature=0)


def recuperar(
    pregunta: str,
    k: int = 4,
    tenant_id: str = TENANT_ID,
):
    """Combina busqueda textual y vectorial dentro del tenant indicado."""
    q = Vector(embedding([pregunta])[0])

    conn = conectar()

    try:
        # 1. Busqueda vectorial.
        filas_vector = conn.execute(
            f"""
            SELECT documento, pagina, texto,
                   embedding <=> %s AS distancia
            FROM {NOMBRE_TABLA}
            WHERE tenant_id = %s
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (q, tenant_id, q, k),
        ).fetchall()

        # 2. Busqueda textual complementaria.
        palabras_ignorar = {
            "que", "como", "cuando", "donde", "cual", "cuales",
            "para", "por", "con", "una", "uno", "del", "las",
            "los", "les", "esta", "este", "tiene", "cubren",
            "cubre", "garantia", "garantía", "seguro", "seguros",
            "sobre", "informacion", "información", "documento",
        }

        palabras = [
            palabra.strip("¿?¡!.,:;()[]{}\"'").lower()
            for palabra in pregunta.split()
        ]

        palabras = list(dict.fromkeys(
            palabra
            for palabra in palabras
            if len(palabra) >= 4 and palabra not in palabras_ignorar
        ))

        filas_texto = []

        if palabras:
            condiciones = " OR ".join(
                ["texto ILIKE %s"] * len(palabras)
            )

            filas_texto = conn.execute(
                f"""
                SELECT documento, pagina, texto,
                       embedding <=> %s AS distancia
                FROM {NOMBRE_TABLA}
                WHERE tenant_id = %s
                  AND ({condiciones})
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (
                    [q, tenant_id]
                    + [f"%{palabra}%" for palabra in palabras]
                    + [q, k]
                ),
            ).fetchall()

        # 3. Prioriza las coincidencias textuales y completa
        # con los resultados vectoriales, sin duplicados.
        resultados = {}

        for fila in filas_texto + filas_vector:
            documento, pagina, texto, distancia = fila
            clave = (documento, pagina, texto)

            if clave not in resultados:
                resultados[clave] = {
                    "documento": documento,
                    "pagina": pagina,
                    "texto": texto,
                    "distancia": distancia,
                }

        return list(resultados.values())[:k]

    finally:
        conn.close()


def recuperar_varias(
    preguntas,
    k: int = 4,
    tenant_id: str = TENANT_ID,
):
    """Combina busquedas conservando la prioridad de cada pregunta."""
    mejores = {}

    for indice, pregunta in enumerate(dict.fromkeys(preguntas)):
        fragmentos = recuperar(
            pregunta,
            k=k,
            tenant_id=tenant_id,
        )

        for posicion, chunk in enumerate(fragmentos):
            clave = (
                chunk["documento"],
                chunk["pagina"],
                chunk["texto"],
            )

            puntuacion = (indice, posicion)

            if (
                clave not in mejores
                or puntuacion < mejores[clave]["puntuacion"]
            ):
                mejores[clave] = {
                    **chunk,
                    "puntuacion": puntuacion,
                }

    ordenados = sorted(
        mejores.values(),
        key=lambda chunk: chunk["puntuacion"],
    )

    return [
        {
            clave: valor
            for clave, valor in chunk.items()
            if clave != "puntuacion"
        }
        for chunk in ordenados[:k]
    ]


# Instrucciones para responder usando exclusivamente los documentos.
PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Eres un asistente especializado en seguros de automovil.\n"
        "Responde utilizando EXCLUSIVAMENTE la informacion del contexto "
        "proporcionado. No utilices conocimientos externos ni inventes "
        "coberturas, condiciones o importes.\n"
        "Reglas:\n"
        "1. Si el contexto no contiene informacion suficiente para responder, "
        "responde exactamente: No lo sé.\n"
        "2. Si respondes, indica la fuente al final entre corchetes, con el "
        "nombre del documento y la pagina. Ejemplo: [documento.pdf, p. 2].\n"
        "3. Responde en español, de forma clara y breve, en 1-3 frases.\n"
        "4. Si los documentos contienen condiciones diferentes, no las "
        "mezcles: indica a que documento corresponde cada informacion.\n"
        "5. No afirmes que una cobertura esta incluida en una poliza concreta "
        "si el contexto no lo demuestra.\n\n"
        "CONTEXTO:\n{contexto}",
    ),
    ("human", "{pregunta}"),
])


def formatear_contexto(chunks) -> str:
    """Incluye el texto recuperado y su fuente."""
    return "\n\n".join(
        f"[{chunk['documento']}, p. {chunk['pagina']}]\n{chunk['texto']}"
        for chunk in chunks
    )


# Instrucciones para reformular preguntas dependientes del historial.
REFORMULAR = """Tu tarea es reformular la ultima pregunta de una conversacion
sobre seguros de automovil para que se pueda entender por si sola.

Reglas:
- Lee el historial para identificar el tema al que se refiere la pregunta.
- Si la pregunta depende de algo anterior, incorpora solo la informacion
  necesaria para aclararla.
- Si introduce un tema nuevo y se entiende por si sola, conservala.
- No cambies el tema, la intencion ni el alcance de la pregunta.
- No supongas que se refiere al tema anterior si no hay indicios.
- No contestes a la pregunta.
- Devuelve solamente la pregunta reformulada.

HISTORIAL:
{historial}

ULTIMA PREGUNTA:
{ultima}

PREGUNTA REFORMULADA:"""


def reformular_pregunta(mensajes) -> str:
    """Reformula preguntas dependientes del historial."""
    if not mensajes:
        return ""

    ultima = mensajes[-1].content

    if len(mensajes) == 1:
        return ultima

    historial = " / ".join(
        f"{'Usuario' if mensaje.type == 'human' else 'Asistente'}: "
        f"{mensaje.content}"
        for mensaje in mensajes[-5:-1]
    )

    resultado = llm.invoke(
        REFORMULAR.format(
            historial=historial,
            ultima=ultima,
        )
    )

    reformulada = resultado.content.strip()

    # Evita devolver una pregunta vacia si el modelo falla.
    return reformulada or ultima


def generar(pregunta: str, chunks) -> str:
    """Genera una respuesta basada exclusivamente en los fragmentos."""
    if not chunks:
        return "No lo sé."

    cadena = PROMPT | llm

    resultado = cadena.invoke({
        "contexto": formatear_contexto(chunks),
        "pregunta": pregunta,
    })

    respuesta = resultado.content.strip()

    if respuesta.lower().startswith("no lo sé"):
        return "No lo sé."

    return respuesta
