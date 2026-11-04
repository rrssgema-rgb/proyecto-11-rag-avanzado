
"""Chatbot RAG de seguros de automovil con LangGraph y memoria.

Uso:
    python 05_chatbot_grafo.py
    python 05_chatbot_grafo.py --chat
"""
import re

import sys

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from rag import (
    TENANT_ID,
    generar,
    recuperar_varias,
    reformular_pregunta,
)


class Estado(MessagesState):
    """Estado del chatbot: historial, pregunta reformulada y contexto."""

    pregunta: str
    contexto: list


def reformular(estado: Estado):
    """Reformula la ultima pregunta para que se entienda por si sola."""
    return {
        "pregunta": reformular_pregunta(estado["messages"])
    }



def nodo_recuperar(estado: Estado):
    original = estado["messages"][-1].content
    reformulada = estado["pregunta"]

    # Buscamos con la pregunta original y con la reformulada.
    # Asi conservamos los resultados que funcionan para preguntas directas
    # y añadimos los que ayudan a interpretar preguntas de seguimiento.
    preguntas_busqueda = [original]

    if reformulada.strip().lower() != original.strip().lower():
        preguntas_busqueda.append(reformulada)

    contexto = recuperar_varias(
        preguntas_busqueda,
        k=4,
        tenant_id=TENANT_ID
    )

    return {"contexto": contexto}




def nodo_generar(estado: Estado):
    mensajes = estado["messages"]
    original = mensajes[-1].content.strip()
    reformulada = estado["pregunta"]

    # Si el usuario pregunta por la fuente de la respuesta anterior,
    # reutilizamos la cita que ya se proporciono en esa respuesta.
    pregunta_fuente = re.search(
        r"\b(documento|fuente|fuentes|pagina|página)\b",
        original,
        flags=re.IGNORECASE
    )

    if pregunta_fuente:
        for mensaje in reversed(mensajes[:-1]):
            if isinstance(mensaje, AIMessage):
                citas = re.findall(
                    r"\[([^\[\]]+\.pdf,\s*p\.\s*\d+)\]",
                    mensaje.content,
                    flags=re.IGNORECASE
                )

                if citas:
                    respuesta = (
                        f"La respuesta anterior cita el documento "
                        f"{citas[0]}."
                    )
                    return {"messages": [AIMessage(content=respuesta)]}
                break

    # Para las demas preguntas, seguimos usando el RAG normalmente.
    if original == reformulada:
        pregunta = original
    else:
        pregunta = (
            f"{original} "
            f"(interpretacion de la pregunta: {reformulada})"
        )

    respuesta = generar(pregunta, estado["contexto"])
    return {"messages": [AIMessage(content=respuesta)]}



# Construccion del grafo: reformular -> recuperar -> generar.
grafo = StateGraph(Estado)

grafo.add_node("reformular", reformular)
grafo.add_node("recuperar", nodo_recuperar)
grafo.add_node("generar", nodo_generar)

grafo.add_edge(START, "reformular")
grafo.add_edge("reformular", "recuperar")
grafo.add_edge("recuperar", "generar")
grafo.add_edge("generar", END)

# La memoria conserva los mensajes por hilo de conversacion.
chatbot = grafo.compile(checkpointer=MemorySaver())


def preguntar(texto: str, hilo: str = "seguros-auto-1"):
    """Envia una pregunta al chatbot y conserva el historial del hilo."""
    return chatbot.invoke(
        {"messages": [HumanMessage(content=texto)]},
        {"configurable": {"thread_id": hilo}},
    )


def mostrar_fuentes(contexto):
    """Devuelve una lista legible de las fuentes recuperadas."""
    fuentes = sorted({
        f"{chunk['documento']} p.{chunk['pagina']}"
        for chunk in contexto
    })

    return ", ".join(fuentes) if fuentes else "ninguna"


if __name__ == "__main__":

    if "--chat" in sys.argv:
        print(
            "Asistente RAG de seguros de automovil. "
            "Escribe una pregunta; pulsa Enter vacio para salir."
        )

        while True:
            texto = input("\nTu: ").strip()

            if not texto:
                break

            resultado = preguntar(texto, hilo="chat-seguros-auto")

            print("Bot:", resultado["messages"][-1].content)
            print("Fuentes:", mostrar_fuentes(resultado["contexto"]))

        sys.exit(0)

    conversacion = [
        "¿Que cubre la responsabilidad civil obligatoria?",
        "¿Que documento contiene esa informacion?",
        "¿Que cubre la garantia de lunas?",
        "¿Cual es el precio exacto de mi poliza personal?",
    ]

    for texto in conversacion:
        resultado = preguntar(texto)

        print(f"USUARIO:     {texto}")
        print(f"REFORMULADA: {resultado['pregunta']}")
        print(f"FUENTES:     {mostrar_fuentes(resultado['contexto'])}")
        print(f"BOT:         {resultado['messages'][-1].content}\n")

    print(
        "Mensajes guardados en el hilo "
        f"'seguros-auto-1': {len(resultado['messages'])}"
    )

    chatbot.get_graph().draw_mermaid_png(
        output_file_path="grafo_rag.png",
        max_retries=5,
    )

    print("Diagrama del grafo guardado en grafo_rag.png")
