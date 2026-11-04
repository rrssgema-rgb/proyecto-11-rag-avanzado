"""Evaluacion de Faithfulness para RAG original y BGE."""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from langchain_ollama import ChatOllama
import ragas_compat  # noqa: F401

from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper

from rag import (
    generar,
    recuperar,
    DOCUMENTOS_LINEA_DIRECTA,
)
from reranker import rerankear


PREGUNTA = "¿Qué gastos de remolcaje cubre Línea Directa en España en caso de rotura de lunas?"

CARPETA = Path(__file__).parent / "documentacion"
CARPETA.mkdir(exist_ok=True)

# RAG original: cuatro fragmentos de LÃ­nea Directa.
print("\nRAG ORIGINAL")

chunks_antes = recuperar(
    PREGUNTA,
    k=4,
    documentos=DOCUMENTOS_LINEA_DIRECTA,
)

# RAG con BGE: doce candidatos de LÃ­nea Directa,
# reordenados y reducidos a cuatro.
print("\nRAG CON BGE")

candidatos = recuperar(
    PREGUNTA,
    k=12,
    documentos=DOCUMENTOS_LINEA_DIRECTA,
)

chunks_despues = rerankear(
    PREGUNTA,
    candidatos,
    k=4,
)

muestras = []

for nombre, chunks in [
    ("RAG ORIGINAL", chunks_antes),
    ("RAG CON BGE", chunks_despues),
]:
    respuesta = generar(PREGUNTA, chunks)

    print(f"\n{nombre}:")
    print("Fragmentos:", len(chunks))

    for chunk in chunks:
        print(
            chunk["documento"],
            "- pÃ¡gina",
            chunk["pagina"],
        )

    print("Respuesta:")
    print(respuesta)

    muestras.append({
        "user_input": PREGUNTA,
        "response": respuesta,
        "retrieved_contexts": [
            chunk["texto"] for chunk in chunks
        ] or ["(sin contexto)"],
    })


juez = LangchainLLMWrapper(
    ChatOllama(
        model="llama3.1:8b",
        temperature=0,
    )
)

metricas = Faithfulness(llm=juez)

print("\nEvaluando Faithfulness...")

resultado = evaluate(
    EvaluationDataset.from_list(muestras),
    metrics=[metricas],
    run_config=RunConfig(
        timeout=900,
        max_workers=1,
        max_retries=2,
    ),
    show_progress=False,
)

tabla = resultado.to_pandas()

tabla.to_csv(
    CARPETA / "16_faithfulness_remolcaje.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nRESULTADOS:")
print(tabla.to_string(index=False))

print(
    "\nEvaluacion guardada en "
    "documentacion/16_faithfulness_remolcaje.csv"
)



