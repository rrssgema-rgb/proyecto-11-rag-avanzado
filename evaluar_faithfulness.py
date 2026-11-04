

"""Evaluación detallada de Faithfulness para RAG original y BGE."""

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from langchain_ollama import ChatOllama
import ragas_compat  # noqa: F401

from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper

from rag import generar, recuperar
from reranker import rerankear


#PREGUNTA = "¿Qué cubre la garantía de lunas?"
PREGUNTA = "¿Qué elementos cubre Línea Directa en la garantía de rotura de lunas?"
CARPETA = Path(__file__).parent / "documentacion"
CARPETA.mkdir(exist_ok=True)

# Recuperación original.
print("\nRAG ORIGINAL")
chunks_antes = recuperar(PREGUNTA, k=4)

# Recuperación y reranking BGE.
print("\nRAG CON BGE")
candidatos = recuperar(PREGUNTA, k=12)
chunks_despues = rerankear(PREGUNTA, candidatos, k=4)

muestras = []

for nombre, chunks in [
    ("RAG ORIGINAL", chunks_antes),
    ("RAG CON BGE", chunks_despues),
]:
    respuesta = generar(PREGUNTA, chunks)

    print(f"\n{nombre}:")
    print(respuesta)

    muestras.append({
        "user_input": PREGUNTA,
        "response": respuesta,
        "retrieved_contexts": [
            chunk["texto"] for chunk in chunks
        ] or ["(sin contexto)"],
    })

juez = LangchainLLMWrapper(
    ChatOllama(model="llama3.1:8b", temperature=0)
)

#metricas = Faithfulness(llm=juez)
metricas = Faithfulness()

print("\nEvaluando Faithfulness con explicaciones...")

resultado = evaluate(
    EvaluationDataset.from_list(muestras),
    metrics=[metricas],
    llm=juez,
    run_config=RunConfig(
        timeout=900,
        max_workers=1,
        max_retries=2,
    ),
    show_progress=False,
)

tabla = resultado.to_pandas()

tabla.to_csv(
    CARPETA / "12_faithfulness_detallado.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nRESULTADOS DETALLADOS")
print(tabla.to_string(index=False))

print("\nEvaluación guardada en documentacion.")
