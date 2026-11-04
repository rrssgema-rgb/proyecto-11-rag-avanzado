# Proyecto 11 · RAG avanzado: evaluación y reranking

## RAG de seguros de automóvil con PostgreSQL, pgvector y BGE Reranker

### 1. Objetivo del proyecto

Este proyecto amplía el sistema RAG desarrollado en el Proyecto 10, incorporando una mejora en la recuperación de información mediante un modelo *reranker*.

El objetivo es comparar el comportamiento del RAG original con una versión mejorada que utiliza **BAAI/bge-reranker-v2-m3** para ordenar los fragmentos recuperados según su relevancia respecto a la pregunta del usuario.

El sistema responde preguntas sobre documentación de seguros de automóvil utilizando exclusivamente la información de los documentos. Cuando no encuentra información suficiente, debe responder:

`No lo sé.`

Además, se evalúa la fidelidad de las respuestas mediante RAGAS, utilizando la métrica Faithfulness.

### 2. Arquitectura del sistema

El proyecto mantiene la arquitectura RAG original:

1. Los documentos PDF se procesan y dividen en fragmentos.
2. Los fragmentos y sus embeddings se almacenan en PostgreSQL con pgvector.
3. Se recuperan fragmentos mediante búsqueda vectorial y textual.
4. El modelo local genera una respuesta utilizando el contexto recuperado.
5. Se cita el documento y la página utilizados.

Sobre esta arquitectura se incorpora el reranker BGE:

- **RAG original:** recupera los 4 primeros fragmentos.
- **RAG con BGE:** recupera inicialmente 12 candidatos y utiliza el reranker para seleccionar y ordenar los 4 fragmentos más relevantes.
- Ambas versiones utilizan el mismo modelo generativo, la misma pregunta y el mismo prompt.

De esta forma se puede comparar el efecto de la nueva etapa de recuperación sin cambiar el modelo que genera las respuestas.

### 3. Tecnologías utilizadas

- Python.
- PostgreSQL y pgvector, alojados en Neon.
- Ollama.
- `nomic-embed-text` para generar embeddings.
- `llama3.1:8b` para generar respuestas y actuar como modelo evaluador.
- LangChain.
- Sentence Transformers.
- `BAAI/bge-reranker-v2-m3` como modelo de reranking.
- RAGAS para evaluar Faithfulness.

El reranker se ejecuta localmente utilizando CPU.

### 4. Documentación utilizada

La carpeta `docs/` contiene los tres documentos PDF empleados por el sistema:

1. `gesauto_condiciones_modelo_2020.pdf`: condiciones generales del modelo GesAuto 2020.
2. `linea_directa_coberturas_complementarias.pdf`: coberturas complementarias de Línea Directa, marzo de 2025.
3. `linea_directa_condiciones_generales.pdf`: condiciones generales de Línea Directa, edición de enero de 2026.

Se trata de ediciones concretas de documentación aseguradora. Sus contenidos no deben interpretarse automáticamente como condiciones vigentes de cualquier póliza.

### 5. Recuperación y reranking

La recuperación original combina búsqueda vectorial y textual dentro del tenant correspondiente.

El nuevo componente `reranker.py` utiliza el modelo `BAAI/bge-reranker-v2-m3`.

Su funcionamiento es el siguiente:

1. Recibe la pregunta y los 12 fragmentos candidatos.
2. Evalúa la relación entre la pregunta y cada fragmento.
3. Asigna una puntuación de relevancia.
4. Ordena los fragmentos de mayor a menor puntuación.
5. Devuelve los 4 primeros para generar la respuesta.

Las puntuaciones del modelo son valores de clasificación (*scores*). No representan porcentajes ni probabilidades de relevancia.

### 6. Evaluación de relevancia

Se ha comparado la recuperación original con la recuperación después de aplicar BGE.

Se utilizaron preguntas sobre la garantía de lunas de los documentos de seguros.

En la pregunta «¿Qué cubre la garantía de lunas?», el RAG original recuperó inicialmente un fragmento de GesAuto que contenía directamente la cobertura de reparación, reposición y colocación de lunas.

BGE reorganizó los resultados y situó en primera posición fragmentos de Línea Directa relacionados con la garantía, aunque el fragmento que contenía la cobertura de GesAuto pasó a una posición posterior.

En la pregunta «¿Qué elementos cubre Línea Directa en la garantía de rotura de lunas?», BGE situó en primera posición un fragmento que contenía el comienzo del artículo 31, mientras que la recuperación original priorizó un fragmento con parte de la enumeración de elementos cubiertos.

La comparación muestra que el reranker puede modificar la prioridad de los fragmentos recuperados. Sin embargo, no garantiza por sí solo que se recupere toda la información necesaria para responder correctamente.

### 7. Evaluación de Faithfulness con RAGAS

Se ha utilizado la métrica Faithfulness de RAGAS para comprobar si las afirmaciones de las respuestas están respaldadas por el contexto recuperado.

La evaluación utiliza `llama3.1:8b` como modelo juez, con temperatura 0.

#### Primera pregunta

**Pregunta:** ¿Qué cubre la garantía de lunas?

| Resultado | RAG original | RAG con BGE |
|---|---:|---:|
| Faithfulness | 1,00 | 0,25 |

Ambas versiones generaron la misma respuesta:

La garantía de lunas cubre la reparación, la reposición y los gastos de colocación de las lunas del vehículo, siempre que su inclusión aparezca expresamente pactada en las condiciones particulares.

La respuesta cita el documento `gesauto_condiciones_modelo_2020.pdf`, página 41.

En esta prueba, el RAG original obtuvo una puntuación de Faithfulness superior a la versión con BGE.

#### Segunda pregunta

**Pregunta:** ¿Qué elementos cubre Línea Directa en la garantía de rotura de lunas?

| Resultado | RAG original | RAG con BGE |
|---|---:|---:|
| Faithfulness | 0,00 | 0,00 |
| Respuesta | No lo sé. | No lo sé. |

En esta evaluación, los contextos recuperados no contenían información suficiente para responder a la pregunta. Ambas versiones activaron el mecanismo de fallback.

Este resultado muestra que el reranking no siempre consigue resolver las limitaciones de la recuperación inicial.

### 8. Mecanismo de fallback

El sistema debe evitar inventar información cuando no dispone de contexto suficiente.

Se han realizado dos comprobaciones:

- Pregunta sin información relacionada con los documentos, utilizando un contexto vacío.
- Pregunta sin relación con la documentación, aunque el sistema haya recuperado fragmentos irrelevantes.

En ambos casos, la respuesta obtenida fue:

`No lo sé.`

Este comportamiento permite reducir las respuestas sin respaldo documental.

### 9. Aislamiento por tenant

La tabla de fragmentos incluye la columna `tenant_id`.

Las consultas de recuperación filtran los resultados por el tenant solicitado para evitar que una búsqueda acceda a fragmentos pertenecientes a otro ámbito.

En la configuración de demostración se utiliza:

`seguros_auto_demo`

La comprobación realizada mostró 1.139 fragmentos para este tenant y cero fragmentos visibles para otro tenant mediante la consulta de prueba.

### 10. Estructura del proyecto

Los principales componentes utilizados son:

| Archivo | Función |
|---|---|
| `db.py` | Conexión a PostgreSQL y generación de embeddings. |
| `rag.py` | Recuperación de fragmentos y generación de respuestas. |
| `reranker.py` | Ordenación de fragmentos mediante BGE. |
| `07_generar_reranker.py` | Prueba de generación con reranking. |
| `08_evaluar_relevancia.py` | Comparación de relevancia entre RAG original y BGE. |
| `evaluar_faithfulness.py` | Evaluación de respuestas mediante RAGAS. |
| `ragas_compat.py` | Compatibilidad utilizada para la evaluación con RAGAS. |
| `probar_reranker.py` | Pruebas del componente reranker. |
| `documentacion/` | Capturas, resultados y evidencias de evaluación. |
| `docs/` | Documentos PDF utilizados por el RAG. |

### 11. Resultados y conclusiones

La incorporación de BGE permite añadir una etapa de clasificación de relevancia después de la recuperación inicial.

Las pruebas realizadas demuestran que:

- El reranker funciona y modifica el orden de los fragmentos recuperados.
- Puede situar en primera posición fragmentos más relacionados con la pregunta.
- No garantiza que toda la información necesaria esté incluida entre los cuatro fragmentos seleccionados.
- La fragmentación de los documentos puede separar información que pertenece a un mismo artículo o cobertura.
- La evaluación de Faithfulness permite detectar diferencias entre las respuestas generadas por ambas versiones.
- El mecanismo de fallback evita responder cuando el contexto no contiene información suficiente.

En las pruebas realizadas no se ha demostrado una mejora general de la calidad de las respuestas con BGE. En la primera pregunta, el RAG original obtuvo mejor Faithfulness; en la segunda, ambas versiones activaron el fallback.

Por tanto, la principal aportación del proyecto es la implementación y evaluación comparativa de una estrategia de reranking, así como la identificación de sus ventajas y limitaciones.

### 12. Seguridad y consideraciones finales

- No publicar el archivo `.env` ni credenciales de conexión.
- No subir el entorno virtual `.venv`.
- No incluir documentos PDF ni archivos ajenos a la entrega si no son necesarios.
- Mantener el aislamiento por `tenant_id` en todas las consultas.
- No ejecutar procesos de indexación o reconstrucción de tablas sobre la base de datos de producción o demostración sin necesidad.
- No interpretar las puntuaciones de BGE como probabilidades.
- No considerar las métricas automáticas de RAGAS como una evaluación humana definitiva.

**Conclusión:** el Proyecto 11 amplía el RAG del Proyecto 10 con un reranker BGE y una evaluación mediante RAGAS, permitiendo analizar de forma práctica cómo cambia la recuperación de información y qué limitaciones permanecen en un sistema RAG aplicado al ámbito asegurador.