"""Indexacion por lotes de documentos del Proyecto 10."""

from db import DIMENSIONES, conectar, embedding
from ingesta import cargar_documentos, trocear

NOMBRE_TABLA = "fragmentos_rag10"
NOMBRE_INDICE = "fragmentos_rag10_hnsw"
NOMBRE_INDICE_TENANT = "idx_fragmentos_rag10_tenant"
TENANT_ID = "seguros_auto_demo"
TAMANO_LOTE = 25


def main():
    chunks = trocear(cargar_documentos())
    print(f"Fragmentos que se van a indexar: {len(chunks)}", flush=True)

    conn = conectar()

    try:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {NOMBRE_TABLA} (
                id SERIAL PRIMARY KEY,
                documento TEXT NOT NULL,
                pagina INT NOT NULL,
                texto TEXT NOT NULL,
                embedding VECTOR({DIMENSIONES}) NOT NULL,
                tenant_id TEXT NOT NULL DEFAULT '{TENANT_ID}'
            )
        """)

        # Garantiza que la columna exista en tablas creadas anteriormente.
        conn.execute(f"""
            ALTER TABLE {NOMBRE_TABLA}
            ADD COLUMN IF NOT EXISTS tenant_id TEXT
            NOT NULL DEFAULT '{TENANT_ID}'
        """)

        # Indice para facilitar el filtrado por tenant.
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {NOMBRE_INDICE_TENANT}
            ON {NOMBRE_TABLA} (tenant_id)
        """)

        # ATENCION: elimina los fragmentos de esta tabla.
        # Solo se ejecuta al solicitar una reindexacion completa.
        conn.execute(f"TRUNCATE TABLE {NOMBRE_TABLA} RESTART IDENTITY")

    finally:
        conn.close()

    total = len(chunks)

    for inicio in range(0, total, TAMANO_LOTE):
        lote = chunks[inicio:inicio + TAMANO_LOTE]
        textos = [c.page_content for c in lote]

        # Generar embeddings fuera de la conexion a Neon.
        vectores = embedding(textos)

        # Abrir una conexion nueva para cada lote.
        conn = conectar()

        try:
            with conn.cursor() as cur:
                cur.executemany(
                    f"""
                    INSERT INTO {NOMBRE_TABLA}
                        (documento, pagina, texto, embedding, tenant_id)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            chunk.metadata["documento"],
                            chunk.metadata["pagina"],
                            texto,
                            vector,
                            TENANT_ID,
                        )
                        for chunk, texto, vector in zip(
                            lote, textos, vectores
                        )
                    ],
                )

            print(
                f"Guardados: {min(inicio + len(lote), total)}/{total}",
                flush=True,
            )

        finally:
            conn.close()

    conn = conectar()

    try:
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {NOMBRE_INDICE}
            ON {NOMBRE_TABLA}
            USING hnsw (embedding vector_cosine_ops)
        """)

        filas = conn.execute(f"""
            SELECT documento, COUNT(*)
            FROM {NOMBRE_TABLA}
            WHERE tenant_id = %s
            GROUP BY documento
            ORDER BY documento
        """, (TENANT_ID,)).fetchall()

        for documento, cantidad in filas:
            print(f"{documento}: {cantidad} fragmentos")

        total_guardado = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM {NOMBRE_TABLA}
            WHERE tenant_id = %s
            """,
            (TENANT_ID,),
        ).fetchone()[0]

        print(f"\nTOTAL GUARDADO: {total_guardado}")

        if total_guardado != total:
            raise RuntimeError(
                f"Se esperaban {total} fragmentos y "
                f"se guardaron {total_guardado}."
            )

        dimensiones = conn.execute(f"""
            SELECT vector_dims(embedding)
            FROM {NOMBRE_TABLA}
            WHERE tenant_id = %s
            LIMIT 1
        """, (TENANT_ID,)).fetchone()

        if dimensiones:
            print("Dimensiones del embedding:", dimensiones[0])

        print("INDEXACION COMPLETADA CORRECTAMENTE.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
