"""Añade aislamiento por tenant a los fragmentos del Proyecto 10."""

from db import conectar

NOMBRE_TABLA = "fragmentos_rag10"
TENANT_ID = "seguros_auto_demo"

conn = conectar()

try:
    conn.execute(f"""
        ALTER TABLE {NOMBRE_TABLA}
        ADD COLUMN IF NOT EXISTS tenant_id TEXT
        NOT NULL DEFAULT '{TENANT_ID}'
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_fragmentos_rag10_tenant
        ON fragmentos_rag10 (tenant_id)
    """)

    print("MIGRACION COMPLETADA")
    print("\nFragmentos por tenant:")

    filas = conn.execute("""
        SELECT tenant_id, COUNT(*)
        FROM fragmentos_rag10
        GROUP BY tenant_id
        ORDER BY tenant_id
    """).fetchall()

    for tenant, cantidad in filas:
        print(f"{tenant}: {cantidad} fragmentos")

    print("\nTotal de fragmentos:")
    print(
        conn.execute(
            "SELECT COUNT(*) FROM fragmentos_rag10"
        ).fetchone()[0]
    )

finally:
    conn.close()
