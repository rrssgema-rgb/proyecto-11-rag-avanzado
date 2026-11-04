"""Ingesta: cargar los documentos y partirlos en chunks (paso 1 del pipeline RAG)."""
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CARPETA_DOCS = Path(__file__).parent / "docs"


def cargar_documentos(carpeta: Path = CARPETA_DOCS, limpiar: bool = True):
    """Lee cada PDF (una pieza por pagina) y cada TXT (una pieza por archivo) de la carpeta."""
    paginas = []
    for ruta in sorted(carpeta.iterdir()):
        if ruta.suffix == ".pdf":
            piezas = PyPDFLoader(str(ruta)).load()           # un Document por pagina
            if limpiar:
                # El PDF trae un salto de linea al final de cada renglon, en mitad de las frases.
                # Si no se quitan, el splitter corta ahi (ver extras/paso4_el_corte_importa.py)
                for p in piezas:
                    p.page_content = re.sub(r"\s*\n\s*", " ", p.page_content)
        elif ruta.suffix in (".txt", ".md"):
            piezas = TextLoader(str(ruta), encoding="utf-8").load()
        else:
            continue
        for p in piezas:
            # Metadatos limpios: de que documento y pagina viene. Es lo que luego permite CITAR
            p.metadata = {"documento": ruta.name, "pagina": p.metadata.get("page", 0) + 1}
        paginas.extend(piezas)
    return paginas


def trocear(paginas, tamano: int = 500, solapamiento: int = 75):
    """Parte el texto en chunks de ~`tamano` caracteres; cada uno repite los ultimos `solapamiento` del anterior."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=tamano,
        chunk_overlap=solapamiento,
        separators=["\n\n", "\n", ". ", " ", ""],   # intenta cortar por parrafo, luego linea, luego frase...
        keep_separator="end",                        # el punto se queda al final de su frase
    )
    return splitter.split_documents(paginas)          # cada chunk hereda los metadatos de su pagina
