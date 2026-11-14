"""Read and normalize source data."""
import logging
from pathlib import Path
from pypdf import PdfReader
from .models import DocumentPage

# Tắt cảnh báo font rác của pypdf
logging.getLogger("pypdf").setLevel(logging.ERROR)


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical page; keep its page number and source."""
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"📖 Reading PDF from {pdf_path.name}...", flush=True)
    reader = PdfReader(str(pdf_path))
    pages = []
    total = len(reader.pages)

    for i, p in enumerate(reader.pages, start=1):
        try:
            text = p.extract_text() or ""
        except Exception:
            text = ""

        is_empty = not bool(text.strip())
        page_obj = DocumentPage(
            document_id=document_id,
            source=source,
            page=i,
            text=text,
            metadata={"empty": is_empty},
        )
        pages.append(page_obj)

    print(f"✅ Loaded {len(pages)}/{total} pages successfully!\n", flush=True)
    return pages


def normalize_text(text: str) -> str:
    """Normalize whitespace and clean text."""
    return " ".join(text.split())
