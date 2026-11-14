"""Chunking utilities for Day06 Data Ingestion lab.
Fixed infinite loops and clean boundaries.
"""

from __future__ import annotations
import re
from typing import List
from .models import Chunk


def _uid(prefix: str, page: int, idx: int) -> str:
    return f"{prefix}_p{page:03d}_{idx:04d}"


def fixed_size(
    page_text: str,
    page_no: int,
    document_id: str = "day6",
    source: str = "pdf",
    size: int = 500,
    overlap: int = 50,
) -> List[Chunk]:
    """Split text into overlapping fixed-size windows."""
    chunks: List[Chunk] = []
    text = page_text.strip()
    if not text:
        return chunks

    start = 0
    idx = 0
    step = max(1, size - overlap)  # Đảm bảo luôn luôn tiến lên phía trước

    while start < len(text):
        end = min(start + size, len(text))
        snippet = text[start:end].strip()
        if snippet:
            chunks.append(
                Chunk(
                    chunk_id=_uid("fix", page_no, idx),
                    document_id=document_id,
                    source=source,
                    page=page_no,
                    text=snippet,
                    metadata={},
                )
            )
            idx += 1
        if end >= len(text):
            break
        start += step

    return chunks


def by_paragraph(
    page_text: str,
    page_no: int,
    document_id: str = "day6",
    source: str = "pdf",
    max_size: int = 800,
) -> List[Chunk]:
    """Split text by paragraphs, merging them up to max_size."""
    chunks: List[Chunk] = []
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
    if not raw_paragraphs:
        return chunks

    cur_parts: list[str] = []
    cur_len = 0
    idx = 0

    for para in raw_paragraphs:
        # Nếu 1 đoạn đơn lẻ quá dài, chia nhỏ nó ra theo max_size
        if len(para) > max_size:
            # Xả phần đang có trước
            if cur_parts:
                chunks.append(
                    Chunk(
                        chunk_id=_uid("par", page_no, idx),
                        document_id=document_id,
                        source=source,
                        page=page_no,
                        text="\n".join(cur_parts),
                        metadata={},
                    )
                )
                idx += 1
                cur_parts = []
                cur_len = 0

            # Chia đoạn dài thành các khúc nhỏ
            for sub_start in range(0, len(para), max_size):
                sub_snippet = para[sub_start : sub_start + max_size].strip()
                if sub_snippet:
                    chunks.append(
                        Chunk(
                            chunk_id=_uid("par", page_no, idx),
                            document_id=document_id,
                            source=source,
                            page=page_no,
                            text=sub_snippet,
                            metadata={},
                        )
                    )
                    idx += 1
            continue

        # Nếu thêm para này vào mà vượt quá max_size thì lưu chunk cũ lại
        if cur_len + len(para) + 1 > max_size:
            if cur_parts:
                chunks.append(
                    Chunk(
                        chunk_id=_uid("par", page_no, idx),
                        document_id=document_id,
                        source=source,
                        page=page_no,
                        text="\n".join(cur_parts),
                        metadata={},
                    )
                )
                idx += 1
                cur_parts = []
                cur_len = 0

        cur_parts.append(para)
        cur_len += len(para) + 1

    # Lưu nốt phần còn sót lại
    if cur_parts:
        chunks.append(
            Chunk(
                chunk_id=_uid("par", page_no, idx),
                document_id=document_id,
                source=source,
                page=page_no,
                text="\n".join(cur_parts),
                metadata={},
            )
        )

    return chunks
