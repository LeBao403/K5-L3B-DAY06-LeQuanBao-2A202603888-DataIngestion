import json
import psycopg
import numpy as np
from typing import List, Optional, Dict

from .models import Chunk


class PgVectorStore:
    """A thin wrapper around a PostgreSQL + pgvector table."""

    def __init__(self, database_url: str, dimensions: int = 384):
        url = database_url.replace("localhost", "127.0.0.1")
        self.conn = psycopg.connect(url, connect_timeout=10, autocommit=True)
        self.dim = dimensions

    def create_schema(self) -> None:
        """Create fresh chunks table with all required columns."""
        sql = f"""
        CREATE EXTENSION IF NOT EXISTS vector;
        DROP TABLE IF EXISTS chunks;
        CREATE TABLE chunks (
            chunk_id    TEXT PRIMARY KEY,
            document_id TEXT,
            source      TEXT,
            page        INT,
            text        TEXT NOT NULL,
            metadata    JSONB,
            embedding   VECTOR({self.dim})
        );
        """
        with self.conn.cursor() as cur:
            cur.execute(sql)

    def upsert(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        """Bulk INSERT-ON-CONFLICT for chunks."""
        sql = """
        INSERT INTO chunks
            (chunk_id, document_id, source, page, text, metadata, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
        ON CONFLICT (chunk_id) DO UPDATE
          SET document_id = EXCLUDED.document_id,
              source      = EXCLUDED.source,
              page        = EXCLUDED.page,
              text        = EXCLUDED.text,
              metadata    = EXCLUDED.metadata,
              embedding   = EXCLUDED.embedding;
        """
        with self.conn.cursor() as cur:
            for chunk, vec in zip(chunks, vectors):
                vec_str = "[" + ",".join(map(str, vec.tolist())) + "]"
                cur.execute(
                    sql,
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.source,
                        chunk.page,
                        chunk.text,
                        json.dumps(chunk.metadata),
                        vec_str,
                    ),
                )

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 3,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        filter_parts: list = []
        filter_params: list = []

        if filters:
            for k, v in filters.items():
                filter_parts.append("metadata->>%s = %s")
                filter_params.extend([k, str(v)])

        filter_clause = ("WHERE " + " AND ".join(filter_parts)) if filter_parts else ""
        vec_str = "[" + ",".join(map(str, query_vec.tolist())) + "]"

        sql = f"""
        SELECT chunk_id, text, page, source, metadata,
               1 - (embedding <=> %s::vector) AS score
        FROM chunks
        {filter_clause}
        ORDER BY score DESC
        LIMIT %s;
        """
        params = [vec_str] + filter_params + [top_k]

        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            {
                "chunk_id": r[0],
                "text": r[1],
                "page": r[2],
                "source": r[3],
                "metadata": r[4],
                "score": float(r[5]),
            }
            for r in rows
        ]
