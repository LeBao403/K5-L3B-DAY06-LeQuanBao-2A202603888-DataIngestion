import os
import json
import csv
from pathlib import Path
from typing import List

from .ingestion import load_pdf, normalize_text
from .chunking import fixed_size, by_paragraph
from .models import Chunk
from .embeddings import MyEmbedder
from .store import PgVectorStore
from .evaluation import evaluate_query, summarize


def main() -> None:
    root_dir = Path(__file__).parents[2]  # K5-L3B-DAY06-LeQuanBao-2A202603888-DataIngestion/
    pdf_path = root_dir / "data" / "CMO-October-2026.pdf"

    print("🚀 Bắt đầu Day 06 Pipeline...\n", flush=True)

    # 1. Load PDF
    pages = load_pdf(pdf_path, document_id="day6", source="CMO")
    if len(pages) < 30:
        raise ValueError(f"Corpus must contain ≥30 pages, got {len(pages)}")

    for p in pages:
        p.text = normalize_text(p.text)

    # 2. Embedding Model
    print("⏳ Khởi tạo model Embeddings...", flush=True)
    embedder = MyEmbedder()

    # 3. Kết nối DB
    db_url = os.getenv("DATABASE_URL", "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab")
    print(f"🔗 Kết nối Database: {db_url}...", flush=True)
    store = PgVectorStore(database_url=db_url, dimensions=embedder.dim)
    store.create_schema()
    print("✔️ Database & Schema đã sẵn sàng!\n", flush=True)

    # 4. Chunking Strategies
    strategies = {
        "A": lambda pg: fixed_size(
            pg.text, pg.page,
            document_id=pg.document_id, source=pg.source,
            size=500, overlap=50,
        ),
        "B": lambda pg: by_paragraph(
            pg.text, pg.page,
            document_id=pg.document_id, source=pg.source,
            max_size=800,
        ),
        "C": lambda pg: by_paragraph(
            pg.text, pg.page,
            document_id=pg.document_id, source=pg.source,
            max_size=800,
        ),
    }

    # 5. Embedding & Upserting
    for strat_name, splitter in strategies.items():
        print(f"✂️ Strategy {strat_name}: Đang cắt text thành các chunks...", flush=True)
        all_chunks: List[Chunk] = []
        for pg in pages:
            chunks = splitter(pg)
            # Strategy C: Gán thêm metadata ngữ cảnh từ trang
            if strat_name == "C":
                for c in chunks:
                    c.metadata["page"] = pg.page
                    c.metadata["source"] = pg.source
            all_chunks.extend(chunks)

        texts = [c.text for c in all_chunks]
        print(f"📦 Strategy {strat_name}: Cắt được {len(texts)} chunks. Đang tạo embeddings...", flush=True)
        vectors = embedder.embed_documents(texts)

        print(f"💾 Strategy {strat_name}: Đang lưu vào Postgres...", flush=True)
        store.upsert(all_chunks, vectors)
        print(f"✔️ Strategy {strat_name}: Đã lưu thành công!\n", flush=True)

    # 6. Benchmark
    bench_path = root_dir / "benchmark" / "questions.json"
    with open(bench_path, "r", encoding="utf-8-sig") as f:
        benchmark = json.load(f)
    print(f"📋 Đang đánh giá {len(benchmark)} câu hỏi benchmark...", flush=True)

    traces = []
    for q in benchmark:
        q_id = q["id"]
        split = q["split"]
        query_vec = embedder.embed_query(q["question"])

        for strat_name in strategies:
            filt = q.get("filter") if strat_name == "C" else None
            # Nếu filter rỗng thì bỏ qua filter
            if not filt:
                filt = None
            hits = store.search(query_vec, top_k=3, filters=filt)
            ev = evaluate_query(q, hits)

            traces.append(
                {
                    "strategy": strat_name,
                    "split": split,
                    "question_id": q_id,
                    "hit1": ev["hit1"],
                    "hit3": ev["hit3"],
                }
            )

    # 7. Summary
    summary = summarize(traces)
    print("\n" + "="*30 + " KẾT QUẢ ĐÁNH GIÁ (SUMMARY) " + "="*30)
    for key, stats in summary.items():
        print(
            f"  {key:<8}: Hit@1 = {stats['hit1_ratio']:>6.2%} | Hit@3 = {stats['hit3_ratio']:>6.2%} "
            f"({stats['hit1']}/{stats['total']} câu | {stats['hit3']}/{stats['total']} câu)"
        )
    print("="*88)

    # 8. Write log to root artifacts/
    root_dir = Path(__file__).parents[2]  # K5-L3B-DAY06-LeQuanBao-2A202603888-DataIngestion/
    log_path = root_dir / "artifacts" / "experiment_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["strategy", "split", "question_id", "hit1", "hit3"],
        )
        writer.writeheader()
        writer.writerows(traces)

    print(f"\n🗂 File log đã được ghi thành công tại: {log_path}\n")


if __name__ == "__main__":
    main()
