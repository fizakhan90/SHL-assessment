"""
Hybrid Retrieval Layer for SHL Assessment Recommender

Combines dense (semantic) search via ChromaDB + sentence-transformers
with sparse (keyword/BM25) search via rank_bm25, then fuses the ranked
results using Reciprocal Rank Fusion (RRF).

"""

import json
import re
import time
import logging
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


CATALOG_PATH = Path(__file__).parent / "shl_catalog_clean.json"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "shl_assessments"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  
DEFAULT_K = 10
RRF_K = 60 

def build_chunk(entry: dict) -> str:
    name = entry["name"]
    test_type = entry.get("test_type_name") or "Assessment"
    description = entry.get("description") or ""
    job_levels = entry.get("job_levels") or ""

    parts = [
        f"{name}.",
        f"Type: {test_type}.",
    ]
    if job_levels:
        parts.append(f"Job levels: {job_levels}.")
    if description:
        parts.append(description)

    return " ".join(parts)

def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9.+#]+", text)
    tokens = [t.rstrip(".+") if len(t.rstrip(".+")) > 0 else t for t in tokens]
    return tokens

class SHLRetriever:
    def __init__(
        self,
        catalog_path: str | Path = CATALOG_PATH,
        force_rebuild: bool = False,
    ):
        t0 = time.perf_counter()

        with open(catalog_path, encoding="utf-8") as f:
            self.catalog: list[dict] = json.load(f)
        logger.info(f"Loaded {len(self.catalog)} assessments from catalog")

        self.chunks: list[str] = []
        self.ids: list[str] = []
        for i, entry in enumerate(self.catalog):
            self.chunks.append(build_chunk(entry))
            self.ids.append(f"shl_{i}")

        self.bm25_corpus = [tokenize(chunk) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.bm25_corpus)
        logger.info("BM25 index built")

        self.model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")

        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        existing_collections = [c.name for c in self.chroma_client.list_collections()]
        needs_build = force_rebuild or COLLECTION_NAME not in existing_collections

        if not needs_build:
            self.collection = self.chroma_client.get_collection(COLLECTION_NAME)
            if self.collection.count() != len(self.catalog):
                logger.warning(
                    f"Collection has {self.collection.count()} docs but catalog "
                    f"has {len(self.catalog)} — rebuilding"
                )
                needs_build = True

        if needs_build:
            if COLLECTION_NAME in existing_collections:
                self.chroma_client.delete_collection(COLLECTION_NAME)
            self.collection = self.chroma_client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},  
            )
            self._index_all()
        else:
            logger.info(
                f"Using existing ChromaDB collection ({self.collection.count()} docs)"
            )

        elapsed = time.perf_counter() - t0
        logger.info(f"SHLRetriever initialized in {elapsed:.2f}s")

    def _index_all(self) -> None:
        logger.info("Embedding and indexing all assessments into ChromaDB...")
        t0 = time.perf_counter()
        embeddings = self.model.encode(
            self.chunks,
            show_progress_bar=True,
            batch_size=64,
        ).tolist()
        metadatas = []
        for entry in self.catalog:
            metadatas.append({
                "name": entry["name"],
                "url": entry["url"],
                "test_type": entry.get("test_type") or "",
                "test_type_name": entry.get("test_type_name") or "",
                "job_levels": entry.get("job_levels") or "",
                "description": (entry.get("description") or "")[:500],  
                "assessment_length_minutes": str(entry.get("assessment_length_minutes") or ""),
                "remote_testing": entry.get("remote_testing") or "",
                "languages": entry.get("languages") or "",
                "fact_sheet_url": entry.get("fact_sheet_url") or "",
            })

        self.collection.add(
            ids=self.ids,
            embeddings=embeddings,
            documents=self.chunks,
            metadatas=metadatas,
        )

        elapsed = time.perf_counter() - t0
        logger.info(f"Indexed {len(self.chunks)} assessments in {elapsed:.2f}s")

    def _dense_search(self, query: str, k: int) -> list[dict]:
        query_embedding = self.model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "id": results["ids"][0][i],
                "rank": i + 1,
                "distance": results["distances"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
            })
        return hits

    def _sparse_search(self, query: str, k: int) -> list[dict]:
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        hits = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                continue  
            entry = self.catalog[idx]
            hits.append({
                "id": self.ids[idx],
                "rank": rank + 1,
                "score": float(scores[idx]),
                "metadata": {
                    "name": entry["name"],
                    "url": entry["url"],
                    "test_type": entry.get("test_type") or "",
                    "test_type_name": entry.get("test_type_name") or "",
                    "job_levels": entry.get("job_levels") or "",
                    "description": (entry.get("description") or "")[:500],
                    "assessment_length_minutes": str(entry.get("assessment_length_minutes") or ""),
                    "remote_testing": entry.get("remote_testing") or "",
                    "languages": entry.get("languages") or "",
                    "fact_sheet_url": entry.get("fact_sheet_url") or "",
                },
            })
        return hits

    @staticmethod
    def _rrf_fuse(
        dense_hits: list[dict],
        sparse_hits: list[dict],
        k: int = DEFAULT_K,
        rrf_k: int = RRF_K,
    ) -> list[dict]:
        fused: dict[str, dict] = {}

        for hit in dense_hits:
            doc_id = hit["id"]
            rrf_score = 1.0 / (rrf_k + hit["rank"])
            if doc_id not in fused:
                fused[doc_id] = {
                    "id": doc_id,
                    "metadata": hit["metadata"],
                    "rrf_score": 0.0,
                    "dense_rank": hit["rank"],
                    "dense_distance": hit.get("distance"),
                    "sparse_rank": None,
                    "sparse_score": None,
                }
            fused[doc_id]["rrf_score"] += rrf_score

        for hit in sparse_hits:
            doc_id = hit["id"]
            rrf_score = 1.0 / (rrf_k + hit["rank"])
            if doc_id not in fused:
                fused[doc_id] = {
                    "id": doc_id,
                    "metadata": hit["metadata"],
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "dense_distance": None,
                    "sparse_rank": hit["rank"],
                    "sparse_score": hit.get("score"),
                }
            else:
                fused[doc_id]["sparse_rank"] = hit["rank"]
                fused[doc_id]["sparse_score"] = hit.get("score")
            fused[doc_id]["rrf_score"] += rrf_score

        ranked = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
        return ranked[:k]

    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_K,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> list[dict]:
        
        t0 = time.perf_counter()
        fetch_k = min(k * 3, len(self.catalog))

        dense_hits = self._dense_search(query, k=fetch_k)
        sparse_hits = self._sparse_search(query, k=fetch_k)

        fused = self._rrf_fuse(dense_hits, sparse_hits, k=k)

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Hybrid retrieval for '{query[:50]}...' → "
            f"{len(fused)} results in {elapsed*1000:.1f}ms"
        )

        return fused

    def retrieve_clean(self, query: str, k: int = DEFAULT_K) -> list[dict]:
        results = self.retrieve(query, k=k)
        clean_results = []
        for r in results:
            meta = r["metadata"]
            clean_results.append({
                "name": meta["name"],
                "url": meta["url"],
                "description": meta["description"],
                "test_type": meta["test_type"],
                "test_type_name": meta["test_type_name"],
                "job_levels": meta["job_levels"],
                "assessment_length_minutes": meta["assessment_length_minutes"],
                "remote_testing": meta["remote_testing"],
                "languages": meta["languages"],
                "fact_sheet_url": meta["fact_sheet_url"],
            })
        return clean_results

def main():
    retriever = SHLRetriever()
    test_queries = [
        "Java developer mid level",
        "personality assessment for sales manager",
        "OPQ32r",
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"{'='*80}")

        results = retriever.retrieve(query, k=10)

        for i, result in enumerate(results):
            meta = result["metadata"]
            print(f"\n  [{i+1}] {meta['name']}")
            print(f"      URL:        {meta['url']}")
            print(f"      Type:       {meta['test_type_name']}")
            print(f"      Job Levels: {meta['job_levels']}")
            print(f"      Duration:   {meta['assessment_length_minutes']} min")
            print(f"      Remote:     {meta['remote_testing']}")
            print(f"      RRF Score:  {result['rrf_score']:.6f}")
            print(f"      Dense Rank: {result['dense_rank']}")
            print(f"      Sparse Rank: {result['sparse_rank']}")


if __name__ == "__main__":
    main()
