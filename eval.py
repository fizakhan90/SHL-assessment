import json
import os
import re
from pathlib import Path
from retrieval import SHLRetriever

TRACES_DIR = Path(r"C:\Users\Hp\Downloads\sample_conversations\GenAI_SampleConversations")
CATALOG_PATH = r"C:\Users\Hp\Desktop\SHL_assesment\shl_catalog_clean.json"

def check_catalog_coverage():
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)

    catalog_names = {d['name'].lower() for d in catalog}

    print(f"{'Assessment':<60} | {'In Catalog'}")
    print("-" * 75)
    for file_path in sorted(TRACES_DIR.glob("*.md")):
        text = file_path.read_text(encoding="utf-8")
        expected = extract_expected_assessments(text)
        for name in expected:
            found = name.lower() in catalog_names
            print(f"{name:<60} | {'OK' if found else 'X MISSING'}")

def extract_raw_query(markdown_text: str) -> str:
    query_parts = []
    blocks = re.split(r"\*\*(User|Agent)\*\*", markdown_text)
    for i in range(1, len(blocks), 2):
        role = blocks[i]
        content = blocks[i + 1].strip()
        if role == "User":
            content = "\n".join(line.lstrip("> ").strip() for line in content.splitlines())
            query_parts.append(content)
    combined = " | ".join(query_parts)
    words = combined.split()
    if len(words) > 300:
        combined = " ".join(words[-300:])
    return combined

def extract_query(markdown_text: str, groq_client=None) -> str:
    from main import synthesize_search_query
    query_parts = []
    blocks = re.split(r"\*\*(User|Agent)\*\*", markdown_text)

    messages = []
    for i in range(1, len(blocks), 2):
        role = blocks[i]
        content = blocks[i + 1].strip()
        if role == "User":
            content = "\n".join(line.lstrip("> ").strip() for line in content.splitlines())
            messages.append({"role": "user", "content": content})

    return synthesize_search_query(messages, groq_client)


def extract_expected_assessments(markdown_text: str) -> list:
    """Extract expected assessment names from the final markdown table."""
    lines = markdown_text.strip().splitlines()
    tables = []
    current_table = []

    for line in lines:
        line = line.strip()
        if line.startswith("|") and line.endswith("|"):
            current_table.append(line)
        else:
            if current_table:
                tables.append(current_table)
                current_table = []

    if current_table:
        tables.append(current_table)

    if not tables:
        return []

    last_table = tables[-1]
    headers = [h.strip().lower() for h in last_table[0].split("|")[1:-1]]
    if "name" not in headers:
        return []

    name_idx = headers.index("name")
    names = []

    for row in last_table[2:]:
        cols = [c.strip() for c in row.split("|")[1:-1]]
        if len(cols) > name_idx:
            name = cols[name_idx]
            name = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", name)
            name = name.replace("**", "").replace("__", "").strip()
            names.append(name)

    return names


def calculate_metrics(expected: list, retrieved: list, k: int = 10):
    """Compute Recall@K, Precision@K, and F1@K."""
    if not expected:
        return 0.0, 0.0, 0.0

    expected_set = set(e.lower() for e in expected)
    retrieved_set = set(r.lower() for r in retrieved[:k])

    hits = len(expected_set.intersection(retrieved_set))

    recall = hits / len(expected_set)
    precision = hits / k if k > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return recall, precision, f1


def main():
    print("=" * 80)
    print("CATALOG COVERAGE CHECK")
    print("=" * 80)
    check_catalog_coverage()

    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION")
    print("=" * 80)

    print("\nLoading SHL Retriever...")
    retriever = SHLRetriever()

    results = []
    gap_analysis = []

    files = list(TRACES_DIR.glob("*.md"))
    print(f"Found {len(files)} trace files.\n")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from groq import Groq
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
    
    if not groq_client:
        print("\nWARNING: GROQ_API_KEY not found in environment! Distillation/Reranking will fall back.\n")

    from main import rerank_assessments

    for file_path in sorted(files):
        trace_id = file_path.stem
        text = file_path.read_text(encoding="utf-8")

        raw_query = extract_raw_query(text)
        distilled_query = extract_query(text, groq_client)
        expected = extract_expected_assessments(text)
        expected = extract_expected_assessments(text)

        if not expected:
            print(f"Skipping {trace_id}: No expected assessments found.")
            continue

        # Mode A: BM25 only (Baseline)
        bm25_hits = retriever.retrieve(raw_query, k=10, mode="bm25")
        bm25_names = [h["metadata"]["name"] for h in bm25_hits]
        bm25_recall, bm25_prec, bm25_f1 = calculate_metrics(expected, bm25_names, k=10)

        # Mode B: Hybrid (Baseline)
        hybrid_hits = retriever.retrieve(raw_query, k=10, mode="hybrid")
        hybrid_names = [h["metadata"]["name"] for h in hybrid_hits]
        hybrid_recall, hybrid_prec, hybrid_f1 = calculate_metrics(expected, hybrid_names, k=10)

        # Mode C: New Pipeline (Distilled Query -> Hybrid K=30 -> Rerank K=10)
        pipeline_candidates = retriever.retrieve_clean(distilled_query, k=30, mode="hybrid")
        pipeline_names_30 = [c["name"] for c in pipeline_candidates]
        
        reranked = rerank_assessments(distilled_query, pipeline_candidates, groq_client)
        pipeline_names_10 = [c["name"] for c in reranked]
        pipeline_recall, pipeline_prec, pipeline_f1 = calculate_metrics(expected, pipeline_names_10, k=10)
        
        print(f"\n--- DEBUG LOGS FOR {trace_id} ---")
        print(f"Raw Query: {raw_query}")
        print(f"Distilled Query: {distilled_query}")
        print(f"30 Retrieved Candidates (before rerank): {pipeline_names_30[:3]}... (showing first 3)")
        print(f"10 Reranked Candidates: {pipeline_names_10}")

        results.append({
            "trace": trace_id,
            "expected_count": len(expected),
            "bm25_r10": bm25_recall,
            "hybrid_r10": hybrid_recall,
            "pipeline_r10": pipeline_recall,
            "bm25_f1": bm25_f1,
            "hybrid_f1": hybrid_f1,
            "pipeline_f1": pipeline_f1,
        })

        expected_lower = [e.lower() for e in expected]
        bm25_lower = [r.lower() for r in bm25_names]
        pipeline_lower = [r.lower() for r in pipeline_names_10]

        missed_by_baseline = []
        for orig_name, lower_name in zip(expected, expected_lower):
            if lower_name not in bm25_lower and lower_name in pipeline_lower:
                missed_by_baseline.append(orig_name)

        if missed_by_baseline:
            gap_analysis.append((trace_id, missed_by_baseline))

    print(f"\n{'=' * 95}")
    print(f"{'Trace':<10} | {'Expected':<8} | {'BM25-R@10':<10} | {'Hybrid-R@10':<11} | {'Pipeline-R@10':<13} | {'BM25-F1':<8} | {'Hybrid-F1':<9} | {'Pipeline-F1':<11}")
    print("-" * 95)

    sum_bm25_r = sum_hyb_r = sum_pipe_r = sum_bm25_f = sum_hyb_f = sum_pipe_f = 0.0

    for r in results:
        sum_bm25_r += r['bm25_r10']
        sum_hyb_r += r['hybrid_r10']
        sum_pipe_r += r['pipeline_r10']
        sum_bm25_f += r['bm25_f1']
        sum_hyb_f += r['hybrid_f1']
        sum_pipe_f += r['pipeline_f1']
        print(
            f"{r['trace']:<10} | {r['expected_count']:<8} | "
            f"{r['bm25_r10']:<10.2f} | {r['hybrid_r10']:<11.2f} | "
            f"{r['pipeline_r10']:<13.2f} | "
            f"{r['bm25_f1']:<8.2f} | {r['hybrid_f1']:<9.2f} | {r['pipeline_f1']:<11.2f}"
        )

    n = len(results)
    print("-" * 95)
    print(
        f"{'Mean':<10} | {'':<8} | {sum_bm25_r/n:<10.2f} | "
        f"{sum_hyb_r/n:<11.2f} | {sum_pipe_r/n:<13.2f} | "
        f"{sum_bm25_f/n:<8.2f} | {sum_hyb_f/n:<9.2f} | {sum_pipe_f/n:<11.2f}"
    )
    print("=" * 95)

    print("\nGap Analysis: Expected assessments missed by Baseline but found by Pipeline.")
    print("-" * 95)
    if not gap_analysis:
        print("None! Baseline found everything that Pipeline found.")
    else:
        for trace_id, missed in gap_analysis:
            print(f"Trace: {trace_id}")
            for m in missed:
                print(f"  - {m}")


if __name__ == "__main__":
    main()