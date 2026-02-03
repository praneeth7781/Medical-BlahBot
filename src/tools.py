from langchain_core.tools import tool
from typing import Dict, List
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import torch
import dotenv
import os
from qdrant_client import QdrantClient

if os.environ.get("QDRANT_API_KEY") is None or os.environ.get("QDRANT_CLUSTER_ENDPOINT") is None:
    dotenv.load_dotenv()
qdrant_client = QdrantClient(
    url=os.environ["QDRANT_CLUSTER_ENDPOINT"],
    api_key=os.environ["QDRANT_API_KEY"]
)

class SearchInput(BaseModel):
    query: str = Field(description="The student's medical question.")
    # num_chunks: int = Field(default=5, description="Number of relevant chunks to retrieve.")

def setup_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def run_query(query_text, n_results):
    device = setup_device()
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").to(device)
    embedding = model.encode(sentences=query_text, device=device).tolist()

    try:
        results = qdrant_client.query_points(
            collection_name="DC_Dutta",
            query=embedding,
            limit=n_results
        )

        hits = results.points
        
        chunks = []
        for hit in hits:
            chunks.append({
                "chunk": hit.payload.get("text", ""),
                "score": hit.score,
                "metadata": {
                    "book_name": hit.payload.get("book_name",""),
                    "chapter_title": hit.payload.get("chapter_title", ""),
                    "section": hit.payload.get("section",""),
                    "subsection": hit.payload.get("subsection", ""),
                    "hierarchy": hit.payload.get("hierarchy", "")
                }
            })
    except Exception as e:
        print(f"Error processing results: {e}")
        chunks = []

    return chunks


@tool(description="Search the medical reference book DC Dutta for relevant information.", args_schema=SearchInput, return_direct=False)
def search_medical_reference(query: str) -> str:
    chunks = run_query(query, 5)

    import json
    return json.dumps(chunks)

def format_chunks_for_prompt(chunks: List[str], metadata: List[Dict]) -> str:
    "Format retrieved chunks and their metadata for inclusion in prompts."
    formatted_chunks = []
    for i, (chunk, meta) in enumerate(zip(chunks, metadata), start=1):
        citation = f"[{i}] {meta['book_name']}, {meta['edition']}, Chapter: {meta['chapter']}: {meta['section']}"
        formatted_chunks.append(f"{chunk}\n\nCitation: {citation}\n")
    return "\n".join(formatted_chunks)

def format_chunks_preview(chunks: List[str], max_preview_length: int = 200) -> str:
    "Create a preview of the retrieved chunks for evaluation prompt."
    previews = []
    for i, chunk in enumerate(chunks, start=1):
        preview = chunk[:max_preview_length].replace('\n', ' ') + ("..." if len(chunk) > max_preview_length else "")
        previews.append(f"- Chunk [{i}] {preview}")
    return "\n".join(previews)

tools = [search_medical_reference]

if __name__ == "__main__":
    test_query = "What is the structure of clitoris?"
    print(search_medical_reference.invoke({"query": test_query, "num_chunks": 5}))