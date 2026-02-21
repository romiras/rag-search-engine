from rag_search_engine.core.chunker import MarkdownChunker
from pathlib import Path


def test_chunk_file(file_path: str):
    print(f"Testing chunking for: {file_path}")
    chunker = MarkdownChunker()
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = chunker.chunk(content)
    print(f"Produced {len(chunks)} chunks.")
    for i, c in enumerate(chunks[:5]):
        print(f"Chunk {i + 1}: {c[:100]}...")
    if len(chunks) > 5:
        print("...")


if __name__ == "__main__":
    import sys

    # Test with one of the files from data/test_docs
    test_file = "data/test_docs/ai.md"
    if len(sys.argv) > 1:
        test_file = sys.argv[1]

    if Path(test_file).exists():
        test_chunk_file(test_file)
    else:
        print(f"File {test_file} not found.")
