import mistune
import json
from pathlib import Path


def print_ast(file_path: str):
    print(f"Printing AST for: {file_path}")
    md = mistune.create_markdown(renderer="ast")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    ast = md(content)
    # Filter to show some structure without too much noise
    print(json.dumps(ast[:10], indent=2))


if __name__ == "__main__":
    import sys

    test_file = "data/test_docs/ai.md"
    if len(sys.argv) > 1:
        test_file = sys.argv[1]

    if Path(test_file).exists():
        print_ast(test_file)
    else:
        print(f"File {test_file} not found.")
