import mistune
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer

class MarkdownChunker:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", max_tokens: int = 256):
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # We use 'ast' renderer to get the structure
        self.md = mistune.create_markdown(renderer='ast')

    def chunk(self, text: str) -> List[str]:
        tokens = self.md(text)
        chunks = []
        header_stack = []

        for token in tokens:
            if token['type'] == 'heading':
                level = token['attrs']['level']
                title = self._get_text_from_children(token.get('children', []))
                
                # Update header stack
                while len(header_stack) >= level:
                    header_stack.pop()
                while len(header_stack) < level - 1:
                    header_stack.append("...")
                header_stack.append(title)
                
            elif token['type'] in ('paragraph', 'list', 'block_code', 'block_quote'):
                breadcrumb = " > ".join(header_stack)
                content = self._get_token_content(token)
                
                if not content.strip():
                    continue

                prefix = f"[{breadcrumb}] | " if breadcrumb else ""
                full_chunk = prefix + content
                
                # Precise truncation using tokenizer
                encoded = self.tokenizer.encode(full_chunk, add_special_tokens=False)
                if len(encoded) > self.max_tokens:
                    # '…' is 1 token
                    truncated_encoded = encoded[:self.max_tokens - 1]
                    full_chunk = self.tokenizer.decode(truncated_encoded).strip() + "…"
                
                chunks.append(full_chunk)
        
        return chunks

    def _get_text_from_children(self, children: List[Dict[str, Any]]) -> str:
        text = ""
        for child in children:
            if 'text' in child:
                text += child['text']
            elif 'raw' in child:
                text += child['raw']
            elif 'children' in child:
                text += self._get_text_from_children(child['children'])
        return text

    def _get_token_content(self, token: Dict[str, Any]) -> str:
        if token['type'] == 'block_code':
            return token.get('raw', '')
        return self._get_text_from_children(token.get('children', []))

if __name__ == "__main__":
    # POC Test
    test_md = """
# Introduction
This is a paragraph in the intro.

## Details
Here is some detail.

### Sub-detail
More specific info.

# Conclusion
The end.

# Long Text
""" + ("This is a very long text. " * 100)
    
    chunker = MarkdownChunker()
    for c in chunker.chunk(test_md):
        tokens_count = len(chunker.tokenizer.encode(c, add_special_tokens=False))
        print(f"CHUNK: {c[:100]}... (len tokens: {tokens_count})")
