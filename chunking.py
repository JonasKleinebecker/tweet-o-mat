from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from transformers import AutoTokenizer

model_name = "JungZoona/T3Q-qwen2.5-14b-v1.0-e3"

tokenizer = AutoTokenizer.from_pretrained(model_name)
MAX_TOKENS = 4096

converter = DocumentConverter()
conv_result = converter.convert("https://www.poewiki.net/wiki/Scourge")
document = conv_result.document

chunker = HybridChunker(tokenizer=tokenizer, max_tokens=MAX_TOKENS, merge_peers=True)
conv_result = converter.convert("https://www.poewiki.net/wiki/Scourge")
document = conv_result.document

chunks = chunker.chunk(dl_doc=document)
chunks = list(chunks)
print(len(chunks))
chunks[0].model_dump()

for chunk in chunks:
    print("------ NEW CHUNK ------")
    print(chunk.text)
