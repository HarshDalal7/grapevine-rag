from app.embeddings_manager import EmbeddingManager
from app.extracting import extract_text_from_pdf, chunk_text

# Step 1: Load & chunk document
pdf_path = "data/raw/GRAPEVINEDATA.pdf"
text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text)

# Step 2: Embed and save
embedder = EmbeddingManager()
embedder.embed_chunks(chunks)
embedder.save_index()

print("✅ Embeddings created and saved.")
