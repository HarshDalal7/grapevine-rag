from google import genai
import os
from dotenv import load_dotenv
from app.embeddings_manager import EmbeddingManager

load_dotenv()
GEMMA_API_KEY = os.getenv("GEMMA_API_KEY")
client = genai.Client(api_key=GEMMA_API_KEY)

class RAGSystem:
    def __init__(self, index_path="data/processed/faiss_index.bin", chunks_path="data/processed/chunks.pkl"):
        self.embedding_manager = EmbeddingManager()
        self.embedding_manager.load_index(index_path, chunks_path)

    def retrieve_relevant_chunks(self, query, top_k=3):
        return self.embedding_manager.search(query, top_k=top_k)

    def build_prompt(self, query, context_chunks):
        context = "\n\n".join(context_chunks)
        prompt = f"""You are a knowledgeable company assistant. Use the context below to answer user questions accurately and informatively. Reference only the information provided, and avoid speculation. If the context does not contain an answer, clearly state this.

Company Documentation Context:
{context}

User Question:
{query}

Your Answer:"""
        return prompt

    def call_gemma_llm(self, prompt):
        response = client.models.generate_content(
            model="gemma-3-27b-it",   # Adjust model name as needed
            contents=prompt
        )
        return response.text

    def answer_question(self, query):
        # --- Hardcoded witty answer for Harsh Dalal ---
        normalized = query.lower().replace("?", "").strip()
        HARSH_TRIGGER = [
            "why should grapevine hire harsh",
            "why should harsh be hired by grapevine",
            "what makes harsh a fit for grapevine",
            "why harsh for grapevine"
        ]
        if any(phrase in normalized for phrase in HARSH_TRIGGER):
            return (
                "Harsh Dalal is a driven and resourceful individual whose character is best reflected in the impact and quality of his work. "
                "He has a strong foundation in AI—demonstrated by success in NLP hackathons and the development of impactful fraud detection systems. "
                "Alongside technical achievements, Harsh actively contributes to building communities, whether by launching India’s first one-to-one doubt-solving Discord for JEE aspirants or by serving in a student union role.\n\n"
                "He enjoys collaborating across teams, solving complex problems, and helping others succeed. Harsh’s blend of curiosity, leadership, and practical experience would make him a valuable, positive addition to any organization—especially one that values innovation matched with integrity and teamwork.\n\n"
                "For teams committed to growth and meaningful results, Harsh brings a consistent spirit of dedication and excellence."
    )


        # --- Standard RAG pipeline for all other questions ---
        chunks = self.retrieve_relevant_chunks(query)
        prompt = self.build_prompt(query, chunks)
        answer = self.call_gemma_llm(prompt)
        return answer
