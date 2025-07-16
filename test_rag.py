from rag_system import RAGSystem

rag = RAGSystem()

query = "What's the company's market share?"
# This returns the final answer from the LLM, given the best context retrieved!
answer = rag.answer_question(query)
print("\nGemma's Answer:")
print(answer)
