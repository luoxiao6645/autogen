import asyncio
import os
from dotenv import load_dotenv

from autogen_ext.memory.chromadb import ChromaDBVectorMemory, ChromaDBVectorMemoryConfig
from autogen_core.memory import MemoryContent, MemoryMimeType

# It's good practice to load environment variables if your embedding model needs API keys,
# though ChromaDB's default SentenceTransformer might not.
# OpenAI embeddings, if used explicitly here, would need it.
load_dotenv()

# --- Configuration ---
PERSISTENCE_DIR = "./chroma_db_store"
COLLECTION_NAME = "personal_agent_docs"

# --- Sample Documents ---
SAMPLE_DOCUMENTS = [
    {
        "id": "doc1",
        "content": """AutoGen is a framework for building applications with multiple agents that can converse with each other to solve tasks.
AutoGen agents are customizable, conversable, and can seamlessly allow human participation.
They can operate in various modes that employ combinations of LLMs, human inputs, and tools.
AutoGen enables building next-gen LLM applications based on multi-agent conversations.
It simplifies the orchestration, automation, and optimization of complex LLM workflows.""",
        "metadata": {"source": "autogen_overview.txt", "category": "framework"}
    },
    {
        "id": "doc2",
        "content": """The UserProxyAgent is an agent that acts as a proxy for the human user.
It can solicit input from the user, or execute code on behalf of the user.
By default, it will solicit human input an LLM call receives a "TERMINATE" signal or if human_input_mode is set to "ALWAYS".
It can also be configured to execute code blocks it receives.
The AssistantAgent is a generic agent that uses an LLM to generate responses.
It can be configured with a system message to define its role and behavior.""",
        "metadata": {"source": "autogen_agents.txt", "category": "components"}
    },
    {
        "id": "doc3",
        "content": """RAG stands for Retrieval Augmented Generation.
It's a technique to improve the quality of LLM responses by grounding them in factual information retrieved from an external knowledge base.
This involves retrieving relevant document chunks and providing them as context to the LLM when generating an answer.
ChromaDB is a popular open-source vector database used for storing and querying embeddings, often used in RAG pipelines.""",
        "metadata": {"source": "rag_and_chroma.txt", "category": "concepts"}
    },
    {
        "id": "doc4", # Added a non-technical document for variety
        "content": """The best way to learn a new skill is through consistent practice and seeking feedback.
Start with the fundamentals, break down complex topics into smaller parts, and apply what you learn in real-world projects.
Don't be afraid to make mistakes, as they are valuable learning opportunities.
Regularly review and reinforce your knowledge.""",
        "metadata": {"source": "learning_tips.txt", "category": "advice"}
    }
]

# --- Basic Text Chunker ---
def simple_chunker(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Splits text into chunks of roughly chunk_size with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start += (chunk_size - overlap)
    return chunks

async def main():
    print(f"Initializing ChromaDB for collection '{COLLECTION_NAME}' with persistence at '{PERSISTENCE_DIR}'...")

    # Ensure persistence directory exists
    os.makedirs(PERSISTENCE_DIR, exist_ok=True)

    # Configure ChromaDBVectorMemory for persistence
    # If you want to use a specific embedding model (e.g., OpenAI),
    # you'd configure it here or ensure ChromaDB client is set up for it.
    # For now, we rely on ChromaDB's default (SentenceTransformer).
    chroma_config = ChromaDBVectorMemoryConfig(
        collection_name=COLLECTION_NAME,
        persistence_path=PERSISTENCE_DIR,
        # embedding_model_name="all-MiniLM-L6-v2" # Example of specifying default SentenceTransformer
    )
    vector_db = ChromaDBVectorMemory(config=chroma_config)

    print(f"Clearing any existing data from collection '{COLLECTION_NAME}'...")
    await vector_db.clear() # Clears all data from the specified collection

    total_chunks_added = 0
    print("Starting to process and index documents...")

    for doc in SAMPLE_DOCUMENTS:
        doc_id = doc["id"]
        content = doc["content"]
        metadata = doc["metadata"]

        print(f"  Processing document: {doc_id} ({metadata.get('source', 'N/A')})")

        # Chunk the document content
        # For simplicity, we're using a very basic chunker. In a real RAG system,
        # more sophisticated chunking (semantic, sentence-based, etc.) would be used.
        chunks = simple_chunker(content, chunk_size=300, overlap=30) # Smaller chunks for dense info

        print(f"    Split into {len(chunks)} chunks.")

        memory_contents_to_add = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = metadata.copy() # shallow copy
            chunk_metadata["original_doc_id"] = doc_id
            chunk_metadata["chunk_index"] = i

            memory_contents_to_add.append(
                MemoryContent(
                    content=chunk_text,
                    mime_type=MemoryMimeType.TEXT,
                    metadata=chunk_metadata
                )
            )

        if memory_contents_to_add:
            await vector_db.add(memory_contents_to_add) # Batch add chunks for this document
            total_chunks_added += len(memory_contents_to_add)
            print(f"    Added {len(memory_contents_to_add)} chunks to ChromaDB.")

    print(f"\nIndexing complete. Total chunks added: {total_chunks_added}.")

    # It's good practice to close the DB connection if the underlying client supports it,
    # though ChromaDBVectorMemory might manage this internally for file-based persistence.
    # await vector_db.close() # Not explicitly available in ChromaDBVectorMemory, client handles it.

if __name__ == "__main__":
    asyncio.run(main())
