"""
Test script for the RAG system
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_system import create_rag_system, RAGConfig


def test_rag_system():
    """Test the RAG system with sample data"""

    print("Testing RAG System")
    print("=" * 50)

    # Create RAG system (without API keys for testing)
    config = RAGConfig(
        persist_directory="./data/test_chroma_db"
    )
    rag_system = create_rag_system(persist_directory=config.persist_directory)

    # Clear any existing data
    print("Clearing existing knowledge base...")
    rag_system.clear_knowledge_base()

    # Ingest sample documents
    print("Ingesting sample documents...")
    sample_docs_path = Path(__file__).parent.parent / "data" / "sample_docs"
    if sample_docs_path.exists():
        rag_system.ingest_documents([str(sample_docs_path)])

    # Get system stats
    stats = rag_system.get_stats()
    print(f"System Stats: {stats}")

    # Test queries (these will work without API keys since we're testing the retrieval)
    test_queries = [
        "What is RAG?",
        "Explain prompt engineering techniques",
        "What are the main components of a neural network?",
        "How do you evaluate generative AI models?",
    ]

    print("\nTesting document retrieval (without LLM generation):")
    print("-" * 50)

    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            # Test just the retrieval part
            search_results = rag_system.vector_store.search(query, n_results=2)

            if search_results['documents'][0]:
                print("Retrieved documents:")
                for i, doc in enumerate(search_results['documents'][0][:2]):
                    print(f"  {i+1}. {doc[:150]}...")
            else:
                print("  No relevant documents found")

        except Exception as e:
            print(f"  Error: {e}")

    print("\nRAG system test completed!")
    print("\nNote: To test full Q&A functionality, set your API keys:")
    print("  export OPENAI_API_KEY='your-key'")
    print("  export ANTHROPIC_API_KEY='your-key'")


if __name__ == "__main__":
    test_rag_system()