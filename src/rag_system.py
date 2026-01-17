"""
RAG (Retrieval Augmented Generation) System
A comprehensive implementation for document Q&A using vector search and LLMs
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

import chromadb

# Import advanced document processor
from .document_processor import AdvancedDocumentProcessor, Document
# Import conversation memory
from .conversation_memory import ConversationManager, ConversationMemory

from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain_anthropic import AnthropicLLM
# from langchain.chains import RetrievalQA  # Temporarily commented out due to import issues
# Using basic Python classes instead of LangChain for simplicity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    collection_name: str = "documents"
    persist_directory: str = "./data/chroma_db"
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None


class DocumentProcessor:
    """Handles document loading and preprocessing using AdvancedDocumentProcessor"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.doc_processor = AdvancedDocumentProcessor()

    def load_documents_from_sources(self, sources: List[str]) -> List[Document]:
        """Load documents from various sources using advanced processor"""
        return self.doc_processor.load_from_mixed_sources(sources)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        return self.text_splitter.split_documents(documents)

    # Keep backward compatibility methods
    def load_text_file(self, file_path: str) -> List[Document]:
        """Load text from a file"""
        return self.doc_processor.load_text_file(file_path)

    def load_pdf_file(self, file_path: str) -> List[Document]:
        """Load text from a PDF file"""
        return self.doc_processor.load_pdf_file(file_path)

    def load_documents_from_directory(self, directory: str) -> List[Document]:
        """Load all documents from a directory"""
        return self.doc_processor.load_documents_from_directory(directory)


class VectorStore:
    """Manages vector database operations"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=config.embedding_model
        )
        self.client = chromadb.PersistentClient(
            path=config.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name
        )

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store"""
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"doc_{i}" for i in range(len(documents))]

        embeddings = self.embedding_function.embed_documents(texts)

        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

        logger.info(f"Added {len(documents)} documents to vector store")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Search for relevant documents"""
        query_embedding = self.embedding_function.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        return results

    def clear_collection(self) -> None:
        """Clear all documents from the collection"""
        try:
            self.client.delete_collection(self.config.collection_name)
            self.collection = self.client.create_collection(
                name=self.config.collection_name
            )
            logger.info("Cleared vector store collection")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")


class LLMManager:
    """Manages different LLM providers"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.providers = {}

        if config.openai_api_key:
            self.providers['openai'] = OpenAI(
                api_key=config.openai_api_key,
                model_name="gpt-3.5-turbo",
                temperature=0.1
            )

        if config.anthropic_api_key:
            self.providers['anthropic'] = AnthropicLLM(
                api_key=config.anthropic_api_key,
                model="claude-3-sonnet-20240229",
                temperature=0.1
            )

    def get_llm(self, provider: str = "openai"):
        """Get LLM instance for specified provider"""
        if provider not in self.providers:
            available = list(self.providers.keys())
            raise ValueError(f"Provider {provider} not available. Available: {available}")

        return self.providers[provider]


class RAGSystem:
    """Main RAG system class"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.document_processor = DocumentProcessor(config)
        self.vector_store = VectorStore(config)
        self.llm_manager = LLMManager(config)

        # Initialize conversation memory
        memory_file = os.path.join(os.path.dirname(config.persist_directory), "conversations.json")
        self.conversation_manager = ConversationManager(
            ConversationMemory(memory_file=memory_file)
        )

        # Custom prompt template for better answers (simplified for now)
        self.qa_prompt = "Use the following context to answer the question."

    def ingest_documents(self, sources: List[str]) -> None:
        """Ingest documents from various sources"""
        # Use the advanced document processor to load from mixed sources
        all_documents = self.document_processor.load_documents_from_sources(sources)

        if all_documents:
            # Split documents into chunks
            split_docs = self.document_processor.split_documents(all_documents)
            # Add to vector store
            self.vector_store.add_documents(split_docs)
            logger.info(f"Successfully ingested {len(split_docs)} document chunks from {len(all_documents)} documents")

            # Log document statistics
            stats = self.document_processor.doc_processor.get_document_stats(all_documents)
            logger.info(f"Document stats: {stats}")

    def query(self, question: str, provider: str = "openai", k: int = 5, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Query the RAG system with optional conversation context"""
        # Handle conversation memory
        if session_id:
            # Add user message to conversation
            self.conversation_manager.add_user_message(session_id, question)

        # Search for relevant documents
        search_results = self.vector_store.search(question, n_results=k)

        if not search_results['documents'][0]:
            response = {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "confidence": 0.0
            }
        else:
            # For now, return retrieval results without LLM generation
            # This allows testing the retrieval system without API keys
            sources = []
            for i, doc in enumerate(search_results['documents'][0]):
                metadata = search_results['metadatas'][0][i] if search_results['metadatas'][0] else {}
                sources.append({
                    "content": doc[:300] + "..." if len(doc) > 300 else doc,
                    "source": metadata.get("source", "Unknown") if metadata else "Unknown"
                })

            # Simple answer extraction from top document
            top_doc = search_results['documents'][0][0] if search_results['documents'][0] else ""
            answer = top_doc[:500] + "..." if len(top_doc) > 500 else top_doc

            response = {
                "answer": answer,
                "sources": sources,
                "confidence": self._calculate_confidence(search_results)
            }

        # Add assistant response to conversation memory
        if session_id:
            self.conversation_manager.add_assistant_message(session_id, response["answer"])

        return response


    def _calculate_confidence(self, search_results) -> float:
        """Calculate confidence score based on search results"""
        if not search_results.get('distances') or not search_results['distances'][0]:
            return 0.0

        # Simple confidence based on distance (lower distance = higher confidence)
        distances = search_results['distances'][0]
        avg_distance = sum(distances) / len(distances)

        # Convert distance to confidence (this is a simple heuristic)
        confidence = max(0.0, 1.0 - (avg_distance / 2.0))
        return round(confidence, 2)

    def start_conversation(self, user_id: str, topic: Optional[str] = None) -> str:
        """Start a new conversation session"""
        return self.conversation_manager.start_conversation(user_id, topic=topic)

    def get_conversation_history(self, session_id: str) -> str:
        """Get formatted conversation history"""
        return self.conversation_manager.get_formatted_history(session_id)

    def clear_conversation(self, session_id: str) -> None:
        """Clear conversation history for a session"""
        self.conversation_manager.memory.clear_conversation(session_id)

    def get_conversation_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get conversation statistics"""
        return self.conversation_manager.memory.get_conversation_stats(session_id)

    def clear_knowledge_base(self) -> None:
        """Clear all documents from the knowledge base"""
        self.vector_store.clear_collection()

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            count = self.vector_store.collection.count()
            return {
                "total_documents": count,
                "embedding_model": self.config.embedding_model,
                "available_providers": list(self.llm_manager.providers.keys())
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}


# Factory function for easy initialization
def create_rag_system(
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    persist_directory: str = "./data/chroma_db"
) -> RAGSystem:
    """Create a RAG system with default configuration"""

    config = RAGConfig(
        openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"),
        persist_directory=persist_directory
    )

    return RAGSystem(config)